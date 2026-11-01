"""Hidden defaults: ``<expr> or <number>``, ``<mapping>.get(<key>, <number>)`` and ``getattr(x, k, <number>)`` in
package code. A number that silently stands in for a missing input is how a certified quantity ends up resting on
a value nobody stated; the input is required, or the case is listed in ``.guards-allow`` with its reason."""
import ast
from pathlib import Path

from .allow import Finding, line_hash, package_files

CHECK = "hidden-defaults"


def _is_number(n):
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        n = n.operand
    return isinstance(n, ast.Constant) and isinstance(n.value, (int, float, complex)) and not isinstance(n.value, bool)


def scan_source(text, path):
    tree = ast.parse(text)
    lines = text.splitlines()
    found = []

    def add(node, reason):
        src = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
        found.append(Finding(CHECK, path, node.lineno, reason, line_hash(CHECK, src)))

    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            for v in node.values[1:]:
                if _is_number(v):
                    add(v, "`<expr> or <number>` supplies a number for a missing input")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "get" and len(node.args) == 2 and _is_number(node.args[1]):
                add(node, "`.get(key, <number>)` supplies a number for a missing key")
            elif isinstance(f, ast.Name) and f.id == "getattr" and len(node.args) == 3 and _is_number(node.args[2]):
                add(node, "`getattr(x, name, <number>)` supplies a number for a missing attribute")
    return found


def check(root, src=(), **_):
    found = []
    for p in package_files(root, src):
        try:
            found += scan_source((Path(root) / p).read_text(encoding="utf-8"), p)
        except (SyntaxError, UnicodeDecodeError):
            continue
    return found
