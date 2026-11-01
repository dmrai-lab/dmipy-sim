"""History wording in docstrings. A docstring says what a thing IS; how it differs from an earlier version belongs
in the pull request or issue text. Flagged: "previously", "no longer", "was replaced", "the old", "Now ..." opening
a sentence, and "used to" in the sense of a past habit (the subject is a pronoun or "which"; "is used to compute"
is not flagged)."""
import ast
import re
from pathlib import Path

from .allow import Finding, line_hash, package_files

CHECK = "history-wording"
_PATTERNS = (
    re.compile(r"\bpreviously\b", re.I),
    re.compile(r"\bno longer\b", re.I),
    re.compile(r"\bwas replaced\b", re.I),
    re.compile(r"\bthe old\b", re.I),
    re.compile(r"(?:^|[.!?]\s+)Now\s"),
    re.compile(r"\b(?:it|this|that|they|we|i|which)\s+used to\b", re.I),
)


def scan_source(text, path):
    tree = ast.parse(text)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        first = node.body[0] if node.body else None
        if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            continue
        for i, raw in enumerate(first.value.value.split("\n")):
            line = raw.strip()
            if any(p.search(line) for p in _PATTERNS):
                found.append(Finding(CHECK, path, first.value.lineno + i,
                                     "history wording in a docstring (say what it is, not what it was)",
                                     line_hash(CHECK, line)))
    return found


def check(root, src=(), **_):
    found = []
    for p in package_files(root, src):
        try:
            found += scan_source((Path(root) / p).read_text(encoding="utf-8"), p)
        except (SyntaxError, UnicodeDecodeError):
            continue
    return found
