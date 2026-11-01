"""Findings, source-file discovery and the ``.guards-allow`` list.

An allow-list line is ``<path>:<line-hash> <reason>``; the hash is printed with every finding, so accepting a case
means copying that token and stating why. The hash covers the guard name and the line's text with whitespace
collapsed, so an entry survives the line moving and stops matching when the line changes."""
import hashlib
import subprocess
from pathlib import Path
from typing import NamedTuple

ALLOW_FILE = ".guards-allow"
NOT_PACKAGE_CODE = {"tests", "test", "tools", "examples", "docs", "benchmarks", "notebooks", "scripts"}


class Finding(NamedTuple):
    check: str
    path: str
    line: int
    reason: str
    key: str = ""

    def __str__(self):
        tail = f"  [allow: {self.path}:{self.key}]" if self.key else ""
        return f"{self.path}:{self.line}: {self.check}: {self.reason}{tail}"


def line_hash(check, text):
    return hashlib.sha1(f"{check}\0{' '.join(text.split())}".encode()).hexdigest()[:10]


def tracked(root):
    """Tracked files of the repository as repo-relative POSIX paths (all files when ``root`` is not a git checkout)."""
    root = Path(root)
    run = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True)
    if run.returncode == 0:
        return [p for p in run.stdout.splitlines() if p]
    return [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]


def package_files(root, src=()):
    """Tracked ``.py`` package files: those under ``src`` (files or directories), else everything outside the
    test, tool, example and documentation directories. Test files are always excluded."""
    out = []
    for p in tracked(root):
        parts = p.split("/")
        if not p.endswith(".py") or parts[-1].startswith("test_") or parts[-1] == "conftest.py":
            continue
        if "tests" in parts[:-1] or "test" in parts[:-1]:
            continue
        if src:
            if any(p == s.rstrip("/") or p.startswith(s.rstrip("/") + "/") for s in src):
                out.append(p)
        elif not NOT_PACKAGE_CODE.intersection(parts[:-1]) and len(parts) > 1:
            out.append(p)
    return out


def load(root):
    """``{(path, hash): reason}`` from the allow-list; a line without a reason is an error."""
    f = Path(root) / ALLOW_FILE
    allowed = {}
    if not f.exists():
        return allowed
    for n, raw in enumerate(f.read_text().splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        head, _, reason = raw.strip().partition(" ")
        path, _, key = head.rpartition(":")
        if not (path and key and reason.strip()):
            raise ValueError(f"{ALLOW_FILE}:{n}: expected '<path>:<line-hash> <reason>'")
        allowed[(path, key)] = reason.strip()
    return allowed


def unallowed(findings, root):
    allowed = load(root)
    return [f for f in findings if (f.path, f.key) not in allowed]


def record(findings, root, reason):
    """Append the findings to the allow-list with ``reason`` (the one-time baseline of a repository); returns the count."""
    have = set(load(root))
    new = sorted({(f.path, f.key) for f in findings if f.key} - have)
    with open(Path(root) / ALLOW_FILE, "a") as out:
        out.writelines(f"{p}:{k} {reason}\n" for p, k in new)
    return len(new)
