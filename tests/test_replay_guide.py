"""The replay guide's code blocks run (docs/replay-guide): a page's blocks in order in one namespace, a block whose
first line is ``# docs: skip`` left out (the hub, a token), on the CPU, in a temporary working directory. A block
that no longer runs is a page that lies."""
from __future__ import annotations
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "replay-guide"
FENCE = re.compile(r"^```python[^\n]*\n(.*?)^```", re.S | re.M)
PAGES = sorted(GUIDE.rglob("*.md"))


def blocks(page: Path):
    text = page.read_text(encoding="utf-8")
    for m in FENCE.finditer(text):
        yield text[: m.start()].count("\n") + 2, m.group(1)


@pytest.mark.parametrize("page", PAGES, ids=[str(p.relative_to(GUIDE)) for p in PAGES])
def test_the_page_runs(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    ns = {"__name__": "__docs__", "__file__": str(page)}
    for line, src in blocks(page):
        first = src.lstrip().splitlines()[0] if src.strip() else ""
        if first.startswith("# docs: skip"):
            continue
        exec(compile(src, f"{page.relative_to(ROOT)}:{line}", "exec"), ns)


def test_every_page_is_linked():
    text = (GUIDE / "README.md").read_text(encoding="utf-8")
    for page in PAGES:
        if page.name != "README.md":
            assert str(page.relative_to(GUIDE)) in text, page
