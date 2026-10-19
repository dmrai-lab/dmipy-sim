"""The replay guide's code blocks run (docs/replay-guide): a page's blocks in order in one namespace, a block whose
first line is ``# docs: skip`` left out (the hub, over the network), on the CPU, in a temporary working directory. A block
that no longer runs is a page that lies.

A page that walks with ``simulate_trajectories`` runs on both bases a pack's bands are on: as written (the walk's saves,
packed on their DST-I) and with the walk accumulated (``accumulate=``, the continuous basis), its pack built from the
coefficient walk at the band the page asks for."""
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


#: the bands a page's walk accumulates on the continuous basis: at least the band any page packs at
GUIDE_ACCUMULATE = 32


def _walks(page):
    return "simulate_trajectories(" in page.read_text(encoding="utf-8")


CASES = [(p, b) for p in PAGES for b in (("dst1", "continuous") if _walks(p) else ("dst1",))]


@pytest.mark.parametrize("page,basis", CASES, ids=[f"{p.relative_to(GUIDE)}-{b}" for p, b in CASES])
def test_the_page_runs(page, basis, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    if basis == "continuous":
        import dmipy_sim
        walk = dmipy_sim.simulate_trajectories

        def accumulated(*args, **kw):
            kw.setdefault("accumulate", GUIDE_ACCUMULATE)
            return walk(*args, **kw)
        monkeypatch.setattr(dmipy_sim, "simulate_trajectories", accumulated)
    ns = {"__name__": "__docs__", "__file__": str(page)}
    for line, src in blocks(page):
        first = src.lstrip().splitlines()[0] if src.strip() else ""
        if first.startswith("# docs: skip"):
            continue
        exec(compile(src, f"{page.relative_to(ROOT)}:{line}", "exec"), ns)
    if "walk" in ns and "pack" in ns:                                  # the page packed the walk it made, on the basis asked
        from dmipy_sim.replay.compression import basis_of
        assert basis_of(ns["pack"].meta) == basis, (basis_of(ns["pack"].meta), basis)


def test_every_page_is_linked():
    text = (GUIDE / "README.md").read_text(encoding="utf-8")
    for page in PAGES:
        if page.name != "README.md":
            assert str(page.relative_to(GUIDE)) in text, page
