"""The cookbook's rungs run (examples/).

An example that no longer executes is documentation that lies, and these are the files a newcomer is told to
start from. Parts I and II are pure CPU, a second or two each; the heavier parts of the cookbook are not run here
and are listed in `examples/README.md` as such.
"""
from __future__ import annotations

import pathlib
import runpy

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNGS = sorted((ROOT / "examples").glob("*/[0-9][0-9]_*.py"))

#: A rung that needs an optional interop dependency, and which one. The rung still runs wherever the
#: dependency is installed; where it is not, the skip says so rather than the ladder going red for a
#: reason that has nothing to do with the engine.
OPTIONAL_DEPENDENCY = {"19_pulseq_in_and_out": "pypulseq"}


@pytest.mark.parametrize("rung", RUNGS, ids=[p.stem for p in RUNGS])
def test_the_rung_runs(rung, monkeypatch, tmp_path, capsys):
    needs = OPTIONAL_DEPENDENCY.get(rung.stem)
    if needs:
        pytest.importorskip(needs)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    runpy.run_path(str(rung), run_name="__cookbook__")
    assert capsys.readouterr().out.strip(), f"{rung.name} printed nothing: a rung should show its result"


def test_every_rung_is_on_the_ladder():
    index = (ROOT / "examples" / "README.md").read_text()
    for rung in RUNGS:
        assert rung.name in index, f"{rung.name} is not listed in examples/README.md"
