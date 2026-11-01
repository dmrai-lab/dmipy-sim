"""The step-law report builds (``tools/gen_step_law.py``, the evidence page ``docs/step-law.md``).

The page is a report generator, not a unit: the ``tiny`` tier runs every section end to end in about a minute and
the tests below check the code path (every section, every figure, the page links them) and the numbers the tiny tier
resolves: the pair's moments, the wall constants of the cylinder and the sphere, the fused engine inside the
operator's standard error, the cost table. The certificate ladder and the phase rule need the ``small`` tier's walkers
and quadrature (about twenty minutes on a CPU), so the evidence page records them and no test does."""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _build(tier, tmp_path_factory, timeout):
    out = tmp_path_factory.mktemp("step_law") / "step-law.md"
    subprocess.run([sys.executable, str(ROOT / "tools" / "gen_step_law.py"), "--tier", tier, "--out", str(out)],
                   check=True, cwd=ROOT, timeout=timeout, env={**os.environ, "PYTHONPATH": str(ROOT)})
    return out, json.loads((out.parent / "step-law" / "step_law.json").read_text())


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return _build("tiny", tmp_path_factory, 600)


def test_the_tiny_tier_builds_every_section_and_figure(tiny):
    out, R = tiny
    text = out.read_text()
    for name in ("laws", "operator", "fused", "certificate", "cost", "outside"):
        assert name in R, name
    for fig in ("laws", "operator", "fused", "certificate"):
        assert (out.parent / "step-law" / f"{fig}.png").stat().st_size > 1000
        assert f"step-law/{fig}.png" in text
    assert R["meta"]["tier"] == "tiny" and R["meta"]["commit"]


def test_the_laws_match_their_moments(tiny):
    """The laws section does not depend on the tier (it costs a second), so the fast job checks it."""
    _, R = tiny
    laws = R["laws"]["laws"]
    assert laws["fixed"]["m4"] == pytest.approx(0.6)
    for name, matched in (("equal pair", 2), ("Gauss pair", 3), ("Gauss triple", 3)):
        m = laws[name]
        assert m["m2"] == pytest.approx(1.0, abs=1e-9) and m["m4"] == pytest.approx(1.0, abs=1e-9)
        if matched == 3:
            assert m["m6"] == pytest.approx(1.0, abs=1e-9)
    assert laws["Gauss pair"]["reach_1e3"] == pytest.approx(1.79, abs=0.03)
    assert laws["fixed"]["reach_1e3"] == pytest.approx(0.17, abs=0.02)
    e = R["laws"]["engine"]
    assert e["m2"] == pytest.approx(1.0, abs=1e-2) and e["m4"] == pytest.approx(1.0, abs=3e-2)
    assert e["longest"] <= laws["Gauss pair"]["longest"] * (1 + 1e-5) and len(e["lengths"]) == 2


def test_the_operator_reads_the_wall_constants(tiny):
    _, R = tiny
    pores = R["operator"]["pores"]
    plane = pores["plane"]["levels"]["0.5"]["coefficient"]
    assert abs(plane["Gauss pair"][0]) < 0.02 and plane["fixed"][0] == pytest.approx(1.0, abs=0.08)
    cyl = pores["cylinder"]["levels"]["0.5"]["coefficient"]
    sph = pores["sphere"]["levels"]["0.5"]["coefficient"]
    assert cyl["Gauss pair"][0] == pytest.approx(0.5, abs=0.08) and sph["Gauss pair"][0] == pytest.approx(0.85, abs=0.08)
    assert cyl["Gaussian (6 atoms)"][0] == pytest.approx(cyl["Gauss pair"][0], abs=0.03)
    assert cyl["fixed"][0] > 1.1 and sph["fixed"][0] > 1.3
    for pool, r in R["operator"]["packing"].items():
        assert r["eps_measured"] <= r["eps_declared"] + 3 * r["eps_se"], pool


def test_the_engine_sits_inside_the_operator(tiny):
    _, R = tiny
    for key, rec in R["fused"].items():
        for sub, r in rec["steps"].items():
            assert abs(r["pull"]) < 4.0, (key, sub, r)


def test_the_cost_table_names_what_binds(tiny):
    _, R = tiny
    for name, r in R["cost"].items():
        assert r["sub_steps"] == max(r["rules"].values()) and r["binds"] in r["rules"]
        assert r["walker_steps_per_s"] > 0 or "not_walked" in r
    assert "lookup_longest_atom" in R["cost"]["icosphere mesh 4 um"]
