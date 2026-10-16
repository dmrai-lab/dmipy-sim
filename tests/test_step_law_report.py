"""The step-law report builds (``tools/gen_step_law.py``, the evidence page ``docs/step-law.md``): the small tier
runs end to end on the CPU and the cases it measures match what the page claims -- the pair's moments, the slab's
exactness under the pair, the wall constants of the cylinder and the sphere, the fused engine inside the operator's
standard error, the certificate resolving a tolerance, and the phase rule's sign. A page that no longer builds, or
whose small cases drift, is a page that lies."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    out = tmp_path_factory.mktemp("step_law") / "step-law.md"
    subprocess.run([sys.executable, str(ROOT / "tools" / "gen_step_law.py"), "--tier", "small", "--out", str(out)],
                   check=True, cwd=ROOT, timeout=1800)
    return out, json.loads((out.parent / "step-law" / "step_law.json").read_text())


def test_the_page_and_its_figures_build(report):
    out, R = report
    text = out.read_text()
    for name in ("laws", "operator", "fused", "certificate", "cost", "outside"):
        assert name in R, name
    for fig in ("laws", "operator", "fused", "certificate"):
        assert (out.parent / "step-law" / f"{fig}.png").stat().st_size > 1000
        assert f"step-law/{fig}.png" in text
    assert R["meta"]["tier"] == "small" and R["meta"]["commit"]


def test_the_laws_match_their_moments(report):
    _, R = report
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


def test_the_operator_reads_the_wall_constants(report):
    _, R = report
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


def test_the_engine_sits_inside_the_operator(report):
    _, R = report
    for key, rec in R["fused"].items():
        for sub, r in rec["steps"].items():
            assert abs(r["pull"]) < 4.0, (key, sub, r)


def test_the_certificate_resolves_at_the_top_of_the_ladder(report):
    """The ladder's floor falls with the walker count; at its top the loosened tolerance is resolved and held, and
    the run at the rule's own tolerance says it does not resolve it, since the record never claims what the
    pilot cannot see."""
    _, R = report
    lad = R["certificate"]["ladder"]
    ratio = lad[0]["ratio"]
    rungs = [r for r in lad if r["ratio"] == ratio]
    assert rungs[-1]["floor"] < rungs[0]["floor"]
    top = rungs[-1]
    assert top["resolves"] and top["floor"] < top["eps_declared"]
    assert top["eps_measured"] < 1.1 * top["eps_declared"] + 3 * top["eps_se"]
    own = [r for r in lad if r["ratio"] == 1]
    assert own and not own[0]["resolves"] and own[0]["floor"] > own[0]["eps_declared"]


def test_the_cost_table_names_what_binds(report):
    _, R = report
    for name, r in R["cost"].items():
        assert r["sub_steps"] == max(r["rules"].values()) and r["binds"] in r["rules"]
        assert r["walker_steps_per_s"] > 0 or "not_walked" in r
    assert "lookup_longest_atom" in R["cost"]["icosphere mesh 4 um"]


def test_the_phase_rule_is_the_slab_s_remaining_term(report):
    """On the slab the pair's dispersion is below the phase rule's error and falls faster than its square with the
    step; the phase rule's error falls as the square of the step (a quarter of the step: a sixteenth of the error)."""
    _, R = report
    for label, rec in R["outside"].items():
        s1, s4 = rec["steps"]["1"], rec["steps"]["4"]
        assert abs(s1["dispersion"]) < 0.5 * abs(s1["phase_rule"]), (label, s1)
        assert abs(s4["dispersion"]) < 0.1 * abs(s1["dispersion"]) + 1e-7, (label, s4)
        assert 1 / 32 < abs(s4["phase_rule"]) / abs(s1["phase_rule"]) < 1 / 8, (label, s1, s4)
    assert "fixed" in R["operator"]["pores"]["plane"]["levels"]["0.5"]["coefficient"]
