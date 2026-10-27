"""The paper families' re-walk driver (``tools/family_rewalk.py``): its walker rule holds the toggle-proof bound
at the floor it is given and is the cheapest set of counts that does, and the envelope it certifies under is
the family's own declaration over the campaign's."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fr():
    spec = importlib.util.spec_from_file_location("family_rewalk", ROOT / "tools" / "family_rewalk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_equal_costs_seed_in_proportion_to_the_water_and_hold_the_floor(fr):
    shares = {"intra": 0.3, "myelin": 0.2, "extra": 0.5}
    n = fr.toggle_proof_counts(shares, 5e-3)
    assert n == {"intra": 12000, "myelin": 8000, "extra": 20000}
    assert fr.toggle_proof_bound(shares, n) <= 5e-3 * (1 + 1e-12)
    assert fr.toggle_proof_counts({"water": 1.0}, 1e-2) == {"water": 10000}


def test_the_cost_weighted_counts_are_the_cheapest_that_hold_the_floor(fr):
    shares, costs, f = {"a": 0.4, "b": 0.6}, {"a": 9.0, "b": 1.0}, 1e-2
    n = fr.toggle_proof_counts(shares, f, costs)
    assert fr.toggle_proof_bound(shares, n) <= f * (1 + 1e-9)
    cost = lambda m: sum(m[p] * costs[p] for p in m)
    for t in np.linspace(0.2, 0.8, 13):                     # every other split on the bound's own surface costs more
        na = shares["a"] ** 2 / (t * f ** 2)
        nb = shares["b"] ** 2 / ((1 - t) * f ** 2)
        assert cost(n) <= cost({"a": na, "b": nb}) * (1 + 1e-3)


def test_a_floor_or_share_that_is_not_positive_is_refused(fr):
    with pytest.raises(ValueError):
        fr.toggle_proof_counts({"water": 1.0}, 0.0)
    with pytest.raises(ValueError):
        fr.toggle_proof_counts({"water": 0.0}, 1e-2)
    with pytest.raises(ValueError):
        fr.toggle_proof_counts({"a": 1.0}, 1e-2, {"a": 0.0})


def test_a_family_declaration_is_stated_over_the_campaign_envelope(fr):
    env = fr.family_envelope({"rho2_list": [1e-5], "bvals": [0.0, 1e9]}, {"B0_list": [3.0]}, max_refocus_pulses=8)
    assert env["scanner"] == fr.ENVELOPE["scanner"]
    assert env["rho2_list"] == [1e-5] and env["bvals"] == [0.0, 1e9]
    assert env["B0_list"] == [3.0] and env["max_refocus_pulses"] == 8
    assert "ogse_periods" in env                             # the library's battery shapes underneath
