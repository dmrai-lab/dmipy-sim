"""Excluding the walkers that changed pool in an impermeable bundle walk (``walk(drop_pool_changes=)``): the walk
without them, the pool's weights renormalised to its total, the refusals (past the bound, a raw walker among them),
and the record the pack's certificate carries. The mask is the backend's (`CoefficientWalk.pool_changed`); here it is
set on a small JAX walk, which is what a backend hands the bundle walk."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from dmipy_sim import Sphere
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec import SpecError
from dmipy_sim.spec.walk import DROP_POOL_CHANGE_RULE, _drop_pool_changes

ENV = dict(bvals=[0.0, 1e9], dirs=[[0, 0, 1]], ogse_periods=[2], shortd_b=1e9, shortd_deltas_frac=[0.05],
           B0_list=[3.0], theta_deg=[90], delta_frac=0.2, Delta_frac=0.5, rho2_list=[1e-5])


def _walk(n=400, mask_at=(5, 17, 300)):
    w = _walk_geometry(n, 2e-9, Sphere(radius=4e-6), 2e-3, seed=1, K=8, require_gpu=False)
    mask = np.zeros(n, bool); mask[list(mask_at)] = True
    return dataclasses.replace(w, pool_changed=mask), mask


def test_the_walkers_that_changed_pool_are_taken_out_and_the_pool_keeps_its_weight():
    w, mask = _walk()
    weights = np.ones(w.n_walkers)
    kept, wts, rec = _drop_pool_changes(w, weights, 0, "intra", 0.01)
    assert kept.n_walkers == w.n_walkers - 3 and rec == dict(n_walked=400, n_dropped=3, fraction=3 / 400)
    np.testing.assert_array_equal(kept.position_coeffs, w.position_coeffs[~mask])
    assert wts.shape == (397,) and wts.sum() == pytest.approx(weights.sum())


def test_past_the_bound_or_through_the_raw_subset_the_walk_is_refused():
    w, _ = _walk(mask_at=tuple(range(10)))
    with pytest.raises(SpecError, match="more than the 1.000%"):
        _drop_pool_changes(w, np.ones(w.n_walkers), 0, "extra", 0.01)
    w, _ = _walk(mask_at=(2,))
    with pytest.raises(SpecError, match="raw walkers changed pool"):
        _drop_pool_changes(w, np.ones(w.n_walkers), 50, "extra", 0.01)


def test_a_walk_without_the_mask_is_refused():
    w, _ = _walk()
    with pytest.raises(SpecError, match="no pool-change mask"):
        _drop_pool_changes(dataclasses.replace(w, pool_changed=None), np.ones(w.n_walkers), 0, "intra", 0.01)


def test_the_pack_certificate_states_the_exclusion():
    w, _ = _walk()
    kept, _, rec = _drop_pool_changes(w, np.ones(w.n_walkers), 0, "intra", 0.01)
    record = dict(rule=DROP_POOL_CHANGE_RULE, max_fraction=0.01, n_dropped_pool_change=3, fraction=3 / 400,
                  pools={"intra": rec})
    kept = dataclasses.replace(kept, work=dict(kept.work or {}, dropped_pool_change=record))
    pk = build_replay_pack(kept, id="test/drop", license="x", citation="x", K=8, envelope=ENV)
    assert pk.meta["fidelity"]["dropped_pool_change"] == record


def test_the_jax_kernels_refuse_the_request():
    with pytest.raises(ValueError, match="JAX kernels carry each walker's label"):
        _walk_geometry(10, 2e-9, Sphere(radius=4e-6), 2e-3, K=8, require_gpu=False, drop_pool_changes=True)
