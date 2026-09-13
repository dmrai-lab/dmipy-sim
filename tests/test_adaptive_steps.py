"""Adaptive stepping (engine/adaptive.py) against the fused producer on a strand pack: the finest class alone
is the fused walk to the bit; with free steps and radius classes the guarantees hold (no tube entered or left,
the box kept, contact non-positive) and the statistics agree within the Monte-Carlo floor."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine.adaptive import simulate_trajectories_adaptive

D = 2e-9


def _pack(interior):
    cls = [np.array([[x, 0, -30e-6], [x, 0.5e-6, 0], [x, 0, 30e-6]]) for x in (-6e-6, 0, 6e-6)]
    return d.PackedCurvedCylinders(cls, [1.0e-6, 1.5e-6, 2.0e-6], interior=interior, box=(np.full(3, -10e-6), np.full(3, 10e-6)))


def test_the_finest_class_alone_is_the_fused_walk():
    g = _pack(False)
    fixed = d.simulate_trajectories(300, D, g, 1e-3, 2.5e-4, seed=3, require_gpu=False, sub_steps=64)
    adaptive = simulate_trajectories_adaptive(300, D, g, 1e-3, 2.5e-4, seed=3, require_gpu=False, sub_steps=64,
                                              steps_per_round=64, n_classes=1, safety_sigma=1e9, walker_batch_size=300)
    assert adaptive.sub_steps == fixed.sub_steps == 64
    # the same noise sequence and the same wall rule; the gathered kernel compiles to a different op order, so
    # float32 rounding differs by ulps (1e-11 m on 1e-5 m coordinates), not by a step
    np.testing.assert_allclose(adaptive.positions, fixed.positions, rtol=0, atol=1e-10)
    np.testing.assert_allclose(adaptive.boundary_local_time, fixed.boundary_local_time, rtol=1e-5, atol=1e-10)
    np.testing.assert_array_equal(adaptive.compartment, fixed.compartment)


@pytest.mark.parametrize("interior", [False, True])
def test_the_guarantees_and_the_statistics(interior):
    g = _pack(interior)
    n = 6000
    a = simulate_trajectories_adaptive(n, D, g, 2e-3, 2.5e-4, seed=1, require_gpu=False, walker_batch_size=n)
    f = d.simulate_trajectories(n, D, g, 2e-3, 2.5e-4, seed=2, require_gpu=False)
    P = a.positions.reshape(-1, 3)
    if interior:
        assert g.inside_any(P).all()                                   # never left its tube
    else:
        assert not g.inside_any(P).any()                               # never entered one
    assert (np.abs(a.positions) <= 10e-6 + 1e-9).all() and (a.boundary_local_time <= 0).all() and a.illegal_crossings == 0
    assert a.stepping["kernel_steps_ratio"] > 1.2 and (interior or a.stepping["free_fraction"] > 0.05)   # intra gains by class, not by distance
    # end-to-end displacement variance per axis and the accumulated contact, within the floor of n walkers
    da = a.positions[:, -1] - a.positions[:, 0]; df = f.positions[:, -1] - f.positions[:, 0]
    va, vf = (da ** 2).mean(0), (df ** 2).mean(0)
    assert np.abs(va / vf - 1.0).max() < 5 * np.sqrt(2.0 / n) * 1.5, va / vf
    ca, cf = a.boundary_local_time.sum(1), f.boundary_local_time.sum(1)
    se = np.sqrt(ca.var() / n + cf.var() / n)
    assert abs(ca.mean() - cf.mean()) < 4 * se, (ca.mean(), cf.mean(), se)


def test_a_geometry_without_wall_scales_is_refused():
    with pytest.raises(TypeError, match="wall_scales"):
        simulate_trajectories_adaptive(10, D, d.Sphere(3e-6), 1e-3, 5e-4, require_gpu=False)


def test_the_candidate_cache_is_the_full_gather():
    """A round stepped against the segments within its reach is the round stepped against the 27-cell gather:
    the same walls are met, so the same positions and contact (to rounding), from the same noise."""
    g = _pack(False)
    kw = dict(seed=5, require_gpu=False, walker_batch_size=2000)
    full = simulate_trajectories_adaptive(2000, D, g, 2e-3, 2.5e-4, candidate_cache=False, **kw)
    cache = simulate_trajectories_adaptive(2000, D, g, 2e-3, 2.5e-4, candidate_cache=True, candidate_k_start=4, **kw)
    assert cache.stepping["candidate_cache"] and cache.stepping["candidate_k"] >= 4
    # the same segments decide every reflection; the two programs round differently at float32, and a grazing
    # step within an ulp of a wall may flip between hit and miss -- a handful of (walker, save) pairs, by less
    # than one step (0.13 um), never a wall crossed
    diff = np.abs(cache.positions - full.positions).max(-1)
    assert (diff > 1e-10).mean() < 2e-3 and diff.max() < 0.13e-6, ((diff > 1e-10).mean(), diff.max())
    dl = np.abs(cache.boundary_local_time - full.boundary_local_time)
    assert (dl > 1e-10).mean() < 2e-3 and not g.inside_any(cache.positions.reshape(-1, 3)).any()
