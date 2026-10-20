"""The raw subset of a coefficient walk, viewed at its block ends: ``CoefficientWalk.saves()`` is a
`PersistentWalk` of the first ``raw_subset`` walkers -- positions and the step description always, the contact,
pool label and bound state at every block end as attributes that are ``None`` when not recorded -- the dense
oracle of a measured certificate and of the tests, never a producer's output and never a pack's input.
"""
import dataclasses

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.coefficient_walk import CoefficientWalk, events_at_instants
from dmipy_sim.engine import core

D = 2e-9


def _saves(n, *a, **kw):
    """The walk with every walker's block ends kept, viewed as a walk of saves."""
    return d.simulate_trajectories(n, *a, raw_subset=n, **kw).saves()


def test_the_default_walk_records_every_tier_the_geometry_supports():
    w = d.simulate_trajectories(64, D, d.Sphere(3e-6), 2e-3, 5e-4, seed=1, require_gpu=False)
    assert isinstance(w, CoefficientWalk) and w.has_surface and w.has_compartments and not w.has_binding and not w.crossing
    free = d.simulate_trajectories(64, D, d.FreeDiffusion(), 2e-3, 5e-4, seed=1, require_gpu=False)
    assert not free.has_surface and free.boundary_bands is None      # no walls: nothing to record, no channel
    with pytest.raises(ValueError, match="tiers must be"):
        d.simulate_trajectories(64, D, d.Sphere(3e-6), 2e-3, 5e-4, seed=1, require_gpu=False, tiers="surface")


def test_a_positions_only_walk_carries_no_channels_and_the_step_description():
    w = _saves(64, D, d.Sphere(3e-6), 2e-3, 5e-4, seed=1, require_gpu=False, tiers=())
    assert isinstance(w, d.PersistentWalk)
    assert w.positions.shape == (64, 5, 3) and w.positions.dtype == np.float32
    assert w.n_walkers == 64 and w.n_t == 5
    assert w.dt == pytest.approx(5e-4) and w.T_max == pytest.approx(2e-3)
    assert w.dt_sim == pytest.approx(w.dt / w.sub_steps)
    assert w.boundary_local_time is None and w.compartment is None and w.bound is None
    assert not (w.has_surface or w.has_compartments or w.has_binding)
    assert w.illegal_crossings == 0 and w.seed == 1
    assert not hasattr(core, "LAST_ILLEGAL_CROSSINGS")


def test_relaxation_walk_carries_surface_and_compartment_channels():
    cw = d.simulate_trajectories(64, D, d.Sphere(3e-6, surface_relaxivity_t2=1e-6), 2e-3, 5e-4, seed=1, require_gpu=False, raw_subset=64)
    w = cw.saves()
    assert w.has_surface and w.has_compartments and not w.has_binding
    assert w.boundary_local_time.shape == (64, 5) and (w.boundary_local_time <= 0).all()
    assert w.compartment.shape == (64, 5) and (w.compartment == 1).all()     # all inside the sphere
    np.testing.assert_allclose(w.boundary_local_time.sum(1), np.asarray(cw.boundary_endpoint)[:, 0], rtol=1e-6)   # the endpoint: float32 sums
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.dt = 1.0
    with pytest.raises(ValueError, match="raw_subset"):
        d.simulate_trajectories(8, D, d.Sphere(3e-6), 2e-3, 5e-4, seed=1, require_gpu=False).saves()


def test_the_two_producers_return_the_same_object():
    g = d.Sphere(2e-6, surface_relaxivity_t2=1e-6)
    plain = d.simulate_trajectories(50, D, g, 2e-3, 5e-4, seed=3, require_gpu=False, f_band=0.5 / 5e-4, raw_subset=50)
    mt = d.simulate_mt_trajectories(50, D, g, 2e-3, 5e-4, kappa_MT=0.0, dwell_time=0.0,
                                    equilibrate_binding="off", seed=3, require_gpu=False, raw_subset=50)
    assert isinstance(mt, CoefficientWalk) and not mt.has_binding and mt.has_surface      # no binding at kappa = 0: the plain walk
    np.testing.assert_array_equal(mt.raw_positions, plain.raw_positions)
    np.testing.assert_array_equal(mt.position_coeffs, plain.position_coeffs)
    np.testing.assert_array_equal(mt.boundary_bands, plain.boundary_bands)
    assert mt.bound_start is None and mt.bound_events is None and mt.saves().bound is None
    bound = d.simulate_mt_trajectories(50, D, g, 2e-3, 5e-4, kappa_MT=1e-5, dwell_time=1e-4, equilibrate_binding="off",
                                       seed=3, require_gpu=False, raw_subset=50)
    assert bound.has_binding and bound.bound_start.shape == (50,) and (bound.bound_start == 0).all()        # off: every walker starts free
    np.testing.assert_array_equal(events_at_instants(bound.bound_events, bound.bound_start, np.arange(5) * 5e-4), bound.raw_bound)


def test_the_bank_refuses_the_view():
    from dmipy_sim.replay.bank import build_replay_pack
    w = _saves(40, D, d.Cylinder(2e-6, (0, 0, 1)), 2e-3, 5e-4, seed=2, require_gpu=False)
    with pytest.raises(TypeError, match="CoefficientWalk"):
        build_replay_pack(w, id="x", K=4, license="x", citation="x")
