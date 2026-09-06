"""A trajectory producer returns one object, `PersistentWalk`, whatever it recorded.

`simulate_trajectories` and `simulate_mt_trajectories` return the same dataclass: positions and
the step description always, the surface, compartment and binding channels as attributes that are
``None`` when not recorded, and the rejected-crossing count on the object rather than in a module
global. The bank reads it directly.
"""
import dataclasses

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine import core
from dmipy_sim.replay.bank import _master_arrays

D = 2e-9


def test_the_default_walk_records_every_tier_the_geometry_supports():
    w = d.simulate_trajectories(64, D, d.Sphere(3e-6), 2e-3, 5e-4, seed=1, require_gpu=False)
    assert w.has_surface and w.has_compartments and not w.has_binding
    free = d.simulate_trajectories(64, D, d.FreeDiffusion(), 2e-3, 5e-4, seed=1, require_gpu=False)
    assert free.has_surface and (free.boundary_local_time == 0).all()      # no walls: an all-zero channel
    with pytest.raises(ValueError, match="tiers must be"):
        d.simulate_trajectories(64, D, d.Sphere(3e-6), 2e-3, 5e-4, seed=1, require_gpu=False, tiers="surface")


def test_a_positions_only_walk_carries_no_channels_and_the_step_description():
    w = d.simulate_trajectories(64, D, d.Sphere(3e-6), 2e-3, 5e-4, seed=1, require_gpu=False, tiers=())
    assert isinstance(w, d.PersistentWalk)
    assert w.positions.shape == (64, 5, 3) and w.positions.dtype == np.float32
    assert w.storage_dtype == np.float32 and w.n_walkers == 64 and w.n_t == 5
    assert w.dt == pytest.approx(5e-4) and w.T_max == pytest.approx(2e-3)
    assert w.dt_sim == pytest.approx(w.dt / w.sub_steps)
    assert w.boundary_local_time is None and w.compartment is None and w.bound_frac is None
    assert not (w.has_surface or w.has_compartments or w.has_binding)
    assert w.illegal_crossings == 0 and w.seed == 1
    assert not hasattr(core, "LAST_ILLEGAL_CROSSINGS")


def test_relaxation_walk_carries_surface_and_compartment_channels():
    w = d.simulate_trajectories(64, D, d.Sphere(3e-6, surface_relaxivity_t2=1e-6), 2e-3, 5e-4,
                                seed=1, require_gpu=False)
    assert w.has_surface and w.has_compartments and not w.has_binding
    assert w.boundary_local_time.shape == (64, 5) and (w.boundary_local_time <= 0).all()
    assert w.compartment.shape == (64, 5) and (w.compartment == 1).all()     # all inside the sphere
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.dt = 1.0


def test_the_two_producers_return_the_same_shape_of_object():
    g = d.Sphere(2e-6, surface_relaxivity_t2=1e-6)
    plain = d.simulate_trajectories(50, D, g, 2e-3, 5e-4, seed=3, require_gpu=False)
    mt = d.simulate_mt_trajectories(50, D, g, 2e-3, 5e-4, kappa_MT=0.0, dwell_time=0.0,
                                    equilibrate_binding="off", seed=3, require_gpu=False)
    assert isinstance(mt, d.PersistentWalk) and mt.has_binding and mt.has_surface
    np.testing.assert_array_equal(mt.positions, plain.positions)
    np.testing.assert_array_equal(mt.boundary_local_time, plain.boundary_local_time)
    assert (mt.bound_frac == 0).all() and mt.compartment is None


def test_the_bank_reads_a_persistent_walk_directly():
    w = d.simulate_trajectories(40, D, d.Cylinder(2e-6, (0, 0, 1)), 2e-3, 5e-4, seed=2, require_gpu=False)
    m = _master_arrays(w)
    assert m["traj"] is w.positions and m["dt_traj"] == w.dt and m["T_max"] == pytest.approx(w.T_max)
    assert m["dlog_b"] is w.boundary_local_time and m["comp"] is w.compartment and m["bfrac"] is None
    assert m["n_walkers"] == 40 and m["seed"] == 2
    m2 = _master_arrays(w._bank_dict(T2_per_comp=[0.05, 0.08], w=np.ones(40)))
    assert m2["T2_per_comp"].tolist() == [0.05, 0.08] and m2["w"].shape == (40,)
    with pytest.raises(TypeError, match="PersistentWalk"):
        _master_arrays((w.positions, w.dt))
