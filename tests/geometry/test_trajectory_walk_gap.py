"""The single source of truth for whether `simulate_trajectories` can walk a geometry at all:
`Geometry.trajectory_walk_gap()`. Both entry points that reach the engine's trajectory producer
-- `engine.core.simulate_trajectories` directly, and `spec.walk.walk_spec` on a single-geometry
spec -- consult this one method before any device work, so a spec whose geometry has no
persistent walk is refused by name instead of discovered by a probe walk.

No production geometry is currently unwalkable (both myelinated-cylinder classes have their own
trajectory step), so the generic refusal path is exercised with a stub.
"""
import jax.numpy as jnp
import pytest

import dmipy_sim as d
import dmipy_sim.engine.core as core
import dmipy_sim.spec.walk as walk_mod
from dmipy_sim.geometry.base import Geometry, LengthScales
from dmipy_sim.spec import walk_spec
from dmipy_sim.spec.substrate import SpecError

D = 2e-9


class _Unwalkable(Geometry):
    """A minimal stub with no persistent walk, for testing the refusal mechanism generically.

    ``_spec_source`` is set to a sentinel so ``spec.build.as_geometry`` -- which every driver
    calls first, and which otherwise needs a real ``spec_of`` registration -- takes the door
    already open rather than trying to build one for a stub class it has never heard of.
    """
    _spec_source = "stub, unused"

    def trajectory_walk_gap(self):
        return "it is a test stub with no boundary rule"

    def init_positions(self, n_walkers, key):
        return jnp.zeros((n_walkers, 3), jnp.float32)

    def reflect(self, r, step):
        raise AssertionError("reflect was called: the gap check should have refused the walk first")

    @property
    def length_scales(self):
        return LengthScales(min_feature=1e-6)


def test_the_default_is_walkable():
    assert d.Cylinder(3e-6, (0, 0, 1)).trajectory_walk_gap() is None


def test_simulate_trajectories_refuses_an_unwalkable_geometry_before_any_device_work(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("seed_walkers was called: a walk was attempted")
    monkeypatch.setattr(core, "seed_walkers", _boom)
    with pytest.raises(NotImplementedError, match=r"_Unwalkable.*test stub"):
        d.simulate_trajectories(8, D, _Unwalkable(), T_max=1e-3, dt_save=1e-3, require_gpu=False)


def test_walk_spec_refuses_an_unwalkable_geometry_before_calling_simulate_trajectories(monkeypatch):
    """walk_spec checks the geometry_from_spec() result's own trajectory_walk_gap() and raises
    without ever importing or calling engine.core.simulate_trajectories."""
    def _boom(*a, **k):
        raise AssertionError("simulate_trajectories was called: a walk was attempted")
    monkeypatch.setattr(core, "simulate_trajectories", _boom)
    monkeypatch.setattr(walk_mod, "geometry_from_spec", lambda spec: _Unwalkable())

    g = d.MyelinatedCylinder(2e-6, 2.8e-6, (0, 0, 1), D, D)   # any single-geometry spec will do
    with pytest.raises(SpecError, match="test stub"):
        walk_spec(g.spec, 8, T_max=1e-3)
