"""`simulate` routes on a capability the geometry declares, `replay_parity`, not on its class name.

A geometry whose producer walk is the fused walk (validated in test_replay_parity.py) says so;
``simulate`` then serves it from a coefficient walk. Everything else, and every run the producer cannot walk,
runs the fused oracle.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine import core
from dmipy_sim.geometry.analytic import PermeableShell, PermeableSlab1D

D = 2e-9
WF = d.set_b(d.pgse([[1, 0, 0]], 3e-3, 8e-3, gradient_strengths=0.1, n_t=40, slew_rate=np.inf), 5e8)


def _programs_used(geometry):
    """Which batch programs a run built on the geometry: 'coefficient walk' keys are the producer,
    'standard' the oracle's scan."""
    return {k[0][0] for k in vars(geometry).get(core._BATCH_CACHE_ATTR, {})}


def test_parity_is_declared_on_the_validated_families_only():
    yes = (d.FreeDiffusion, d.Box1D, d.Sphere, d.Cylinder, d.Ellipsoid, d.PackedCylinders, d.PackedSpheres, d.Mesh)
    no = (d.MyelinatedCylinder, d.PackedMyelinatedCylinders, d.CurvedCylinder, d.PackedCurvedCylinders,
          PermeableSlab1D, PermeableShell)
    assert all(g.replay_parity for g in yes) and not any(g.replay_parity for g in no)
    assert not hasattr(core, "_REPLAY_AUTO_GEOM_NAMES") and not hasattr(core, "_replay_auto_allowed")


def test_simulate_follows_the_declaration_not_the_name():
    sphere = d.Sphere(3e-6)
    d.simulate(200, D, WF, sphere, seed=0, require_gpu=False)
    assert _programs_used(sphere) >= {"coefficient walk"} and "standard" not in _programs_used(sphere)

    class Renamed(d.Sphere):                       # a subclass keeps the capability with the walk
        pass
    r = Renamed(3e-6)
    d.simulate(200, D, WF, r, seed=0, require_gpu=False)
    assert "coefficient walk" in _programs_used(r)

    class Unvalidated(d.Sphere):                   # ...and can renounce it
        replay_parity = False
    u = Unvalidated(3e-6)
    d.simulate(200, D, WF, u, seed=0, require_gpu=False)
    assert _programs_used(u) == {"standard"}


def test_a_run_the_producer_cannot_walk_is_the_oracles_even_with_parity():
    """No diffusivity: the producer sizes its step from D, the oracle reads it from the geometry."""
    s = d.Sphere(3e-6)
    assert core._replay_gap(s, None) is not None and core._replay_gap(s, D) is None
    mc = d.MyelinatedCylinder(
        2e-6,
        3e-6,
        (0, 0, 1),
        compartments=d.Compartments(intra=d.Pool(D=D, water_fraction=1.0), myelin=d.Pool(water_fraction=0.0), extra=d.Pool(D=D, water_fraction=0.0)),
    )
    d.simulate(200, None, WF, mc, seed=0, require_gpu=False)
    assert "coefficient walk" not in _programs_used(mc) and _programs_used(mc)
