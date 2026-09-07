"""`engine="auto"` routes on a capability the geometry declares, `replay_parity`, not on its class name.

A geometry whose replay-producer walk is the fused walk (validated in test_replay_parity.py) says
so; auto then serves it from a persistent walk when nothing requested is fused-only. Everything
else runs fused, and an explicit engine="replay" is still honoured wherever the replay can serve.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine import core
from dmipy_sim.geometry.analytic import PermeableShell, PermeableSlab1D

D = 2e-9
WF = d.set_b(d.pgse(delta=3e-3, DELTA=8e-3, G_magnitude=0.1, bvecs=[[1, 0, 0]], n_t=40, slew_rate=np.inf), 5e8)


def _engines_used(geometry):
    """Which batch programs a run built on the geometry: 'traj*' keys are the replay producer,
    'standard' the fused scan."""
    return {k[0][0] for k in vars(geometry).get(core._BATCH_CACHE_ATTR, {})}


def test_parity_is_declared_on_the_validated_families_only():
    yes = (d.FreeDiffusion, d.Box1D, d.Sphere, d.Cylinder, d.Ellipsoid, d.PackedCylinders, d.PackedSpheres, d.Mesh)
    no = (d.MyelinatedCylinder, d.PackedMyelinatedCylinders, d.CurvedCylinder, d.PackedCurvedCylinders,
          PermeableSlab1D, PermeableShell)
    assert all(g.replay_parity for g in yes) and not any(g.replay_parity for g in no)
    assert not hasattr(core, "_REPLAY_AUTO_GEOM_NAMES") and not hasattr(core, "_replay_auto_allowed")


def test_auto_follows_the_declaration_not_the_name():
    sphere = d.Sphere(3e-6)
    d.simulate(200, D, WF, sphere, seed=0, require_gpu=False)
    assert _engines_used(sphere) >= {"traj"} and "standard" not in _engines_used(sphere)

    class Renamed(d.Sphere):                       # a subclass keeps the capability with the walk
        pass
    r = Renamed(3e-6)
    d.simulate(200, D, WF, r, seed=0, require_gpu=False)
    assert "traj" in _engines_used(r)

    class Unvalidated(d.Sphere):                   # ...and can renounce it
        replay_parity = False
    u = Unvalidated(3e-6)
    d.simulate(200, D, WF, u, seed=0, require_gpu=False)
    assert _engines_used(u) == {"standard"}
    # an explicit replay request is not gated by parity, only by what the replay can serve
    d.simulate(200, D, WF, Unvalidated(3e-6), seed=0, require_gpu=False, engine="replay")


def test_a_fused_only_request_falls_back_even_with_parity():
    s = d.Sphere(3e-6)
    d.simulate(200, D, WF, s, seed=0, require_gpu=False, return_compartments="final")
    assert _engines_used(s) == {"standard"}
