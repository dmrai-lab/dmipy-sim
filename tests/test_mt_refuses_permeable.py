"""The magnetization-transfer walk refuses, by name, the permeable walls it does not walk, and walks the rest.

The walk binds at a permeable wall after deciding the crossing (dmrai-lab/dmipy-sim#909;
``test_mt_permeable_walk`` and ``test_mt_permeable_packed`` hold it to the continuum). Every geometry with walls is
walked at a permeability of one number, one per object or a schedule over time, at every entry point; a Mesh with a
permeable wall is refused; the forward vector-Bloch MT path takes a permeability constant in time and refuses a schedule.
An impermeable MT walk is unchanged.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.geometry.base import PermeabilitySchedule
from dmipy_sim.engine.core import _walk_geometry, walk_cost
from dmipy_sim.engine.mt_walk import simulate_mt_trajectories

D = 2e-9
MT = dict(kappa_MT=2e-6, dwell_time=5e-3)
WALK = dict(T_max=2e-3, K=8, seed=3)


def _packed(permeability):
    radii = np.array([2e-6, 2e-6])
    centers = np.array([[3e-6, 3e-6], [9e-6, 9e-6]])
    return d.PackedCylinders(radii, centers, 12e-6, permeability=permeability)


SCHEDULES = {
    "cylinder-schedule": lambda: d.Cylinder(radius=2e-6, orientation=(0, 0, 1),
                                            permeability=PermeabilitySchedule([1e-3], [0.0, 1e-5])),
    "packed-schedule": lambda: _packed(PermeabilitySchedule([1e-3], [[1e-5, 0.0], [0.0, 1e-5]])),
    "slab-schedule": lambda: d.PermeableSlab1D(6e-6, PermeabilitySchedule([1e-3], [0.0, 1e-5])),
}
# every permeability form the walk takes: a number, a number per object, a schedule
FORMS = dict(SCHEDULES, **{
    "cylinder-scalar": lambda: d.Cylinder(radius=2e-6, orientation=(0, 0, 1), permeability=1e-5),
    "packed-scalar": lambda: _packed(1e-5),
    "packed-per-object": lambda: _packed([1e-5, 0.0]),
    "slab-scalar": lambda: d.PermeableSlab1D(6e-6, 1e-5),
    "slab-per-object": lambda: d.PermeableSlab1D(7e-6, [2e-5, 2.5e-6], membranes=[2e-6, 5e-6]),
})


@pytest.mark.parametrize("form", sorted(FORMS))
def test_every_entry_point_walks_a_permeable_mt_geometry(form):
    g = FORMS[form]()
    w = _walk_geometry(16, D, g, **WALK, **MT)
    assert w.bound_events is not None
    simulate_mt_trajectories(16, D, g, **WALK, **MT)
    walk_cost(16, D, g, **WALK, **MT)


@pytest.mark.parametrize("form", sorted(SCHEDULES))
def test_the_forward_vector_bloch_oracle_refuses_a_schedule(form):
    from dmipy_replay import ScannerSequence
    from dmipy_sim.engine.bloch import _simulate_bloch
    wf = ScannerSequence(G=np.zeros((1, 11, 3)), dt=1e-4)
    with pytest.raises(NotImplementedError, match=r"varies in time.*#909"):
        _simulate_bloch(8, D, wf, SCHEDULES[form](), seed=1, **MT)


def test_mesh_permeability_is_refused():
    trimesh = pytest.importorskip("trimesh")
    from dmipy_sim.geometry.mesh import Mesh
    UM = 1e-6
    ico = trimesh.creation.icosphere(subdivisions=2, radius=2.0)
    V, F = np.asarray(ico.vertices) * UM, np.asarray(ico.faces, np.int64)
    edge = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
    kw = dict(periodic=False, voxel_min=V.min(0) - 0.5 * UM, voxel_max=V.max(0) + 0.5 * UM,
              feature_radius=0.5 * float(np.median(edge)))
    refused = 0
    for permeability in (1e-6, {"out": 1e-6, "in": 0.0}):
        try:
            g = Mesh(V, F, permeability=permeability, **kw)
        except (TypeError, ValueError, NotImplementedError):
            continue                                    # a spelling this Mesh does not take
        with pytest.raises(NotImplementedError, match=r"Mesh.*#909"):
            _walk_geometry(8, D, g, **WALK, **MT)
        refused += 1
    assert refused


@pytest.mark.parametrize("make", [
    lambda: d.Cylinder(radius=2e-6, orientation=(0, 0, 1)),
    lambda: _packed(None),
    lambda: d.PermeableSlab1D(6e-6, 0.0),
])
def test_an_impermeable_mt_walk_still_walks(make):
    w = _walk_geometry(16, D, make(), **WALK, **MT)
    assert w.bound_events is not None
