"""The magnetization-transfer walk refuses the permeable geometries it does not walk, by name, at every entry point.

The walk binds at a permeable wall after deciding the crossing (dmrai-lab/dmipy-sim#909; ``test_mt_permeable_walk``)
for a permeability constant in time on a geometry that carries no side. A permeability over time, a geometry that
carries the walker's side (packed cylinders) and a Mesh are refused; the forward vector-Bloch MT path refuses every
permeable wall; an impermeable MT walk is unchanged.
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


# geometry factory per permeability form the walk refuses: a schedule, a geometry that carries the walker's side
FORMS = {
    "cylinder-schedule": lambda: d.Cylinder(radius=2e-6, orientation=(0, 0, 1),
                                            permeability=PermeabilitySchedule([1e-3], [0.0, 1e-5])),
    "packed-scalar": lambda: _packed(1e-5),
    "packed-per-object": lambda: _packed([1e-5, 0.0]),
    "packed-schedule": lambda: _packed(PermeabilitySchedule([1e-3], [[1e-5, 0.0], [0.0, 1e-5]])),
    "slab-schedule": lambda: d.PermeableSlab1D(6e-6, PermeabilitySchedule([1e-3], [0.0, 1e-5])),
}


@pytest.mark.parametrize("form", sorted(FORMS))
def test_every_entry_point_refuses_a_permeable_mt_walk(form):
    g = FORMS[form]()
    named = pytest.raises(NotImplementedError, match=r"permeable .*#909")
    with named:
        _walk_geometry(16, D, g, **WALK, **MT)
    with named:
        simulate_mt_trajectories(16, D, g, **WALK, **MT)
    with named:
        walk_cost(16, D, g, **WALK, **MT)


def test_the_message_names_the_combination():
    g = FORMS["packed-scalar"]()
    with pytest.raises(NotImplementedError) as e:
        _walk_geometry(16, D, g, **WALK, **MT)
    msg = str(e.value)
    assert "PackedCylinders" in msg and "kappa_MT" in msg and "carries the walker's side" in msg and "#909" in msg


def test_the_forward_vector_bloch_oracle_refuses_too():
    from dmipy_replay import ScannerSequence
    from dmipy_sim.engine.bloch import _simulate_bloch
    wf = ScannerSequence(G=np.zeros((1, 11, 3)), dt=1e-4)
    with pytest.raises(NotImplementedError, match="#909"):
        _simulate_bloch(8, D, wf, d.Cylinder(radius=2e-6, orientation=(0, 0, 1), permeability=1e-5), seed=1, **MT)


def test_mesh_permeability_is_refused():
    trimesh = pytest.importorskip("trimesh")
    from dmipy_sim.geometry.mesh import Mesh
    UM = 1e-6
    ico = trimesh.creation.icosphere(subdivisions=2, radius=2.0)
    V, F = np.asarray(ico.vertices) * UM, np.asarray(ico.faces, np.int64)
    edge = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
    kw = dict(periodic=False, voxel_min=V.min(0) - 0.5 * UM, voxel_max=V.max(0) + 0.5 * UM,
              feature_radius=0.5 * float(np.median(edge)))
    for permeability in (1e-6, {"out": 1e-6, "in": 0.0}):
        try:
            g = Mesh(V, F, permeability=permeability, **kw)
        except (TypeError, ValueError, NotImplementedError):
            continue                                    # a spelling this Mesh does not take
        with pytest.raises(NotImplementedError, match="#909"):
            _walk_geometry(8, D, g, **WALK, **MT)


@pytest.mark.parametrize("make", [
    lambda: d.Cylinder(radius=2e-6, orientation=(0, 0, 1), permeability=1e-5),
    lambda: d.PermeableSlab1D(6e-6, 1e-5),
    lambda: d.PermeableSlab1D(7e-6, [2e-5, 2.5e-6], membranes=[2e-6, 5e-6]),
    lambda: d.Cylinder(radius=2e-6, orientation=(0, 0, 1)),
    lambda: _packed(None),
    lambda: d.PermeableSlab1D(6e-6, 0.0),
])
def test_an_mt_walk_that_is_impermeable_or_permeable_in_time_constant_walks(make):
    w = _walk_geometry(16, D, make(), **WALK, **MT)
    assert w.bound_events is not None
