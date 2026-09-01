"""The plain mesh walk was sub-stepped by a meshing parameter, not a pore.

The auto-tune set ``step_l = R/6`` where ``_geometry_radius`` returns ``feature_radius`` for a Mesh -- so
refining a mesh tightened the step while the pore stayed the size it always was. What actually bounds a mesh
step is the 27-cell collision lookup: outrun it and the wall is missed outright. Same substitution
``mt_sub_steps`` makes for the binding walk (dmipy-sim#56), for the same reason.

Measured on the 366-fibre CACTUS bundle: the old rule asked 1256, the collision criterion 97, and across
97 -> 314 apparent D_perp scatters +/-1.3% with no trend, boundary local time moves +0.16%, containment is
flat. The convergence itself is the slow test at the bottom; these are the cheap rule assertions.
"""
from __future__ import annotations

import numpy as np
import pytest

from dmipy_sim.geometries import Sphere
from dmipy_sim.physics import collision_sub_steps, walk_sub_steps, _geometry_radius

trimesh = pytest.importorskip("trimesh")

UM = 1e-6
D = 2e-9


def _mesh_sphere(radius_um=2.0, subdivisions=3, **kw):
    from dmipy_sim.mesh import Mesh
    ico = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius_um)
    V = np.asarray(ico.vertices, float) * UM
    F = np.asarray(ico.faces, np.int64)
    e = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
    return Mesh(V, F, periodic=False, voxel_min=V.min(0) - 0.5 * UM, voxel_max=V.max(0) + 0.5 * UM,
                feature_radius=0.5 * float(np.median(e)), **kw)


def _old_rule(geometry, dt, divisor=216.0):
    R = _geometry_radius(geometry)
    return max(1, int(np.ceil(dt / (float(R) ** 2 / (divisor * D)))))


def test_a_mesh_uses_the_collision_criterion():
    mesh = _mesh_sphere()
    dt = 2e-5
    assert walk_sub_steps(mesh, D, dt) == collision_sub_steps(mesh, D, dt)
    # and that is far below what keying off feature_radius asked for
    assert walk_sub_steps(mesh, D, dt) < 0.25 * _old_rule(mesh, dt), (
        f"{walk_sub_steps(mesh, D, dt)} vs old {_old_rule(mesh, dt)}")


def test_refining_a_mesh_no_longer_multiplies_the_cost_quadratically():
    """The defect, stated as a ratio: the old rule scaled with the TRIANGLES, not the pore.

    Both spheres below are the same 2um pore. Under the old rule the finer mesh costs ~4x more sub-steps for
    the same physics; under the collision criterion the cost tracks the cell size, which is the thing a step
    actually has to respect.
    """
    dt = 2e-5
    coarse, fine = _mesh_sphere(subdivisions=2), _mesh_sphere(subdivisions=3)
    old_ratio = _old_rule(fine, dt) / _old_rule(coarse, dt)
    assert old_ratio > 3.0, f"test geometry no longer shows the old rule's scaling (ratio {old_ratio:.2f})"
    # the new rule still tightens with resolution (the lookup really does shrink) but from a far lower base
    assert walk_sub_steps(fine, D, dt) < _old_rule(fine, dt) / 3.0


def test_an_analytic_pore_is_unchanged():
    """R/6 is a real criterion for an analytic sphere -- only the mesh path was keyed to the wrong length."""
    sph = Sphere(radius=2e-6)
    dt = 2e-5
    assert walk_sub_steps(sph, D, dt) == _old_rule(sph, dt)


def test_a_permeable_mesh_keeps_the_fine_rule():
    """Deliberately out of scope: crossing probability is step-size sensitive in a way the collision
    criterion says nothing about, and that regime is not measured here."""
    mesh = _mesh_sphere(permeability=1e-5)
    dt = 2e-5
    assert walk_sub_steps(mesh, D, dt) == _old_rule(mesh, dt, divisor=3750.0)
    assert walk_sub_steps(mesh, D, dt) > collision_sub_steps(mesh, D, dt)


@pytest.mark.slow
def test_the_observables_are_converged_at_the_collision_criterion():
    """Refining 4x beyond the collision criterion must not move the walk's observables.

    `simulate_trajectories` takes no sub_steps argument, so the step is varied through the attribute the rule
    actually reads. For a MESH that is now `cell_size`, not `radius` -- this change is what moved it. Setting
    `radius` here would silently leave every run at the same step and compare a configuration against
    itself, which is how this test first passed while asserting nothing.
    """
    from dmipy_sim.core import simulate_trajectories
    mesh = _mesh_sphere()
    dt_save, T_max, n = 2e-5, 5e-3, 3000
    n_coll = collision_sub_steps(mesh, D, dt_save)

    out = {}
    steps = {}
    for mult in (1, 4):
        target = n_coll * mult
        # n = ceil((L/(0.9*cell))^2), L = sqrt(6 D dt) -> invert for a target n
        L = float(np.sqrt(6.0 * D * dt_save))
        mesh.cell_size = float(L / (0.9 * np.sqrt(target)))
        o = simulate_trajectories(n, D, mesh, T_max=T_max, dt_save=dt_save, seed=3,
                                  save_relaxation_data=True, require_gpu=False)
        steps[mult] = int(o[2])
        tr = np.asarray(o[0], np.float64)
        disp = tr[:, -1, :] - tr[:, 0, :]
        out[mult] = (float((disp ** 2).sum(axis=1).mean() / (6.0 * T_max)),
                     float(-np.asarray(o[4], np.float64).sum(axis=1).mean()))

    # guard the lever itself: if the two runs used the same sub-step count this test proves nothing
    assert steps[4] >= 3 * steps[1], (
        f"refinement did not take effect (sub_steps {steps[1]} -> {steps[4]}); the step is driven by "
        f"cell_size for a mesh, so a lever on the wrong attribute makes this test vacuous")
    d_rel = abs(out[4][0] - out[1][0]) / out[1][0]
    lt_rel = abs(out[4][1] - out[1][1]) / out[1][1]
    assert d_rel < 0.05, f"apparent D moved {100*d_rel:.2f}% refining 4x past the collision criterion"
    assert lt_rel < 0.05, f"boundary local time moved {100*lt_rel:.2f}% refining 4x"
