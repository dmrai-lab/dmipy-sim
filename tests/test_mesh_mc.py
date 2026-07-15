"""Monte-Carlo accuracy of the mesh geometry vs the analytic geometries.

A triangulated shape must reproduce the analytic signal of the same shape to the
MC noise floor.  We isolate the faceting bias by running the mesh and the analytic
geometry through the IDENTICAL waveform / seed / walker count (same MC noise
realisation), so the residual is purely the surface discretisation.

Resolution note: restricted diffusion and surface relaxivity reach the noise floor
at moderate resolution; permeability needs a finer mesh (its bias falls ~O(h^2)),
so the permeability test uses a higher subdivision.

Marked slow (see conftest) — heavy CPU Monte-Carlo; runs in the weekly job.
"""
import numpy as np
import numpy.testing as npt
import jax
import jax.numpy as jnp
import pytest

from dmipy_sim import simulate, Sphere, Cylinder, Mesh, set_b
from dmipy_sim.waveforms import Waveform

trimesh = pytest.importorskip("trimesh")

D = 2e-9
R = 5e-6
SEED = 123


def _pgse(nb, n_t, TE=40e-3, axis=0):
    dt = TE / (n_t - 1)
    G = np.zeros((1, n_t, 3), np.float32)
    G[0, 1:int(0.4 * n_t), axis] = 1.0
    G[0, -int(0.4 * n_t):-1, axis] = -1.0
    Gt = jnp.tile(jnp.array(G), (nb, 1, 1))
    return set_b(Waveform(G=Gt, dt=dt, echo_idx=n_t - 1), np.linspace(1, 2e9, nb))


def _ico(sub, r=R):
    m = trimesh.creation.icosphere(subdivisions=sub, radius=r)
    return np.asarray(m.vertices, float), np.asarray(m.faces, int)


def _open_tube(r, L, nt=48, nz=40):
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    zs = np.linspace(0, L, nz)
    V = np.array([[r * np.cos(t), r * np.sin(t), z] for z in zs for t in th])
    F = []
    for iz in range(nz - 1):
        for j in range(nt):
            a = iz * nt + j; b = iz * nt + (j + 1) % nt
            c = (iz + 1) * nt + j; d = (iz + 1) * nt + (j + 1) % nt
            F.append([a, b, d]); F.append([a, d, c])
    return V, np.array(F)


def test_diffusion_matches_analytic_sphere():
    V, F = _ico(4)
    wf = _pgse(8, 400)
    s_mesh = np.asarray(simulate(3000, D, wf, Mesh(V, F), seed=SEED))
    s_ana = np.asarray(simulate(3000, D, wf, Sphere(radius=R), seed=SEED))
    npt.assert_allclose(s_mesh, s_ana, atol=0.02,
                        err_msg="mesh sphere diffusion vs analytic Sphere")


def test_surface_relaxivity_matches_analytic_sphere():
    V, F = _ico(4)
    wf = _pgse(8, 400)
    rho = 5e-6
    s_mesh = np.asarray(simulate(3000, D, wf, Mesh(V, F, surface_relaxivity_t2=rho), seed=SEED))
    s_ana = np.asarray(simulate(3000, D, wf, Sphere(radius=R, surface_relaxivity_t2=rho), seed=SEED))
    npt.assert_allclose(s_mesh, s_ana, atol=0.02,
                        err_msg="mesh surface relaxivity vs analytic Sphere")


def test_permeability_matches_analytic_sphere():
    # permeability faceting bias ~O(h^2) -> needs a fine mesh (subdiv 5)
    V, F = _ico(5)
    wf = _pgse(6, 250, TE=15e-3)
    kappa = 2e-5
    s_mesh = np.asarray(simulate(1500, D, wf, Mesh(V, F, permeability=kappa), seed=SEED))
    s_ana = np.asarray(simulate(1500, D, wf, Sphere(radius=R, permeability=kappa), seed=SEED))
    npt.assert_allclose(s_mesh, s_ana, atol=0.025,
                        err_msg="mesh permeability vs analytic Sphere (needs fine mesh)")


def test_periodic_tube_matches_infinite_cylinder():
    r, L = 4e-6, 12e-6
    V, F = _open_tube(r, L)
    g = Mesh(V, F, periodic=(False, False, True),
             voxel_min=[-r - 2e-6, -r - 2e-6, 0.0], voxel_max=[r + 2e-6, r + 2e-6, L],
             feature_radius=r)
    wf = _pgse(8, 400)
    s_mesh = np.asarray(simulate(3000, D, wf, g, seed=SEED))
    s_cyl = np.asarray(simulate(3000, D, wf,
                                Cylinder(radius=r, orientation=np.array([0., 0., 1.])), seed=SEED))
    npt.assert_allclose(s_mesh, s_cyl, atol=0.02,
                        err_msg="periodic tube mesh vs analytic infinite Cylinder")


def test_positions_full_selects_permeated_walkers():
    """return_positions='full' + return_compartments='full' on a permeable mesh
    yields per-step positions and compartment ids, so walkers that permeated can
    be selected and their trajectories visualised."""
    V, F = _ico(3)
    g = Mesh(V, F, permeability=2e-5)
    wf = _pgse(2, 120, TE=15e-3)
    _, pos, origin, comp = simulate(800, D, wf, g, seed=SEED,
                                    return_positions='full', return_compartments='full')
    assert pos.shape == (800, 120, 3)
    assert comp.shape == (800, 120)
    permeated = (comp != comp[:, :1]).any(axis=1)
    assert int(permeated.sum()) > 0                       # some walkers crossed the membrane


def test_orientation_signal_parity():
    """Orienting the substrate in the bore must be equivalent to rotating the
    acquisition: a tube whose axis is placed along lab-x, probed with a gradient
    perpendicular to lab-x, matches the same tube along mesh-z probed
    perpendicular to mesh-z (both are the identical perpendicular measurement)."""
    r, L = 4e-6, 12e-6
    V, F = _open_tube(r, L)
    vmin = [-r - 2e-6, -r - 2e-6, 0.0]; vmax = [r + 2e-6, r + 2e-6, L]

    g_ref = Mesh(V, F, periodic=(False, False, True),
                 voxel_min=vmin, voxel_max=vmax, feature_radius=r)
    s_ref = np.asarray(simulate(3000, D, _pgse(8, 400, axis=0), g_ref, seed=SEED))

    # same tube, axis oriented along lab-x; probe perpendicular = lab-z
    g_rot = Mesh(V, F, periodic=(False, False, True),
                 voxel_min=vmin, voxel_max=vmax, feature_radius=r,
                 orientation=[1.0, 0.0, 0.0])
    s_rot = np.asarray(simulate(3000, D, _pgse(8, 400, axis=2), g_rot, seed=SEED))

    npt.assert_allclose(s_rot, s_ref, atol=0.02,
                        err_msg="oriented substrate != rotated acquisition")
