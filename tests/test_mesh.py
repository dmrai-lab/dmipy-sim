"""Fast unit / smoke tests for the triangular-mesh geometry (dmipy_sim.mesh).

Heavy Monte-Carlo accuracy checks (mesh vs analytic to the MC noise floor) live in
test_mesh_mc.py, which is marked slow.  These tests are quick (small N, tiny meshes)
so they run on every PR.  Meshes are generated on the fly with trimesh — the big
research PLYs are a manual stress test, never committed to the suite.
"""
import warnings

import numpy as np
import numpy.testing as npt
import jax
import jax.numpy as jnp
import pytest

from dmipy_sim import simulate, Sphere, Mesh, load_ply, set_b
from dmipy_sim.waveforms import Waveform

trimesh = pytest.importorskip("trimesh")

D = 2e-9
R = 5e-6


def _short_wf(nb=3, n_t=80):
    T = 40e-3
    dt = T / (n_t - 1)
    G = np.zeros((1, n_t, 3), np.float32)
    G[0, 1:int(0.4 * n_t), 0] = 1.0
    G[0, -int(0.4 * n_t):-1, 0] = -1.0
    Gt = jnp.tile(jnp.array(G), (nb, 1, 1))
    return set_b(Waveform(G=Gt, dt=dt, echo_idx=n_t - 1), np.linspace(1, 1.5e9, nb))


def _icosphere(sub=3, r=R):
    m = trimesh.creation.icosphere(subdivisions=sub, radius=r)
    return np.asarray(m.vertices, float), np.asarray(m.faces, int)


def _open_tube(r=4e-6, L=12e-6, nt=32, nz=24):
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


def test_construction_and_attributes():
    V, F = _icosphere(3)
    g = Mesh(V, F)
    assert g.radius > 0
    assert g.C > 0 and g.overflow == 0
    assert g.n_ghost == 0                       # closed mesh -> no periodic ghosts
    assert g.surface_relaxivity_t2 is None and g.permeability is None


def test_classify_position_inside_outside():
    V, F = _icosphere(3)
    g = Mesh(V, F)
    pts = jnp.array([[0., 0., 0.],               # centre -> inside (0)
                     [0.9 * R, 0., 0.],           # inside
                     [2 * R, 0., 0.]])            # outside (1)
    lab = np.asarray(jax.vmap(g.classify_position)(pts))
    assert lab[0] == 0 and lab[1] == 0 and lab[2] == 1


def test_seed_containment_closed():
    V, F = _icosphere(3)
    g = Mesh(V, F)
    pts = np.asarray(g.init_positions(500, jax.random.PRNGKey(0), intra=True))
    assert (np.linalg.norm(pts, axis=1) < R).mean() > 0.99


def test_permeability_none_matches_default():
    V, F = _icosphere(2)
    wf = _short_wf()
    s_def = simulate(400, D, wf, Mesh(V, F), seed=1)
    s_none = simulate(400, D, wf, Mesh(V, F, permeability=None), seed=1)
    npt.assert_array_equal(np.asarray(s_def), np.asarray(s_none))


def test_load_ply_roundtrip(tmp_path):
    V, F = _icosphere(2)
    m = trimesh.Trimesh(vertices=V, faces=F, process=False)
    p = tmp_path / "ico.ply"
    m.export(p)
    V2, F2 = load_ply(str(p))
    assert V2.shape == V.shape and F2.shape == F.shape
    # scale argument converts units
    V3, _ = load_ply(str(p), scale=2.0)
    npt.assert_allclose(V3, V2 * 2.0, rtol=1e-6)
    # from_ply builds a usable geometry
    g = Mesh.from_ply(str(p), feature_radius=R)
    assert isinstance(g, Mesh) and g.C > 0


def test_periodic_ghosts_created():
    V, F = _open_tube()
    closed = Mesh(V, F)
    periodic = Mesh(V, F, periodic=(False, False, True),
                    voxel_min=[-6e-6, -6e-6, 0.0], voxel_max=[6e-6, 6e-6, 12e-6],
                    feature_radius=4e-6)
    assert closed.n_ghost == 0
    assert periodic.n_ghost > 0                  # z-seam triangles replicated


def test_permeability_coarseness_warning_and_report():
    # coarse mesh + permeability -> loud warning; quality_report flags it
    Vc, Fc = _icosphere(2)                        # edge/feature ~0.3 (coarse)
    with pytest.warns(UserWarning, match="coarse"):
        gc = Mesh(Vc, Fc, permeability=2e-5)
    rep = gc.quality_report(verbose=False)
    assert rep["permeability_noise_floor"] is False
    assert rep["diffusion_noise_floor"] is True and rep["relaxivity_noise_floor"] is True
    assert rep["edge_feature_ratio"] > 0.05
    # fine mesh -> no permeability warning
    Vf, Ff = _icosphere(5)                        # edge/feature ~0.04 (fine)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gf = Mesh(Vf, Ff, permeability=2e-5)
    assert gf.quality_report(verbose=False)["permeability_noise_floor"] is True
    # coarse mesh WITHOUT permeability -> no warning (diffusion/relaxivity fine)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Mesh(Vc, Fc)


def test_orientation_is_acquisition_rotation():
    """Placement in the bore is applied as an acquisition rotation, so the walk
    stays in the mesh's native frame (unchanged containment) and `orientation`
    only records the mesh->lab rotation used to rotate the gradient in simulate."""
    V, F = _open_tube(r=4e-6, L=12e-6)
    kw = dict(periodic=(False, False, True),
              voxel_min=[-6e-6, -6e-6, 0.0], voxel_max=[6e-6, 6e-6, 12e-6],
              feature_radius=4e-6)
    g_plain = Mesh(V, F, **kw)
    g_orient = Mesh(V, F, orientation=[1.0, 0.0, 0.0], **kw)
    assert g_plain._orient_R is None
    assert g_orient._orient_R is not None                 # mesh->lab rotation stored
    # walk is native-frame in both -> radial confinement about the mesh-z axis
    _, pos = simulate(400, D, _short_wf(1, 120), g_orient, seed=1, return_positions=True)
    pos = np.asarray(pos)
    assert (np.linalg.norm(pos[:, :2], axis=1) < 4e-6 * 1.03).mean() > 0.99


def test_viz_functions_render(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from dmipy_sim import (plot_mesh_section, plot_walkers_3d, plot_cell_surface,
                           walk_paths, plot_trajectories)
    import matplotlib.pyplot as plt
    V, F = _icosphere(3)
    g = Mesh(V, F)
    w = np.asarray(g.init_positions(300, jax.random.PRNGKey(0), intra=True))
    p = tmp_path / "sec.png"; plot_mesh_section(g, walkers=w, save=str(p)); assert p.exists()
    p = tmp_path / "w3d.png"; plot_walkers_3d(g, w, sub_box=5e-6, save=str(p)); assert p.exists()
    p = tmp_path / "cell.png"; plot_cell_surface(g, save=str(p)); assert p.exists()
    paths = walk_paths(g, 20, 25, diffusivity=D, dt=2e-4, seed=0)
    assert paths.shape == (20, 26, 3)
    p = tmp_path / "traj.png"; plot_trajectories(g, paths, save=str(p)); assert p.exists()
    plt.close("all")


def test_periodic_tube_confinement_and_zwrap():
    """Smoke: walkers stay radially inside an open periodic tube and wrap in z."""
    V, F = _open_tube(r=4e-6, L=12e-6)
    g = Mesh(V, F, periodic=(False, False, True),
             voxel_min=[-6e-6, -6e-6, 0.0], voxel_max=[6e-6, 6e-6, 12e-6],
             feature_radius=4e-6)
    _, pos = simulate(600, D, _short_wf(1, 120), g, seed=1, return_positions=True)
    pos = np.asarray(pos)
    rad = np.linalg.norm(pos[:, :2], axis=1)
    assert (rad < 4e-6 * 1.03).mean() > 0.99      # radial confinement (no wall leak)
    # continuous z is unbounded (periodic wrap keeps geometry query in-box but the
    # returned position accumulates), so some walkers move beyond [0, L]
    assert pos[:, 2].max() > 12e-6 or pos[:, 2].min() < 0.0
