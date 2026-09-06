"""One compartment-id convention: 0 the extra / free pool, enclosed pools positive.

`classify_position` is the single source of a walker's pool, and every channel that reports one
-- `return_compartments`, `simulate_trajectories`' `comp_traj`, the myelin kernels' internal codes
via `pool_of` -- agrees with it. The slow test at the end is what the convention is for: a
permeable mesh with per-compartment T2 replays to the fused engine's answer, because the
occupancy channel and the T2 array are indexed by the same id.
"""
import numpy as np
import pytest

import dmipy_sim as d

D = 2e-9
EXTRA, INTRA, MYELIN = 0, 1, 2


def _geometries():
    from dmipy_sim.geometry import mesh_shapes
    c, L, _ = d.pack_cylinders([1e-6] * 4, target_vf=0.3, seed=0)
    cs, Ls, _ = d.pack_spheres([1e-6] * 4, target_vf=0.1, seed=0)
    V, F = mesh_shapes.icosphere(3e-6, subdivisions=2)
    cl = np.stack([np.zeros(8), np.zeros(8), np.linspace(0, 2e-5, 8)], axis=1)
    return {
        "Box1D": d.Box1D(4e-6),
        "Sphere": d.Sphere(3e-6),
        "Cylinder": d.Cylinder(3e-6, (0, 0, 1)),
        "Ellipsoid": d.Ellipsoid((3e-6, 4e-6, 5e-6)),
        "PermeableSlab1D": d.geometry.PermeableSlab1D(4e-6, permeability=1e-5),
        "PermeableShell": d.geometry.PermeableShell(3e-6, 5e-6, permeability=1e-5),
        "PackedCylinders": d.PackedCylinders([1e-6] * 4, c, L),
        "PackedSpheres": d.PackedSpheres([1e-6] * 4, cs, Ls),
        "CurvedTube": d.CurvedTube(cl, radius=2e-6),
        "MultiShellCurvedTube": d.MultiShellCurvedTube(cl, r_in=2e-6, r_out=3e-6),
        "Mesh": d.Mesh(V, F, feature_radius=1e-6),
    }


@pytest.mark.parametrize("name", ["Sphere", "Cylinder", "Ellipsoid", "PermeableShell", "CurvedTube",
                                  "Mesh"])
def test_inside_is_one_and_outside_is_zero(name):
    import jax
    g = _geometries()[name]
    r0 = g.init_positions(256, jax.random.PRNGKey(0))         # seeded inside
    # the exact labels; a mesh's per-step classifier is a near-field test that is undecidable
    # away from its walls (#33), which is why the engine carries the label instead
    lab = np.asarray(g.classify_positions_exact(r0))
    assert (lab == INTRA).all(), f"{name}: seeds inside must carry pool id 1"
    far = np.asarray(r0) + np.array([1e-3, 0.0, 0.0], np.float32)
    assert (np.asarray(jax.vmap(g.classify_position)(far)) == EXTRA).all(), \
        f"{name}: a point far outside must carry pool id 0"


def test_the_remaining_geometries_follow_the_same_convention():
    import jax
    import jax.numpy as jnp
    G = _geometries()
    z = jnp.zeros(3, jnp.float32)
    assert int(G["Box1D"].classify_position(jnp.asarray([2e-6, 0, 0], jnp.float32))) == INTRA
    assert int(G["PermeableSlab1D"].classify_position(jnp.asarray([1e-6, 0, 0], jnp.float32))) == INTRA
    assert int(G["PermeableSlab1D"].classify_position(jnp.asarray([3e-6, 0, 0], jnp.float32))) == EXTRA
    ms = G["MultiShellCurvedTube"]
    assert int(ms.classify_position(z)) == INTRA
    assert int(ms.classify_position(jnp.asarray([2.5e-6, 0, 0], jnp.float32))) == MYELIN
    assert int(ms.classify_position(jnp.asarray([5e-6, 0, 0], jnp.float32))) == EXTRA
    for name in ("PackedCylinders", "PackedSpheres"):
        g = G[name]
        r0 = g.init_positions(64, jax.random.PRNGKey(0))       # seeded in the extra space
        assert (np.asarray(jax.vmap(g.classify_position)(r0)) == EXTRA).all()
        assert int(g.classify_position(jnp.asarray(np.r_[g._centers_jax[0], 0.0][:3], jnp.float32))) == 1
    assert int(d.FreeDiffusion().classify_position(z)) == EXTRA


def test_myelin_kernels_report_pool_ids():
    import jax
    import jax.numpy as jnp
    mc = d.MyelinatedCylinder(3e-6, 5e-6, (0, 0, 1), 2e-9, 2e-9)
    assert [int(mc.classify_position(jnp.asarray([x, 0, 0], jnp.float32))) for x in (0.0, 4e-6, 6e-6)] \
        == [INTRA, MYELIN, EXTRA]
    assert [int(v) for v in mc.pool_of(jnp.arange(3))] == [EXTRA, INTRA, MYELIN]   # carries pool ids
    L = float(np.sqrt(np.pi * 3 * (1e-6 / 0.7) ** 2 / 0.5))
    _, _, c = d.pack_myelinated_cylinders([1e-6] * 3, 0.7, None, cell_size=L, seed=0)
    pm = d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4)
    enc = jnp.asarray([0, 1, 3, 5, 8])
    assert [int(v) for v in pm.pool_of(enc)] == [EXTRA, INTRA, INTRA, MYELIN, MYELIN]
    r0 = pm.init_positions(200, jax.random.PRNGKey(0))
    assert (np.asarray(pm.pool_of(pm._init_compartments))
            == np.asarray(pm.pool_of(jax.vmap(pm.classify_position)(r0)))).mean() > 0.99


def test_simulate_and_simulate_trajectories_report_the_same_ids():
    """The same PackedMyelinatedCylinders labels its walkers the same way on both entry points."""
    L = float(np.sqrt(np.pi * 3 * (1e-6 / 0.7) ** 2 / 0.5))
    _, _, c = d.pack_myelinated_cylinders([1e-6] * 3, 0.7, None, cell_size=L, seed=0)
    pm = d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4)
    wf = d.set_b(d.pgse(delta=2e-3, DELTA=4e-3, G_magnitude=0.05, bvecs=[[1, 0, 0]], n_t=30,
                        slew_rate=np.inf), 5e8)
    _, origin, final = d.simulate(300, None, wf, pm, seed=0, return_compartments="final",
                                  require_gpu=False)
    out = d.simulate_trajectories(300, 2e-9, pm, T_max=6e-3, dt_save=2e-3, seed=0, require_gpu=False)
    comp0 = np.asarray(out.compartment)[:, 0]
    assert set(np.unique(origin)) <= {EXTRA, INTRA, MYELIN}
    assert (np.asarray(origin) == comp0).all()
    assert (comp0 == MYELIN).mean() > 0.05 and (comp0 == INTRA).mean() > 0.05


@pytest.mark.parametrize("name", ["Sphere", "Cylinder", "Ellipsoid", "PermeableSlab1D",
                                  "PermeableShell", "PackedCylinders", "Mesh"])
def test_comp_traj_is_the_geometrys_own_label(name):
    """`simulate_trajectories`' compartment channel at t=0 is the classifier collapsed to two pools."""
    import jax
    g = _geometries()[name]
    r0 = g.init_positions(128, jax.random.PRNGKey(3))
    out = d.simulate_trajectories(128, D, g, T_max=2e-4, dt_save=1e-4, seed=3, r0=r0, require_gpu=False)
    comp = np.asarray(out.compartment)[:, 0]                          # over the first save interval
    want = np.minimum(np.asarray(g.classify_positions_exact(r0)), 1)
    if g.permeability is not None:
        # a permeable wall may be crossed within the interval
        assert np.abs(comp - want).max() <= 1.0 and (np.round(comp) == want).mean() > 0.95
    elif name == "Mesh":
        # the carried label is re-derived only within reach of a wall, where the nearest-triangle
        # test is approximate to about a triangle (#33)
        assert (comp == want).mean() > 0.95
    else:
        assert (comp == want).all()


@pytest.mark.slow
def test_permeable_mesh_with_per_compartment_T2_replays_the_fused_engine():
    """Per-compartment T2 on a permeable mesh: the occupancy channel and the T2 array share the id."""
    from dmipy_sim.geometry import mesh_shapes
    V, F = mesh_shapes.icosphere(3e-6, subdivisions=3)
    m = d.Mesh(V, F, feature_radius=0.5e-6, permeability=2e-5, intra={"T2": 0.03}, extra={"T2": 0.3})
    wf = d.set_b(d.pgse(delta=3e-3, DELTA=9e-3, G_magnitude=0.1, bvecs=[[1, 0, 0]], n_t=120,
                        slew_rate=np.inf), 5e8)
    N = 6_000
    s_f = np.asarray(d.simulate(N, D, wf, m, seed=2, engine="fused", require_gpu=False)).ravel()
    s_r = np.asarray(d.simulate(N, D, wf, m, seed=2, engine="replay", require_gpu=False)).ravel()
    assert abs(s_f[0] - s_r[0]) < max(0.02, 3.0 / np.sqrt(N)), f"fused {s_f[0]:.4f} vs replay {s_r[0]:.4f}"
    # and the intra T2 actually acted: a walk that started inside decays toward exp(-TE/T2_intra)
    assert s_f[0] < 0.9


@pytest.mark.parametrize("sub_steps", [43, 97, 200])
def test_the_discrete_label_is_rounded_not_truncated(sub_steps):
    """The occupancy is comp_sum / sub_steps; XLA may compute it as comp_sum * (1 / sub_steps), which for
    some counts is 0.99999994. Truncating that to int8 labelled every intra walker "extra" at 97 sub-steps."""
    out = d.simulate_trajectories(32, D, d.Cylinder(3e-6, (0, 0, 1)), T_max=2e-3, dt_save=5e-4, seed=0,
                                  require_gpu=False, sub_steps=sub_steps)
    assert (np.asarray(out.compartment) == 1).all()
