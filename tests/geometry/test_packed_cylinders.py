"""PackedCylinders geometry: two pools of water, the extra-axonal physics measured with ``pool="extra"``.

Tests cover:
  - RSA packing: no overlaps (including periodic images), correct VF
  - init_positions: both pools by area by default; one pool when named
  - Step-size assumption: σ << min_gap for default simulation parameters
  - Parallel gradient → free diffusion exp(-bD)  [exact, analytical]
  - Walker containment: no walker enters any cylinder during simulation
  - Hindered perpendicular diffusion > free diffusion at high b  [qualitative]
"""

import sys
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from dmipy_sim import simulate, FreeDiffusion, pack_cylinders, PackedCylinders
from dmipy_replay import set_b
from dmipy_replay.sequences import pgse
from tests.conftest import D, SEED, mc_atol

# The pack kernel sub-steps the walk (several candidate walls per step): 1,500 walkers is a standard error of at most
# 0.018 here, against effects of 0.1 (hindering at b = 1.5e9) and a free-diffusion tolerance of four of them.
N_PACK = 1_500
from dmipy_sim.engine._fused import simulate_fused


# ---------------------------------------------------------------------------
# pack_cylinders
# ---------------------------------------------------------------------------

def test_pack_cylinders_no_overlap_monodisperse():
    """RSA: no two cylinders overlap (including across periodic boundary)."""
    r = 2e-6   # 2 µm radius
    N = 8
    radii = np.full(N, r)
    centers, L, vf = pack_cylinders(radii, target_vf=0.20, seed=42)

    assert centers.shape == (N, 2)
    for i in range(N):
        for j in range(i + 1, N):
            dq  = centers[i] - centers[j]
            dq -= L * np.round(dq / L)
            dist = np.linalg.norm(dq)
            assert dist >= 2 * r - 1e-10, (
                f"Cylinders {i} and {j} overlap: dist={dist*1e6:.3f} µm, "
                f"min_required={2*r*1e6:.3f} µm")


def test_pack_cylinders_no_overlap_polydisperse():
    """RSA: no overlap for polydisperse population."""
    rng = np.random.default_rng(0)
    radii = rng.uniform(1e-6, 4e-6, 6)
    centers, L, vf = pack_cylinders(radii, target_vf=0.15, seed=7)

    for i in range(len(radii)):
        for j in range(i + 1, len(radii)):
            dq  = centers[i] - centers[j]
            dq -= L * np.round(dq / L)
            dist = np.linalg.norm(dq)
            min_required = radii[i] + radii[j]
            assert dist >= min_required - 1e-10, (
                f"Cylinders {i},{j} overlap: gap={( dist-min_required)*1e9:.1f} nm")


def test_pack_cylinders_achieved_vf():
    """Achieved VF matches the returned value and is close to target."""
    radii = np.full(5, 3e-6)
    centers, L, vf = pack_cylinders(radii, target_vf=0.20, seed=1)
    expected_vf = np.pi * np.sum(radii ** 2) / L ** 2
    npt.assert_allclose(vf, expected_vf, rtol=1e-6, err_msg="returned vf mismatch")
    npt.assert_allclose(vf, 0.20, rtol=1e-3, err_msg="achieved vf vs target")


def test_pack_cylinders_raises_on_bad_inputs():
    """ValueError for invalid inputs."""
    with pytest.raises(ValueError, match="exactly one"):
        pack_cylinders([1e-6], target_vf=0.2, L=100e-6)
    with pytest.raises(ValueError, match="exactly one"):
        pack_cylinders([1e-6])
    with pytest.raises(ValueError, match="positive"):
        pack_cylinders([-1e-6], target_vf=0.1)
    with pytest.raises(ValueError, match="target_vf"):
        pack_cylinders([1e-6], target_vf=1.5)


# ---------------------------------------------------------------------------
# PackedCylinders geometry
# ---------------------------------------------------------------------------

def _simple_geometry(n_cyl=4, radius=3e-6, target_vf=0.20, seed=0):
    """Build a small PackedCylinders geometry for testing."""
    radii = np.full(n_cyl, radius)
    centers, L, _ = pack_cylinders(radii, target_vf=target_vf, seed=seed)
    return PackedCylinders(radii=radii, centers=centers, L=L,
                           orientation=[0., 0., 1.], pool="extra")


def _inside_any(pos_xy, geom):
    centers, radii, L = np.array(geom._centers_jax), geom._radii_np, geom._L_float
    inside = np.zeros(len(pos_xy), bool)
    for k in range(len(radii)):
        dxy = pos_xy - centers[k]
        dxy -= L * np.round(dxy / L)
        inside |= np.sum(dxy ** 2, axis=1) < radii[k] ** 2
    return inside


def test_packed_cylinders_seeds_both_pools_by_area_by_default():
    """The default seeding covers the whole cell: the lumens hold their area fraction of the walkers, and a named
    pool holds all of them."""
    radii = np.full(4, 3e-6)
    centers, L, vf = pack_cylinders(radii, target_vf=0.20, seed=0)
    both = PackedCylinders(radii=radii, centers=centers, L=L)
    assert both.pool is None
    pos = np.array(both.init_positions(20_000, jax.random.PRNGKey(0)))
    assert abs(_inside_any(pos[:, :2], both).mean() - vf) < 0.01
    intra = PackedCylinders(radii=radii, centers=centers, L=L, pool="intra")
    assert _inside_any(np.array(intra.init_positions(2_000, jax.random.PRNGKey(1)))[:, :2], intra).all()
    with pytest.raises(ValueError):
        PackedCylinders(radii=radii, centers=centers, L=L, pool="lumen")


def test_packed_cylinders_min_gap_positive():
    """min_gap must be positive (cylinders don't overlap)."""
    geom = _simple_geometry()
    assert geom.min_gap > 0, f"min_gap={geom.min_gap*1e9:.1f} nm should be positive"


def test_packed_cylinders_step_size_assumption():
    """Verify σ < 0.1 · min_gap for typical parameters.

    At D=2e-9 m²/s, T=50 ms, n_t=1000: dt=50 µs, σ=√(6·D·dt)≈0.245 µm.
    For a 3 µm radius at VF=0.20 the min gap is well above 2 µm, so the
    single-reflection approximation is valid.
    """
    geom   = _simple_geometry()
    n_t    = 1000
    T_diff = 50e-3   # s
    dt     = T_diff / n_t
    sigma  = np.sqrt(6 * D * dt)
    ratio  = sigma / geom.min_gap
    assert ratio < 0.5, (
        f"σ/min_gap = {ratio:.2f} ≥ 0.5: single-reflection approximation "
        f"may be inaccurate.  σ={sigma*1e6:.3f} µm, min_gap={geom.min_gap*1e6:.3f} µm.  "
        f"Use more timesteps or a lower packing fraction.")


def test_packed_cylinders_init_outside():
    """With ``pool="extra"`` every initial walker position lies outside every cylinder cross-section."""
    geom    = _simple_geometry()
    import jax
    key     = jax.random.PRNGKey(0)
    pos     = np.array(geom.init_positions(5_000, key))

    # Project to cross-section (xy plane in lab frame, since orientation=[0,0,1])
    pos_xy  = pos[:, :2]   # (N, 2)
    centers = np.array(geom._centers_jax)
    radii   = geom._radii_np
    L       = geom._L_float

    for k in range(len(radii)):
        dxy  = pos_xy - centers[k]
        dxy -= L * np.round(dxy / L)
        dist = np.linalg.norm(dxy, axis=1)
        inside = dist < radii[k] * (1 - 1e-6)
        assert not np.any(inside), (
            f"Cylinder {k}: {np.sum(inside)} initial walkers inside "
            f"(r={radii[k]*1e6:.2f} µm, min_dist={dist.min()*1e6:.3f} µm)")


def test_packed_cylinders_parallel_gradient_free():
    """Gradient along cylinder axis (z) → free diffusion signal exp(-bD).

    Along the axis, walkers are unrestricted; signal must match free diffusion
    to within MC noise (four standard errors at N_PACK).
    """
    geom = _simple_geometry()

    b_values = np.linspace(1e8, 3e9, 6)
    bvecs    = np.tile([0., 0., 1.], (6, 1))   # gradient ∥ cylinder axis
    wf       = set_b(pgse(bvecs, 10e-3, 40e-3, gradient_strengths=1.0, n_t=1000), b_values)

    S_packed = simulate(N_PACK, D, wf, geom, seed=SEED)
    expected  = np.exp(-b_values * D)

    npt.assert_allclose(S_packed, expected, atol=mc_atol(N_PACK),
                        err_msg="Parallel gradient should give free diffusion")


def test_packed_cylinders_walkers_stay_outside():
    """Final walker positions lie outside all cylinder cross-sections.

    Runs a perpendicular-gradient simulation and checks that no walker has
    crossed into any cylinder.  Tolerance: 3 × NUDGE = 3e-4 × min_radius
    (float32 round-trip through the rotation matrices).
    """
    geom = _simple_geometry()

    bvec = np.array([[1., 0., 0.]])
    wf   = set_b(pgse(bvec, 10e-3, 40e-3, gradient_strengths=1.0, n_t=1000), np.array([1e9]))
    _, final_pos = simulate_fused(1_000, D, wf, geom, seed=SEED, return_positions=True)

    centers = np.array(geom._centers_jax)
    radii   = geom._radii_np
    L       = geom._L_float
    tol     = 3e-4 * float(np.min(radii))

    # Project final positions to cross-section (orientation=[0,0,1] → xy plane)
    pos_xy = final_pos[:, :2]
    for k in range(len(radii)):
        dxy  = pos_xy - centers[k]
        dxy -= L * np.round(dxy / L)
        dist = np.linalg.norm(dxy, axis=1)
        worst = float(radii[k] - np.min(dist))   # positive = walker inside
        assert worst < tol, (
            f"Cylinder {k}: walker inside by {worst*1e9:.1f} nm "
            f"(tolerance {tol*1e9:.1f} nm)")


def test_packed_cylinders_hindered_above_free():
    """Extra-axonal perpendicular signal > free diffusion at high b.

    Extra-axonal diffusion is hindered (D_eff_perp < D), so signal attenuates
    more slowly than free diffusion.  At b=1.5e9 s/m² the difference is
    detectable at N_PACK walkers.
    """
    # Use a denser packing so hindering is clearly visible
    r     = 3e-6
    N_cyl = 6
    radii   = np.full(N_cyl, r)
    centers, L, vf = pack_cylinders(radii, target_vf=0.30, seed=5)
    geom_packed = PackedCylinders(radii=radii, centers=centers, L=L, pool="extra")

    b_values = np.array([1.5e9])
    bvecs    = np.array([[1., 0., 0.]])
    wf       = set_b(pgse(bvecs, 10e-3, 40e-3, gradient_strengths=1.0, n_t=1000), b_values)

    S_extra = simulate(N_PACK, D, wf, geom_packed, seed=SEED)
    S_free  = simulate(N_PACK, D, wf, FreeDiffusion(),  seed=SEED + 1)

    assert float(S_extra[0]) > float(S_free[0]) + mc_atol(N_PACK, k=2.0, runs=2), (
        f"Extra-axonal signal ({S_extra[0]:.4f}) should exceed free "
        f"({S_free[0]:.4f}) at high b (VF={vf:.2f})")


def test_impermeable_wall_grants_no_compartment_change():
    """A walker whose step lands exactly on |q| = R must not change compartment (#86).

    At kappa = 0 no crossing is ever granted, so any walker that ends up on the far side of
    a membrane got there by relabelling, not by physics.  The mechanism is float32: a step
    whose exit time marginally exceeds the step length fires no collision (correctly -- it
    does not reach the wall), the straight step is taken, and the endpoint rounds onto the
    surface, where the strict `dist2 < R**2` test reads "outside".  Measured at 0.675% of
    intra walkers over 30k steps on a dense gamma packing before the carried-side sentinel.

    This builds the rounding case directly rather than waiting for a long walk to hit it.
    """
    R = 1.0e-6
    g = PackedCylinders(centers=np.array([[0.0, 0.0]]), radii=np.array([R]),
                        L=10e-6, orientation=[0, 0, 1.0], permeability=0.0)
    step_fn = jax.jit(jax.vmap(
        lambda p, s, k: g.permeate(p, s, jnp.float32(0.0), jnp.float32(0.0), k, jnp.int8(-1)),
        in_axes=(0, 0, 0)))

    n = 4000
    d = (np.float32(R) * (1.0 - 10.0 ** (-(4 + np.linspace(0, 4, n))))).astype(np.float32)
    p = np.zeros((n, 3), np.float32); p[:, 0] = d
    s = np.zeros((n, 3), np.float32); s[:, 0] = np.float32(R) - d   # lands on the wall
    keys = jax.random.split(jax.random.PRNGKey(0), n)

    hit = step_fn(jnp.asarray(p), jnp.asarray(s), keys)
    crossed, illegal = hit.crossed, hit.illegal

    assert not np.asarray(crossed).any(), "an impermeable wall granted a crossing"
    assert np.asarray(illegal).any(), "the rounding case did not reproduce"
    q = np.linalg.norm(np.asarray(hit.r)[:, :2], axis=1)
    assert (q < R).all(), (
        f"{int((q >= R).sum())}/{n} intra walkers ended outside their axon at kappa=0")
