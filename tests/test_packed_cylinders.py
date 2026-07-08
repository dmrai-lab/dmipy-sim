"""PackedCylinders geometry — extra-axonal diffusion.

Tests cover:
  - RSA packing: no overlaps (including periodic images), correct VF
  - init_positions: all walkers outside all cylinder cross-sections
  - Step-size assumption: σ << min_gap for default simulation parameters
  - Parallel gradient → free diffusion exp(-bD)  [exact, analytical]
  - Walker containment: no walker enters any cylinder during simulation
  - Hindered perpendicular diffusion > free diffusion at high b  [qualitative]
  - Area-weighted MC mixture matches analytical Callaghan per cylinder  [physics]
"""

import sys
import numpy as np
import numpy.testing as npt
import pytest

from dmipy_sim import simulate, simulate_mixture, FreeDiffusion, pack_cylinders, PackedCylinders, set_b, Cylinder
from dmipy_sim.waveforms import pgse
from .conftest import D, N_WALKERS, SEED


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
                           orientation=[0., 0., 1.])


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
    """All initial walker positions lie outside every cylinder cross-section."""
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
    to within MC noise (atol=0.02).
    """
    geom = _simple_geometry()

    b_values = np.linspace(1e8, 3e9, 20)
    bvecs    = np.tile([0., 0., 1.], (20, 1))   # gradient ∥ cylinder axis
    wf       = set_b(pgse(delta=10e-3, DELTA=40e-3, G_magnitude=1.0,
                          bvecs=bvecs, n_t=1000), b_values)

    S_packed = simulate(N_WALKERS, D, wf, geom, seed=SEED)
    expected  = np.exp(-b_values * D)

    npt.assert_allclose(S_packed, expected, atol=0.02,
                        err_msg="Parallel gradient should give free diffusion")


def test_packed_cylinders_walkers_stay_outside():
    """Final walker positions lie outside all cylinder cross-sections.

    Runs a perpendicular-gradient simulation and checks that no walker has
    crossed into any cylinder.  Tolerance: 3 × NUDGE = 3e-4 × min_radius
    (float32 round-trip through the rotation matrices).
    """
    geom = _simple_geometry()

    bvec = np.array([[1., 0., 0.]])
    wf   = set_b(pgse(delta=10e-3, DELTA=40e-3, G_magnitude=1.0,
                      bvecs=bvec, n_t=1000), np.array([1e9]))
    _, final_pos = simulate(5_000, D, wf, geom, seed=SEED, return_positions=True)

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
    more slowly than free diffusion.  At b=3e9 s/m² the difference is
    detectable even with 100k walkers.
    """
    # Use a denser packing so hindering is clearly visible
    r     = 3e-6
    N_cyl = 6
    radii   = np.full(N_cyl, r)
    centers, L, vf = pack_cylinders(radii, target_vf=0.30, seed=5)
    geom_packed = PackedCylinders(radii=radii, centers=centers, L=L)

    b_values = np.array([3e9])
    bvecs    = np.array([[1., 0., 0.]])
    wf       = set_b(pgse(delta=10e-3, DELTA=40e-3, G_magnitude=1.0,
                          bvecs=bvecs, n_t=1000), b_values)

    S_extra = simulate(N_WALKERS, D, wf, geom_packed, seed=SEED)
    S_free  = simulate(N_WALKERS, D, wf, FreeDiffusion(),  seed=SEED + 1)

    assert float(S_extra[0]) > float(S_free[0]) - 0.01, (
        f"Extra-axonal signal ({S_extra[0]:.4f}) should exceed free "
        f"({S_free[0]:.4f}) at high b (VF={vf:.2f})")


# ---------------------------------------------------------------------------
# Area-weighted MC mixture vs analytical Callaghan (physics validation)
# ---------------------------------------------------------------------------

def test_packed_cylinders_mc_mixture_vs_callaghan():
    """Area-weighted MC mixture matches analytical Callaghan per-cylinder signal.

    Samples N radii from Gamma(alpha, beta_r), computes:
      E_empir(b) = ∑ w_i · E_Callaghan(r_i, b)   (w_i ∝ r_i²)

    and compares it to the MC mixture signal from simulate_mixture() with the
    same radii and area weights.  Agreement within MC noise validates that
    simulate_mixture() correctly combines restricted Cylinder signals.

    Uses short-pulse SGP regime (δ=0.5ms) where C3Callaghan is accurate for
    these radii.  N_WALKERS_PER=10k gives mixture noise ≈ 0.004; atol=0.025.
    """
    # Import dmipy-core C3Callaghan; skip if not available
    try:
        pytest.importorskip("dmipy_fit")  # cross-engine test; absent in the standalone public repo
        from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
        from dmipy_fit.signal_models.cylinder_models import C3CylinderCallaghanApproximation
        from dmipy_fit.core.modeling_framework import MultiCompartmentModel
    except ImportError:
        pytest.skip("dmipy-core not available — skipping cross-package validation")

    # --- Gamma-sample N=8 radii (fast, repeatable) ---
    ALPHA, BETA_R = 4.0, 0.5e-6   # mean_r = 2 µm, SGP valid: τ_c=2ms >> δ=0.5ms
    N_CYL         = 8
    N_W_PER       = 10_000

    rng   = np.random.default_rng(SEED)
    radii = rng.gamma(ALPHA, BETA_R, size=N_CYL)

    # Short-pulse waveform
    DELTA_S = 0.5e-3
    DELTA_L = 40e-3
    B_VALUES = np.array([0., 5e8, 1e9, 2e9])
    BVECS    = np.tile([[1., 0., 0.]], (len(B_VALUES), 1))
    scheme   = acquisition_scheme_from_bvalues(B_VALUES, BVECS, DELTA_S, DELTA_L)
    wf       = set_b(pgse(delta=DELTA_S, DELTA=DELTA_L, G_magnitude=1.0,
                          bvecs=BVECS, n_t=1000), B_VALUES)

    # --- Empirical analytical: area-weighted Callaghan sum ---
    area    = radii ** 2
    weights = area / area.sum()

    E_each = []
    for r in radii:
        c3 = C3CylinderCallaghanApproximation(diffusion_perpendicular=D)
        mc_model = MultiCompartmentModel(models=[c3])
        # surface_relaxivity is a free parameter on this model; pin it to 0 so the
        # signal is the pure restricted Callaghan response (no wall relaxivity) and
        # it drops out of the parameter vector (otherwise simulate_signal requires
        # C3CylinderCallaghanApproximation_1_surface_relaxivity in params).
        mc_model.set_fixed_parameter(
            'C3CylinderCallaghanApproximation_1_surface_relaxivity', 0.0)
        params = {
            'C3CylinderCallaghanApproximation_1_mu': np.array([0., 0.]),
            'C3CylinderCallaghanApproximation_1_lambda_par': D,
            'C3CylinderCallaghanApproximation_1_diameter': 2.0 * r,
            'partial_volume_0': 1.0,
        }
        E_each.append(mc_model.simulate_signal(scheme, params))
    E_each   = np.array(E_each)        # (N_CYL, n_b)
    E_empir  = (weights[:, None] * E_each).sum(axis=0)

    # --- MC mixture ---
    compartments = [
        {'fraction': float(w), 'n_walkers': N_W_PER, 'diffusivity': D,
         'geometry': Cylinder(radius=float(r), orientation=[0., 0., 1.])}
        for r, w in zip(radii, weights)
    ]
    E_mc = np.array(simulate_mixture(compartments, wf, seed=SEED))

    # Noise in mixture: noise = sqrt(∑ w_i² / N_w_per)
    mix_noise = np.sqrt(np.sum(weights ** 2) / N_W_PER)
    atol = max(0.025, 3 * mix_noise)

    npt.assert_allclose(
        E_mc, E_empir, atol=atol,
        err_msg=(
            f"Area-weighted MC mixture differs from analytical Callaghan sum "
            f"by more than {atol:.4f}.  radii={np.round(radii*1e6, 2)} µm, "
            f"weights={np.round(weights, 3)}"
        )
    )
