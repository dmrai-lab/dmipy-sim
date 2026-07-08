"""Tests for dmipy_sim.noise — Rician and NC-chi measurement noise.

Validation approach: use the exact second-moment identity for the
Rice distribution:
    E[r²] = ν² + 2σ²   for r ~ Rice(ν, σ)

With N realisations, the MC estimate of E[r²] should agree with ν² + 2σ²
to within ~4σ² / sqrt(N) (standard error of the squared estimator).
We use N = 200_000 which gives SE < 0.1% for σ = 0.05.
"""

import numpy as np
import pytest

from dmipy_sim.noise import (
    add_rician_noise,
    add_rician_noise_batch,
    add_nc_chi_noise,
    estimate_sigma,
)


# ── Second-moment identity: E[r²] = ν² + 2σ² ────────────────────────────────

def _check_second_moment(nu, sigma, n_real=200_000, rtol=0.01):
    """Verify E[r²] ≈ ν² + 2σ² via Monte Carlo."""
    signal = np.full(1, nu, dtype=np.float32)
    noisy = add_rician_noise_batch(signal, sigma, n_realizations=n_real, seed=42)
    # noisy shape: (n_real, 1)
    em2 = float(np.mean(noisy[:, 0] ** 2))
    expected = nu ** 2 + 2 * sigma ** 2
    np.testing.assert_allclose(em2, expected, rtol=rtol,
        err_msg=f"E[r²] mismatch for ν={nu}, σ={sigma}: got {em2:.6f}, expected {expected:.6f}")


@pytest.mark.parametrize("nu, sigma", [
    (0.9,  0.05),   # high SNR
    (0.5,  0.05),   # mid SNR
    (0.1,  0.05),   # low SNR (high b)
    (0.0,  0.05),   # noise floor (ν=0)
    (0.9,  0.02),   # high SNR, low noise
    (0.1,  0.10),   # very low SNR
])
def test_rician_second_moment(nu, sigma):
    """E[r²] = ν² + 2σ² for Rice(ν, σ) — verifies generation model."""
    _check_second_moment(nu, sigma)


# ── Mean at zero signal: E[r|ν=0] = σ√(π/2) (Rayleigh) ────────────────────

def test_rayleigh_mean_at_zero_signal():
    """At ν=0, Rice reduces to Rayleigh: E[r] = σ√(π/2)."""
    sigma = 0.05
    signal = np.zeros(1, dtype=np.float32)
    noisy = add_rician_noise_batch(signal, sigma, n_realizations=200_000, seed=7)
    mean_r = float(np.mean(noisy[:, 0]))
    expected = sigma * np.sqrt(np.pi / 2)
    np.testing.assert_allclose(mean_r, expected, rtol=0.01,
        err_msg=f"Rayleigh mean: got {mean_r:.5f}, expected {expected:.5f}")


# ── Determinism: same seed → identical output ───────────────────────────────

def test_rician_seed_determinism():
    """Same seed must produce byte-identical output."""
    signal = np.array([0.8, 0.5, 0.1, 0.0], dtype=np.float32)
    r1 = add_rician_noise(signal, sigma=0.05, seed=99)
    r2 = add_rician_noise(signal, sigma=0.05, seed=99)
    np.testing.assert_array_equal(r1, r2, err_msg="Same seed → same output")


def test_rician_different_seeds():
    """Different seeds must produce different outputs."""
    signal = np.array([0.8, 0.5, 0.1], dtype=np.float32)
    r1 = add_rician_noise(signal, sigma=0.05, seed=1)
    r2 = add_rician_noise(signal, sigma=0.05, seed=2)
    assert not np.allclose(r1, r2), "Different seeds should differ"


# ── Output shape and dtype ───────────────────────────────────────────────────

def test_rician_shape_and_dtype():
    signal = np.linspace(0, 1, 20, dtype=np.float32)
    noisy = add_rician_noise(signal, sigma=0.02, seed=0)
    assert noisy.shape == signal.shape
    assert noisy.dtype == np.float32


def test_rician_batch_shape():
    signal = np.linspace(0, 1, 15, dtype=np.float32)
    noisy = add_rician_noise_batch(signal, sigma=0.05, n_realizations=50, seed=0)
    assert noisy.shape == (50, 15)
    assert noisy.dtype == np.float32


# ── Non-negativity: magnitude is always ≥ 0 ─────────────────────────────────

def test_rician_non_negative():
    signal = np.zeros(100, dtype=np.float32)
    noisy = add_rician_noise(signal, sigma=0.05, seed=0)
    assert np.all(noisy >= 0), "Rician magnitude must be non-negative"


# ── sigma=0 guard ────────────────────────────────────────────────────────────

def test_rician_sigma_zero_raises():
    with pytest.raises(ValueError, match="sigma must be positive"):
        add_rician_noise(np.array([0.5]), sigma=0.0)


def test_rician_sigma_negative_raises():
    with pytest.raises(ValueError, match="sigma must be positive"):
        add_rician_noise(np.array([0.5]), sigma=-0.01)


# ── NC-chi: L=1 should match Rician (identical seed) ────────────────────────

def test_nc_chi_l1_matches_rician():
    """NC-chi with 1 coil must equal Rician (same seed, same signal)."""
    signal = np.array([0.8, 0.5, 0.2], dtype=np.float32)
    r_rician  = add_rician_noise(signal, sigma=0.05, seed=3)
    r_nc_chi1 = add_nc_chi_noise(signal, sigma=0.05, n_coils=1, seed=3)
    np.testing.assert_allclose(r_rician, r_nc_chi1, rtol=1e-5,
        err_msg="NC-chi L=1 should equal Rician")


# ── NC-chi: second moment identity for L coils ───────────────────────────────
# With signal_per_coil = ν/sqrt(L) and L independent pairs (n_real, n_imag):
#   E[sos] = L · ((ν/sqrt(L))² + 2σ²) = ν² + 2Lσ²
# So E[r²] = E[sqrt(sos)²] = E[sos] = ν² + 2·L·σ²
# The noise floor grows with coil count — this is physically correct for SOS.

def test_nc_chi_second_moment():
    """NC-chi second moment E[r²] = ν² + 2·L·σ² (L independent noise channels)."""
    nu, sigma, n_coils = 0.5, 0.05, 4
    signal = np.full(1, nu, dtype=np.float32)
    results = np.array([
        add_nc_chi_noise(signal, sigma, n_coils, seed=s)
        for s in range(2000)
    ]).ravel()
    em2 = float(np.mean(results ** 2))
    expected = nu ** 2 + 2 * n_coils * sigma ** 2
    np.testing.assert_allclose(em2, expected, rtol=0.03,
        err_msg=f"NC-chi E[r²]: got {em2:.5f}, expected {expected:.5f}")


# ── estimate_sigma ────────────────────────────────────────────────────────────

def test_estimate_sigma_gaussian():
    """estimate_sigma should recover σ from many b0 measurements."""
    rng = np.random.default_rng(0)
    sigma_true = 0.03
    # Simulate b0 measurements: ν=1, add Rician, divide by 1 = noisy ~N(1, σ²)
    n_b0 = 5000
    b0 = 1.0 + rng.normal(0, sigma_true, size=n_b0)
    sigma_std, sigma_mad = estimate_sigma(b0)
    np.testing.assert_allclose(sigma_std, sigma_true, rtol=0.05)
    np.testing.assert_allclose(sigma_mad, sigma_true, rtol=0.05)
