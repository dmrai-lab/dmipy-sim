"""Measurement noise models for synthetic MRI data.

dmipy-sim produces noiseless signals: simulate() returns mean(cos(φ)),
the ensemble-averaged phase — the latent signal ν = E(b) ∈ [0, 1].

This module adds the measurement process: thermal coil noise added in
the complex domain, magnitude reconstruction, giving Rician-distributed
observations.

Physical model (single-coil / pre-combined magnitude):
  M_noisy = (ν + n_real) + i·n_imag,   n_real, n_imag ~ N(0, σ²)
  r = |M_noisy| = sqrt((ν + n_real)² + n_imag²)
  r ~ Rice(ν, σ)

This is NOT equivalent to clipping Gaussian noise.  At low SNR (ν ≈ 0),
r ~ Rayleigh(σ) with E[r] = σ√(π/2) > 0 — the Rician noise floor.

Multi-coil SOS reconstruction (L coils):
  r = sqrt(Σᵢ |Mᵢ_noisy|²) / sqrt(L)   (normalised to preserve scale)
  r ~ NoncentralChi(2L dof, ν, σ) / sqrt(L) (approximately)

Convention: all signals are normalised to S₀ = 1.
σ = σ_physical / S₀ = 1 / SNR₀  (SNR at b = 0 measurement).
Typical values: σ = 0.01 (SNR₀ = 100), σ = 0.05 (SNR₀ = 20).
"""

import numpy as np
import jax
import jax.numpy as jnp


def add_rician_noise(signal, sigma, seed=0):
    """Add Rician-distributed magnitude noise to noiseless signals.

    Generates two independent Gaussian noise channels and takes the
    magnitude, producing Rice(ν, σ) observations.

    Parameters
    ----------
    signal : array_like, shape (n_measurements,)
        Noiseless normalised signal from simulate(), values in [0, 1].
    sigma : float
        Noise standard deviation in the same normalised units.
        sigma = 1 / SNR₀  where SNR₀ is the SNR at b = 0.
    seed : int
        JAX PRNG seed. Same seed → identical noise realisation.

    Returns
    -------
    noisy : np.ndarray, shape (n_measurements,), float32
        Rician-noisy magnitude signal.

    Notes
    -----
    Second moment check: E[r²] = ν² + 2σ² for Rice(ν, σ).
    This can be used to verify the noise generation (see test_noise.py).
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    signal = np.asarray(signal, dtype=np.float32)
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    n_real = sigma * jax.random.normal(k1, shape=signal.shape, dtype=jnp.float32)
    n_imag = sigma * jax.random.normal(k2, shape=signal.shape, dtype=jnp.float32)
    noisy = jnp.sqrt((signal + n_real) ** 2 + n_imag ** 2)
    return np.array(noisy, dtype=np.float32)


def add_rician_noise_batch(signal, sigma, n_realizations, seed=0):
    """Add Rician noise to produce multiple independent noise realisations.

    Parameters
    ----------
    signal : array_like, shape (n_measurements,)
        Noiseless normalised signal.
    sigma : float
        Noise standard deviation.
    n_realizations : int
        Number of independent noise realisations to generate.
    seed : int
        JAX PRNG seed.

    Returns
    -------
    noisy : np.ndarray, shape (n_realizations, n_measurements), float32
        Each row is one independent Rician-noisy realisation.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    signal = np.asarray(signal, dtype=np.float32)
    n_meas = signal.shape[0]
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    n_real = sigma * jax.random.normal(k1, shape=(n_realizations, n_meas), dtype=jnp.float32)
    n_imag = sigma * jax.random.normal(k2, shape=(n_realizations, n_meas), dtype=jnp.float32)
    noisy = jnp.sqrt((signal[None, :] + n_real) ** 2 + n_imag ** 2)
    return np.array(noisy, dtype=np.float32)


def add_nc_chi_noise(signal, sigma, n_coils, seed=0):
    """Add noncentral-chi magnitude noise (multi-coil SOS reconstruction).

    For L coils with equal sensitivity, the sum-of-squares (SOS) magnitude
    follows a noncentral chi distribution with 2L degrees of freedom.
    Rician is the L = 1 special case.

    Implementation: L independent Rician channels summed in quadrature,
    normalised by 1/sqrt(L) so that the noise floor σ_eff is preserved.

    Parameters
    ----------
    signal : array_like, shape (n_measurements,)
        Noiseless normalised signal, values in [0, 1].
    sigma : float
        Per-coil noise standard deviation. Same σ in each coil.
        The effective noise floor of the SOS magnitude ≈ σ√(2L)·√(π/4L)
        = σ√(π/2) for all L (same as Rician), so σ interpretation is
        consistent across coil counts.
    n_coils : int
        Number of coils L ≥ 1. L = 1 gives Rician.
    seed : int
        JAX PRNG seed.

    Returns
    -------
    noisy : np.ndarray, shape (n_measurements,), float32
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if n_coils < 1:
        raise ValueError(f"n_coils must be >= 1, got {n_coils}")
    signal = np.asarray(signal, dtype=np.float32)
    n_meas = signal.shape[0]
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 2 * n_coils)

    # Each coil gets signal / sqrt(n_coils) of the total signal power
    signal_per_coil = signal / np.sqrt(n_coils)
    sos = jnp.zeros(n_meas, dtype=jnp.float32)
    for i in range(n_coils):
        n_real = sigma * jax.random.normal(keys[2 * i],     shape=(n_meas,), dtype=jnp.float32)
        n_imag = sigma * jax.random.normal(keys[2 * i + 1], shape=(n_meas,), dtype=jnp.float32)
        sos = sos + (signal_per_coil + n_real) ** 2 + n_imag ** 2
    return np.array(jnp.sqrt(sos), dtype=np.float32)


def estimate_sigma(b0_signals):
    """Estimate noise standard deviation from repeated b = 0 measurements.

    Uses the standard deviation of the b0 signal as a proxy for σ.
    Valid when the b0 measurements span multiple acquisitions (e.g.
    interleaved b0 volumes), not when all b0s share the same noise
    realisation.

    For a single array of b0 voxel values (across voxels or repetitions),
    also returns the MAD-based estimate which is more robust to outliers.

    Parameters
    ----------
    b0_signals : array_like, shape (n_b0,)
        Normalised b = 0 signal values (after dividing by mean b0).

    Returns
    -------
    sigma_std : float
        Standard deviation estimate (direct).
    sigma_mad : float
        MAD-based estimate: MAD / 0.6745 (consistent with Gaussian σ).
    """
    b0 = np.asarray(b0_signals, dtype=np.float64)
    sigma_std = float(np.std(b0))
    mad = np.median(np.abs(b0 - np.median(b0)))
    sigma_mad = float(mad / 0.6745)
    return sigma_std, sigma_mad
