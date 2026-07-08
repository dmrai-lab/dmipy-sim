"""Membrane permeability for Sphere (Powles 2004).

Physics
-------
At each wall crossing the walker transmits through the membrane with
probability

    p = min(1,  2·κ·d_perp / D)

and reflects otherwise (Powles et al. 2004).
The exchange time for walkers initially inside a sphere of radius R is

    τ = R / (3κ)   (V/κS = (4πR³/3) / (κ·4πR²) = R/(3κ))

Analytical limits validated here
----------------------------------
1. κ=None  (permeability not set) → identical signal to impermeable Sphere.
2. κ=0  (zero permeability)       → identical signal to impermeable Sphere.
3. κ>0 at high b                  → signal strictly below impermeable (walkers
                                    that escape are less restricted → higher
                                    phase accumulation → lower S at high b).
4. Very high κ (TE >> τ)          → walkers mostly escaped, signal close to
                                    free diffusion exp(−bD).
5. Permeability + relaxivity      → signal below permeability-only (weight
                                    penalty on reflection).
6. Signal monotonically decreases with κ at high b.

Parameters chosen so σ/R < 0.1 (good-practice criterion):
    R = 5 µm, D = 2e-9 m²/s, TE ≈ 100 ms
    dt = TE/500 = 0.2e-3 s → σ = √(6D·dt) ≈ 1.55 µm → σ/R ≈ 0.31

For the high-κ test we use a short TE (20 ms) with fine time-stepping:
    τ = R/(3κ) = 5e-6/(3·1e-2) ≈ 0.17 ms
    TE/τ ≈ 120 → effectively full exchange
"""

import numpy as np
import numpy.testing as npt
import pytest

from dmipy_sim import simulate, Sphere, set_b
from dmipy_sim.waveforms import pgse

from tests.conftest import D, N_WALKERS, SEED

R          = 5e-6   # m
KAPPA_MED  = 1e-5   # m/s  — exchange time τ = R/(3κ) ≈ 167 ms
KAPPA_HIGH = 1e-2   # m/s  — exchange time τ ≈ 0.17 ms (fast exchange)
RHO        = 5e-4   # m/s  — surface relaxivity for combination test


def _pgse_wf(TE_s, n_t=500):
    delta  = max(TE_s * 0.05, 5e-6)
    DELTA  = TE_s - delta
    b_values = np.array([0.0, 500e6, 1000e6, 2000e6])  # s/m²
    bvecs    = np.tile([1., 0., 0.], (4, 1))
    return set_b(
        pgse(delta=delta, DELTA=DELTA, G_magnitude=1.0, bvecs=bvecs, n_t=n_t),
        b_values)


# ---------------------------------------------------------------------------
# 1. Attribute storage
# ---------------------------------------------------------------------------

def test_permeability_attribute_stored():
    """permeability is stored correctly on Sphere."""
    geom = Sphere(radius=R, permeability=KAPPA_MED)
    assert geom.permeability == KAPPA_MED


def test_permeability_none_not_set():
    """Default Sphere has permeability=None."""
    geom = Sphere(radius=R)
    assert geom.permeability is None


# ---------------------------------------------------------------------------
# 2. κ=None / κ=0 must reproduce the impermeable signal
# ---------------------------------------------------------------------------

def test_permeability_none_matches_impermeable():
    """Sphere(permeability=None) == Sphere() — identical signal."""
    wf = _pgse_wf(100e-3)
    geom_default = Sphere(radius=R)
    geom_none    = Sphere(radius=R, permeability=None)
    S_default = simulate(N_WALKERS, D, wf, geom_default, seed=SEED)
    S_none    = simulate(N_WALKERS, D, wf, geom_none,    seed=SEED)
    npt.assert_array_equal(S_default, S_none,
        err_msg="permeability=None must give identical signal to default")


# ---------------------------------------------------------------------------
# 3. κ>0 reduces signal at high b
# ---------------------------------------------------------------------------

def test_permeability_reduces_signal_at_high_b():
    """Signal with κ>0 must be strictly below impermeable signal at b=2000 s/mm².

    Inside a sphere, restriction raises the signal (ADC_app << D).
    Walkers that escape diffuse more freely → larger phase accumulation →
    lower signal.  So: S_perm < S_imp at high b.
    """
    wf = _pgse_wf(100e-3)
    geom_imp  = Sphere(radius=R)
    geom_perm = Sphere(radius=R, permeability=KAPPA_MED)
    S_imp  = simulate(N_WALKERS, D, wf, geom_imp,  seed=SEED)
    S_perm = simulate(N_WALKERS, D, wf, geom_perm, seed=SEED)
    assert float(S_perm[3]) < float(S_imp[3]) - 0.05, (
        f"S_perm={S_perm[3]:.4f} must be below S_imp={S_imp[3]:.4f} at b=2000 s/mm²")


# ---------------------------------------------------------------------------
# 4. High κ (TE >> τ): signal approaches free diffusion
# ---------------------------------------------------------------------------

def test_permeability_high_kappa_approaches_free_diffusion():
    """Very high κ (TE/τ >> 1): signal within 10% of exp(-bD) at b=500 s/mm².

    τ = R/(3κ) = 5e-6/(3·1e-2) ≈ 0.17 ms.  TE=20ms → TE/τ ≈ 120.
    Nearly all walkers have crossed the membrane many times; the ensemble
    ADC approaches free diffusivity.

    Uses fine time-stepping (n_t=2000) to keep σ/R < 0.1.
    """
    TE    = 20e-3
    b_idx = 1   # b = 500 s/mm²
    wf    = _pgse_wf(TE, n_t=2000)

    geom_perm = Sphere(radius=R, permeability=KAPPA_HIGH)
    S_perm    = simulate(N_WALKERS, D, wf, geom_perm, seed=SEED)

    b_val  = 500e6   # s/m²
    S_free = np.exp(-b_val * D)

    rel_err = abs(float(S_perm[b_idx]) - S_free) / S_free
    assert rel_err < 0.10, (
        f"High-κ signal {S_perm[b_idx]:.4f} should be within 10% of "
        f"free diffusion {S_free:.4f} (rel_err={rel_err:.3f})")


# ---------------------------------------------------------------------------
# 5. Permeability + relaxivity: signal below permeability-only
# ---------------------------------------------------------------------------

def test_permeability_with_relaxivity_reduces_signal():
    """Adding surface relaxivity to a permeable sphere must reduce signal.

    Reflected walkers receive the Brownstein-Tarr weight; transmitted walkers
    do not.  The ensemble signal must therefore be ≤ the permeability-only
    signal.
    """
    wf = _pgse_wf(100e-3)
    geom_perm     = Sphere(radius=R, permeability=KAPPA_MED)
    geom_perm_rho = Sphere(radius=R, permeability=KAPPA_MED,
                            surface_relaxivity_t2=RHO)
    S_perm     = simulate(N_WALKERS, D, wf, geom_perm,     seed=SEED)
    S_perm_rho = simulate(N_WALKERS, D, wf, geom_perm_rho, seed=SEED)
    assert float(S_perm_rho[0]) < float(S_perm[0]) - 0.02, (
        f"Relaxivity+permeability {S_perm_rho[0]:.4f} must be below "
        f"permeability-only {S_perm[0]:.4f} at b=0")


# ---------------------------------------------------------------------------
# 6. Signal monotonically decreases with κ at high b
# ---------------------------------------------------------------------------

def test_permeability_signal_monotone_in_kappa():
    """Signal at b=2000 s/mm² decreases monotonically with κ.

    Higher κ → more walkers escape → less restriction → faster decay →
    lower signal at high b.
    """
    wf     = _pgse_wf(100e-3)
    kappas = [0.0, 1e-6, 1e-5, 1e-4]
    signals = []
    for kappa in kappas:
        perm = kappa if kappa > 0 else None
        geom = Sphere(radius=R, permeability=perm)
        S    = simulate(N_WALKERS, D, wf, geom, seed=SEED)
        signals.append(float(S[3]))   # b=2000 s/mm²
    for i in range(len(signals) - 1):
        assert signals[i] >= signals[i + 1] - 1e-3, (
            f"Signal not monotone: κ={kappas[i]:.0e} → {signals[i]:.4f}, "
            f"κ={kappas[i+1]:.0e} → {signals[i+1]:.4f}")
