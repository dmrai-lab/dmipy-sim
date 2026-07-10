"""Membrane permeability for Ellipsoid (Powles 2004).

Physics
-------
Same Powles (2004) probabilistic crossing as for Sphere and Cylinder:

    p = min(1,  2·κ·d_perp / D)

For a general ellipsoid the exchange time is

    τ = V / (κ · S)

For a prolate ellipsoid with semi-axes [a, a, c] (c > a):
    V = (4π/3) · a² · c
    S ≈ 2π · a² · (1 + (c/a) · arcsin(e) / e)    where e = √(1 - a²/c²)

Analytical limits validated here
----------------------------------
1. κ=None  → identical signal to impermeable Ellipsoid.
2. κ>0 at high b → signal below impermeable (escaped walkers diffuse
   more freely along the long axis → greater signal decay).
3. Very high κ (TE >> τ) → ensemble ADC approaches free diffusion
   (tested at b=500 s/mm², b=1000 s/mm²).
4. Permeability + relaxivity → signal below permeability-only.
5. Signal monotonically decreases with κ at high b.

Parameters: a=3 µm, c=9 µm (prolate, aspect ratio 3), D=2e-9 m²/s
    σ = √(6D·dt) with dt = TE/n_t; n_t=500, TE=100ms → σ ≈ 1.55 µm
    σ/a ≈ 0.52 (borderline for good practice, sufficient for monotone tests)
    Use n_t=2000 for quantitative high-κ test.
"""

import numpy as np
import numpy.testing as npt

from dmipy_sim import simulate, Ellipsoid, set_b
from dmipy_sim.waveforms import pgse

from tests.conftest import D, N_WALKERS, SEED

A          = 3e-6   # m  — short semi-axis
C_AXIS     = 9e-6   # m  — long semi-axis
SEMIAXES   = [A, A, C_AXIS]
KAPPA_MED  = 1e-5   # m/s
KAPPA_HIGH = 5e-3   # m/s
RHO        = 5e-4   # m/s


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
    """permeability is stored correctly on Ellipsoid."""
    geom = Ellipsoid(semiaxes=SEMIAXES, permeability=KAPPA_MED)
    assert geom.permeability == KAPPA_MED


def test_permeability_none_not_set():
    """Default Ellipsoid has permeability=None."""
    geom = Ellipsoid(semiaxes=SEMIAXES)
    assert geom.permeability is None


# ---------------------------------------------------------------------------
# 2. κ=None must reproduce the impermeable signal
# ---------------------------------------------------------------------------

def test_permeability_none_matches_impermeable():
    """Ellipsoid(permeability=None) == Ellipsoid() — identical signal."""
    wf           = _pgse_wf(100e-3)
    geom_default = Ellipsoid(semiaxes=SEMIAXES)
    geom_none    = Ellipsoid(semiaxes=SEMIAXES, permeability=None)
    S_default    = simulate(N_WALKERS, D, wf, geom_default, seed=SEED)
    S_none       = simulate(N_WALKERS, D, wf, geom_none,    seed=SEED)
    npt.assert_array_equal(S_default, S_none,
        err_msg="permeability=None must give identical signal to default")


# ---------------------------------------------------------------------------
# 3. κ>0 reduces signal at high b
# ---------------------------------------------------------------------------

def test_permeability_reduces_signal_at_high_b():
    """Signal with κ>0 must be strictly below impermeable at b=2000 s/mm².

    Escaping walkers have access to the long axis → larger ADC_app →
    faster signal decay at high b compared to fully confined walkers.
    """
    wf        = _pgse_wf(100e-3)
    geom_imp  = Ellipsoid(semiaxes=SEMIAXES)
    geom_perm = Ellipsoid(semiaxes=SEMIAXES, permeability=KAPPA_MED)
    S_imp     = simulate(N_WALKERS, D, wf, geom_imp,  seed=SEED)
    S_perm    = simulate(N_WALKERS, D, wf, geom_perm, seed=SEED)
    assert float(S_perm[3]) < float(S_imp[3]) - 0.05, (
        f"S_perm={S_perm[3]:.4f} must be below S_imp={S_imp[3]:.4f} "
        f"at b=2000 s/mm²")


# ---------------------------------------------------------------------------
# 4. High κ: signal approaches free diffusion
# ---------------------------------------------------------------------------

def test_permeability_high_kappa_approaches_free_diffusion():
    """Very high κ: signal within 15% of exp(-bD) at b=500 s/mm².

    Uses fine time-stepping (n_t=2000).  15% tolerance accounts for the
    finite TE/τ ratio and the fact that walkers start inside the ellipsoid
    and must diffuse back out multiple times.
    """
    TE    = 20e-3
    b_idx = 1   # b = 500 s/mm²
    wf    = _pgse_wf(TE, n_t=2000)

    geom_perm = Ellipsoid(semiaxes=SEMIAXES, permeability=KAPPA_HIGH)
    S_perm    = simulate(N_WALKERS, D, wf, geom_perm, seed=SEED)

    b_val  = 500e6
    S_free = np.exp(-b_val * D)

    rel_err = abs(float(S_perm[b_idx]) - S_free) / S_free
    assert rel_err < 0.15, (
        f"High-κ signal {S_perm[b_idx]:.4f} should be within 15% of "
        f"free diffusion {S_free:.4f} (rel_err={rel_err:.3f})")


# ---------------------------------------------------------------------------
# 5. Permeability + relaxivity: signal below permeability-only
# ---------------------------------------------------------------------------

def test_permeability_with_relaxivity_reduces_signal():
    """Adding surface relaxivity to a permeable ellipsoid must reduce signal."""
    wf            = _pgse_wf(100e-3)
    geom_perm     = Ellipsoid(semiaxes=SEMIAXES, permeability=KAPPA_MED)
    geom_perm_rho = Ellipsoid(semiaxes=SEMIAXES, permeability=KAPPA_MED,
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
    """Signal at b=2000 s/mm² decreases monotonically with κ."""
    wf      = _pgse_wf(100e-3)
    kappas  = [0.0, 1e-6, 1e-5, 1e-4]
    signals = []
    for kappa in kappas:
        perm = kappa if kappa > 0 else None
        geom = Ellipsoid(semiaxes=SEMIAXES, permeability=perm)
        S    = simulate(N_WALKERS, D, wf, geom, seed=SEED)
        signals.append(float(S[3]))
    for i in range(len(signals) - 1):
        assert signals[i] >= signals[i + 1] - 1e-3, (
            f"Signal not monotone: κ={kappas[i]:.0e} → {signals[i]:.4f}, "
            f"κ={kappas[i+1]:.0e} → {signals[i+1]:.4f}")
