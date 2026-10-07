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

from tests.conftest import pgse_wf
import numpy.testing as npt

from dmipy_sim import simulate, Ellipsoid, set_b
from dmipy_sim.sequences import pgse

from tests.conftest import D, N_WALKERS, N_EXACT, SEED

A          = 3e-6   # m  — short semi-axis
C_AXIS     = 9e-6   # m  — long semi-axis
SEMIAXES   = [A, A, C_AXIS]
KAPPA_MED  = 1e-5   # m/s
KAPPA_HIGH = 6e-6   # m/s — derived in test_permeability_high_kappa_approaches_free_diffusion
                     # below from the engine's own sub-step budget, together with its TE
RHO        = 5e-4   # m/s


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
    wf           = pgse_wf(100e-3)
    geom_default = Ellipsoid(semiaxes=SEMIAXES)
    geom_none    = Ellipsoid(semiaxes=SEMIAXES, permeability=None)
    # exact-equality assertion -> scale-free; see N_EXACT in conftest (#93)
    S_default    = simulate(N_EXACT, D, wf, geom_default, seed=SEED)
    S_none       = simulate(N_EXACT, D, wf, geom_none,    seed=SEED)
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
    wf        = pgse_wf(100e-3)
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

    15% tolerance accounts for the finite TE/τ ratio and the fact that walkers start inside the
    ellipsoid and must diffuse back out multiple times. ``slew_rate=np.inf`` keeps the lobes
    square: ``pgse_wf``'s unit-amplitude reference build ramps at a FIXED 5 ms regardless of TE,
    so a short TE whose lobe is narrower than 5 ms is refused by the assembler as inexpressible
    (dmrai-lab/dmipy-sim#475) -- not a bug in the assembler, a placeholder amplitude too large
    for this TE.

    KAPPA_HIGH and TE are DERIVED below, not chosen by search, from the engine's own sub-step
    budget (``dmipy_sim.engine.physics``) -- the same derivation as the cylinder/sphere siblings
    (``test_permeability_crossing.py``, ``test_sphere_permeability.py``), but the ellipsoid uses
    TWO different radii: the sub-step rules read ``length_scales.min_feature = min(semiaxes) = A``
    (``walk_sub_steps``'s R/6 rule has no other notion of "the pore" for this geometry), while the
    exchange time is the solid's own ``tau = V / (kappa S)`` with its exact volume and surface
    area, which for ``SEMIAXES = [3e-6, 3e-6, 9e-6]`` (a prolate spheroid) gives an equivalent
    radius ``R_eq = 3 V / S = 3.661e-6 m``.

      * ``crossing_sub_steps`` needs ``n_cross = ceil(6 D dt / step_max**2)``,
        ``step_max = C D / (2 kappa)`` (``C = CROSSING_P_MAX = 3e-3``);
      * ``walk_sub_steps`` needs ``n_refl = ceil(dt / ((A/6)**2 / (6 D)))``, kappa-independent;
      * total sub-steps over the walk is ``TE * max(216 D / A**2, 24 kappa**2 / (C**2 D))``,
        independent of n_t; neither rule wastes budget against the other at
        ``kappa* = 3 D C / A``;
      * for a walker-step budget ``B = N * TE * 216 D / A**2`` (N = 5,000, this file's N_EXACT),
        ``TE = B A**2 / (216 D N)`` and the achieved ``TE / tau = TE * 3 kappa* / R_eq``
        (the ellipsoid's tau uses ``R_eq``, not ``A``, so this does not collapse to as clean a
        closed form in B as the cylinder/sphere siblings; it is evaluated numerically below).
      * solving ``TE / tau >= 10`` at A = 3e-6 m, R_eq = 3.661e-6 m, D = 2e-9 m^2/s, C = 3e-3,
        N = 5,000 gives ``kappa = 6.0e-6 m/s``, ``TE = 2.034 s``, ``n_sub = 98`` per save (both
        rules bind, 1000x under the 100,000 cap) at ``n_t = 1,000``: ``TE / tau = 10.0``
        exactly, total walker-steps ``5,000 * 1,000 * 98 ~= 4.90e8`` -- measured at ~3.5 minutes
        wall clock on this machine (CPU), inside a 10-minute bound. N drops from N_WALKERS
        (100,000) to N_EXACT (5,000, this module's reduced ordinal count) because N_WALKERS at
        this budget would take ~20x as long for the same statistical qualitative check.
    """
    TE    = 2.0337916666666667   # s -- derived above: B A**2 / (216 D N) at B = 97,622
    n_t   = 1000
    b_idx = 1   # b = 500 s/mm²
    wf    = pgse_wf(TE, n_t=n_t, slew_rate=np.inf)

    geom_perm = Ellipsoid(semiaxes=SEMIAXES, permeability=KAPPA_HIGH)
    S_perm    = simulate(N_EXACT, D, wf, geom_perm, seed=SEED)

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
    wf            = pgse_wf(100e-3)
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
    wf      = pgse_wf(100e-3)
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
