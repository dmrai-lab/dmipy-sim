"""Membrane permeability for Cylinder (Powles 2004).

Physics
-------
At each wall crossing the walker transmits through the membrane with
probability

    p = min(1,  2·κ·d_perp / D)

and reflects otherwise (Powles et al. 2004; same formula as disimpy).
The exchange time for walkers initially inside a cylinder of radius R is

    τ = R / (2κ)   (V/κS = πR²L / (κ·2πRL) = R/(2κ))

Analytical limits validated here
----------------------------------
1. κ=None  (permeability not set) → identical signal to impermeable Cylinder.
2. κ=0  (zero permeability)       → identical signal to impermeable Cylinder.
3. κ>0 at high b                  → signal strictly above impermeable (walkers
                                    that escape are less restricted → higher S).
4. Very high κ (TE >> τ)          → walkers mostly escaped, signal close to
                                    free diffusion exp(−bD).  Tested at
                                    b = 500 s/mm² where the gap between
                                    restricted and free is large enough to give
                                    a clear bound.

Combination with surface relaxivity
-------------------------------------
5. Permeability + relaxivity simultaneously: signal must be below the
   permeability-only signal (relaxivity reduces signal on reflection).

Parameters chosen so σ/R < 0.1 (good-practice criterion):
    R = 5 µm,  D = 2e-9 m²/s,  n_t = 500,  TE ≈ 100 ms
    σ = √(6D·dt) ≈ √(6·2e-9·0.2e-3) ≈ 1.55 µm → σ/R ≈ 0.31

For the high-κ test we use a short TE (20 ms) with fine time-stepping:
    τ = R/(2κ) = 5e-6/(2·1e-2) = 0.25 ms
    TE/τ ≈ 80  → ≈ 100% of walkers have had exchange opportunity
"""

import numpy as np

from tests.conftest import pgse_wf
import numpy.testing as npt
import pytest

from dmipy_sim import simulate, Cylinder, set_b
from dmipy_sim.sequences import pgse

from tests.conftest import D, N_WALKERS, SEED

# Every test in this module makes an ORDINAL or IDENTITY claim -- monotone in kappa, A < B,
# "approaches free diffusion", or an exact `assert_array_equal` -- none of which needs the
# absolute accuracy that N_WALKERS = 100,000 buys. All comparisons share `seed=SEED`, so the
# walks are correlated and the DIFFERENCES carry far less noise than 0.32% per signal.
#
# Measured on the monotonicity sweep, which is the tightest claim here:
#     N=100,000  62.9 s  monotone, min gap between adjacent signals +3.04e-02
#     N= 20,000  46.1 s  monotone, min gap                          +2.89e-02
#     N=  5,000  42.3 s  monotone, min gap                          +3.00e-02
# The gap deciding the ordering is 30x the 1e-3 slack the assertion allows, and it is stable
# at every N -- so the ordering is not what the walkers were buying.
#
# NOT applied to the modules that compare against an ANALYTIC reference to 1-2% (test_cylinder,
# test_free_1d, test_box_1d, ...): those genuinely need ~90,000 walkers at 3 sigma. See #93.
N_ORDINAL = 5_000

R   = 5e-6   # m
KAPPA_MED  = 1e-5  # m/s  — exchange time τ = R/(2κ) ≈ 250 ms
KAPPA_HIGH = 3.6e-6  # m/s — derived in test_permeability_high_kappa_approaches_free_diffusion
                      # below from the engine's own sub-step budget, together with its TE
RHO        = 5e-4  # m/s  — surface relaxivity for combination test


# ---------------------------------------------------------------------------
# 1. Attribute storage
# ---------------------------------------------------------------------------

def test_permeability_attribute_stored():
    """permeability is stored correctly on Cylinder."""
    geom = Cylinder(radius=R, orientation=[0, 0, 1], permeability=KAPPA_MED)
    assert geom.permeability == KAPPA_MED


def test_permeability_none_not_set():
    """Default Cylinder has permeability=None."""
    geom = Cylinder(radius=R, orientation=[0, 0, 1])
    assert geom.permeability is None


# ---------------------------------------------------------------------------
# 2. κ=None / κ=0 must reproduce the impermeable signal
# ---------------------------------------------------------------------------

def test_permeability_none_matches_impermeable():
    """Cylinder(permeability=None) == Cylinder() — identical signal."""
    wf = pgse_wf(100e-3)
    geom_default = Cylinder(radius=R, orientation=[0, 0, 1])
    geom_none    = Cylinder(radius=R, orientation=[0, 0, 1], permeability=None)
    S_default = simulate(N_ORDINAL, D, wf, geom_default, seed=SEED)
    S_none    = simulate(N_ORDINAL, D, wf, geom_none,    seed=SEED)
    npt.assert_array_equal(S_default, S_none,
        err_msg="permeability=None must give identical signal to default")


# ---------------------------------------------------------------------------
# 3. κ>0 increases signal vs impermeable at high b
# ---------------------------------------------------------------------------

def test_permeability_reduces_signal_at_high_b():
    """Signal with κ>0 must be strictly below impermeable signal at b=2000 s/mm².

    Inside a cylinder, restriction raises the signal (ADC_app << D).
    Walkers that escape diffuse freely → larger phase accumulation → lower
    signal.  So: S_perm < S_imp at high b.
    """
    wf = pgse_wf(100e-3)
    geom_imp  = Cylinder(radius=R, orientation=[0, 0, 1])
    geom_perm = Cylinder(radius=R, orientation=[0, 0, 1], permeability=KAPPA_MED)
    S_imp  = simulate(N_ORDINAL, D, wf, geom_imp,  seed=SEED)
    S_perm = simulate(N_ORDINAL, D, wf, geom_perm, seed=SEED)
    # b=2000 s/mm² is index 3 in our b_values array
    assert float(S_perm[3]) < float(S_imp[3]) - 0.05, (
        f"S_perm={S_perm[3]:.4f} must be below S_imp={S_imp[3]:.4f} at b=2000 s/mm²")


# ---------------------------------------------------------------------------
# 4. High κ (TE >> τ): signal approaches free diffusion
# ---------------------------------------------------------------------------

def test_permeability_high_kappa_approaches_free_diffusion():
    """Very high κ (TE/τ >> 1): signal within 10% of exp(-bD) at b=500 s/mm².

    ``slew_rate=np.inf`` keeps the lobes square: ``pgse_wf``'s unit-amplitude reference build
    (``gradient_strengths=1.0``) ramps at a FIXED 5 ms (``1.0 / DEFAULT_SLEW_RATE``) regardless of
    TE, so a short TE whose lobe is narrower than 5 ms is refused by the assembler as inexpressible
    (dmrai-lab/dmipy-sim#475) -- not a bug in the assembler, a placeholder amplitude too large for
    this TE.

    KAPPA_HIGH and TE are DERIVED below, not chosen by search, from the engine's own sub-step
    budget (``dmipy_sim.engine.physics``), because the obvious choice -- the original κ=1e-2 m/s
    at TE=20 ms -- asks ``resolve_sub_steps`` for 1.33e6 sub-steps per save (13x its 100,000 cap):
    at the save grid this TE needs, a step small enough to keep the crossing rule's per-hit
    probability near first order is a few e-10 m, no matter how n_t is chosen.

      * the crossing rule (``crossing_sub_steps``, ``CROSSING_P_MAX = C = 3e-3``) bounds the
        per-hit transmission probability: the largest step is
        ``step_max = C D / (2 kappa)``, so a save of ``dt`` needs
        ``n_cross = ceil(6 D dt / step_max**2)`` sub-steps.
      * the reflection rule (``walk_sub_steps``, R/6) needs
        ``n_refl = ceil(dt / ((R/6)**2 / (6 D)))``, independent of kappa.
      * ``resolve_sub_steps`` takes ``max(n_cross, n_refl)``; both are linear in dt, so the
        TOTAL sub-steps over the whole walk, ``n_t * n_sub``, is
        ``TE * max(216 D / R**2, 24 kappa**2 / (C**2 D))`` -- independent of how finely the
        n_t grid chops it.
      * neither rule wastes budget against the other at ``kappa* = 3 D C / R`` (where the two
        match); there, for a walker-step budget ``B = N * TE * 216 D / R**2`` (N = 5,000, the
        file's N_ORDINAL), ``TE = B R**2 / (216 D N)`` and the achieved exchange ratio
        ``TE / tau`` (``tau = R / (2 kappa)`` for a cylinder) is ``2 B C / (72 N)``.
      * solving ``TE / tau >= 10`` (near-complete exchange -- empirically, TE/tau = 6 at
        κ=1.5e-5, TE=1 s left a 21% relative error, too far above this test's 10% tolerance) for
        B at R = 5e-6 m, D = 2e-9 m^2/s, C = 3e-3, N = 5,000 gives
        ``kappa = 3.6e-6 m/s``, ``TE = 6.944 s``, ``n_sub = 101`` per save (both rules bind,
        1000x under the 100,000 cap) at ``n_t = 1,200``: ``TE / tau = 10.0`` exactly, total
        walker-steps ``5,000 * 1,200 * 101 ~= 6.06e8`` -- measured at ~4.3 minutes wall clock on
        this machine (CPU), inside a 10-minute bound.
    """
    TE = 6.944444444444445   # s -- derived above: B(R**2)/(216 D N) at B = 120,000
    n_t = 1200
    b_idx = 1  # b = 500 s/mm² = 500e6 s/m²
    wf = pgse_wf(TE, n_t=n_t, slew_rate=np.inf)

    geom_perm = Cylinder(radius=R, orientation=[0, 0, 1], permeability=KAPPA_HIGH)
    S_perm = simulate(N_ORDINAL, D, wf, geom_perm, seed=SEED)

    # free-diffusion reference at b=500 s/mm²
    b_val = 500e6  # s/m²
    S_free = np.exp(-b_val * D)

    rel_err = abs(float(S_perm[b_idx]) - S_free) / S_free
    assert rel_err < 0.10, (
        f"High-κ signal {S_perm[b_idx]:.4f} should be within 10% of "
        f"free diffusion {S_free:.4f} (rel_err={rel_err:.3f})")


# ---------------------------------------------------------------------------
# 5. Permeability + relaxivity: signal below permeability-only
# ---------------------------------------------------------------------------

def test_permeability_with_relaxivity_reduces_signal():
    """Adding surface relaxivity to a permeable cylinder must reduce signal.

    Reflected walkers receive the Brownstein-Tarr weight; transmitted walkers
    do not.  The ensemble signal must therefore be ≤ the permeability-only
    signal.
    """
    wf = pgse_wf(100e-3)
    geom_perm     = Cylinder(radius=R, orientation=[0, 0, 1],
                             permeability=KAPPA_MED)
    geom_perm_rho = Cylinder(radius=R, orientation=[0, 0, 1],
                             permeability=KAPPA_MED,
                             surface_relaxivity_t2=RHO)
    S_perm     = simulate(N_ORDINAL, D, wf, geom_perm,     seed=SEED)
    S_perm_rho = simulate(N_ORDINAL, D, wf, geom_perm_rho, seed=SEED)
    # Check at b=0 (only relaxivity reduces signal; diffusion doesn't matter)
    assert float(S_perm_rho[0]) < float(S_perm[0]) - 0.02, (
        f"Relaxivity+permeability {S_perm_rho[0]:.4f} must be below "
        f"permeability-only {S_perm[0]:.4f} at b=0")


# ---------------------------------------------------------------------------
# 6. Signal monotonically increases with κ at high b
# ---------------------------------------------------------------------------

def test_permeability_signal_monotone_in_kappa():
    """Signal at b=2000 s/mm² decreases monotonically with κ.

    Higher κ → more walkers escape → less restriction → faster decay → lower
    signal at high b.
    """
    wf = pgse_wf(100e-3)
    kappas = [0.0, 1e-6, 1e-5, 1e-4]
    signals = []
    for kappa in kappas:
        perm = kappa if kappa > 0 else None
        geom = Cylinder(radius=R, orientation=[0, 0, 1], permeability=perm)
        S = simulate(N_ORDINAL, D, wf, geom, seed=SEED)
        signals.append(float(S[3]))  # b=2000 s/mm²
    for i in range(len(signals) - 1):
        assert signals[i] >= signals[i + 1] - 1e-3, (
            f"Signal not monotone: κ={kappas[i]:.0e} → {signals[i]:.4f}, "
            f"κ={kappas[i+1]:.0e} → {signals[i+1]:.4f}")
