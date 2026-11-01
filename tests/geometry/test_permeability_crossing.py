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

from dmipy_sim.engine._fused import simulate_fused
from dmipy_sim import simulate, Cylinder, set_b
from dmipy_sim.sequences import pgse

from tests.conftest import D, N_EXACT, SEED, n_resolving, se_tol

# Every test in this module makes an ORDINAL or IDENTITY claim -- monotone in kappa, A < B,
# "approaches free diffusion", or an exact `assert_array_equal` -- none of which needs the
# absolute accuracy that N_WALKERS = 100,000 buys. An ordinal claim "A < B" is held at its own
# noise: B must sit below A by `se_tol`, the z = 3.29 standard errors of their difference, and the
# walker count is the smallest at which the effect that was measured clears that margin.
#
# Measured (1,500 walkers, 4 cores):
#     b = 2000 s/mm^2: S_imp = 0.926, S_perm(kappa = 1e-5) = 0.654   -> a gap of 0.272
#     b = 0, kappa = 1e-5: S = 1 without relaxivity, 0.006 with RHO   -> a gap of 0.994
# The walker spread of a signal in [0, 1] is at most SD_COS = 0.707 (a bound; 0.25 at S = 0.93, 0.5 at S = 0.65), so
# the gap of two walks of N walkers has the margin ``se_tol(N, 0.5, runs=2)``: 0.10 at N = 500, a third of the gap.
# The relaxivity claim is a gap of 0.99 against weights that cannot spread beyond 0.5: 100 walkers.
#
# NOT applied to the modules that compare against an ANALYTIC reference to 1-2% (test_cylinder,
# test_free_1d, test_box_1d, ...): those genuinely need ~90,000 walkers at 3 sigma. See #93.
N_ORDINAL = 500
N_RELAXIVITY = 100
SD_ORDINAL = 0.5

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
    # exact-equality assertion: scale-free, see N_EXACT in conftest
    S_default = simulate(N_EXACT, D, wf, geom_default, seed=SEED)
    S_none    = simulate(N_EXACT, D, wf, geom_none,    seed=SEED)
    npt.assert_array_equal(S_default, S_none,
        err_msg="permeability=None must give identical signal to default")


_WALKS = {}


def _signal(kappa, rho=None, n=N_ORDINAL):
    """The signal of one permeable walk at ``n`` walkers, computed once per (kappa, rho, n) and shared by the tests
    below (the reduces and monotone tests read the kappa = None and kappa = KAPPA_MED walks)."""
    if (kappa, rho, n) not in _WALKS:
        geom = Cylinder(radius=R, orientation=[0, 0, 1], permeability=kappa, surface_relaxivity_t2=rho)
        _WALKS[(kappa, rho, n)] = np.asarray(simulate(n, D, pgse_wf(100e-3), geom, seed=SEED))
    return _WALKS[(kappa, rho, n)]


# ---------------------------------------------------------------------------
# 3. κ>0 increases signal vs impermeable at high b
# ---------------------------------------------------------------------------

def test_permeability_reduces_signal_at_high_b():
    """Signal with kappa > 0 is strictly below the impermeable signal at b = 2000 s/mm^2: an escaped walker diffuses
    freely and accumulates more phase than a restricted one."""
    S_imp, S_perm = _signal(None), _signal(KAPPA_MED)
    margin = se_tol(N_ORDINAL, SD_ORDINAL, runs=2)
    assert float(S_perm[3]) < float(S_imp[3]) - margin, (
        f"S_perm={S_perm[3]:.4f} must be below S_imp={S_imp[3]:.4f} by {margin:.3f} at b=2000 s/mm²")


# ---------------------------------------------------------------------------
# 4. High κ (TE >> τ): signal approaches free diffusion
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_permeability_high_kappa_approaches_free_diffusion():
    """Very high κ (TE/τ >> 1): signal within 10% of exp(-bD) at b=500 s/mm².

    Claim: at TE / tau = 10 the exchange is complete and the signal is the free one; at TE / tau = 6 it was not (a
    21 % relative error at kappa = 1.5e-5, TE = 1 s). Statistic: the walker mean of cos(phase) at b = 500 s/mm^2, whose
    per-walker standard deviation for free diffusion is ``sqrt((1 + S^4) / 2 - S^2) = 0.61`` at ``S = exp(-1)``.
    The 21 % error is resolved, i.e. the tolerance is half of it (10.5 %, z = 3.29), by
    ``n_resolving(0.21 S, 0.61)`` = 2,706 walkers.

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
        match); there ``tau = R / (2 kappa)`` is the cylinder's exchange time, ``TE = K tau`` for
        the exchange ratio ``K = TE / tau``, and the sub-steps per walker are ``36 K / C``.
      * ``K = 10`` (near-complete exchange) at R = 5e-6 m, D = 2e-9 m^2/s, C = 3e-3 gives
        ``kappa = 3.6e-6 m/s``, ``TE = 6.944 s``, ``n_sub = 101`` per save (both rules bind,
        1000x under the 100,000 cap) at ``n_t = 1,200``: 1.2e5 sub-steps per walker, so
        3.3e8 walker-steps for the walkers above.
    """
    TE = 6.944444444444445   # s -- K R / (2 kappa*) for K = 10
    n_t = 1200
    b_idx = 1  # b = 500 s/mm² = 500e6 s/m²
    wf = pgse_wf(TE, n_t=n_t, slew_rate=np.inf)

    # free-diffusion reference at b=500 s/mm²
    b_val = 500e6  # s/m²
    S_free = np.exp(-b_val * D)
    sd_free = np.sqrt((1.0 + S_free ** 4) / 2.0 - S_free ** 2)
    n = n_resolving(0.21 * S_free, sd_free)

    geom_perm = Cylinder(radius=R, orientation=[0, 0, 1], permeability=KAPPA_HIGH)
    S_perm = simulate_fused(n, D, wf, geom_perm, seed=SEED)

    tol = se_tol(n, sd_free)
    assert abs(float(S_perm[b_idx]) - S_free) < tol, (
        f"High-κ signal {S_perm[b_idx]:.4f} should be within {tol:.4f} ({tol / S_free:.1%}) of "
        f"free diffusion {S_free:.4f} (N = {n})")


# ---------------------------------------------------------------------------
# 5. Permeability + relaxivity: signal below permeability-only
# ---------------------------------------------------------------------------

def test_permeability_with_relaxivity_reduces_signal():
    """Surface relaxivity on a permeable cylinder reduces the signal at b = 0 (reflected walkers take the
    Brownstein-Tarr weight, transmitted ones do not)."""
    S_perm, S_perm_rho = _signal(KAPPA_MED, n=N_RELAXIVITY), _signal(KAPPA_MED, RHO, n=N_RELAXIVITY)
    margin = se_tol(N_RELAXIVITY, SD_ORDINAL, runs=2)
    assert float(S_perm_rho[0]) < float(S_perm[0]) - margin, (
        f"Relaxivity+permeability {S_perm_rho[0]:.4f} must be below "
        f"permeability-only {S_perm[0]:.4f} by {margin:.3f} at b=0")


# ---------------------------------------------------------------------------
# 6. Signal monotonically increases with κ at high b
# ---------------------------------------------------------------------------

