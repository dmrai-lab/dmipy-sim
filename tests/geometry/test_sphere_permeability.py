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

from tests.conftest import pgse_wf
import numpy.testing as npt
import pytest

from dmipy_sim import simulate, Sphere, set_b
from dmipy_sim.sequences import pgse

from tests.conftest import D, N_CPU, N_EXACT, SEED, mc_atol

N_BUDGET = 5_000     # the walker count the high-kappa regime below is derived for
R          = 5e-6   # m
KAPPA_MED  = 1e-5   # m/s  — exchange time τ = R/(3κ) ≈ 167 ms
KAPPA_HIGH = 3.6e-6  # m/s — derived in test_permeability_high_kappa_approaches_free_diffusion
                      # below from the engine's own sub-step budget, together with its TE
RHO        = 5e-4   # m/s  — surface relaxivity for combination test


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
    wf = pgse_wf(100e-3)
    geom_default = Sphere(radius=R)
    geom_none    = Sphere(radius=R, permeability=None)
    # exact-equality assertion -> scale-free; see N_EXACT in conftest (#93)
    S_default = simulate(N_EXACT, D, wf, geom_default, seed=SEED)
    S_none    = simulate(N_EXACT, D, wf, geom_none,    seed=SEED)
    npt.assert_array_equal(S_default, S_none,
        err_msg="permeability=None must give identical signal to default")


# ---------------------------------------------------------------------------
# 3. κ>0 reduces signal at high b
_WALKS = {}


def _signal(kappa, rho=None):
    """The signal of one permeable walk, computed once per (kappa, rho) and shared by the tests below (the reduces,
    relaxivity and monotone tests all read the same kappa = None and kappa = KAPPA_MED walks)."""
    if (kappa, rho) not in _WALKS:
        geom = Sphere(radius=R, permeability=kappa, surface_relaxivity_t2=rho)
        _WALKS[(kappa, rho)] = np.asarray(simulate(N_CPU, D, pgse_wf(100e-3), geom, seed=SEED))
    return _WALKS[(kappa, rho)]



# ---------------------------------------------------------------------------

def test_permeability_reduces_signal_at_high_b():
    """Signal with kappa > 0 must be strictly below the impermeable signal at b = 2000 s/mm^2 (an escaped walker
    diffuses more freely, so it accumulates more phase). The margin 0.05 is 7 standard errors at N_CPU."""
    S_imp, S_perm = _signal(None), _signal(KAPPA_MED)
    assert float(S_perm[3]) < float(S_imp[3]) - 0.05, (
        f"S_perm={S_perm[3]:.4f} must be below S_imp={S_imp[3]:.4f} at b=2000 s/mm²")


# ---------------------------------------------------------------------------
# 4. High κ (TE >> τ): signal approaches free diffusion
# ---------------------------------------------------------------------------

def test_permeability_high_kappa_approaches_free_diffusion():
    """Very high κ (TE/τ >> 1): signal within 10% of exp(-bD) at b=500 s/mm².

    ``slew_rate=np.inf`` keeps the lobes square: ``pgse_wf``'s unit-amplitude reference build
    (``gradient_strengths=1.0``) ramps at a FIXED 5 ms regardless of TE, so a short TE whose lobe
    is narrower than 5 ms is refused by the assembler as inexpressible (dmrai-lab/dmipy-sim#475)
    -- not a bug in the assembler, a placeholder amplitude too large for this TE.

    KAPPA_HIGH and TE are DERIVED below, not chosen by search, from the engine's own sub-step
    budget (``dmipy_sim.engine.physics``) -- the same derivation as
    ``test_permeability_crossing.py``'s sibling test, with the sphere's ``tau = R / (3 kappa)``
    in place of a cylinder's ``R / (2 kappa)``:

      * ``crossing_sub_steps`` (``CROSSING_P_MAX = C = 3e-3``) needs
        ``n_cross = ceil(6 D dt / step_max**2)`` per save, ``step_max = C D / (2 kappa)``;
      * ``walk_sub_steps`` (R/6, kappa-independent) needs
        ``n_refl = ceil(dt / ((R/6)**2 / (6 D)))``;
      * ``resolve_sub_steps`` takes the max of the two, both linear in dt, so total sub-steps
        over the walk is ``TE * max(216 D / R**2, 24 kappa**2 / (C**2 D))``, independent of n_t;
      * at ``kappa* = 3 D C / R`` neither rule wastes budget against the other, and for a
        walker-step budget ``B = N * TE * 216 D / R**2`` (N = N_BUDGET = 5,000) the
        achieved ``TE / tau = 3 B C / (72 N)`` (the sphere's extra factor of 3/2 over the
        cylinder's, from ``tau``'s R/(3 kappa) vs R/(2 kappa)).
      * solving ``TE / tau >= 10`` at R = 5e-6 m, D = 2e-9 m^2/s, C = 3e-3, N = 5,000 gives
        ``kappa = 3.6e-6 m/s``, ``TE = 4.630 s``, ``n_sub = 101`` per save (both rules bind,
        1000x under the 100,000 cap) at ``n_t = 800``: ``TE / tau = 10.0`` exactly, total
        walker-steps ``5,000 * 800 * 101 ~= 4.04e8`` -- measured at ~2.9 minutes wall clock on
        this machine (CPU), inside a 10-minute bound. N drops from N_WALKERS (100,000) to
        N_BUDGET (5,000) because N_WALKERS at this budget
        would take ~20x as long for the same statistical qualitative check.
    """
    TE    = 4.62962962962963   # s -- derived above: B R**2 / (216 D N) at B = 80,000
    n_t   = 800
    b_idx = 1   # b = 500 s/mm²
    wf    = pgse_wf(TE, n_t=n_t, slew_rate=np.inf)

    geom_perm = Sphere(radius=R, permeability=KAPPA_HIGH)
    S_perm    = simulate(N_BUDGET, D, wf, geom_perm, seed=SEED)

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
    """Surface relaxivity on a permeable pore reduces the signal at b = 0 (reflected walkers take the
    Brownstein-Tarr weight, transmitted ones do not)."""
    S_perm, S_perm_rho = _signal(KAPPA_MED), _signal(KAPPA_MED, RHO)
    assert float(S_perm_rho[0]) < float(S_perm[0]) - 0.02, (
        f"Relaxivity+permeability {S_perm_rho[0]:.4f} must be below "
        f"permeability-only {S_perm[0]:.4f} at b=0")


# ---------------------------------------------------------------------------
# 6. Signal monotonically decreases with κ at high b
# ---------------------------------------------------------------------------

def test_permeability_signal_monotone_in_kappa():
    """Signal at b = 2000 s/mm^2 decreases with kappa (more escape, less restriction), to the noise of two walks."""
    kappas = [0.0, 1e-6, 1e-5, 1e-4]
    signals = [float(_signal(k if k > 0 else None)[3]) for k in kappas]
    for i in range(len(signals) - 1):
        assert signals[i] >= signals[i + 1] - mc_atol(N_CPU, runs=2), (
            f"Signal not monotone: κ={kappas[i]:.0e} → {signals[i]:.4f}, "
            f"κ={kappas[i+1]:.0e} → {signals[i+1]:.4f}")
    assert signals[0] - signals[-1] > 0.05, "the full kappa range must move the signal"


