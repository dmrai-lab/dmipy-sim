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
4. Very high κ (TE >> τ)          → the transmit probability saturates at one:
                                    every impact that crosses is granted
                                    (tests/geometry/test_permeable_crossings.py).
                                    The walk to complete exchange costs 3e8
                                    walker-sub-steps (the crossing rule keeps
                                    the probability per hit at 3e-3) and is not
                                    run.

Combination with surface relaxivity
-------------------------------------
5. Permeability + relaxivity simultaneously: signal must be below the
   permeability-only signal (relaxivity reduces signal on reflection).

Parameters chosen so σ/R < 0.1 (good-practice criterion):
    R = 5 µm,  D = 2e-9 m²/s,  n_t = 500,  TE ≈ 100 ms
    σ = √(6D·dt) ≈ √(6·2e-9·0.2e-3) ≈ 1.55 µm → σ/R ≈ 0.31
"""

import numpy as np

from tests.conftest import pgse_wf
import numpy.testing as npt
import pytest

from dmipy_sim import simulate, Cylinder
from dmipy_replay import set_b
from dmipy_replay.sequences import pgse

from tests.conftest import D, N_EXACT, SEED, se_tol

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

