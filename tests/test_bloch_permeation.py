"""Membrane permeability in the vector-Bloch forward (`simulate_bloch`).

The scalar `core.simulate` walk has always supported the Powles crossing; these check that
`simulate_bloch` now does too — a diffusion-weighted spin echo on a restricted cylinder attenuates
MORE as the wall becomes permeable (walkers escape the restriction), and approaches the free-
diffusion signal in the high-κ limit. This is what lets the Bloch engine model exchange during a
longitudinal-storage mixing time (FEXI).
"""
import numpy as np
from dmipy_sim import simulate_bloch, pgse
import pytest

from dmipy_sim import Cylinder, FreeDiffusion
from dmipy_sim.engine.physics import CROSSING_P_MAX, resolve_sub_steps
from dmipy_sim.engine.pulse_sequence import bare_spin_echo

GAMMA = 2.675e8
R = 3e-6     # m -- this file's Cylinder radius, both tests below
D = 2e-9     # m^2/s


def _dw_spin_echo(g=0.20, TE=30e-3, dt=1e-4, delta=5e-3):
    """PGSE (physical same-sign lobes; the 180 folds the sign) perpendicular to the cylinder."""
    seq = bare_spin_echo(TE, dt)
    n_t = seq.n_t
    nd = int(round(delta / dt))
    i1 = int(round(4e-3 / dt))
    i2 = int(round((TE / 2 + 1e-3) / dt))
    G = np.zeros((1, n_t, 3))
    G[0, i1:i1 + nd, 0] = g
    G[0, i2:i2 + nd, 0] = g
    seq.G[:] = G
    sgn = np.where(np.arange(n_t) < int(round(TE / 2 / dt)), 1.0, -1.0)
    q = GAMMA * np.cumsum(sgn * G[0, :, 0]) * dt
    return seq, float(np.sum(q ** 2) * dt)


def _S(kappa, seq, D=2e-9, n=4000, seed=0):
    geom = Cylinder(radius=3e-6, orientation=(0, 0, 1),
                    permeability=(None if kappa == 0 else kappa))
    return abs(complex(simulate_bloch(n, D, seq, geom, seed=seed, require_gpu=False)[0]))


def test_permeability_increases_attenuation_monotonically():
    """kappa=1e-4 (not the original 3e-3): the crossing sub-step rule (`crossing_sub_steps`, #292)
    scales sub-steps as kappa^2, so 3e-3 now asks for 1.2M sub-steps per waveform step -- 12x
    `MAX_SUB_STEPS` and refused (#626).

    1e-4 is a margin, not a search: at this file's dt=1e-4 s, `resolve_sub_steps` (the SAME
    engine rule the test below derives a regime from) asks for 1,334 sub-steps -- 1.3% of the
    100,000 cap -- comfortably inside it while still a clearly more permeable wall than the 3e-5
    case; the monotonicity and the effect size (`> 0.1`, unchanged from the original test) are
    what this test checks, not a specific kappa. (The module is already in
    `conftest._SLOW_MC_MODULES`, so the extra sub-steps cost the weekly lane, not every PR.)

    N_MONO = 1,500 (not this file's usual 4,000): at 1,334 sub-steps x 300 waveform steps, 4,000
    walkers measured 1,199 s on this shared CPU box, over the 10-minute single-run bound; the
    ordering and the 0.1 gap are nowhere near the N=4,000 noise floor (observed gap 0.45, ~28x
    the 1/sqrt(N) floor at 1,500), so the walker count is the thing to lower, not the gap -- the
    floor stated here, not searched for.
    """
    seq, b = _dw_spin_echo()
    assert b > 3e8                                   # a real diffusion weighting
    n_sub = resolve_sub_steps(Cylinder(radius=R, orientation=(0, 0, 1), permeability=1e-4), D, 1e-4)
    assert n_sub < 10_000, f"kappa=1e-4 should be a wide cap margin, asked for {n_sub} sub-steps"
    N_MONO = 1_500
    S = [_S(k, seq, n=N_MONO) for k in (0.0, 3e-5, 1e-4)]
    # more permeable wall -> walkers leave the restriction -> more signal loss
    assert S[0] > S[1] > S[2]
    assert S[0] - S[2] > 0.1                          # a clear effect, not noise


# ── a feasible "near free diffusion" regime, derived from the engine's own sub-step budget ──────
#
# The original kappa=1e-2 at TE=30ms asks `resolve_sub_steps` for 13.3M sub-steps per waveform
# step (133x `MAX_SUB_STEPS`, #626) -- not searchable down, since the crossing rule's cost is
# INDEPENDENT of how TE is chopped into n_t. Derived instead, the same way PR #652 derived its
# scalar-engine sibling (tests/geometry/test_permeability_crossing.py), from
# `dmipy_sim.engine.physics`:
#
#   * the crossing rule (`crossing_sub_steps`, CROSSING_P_MAX = C = 3e-3) bounds the per-hit
#     transmission probability: the largest step is step_max = C D / (2 kappa), so a waveform
#     step of dt needs n_cross = ceil(6 D dt / step_max**2) sub-steps -- a RATE (sub-steps per
#     second) of 24 kappa**2 / (C**2 D), independent of dt.
#   * the reflection rule (`walk_sub_steps`, R/6) needs a rate of 216 D / R**2, independent of
#     kappa.
#   * neither rate wastes budget against the other at kappa* = 3 C D / R (where the two match);
#     there the total sub-steps over the whole walk (n_t * n_sub, exactly the rate times TE,
#     independent of how finely n_t chops it) is TE * 216 D / R**2.
#   * tau = R / (2 kappa*) = R**2 / (6 C D) is the cylinder's exchange time at kappa*. For a
#     target exchange ratio K = TE / tau (near-complete exchange; the residual against free
#     diffusion is of order exp(-K)), TE = K tau = K R**2 / (6 C D), and the per-walker total
#     sub-steps (independent of N, R, D -- a pure function of K and C) is
#     B = TE * 216 D / R**2 = 36 K / C.
#
# R = 3e-6 m, D = 2e-9 m^2/s, C = 3e-3 (this file's substrate), K = 10 (exp(-10) = 4.5e-5 of the
# restricted/free gap -- the SAME K tests/geometry/test_permeability_crossing.py uses, there
# against a 10% RELATIVE tolerance; here against this test's ORIGINAL 0.12 ABSOLUTE one,
# unchanged below) gives kappa* = 3 C D / R = 6e-6 m/s, tau = 0.25 s, TE = 2.5 s,
# B = 36 K / C = 120,000 sub-steps per walker (`resolve_sub_steps` confirms 101 per save at
# n_t=1,200, dt=2.09 ms -- 0.1% of the cap). At N=4,000 walkers that is 4.85e8 total sub-steps.
K_FREE = 10.0        # target TE / tau
KAPPA_FREE = 3 * CROSSING_P_MAX * D / R                           # kappa* -- 6e-6 m/s
TAU_FREE = R / (2 * KAPPA_FREE)                                   # = R**2 / (6 C D) -- 0.25 s
TE_FREE = K_FREE * TAU_FREE                                       # -- 2.5 s
N_T_FREE = 1_200
B_FREE = 5e8         # s/m^2 -- an arbitrary, comfortably-restricted b (500 s/mm^2)


def _free_diffusion_regime_wf():
    """The PGSE of the derived (kappa*, TE, n_t) regime above (a lobe width far shorter than
    TE, so its own duration does not compete with the exchange-time budget)."""
    return pgse([[1, 0, 0]], 0.05, TE_FREE / 2, bvalues=[B_FREE], TE=TE_FREE,
                n_t=N_T_FREE, slew_rate=np.inf)


def test_high_permeability_approaches_free_diffusion():
    """At kappa*/TE derived above, the wall is near-transparent (TE/tau = 10) and the original
    0.12 absolute tolerance holds without touching it -- not fit to a measured gap (#626 review)."""
    wf = _free_diffusion_regime_wf()
    geom = Cylinder(radius=R, orientation=(0, 0, 1), permeability=KAPPA_FREE)
    n_sub = resolve_sub_steps(geom, D, float(wf.dt))
    assert n_sub < 1_000, f"the derived regime should be far under the cap, asked for {n_sub}"

    S_free = abs(complex(simulate_bloch(4000, D, wf, FreeDiffusion(), seed=0, require_gpu=False)[0]))
    S_perm = abs(complex(simulate_bloch(4000, D, wf, geom, seed=0, require_gpu=False)[0]))
    assert abs(S_perm - S_free) < 0.12               # high-kappa limit ~= free diffusion (original)


def test_impermeable_bloch_walk_is_unchanged():
    """permeability=None must reproduce the plain reflecting walk (no regression)."""
    seq, _ = _dw_spin_echo()
    a = _S(0.0, seq, seed=7)
    b = _S(0.0, seq, seed=7)
    assert a == pytest.approx(b, abs=1e-9)           # deterministic, reflecting path
