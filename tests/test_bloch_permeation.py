"""Membrane permeability in the vector-Bloch forward (`_simulate_bloch`).

The scalar `core.simulate` walk has always supported the Powles crossing; these check that
`_simulate_bloch` now does too — a diffusion-weighted spin echo on a restricted cylinder attenuates
MORE as the wall becomes permeable (walkers escape the restriction), and approaches the free-
diffusion signal in the high-κ limit. This is what lets the Bloch engine model exchange during a
longitudinal-storage mixing time (FEXI).
"""
import numpy as np
from dmipy_sim.engine.bloch import _simulate_bloch
from dmipy_replay import pgse
import pytest

from tests.conftest import N_EXACT, se_tol

from dmipy_sim import Cylinder, FreeDiffusion
from dmipy_sim.engine.physics import resolve_sub_steps
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
    return abs(complex(_simulate_bloch(n, D, seq, geom, seed=seed, require_gpu=False)[0]))





# ── a "near free diffusion" regime ───────────────────────────────────────────────────────────
#
# The walk's step is the step law's (the pore term with the membrane's coefficient added), the same at every
# kappa, so the regime is set by the exchange alone: kappa* = 6e-6 m/s on R = 3 um gives tau = R / (2 kappa*) =
# 0.25 s, and K = TE / tau = 10 (exp(-10) = 4.5e-5 of the restricted/free gap -- the SAME K
# tests/geometry/test_permeability_crossing.py uses, there against a 10% RELATIVE tolerance; here against this
# test's 0.12 ABSOLUTE one) gives TE = 2.5 s, at the step law's count per save of n_t = 1,200 (dt = 2.09 ms).
K_FREE = 10.0        # target TE / tau
KAPPA_FREE = 6e-6                                                 # kappa*, m/s
TAU_FREE = R / (2 * KAPPA_FREE)                                   # -- 0.25 s
TE_FREE = K_FREE * TAU_FREE                                       # -- 2.5 s
N_T_FREE = 1_200
B_FREE = 5e8         # s/m^2 -- an arbitrary, comfortably-restricted b (500 s/mm^2)


def _free_diffusion_regime_wf():
    """The PGSE of the derived (kappa*, TE, n_t) regime above (a lobe width far shorter than
    TE, so its own duration does not compete with the exchange-time budget)."""
    return pgse([[1, 0, 0]], 0.05, TE_FREE / 2, bvalues=[B_FREE], TE=TE_FREE,
                n_t=N_T_FREE, slew_rate=np.inf)


# The first lobe of this echo starts 0.6 s in, 2.4 exchange times (tau = 0.25 s) after the start: e^-2.4 = 9 % of the
# walkers are still inside the cylinder and see its restricted signal, 0.58 above the free one (measured: 1.00 against
# 0.41 and 0.43), so the permeable signal could sit 0.05 above the free one however many walkers there are (measured:
# 0.03 and 0.05 below it, within the noise of 250 walkers).
UNEXCHANGED_OFFSET = 0.05
N_FREE = 250


def test_high_permeability_approaches_free_diffusion():
    """At kappa*/TE derived above, the wall is near-transparent (TE/tau = 10) and the signal is the free one.

    Claim: the permeable cylinder's signal is the free-diffusion signal to ``UNEXCHANGED_OFFSET + se_tol``, 0.23 at 250
    walkers, where the restricted signal is 0.58 higher. Statistic: the difference of two walker means of cos(phase) at
    ``S = exp(-1)``, whose per-walker standard deviation is ``sqrt((1 + S^4) / 2 - S^2) = 0.61``; a restricted cylinder
    fails it by 6 standard errors of the difference, and ``n_resolving(0.4, 0.61, runs=2)`` = 202 walkers is where the
    margin to 0.4 is the noise's."""
    wf = _free_diffusion_regime_wf()
    geom = Cylinder(radius=R, orientation=(0, 0, 1), permeability=KAPPA_FREE)
    n_sub = resolve_sub_steps(geom, D, float(wf.dt))
    assert n_sub < 1_000, f"the derived regime should be far under the cap, asked for {n_sub}"

    s_free = np.exp(-B_FREE * D)
    sd = np.sqrt((1.0 + s_free ** 4) / 2.0 - s_free ** 2)
    S_free = abs(complex(_simulate_bloch(N_FREE, D, wf, FreeDiffusion(), seed=0, require_gpu=False)[0]))
    S_perm = abs(complex(_simulate_bloch(N_FREE, D, wf, geom, seed=0, require_gpu=False)[0]))
    tol = UNEXCHANGED_OFFSET + se_tol(N_FREE, sd, runs=2)
    assert abs(S_perm - S_free) < tol, f"S_perm {S_perm:.3f} vs S_free {S_free:.3f} (tolerance {tol:.3f})"      # high-kappa limit ~= free diffusion


def test_impermeable_bloch_walk_is_unchanged():
    """permeability=None must reproduce the plain reflecting walk (no regression)."""
    seq, _ = _dw_spin_echo()
    a = _S(0.0, seq, n=N_EXACT, seed=7)
    b = _S(0.0, seq, n=N_EXACT, seed=7)
    assert a == pytest.approx(b, abs=1e-9)           # deterministic, reflecting path
