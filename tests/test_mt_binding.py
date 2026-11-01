"""MT binding in the forward vector-Bloch walk (Piece D of the MT staging ladder).

The impact-angle stick rule (p = min(1, 2 (kappa_MT/D) d_perp), consuming the same
boundary-local-time channel as surface relaxivity) is fused into the forward engine:
a free spin sticks at a wall, freezes for an exponential dwell, and relaxes with the
bound pool (T2_bound, T1_bound) blended by occupancy.  Validated against the
two-region exchange law and the two-pool Bloch--McConnell oracle.

"""

from dataclasses import replace
from dmipy_sim.engine.bloch import _simulate_bloch
import numpy as np
from dmipy_sim import ScannerSequence
from dmipy_sim import RFEvent
import pytest

from dmipy_sim import Sphere
from dmipy_sim.engine import mt

from tests.conftest import se_tol

EXC = [RFEvent(0.0, 90.0, axis_deg=90.0, duration_s=0.0, offset_hz=0.0)]


def _zero_wf(n_t, dt):
    return ScannerSequence(G=np.zeros((1, n_t, 3), dtype=np.float64), dt=dt)


# ── emergent exchange law: f_b = k_f/(k_f+k_r) on a sphere (S/V = 3/R) ───────────
@pytest.mark.parametrize("k_f,k_r", [(50.0, 100.0), (30.0, 150.0), (80.0, 80.0)])
def test_emergent_bound_fraction(k_f, k_r):
    R, D = 5e-6, 1e-9
    kappa_MT, dwell = k_f * R / 3.0, 1.0 / k_r
    dt, T_max = 1e-3, 0.04
    n_t = int(round(T_max / dt)) + 1
    _, bfrac = _simulate_bloch(8000, D, replace(_zero_wf(n_t, dt), rf=EXC), Sphere(radius=R), kappa_MT=kappa_MT, dwell_time=dwell, seed=3, return_bound_frac=True)
    f_b = float(np.mean(bfrac[-15:]))                 # equilibrium: the last 15 ms, from 4.5 relaxation times 1/(k_f+k_r) on
    assert f_b == pytest.approx(k_f / (k_f + k_r), rel=0.08)


# The late-time bound fraction (the mean of the last 20 ms) has the relative spread 0.8 per walker: its time samples are
# binomial (sd 0.0122 at 1,500 walkers, measured 0.0136) and correlated over 1/(k_f + k_r) = 6.7 ms, so the 20 ms are
# about three independent ones. The binding probability per hit is not small at 150 sub-steps (0.03), so the coarser
# run reads below the finer by the first-order effect of the step: -1.9 % and -5.1 % on two seeds of 1,500 walkers,
# 3 % allowed. Two runs agree to ``BIND_BIAS + se_tol(N, SD_BOUND_REL, runs=2)``, 9 % at 3,900 walkers: a 15 %
# difference fails with the margin the noise passes it.
BIND_BIAS = 0.03
SD_BOUND_REL = 0.8
N_BIND = 3_900


@pytest.mark.gpu
def test_binding_timestep_independence():
    R, D, k_f, k_r = 5e-6, 1e-9, 50.0, 100.0
    kappa_MT, dwell = k_f * R / 3.0, 1.0 / k_r
    dt, T_max = 1e-3, 0.06
    n_t = int(round(T_max / dt)) + 1

    def run(nsub):
        _, bf = _simulate_bloch(N_BIND, D, replace(_zero_wf(n_t, dt), rf=EXC), Sphere(radius=R), kappa_MT=kappa_MT, dwell_time=dwell, seed=3, sub_steps=nsub, return_bound_frac=True)
        return float(np.mean(bf[-20:]))

    assert run(150) == pytest.approx(run(400), rel=BIND_BIAS + se_tol(N_BIND, SD_BOUND_REL, runs=2))   # timestep-independent


# ── transverse exchange vs the two-pool Bloch--McConnell oracle (well-mixed) ─────
# The oracle is the well-mixed limit; the sphere mixes in R^2/D = 2 ms against 1/k_f = 33 ms, and the walk sits below
# it by that finite mixing: -3.2 % on average over three seeds of 1,500 walkers (-5.4, -3.0, -2.9 %). The walkers'
# relative spread about it is 0.50 per walker (a seed-to-seed spread of 1.3 % at 1,500). The match is therefore held
# to the finite-mixing offset plus the noise: ``MIXING_OFFSET + se_tol(N, SD_REL)``, 7 % at the 2,200 walkers that
# make the noise as large as the offset (z = 3.29), where 12,000 held 7 % with the noise a third of it.
MIXING_OFFSET = 0.035
SD_REL = 0.50
N_EXCHANGE = 2_200


@pytest.mark.gpu
def test_transverse_exchange_matches_oracle():
    R, D = 2e-6, 2e-9                 # mixing R^2/D = 2 ms << 1/k_f = 33 ms
    T2a, T2b, k_f, k_r = 80e-3, 1e-5, 30.0, 100.0
    kappa_MT, dwell = k_f * R / 3.0, 1.0 / k_r
    dt, T_max = 1e-3, 0.03
    n_t = int(round(T_max / dt)) + 1
    echo = list(range(1, n_t))
    geom = Sphere(radius=R)

    # equilibrate_binding='off': the oracle excites the FREE pool only (s0=[1,0,0,...]),
    # which the all-free start matches.  With the default pre-equilibrated bound pool, the
    # 90 would excite-and-kill (~10 us T2b) the bound spins too, lowering the initial free
    # transverse below the oracle's -- so this exchange-dynamics check pins the all-free IC.
    S_mt = np.abs(_simulate_bloch(N_EXCHANGE, D, replace(_zero_wf(n_t, dt), rf=EXC, readout=echo), geom, T2=T2a, kappa_MT=kappa_MT, dwell_time=dwell, T2_bound=T2b, T1_bound=1.0, seed=5, equilibrate_binding='off')[0])
    S0 = np.abs(_simulate_bloch(50, D, replace(_zero_wf(n_t, dt), rf=EXC, readout=echo), geom, T2=T2a, seed=5)[0])       # no MT -> plain T2a: no gradient, so every walker decays alike and the 50 are exact
    ratio_mc = S_mt / S0

    A = mt.two_pool_generator(R1a=0.1, R2a=1 / T2a, R1b=1.0, R2b=1 / T2b, k_f=k_f, k_r=k_r)
    s0 = np.array([1., 0, 0, 0, 0, 0, 1.])            # only the free pool is excited
    t = np.array(echo) * dt
    S_or = np.array([abs(complex(*mt.evolve_two_pool(s0, tt, A)[[0, 1]])) for tt in t])
    ratio_or = S_or / np.exp(-t / T2a)

    m = t > 0.006                                     # after initial mixing
    rel = np.abs(ratio_mc[m] - ratio_or[m]) / ratio_or[m]
    tol = MIXING_OFFSET + se_tol(N_EXCHANGE, SD_REL)
    assert np.max(rel) < tol, f"max rel diff {np.max(rel):.3f} (tolerance {tol:.3f})"
