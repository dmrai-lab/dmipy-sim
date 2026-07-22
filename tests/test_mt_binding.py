"""MT binding in the forward vector-Bloch walk (Piece D of the MT staging ladder).

The impact-angle stick rule (p = min(1, 2 (kappa_MT/D) d_perp), consuming the same
boundary-local-time channel as surface relaxivity) is fused into the forward engine:
a free spin sticks at a wall, freezes for an exponential dwell, and relaxes with the
bound pool (T2_bound, T1_bound) blended by occupancy.  Validated against the
two-region exchange law and the two-pool Bloch--McConnell oracle.

Heavy Monte-Carlo (fine binding sub-steps) -> GPU-recommended; marked slow.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from dmipy_sim import Sphere, simulate_bloch
from dmipy_sim import mt

pytestmark = pytest.mark.slow

EXC = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': 90.0, 'duration_s': 0.0, 'offset_hz': 0.0}]


def _zero_wf(n_t, dt):
    return SimpleNamespace(G=np.zeros((1, n_t, 3), dtype=np.float64), dt=dt)


# ── emergent exchange law: f_b = k_f/(k_f+k_r) on a sphere (S/V = 3/R) ───────────
@pytest.mark.parametrize("k_f,k_r", [(50.0, 100.0), (30.0, 150.0), (80.0, 80.0)])
def test_emergent_bound_fraction(k_f, k_r):
    R, D = 5e-6, 1e-9
    kappa_MT, dwell = k_f * R / 3.0, 1.0 / k_r
    dt, T_max = 1e-3, 0.06
    n_t = int(round(T_max / dt)) + 1
    _, bfrac = simulate_bloch(8000, D, _zero_wf(n_t, dt), Sphere(radius=R), EXC,
                              kappa_MT=kappa_MT, dwell_time=dwell, seed=3,
                              return_bound_frac=True)
    f_b = float(np.mean(bfrac[-20:]))                 # equilibrium (last 20 ms)
    assert f_b == pytest.approx(k_f / (k_f + k_r), rel=0.08)


def test_binding_timestep_independence():
    R, D, k_f, k_r = 5e-6, 1e-9, 50.0, 100.0
    kappa_MT, dwell = k_f * R / 3.0, 1.0 / k_r
    dt, T_max = 1e-3, 0.06
    n_t = int(round(T_max / dt)) + 1

    def run(nsub):
        _, bf = simulate_bloch(6000, D, _zero_wf(n_t, dt), Sphere(radius=R), EXC,
                               kappa_MT=kappa_MT, dwell_time=dwell, seed=3,
                               sub_steps=nsub, return_bound_frac=True)
        return float(np.mean(bf[-20:]))

    assert run(150) == pytest.approx(run(400), rel=0.06)   # timestep-independent


# ── transverse exchange vs the two-pool Bloch--McConnell oracle (well-mixed) ─────
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
    S_mt = np.abs(simulate_bloch(12000, D, _zero_wf(n_t, dt), geom, EXC, T2=T2a,
                                 kappa_MT=kappa_MT, dwell_time=dwell, T2_bound=T2b,
                                 T1_bound=1.0, seed=5, echo_steps=echo,
                                 equilibrate_binding='off')[0])
    S0 = np.abs(simulate_bloch(12000, D, _zero_wf(n_t, dt), geom, EXC, T2=T2a,
                               seed=5, echo_steps=echo)[0])       # no MT -> plain T2a
    ratio_mc = S_mt / S0

    A = mt.two_pool_generator(R1a=0.1, R2a=1 / T2a, R1b=1.0, R2b=1 / T2b, k_f=k_f, k_r=k_r)
    s0 = np.array([1., 0, 0, 0, 0, 0, 1.])            # only the free pool is excited
    t = np.array(echo) * dt
    S_or = np.array([abs(complex(*mt.evolve_two_pool(s0, tt, A)[[0, 1]])) for tt in t])
    ratio_or = S_or / np.exp(-t / T2a)

    m = t > 0.006                                     # after initial mixing
    rel = np.abs(ratio_mc[m] - ratio_or[m]) / ratio_or[m]
    assert np.max(rel) < 0.07, f"max rel diff {np.max(rel):.3f}"


def test_mt_attenuates_more_for_higher_kf():
    R, D = 2e-6, 2e-9
    T2a, T2b = 80e-3, 1e-5
    dt, T_max = 1e-3, 0.03
    n_t = int(round(T_max / dt)) + 1

    def last_echo(k_f):
        kappa_MT, dwell = k_f * R / 3.0, 1.0 / 100.0
        S = np.abs(simulate_bloch(8000, D, _zero_wf(n_t, dt), Sphere(radius=R), EXC,
                                  T2=T2a, kappa_MT=kappa_MT, dwell_time=dwell,
                                  T2_bound=T2b, seed=7, echo_steps=[n_t - 1])[0])
        return float(S[0])

    assert last_echo(60.0) < last_echo(15.0)          # more binding -> more signal loss
