"""MT is active during LONGITUDINAL storage (PGSTE mixing time) -- sim coverage.

The forward vector-Bloch engine carries M=(Mx,My,Mz) and applies T1 (blended toward
T1_bound by binding occupancy) every step, and the free<->bound exchange is emergent
(walkers carry their Mz across bind/release).  So magnetization stored along z -- a
PGSTE mixing time, or the residual Mz of an imperfect pulse -- already experiences MT,
with NO engine change.  Here we invert both pools (180), let them recover during a
pure longitudinal storage delay (no RF, no gradient), and check the ensemble Mz(t)
against the two-pool Bloch-McConnell longitudinal oracle.

This is why MT is the residual confound for PGSTE permeability: susceptibility and
surface relaxivity are transverse-gated (off during storage), MT is not.

Heavy Monte-Carlo -> GPU-recommended; marked slow.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from dmipy_sim import Sphere, simulate_bloch
from dmipy_sim import mt

pytestmark = pytest.mark.slow

R, D = 5e-6, 1e-9
k_f, k_r = 30.0, 100.0                       # -> f_b = k_f/(k_f+k_r) = 0.23
KAPPA_MT, DWELL = k_f * R / 3.0, 1.0 / k_r   # sphere S/V = 3/R
T1A, T1B = 1.0, 0.4
INVERT = [{'t_s': 0.0, 'flip_deg': 180.0, 'axis_deg': 0.0, 'duration_s': 0.0, 'offset_hz': 0.0}]


def _stored_mz(TM, *, with_mt):
    dt = 1e-3
    n_t = int(round(TM / dt)) + 1
    wf = SimpleNamespace(G=np.zeros((1, n_t, 3)), dt=dt)
    kw = dict(T2=80e-3, T1=T1A, M0=1.0, return_mz=True, seed=4)
    if with_mt:
        # equilibrate_binding='off': this coverage test's oracle idealises a CLEAN 180
        # inversion of BOTH pools (s0 = -M0a, -M0b).  Physically a hard pulse cannot invert
        # the broad (10 us-T2b) bound pool -- its nutation dephases mid-pulse -- so the
        # idealised oracle only matches when no bound pool exists at the inversion and the
        # bound pool is then populated by the inverted free spins during storage (all-free
        # start).  With the default burn-in the pre-populated bound pool is NOT inverted and
        # the idealisation breaks; that regime needs a finite-pulse oracle (future work).
        kw.update(kappa_MT=KAPPA_MT, dwell_time=DWELL, T2_bound=1e-5, T1_bound=T1B,
                  equilibrate_binding='off')
    _, mz = simulate_bloch(4000, D, wf, Sphere(radius=R), INVERT, **kw)
    return float(mz[0])


def _oracle_total_mz(TM):
    """Two-pool TOTAL longitudinal magnetisation after inverting both pools.

    The MC ensemble (all spins M0=1, fraction f_b bound) maps to a two-pool system with
    M0a = 1 - f_b (free), M0b = f_b (bound); the mean Mz is the pool total Mza + Mzb."""
    f_b = k_f / (k_f + k_r)
    M0a, M0b = 1.0 - f_b, f_b
    A = mt.two_pool_generator(R1a=1 / T1A, R2a=1e3, R1b=1 / T1B, R2b=1e3,
                              k_f=k_f, k_r=k_r, M0a=M0a, M0b=M0b)
    s0 = np.array([0, 0, -M0a, 0, 0, -M0b, 1.0])
    s = mt.evolve_two_pool(s0, TM, A)
    return s[2] + s[5]


def test_mt_changes_stored_magnetization():
    """MT is active during pure longitudinal storage and the effect grows with TM."""
    d_short = abs(_stored_mz(0.05, with_mt=True) - _stored_mz(0.05, with_mt=False))
    d_long = abs(_stored_mz(0.30, with_mt=True) - _stored_mz(0.30, with_mt=False))
    assert d_long > d_short > 1e-3                # MT accrues during longitudinal storage


def test_storage_recovery_matches_two_pool_oracle():
    """Ensemble Mz(TM) under exchange matches the two-pool longitudinal oracle."""
    for TM in (0.05, 0.15, 0.4):
        mc = _stored_mz(TM, with_mt=True)
        orc = _oracle_total_mz(TM)
        assert abs(mc - orc) < 0.05, f"TM={TM}: MC {mc:.3f} vs oracle {orc:.3f}"
