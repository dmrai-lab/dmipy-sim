"""Emergent MT saturation / Z-spectrum end-to-end (Piece E of the MT staging ladder).

An off-resonance CW saturation pulse, propagated through the forward vector-Bloch
engine on an MT-binding substrate, saturates the broad (short-T2b) bound pool over a
wide offset range while sparing the narrow free-water line; the saturation transfers
to the free pool through the walk's emergent bind/release exchange -- the MT dip and
its Z-spectrum EMERGE (no super-Lorentzian lineshape is imposed).  Validated against
the analytic two-pool Z-spectrum oracle (dmipy_sim.mt.mt_z_spectrum).

Heavy Monte-Carlo -> GPU-recommended; marked slow.  Fine dt so the carrier
2*pi*offset*dt does not alias.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from dmipy_sim import Sphere, simulate_bloch, gradient_echo, prepend_mt_prep, run_bloch_sequence
from dmipy_sim import mt

pytestmark = pytest.mark.slow

# well-mixed sphere (mixing R^2/D = 2 ms << 1/k_f = 25 ms); bound pool broad (T2b ~ 10 us)
R, D = 2e-6, 2e-9
k_f, k_r = 40.0, 80.0
T2a, T1a, T2b, T1b = 80e-3, 1.0, 1e-5, 1.0
W1_HZ, T_SAT, DT = 200.0, 0.025, 2e-5
KAPPA_MT, DWELL = k_f * R / 3.0, 1.0 / k_r


def _mz(offset_hz, *, with_mt):
    n_t = int(round(T_SAT / DT)) + 1
    wf = SimpleNamespace(G=np.zeros((1, n_t, 3)), dt=DT)
    flip = 360.0 * W1_HZ * T_SAT                            # CW over the window
    rf = [{'t_s': T_SAT / 2, 'flip_deg': flip, 'axis_deg': 0.0,
           'duration_s': T_SAT, 'offset_hz': offset_hz}]
    kw = dict(T2=T2a, T1=T1a, return_mz=True, seed=3)
    if with_mt:
        kw.update(kappa_MT=KAPPA_MT, dwell_time=DWELL, T2_bound=T2b, T1_bound=T1b)
    _, mz = simulate_bloch(4000, D, wf, Sphere(radius=R), rf, **kw)
    return float(mz[0])


def test_emergent_zspectrum_matches_oracle():
    # Include off-resonance WINGS (a few kHz), where the bound-pool exchange rate k_f
    # actually shapes the Z-spectrum -- a near-resonance-only check is insensitive to k_f
    # and misses discretisation errors in the saturation transfer.  With the float64
    # magnetisation evolution + effective-field rotation, the emergent MC matches the
    # two-pool oracle to the Monte-Carlo noise floor across the whole spectrum.
    offsets = np.array([0.0, 500.0, 1000.0, 3000.0, 8000.0])
    mc = np.array([_mz(o, with_mt=True) for o in offsets])
    Z = mt.mt_z_spectrum(offsets, w1_hz=W1_HZ, t_sat=T_SAT, T1a=T1a, T2a=T2a,
                         T1b=T1b, T2b=T2b, k_f=k_f, k_r=k_r)
    rel = np.abs(mc - Z) / np.maximum(np.abs(Z), 0.05)
    assert np.max(rel) < 0.04, f"MC {np.round(mc,3)} vs oracle {np.round(Z,3)}"


def test_mt_dip_shape_and_specificity():
    mz0, mz500, mz2k = _mz(0.0, with_mt=True), _mz(500.0, with_mt=True), _mz(2000.0, with_mt=True)
    assert mz0 < mz500 < mz2k                              # Z-spectrum recovers off-resonance
    # off-resonance the dip needs a bound pool: MT vs no-MT at 2 kHz
    mtr = 1.0 - mz2k / _mz(2000.0, with_mt=False)
    assert mtr > 0.03                                      # a real off-resonance MT effect
    assert _mz(2000.0, with_mt=False) > 0.9               # free pool spared without a bound pool


# ── the full C+D+E flow: MT-prepped GRE readout -> off-resonance MTR emerges ─────
def test_mt_prep_gre_offresonance_mtr():
    gre = gradient_echo(TE=2e-3, dt=DT)
    prep = dict(offset_hz=1500.0, duration_s=10e-3, b1_hz=W1_HZ, spoiler_s=1e-3, n_cycles=32.0)
    seq = prepend_mt_prep(gre, prep)
    base = dict(T2=T2a, T1=T1a, seed=3)
    mt_kw = dict(kappa_MT=KAPPA_MT, dwell_time=DWELL, T2_bound=T2b, T1_bound=T1b)
    S_sat = abs(run_bloch_sequence(seq, 4000, D, Sphere(radius=R), **base, **mt_kw)[0])
    S_ref = abs(run_bloch_sequence(gradient_echo(TE=2e-3, dt=DT), 4000, D,
                                   Sphere(radius=R), **base, **mt_kw)[0])
    mtr = 1.0 - S_sat / S_ref
    assert mtr > 0.03                                      # emergent off-resonance MTR (bound pool)
