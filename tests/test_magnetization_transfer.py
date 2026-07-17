"""Magnetization transfer (two-rate model) — the gated transverse / stored longitudinal split.

MT has two channels that sit on opposite sides of the coherence gate:
  * TRANSVERSE MT (bound-pool exchange sink, R2_MT): accrues only while transverse,
    so a stimulated echo (transverse only during the two 2δ encodings) pauses it and
    retains MORE signal than a spin echo (transverse the whole TE).
  * LONGITUDINAL MT (cross-relaxation, R1_MT): accrues only while stored longitudinally,
    so PGSE (no storage) is untouched while PGSTE pays it over the mixing time T_M —
    exactly exp(-R1_MT·T_M).

This is the honest MT trade-off: PGSTE gains on transverse MT but pays on longitudinal MT.
"""
import numpy as np
import pytest

from dmipy_sim import simulate, set_b, FreeDiffusion
from dmipy_sim.waveforms import pgse, pgste

D = 1e-9
N = 4000
DELTA_ENC, DELTA = 5e-3, 40e-3
TM = DELTA - DELTA_ENC
BOX = 12e-6


def _sig(seq, **mt):
    wf = (pgse(delta=DELTA_ENC, DELTA=DELTA, G_magnitude=1.0, bvecs=[[1, 0, 0]],
               n_t=1000, slew_rate=np.inf) if seq == 'pgse'
          else pgste(delta=DELTA_ENC, TM=TM, G_magnitude=1.0, bvecs=[[1, 0, 0]],
                     n_t=1000, slew_rate=np.inf))
    wf = set_b(wf, 0.0)
    r0 = np.random.default_rng(0).uniform(-BOX, BOX, (N, 3)).astype(np.float32)
    return float(np.asarray(simulate(N, D, wf, FreeDiffusion(), seed=1, r0=r0,
                                     require_gpu=False, **mt))[0])


def test_transverse_mt_is_gated_pgste_retains_more():
    R = 5.0
    b_pgse, b_pgste = _sig('pgse'), _sig('pgste')
    keep_pgse = _sig('pgse', mt_transverse_rate=R) / b_pgse
    keep_pgste = _sig('pgste', mt_transverse_rate=R) / b_pgste
    assert keep_pgste > keep_pgse + 0.05           # PGSTE gates the transverse channel


def test_longitudinal_mt_hits_pgste_not_pgse():
    R = 5.0
    b_pgse, b_pgste = _sig('pgse'), _sig('pgste')
    keep_pgse = _sig('pgse', mt_longitudinal_rate=R) / b_pgse
    keep_pgste = _sig('pgste', mt_longitudinal_rate=R) / b_pgste
    # PGSE has no storage -> untouched; PGSTE pays over T_M
    assert keep_pgse == pytest.approx(1.0, abs=0.01)
    assert keep_pgste < keep_pgse - 0.05


def test_longitudinal_mt_matches_analytic_storage_loss():
    R = 5.0
    keep_pgste = _sig('pgste', mt_longitudinal_rate=R) / _sig('pgste')
    assert keep_pgste == pytest.approx(np.exp(-R * TM), abs=0.01)


def test_mt_composes_and_no_mt_is_noop():
    # zero rates must exactly reproduce the no-MT signal
    assert _sig('pgse', mt_transverse_rate=0.0, mt_longitudinal_rate=0.0) == pytest.approx(
        _sig('pgse'), abs=1e-6)
