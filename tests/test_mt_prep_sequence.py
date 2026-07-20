"""Sequence builder + emergent voxel-scale crusher (Piece C of the MT staging ladder).

Pins the lean forward-native ``dmipy_sim.pulse_sequence`` (readout builders, the
MT-prep saturation block, the crusher) against the forward engine.  MT *binding* is
Piece D; here the substrate has no bound pool, so an off-resonance saturation pulse
mainly demonstrates the plumbing + free-water specificity (the broad bound-pool dip
is added in Piece E).  Free diffusion, small walker counts, fast.
"""
import numpy as np
import pytest

from dmipy_sim import FreeDiffusion
from dmipy_sim.pulse_sequence import (BlochSequence, gradient_echo, spin_echo,
                                      prepend_mt_prep, run_bloch_sequence)

D = 2e-9


# ── structure: the MT-prep block prepends correctly ─────────────────────────────
def test_mt_prep_structure():
    dt = 1e-4
    gre = gradient_echo(TE=2e-3, dt=dt)
    n0 = gre.n_t
    prep = dict(offset_hz=800.0, duration_s=5e-3, b1_hz=100.0, spoiler_s=1e-3)
    seq = prepend_mt_prep(gre, prep)
    n_sat, n_spoil = int(round(5e-3 / dt)), int(round(1e-3 / dt))
    assert seq.n_t == n0 + n_sat + n_spoil
    sat = seq.rf_events[0]
    assert sat['offset_hz'] == 800.0
    assert sat['flip_deg'] == pytest.approx(360.0 * 100.0 * 5e-3)   # 360*b1_hz*dur
    assert seq.crusher is not None and len(seq.crusher['windows_s']) == 1
    # the original excitation is shifted by the whole prep block
    assert seq.rf_events[1]['t_s'] == pytest.approx((n_sat + n_spoil) * dt)


# ── spin-echo readout: pure T2 (nothing to refocus) -> exp(-TE/T2) ──────────────
def test_spin_echo_readout_T2():
    dt, TE, T2 = 2e-4, 20e-3, 0.06
    seq = spin_echo(TE, dt)
    s = run_bloch_sequence(seq, 2000, D, FreeDiffusion(), T2=T2, seed=0)
    assert abs(s[0]) == pytest.approx(np.exp(-TE / T2), rel=0.03)


# ── emergent crusher: a voxel-scale window dephases the transverse residual ──────
def test_crusher_dephases_transverse():
    dt, n_t = 1e-4, 100
    G = np.zeros((1, n_t, 3))
    exc = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': 90.0, 'duration_s': 0.0, 'offset_hz': 0.0}]
    base = BlochSequence(G=G, dt=dt, rf_events=exc, complex_signal=True)
    crushed = BlochSequence(G=G, dt=dt, rf_events=exc, complex_signal=True,
                            crusher=dict(windows_s=[(dt, n_t * dt)], n_cycles=32.0))
    s_free = run_bloch_sequence(base, 4000, D, FreeDiffusion(), seed=0)
    s_crush = run_bloch_sequence(crushed, 4000, D, FreeDiffusion(), seed=0)
    assert abs(s_free[0]) == pytest.approx(1.0, rel=1e-2)   # coherent, undephased
    assert abs(s_crush[0]) < 0.1                            # ensemble dephased away


# ── MT-prep free-water specificity (no bound pool): on-res saturates, off-res spared
def test_mt_prep_free_water_specificity():
    dt = 1e-4
    gre = gradient_echo(TE=2e-3, dt=dt)

    def mtr(offset_hz, flip_deg):
        prep = dict(offset_hz=offset_hz, duration_s=5e-3, flip_deg=flip_deg,
                    spoiler_s=1e-3, n_cycles=32.0)
        s = run_bloch_sequence(prepend_mt_prep(gre, prep), 4000, D, FreeDiffusion(),
                               T2=0.05, T1=1.0, seed=0)
        return abs(s[0])

    base = mtr(0.0, 0.0)                          # flip 0 -> no saturation (same timeline)
    on = 1.0 - mtr(0.0, 90.0) / base             # on-resonance 90 -> tipped then crushed
    off = 1.0 - mtr(5000.0, 90.0) / base         # 5 kHz off-resonance -> free pool spared
    assert on > 0.7                              # direct saturation of the free pool
    assert off < 0.15                            # narrow free line spared far off-resonance
    assert on > off
