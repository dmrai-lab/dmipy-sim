"""The effective gate is binary across a finite RF pulse, every pulse acting at its centre.

``chi_perp`` across a finite pulse is the transverse FRACTION of the pathway, which relaxation reads. A phase
accrues through ``ScannerSequence.effective_gate`` = ``rf.sign(t)`` times ``RFSchedule.transverse_gate``, which
switches at each pulse's centre: to first order a symmetric pulse turns an off-resonance's phase as an
instantaneous one at its centre does. The fractional profile on a grid that starts at the excitation's centre
keeps half of the excitation and leaves a static field unrefocused; a magnet's background gradient is such a
field. Hard-pulse schedules have a binary ``chi_perp`` already and are unchanged.
"""
import numpy as np
import pytest

from dmipy_sim import sequences
from dmipy_sim.acquisition.rf import RFEvent, RFSchedule
from dmipy_sim.acquisition.scanner_sequence import ScannerSequence
from dmipy_sim.acquisition.timing import SequenceTiming

G0 = 1e-3                                                             # a magnet's own gradient, T/m


def _finite_se():
    tm = SequenceTiming(t_excite=3e-3, t_refocus=5e-3, t_readout_pre_echo=10e-3)
    return sequences.pgse([[0, 0, 1]], 7.6e-3, 30e-3, gradient_strengths=0.04, TE=60e-3, n_t=6001,
                          slew_rate=np.inf, timing=tm)


def _finite_ste(dur=3e-3, dt=1e-5):
    """Three 3 ms pulses (excite at 0, store at 12 ms, recall TM = 38.3 ms later), a 7.6 ms lobe 2 ms after the
    excitation and after the recall, TE 62.3 ms; no gradient inside any pulse window."""
    t_store, TM, delta, lead = 12e-3, 38.3e-3, 7.6e-3, 2e-3
    n_t = int(round((2 * t_store + TM) / dt)) + 1
    t = np.arange(n_t) * dt
    G = np.zeros((1, n_t, 3))
    t_rec = t_store + TM
    G[0, (t >= lead) & (t < lead + delta), 2] = 0.04
    G[0, (t >= t_rec + lead) & (t < t_rec + lead + delta), 2] = 0.04
    rf = RFSchedule([RFEvent(0.0, 90, "Mz→Mxy", duration_s=dur), RFEvent(t_store, 90, "store", duration_s=dur),
                     RFEvent(t_rec, 90, "recall", duration_s=dur)])
    return ScannerSequence(G=G, dt=dt, rf=rf)


def _background_moment(seq):
    """The net moment at the echo of the magnet's gradient alone (T s / m)."""
    bg = seq.with_gradient(np.zeros_like(np.asarray(seq.G))).with_background_gradient([0.0, 0.0, G0])
    return float(np.cumsum(np.asarray(bg.G_eff, np.float64) * bg.dt, axis=1)[0, bg.echo_idx - 1, 2])


@pytest.mark.parametrize("make", [_finite_se, _finite_ste], ids=["spin-echo", "stimulated-echo"])
def test_a_finite_pulse_turns_the_phase_at_its_centre(make):
    """The effective gate is the sign times the binary transverse gate; chi_perp stays the fractional profile."""
    seq = make()
    t = np.arange(seq.n_t) * seq.dt
    assert seq.chi_perp is not None and not np.isin(seq.chi_perp, (0.0, 1.0)).all()
    assert set(np.unique(seq.effective_gate).tolist()) <= {-1.0, 0.0, 1.0}
    hard = seq.rf.sign(t) * seq.rf.transverse_gate(seq.n_t, seq.dt)
    np.testing.assert_array_equal(seq.effective_gate, hard.astype(np.float32))


@pytest.mark.parametrize("make", [_finite_se, _finite_ste], ids=["spin-echo", "stimulated-echo"])
def test_a_finite_pulse_echo_refocuses_a_magnets_gradient(make):
    """With every pulse on a sample, a static gradient is refocused to rounding at the echo; the fractional profile
    left 2.7e-4 s x g0 of it from the half-excitation the grid keeps."""
    seq = make()
    assert abs(_background_moment(seq)) < 1e-6 * seq.dt * G0


@pytest.mark.parametrize("make", [
    lambda: sequences.pgse([[1, 0, 0]], 7.6e-3, 45.9e-3, bvalues=[1e9], TE=53.5e-3, n_t=1000, slew_rate=np.inf),
    lambda: sequences.pgste([[1, 0, 0]], 7.6e-3, 38.3e-3, bvalues=[1e9], TE=53.5e-3, n_t=1000, slew_rate=np.inf),
    lambda: sequences.gre(20e-3, gradient_directions=[[1, 0, 0]], bvalues=[5e8], delta=3e-3, Delta=8e-3, n_t=801,
                          slew_rate=np.inf),
    lambda: sequences.cpmg(4, 10e-3, n_t_per_echo=100),
], ids=["pgse", "pgste", "gre", "cpmg"])
def test_hard_pulses_keep_their_gate(make):
    """A hard-pulse schedule's transverse gate is its chi_perp: the effective gate is the sign times chi_perp."""
    seq = make()
    s = seq.rf.sign(np.arange(seq.n_t) * seq.dt)
    expect = s if seq.chi_perp is None else s * np.asarray(seq.chi_perp, np.float32)
    np.testing.assert_array_equal(seq.effective_gate, expect)
