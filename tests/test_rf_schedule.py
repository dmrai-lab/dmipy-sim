"""The RF schedule is the source of a waveform's coherence attributes.

`chi_perp`, `TM`, `stimulated_echo` and `echo_indices` are derived from `rf_events` at
construction, on the one `ScannerSequence`; the constructors no longer carry them as
flags, and a flag passed explicitly must agree with the schedule.
"""
import numpy as np
from dmipy_sim import RFEvent
import pytest

import dmipy_sim as d
from dmipy_sim.acquisition.scanner_sequence import ScannerSequence
from dmipy_sim.acquisition.rf import RFSchedule
from dmipy_sim import sequences as _seqmod


def test_pgste_mask_storage_time_and_stimulated_echo_come_from_the_schedule():
    wf = d.pgste([[1, 0, 0]], 5e-3, 20e-3, gradient_strengths=0.1, n_t=300, slew_rate=np.inf)
    n_pulse = int(round(5e-3 / wf.dt))
    i_recall = int(round(wf.rf[2].t_s / wf.dt))
    expect = np.ones(300, bool)
    expect[n_pulse:i_recall] = False                      # what the constructor used to hard-code
    np.testing.assert_array_equal(np.asarray(wf.chi_perp), expect)
    assert wf.stimulated_echo and wf.TM == pytest.approx(20e-3, abs=2 * wf.dt)
    assert wf.readout == (wf.n_t - 1,)


def test_spin_echo_constructors_are_all_transverse_with_the_echo_at_the_end():
    for wf in (d.pgse([[1, 0, 0]], 5e-3, 20e-3, gradient_strengths=0.1, n_t=200),
               d.ogse([[1, 0, 0]], 100.0, (40e-3) / 2, gradient_strengths=0.1, shape="cosine", slew_rate=np.inf, n_t=400),
               d.ogse([[1, 0, 0]], 3 / (2 * (10e-3)), 10e-3, gradient_strengths=0.1, shape="trapezoid", Delta=15e-3, slew_rate=200e3, n_t=300)):
        assert wf.chi_perp is None and wf.TM is None and not wf.stimulated_echo and wf.readout == (wf.n_t - 1,)
        chi, TM, ste, echoes = wf.rf.coherence(wf.G.shape[1], wf.dt)
        assert chi.all() and len(echoes) == 1 and abs(round(echoes[0] / wf.dt) - wf.echo_idx) <= 2


def test_cpmg_echo_indices_are_the_echo_times():
    wf = d.cpmg(4, 10e-3, gradient_strengths=0.02, gradient_directions=[[0, 0, 1]], n_t_per_echo=50)
    np.testing.assert_array_equal(wf.readout, np.arange(1, 5) * 50)     # k*TE on the grid
    seq = _seqmod.cpmg(4, 10e-3, bvalues=1e9, n_t_per_echo=50)
    n_t = seq.G.shape[1]
    np.testing.assert_array_equal(seq.readout, np.minimum(np.arange(1, 5) * 50, n_t - 1))
    assert seq.chi_perp is None and not seq.stimulated_echo


def test_the_coherence_state_is_derived_and_a_readout_off_the_echo_is_refused():
    """A ScannerSequence derives chi_perp, TM, stimulated_echo and its echoes from the schedule -- there is no
    flag to disagree with it. The one thing a caller states is where the signal is read, and a readout that
    is not where the schedule forms its echo is refused."""
    G = np.zeros((1, 100, 3), np.float32)
    rf = [RFEvent(0.0, 90), RFEvent(50 * 1e-4, 180)]
    se = ScannerSequence(G=G, dt=1e-4, readout=(99,), rf=rf)
    assert se.chi_perp is None and not se.stimulated_echo and se.TM is None and se.echoes == (pytest.approx(100e-4),)
    assert ScannerSequence(G=G, dt=1e-4, rf=rf).readout == (99,)                # defaults to the schedule's echo
    with pytest.raises(ValueError, match="forms its echo at sample"):
        ScannerSequence(G=G, dt=1e-4, readout=(20,), rf=rf)
    train = [RFEvent(0.0, 90)] + [RFEvent((k + 0.5) * 20e-4, 180) for k in range(4)]
    assert ScannerSequence(G=G, dt=1e-4, rf=train).readout == (20, 40, 60, 80)
    with pytest.raises(ValueError, match="disagrees with the schedule's echoes"):
        ScannerSequence(G=G, dt=1e-4, readout=(10, 20), rf=train)
    bare = ScannerSequence(G=G, dt=1e-4, readout=(99,))                        # no schedule: transverse, no echo
    assert bare.chi_perp is None and not bare.stimulated_echo and bare.echoes == ()



def test_the_bookkeeping_follows_excite_store_recall_and_refocus():
    dt = 1e-3
    rf = [RFEvent(0.0, 90), RFEvent(10e-3, 90), RFEvent(30e-3, 90),
          RFEvent(35e-3, 180)]
    chi, TM, ste, echoes = RFSchedule(rf).coherence(50, dt)
    assert chi[:10].all() and not chi[10:30].any() and chi[30:].all()
    assert TM == pytest.approx(20e-3) and ste
    assert echoes == [pytest.approx(40e-3)]           # 2*35 - 30: refocused from the recall
