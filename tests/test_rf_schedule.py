"""The RF schedule is the source of a waveform's coherence attributes.

`chi_perp`, `TM`, `stimulated_echo` and `echo_indices` are derived from `rf_events` at
construction, for `Waveform` and `Sequence` alike; the constructors no longer carry them as
flags, and a flag passed explicitly must agree with the schedule.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.waveforms import Waveform, rf_schedule_coherence, apply_rf_schedule
from dmipy_sim.sequences import Sequence


def test_pgste_mask_storage_time_and_stimulated_echo_come_from_the_schedule():
    wf = d.pgste(delta=5e-3, TM=20e-3, G_magnitude=0.1, bvecs=[[1, 0, 0]], n_t=300, slew_rate=np.inf)
    n_pulse = int(round(5e-3 / wf.dt))
    i_recall = int(round(wf.rf_events[2]['t_s'] / wf.dt))
    expect = np.ones(300, bool)
    expect[n_pulse:i_recall] = False                      # what the constructor used to hard-code
    np.testing.assert_array_equal(np.asarray(wf.chi_perp), expect)
    assert wf.stimulated_echo and wf.TM == pytest.approx(20e-3, abs=2 * wf.dt)
    assert wf.echo_indices is None


def test_spin_echo_constructors_are_all_transverse_with_the_echo_at_the_end():
    for wf in (d.pgse(delta=5e-3, DELTA=20e-3, G_magnitude=0.1, bvecs=[[1, 0, 0]], n_t=200),
               d.ogse(frequency=100.0, T_total=40e-3, G_magnitude=0.1, bvecs=[[1, 0, 0]], n_t=400),
               d.trapezoidal_ogse(N=3, delta=10e-3, DELTA=15e-3, G_magnitude=0.1, bvecs=[[1, 0, 0]], n_t=300)):
        assert wf.chi_perp is None and wf.TM is None and not wf.stimulated_echo and wf.echo_indices is None
        chi, TM, ste, echoes = rf_schedule_coherence(wf.rf_events, wf.G.shape[1], wf.dt)
        assert chi.all() and len(echoes) == 1 and abs(round(echoes[0] / wf.dt) - wf.echo_idx) <= 2


def test_cpmg_echo_indices_are_the_echo_times():
    wf = d.cpmg(4, 10e-3, 0.02, [[0, 0, 1]], n_t_per_echo=50)
    np.testing.assert_array_equal(wf.echo_indices, np.arange(1, 5) * 50)     # k*TE on the grid
    seq = Sequence.from_cpmg(4, 10e-3, bvalues=1e9, n_t_per_echo=50)
    n_t = seq.G.shape[1]
    np.testing.assert_array_equal(seq.echo_indices, np.minimum(np.arange(1, 5) * 50, n_t - 1))
    assert seq.chi_perp is None and not seq.stimulated_echo


def test_a_flag_that_disagrees_with_the_schedule_is_refused():
    G = np.zeros((1, 100, 3), np.float32)
    rf = [{'t_s': 0.0, 'flip_deg': 90}, {'t_s': 50 * 1e-4, 'flip_deg': 180}]
    with pytest.raises(ValueError, match="chi_perp"):
        Waveform(G=G, dt=1e-4, echo_idx=99, rf_events=rf, chi_perp=np.zeros(100, bool))
    with pytest.raises(ValueError, match="stimulated_echo"):
        Waveform(G=G, dt=1e-4, echo_idx=99, rf_events=rf, stimulated_echo=True)
    with pytest.raises(ValueError, match="echo_indices"):
        Waveform(G=G, dt=1e-4, echo_idx=99, rf_events=rf, echo_indices=[10, 20])
    with pytest.raises(ValueError, match="echo_idx"):
        Waveform(G=G, dt=1e-4, echo_idx=20, rf_events=rf)
    ok = Waveform(G=G, dt=1e-4, echo_idx=99, rf_events=rf, echo_indices=[100])   # agrees (within rounding)
    assert ok.echo_indices is not None
    bare = Waveform(G=G, dt=1e-4, echo_idx=99)                                   # no schedule: flags as given
    assert bare.chi_perp is None and not bare.stimulated_echo


def test_the_bookkeeping_follows_excite_store_recall_and_refocus():
    dt = 1e-3
    rf = [{'t_s': 0.0, 'flip_deg': 90}, {'t_s': 10e-3, 'flip_deg': 90}, {'t_s': 30e-3, 'flip_deg': 90},
          {'t_s': 35e-3, 'flip_deg': 180}]
    chi, TM, ste, echoes = rf_schedule_coherence(rf, 50, dt)
    assert chi[:10].all() and not chi[10:30].any() and chi[30:].all()
    assert TM == pytest.approx(20e-3) and ste
    assert echoes == [pytest.approx(40e-3)]           # 2*35 - 30: refocused from the recall
