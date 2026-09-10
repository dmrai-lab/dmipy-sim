"""The timing budget on the object (#173 piece 4).

``SequenceTiming`` is where the gradient may not be on: the excitation lead-in, the refocusing window at TE/2,
the readout tail. Its windows are derived, asymmetric by construction, and refused below the echo time both
encoding windows exist for. A finite pulse makes the coherence mask a fractional pathway profile -- a hard
pulse keeps the binary one -- and the scalar engine gates T2 / T1 by it. ``from_pgse(timing=)`` builds to a
budget; ``validate()`` refuses gradient inside its windows; a ``.seq`` carries it both ways.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences as S
from dmipy_sim.acquisition.rf import RFEvent, RFSchedule
from dmipy_sim.acquisition.timing import SequenceTiming


def test_windows_are_derived_and_asymmetric():
    st = SequenceTiming.from_readout(t_excite=2e-3, t_refocus=4e-3, readout_duration=30e-3, partial_fourier=0.75)
    assert st.t_readout_pre_echo == pytest.approx(10e-3)                 # 30 ms * (0.25 / 0.75)
    TE = 80e-3
    (pre0, pre1), (post0, post1) = st.encoding_windows(TE)
    assert pre0 == st.t_lead and pre1 == TE / 2 - 2e-3 and post0 == TE / 2 + 2e-3 and post1 == TE - 10e-3
    assert (pre1 - pre0) > 1.1 * (post1 - post0)                          # asymmetric, derived
    t = np.linspace(0.0, TE, 801)
    on = st.on_mask(t, TE)
    assert on[t < st.t_lead].sum() == 0 and on[np.abs(t - TE / 2) <= 2e-3].sum() == 0 and on[t > TE - 10e-3].sum() == 0
    assert on.sum() == pytest.approx(((pre1 - pre0) + (post1 - post0)) / (t[1] - t[0]), abs=4)
    assert [w for _, _, w in st.windows(TE)] == ["excitation", "refocus", "readout"]


def test_min_te_guards_the_windows_and_bad_budgets_are_refused():
    st = SequenceTiming.from_readout(t_excite=2e-3, t_refocus=4e-3, readout_duration=30e-3, partial_fourier=0.75)
    assert st.min_TE() == pytest.approx(2.0 * (10e-3 + 2e-3))              # the readout tail binds here
    with pytest.raises(ValueError, match="below min_TE"):
        st.windows(st.min_TE() - 1e-3)
    with pytest.raises(ValueError, match="below min_TE"):
        SequenceTiming(2e-3, 4e-3, 10e-3, TE=5e-3)
    with pytest.raises(ValueError, match="partial_fourier"):
        SequenceTiming.from_readout(t_excite=2e-3, t_refocus=4e-3, readout_duration=30e-3, partial_fourier=0.3)
    with pytest.raises(ValueError, match=">= 0"):
        SequenceTiming(-1e-3, 4e-3, 10e-3)
    assert SequenceTiming.from_dict(st.to_dict()) == st


def test_the_budget_implies_a_finite_schedule():
    st = SequenceTiming(t_excite=3e-3, t_refocus=6e-3, t_readout_pre_echo=14e-3)
    rf = st.rf_events(60e-3)
    assert isinstance(rf, RFSchedule) and [e.label for e in rf] == ['Mz→Mxy', 'refocus']
    assert rf[0].t_s == 0.0 and rf[0].duration_s == 3e-3 and rf[1].t_s == 30e-3 and rf[1].duration_s == 6e-3


def test_a_finite_pulse_makes_the_coherence_mask_a_pathway_profile_and_a_hard_one_keeps_it_binary():
    n_t, dt = 401, 1e-4
    hard = RFSchedule((RFEvent(0.0, 90, 'Mz→Mxy'), RFEvent(20e-3, 180, 'refocus')))
    chi, TM, ste, echoes = hard.coherence(n_t, dt)
    assert chi.dtype == bool and chi.all() and TM is None and not ste and echoes == pytest.approx([40e-3])
    fin = RFSchedule((RFEvent(0.0, 90, 'Mz→Mxy'), RFEvent(20e-3, 180, 'refocus', duration_s=4e-3)))
    chi, TM, ste, echoes = fin.coherence(n_t, dt)
    assert chi.dtype == np.float64 and echoes == pytest.approx([40e-3])   # the echo is where it was
    t = np.arange(n_t) * dt
    win = np.abs(t - 20e-3) <= 2e-3 + 1e-12
    assert chi[~win].min() == 1.0                                           # transverse outside the pulse
    assert chi[win].min() == pytest.approx(0.5) and chi[win][0] == pytest.approx(1.0) and chi[win][-1] == pytest.approx(1.0)
    # a quarter of the pulse is spent longitudinal: the ensemble mean of finite_180_longitudinal_dwell
    assert (1.0 - chi[win]).sum() * dt == pytest.approx(4e-3 / 4, rel=0.03)
    # excitation tips z into the plane as sin^2; a store tips it back as cos^2; a recall as sin^2
    pgste = RFSchedule((RFEvent(2e-3, 90, 'Mz→Mxy', duration_s=4e-3), RFEvent(10e-3, 90, 'store', duration_s=2e-3),
                        RFEvent(30e-3, 90, 'recall', duration_s=2e-3)))
    chi, TM, ste, _ = pgste.coherence(n_t, dt)
    assert ste and TM == pytest.approx(20e-3)
    assert chi[0] == pytest.approx(0.0) and chi[40] == pytest.approx(1.0) and chi[20] == pytest.approx(0.5)
    assert chi[100] == pytest.approx(0.5) and chi[150] == pytest.approx(0.0) and chi[300] == pytest.approx(0.5)


def test_the_scalar_engine_gates_relaxation_by_the_fractional_mask():
    """Zero gradient, a finite 180: the signal is exp(-(T_perp / T2) - (T_par / T1)) with the transverse and
    longitudinal times read off the profile -- the check #25 could not write while nothing produced one."""
    n_t, dt, T2, T1 = 401, 1e-4, 30e-3, 300e-3
    rf = RFSchedule((RFEvent(0.0, 90, 'Mz→Mxy'), RFEvent(20e-3, 180, 'refocus', duration_s=8e-3)))
    wf = d.Waveform(G=np.zeros((1, n_t, 3), np.float32), dt=dt, echo_idx=n_t - 1, rf_events=rf)
    chi = np.asarray(wf.chi_perp, float)
    assert 0.0 < chi.min() < 1.0                                            # fractional, and reachable
    S_ = d.simulate(500, 2e-9, wf, d.FreeDiffusion(), T2=T2, T1=T1, seed=0, require_gpu=False)
    expected = np.exp(-(chi.sum() * dt) / T2 - ((1.0 - chi).sum() * dt) / T1)
    assert float(S_[0]) == pytest.approx(expected, rel=2e-3)
    hard = d.Waveform(G=np.zeros((1, n_t, 3), np.float32), dt=dt, echo_idx=n_t - 1,
                      rf_events=RFSchedule((RFEvent(0.0, 90, 'Mz→Mxy'), RFEvent(20e-3, 180, 'refocus'))))
    S_hard = d.simulate(500, 2e-9, hard, d.FreeDiffusion(), T2=T2, T1=T1, seed=0, require_gpu=False)
    assert float(S_hard[0]) == pytest.approx(np.exp(-n_t * dt / T2), rel=2e-3) and S_[0] > S_hard[0]


def test_from_pgse_builds_to_a_budget():
    st = SequenceTiming(t_excite=3e-3, t_refocus=6e-3, t_readout_pre_echo=14e-3)
    B = np.array([1e9, 2e9]); D2 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    seq = S.pgse(B, D2, 8e-3, 24e-3, n_t=1200, slew_rate=200.0, timing=st)
    assert seq.timing is st and [e.duration_s for e in seq.rf_events] == [3e-3, 6e-3]
    TE = float(seq.TE[0])
    assert TE == pytest.approx(seq.rf_events.refocus_time * 2) and TE >= st.min_TE()
    t = np.arange(seq.G.shape[1]) * seq.dt
    assert np.all(np.abs(np.asarray(seq.G))[:, st.on_mask(t, TE) == 0.0, :] == 0.0)   # nothing in an off window
    np.testing.assert_allclose(d.calc_b(seq), B, rtol=1e-6)
    chi = np.asarray(seq.chi_perp, float)
    assert 0.0 < chi.min() < 1.0 and seq.refocusing_residual < 1e-6
    assert seq.refocus_gap >= st.t_refocus
    seq.validate()
    # the pair without a budget is the idealised one; a gap too narrow for the 180 is refused, so is a short TE
    assert S.pgse(B, D2, 8e-3, 24e-3, n_t=600).timing is None
    with pytest.raises(ValueError, match="narrower than the refocusing window"):
        S.pgse(B, D2, 8e-3, 12e-3, n_t=600, slew_rate=np.inf, timing=SequenceTiming(1e-3, 5e-3, 1e-3))
    with pytest.raises(ValueError, match="below the"):
        S.pgse(B, D2, 8e-3, 24e-3, TE=40e-3, n_t=600, slew_rate=np.inf, timing=st)


def test_refocus_gap_is_derived_from_the_gradient_and_the_schedule():
    seq = S.pgse([1e9], [[1.0, 0.0, 0.0]], 8e-3, 24e-3, n_t=640, slew_rate=np.inf)
    assert seq.refocus_gap == pytest.approx(16e-3, abs=seq.dt)             # Delta - delta
    assert S.ogse([1e9], [[1.0, 0.0, 0.0]], 100.0, 20e-3, n_t=400, slew_rate=np.inf).refocus_gap is None   # no 180
    assert S.cpmg(2, 20e-3, bvalues=[1e9, 1e9], n_t_per_echo=100).refocus_gap == 0.0                       # on the gradient


def test_validate_refuses_gradient_inside_a_budget_window():
    st = SequenceTiming(t_excite=3e-3, t_refocus=2e-3, t_readout_pre_echo=2e-3)      # fits a 32 ms echo
    seq = S.pgse([1e9], [[1.0, 0.0, 0.0]], 8e-3, 24e-3, n_t=600, slew_rate=np.inf)   # idealised: lobe 1 at t = 0
    seq.timing = st
    with pytest.raises(ValueError, match="excitation window"):
        seq.validate()
    seq.timing = SequenceTiming(t_excite=3e-3, t_refocus=6e-3, t_readout_pre_echo=14e-3)   # min_TE 34 ms > its 32 ms
    with pytest.raises(ValueError, match="below min_TE"):
        seq.validate()


pypulseq = pytest.importorskip("pypulseq")


def test_a_seq_file_carries_the_budget_both_ways(tmp_path):
    from dmipy_sim.sequences.pulseq import to_pulseq, from_pulseq
    st = SequenceTiming(t_excite=3e-3, t_refocus=6e-3, t_readout_pre_echo=14e-3)
    seq = S.pgse([1e9], [[1.0, 0.0, 0.0]], 8e-3, 24e-3, n_t=1200, slew_rate=200.0, timing=st)
    p = tmp_path / "budget.seq"
    to_pulseq(seq, filename=str(p), native_rf=False)
    back = from_pulseq(str(p))
    assert back.timing is not None
    assert back.timing.t_excite == pytest.approx(3e-3) and back.timing.t_refocus == pytest.approx(6e-3)
    assert back.timing.t_readout_pre_echo == pytest.approx(14e-3)
