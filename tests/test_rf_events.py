"""One RF dialect (#173 piece 2): ``RFEvent`` is what every builder emits and every reader reads; a hard
pulse, a finite hard pulse and a shaped pulse are the same object at three settings; ``RFSchedule`` is the
schedule, valid by construction and the one place anything is derived from the pulses; nothing else is
accepted anywhere -- the only dict an event is read from is its own serialised record.
"""
import json

import numpy as np
import pytest

import dmipy_sim
from dmipy_sim.acquisition.rf import B1Pulse, RFEvent, RFSchedule
from dmipy_sim.constants import GAMMA

IDEAL = RFSchedule((RFEvent(0.0, 90, 'Mz→Mxy'), RFEvent(0.02, 180, 'refocus')))
FINITE = RFSchedule((RFEvent(0.0, 90.0, axis_deg=90.0), RFEvent(0.02, 180.0, duration_s=2e-3, offset_hz=40.0)))


def _const_pulse(flip_deg, duration, n=200, phase=0.0):
    """A constant-amplitude B1(t) whose gamma * int |B1| dt is ``flip_deg``."""
    b1 = np.deg2rad(flip_deg) / (GAMMA * duration) * np.exp(1j * phase)
    return B1Pulse.from_samples(np.full(n, b1), duration / n)


def test_a_schedule_is_events_in_time_order_and_nothing_else_is_accepted():
    """There is one way to spell a pulse and one way to hold a schedule. A dict -- either of the two spellings the
    builders once used -- is refused with the constructor to use, not converted."""
    assert RFSchedule() == () and RFSchedule(None) == () and RFSchedule([]) == () and not RFSchedule()
    assert RFSchedule(IDEAL) is IDEAL and RFSchedule(list(IDEAL)) == IDEAL
    assert [e.t_s for e in RFSchedule(tuple(reversed(IDEAL)))] == [0.0, 0.02]          # sorted by time
    assert isinstance(IDEAL, tuple) and IDEAL[1].label == 'refocus' and len(IDEAL) == 2
    for legacy in ([{'t_s': 0.0, 'label': 'Mz→Mxy', 'flip_deg': 90}],
                   [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': 90.0, 'duration_s': 0.0, 'offset_hz': 0.0}],
                   [{'t_s': 0.0, 'flip_deg': 90.0, 'duration_s': 1e-3, 'b1_envelope': [0.2, 1.0, 0.2]}],
                   [("t_s", 0.0)]):
        with pytest.raises(TypeError, match="an RF event is an RFEvent"):
            RFSchedule(legacy)
    with pytest.raises(ValueError, match="needs flip_deg"):
        RFEvent(0.0)
    with pytest.raises(ValueError, match="not an RFEvent.to_dict record"):
        RFEvent.from_dict({'t_s': 0.0, 'flip_deg': 90.0, 'b1_envelope': [1.0]})


def test_the_schedule_is_the_one_place_things_are_derived_from_the_pulses():
    """Refocus time, the spin-echo sign, the coherence mask and the mixing time come from the schedule; a gradient echo (no pulse) has none of them."""
    se = IDEAL
    assert se.refocus_time == 0.02 and RFSchedule().refocus_time is None
    t = np.linspace(0.0, 0.04, 9)
    np.testing.assert_array_equal(se.sign(t), np.where(t >= 0.02, -1.0, 1.0))
    np.testing.assert_array_equal(RFSchedule().sign(t), np.ones_like(t))
    chi, TM, ste, echoes = se.coherence(41, 1e-3)
    assert chi.all() and TM is None and not ste and echoes == pytest.approx([0.04])
    assert se.mixing_time == (None, False)
    pgste = RFSchedule((RFEvent(0.0, 90, 'Mz→Mxy'), RFEvent(0.01, 90, 'store'), RFEvent(0.03, 90, 'recall')))
    assert pgste.mixing_time == (pytest.approx(0.02), True)
    chi, TM, ste, _ = pgste.coherence(41, 1e-3)
    assert TM == pytest.approx(0.02) and ste and not chi[15] and chi[5] and chi[35]
    np.testing.assert_array_equal(pgste.sign(t), np.where(t >= 0.03, -1.0, 1.0))         # flips at RECALL
    unlabelled = RFSchedule((RFEvent(0.0, 90), RFEvent(0.01, 90), RFEvent(0.03, 90)))
    assert unlabelled.mixing_time == (pytest.approx(0.02), True)
    assert [e.t_s for e in se.shifted(0.005)] == pytest.approx([0.005, 0.025]) and isinstance(se.shifted(1.0), RFSchedule)


def test_an_envelope_is_the_shape_and_the_declared_values_must_agree_with_it():
    p = _const_pulse(90.0, 1e-3)
    e = RFEvent(0.01, envelope=p)
    assert e.flip_deg == pytest.approx(90.0) and e.duration_s == pytest.approx(1e-3) and not e.is_hard
    assert RFEvent(0.01, 90.0, envelope=p).flip_deg == pytest.approx(90.0)          # agreeing declaration is fine
    with pytest.raises(ValueError, match="disagrees with the envelope"):
        RFEvent(0.01, 180.0, envelope=p)
    with pytest.raises(ValueError, match="disagrees with the envelope"):
        RFEvent(0.01, envelope=p, duration_s=5e-3)
    with pytest.raises(TypeError):
        RFEvent(0.01, 90.0, envelope=[1.0, 1.0])


def test_flip_split_is_one_pulse_at_three_settings():
    hard = RFEvent(0.0, 90.0)
    d, a = hard.flip_split(1)
    assert d == pytest.approx([np.pi / 2]) and a == pytest.approx([0.0])
    finite = RFEvent(0.0, 90.0, axis_deg=90.0, duration_s=1e-3)
    d, a = finite.flip_split(5)
    assert d == pytest.approx(np.full(5, np.pi / 10)) and a == pytest.approx(np.full(5, np.pi / 2))
    shaped = RFEvent(0.0, envelope=_const_pulse(90.0, 1e-3, phase=np.pi / 2), axis_deg=30.0)
    d, a = shaped.flip_split(8)
    assert d.sum() == pytest.approx(np.pi / 2) and d == pytest.approx(np.full(8, np.pi / 16))
    assert a == pytest.approx(np.full(8, np.deg2rad(30.0) + np.pi / 2))        # axis_deg + the B1 phase
    # a shaped pulse: the flip lands where |B1| is, the total is exact at any nsub
    w = np.array([0.0, 1.0, 3.0, 1.0, 0.0]); b1 = w / (GAMMA * w.sum() * 2e-4) * np.pi
    tri = RFEvent(0.0, envelope=B1Pulse.from_samples(b1, 2e-4))
    d5, _ = tri.flip_split(5)
    assert d5 == pytest.approx(np.pi * w / w.sum()) and tri.flip_split(7)[0].sum() == pytest.approx(np.pi)


def test_both_rasterisers_read_the_one_object():
    """The engine's and the replay's rasterisers read the same RFEvent: a hard pulse is one rotation at its
    instant in both, a finite pulse the same sub-rotations in both."""
    from dmipy_sim.engine.bloch import _build_rf_schedule
    from dmipy_sim.replay.trajectories import _bloch_timeline
    dt, n_t = 1e-4, 300
    dflip, axis, carrier = _build_rf_schedule(FINITE, dt, n_t)
    assert dflip.sum() == pytest.approx(np.deg2rad(270.0)) and np.count_nonzero(dflip) == 1 + 20
    _, rots, windows = _bloch_timeline(FINITE, n_t, dt)
    assert len(rots) == 1 + 20 and sum(f for _, f, _ in rots) == pytest.approx(np.deg2rad(270.0))
    assert windows[1][2] == pytest.approx(2 * np.pi * 40.0)


def test_events_round_trip_through_plain_data():
    p = _const_pulse(90.0, 1e-3, n=16, phase=0.3)
    ev = (RFEvent(0.0, 90.0, 'Mz→Mxy'), RFEvent(0.01, 180.0, 'refocus', axis_deg=45.0, duration_s=2e-3, offset_hz=10.0),
          RFEvent(0.02, envelope=p, label='refocus'))
    back = RFSchedule.from_dicts(json.loads(json.dumps(RFSchedule(ev).to_dicts())))
    assert back == RFSchedule(ev) and back[2].envelope.dt == p.dt and np.allclose(back[2].envelope.b1, p.b1)
    assert RFEvent(0.0, 90.0) == RFEvent(0.0, 90.0) and RFEvent(0.0, 90.0) != RFEvent(0.0, 90.0, 'x')
    assert len({RFEvent(0.0, 90.0), RFEvent(0.0, 90.0)}) == 1


def test_a_bloch_sequence_now_carries_coherence_labels_so_it_un_folds():
    """The finite-pulse builders emitted flips without labels; now their 180 says 'refocus', so the same
    RFSchedule.sign un-folds a ScannerSequence."""
    from dmipy_sim.engine.pulse_sequence import bare_spin_echo, fexi
    se = bare_spin_echo(20e-3, 1e-4)
    assert [e.label for e in se.rf] == ['Mz→Mxy', 'refocus']
    s = se.rf.sign(np.arange(se.n_t) * se.dt)
    assert s[:se.n_t // 2].min() == 1.0 and s[se.n_t // 2 + 1:].max() == -1.0
    fx = fexi(4e-3, 20e-3, 1e-4, g_filter=0.05, g_detect=0.05)
    assert [e.label for e in fx.rf] == ['Mz→Mxy', 'store', 'recall']


def test_public_names():
    assert dmipy_sim.RFEvent is RFEvent and dmipy_sim.RFSchedule is RFSchedule
