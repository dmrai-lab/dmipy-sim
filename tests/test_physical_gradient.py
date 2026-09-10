"""The stored gradient is the PHYSICAL one; the effective one is derived (#173 piece 3).

``G`` is what the scanner plays; ``G_eff = G * rf_events.sign(t)`` is what the phase integral walks. Every
builder's ``G_eff`` is, bit for bit, the gradient it stored before the flip (the fixture was generated from the
pre-piece-3 tree); a shorter-timing row is placed symmetric about the one 180 instead of starting at t = 0
with the pulse inside its lobe; the two constructors that folded a 180 without declaring it now declare it;
and ``validate()`` refuses a finite pulse over a live gradient and an unrefocused echo.
"""
from pathlib import Path

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences as S
from dmipy_sim.acquisition.rf import RFEvent, RFSchedule
from dmipy_sim.acquisition.waveforms import b_from_gradient
from dmipy_sim.sequences import Sequence

FIX = np.load(Path(__file__).parent / "fixtures" / "effective_gradient_pre_piece3.npz")
bv = np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]]); B = np.array([1e9, 2e9]); D2 = bv


def _builders():
    """The recipes the fixture was generated from, on this tree."""
    G = np.zeros((1, 200, 3), np.float32); G[0, :50, 0] = 0.05; G[0, 50:100, 0] = -0.05
    return {
        "waveforms.pgse.square": d.set_b(d.pgse(4e-3, 20e-3, 0.05, bv, 240, slew_rate=np.inf), B),
        "waveforms.pgse.slew": d.set_b(d.pgse(4e-3, 20e-3, 0.05, bv, 240, slew_rate=200.0), B),
        "waveforms.pgste": d.set_b(d.pgste(4e-3, 20e-3, 0.05, bv, 240), B),
        "waveforms.ogse.even": d.set_b(d.ogse(100.0, 40e-3, 0.05, bv, 400), B),
        "waveforms.ogse.odd": d.set_b(d.ogse(100.0, 40e-3, 0.05, bv, 401), B),
        "waveforms.trapezoidal_ogse": d.set_b(d.trapezoidal_ogse(4, 20e-3, 24e-3, 0.05, bv, 400, slew_rate=200.0), B),
        "waveforms.cpmg": d.cpmg(3, 20e-3, 0.05, bv, n_t_per_echo=50),
        "waveforms.ste": d.ste(4e-3, 20e-3, 0.05, 240),
        "waveforms.pte": d.pte(4e-3, 20e-3, 0.05, [0.0, 0.0, 1.0], 240),
        "Sequence.pgse.square": S.pgse(B, D2, 4e-3, 20e-3, n_t=240, slew_rate=np.inf),
        "Sequence.pgse.slew": S.pgse(B, D2, 4e-3, 20e-3, n_t=240, slew_rate=200.0),
        "Sequence.cpmg": S.cpmg(3, 20e-3, bvalues=[0.0, 5e8, 1e9], n_t_per_echo=50),
        "Sequence.ogse.square": S.ogse(B, D2, 100.0, 20e-3, n_t=400, slew_rate=np.inf),
        "Sequence.ogse.twotrain": S.ogse(B, D2, 100.0, 20e-3, n_t=400, slew_rate=200.0, refocus_duration=4e-3),
        "Sequence.ste": S.ste(B, 4e-3, 20e-3, n_t=240),
        "Sequence.pte": S.pte(B, [0.0, 0.0, 1.0], 4e-3, 20e-3, n_t=240),
        "Sequence.from_waveform": S.from_waveform(G, 1e-4, bv[:1]),
    }


def _unshifted_rows(key, obj):
    """The rows a builder places exactly where it did before the flip: every row of every builder except a
    slew-limited Sequence PGSE row whose ramp is shorter than the longest row's -- that row's pair is now centred
    on the 180 (moved right by half the ramp difference), where it used to start at t = 0."""
    n = np.asarray(obj.G).shape[0]
    if key != "Sequence.pgse.slew":
        return list(range(n))
    eps = np.minimum(np.asarray(obj.gradient_strengths) / 200.0, np.asarray(obj.delta))
    return [m for m in range(n) if eps[m] == eps.max()]


@pytest.mark.parametrize("key", sorted(k for k in FIX.files if k != "Sequence.pgse.mixedDelta.square"))
def test_the_effective_gradient_is_bit_for_bit_what_was_stored_before(key):
    obj = _builders()[key]
    G_eff, old = np.asarray(obj.G_eff, np.float32), FIX[key]
    rows = _unshifted_rows(key, obj)
    assert rows, key
    np.testing.assert_array_equal(G_eff[rows], old[rows])
    for m in set(range(old.shape[0])) - set(rows):                         # a re-placed row: same pulse, same b
        assert not np.array_equal(G_eff[m], old[m])
        assert b_from_gradient(G_eff[m][None], obj.dt)[0] == pytest.approx(b_from_gradient(old[m][None], obj.dt)[0], rel=1e-6)
        assert np.count_nonzero(np.abs(G_eff[m]).sum(1)) == pytest.approx(np.count_nonzero(np.abs(old[m]).sum(1)), abs=2)


def test_the_effective_gradient_is_the_physical_one_through_the_schedule():
    for key, obj in _builders().items():
        G, G_eff = np.asarray(obj.G, np.float32), np.asarray(obj.G_eff, np.float32)
        s = RFSchedule(obj.rf_events).sign(np.arange(G.shape[1]) * float(obj.dt))
        np.testing.assert_array_equal(G_eff, G * s[None, :, None], err_msg=key)
        np.testing.assert_array_equal(G_eff * s[None, :, None], G, err_msg=key)     # s is its own inverse
        if np.any(s < 0):                                                 # a 180, or a stimulated echo's recall
            assert not np.array_equal(G, G_eff), f"{key}: the schedule flips but folds nothing"
        else:
            np.testing.assert_array_equal(G, G_eff, err_msg=f"{key}: nothing flips, one gradient")


def test_a_spin_echo_stores_same_sign_lobes_and_a_cpmg_a_constant_gradient():
    for wf in (d.pgse(4e-3, 20e-3, 0.05, bv, 240), S.pgse(B, D2, 4e-3, 20e-3, n_t=240)):
        G = np.asarray(wf.G)
        for m in range(2):
            ax = int(np.argmax(np.abs(bv[m])))
            half = G.shape[1] // 2
            assert np.sign(G[m, :half, ax].sum()) == np.sign(G[m, half:, ax].sum()) > 0
    cp = d.cpmg(3, 20e-3, 0.05, bv, n_t_per_echo=50)
    G = np.asarray(cp.G)
    for m in range(2):
        assert np.all(G[m] == G[m][0]), "Carr-Purcell: the physical gradient is constant under the 180 train"


def test_a_shorter_row_is_placed_symmetric_about_the_one_180():
    """Two measurements, Delta = [20, 12] ms, one 180 at T_total/2: the shorter row used to start at t = 0 and have the
    pulse inside its second lobe (|G| = 0.153 T/m there). Now its pair is centred on the 180, its b is unchanged,
    the 180 sits on zero gradient for every row, every row refocuses, and the longest row is untouched."""
    seq = S.pgse(B, D2, np.array([4e-3, 4e-3]), np.array([20e-3, 12e-3]), n_t=240, slew_rate=np.inf)
    old = FIX["Sequence.pgse.mixedDelta.square"]
    G, G_eff = np.asarray(seq.G), np.asarray(seq.G_eff)
    np.testing.assert_array_equal(G_eff[0], old[0])                       # the longest row: bit for bit
    assert not np.array_equal(G_eff[1], old[1])                           # the shorter row moved
    np.testing.assert_allclose(b_from_gradient(G_eff, seq.dt), B, rtol=1e-6)
    t180 = seq.rf_events.refocus_time
    k = int(round(t180 / seq.dt))
    assert np.all(G[:, k - 1:k + 2, :] == 0.0), "the 180 sits on zero gradient for every row"
    assert seq.refocusing_residual < 1e-6
    t = np.arange(G.shape[1]) * seq.dt
    for m in range(2):                                                    # each pair is centred on the 180
        on = np.flatnonzero(np.abs(G[m]).sum(1) > 0)
        assert (t[on[0]] + t[on[-1]]) / 2.0 == pytest.approx(t180, abs=seq.dt)


def test_the_two_train_ogse_and_the_btensor_waveform_declare_their_180():
    og = S.ogse(B, D2, 100.0, 20e-3, n_t=400, slew_rate=200.0, refocus_duration=4e-3)
    assert og.rf_events.refocus_time == pytest.approx(22e-3) and og.refocusing_residual < 1e-6
    k = int(round(og.rf_events.refocus_time / og.dt))
    assert np.all(np.asarray(og.G)[:, k, :] == 0.0)                     # in the gap
    assert not S.ogse(B, D2, 100.0, 20e-3, n_t=400, slew_rate=np.inf).rf_events   # the single-cosine limit: no pulse
    # a physical same-sign pair with its 180 at TE/2 is a spin echo; an off-centre 180 is refused, not allowed
    G = np.zeros((1, 200, 3), np.float32); G[0, 20:60, 2] = 0.05; G[0, 140:180, 2] = 0.05
    seq = Sequence.from_btensor_waveform(G, 1e-4)
    assert seq.rf_events.refocus_time == pytest.approx(100 * 1e-4) and seq.refocusing_residual < 1e-6
    assert seq.bvalues[0] > 0 and np.sign(np.asarray(seq.G_eff)[0, 30, 2]) == -np.sign(np.asarray(seq.G_eff)[0, 150, 2])
    with pytest.raises(ValueError, match="not TE/2"):
        Sequence.from_btensor_waveform(G, 1e-4, echo_idx=60)


def test_validate_refuses_a_finite_pulse_over_a_live_gradient_and_an_unrefocused_echo():
    seq = S.pgse(B, D2, 4e-3, 20e-3, n_t=240, slew_rate=np.inf)
    seq.validate()
    t180 = seq.rf_events.refocus_time
    seq.rf_events = RFSchedule([RFEvent(0.0, 90, 'Mz→Mxy'), RFEvent(t180, 180, 'refocus', duration_s=30e-3)])
    with pytest.raises(ValueError, match="gradient is on during the 180"):
        seq.validate()
    G = np.zeros((1, 200, 3), np.float32); G[0, :50, 0] = 0.05                     # one lobe: never refocuses
    with pytest.raises(ValueError, match="not refocused"):
        S.from_waveform(G, 1e-4, bv[:1])


def test_the_ogse_waveform_declares_its_180_at_exactly_half_the_span():
    for n_t in (400, 401):
        wf = d.ogse(100.0, 40e-3, 0.05, bv, n_t)
        assert wf.rf_events.refocus_time == pytest.approx(20e-3, abs=1e-15)


def test_the_scalar_engine_and_b_read_the_effective_gradient():
    wf = d.set_b(d.pgse(4e-3, 20e-3, 0.05, bv, 240), B)
    np.testing.assert_allclose(d.calc_b(wf), B, rtol=1e-6)
    assert np.all(b_from_gradient(wf.G, wf.dt) > 1.2 * B)                 # the physical pair alone would not refocus
    E = d.simulate(2000, 2e-9, wf, d.FreeDiffusion(), seed=0, require_gpu=False)
    np.testing.assert_allclose(E, np.exp(-B * 2e-9), atol=0.03)
