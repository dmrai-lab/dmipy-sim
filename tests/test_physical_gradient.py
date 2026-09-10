"""The stored gradient is the PHYSICAL one; the effective one is derived (#173 pieces 3 and 5b).

``G`` is what the scanner plays; ``G_eff = G * rf.sign(t)`` is what the phase integral walks. For every builder
the two agree where nothing flips and differ where a 180 (or a recall) is declared; a spin echo plays the same
block on both sides of its 180 so ``q(TE) = 0`` sample for sample; every row's pair is centred on the one 180;
the readout sample carries no gradient; ``validate()`` refuses a finite pulse over a live gradient and an
unrefocused echo.
"""
from dataclasses import replace

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences as S
from dmipy_sim.acquisition.rf import RFEvent, RFSchedule
from dmipy_sim.acquisition.timing import SequenceTiming
from dmipy_sim.acquisition.waveforms import b_from_gradient

bv = np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]]); B = np.array([1e9, 2e9]); D2 = bv
TM = SequenceTiming(t_excite=2e-3, t_refocus=4e-3, t_readout_pre_echo=3e-3)


def _builders():
    """One instance of every family, on both routes and with and without a budget."""
    G = np.zeros((1, 200, 3), np.float32); G[0, :50, 0] = 0.05; G[0, 50:100, 0] = -0.05
    return {
        "pgse.square": S.pgse(D2, 4e-3, 20e-3, bvalues=B, n_t=240, slew_rate=np.inf),
        "pgse.slew": S.pgse(D2, 4e-3, 20e-3, bvalues=B, n_t=240),
        "pgse.amplitude": S.pgse(D2, 4e-3, 20e-3, gradient_strengths=0.05, n_t=240),
        "pgse.timing": S.pgse(D2, 4e-3, 20e-3, bvalues=B, n_t=600, timing=TM),
        "pgse.TE": S.pgse(D2, 4e-3, 20e-3, bvalues=B, TE=60e-3, n_t=600),
        "pgste": S.pgste(D2, 4e-3, 20e-3, bvalues=B, n_t=400),
        "pgste.timing": S.pgste(D2, 4e-3, 20e-3, bvalues=B, n_t=600, timing=TM),
        "ogse.cosine": S.ogse(D2, 100.0, 20e-3, shape="cosine", bvalues=B, n_t=400, slew_rate=np.inf),
        "ogse.cosine.odd": S.ogse(D2, 100.0, 20e-3, shape="cosine", bvalues=B, n_t=401, slew_rate=np.inf),
        "ogse.trapezoid": S.ogse(D2, 100.0, 20e-3, bvalues=B / 10, n_t=400),
        "ogse.timing": S.ogse(D2, 100.0, 20e-3, bvalues=B / 10, n_t=600, timing=TM),
        "cpmg": S.cpmg(3, 20e-3, gradient_directions=bv, bvalues=[5e8, 1e9], n_t_per_echo=50),
        "cpmg.alternate": S.cpmg(3, 20e-3, gradient_directions=bv, bvalues=[5e8, 1e9], n_t_per_echo=50, polarity="alternate"),
        "cpmg.timing": S.cpmg(3, 40e-3, gradient_directions=bv, bvalues=[5e8, 1e9], n_t_per_echo=200, timing=TM),
        "gre": S.gre(30e-3, gradient_directions=bv, bvalues=B, delta=4e-3, Delta=12e-3, n_t=300),
        "ste": S.ste(24e-3, bvalues=B / 5, n_t=240),
        "pte": S.pte([0.0, 0.0, 1.0], 24e-3, bvalues=B / 5, n_t=240),
        "from_waveform": S.from_waveform(G, 1e-4, bv[:1]),
    }


def test_the_effective_gradient_is_the_physical_one_through_the_schedule():
    for key, obj in _builders().items():
        G, G_eff = np.asarray(obj.G, np.float32), np.asarray(obj.G_eff, np.float32)
        s = RFSchedule(obj.rf).sign(np.arange(G.shape[1]) * float(obj.dt))
        np.testing.assert_array_equal(G_eff, G * s[None, :, None], err_msg=key)
        np.testing.assert_array_equal(G_eff * s[None, :, None], G, err_msg=key)     # s is its own inverse
        if np.any(s < 0):                                                 # a 180, or a stimulated echo's recall
            assert not np.array_equal(G, G_eff), f"{key}: the schedule flips but folds nothing"
        else:
            np.testing.assert_array_equal(G, G_eff, err_msg=f"{key}: nothing flips, one gradient")


def test_every_builder_refocuses_at_its_readout_and_leaves_the_readout_sample_free():
    for key, obj in _builders().items():
        assert obj.refocusing_residual < 1e-9, key
        q = np.cumsum(np.asarray(obj.G_eff, np.float64) * obj.dt, axis=1)
        for i in obj.readout:                                             # every echo of a train too
            assert np.abs(q[:, i - 1]).max() <= 1e-9 * np.abs(q).max(), key
        assert np.all(np.asarray(obj.G)[:, obj.n_t - 1, :] == 0.0), f"{key}: the readout sample acts over nothing"
        np.testing.assert_allclose(b_from_gradient(obj.G_eff, obj.dt), obj.encoding.bvalues, rtol=1e-6, err_msg=key)


def test_a_spin_echo_plays_the_same_block_on_both_sides_of_its_180():
    for key in ("pgse.square", "pgse.slew", "pgse.timing", "pgse.TE", "ogse.cosine", "ogse.trapezoid", "ogse.timing"):
        obj = _builders()[key]
        G = np.asarray(obj.G)
        t180 = obj.rf.refocus_time
        assert t180 == pytest.approx(obj.T / 2.0), key                    # the 180 at exactly TE/2
        t = np.arange(obj.n_t) * obj.dt
        for m in range(obj.n_meas):
            pre, post = G[m, t < t180], G[m, t >= t180]
            on_pre, on_post = np.flatnonzero(np.abs(pre).sum(1)), np.flatnonzero(np.abs(post).sum(1))
            np.testing.assert_array_equal(pre[on_pre[0]:on_pre[-1] + 1], post[on_post[0]:on_post[-1] + 1],
                                          err_msg=f"{key} row {m}: the two blocks differ")
            # centred on the 180: the same dead time before the pair as after it, to a sample
            assert (t[on_pre[0]] + t[t >= t180][on_post[-1]]) / 2.0 == pytest.approx(t180, abs=obj.dt), key


def test_a_shorter_row_is_placed_symmetric_about_the_one_180():
    """Two measurements, Delta = [20, 12] ms, one 180 at TE/2: the shorter row's pair is centred on the 180 with the
    pulse in its own gap, its b is exact, the 180 sits on zero gradient for every row, and every row refocuses."""
    seq = S.pgse(D2, np.array([4e-3, 4e-3]), np.array([20e-3, 12e-3]), bvalues=B, n_t=240, slew_rate=np.inf)
    G = np.asarray(seq.G)
    np.testing.assert_allclose(b_from_gradient(seq.G_eff, seq.dt), B, rtol=1e-6)
    t180 = seq.rf.refocus_time
    k = int(round(t180 / seq.dt))
    assert np.all(G[:, k - 1:k + 2, :] == 0.0), "the 180 sits on zero gradient for every row"
    assert seq.refocusing_residual < 1e-9
    t = np.arange(G.shape[1]) * seq.dt
    for m in range(2):                                                    # each pair is centred on the 180
        on = np.flatnonzero(np.abs(G[m]).sum(1) > 0)
        assert (t[on[0]] + t[on[-1]]) / 2.0 == pytest.approx(t180, abs=seq.dt)
    assert seq.T == pytest.approx(24e-3, abs=seq.dt)                      # the longest row sets TE


def test_a_train_is_carr_purcell_at_either_polarity():
    """The physical gradient of a constant-polarity train is one sign throughout (constant in the square limit);
    an alternating train flips sign every interval; both have the same bipolar ``G_eff`` per interval."""
    const = S.cpmg(3, 20e-3, gradient_directions=bv, gradient_strengths=0.05, n_t_per_echo=50, slew_rate=np.inf)
    alt = S.cpmg(3, 20e-3, gradient_directions=bv, gradient_strengths=0.05, n_t_per_echo=50, slew_rate=np.inf,
                 polarity="alternate")
    Gc, Ga = np.asarray(const.G), np.asarray(alt.G)
    for m in range(2):
        ax = int(np.argmax(np.abs(bv[m])))
        assert np.all(Gc[m, :-1, ax] == Gc[m, 0, ax]) and Gc[m, 0, ax] > 0, "Carr-Purcell: constant under the train"
        signs = [np.sign(Ga[m, k * 50 + 25, ax]) for k in range(3)]
        assert signs == [1.0, -1.0, 1.0], "alternate: one sign per interval"
    # the same bipolar pair per interval in magnitude; constant polarity alternates the pair's phase each echo
    np.testing.assert_array_equal(np.abs(np.asarray(const.G_eff)), np.abs(np.asarray(alt.G_eff)))
    q = np.cumsum(np.asarray(alt.G_eff, np.float64) * alt.dt, axis=1)
    assert np.abs(q[:, [i - 1 for i in alt.readout]]).max() < 1e-9 * np.abs(q).max()
    ramped = S.cpmg(3, 20e-3, gradient_directions=bv, gradient_strengths=0.05, n_t_per_echo=50)
    assert ramped.refocusing_residual < 1e-9 and np.count_nonzero(np.asarray(ramped.G)[0, :, 0]) < np.count_nonzero(Gc[0, :, 0]) + 1


def test_the_btensor_waveform_declares_its_180():
    # a physical same-sign pair with its 180 at TE/2 is a spin echo; an off-centre 180 is refused, not allowed
    G = np.zeros((1, 200, 3), np.float32); G[0, 20:60, 2] = 0.05; G[0, 140:180, 2] = 0.05
    seq = S.from_btensor_waveform(G, 1e-4)
    assert seq.rf.refocus_time == pytest.approx(100 * 1e-4) and seq.refocusing_residual < 1e-6
    assert seq.encoding.bvalues[0] > 0 and np.sign(np.asarray(seq.G_eff)[0, 30, 2]) == -np.sign(np.asarray(seq.G_eff)[0, 150, 2])
    with pytest.raises(ValueError, match="forms its echo at sample"):
        replace(seq, readout=(60,))


def test_validate_refuses_a_finite_pulse_over_a_live_gradient_and_an_unrefocused_echo():
    seq = S.pgse(D2, 4e-3, 20e-3, bvalues=B, n_t=240, slew_rate=np.inf)
    seq.validate()
    t180 = seq.rf.refocus_time
    seq = replace(seq, rf=RFSchedule([RFEvent(0.0, 90, 'Mz→Mxy'), RFEvent(t180, 180, 'refocus', duration_s=30e-3)]))
    with pytest.raises(ValueError, match="gradient is on during the 180"):
        seq.validate()
    G = np.zeros((1, 200, 3), np.float32); G[0, :50, 0] = 0.05                     # one lobe: never refocuses
    with pytest.raises(ValueError, match="not refocused"):
        S.from_waveform(G, 1e-4, bv[:1])


def test_the_scalar_engine_and_b_read_the_effective_gradient():
    wf = d.set_b(d.pgse(bv, 4e-3, 20e-3, gradient_strengths=0.05, n_t=240), B)
    np.testing.assert_allclose(d.calc_b(wf), B, rtol=1e-6)
    assert np.all(b_from_gradient(wf.G, wf.dt) > 1.2 * B)                 # the physical pair alone would not refocus
    E = d.simulate(2000, 2e-9, wf, d.FreeDiffusion(), seed=0, require_gpu=False)
    np.testing.assert_allclose(E, np.exp(-B * 2e-9), atol=0.03)
