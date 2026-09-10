"""ScannerSequence is the one acquisition object (#173 piece 5a): what the scanner does from t = 0 to the
readout. What it stores (physical G, dt, the schedule, the readout, a budget, an encoding) and what it derives
(G_eff, chi_perp, TM, the echoes, b) are pinned here, with the builders that had no public home before: the
stimulated echo, the gradient echo, a precomputed PGSTE, and Protocol for a multi-TE scheme."""
from dataclasses import replace

import numpy as np
from dmipy_sim.acquisition.timing import SequenceTiming
import pytest

import dmipy_sim as d
from dmipy_sim import sequences as S
from dmipy_sim.acquisition.rf import RFEvent, RFSchedule
from dmipy_sim.acquisition.scanner_sequence import Encoding, Protocol, ScannerSequence
from dmipy_sim.acquisition.waveforms import b_from_gradient

B = [1e9, 2e9]
D2 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_stores_the_physical_gradient_and_derives_the_effective_one():
    seq = S.pgse(D2, 4e-3, 20e-3, bvalues=B, n_t=240)
    G, Ge = np.asarray(seq.G), np.asarray(seq.G_eff)
    k = int(round(seq.rf.refocus_time / seq.dt))
    assert G.dtype == np.float32 and G.shape == (2, 240, 3) and Ge.shape == G.shape
    np.testing.assert_array_equal(Ge[:, :k], G[:, :k])                  # before the 180: the same
    np.testing.assert_array_equal(Ge[:, k:], -G[:, k:])                 # after it: negated
    assert np.sign(G[0, 30, 0]) == np.sign(G[0, 200, 0])                # physical lobes share a polarity
    np.testing.assert_allclose(seq.b(), B, rtol=1e-6)                   # the declared b IS the integral of G_eff
    np.testing.assert_allclose(b_from_gradient(seq.G_eff, seq.dt), B, rtol=1e-6)
    assert seq.readout == (239,) and seq.echo_idx == 239 and seq.T == pytest.approx(239 * seq.dt)
    assert seq.chi_perp is None and seq.TM is None and not seq.stimulated_echo
    assert seq.echoes == (pytest.approx(2 * seq.rf.refocus_time),)
    assert seq.refocusing_residual < 1e-6 and seq.family == "pgse" and seq.build_spec[0] == "pgse"


def test_the_object_is_frozen_and_changed_through_replace():
    seq = S.pgse(D2, 4e-3, 20e-3, bvalues=B, n_t=240)
    with pytest.raises(AttributeError):
        seq.G = np.zeros_like(seq.G)
    half = seq.with_gradient(np.asarray(seq.G) / np.sqrt(2.0))
    np.testing.assert_allclose(half.b(), np.asarray(B) / 2.0, rtol=1e-6)
    assert half.rf is seq.rf and half.encoding is seq.encoding            # only G changed
    with pytest.raises(ValueError, match="keep the shape"):
        seq.with_gradient(np.zeros((1, 240, 3)))
    assert replace(seq, encoding=None).encoding is None
    with pytest.raises(ValueError, match="measurements"):
        replace(seq, encoding=Encoding(bvalues=np.array([1e9]), gradient_directions=D2[:1]))


def test_readout_defaults_to_the_grid_end_and_must_sit_on_the_echo():
    G = np.zeros((1, 100, 3), np.float32)
    se = ScannerSequence(G=G, dt=1e-4, rf=[RFEvent(0.0, 90), RFEvent(50e-4, 180)])
    assert se.readout == (99,) and se.echoes == (pytest.approx(100e-4),)
    with pytest.raises(ValueError, match="forms its echo at sample"):
        ScannerSequence(G=G, dt=1e-4, rf=[RFEvent(0.0, 90), RFEvent(20e-4, 180)])   # the grid runs past the echo
    with pytest.raises(ValueError, match="must lie in"):
        ScannerSequence(G=G, dt=1e-4, readout=(100,))
    bare = ScannerSequence(G=G, dt=1e-4)
    assert bare.rf == RFSchedule() and bare.readout == (99,) and bare.chi_perp is None and bare.echoes == ()
    with pytest.raises(TypeError):
        ScannerSequence(G=G, dt=1e-4, rf=[{"t_s": 0.0, "flip_deg": 90}])              # events, not dicts


def test_validate_refuses_gradient_through_a_finite_pulse_but_not_through_a_hard_one():
    G = np.zeros((1, 200, 3), np.float32); G[0, 20:60, 0] = 0.05; G[0, 140:180, 0] = 0.05
    hard = ScannerSequence(G=G, dt=1e-4, rf=[RFEvent(0.0, 90), RFEvent(100e-4, 180)])
    assert hard.validate() is hard
    finite = replace(hard, rf=[RFEvent(0.0, 90), RFEvent(100e-4, 180, duration_s=2e-3)])   # 90-110 samples: off
    assert finite.validate() is finite and 0.0 < float(np.asarray(finite.chi_perp)[100]) < 1.0
    G2 = G.copy(); G2[0, 95:105, 0] = 0.01
    with pytest.raises(ValueError, match="finite pulse needs zero gradient"):
        replace(finite, G=G2).validate()
    G3 = G.copy(); G3[0, 140:180, 0] = 0.04                                                 # unbalanced pair
    with pytest.raises(ValueError, match="not refocused"):
        replace(hard, G=G3).validate()


def test_pgste_is_a_stimulated_echo_with_its_lobes_physical_and_same_sign():
    seq = S.pgste(D2, 4e-3, 30e-3, bvalues=B, n_t=400)
    assert seq.stimulated_echo and seq.TM == pytest.approx(30e-3, abs=2 * seq.dt)
    assert [e.label for e in seq.rf] == ["Mz→Mxy", "store", "recall"] and [e.flip_deg for e in seq.rf] == [90, 90, 90]
    chi = np.asarray(seq.chi_perp)
    st, rc = (int(round(e.t_s / seq.dt)) for e in seq.rf[1:])
    assert not chi[st:rc].any() and chi[:st].all() and chi[rc + 1:].all()      # stored along z over TM
    assert np.all(np.asarray(seq.G)[:, st + 1:rc] == 0.0)                          # no gradient while stored
    G = np.asarray(seq.G)
    assert np.sign(G[0, :st][G[0, :st, 0] != 0, 0][0]) == np.sign(G[0, rc:][G[0, rc:, 0] != 0, 0][0])
    np.testing.assert_allclose(seq.b(), B, rtol=1e-6)
    assert seq.readout == (399,) and seq.refocusing_residual < 1e-3        # off-grid recall: rasterised to the validate tolerance
    assert seq.encoding.ste_flip_angles == (90.0, 90.0, 90.0) and np.allclose(seq.encoding.delta, 4e-3)
    tipped = S.pgste(D2, 4e-3, 30e-3, bvalues=B, n_t=400, ste_flip_angles=(90.0, 60.0, 60.0))
    assert [e.flip_deg for e in tipped.rf] == [90, 60, 60] and tipped.stimulated_echo   # the label says the role
    # the same physics from the amplitude-first builder
    w = d.pgste(D2, 4e-3, 30e-3, gradient_strengths=0.05, n_t=400)
    assert w.stimulated_echo and w.TM == pytest.approx(30e-3, abs=2 * w.dt) and w.family == "pgste"


def test_from_pgste_waveform_reads_a_played_stimulated_echo_and_refuses_an_unmatched_one():
    ref = S.pgste(D2[:1], 4e-3, 30e-3, bvalues=B[:1], n_t=400)
    st, rc = (int(round(e.t_s / ref.dt)) for e in ref.rf[1:])
    back = S.from_pgste_waveform(np.asarray(ref.G), ref.dt, store_idx=st, recall_idx=rc, gradient_directions=D2[:1])
    assert back.stimulated_echo and back.TM == pytest.approx(ref.TM, abs=ref.dt) and back.readout == ref.readout
    np.testing.assert_allclose(back.b(), ref.b(), rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(back.G_eff), np.asarray(ref.G_eff))
    G = np.asarray(ref.G).copy(); G[0, rc:, 0] *= 0.8                                 # a recall lobe that does not match
    with pytest.raises(ValueError):
        S.from_pgste_waveform(G, ref.dt, store_idx=st, recall_idx=rc, gradient_directions=D2[:1])


def test_gre_is_a_bipolar_pair_with_no_180_or_a_pure_fid():
    seq = S.gre(20e-3, bvalues=[1e9], gradient_directions=D2[:1], delta=4e-3, Delta=10e-3, n_t=200)
    assert len(seq.rf) == 1 and seq.rf[0].flip_deg == 90 and seq.rf.refocus_time is None
    G = np.asarray(seq.G)[0, :, 0]
    assert np.sign(G[G != 0][0]) == -np.sign(G[G != 0][-1])                              # self-refocusing: bipolar
    np.testing.assert_array_equal(np.asarray(seq.G_eff), np.asarray(seq.G))                # nothing to un-fold
    np.testing.assert_allclose(seq.b(), [1e9], rtol=1e-6)
    assert seq.T == pytest.approx(20e-3, abs=seq.dt) and seq.encoding.TE[0] == pytest.approx(20e-3)
    fid = S.gre(20e-3, n_t=200)
    assert fid.n_meas == 1 and float(np.abs(fid.G).max()) == 0.0 and fid.b()[0] == 0.0 and fid.family == "gre"


def test_a_protocol_is_a_tuple_of_sequences_one_te_each_in_acquisition_order():
    a = S.pgse(D2, 4e-3, 20e-3, bvalues=B, n_t=240)
    b = S.pgse(D2, 4e-3, 40e-3, bvalues=B, n_t=360)
    p = Protocol([a, b])
    assert isinstance(p, tuple) and len(p) == 2 and p.n_meas == 4
    assert p.echo_times == pytest.approx((a.T, b.T)) and a.T < b.T
    assert [r.tolist() for r in p.rows] == [[0, 1], [2, 3]]                       # one after the other by default
    q = Protocol([a, b], rows=[[0, 2], [1, 3]])                                   # interleaved in the acquisition
    np.testing.assert_array_equal(q.scatter([np.array([10, 30]), np.array([20, 40])]), [10, 20, 30, 40])
    with pytest.raises(ValueError, match="partition"):
        Protocol([a, b], rows=[[0, 1], [1, 3]])
    with pytest.raises(TypeError, match="holds ScannerSequences"):
        Protocol([a, np.asarray(b.G)])
    assert d.Protocol is Protocol and d.ScannerSequence is ScannerSequence and d.Encoding is Encoding
    # simulate takes it: one walk per sequence, the signal in acquisition order
    E = d.simulate(1500, 2e-9, q, d.FreeDiffusion(), seed=0, require_gpu=False)
    Ea = d.simulate(1500, 2e-9, a, d.FreeDiffusion(), seed=0, require_gpu=False)
    Eb = d.simulate(1500, 2e-9, b, d.FreeDiffusion(), seed=0, require_gpu=False)
    np.testing.assert_allclose(E, [Ea[0], Eb[0], Ea[1], Eb[1]], atol=1e-6)
    with pytest.raises(ValueError, match="one sequence"):
        d.simulate(100, 2e-9, q, d.FreeDiffusion(), seed=0, require_gpu=False, return_positions=True)


def test_every_sequence_builder_is_validated_and_declares_its_family():
    bv = D2[:1]
    built = {
        "pgse": S.pgse(bv, 4e-3, 20e-3, bvalues=[1e9], n_t=240),
        "pgste": S.pgste(bv, 4e-3, 20e-3, bvalues=[1e9], n_t=240),
        "gre": S.gre(20e-3, bvalues=[1e9], gradient_directions=bv, delta=4e-3, Delta=10e-3, n_t=200),
        "cpmg": S.cpmg(3, 20e-3, bvalues=[1e9] * 3, n_t_per_echo=50),
        "ogse": S.ogse(bv, 100.0, 20e-3, bvalues=[1e8], timing=SequenceTiming(t_excite=0.0, t_refocus=4e-3, t_readout_pre_echo=0.0), n_t=400),
        "ste": S.ste(4e-3 + 20e-3, bvalues=[1e8], n_t=240),
        "pte": S.pte([0.0, 0.0, 1.0], 4e-3 + 20e-3, bvalues=[1e8], n_t=240),
    }
    for name, seq in built.items():
        assert isinstance(seq, ScannerSequence) and seq.family == name, name
        assert seq.encoding is not None and seq.build_spec[0] == name, name
        assert seq.validate() is seq and seq.refocusing_residual < 1e-3, name
    assert built["cpmg"].readout == tuple(np.asarray(built["cpmg"].readout)) and len(built["cpmg"].readout) == 3


def test_the_readers_of_a_played_gradient_take_a_budget():
    """A designer's output built to a budget arrives with it: the 90 / 180 (or the three 90s) are finite, the
    sequence carries the budget and validate() holds the gradient to its windows."""
    tm = SequenceTiming(t_excite=2e-3, t_refocus=4e-3, t_readout_pre_echo=3e-3)
    ref = S.pgse(D2[:1], 4e-3, 20e-3, bvalues=B[:1], n_t=400, timing=tm)
    back = S.from_btensor_waveform(np.asarray(ref.G), ref.dt, timing=tm)
    assert back.timing is tm and [e.duration_s for e in back.rf] == [2e-3, 4e-3]
    np.testing.assert_allclose(back.b(), ref.b(), rtol=1e-6)
    ste_ = S.pgste(D2[:1], 4e-3, 30e-3, bvalues=B[:1], n_t=400, timing=tm)
    st, rc = (int(round(e.t_s / ste_.dt)) for e in ste_.rf[1:])
    back = S.from_pgste_waveform(np.asarray(ste_.G), ste_.dt, store_idx=st, recall_idx=rc, timing=tm)
    assert back.timing is tm and [e.duration_s for e in back.rf] == [2e-3] * 3 and back.stimulated_echo
    G = np.asarray(ref.G).copy(); G[0, 16:19, 0] = 0.01                                    # into the lead-in
    with pytest.raises(ValueError, match="lead-in window|during the 90 pulse"):
        S.from_btensor_waveform(G, ref.dt, timing=tm)
