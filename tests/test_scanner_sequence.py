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


def test_a_prescription_places_the_acquisition_in_the_bore_and_derives_nothing():
    """The acquisition in space: isocenter, axes, voxel size, matrix. Optional, keyword-only, and the waveform,
    the schedule and b are the same with or without it (ACQUISITION.md 3.6, 5.7)."""
    from dmipy_sim import Prescription, sequences
    from dmipy_sim.phantom import Grid
    p = Prescription(voxel_size_m=(1.5e-3,) * 3, matrix=(40, 40, 1))
    assert p.isocenter_m == (0.0, 0.0, 0.0) and p.axes == "RAS" and p.fov_m == (0.06, 0.06, 1.5e-3)
    np.testing.assert_allclose(p.origin_m, (-0.02925, -0.02925, 0.0))            # the FOV centred on the isocenter
    assert Prescription.from_dict(p.to_dict()) == p
    with pytest.raises(ValueError, match="axes"):
        Prescription(voxel_size_m=(1e-3,) * 3, matrix=(2, 2, 2), axes="RRS")
    with pytest.raises(TypeError):
        Prescription((0, 0, 0), (1e-3,) * 3, (2, 2, 2))
    seq = sequences.pgse([[1, 0, 0]], 0.010, 0.030, bvalues=[1e9], TE=0.060)
    assert seq.prescription is None
    seq_p = seq.with_prescription(p)
    assert seq_p.prescription == p and seq_p.G is seq.G and seq_p.rf == seq.rf and seq_p.b()[0] == seq.b()[0]
    with pytest.raises(TypeError):
        seq.with_prescription({"matrix": (1, 1, 1)})
    g = Grid.from_prescription(p)
    assert g.shape == (40, 40, 1) and g.axes == "RAS"
    np.testing.assert_allclose(g.origin_m, p.origin_m); np.testing.assert_allclose(g.isocenter_m, p.isocenter_m)


# ── a magnet's own gradient (dmipy-sim#285 item 3) ──────────────────────────────────────────────────
def _pgse_1e9():
    return d.pgse([[1.0, 0, 0]], 0.008, 0.030, bvalues=[1e9], TE=0.05, n_t=600)


def test_a_background_gradient_is_added_everywhere_and_remembered():
    """A magnet does not switch off, so its gradient is on through the pulses and the dead times that a
    builder guarantees are clear. The object therefore records what the magnet added, and the builder's
    guarantees are judged on what the builder laid out."""
    seq = _pgse_1e9()
    g = [1.4e-3, 0.0, 0.0]
    bg = seq.with_background_gradient(g)
    # G is stored float32, so the difference carries the storage's rounding, not the transform's
    np.testing.assert_allclose(np.asarray(bg.G) - np.asarray(seq.G),
                               np.broadcast_to(np.float32(g), seq.G.shape), rtol=1e-5, atol=1e-8)
    assert np.abs(np.asarray(bg.G)[..., 0]).min() > 0.0    # on at EVERY sample, dead time included
    assert np.abs(np.asarray(seq.G)[..., 0]).min() == 0.0  # where the built waveform is off
    np.testing.assert_allclose(bg.designed_gradient, seq.G, rtol=1e-5, atol=1e-8)
    assert bg.background_gradient == ((1.4e-3, 0.0, 0.0),)
    bg.validate()                                          # the builder's guarantees still hold of the design


def test_the_background_gradients_effect_on_b_is_a_cross_term():
    """The magnet's own b is negligible; what moves the b-value is its CROSS term with the pulsed gradient.
    So reversing the background reverses the effect, and the two straddle the asked-for b symmetrically --
    which is the ADC error a low-field magnet produces, and why it is signed per direction."""
    seq = _pgse_1e9()
    b0 = float(seq.b()[0])
    plus = float(seq.with_background_gradient([1.4e-3, 0, 0]).b()[0])
    minus = float(seq.with_background_gradient([-1.4e-3, 0, 0]).b()[0])
    perp = float(seq.with_background_gradient([0, 0, 1.4e-3]).b()[0])
    alone = float(seq.with_gradient(np.zeros_like(seq.G)).with_background_gradient([1.4e-3, 0, 0]).b()[0])

    assert plus > b0 > minus                                          # signed: it is a cross term
    assert abs((plus - b0) - (b0 - minus)) < 0.05 * (plus - b0)       # and very nearly antisymmetric
    np.testing.assert_allclose(0.5 * (plus + minus) - b0, alone, rtol=0.05)   # what is left is its own b
    assert alone < 1e-2 * b0                                          # which is negligible on its own
    assert abs(perp - b0) < 0.1 * (plus - b0)                         # perpendicular: almost nothing
    assert 0.05 < (plus - b0) / b0 < 0.10    # 1.4 mT/m on a Swoop: "up to 7 % of the diffusion gradient"


def test_the_declared_b_is_what_was_asked_and_b_is_what_is_played():
    """The encoding records the prescription at isocentre; `b()` reports the waveform that is actually
    played. Away from isocentre they differ, and that difference IS the measurement error."""
    seq = _pgse_1e9()
    bg = seq.with_background_gradient([1.4e-3, 0, 0])
    assert bg.encoding is not None and bg.encoding.bvalues[0] == seq.encoding.bvalues[0]
    assert not np.isclose(float(bg.b()[0]), float(bg.encoding.bvalues[0]), rtol=1e-3)


def test_a_background_gradient_is_refused_twice_over_and_in_the_wrong_shape():
    seq = _pgse_1e9()
    with pytest.raises(ValueError, match="one vector or one per measurement"):
        seq.with_background_gradient([[1e-3, 0, 0], [2e-3, 0, 0]])
    once = seq.with_background_gradient([1e-3, 0, 0])
    with pytest.raises(ValueError, match="already carries a background gradient"):
        once.with_background_gradient([1e-3, 0, 0])


def test_a_background_gradient_may_be_given_per_measurement():
    """Different voxels sit at different places in the bore, so an image asks for one vector per row."""
    two = d.pgse([[1.0, 0, 0], [1.0, 0, 0]], 0.008, 0.030, bvalues=[1e9, 1e9], TE=0.05, n_t=600)
    bg = two.with_background_gradient([[1.4e-3, 0, 0], [-1.4e-3, 0, 0]])
    b = bg.b()
    assert b[0] > float(two.b()[0]) > b[1]
    assert len(bg.background_gradient) == 2


# ── the gradient coils' own concomitant field (dmipy-sim#285 item 4) ────────────────────────────────
def test_the_concomitant_term_is_zero_at_isocentre_and_scales_as_one_over_B0():
    """Maxwell's equations make a gradient coil produce more than its z component. The extra field vanishes
    at isocentre and goes as 1/B0 to leading order, which is the whole reason it is a low-field problem and
    not a 3 T one. The term is the exact field magnitude's departure from ``B0 + B_n``, so the 1/B0 law
    holds up to the next order, ``(|B_perp| / B0)^2``: two per cent here, where the transverse field the
    89 mT/m gradient makes at 10 cm is a seventh of the 64 mT field."""
    seq = _pgse_1e9()
    np.testing.assert_allclose(seq.with_concomitant([0, 0, 0], 0.064).G, seq.G, atol=1e-12)

    off = lambda B0: np.abs(np.asarray(seq.with_concomitant([0, 0, 0.10], B0).G) - np.asarray(seq.G)).max()
    low, high = off(0.064), off(3.0)
    np.testing.assert_allclose(low / high, 3.0 / 0.064, rtol=0.02)      # 47x, the field ratio to leading order
    assert not np.isclose(low / high, 3.0 / 0.064, rtol=1e-3)           # and NOT exactly: the next order is real
    assert low > 0.4 * 24.4e-3     # at 64 mT and 10 cm it is half a Swoop's entire gradient ceiling


def test_the_concomitant_term_does_not_reverse_with_the_coils():
    """It is QUADRATIC in G to leading order, so reversing the gradient leaves it the same -- which is why a
    symmetric pair refocuses the pulsed gradient and not this, and why an unbalanced train does not refocus
    it at all. The exact magnitude adds an odd part: the field the gradient itself makes along B0 either
    adds to or subtracts from it, and ``|B|`` knows which. In the extra GRADIENT that part is
    ``-G |B_perp|^2 / B0^2`` to leading order, which is ``|B_perp| / B0`` of the even term -- eleven per cent
    here, where 89 mT/m at 8 cm makes a transverse field a ninth of the 64 mT static one. It is linear in
    ``G``, so it is an encoding-gradient rescale of ``(|B_perp| / B0)^2``, and a 180 refocuses it like the
    pulsed gradient."""
    seq = _pgse_1e9()
    flipped = seq.with_gradient(-np.asarray(seq.G))
    gc = np.asarray(seq.with_concomitant([0.02, 0, 0.08], 0.064).G) - np.asarray(seq.G)
    gc_flipped = np.asarray(flipped.with_concomitant([0.02, 0, 0.08], 0.064).G) - np.asarray(flipped.G)
    scale = np.abs(gc).max()
    odd = np.abs(gc - gc_flipped).max() / scale
    B_perp_over_B0 = 0.089 * 0.08 / 0.064
    assert odd < 1.5 * B_perp_over_B0, f"the odd part is {odd:.1%} of the term, more than |B_perp| / B0 allows"
    assert odd > 0.5 * B_perp_over_B0, "the odd part vanished: the term has been truncated back to its even leading order"


def test_the_two_magnet_terms_compose_and_are_recoverable():
    """A voxel off isocentre sees both: the magnet's own gradient and the coils' concomitant field. They add,
    and the builder's design is still recoverable from underneath both."""
    seq = _pgse_1e9()
    both = seq.with_background_gradient([1.4e-3, 0, 0]).with_concomitant([0, 0, 0.08], 0.064)
    np.testing.assert_allclose(both.designed_gradient, seq.G, rtol=1e-4, atol=1e-7)
    assert both.background_gradient is not None and both.concomitant["B0_T"] == 0.064
    both.validate()                                     # the builder's guarantees are about the design
    only_bg = seq.with_background_gradient([1.4e-3, 0, 0])
    only_cc = seq.with_concomitant([0, 0, 0.08], 0.064)
    np.testing.assert_allclose(np.asarray(both.G) - np.asarray(seq.G),
                               (np.asarray(only_bg.G) - np.asarray(seq.G))
                               + (np.asarray(only_cc.G) - np.asarray(seq.G)), rtol=1e-4, atol=1e-9)


def test_the_concomitant_term_is_refused_twice_over_and_at_a_nonsense_field():
    seq = _pgse_1e9()
    with pytest.raises(ValueError, match="B0_T must be positive"):
        seq.with_concomitant([0, 0, 0.1], 0.0)
    with pytest.raises(ValueError, match="one point or one per measurement"):
        seq.with_concomitant([[0, 0, 0.1], [0, 0, 0.2]], 0.064)
    once = seq.with_concomitant([0, 0, 0.1], 0.064)
    with pytest.raises(ValueError, match="already carries a concomitant term"):
        once.with_concomitant([0, 0, 0.1], 0.064)


# ── against the Swoop paper's reported numbers (dmipy-sim#285) ──────────────────────────────────────
def _swoop_protocol(directions):
    """The published protocol: b = 945 s/mm2, delta 35 ms, Delta 42 ms (Gholam 2025 / O'Halloran 2022)."""
    n = len(directions)
    return d.pgse(directions, 0.035, 0.042, bvalues=[945e6] * n, TE=0.090, n_t=900)


def test_the_background_gradient_reproduces_the_papers_ADC_error_at_8_cm():
    """Gholam 2025 corrects ADC errors of up to 16.1 % at 8 cm from isocentre, from a magnet gradient of up
    to 1.4 mT/m there. Driving the transform with their gradient must land on their error, and it must be the
    ALIGNED case that does: the error is a cross term, so it is largest when the two gradients are parallel
    and vanishes when they are perpendicular."""
    seq = _swoop_protocol([[1.0, 0, 0]])
    b0 = float(seq.b()[0])
    aligned = float(seq.with_background_gradient([1.4e-3, 0, 0]).b()[0]) / b0 - 1.0
    against = float(seq.with_background_gradient([-1.4e-3, 0, 0]).b()[0]) / b0 - 1.0
    across = float(seq.with_background_gradient([0, 1.4e-3, 0]).b()[0]) / b0 - 1.0

    assert 0.10 < aligned < 0.30, f"aligned error {aligned:.3f} is nowhere near the paper's 0.161"
    assert against < 0 < aligned and abs(abs(against) - aligned) < 0.3 * aligned
    assert abs(across) < 0.05 * aligned            # perpendicular: the cross term is gone
    # the paper's figure sits inside the range the directions span, which is what "up to 16.1 %" means
    assert against < 0.161 < aligned


def test_at_3_T_the_same_magnet_error_would_be_a_low_field_problem_only():
    """The background gradient is a property of the magnet, so it does not scale with B0 -- but a 3 T magnet
    is shimmed to parts per million and has no such gradient. The concomitant term DOES scale, and that one
    is the reason the same sequence is safe at 3 T and not at 64 mT."""
    seq = _swoop_protocol([[0.577, 0.577, 0.577]])
    b0 = float(seq.b()[0])
    at_64mT = float(seq.with_concomitant([0.0462, 0.0462, 0.0462], 0.064).b()[0]) / b0 - 1.0
    at_3T = float(seq.with_concomitant([0.0462, 0.0462, 0.0462], 3.0).b()[0]) / b0 - 1.0
    assert at_64mT > 20 * at_3T > 0.0
    # and on a SYMMETRIC spin echo it stays small: the 180 cancels most of a term the coils do not reverse
    assert at_64mT < 0.02, f"{at_64mT:.4f}: a symmetric PGSE should refocus most of the concomitant term"
