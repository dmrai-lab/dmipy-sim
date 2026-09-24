"""A diffusion-prepared RF train over a whole ODF phantom -- what the vector-Bloch route cannot do.

`replay_bloch` needs one rotation per slot, so it refuses an `odf_sh` phantom (dmipy-sim#338) and a brain
is `odf_sh`. Decomposing the train into microscopic gates removes the obstacle: each gate is an ordinary
phase sum, and the pose expansion has always carried those.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.phantom import Grid, Inert, ODF, PackSubstrate, Phantom
from dmipy_sim.replay import read_rpk
from dmipy_sim.replay.bank import build_replay_pack

SH = (6, 6, 2)


@pytest.fixture(scope="module")
def pack_path(tmp_path_factory):
    out = tmp_path_factory.mktemp("pk") / "wm.rpk"
    walk = d.simulate_trajectories(600, 2e-9, d.FreeDiffusion(), 0.14, 2.5e-4, seed=0, require_gpu=False)
    build_replay_pack(walk, id="test/wm", license="x", citation="x", K=10, out_path=str(out), segment_T=0.14)   # one 140 ms window
    return str(out)


@pytest.fixture(scope="module")
def brain(pack_path):
    grid = Grid(shape=SH, voxel_size_m=(2.5e-2,) * 3,
                origin_m=tuple(-0.5 * (n - 1) * 2.5e-2 for n in SH), isocenter_m=(0.0, 0.0, 0.0))
    c = np.zeros(SH + (45,), np.float32)
    c[..., 0] = 1.0 / np.sqrt(4 * np.pi)
    c[..., 3] = 0.3
    wm = PackSubstrate(pack_path, m0=0.7, name="wm")
    ph = Phantom.compose(grid, fractions={wm: np.ones(SH, np.float32)},
                         orientation={wm: ODF(c, basis="mrtrix3")}, remainder=Inert(name="bg"))
    return ph, read_rpk(pack_path), grid


def _train(n_echo=3, beta=150.0):
    return sequences.splice([[1.0, 0, 0]], 15e-3, 25e-3, n_echo, 10e-3, bvalues=[1e9],
                            TE_prep=80e-3, beta_deg=beta, n_t_per_echo=40)


def test_a_train_replays_over_an_odf_phantom_which_the_bloch_route_refuses(brain):
    """The point of the whole decomposition, stated as a test."""
    ph, pack, _grid = brain
    assert ph.mode == "odf_sh"
    with pytest.raises(ValueError, match="frames-mode"):
        ph.file.replay_bloch(_train(), packs={0: pack})
    S = ph.replay_train(_train(), packs={0: pack})
    assert S.shape[:3] == SH and np.all(np.isfinite(S))
    assert np.nanmean(np.abs(S)) > 0


def test_the_flip_angle_changes_the_signal_as_the_pathways_say(brain):
    """A reduced flip is not a scale factor: the pathways part and the echoes stop being equal. A route that
    merely attenuated would pass a weaker test than this one."""
    ph, pack, _g = brain
    perfect = np.abs(ph.replay_train(_train(beta=180.0), packs={0: pack}))
    reduced = np.abs(ph.replay_train(_train(beta=120.0), packs={0: pack}))
    assert np.nanmean(reduced) < np.nanmean(perfect)

    # and echo to echo, a perfect train is flat where a reduced one is not (no relaxation, no train gradient)
    flat = [np.nanmean(np.abs(ph.replay_train(_train(beta=180.0), echo=e, packs={0: pack}))) for e in range(3)]
    bent = [np.nanmean(np.abs(ph.replay_train(_train(beta=120.0), echo=e, packs={0: pack}))) for e in range(3)]
    assert np.ptp(flat) < 1e-6 * np.mean(flat)
    assert np.ptp(bent) > 1e-2 * np.mean(bent)


def test_a_transmit_map_costs_a_reweighting_not_a_reexpansion(brain):
    """The cost claim. The gate expansions do not depend on the transmit scale, so a map with many scales
    costs one state propagation each rather than one magnetisation propagation per voxel -- and binning
    bounds that count for a map a machine produced, which is smooth."""
    ph, pack, grid = brain
    smooth = np.linspace(0.8, 1.0, ph.n_voxels).reshape(SH)
    fine, coarse = {}, {}
    S_fine = ph.replay_train(_train(), transmit=smooth, transmit_tolerance=1e-2, packs={0: pack}, report=fine)
    S_coarse = ph.replay_train(_train(), transmit=smooth, transmit_tolerance=1e-1, packs={0: pack}, report=coarse)
    assert fine["n_scales"] > coarse["n_scales"]                 # binning is what bounds the count
    assert coarse["n_gates"] == fine["n_gates"]                  # and it does not touch the expansions
    rel = np.nanmax(np.abs(np.abs(S_fine) - np.abs(S_coarse))) / np.nanmax(np.abs(S_fine))
    assert rel < 0.2, f"a tenth of a flip angle moved the signal by {rel:.3f}"


def test_the_gate_count_is_set_by_the_preparation_not_the_train(brain):
    """A longer train does not cost more expansions -- which is what makes seventy echoes possible at all."""
    ph, pack, _g = brain
    counts = []
    for n_echo in (2, 4, 8):
        rep = {}
        ph.replay_train(_train(n_echo=n_echo), packs={0: pack}, report=rep)
        counts.append(rep["n_gates"])
    assert len(set(counts)) == 1, f"gate count moved with the train: {counts}"


def test_the_accelerated_route_agrees_with_the_plain_one(brain):
    """The GPU path is a different arrangement of the same contraction, so it must agree to the precision it
    works in -- float32 -- and not merely correlate."""
    jax = pytest.importorskip("jax")
    ph, pack, _g = brain
    smooth = np.linspace(0.85, 1.0, ph.n_voxels).reshape(SH)
    a = ph.replay_train(_train(), transmit=smooth, packs={0: pack}, jax=False)
    b = ph.replay_train(_train(), transmit=smooth, packs={0: pack}, jax=True)
    rel = np.nanmax(np.abs(np.abs(a) - np.abs(b))) / max(np.nanmax(np.abs(a)), 1e-30)
    assert rel < 1e-5, f"the two routes differ by {rel:.2e}"


# ── a drifting magnet (dmipy-sim#285 item 5) ────────────────────────────────────────────────────────
def test_off_resonance_is_gated_like_the_gradient_not_applied_at_the_end(brain):
    """The correctness point. A field offset accrues only while magnetisation is TRANSVERSE, so it is gated
    by the same F+/F-/Z pattern the gradient is. The refocused pathway comes back to zero signed transverse
    time and so refocuses an offset exactly; a pathway that slept through an interval does not. A train
    therefore has no single coherence gate, and applying one factor at the end would be wrong for every
    pathway but one."""
    from dmipy_sim.replay.pathways import train_response
    _ph, pack, _g = brain
    tr = train_response(pack, _train(), keep=(8, 0))
    tau = tr.tau
    assert any(abs(v) < 1e-9 for v in tau.values()), "no refocused pathway"
    assert any(abs(v) > 1e-3 for v in tau.values()), "no pathway that sleeps"
    # the one that never leaves the transverse plane accrues the whole preparation
    assert max(tau.values()) == pytest.approx(2 * max(v for v in tau.values() if 0 < v < max(tau.values())),
                                              rel=1e-6)


def test_a_spin_echo_refocuses_a_drifting_magnet_exactly(brain):
    """One pathway survives a perfect 180, its signed transverse time is zero, so a UNIFORM offset leaves
    both magnitude and phase untouched however far the magnet has drifted."""
    from dmipy_sim.replay.pathways import train_response
    from dmipy_sim.replay import so3
    _ph, pack, _g = brain
    se = _train(n_echo=1, beta=180.0)
    tr = train_response(pack, se, keep=(8, 0))
    A = so3.so3_design(8, np.eye(3)[None], 0)
    base = complex((A @ tr.at(1.0, echo=-1, dw=0.0).coeffs.T).reshape(-1)[0])
    for hz in (100.0, 1000.0, 2725.0):                       # far beyond any drift this magnet reaches
        s = complex((A @ tr.at(1.0, echo=-1, dw=2 * np.pi * hz).coeffs.T).reshape(-1)[0])
        assert abs(s) / abs(base) == pytest.approx(1.0, abs=1e-9)
        assert abs(np.angle(s / base)) < 1e-9


def test_a_refocusing_train_is_exactly_insensitive_to_a_uniform_offset(brain):
    """#285 filed drift as low priority on the assumption that "the SPLICE magnitude combination is
    insensitive to a slow phase ramp". It is more than insensitive: with hard pulses it is EXACT, and the
    reason is structural rather than a cancellation that happens to be good.

    A uniform offset multiplies every transverse coefficient by the same factor, so it can rotate the
    signal but never redistribute it between coherence orders. The train's echo is carried by the refocused
    pathway, whose signed transverse time is zero, so there is no second pathway to interfere with and the
    magnitude is untouched at any offset.

    An earlier version of this test recorded a bounded +-1.5 % envelope. That was an artefact: the
    off-resonance operator conjugated by the sign of the dephasing index, which made it non-uniform and let
    it move amplitude between orders -- something no static field offset can do. See
    tests/test_epg_off_resonance.py, which checks the operator against an isochromat sum.

    What this does NOT say is that a real magnet's drift is harmless. It says a UNIFORM offset is refocused
    under HARD pulses. Finite pulses excite about a tilted effective field when off-resonant, so the flip
    angle itself becomes offset-dependent -- a channel this route does not model."""
    from dmipy_sim.replay.pathways import train_response
    from dmipy_sim.replay import so3
    _ph, pack, _g = brain
    tr = train_response(pack, _train(n_echo=4, beta=150.0), keep=(8, 0))
    A = so3.so3_design(8, np.eye(3)[None], 0)

    def at(hz):
        return complex((A @ tr.at(1.0, echo=-1, dw=2 * np.pi * hz).coeffs.T).reshape(-1)[0])

    base = at(0.0)
    for hz in (13.0, 100.0, 545.0, 1400.0, 2725.0):
        assert abs(at(hz)) / abs(base) == pytest.approx(1.0, abs=1e-9)
    # the echo really is carried by the pathway that refocuses, which is why
    live = [g for g, w in tr.weights(1.0).items() if abs(w[-1]) > 1e-6]
    assert len(live) == 1 and tr.tau[live[0]] == pytest.approx(0.0, abs=1e-12)


def _free_water_brain(T2=None):
    """A phantom of free water alone on the small grid: the closed form's train has no pose to expand."""
    from dmipy_sim.phantom import FreeWater
    from dmipy_sim.spec import Tissue
    grid = Grid(shape=SH, voxel_size_m=(2.5e-2,) * 3,
                origin_m=tuple(-0.5 * (n - 1) * 2.5e-2 for n in SH), isocenter_m=(0.0, 0.0, 0.0))
    csf = FreeWater(m0=1.0, name="csf", tissue=Tissue(D=3e-9, T2=T2))
    return Phantom.compose(grid, fractions={csf: np.ones(SH, np.float32)}, orientation={}, remainder=Inert(name="bg"))


def test_a_closed_form_under_a_train_is_the_pack_route_at_b_zero(brain):
    """With no gradient every gate's factor is one, so the closed form's pathway sum must equal the pack's to
    the arithmetic, at every echo and transmit scale -- and below one where the crushers have removed
    pathways, which a single static spin under the same pulses would keep."""
    ph_pack, pack, _grid = brain
    ph = _free_water_brain()
    for beta, kappa in ((180.0, 1.0), (150.0, 1.0), (150.0, 0.8)):
        train = sequences.splice([[1.0, 0, 0]], 15e-3, 25e-3, 3, 10e-3, bvalues=[0.0],
                                 TE_prep=80e-3, beta_deg=beta, n_t_per_echo=40)
        for echo in (0, 1, 2):
            got = np.nanmean(ph.replay_train(train, echo=echo, transmit=kappa, complex_signal=True)[..., 0])
            want = np.nanmean(ph_pack.replay_train(train, echo=echo, transmit=kappa, packs={0: pack}, complex_signal=True)[..., 0]) / 0.7
            assert abs(got - want) < 1e-6, (beta, kappa, echo, got, want)
            if beta < 180.0 and echo > 0:
                assert abs(got) < 1.0 - 1e-3


def test_a_closed_form_takes_each_pathway_at_its_own_b_and_the_pack_agrees(brain):
    """Under a diffusion preparation the stimulated pathways carry their own b: free water's amplitude must fall
    below its b = 0 value by the pathways' own Gaussian factors, and a free-water pack under the same train
    must agree with the closed form to its Monte-Carlo scatter."""
    from dmipy_sim.phantom import FreeWater
    from dmipy_sim.spec import Tissue
    ph_pack, pack, grid = brain
    D = float(pack.meta["walk_params"]["diffusivity"])
    csf = FreeWater(m0=0.7, name="fw", tissue=Tissue(D=D))
    ph_form = Phantom.compose(grid, fractions={csf: np.ones(SH, np.float32)}, orientation={}, remainder=Inert(name="bg"))
    for beta in (180.0, 120.0):
        train = _train(beta=beta)
        S_form = np.abs(np.nanmean(ph_form.replay_train(train, echo=1)[..., 0]))
        S_pack = np.abs(np.nanmean(ph_pack.replay_train(train, echo=1, packs={0: pack})[..., 0]))
        assert 0 < S_form < 0.7                                                # attenuated below m0
        if beta == 180.0:                                                      # one pathway: the preparation's own b
            assert abs(S_form - 0.7 * np.exp(-float(train.b()[0]) * D)) < 1e-6 * S_form
        # the fixture pack is 600 walkers at K = 10 over 56 saves: its codec error at b = 1000 s/mm^2 is
        # the pack's, not the closed form's, and is what the tolerance here allows for
        assert abs(S_form - S_pack) < 0.12 * S_pack, (beta, S_form, S_pack)


def test_a_mixed_phantom_composes_a_pack_and_a_closed_form():
    """White matter from a pack beside cerebrospinal fluid in closed form, in one train replay: the voxel is
    the fraction-weighted sum of the two routes, and neither is refused or dropped."""
    from dmipy_sim.phantom import FreeWater
    from dmipy_sim.spec import Tissue
    import tempfile, os
    out = os.path.join(tempfile.mkdtemp(), "wm.rpk")
    walk = d.simulate_trajectories(300, 2e-9, d.FreeDiffusion(), 0.14, 2.5e-4, seed=1, require_gpu=False)
    build_replay_pack(walk, id="test/wm2", license="x", citation="x", K=10, out_path=out, segment_T=0.14)   # one 140 ms window
    pack = read_rpk(out)
    grid = Grid(shape=SH, voxel_size_m=(2.5e-2,) * 3,
                origin_m=tuple(-0.5 * (n - 1) * 2.5e-2 for n in SH), isocenter_m=(0.0, 0.0, 0.0))
    c = np.zeros(SH + (45,), np.float32); c[..., 0] = 1.0 / np.sqrt(4 * np.pi)
    wm = PackSubstrate(out, m0=0.7, name="wm"); csf = FreeWater(m0=1.0, name="csf", tissue=Tissue(D=3e-9))
    f_wm = np.full(SH, 0.6, np.float32); f_csf = np.full(SH, 0.4, np.float32)
    both = Phantom.compose(grid, fractions={wm: f_wm, csf: f_csf}, orientation={wm: ODF(c, basis="mrtrix3")}, remainder=Inert(name="bg"))
    only_wm = Phantom.compose(grid, fractions={wm: f_wm}, orientation={wm: ODF(c, basis="mrtrix3")}, remainder=Inert(name="bg"))
    only_csf = Phantom.compose(grid, fractions={csf: f_csf}, orientation={}, remainder=Inert(name="bg"))
    train = _train(beta=150.0)
    rep = {}
    S = both.replay_train(train, packs={0: pack}, complex_signal=True, report=rep)
    S1 = only_wm.replay_train(train, packs={0: pack}, complex_signal=True)
    S2 = only_csf.replay_train(train, complex_signal=True)
    assert rep["n_closed_forms"] == 1 and rep["n_gates"] >= 1
    np.testing.assert_allclose(S, S1 + S2, rtol=1e-6, atol=1e-9)


def test_an_oriented_closed_form_is_refused_under_a_train():
    from dmipy_sim.replay.pathways import closed_form_train

    class Stick:
        oriented = True

        def response(self, seq, pose=None):
            return np.ones(seq.n_meas)

    with pytest.raises(ValueError, match="oriented closed form"):
        closed_form_train(Stick(), _train())


def test_a_band_is_read_wider_by_zeros_and_narrower_by_truncation():
    """Two packs whose responses were projected at different bands compose at one: the narrower is zero-extended
    (exact, since a projection at the band a response needs has nothing above it), the wider truncated; a band
    narrower in one index and wider in the other is refused."""
    from dmipy_sim.replay.so3 import rebanded, so3_index, extend_coeffs, truncate_coeffs
    rng = np.random.default_rng(0)
    c2 = rng.normal(size=(3, len(so3_index(2, 0))))
    wide = rebanded(c2, 2, 0, 8, 0)
    assert wide.shape == (3, len(so3_index(8, 0)))
    np.testing.assert_array_equal(truncate_coeffs(wide, 8, 0, 2, 0), c2)      # the round trip is exact
    assert np.count_nonzero(wide) == np.count_nonzero(c2)                    # and nothing was invented above
    c8 = rng.normal(size=(3, len(so3_index(8, 0))))
    np.testing.assert_array_equal(rebanded(c8, 8, 0, 2, 0), truncate_coeffs(c8, 8, 0, 2, 0))
    np.testing.assert_array_equal(extend_coeffs(c2, 2, 0, 2, 0), c2)
    with pytest.raises(ValueError, match="narrower in one"):
        rebanded(rng.normal(size=(1, len(so3_index(4, 2)))), 4, 2, 8, 0)


def test_a_b_zero_train_composes_packs_of_different_bands(brain):
    """The case that was refused: a b = 0 preparation, whose gate a pack expands at order two, beside a
    substrate expanded wider -- here the same pack asked for the distribution's order eight -- replays and equals
    the single-pack replay, since the zeros above order two are exact."""
    ph, pack, grid = brain
    train = sequences.splice([[1.0, 0, 0]], 15e-3, 25e-3, 3, 10e-3, bvalues=[0.0],
                             TE_prep=80e-3, beta_deg=150.0, n_t_per_echo=40)
    c = np.zeros(SH + (45,), np.float32); c[..., 0] = 1.0 / np.sqrt(4 * np.pi); c[..., 3] = 0.3
    wm = PackSubstrate(pack, m0=0.7, name="wm"); wm2 = PackSubstrate(pack, m0=0.7, name="wm2")
    two = Phantom.compose(grid, fractions={wm: np.full(SH, 0.5, np.float32), wm2: np.full(SH, 0.5, np.float32)},
                          orientation={wm: ODF(c, basis="mrtrix3"), wm2: ODF(c, basis="mrtrix3")}, remainder=Inert(name="bg"))
    S2 = two.replay_train(train, echo=1, packs={0: pack, 1: pack}, complex_signal=True)
    S1 = ph.replay_train(train, echo=1, packs={0: pack}, complex_signal=True)
    np.testing.assert_allclose(S2, S1, rtol=1e-9, atol=1e-12)


def test_a_pathway_gated_waveform_carries_its_sign_as_its_coherence():
    """The gated copy of a waveform has no RF schedule: its pathway's sign is what the static field is folded
    with and what the relaxation follows -- transverse where |sign| = 1, stored where 0 -- and a closed form reads
    the transverse time as its echo time and the stored time as its mixing time."""
    from dataclasses import replace
    from dmipy_sim.phantom.substrates import _echo_time
    from dmipy_sim.replay._replay_kernel import field_gate
    from dmipy_sim.replay.pathways import gate_sign, sequence_events
    train = _train(n_echo=2, beta=150.0)
    n_t, dt = train.n_t, float(train.dt)
    prep, tr, on, ro, edges = sequence_events(train)
    # a stored-then-recalled pathway: +1 over the first gradient, 0 while stored, -1 after the recall; and over
    # the train the refocused reading, flipped at every pulse
    kinds = ("F+", "Z", "F-")
    s = gate_sign(kinds[:len(edges) - 1] if len(edges) - 1 <= 3 else kinds + ("F-",) * (len(edges) - 4), edges, n_t, dt, train=tr)
    assert set(np.unique(s)) <= {-1.0, 0.0, 1.0} and (s == 0).any() and (s == -1).any() and (s == 1).any()
    w = replace(train, G=np.zeros_like(np.asarray(train.G)), rf=None, gate=s.astype(np.float32), family="waveform",
                crusher=None, readout=None, voxel_scale="declared", encoding=None)
    np.testing.assert_array_equal(w.chi_perp, np.abs(s))
    assert abs(w.TM - dt * np.sum(s[:-1] == 0)) < 1e-12
    assert abs(_echo_time(w) - (w.T - w.TM)) < 1e-12
    # the field gate is the sign itself, integrated: zero net for a balanced sign, the sign's sum otherwise
    fg = field_gate(w, n_t, dt)
    assert abs(fg.sum() - s[:-1].sum()) < 1e-6
    with pytest.raises(ValueError, match="-1, 0 or 1"):
        replace(w, gate=np.full(n_t, 0.5, np.float32))
    with pytest.raises(ValueError, match="one or the other"):
        replace(train, gate=s.astype(np.float32))


@pytest.fixture(scope="module")
def field_pack(tmp_path_factory):
    """A sheathed axon alone in a wide cell with its field basis: the pack a static field needs (C3 path)."""
    from dmipy_sim.fields.susceptibility_field import field_grid_of
    D0 = 2.0e-9
    g = d.PackedMyelinatedCylinders([1.0e-6], 0.7, [[0.0, 0.0]], 30e-6, N_max=2, D_intra=D0, D_extra=D0)
    walk = d.simulate_trajectories(2000, D0, g, 6e-3, 3e-4, seed=0, require_gpu=False)
    out = tmp_path_factory.mktemp("pk") / "sheathed.rpk"
    build_replay_pack(walk, id="test/sheathed", license="x", citation="x", K=8, out_path=str(out),
                      envelope=dict(bvals=[0.0, 1e8], dirs=[[0, 0, 1], [1, 0, 0]], delta_frac=0.2, Delta_frac=0.5,
                                    ogse_periods=[1], shortd_b=1e8, shortd_deltas_frac=[0.2]),
                      field=field_grid_of(g, res=0.2e-6), susc_path_K=16)
    return str(out)


def test_a_spin_echo_under_a_field_refocuses_on_the_train_route_as_on_the_phase_sum(field_pack):
    """A spin echo at b = 0 in a field is one pathway with the sign (+1, -1): the train route must return what the
    phase-sum route returns for the same sequence, which it did not while the gated copy carried no sign and the
    field went unrefocused (a T2*-like loss of a tenth at 7 T)."""
    from dmipy_sim.spec import Tissue
    pack = read_rpk(field_pack)
    grid = Grid(shape=SH, voxel_size_m=(2.5e-2,) * 3,
                origin_m=tuple(-0.5 * (n - 1) * 2.5e-2 for n in SH), isocenter_m=(0.0, 0.0, 0.0))
    c = np.zeros(SH + (45,), np.float32); c[..., 0] = 1.0 / np.sqrt(4 * np.pi); c[..., 3] = 0.3
    wm = PackSubstrate(field_pack, m0=1.0, name="wm", tissue=Tissue(chi_iso=-1e-5, chi_aniso=0.0))   # a large chi: the fixture walk is 6 ms
    ph = Phantom.compose(grid, fractions={wm: np.ones(SH, np.float32)}, orientation={wm: ODF(c, basis="mrtrix3")}, remainder=Inert(name="bg"))
    se = sequences.pgse([[0.0, 0.0, 1.0]], 1e-3, 3e-3, bvalues=[0.0], TE=6e-3, n_t=60)
    for B0 in (None, 7.0):                                   # in magnitude: the pathway carries the pulses' phase, the phase sum does not
        S_sum = np.abs(ph.replay(se, scanner=B0, packs={0: pack}, complex_signal=True)[..., 0])
        S_train = np.abs(ph.replay_train(se, scanner=B0, packs={0: pack}, complex_signal=True)[..., 0])
        np.testing.assert_allclose(S_train, S_sum, rtol=1e-6, atol=1e-9)
    # and the field does something at 7 T on a gradient echo, so the agreement is not trivial
    gre = sequences.gre(6e-3, n_t=60)
    S_none = ph.replay(gre, scanner=None, packs={0: pack})[..., 0]
    S_7 = ph.replay(gre, scanner=7.0, packs={0: pack})[..., 0]
    assert np.nanmax(np.abs(S_7 / S_none - 1.0)) > 1e-3
