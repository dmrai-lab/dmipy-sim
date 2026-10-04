"""``ReplayPack.resegment`` (#525): a one-window pack stored in windows of a shorter duration replays as its parent
on every tier and every route, to the resegmented pack's certified floor, whatever window an acquisition spans; a
prefix of its windows is a range of it; a pose expansion reads the windows an acquisition reaches and no more."""
import time

import numpy as np
import numpy.testing as npt
import pytest

import dmipy_sim as d
from dmipy_sim import build_replay_pack, sequences
from dmipy_sim.replay import ReplayPack, read_rpk
from dmipy_sim.replay import bank
from dmipy_sim.replay.replay import _duration
from dmipy_sim.spec.tissue import Tissue
from tests.test_bank import _lean_env, _susc_master, _sample_susc
from tests.test_pack_segments import C2_DT, C2_N_T, _c2_slab_master

N_T, DT = 197, 5e-4                      # 98 ms of walk: four windows of 49 steps (24.5 ms)
T = (N_T - 1) * DT
TIS = Tissue(T2={"extra": 0.05, "intra": 0.02}, T1={"extra": 1.0, "intra": 0.5}, rho2=2e-5, chi_iso=1e-7)
KNOBS = {"bare": dict(), "relaxation": dict(tissue=Tissue(T2={"extra": 0.05, "intra": 0.02}, T1={"extra": 1.0, "intra": 0.5})),
         "contact": dict(tissue=Tissue(rho2=2e-5)), "field": dict(tissue=Tissue(chi_iso=1e-7), scanner=3.0),
         "all": dict(tissue=TIS, scanner=3.0)}


def _master(n_t=N_T):
    """The slab walk with C1 (a third of the walkers cross into pool 1 in the third window), C2 and the C3 grid."""
    m = _susc_master()
    for k in ("traj", "comp", "dlog_b"):
        m[k] = np.asarray(m[k])[:, :n_t]
    m["T_max"] = (n_t - 1) * DT
    m["dlog_b"] = -m["dlog_b"]                       # a contact lowers the weight: the pack's sign convention
    comp = np.zeros((m["traj"].shape[0], n_t), np.int8); comp[:1000, 120:] = 1
    m["comp"] = comp
    m["substrate"] = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6).spec.to_dict()   # names for the two pools the walk labels
    return _sample_susc(m)         # sampled after the slice to n_t, so the samples match the sliced trajectory


def _parent(n_t=N_T):
    """The walk as ONE window, lossy in the positions (K = 48 over 98 ms), the contact bridge and the path channel."""
    env = dict(_lean_env(), B0_list=[3.0], theta_deg=[0, 90], max_refocus_pulses=8)
    return build_replay_pack(_master(n_t), id="t/parent", license="CC-BY-4.0", citation="test", envelope=env, K=48,
                             blt_temporal_K=8, susc_path_K=32, susc_path_bits=16, segment_T=(n_t - 1) * DT)


@pytest.fixture(scope="module")
def packs(tmp_path_factory):
    parent = _parent()
    p = tmp_path_factory.mktemp("reseg")
    parent.save(p / "parent.rpk")
    parent = read_rpk(str(p / "parent.rpk"))
    seg = parent.resegment(T / 4, out_path=p / "seg.rpk", id="t/seg")
    return parent, seg, p


def _dirs(n, seed=0):
    u = np.random.default_rng(seed).normal(size=(n, 3))
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def _waveforms():
    """300 diffusion measurements and a train: PGSE inside window 0, PGSE across two windows, PGSTE across three,
    an eight-echo train across all four."""
    dirs = _dirs(50)
    two = np.concatenate([dirs, dirs]); b = lambda b1, b2: [b1] * 50 + [b2] * 50
    return {"pgse_1": sequences.pgse(two, 4e-3, 10e-3, bvalues=b(0.5e9, 1e9), TE=20e-3, n_t=400, slew_rate=np.inf),
            "pgse_2": sequences.pgse(two, 8e-3, 25e-3, bvalues=b(1e9, 2e9), TE=45e-3, n_t=900, slew_rate=np.inf),
            "pgste_3": sequences.pgste(two, 5e-3, 50e-3, bvalues=b(0.5e9, 1e9), TE=70e-3, n_t=1400, slew_rate=np.inf),
            "train_4": sequences.cpmg(8, 12e-3, gradient_directions=[[1.0, 0, 0]], bvalues=[0.0], n_t_per_echo=40)}


def test_the_windows_and_their_record(packs):
    parent, seg, p = packs
    assert parent.n_segments == 1 and seg.n_segments == 4
    assert seg.segments["n_t"] == 50 and seg.segments["T"] == pytest.approx(T / 4) and seg.n_t == N_T
    assert seg.K == 12                                                       # round(48 x 49 / 196): the bands per second kept
    assert seg.temporal_bandwidth_hz == pytest.approx(parent.temporal_bandwidth_hz)
    r = seg.meta["provenance"]["resegmented"]
    assert r["parent_id"] == "t/parent" and r["n_seg"] == 4 and r["saves_dropped"] == 0 and r["T_seg"] == pytest.approx(T / 4)
    assert len(r["parent_sha256"]) == 64 and r["K"] == 12 and r["parent_K"] == 48
    f = seg.fidelity
    assert f["certified"] == "bounded" and len(f["segments"]) == 4 and f["within_2x_floor"]
    for w in f["segments"]:
        assert w["err_max"] <= 2.0 * w["floor_max"] and "err_susc_path" in w and "err_surface" in w
    ch = seg.meta["compression"]["channels"]
    assert ch["susceptibility_grid"] == parent.meta["compression"]["channels"]["susceptibility_grid"]
    assert ch["compartment"]["columns"][0]["kind"] != "static"               # the walk crosses: runs in every window
    per_window = {k[3:] for k in seg.arrays if k.startswith("s3/")}
    assert {"pos_x", "blt_bridge_dst", "comp_rle_vals", "susc_path_dct"} <= per_window
    np.testing.assert_array_equal(seg.spin_weights, parent.spin_weights)
    back = read_rpk(str(p / "seg.rpk"))
    assert set(back.arrays) == set(seg.arrays)
    for k in seg.arrays:
        npt.assert_array_equal(np.asarray(back.arrays[k]), np.asarray(seg.arrays[k]), err_msg=k)


def test_the_boundary_saves_are_the_parents(packs):
    """Every window's endpoints are the parent's decoded saves at its ends, exactly (the bridge holds them)."""
    parent, seg, _ = packs
    X = parent.positions()
    for i in range(4):
        C = seg.segment(i).position_coeffs
        npt.assert_allclose(C[:, 0, :], X[:, 49 * i, :], rtol=0, atol=1e-6 * np.abs(X).max())
        npt.assert_allclose(C[:, 0, :] + C[:, 1, :], X[:, 49 * (i + 1), :], rtol=0, atol=1e-6 * np.abs(X).max())


@pytest.mark.parametrize("name", ["pgse_1", "pgse_2", "pgste_3", "train_4"])
@pytest.mark.parametrize("tier", list(KNOBS))
def test_the_windows_replay_as_the_parent(packs, name, tier):
    parent, seg, _ = packs
    seq = _waveforms()[name]
    a = parent.replay(seq, complex_signal=True, **KNOBS[tier])
    b = seg.replay(seq, complex_signal=True, **KNOBS[tier])
    assert np.abs(a - b).max() <= seg.fidelity["floor_max"]


@pytest.mark.parametrize("name", ["pgse_1", "pgse_2", "pgste_3"])
def test_the_pose_expansion_is_the_parents(packs, name):
    parent, seg, _ = packs
    seq = _waveforms()[name]
    rp = parent.pose_response(seq, tissue=TIS, scanner=3.0, keep=(8, 0))
    rs = seg.pose_response(seq, tissue=TIS, scanner=3.0, keep=(8, 0))
    floor = seg.fidelity["floor_max"]
    assert np.abs(rp.coeffs - rs.coeffs).max() <= floor
    from scipy.spatial.transform import Rotation
    for R in Rotation.random(4, random_state=1).as_matrix():
        assert np.abs(rp.at(R) - rs.at(R)).max() <= floor


def test_a_one_window_acquisition_reads_window_zero_alone(packs):
    """The windows past the acquisition are not read: blanked, the pose expansion and the replay are unchanged."""
    parent, seg, _ = packs
    seq = _waveforms()["pgse_1"]
    blind = ReplayPack({k: (np.full_like(v, np.nan) if "/" in k and np.issubdtype(np.asarray(v).dtype, np.floating) else v)
                        for k, v in seg.arrays.items()}, seg.meta)
    npt.assert_array_equal(blind.replay(seq, tissue=TIS, scanner=3.0, complex_signal=True),
                           seg.replay(seq, tissue=TIS, scanner=3.0, complex_signal=True))
    npt.assert_array_equal(blind.pose_response(seq, tissue=TIS, scanner=3.0, keep=(8, 0)).coeffs,
                           seg.pose_response(seq, tissue=TIS, scanner=3.0, keep=(8, 0)).coeffs)


def test_a_truncated_resegmented_pack_is_its_first_window(packs, tmp_path):
    parent, seg, _ = packs
    one = seg.truncate(1, out_path=tmp_path / "one.rpk")
    assert one.n_segments == 1 and one.n_t == 50
    seq = _waveforms()["pgse_1"]
    for knobs in KNOBS.values():
        npt.assert_array_equal(one.replay(seq, complex_signal=True, **knobs), seg.replay(seq, complex_signal=True, **knobs))
    npt.assert_array_equal(one.pose_response(seq, tissue=TIS, scanner=3.0, keep=(8, 0)).coeffs,
                           seg.pose_response(seq, tissue=TIS, scanner=3.0, keep=(8, 0)).coeffs)


def test_the_trailing_remainder_is_dropped_within_its_bound_and_refused_beyond(tmp_path):
    """197 steps have no divisor near 49: one save is dropped at a 1 % bound and the walk refused at 1e-3."""
    parent = _parent(n_t=N_T + 1)
    with pytest.raises(ValueError, match="drops at most"):
        parent.resegment(T / 4, out_path=tmp_path / "x.rpk")
    seg = parent.resegment(T / 4, out_path=tmp_path / "seg.rpk", max_dropped_fraction=1e-2)
    assert seg.n_segments == 4 and seg.n_t == N_T and seg.meta["provenance"]["resegmented"]["saves_dropped"] == 1
    seq = _waveforms()["pgse_2"]
    a = parent.replay(seq, complex_signal=True, **KNOBS["all"])
    assert np.abs(a - seg.replay(seq, complex_signal=True, **KNOBS["all"])).max() <= seg.fidelity["floor_max"]
    with pytest.raises(ValueError, match="already stores"):
        seg.resegment(T / 8, out_path=tmp_path / "y.rpk")


def test_the_channels_decoded_on_a_window_are_the_whole_ones_cut():
    """Positions, the contact bridge and the path series evaluated at saves ``start .. stop - 1`` equal the whole
    decode cut there."""
    from dmipy_sim.replay import compression as cx
    from dmipy_sim.replay.bank import susc_path_decode
    rng = np.random.default_rng(8)
    n_w, n_t = 300, 240
    traj = np.cumsum(rng.normal(size=(n_w, n_t, 3)) * 1e-7, axis=1)
    arrays, meta, _ = cx.encode(traj, "bridge_dst", 24, device="numpy")
    C = cx.read_position_coeffs(arrays, dtype=np.float64)
    ref = cx._bridge_positions(C, n_t)
    dlog = -(rng.exponential(1e-3, size=(n_w, n_t)) * (rng.uniform(size=(n_w, n_t)) < 0.2))
    ba, bm = cx.encode_boundary_bridge(dlog, K=12, device="numpy")
    whole = cx.decode_boundary_bridge(ba, bm)
    Cp = rng.normal(size=(n_w, 7, 9)).astype(np.float32)
    pa, pm = {"susc_path_dct": Cp}, dict(n_t=n_t, channels=["iso_local", "iso_P_xx", "iso_P_yy", "iso_P_zz", "iso_P_xy", "iso_P_xz", "iso_P_yz"])
    pwhole, _ = susc_path_decode(pa, pm)
    for start, stop in ((0, 60), (59, 120), (120, 240), (1, 2), (239, 240)):
        npt.assert_allclose(cx.decode_saves(C, n_t, start, stop, device="numpy"), ref[:, start:stop], atol=1e-6 * np.abs(ref).max())
        part = cx.decode_boundary_bridge(ba, bm, start=start, n_cut=stop)
        assert part.shape == (n_w, stop - start)
        npt.assert_allclose(part, whole[:, start:stop], atol=1e-6 * np.abs(whole).max())
        npt.assert_allclose(susc_path_decode(pa, pm, start=start, n_cut=stop)[0], pwhole[:, :, start:stop], atol=1e-9)


def _timing_packs(tmp_path):
    """A free walk long in saves and wide in band (4001 saves, K = 2000), and the same walk in four windows: the
    cost of a pose expansion is then the band contraction and the saves the acquisition spans."""
    rng = np.random.default_rng(2)
    n_w, n_t, dt, D = 4000, 4001, 2.5e-5, 2e-9
    traj = np.cumsum(rng.normal(0.0, np.sqrt(2 * D * dt), size=(n_w, n_t, 3)), axis=1)
    m = dict(traj=traj, dt_traj=dt, T_max=(n_t - 1) * dt, w=np.ones(n_w), D_intra=D, n_walkers=n_w, seed=0)
    env = dict(_lean_env(), ogse_periods=[], shortd_deltas_frac=[])
    parent = build_replay_pack(m, id="t/long", license="x", citation="x", envelope=env, K=2000, segment_T=(n_t - 1) * dt)
    parent.save(tmp_path / "long.rpk")
    parent = read_rpk(str(tmp_path / "long.rpk"))
    return parent, parent.resegment(0.025, out_path=tmp_path / "long_seg.rpk", envelope=env)


def test_a_pose_expansion_costs_the_saves_it_spans(tmp_path):
    """The one-window acquisition on the resegmented pack reads one window of 1001 saves and 500 bands where the
    parent reads its whole walk of 4001 saves and 2000 bands, and the expansion is the parent's to the floor. Its
    time is held to a loose ordering only, the best of five on a shared machine (measured here between 0.46 and 0.82
    of the parent's under load; the ratio on the 1 s CACTUS pack is in #525)."""
    parent, seg = _timing_packs(tmp_path)
    assert seg.n_segments == 4 and seg.K == 500
    seq = sequences.pgse(_dirs(30), 4e-3, 10e-3, bvalues=[1e9] * 30, TE=20e-3, n_t=800, slew_rate=np.inf)
    spanned = lambda pk: [(int(w.K), int(n_s)) for w, _, n_s in pk._prepare(seq, tissue=None, scanner=None, orientation=None,
                                                                            compartment=None)["windows"]]
    assert spanned(parent) == [(2000, 4001)] and spanned(seg) == [(500, 1001)]

    def best(pk):
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); r = pk.pose_response(seq, keep=(8, 0)); ts.append(time.perf_counter() - t0)
        return min(ts), r

    t_parent, rp = best(parent)
    t_seg, rs = best(seg)
    assert np.abs(rp.coeffs - rs.coeffs).max() <= seg.fidelity["floor_max"]
    assert t_seg < t_parent, (t_seg, t_parent)


def test_the_contact_envelope_is_stated_and_held(packs):
    """The pack states the largest rho2 / D its contact tier serves (RPK.md 8.7); a relaxivity inside replays as the
    parent within the floor, one beyond is refused by name on every route."""
    parent, seg, _ = packs
    top = seg.rho2_over_D_max
    assert top is not None and top > 0 and seg.meta["replay_envelope"]["tissue"]["rho2_over_D_max"] == top
    r = seg.meta["provenance"]["surface_envelope_restated"]
    assert r["rho2_over_D_max"] == top == min(r["per_window"]) and r["reference"] == "t/parent"
    for f in seg.fidelity["segments"]:
        assert f["surface_rho2_over_D_max"] == top and f["err_surface"] <= 2.0 * f["floor_surface"]
    D = seg.diffusivity
    seq = _waveforms()["pgse_2"]
    inside = Tissue(rho2=0.9 * top * D)
    a = parent.replay(seq, complex_signal=True, tissue=inside)
    assert np.abs(a - seg.replay(seq, complex_signal=True, tissue=inside)).max() <= seg.fidelity["floor_max"]
    beyond = Tissue(rho2=1.5 * top * D)
    for route in (lambda: seg.replay(seq, tissue=beyond), lambda: seg.walker_primitives(seq).signals(beyond),
                  lambda: seg.pose_response(seq, tissue=beyond, keep=(8, 0))):
        with pytest.raises(ValueError, match="contact envelope"):
            route()


def test_rho_over_D_max_is_refused_unconditionally():
    """A pack's ``replay_envelope.tissue`` carrying a retired key for this bound -- ``rho_over_D_max``
    (pre-#581) or the underscored ``rho_2_over_D_max`` (pre-#592) -- is refused by name at every
    ``rpk_schema_version``, recorded or not: there is no version-gated migration (dmipy-sim#592, "no legacy
    rho"). The current spelling needs no version and no escape hatch."""
    def _pk(key, version=None):
        meta = {"replay_envelope": {"tissue": {key: 42.0}}}
        if version is not None:
            meta["rpk_schema_version"] = version
        return ReplayPack({"pos_x_ends": np.zeros((1, 2), np.float32)}, meta)

    for key in ("rho_over_D_max", "rho_2_over_D_max"):
        for version in (None, "0.4", "0.5", "0.6", "0.5.0-draft"):
            with pytest.raises(ValueError, match=f"retired key {key!r}"):
                _pk(key, version).rho2_over_D_max
    assert _pk("rho2_over_D_max", "0.6").rho2_over_D_max == 42.0


def test_the_envelope_edge_is_where_band_ripple_outgrows_the_floor():
    """A contact channel whose cumulative sum rises for a few walkers (band ripple: a contact only lowers the weight)
    is served up to the rho2 / D where that gain reaches the floor, and no further; a monotone one to the top."""
    from dmipy_sim.replay.bank import surface_envelope
    rng = np.random.default_rng(4)
    n_w, n_t = 20000, 200
    ell = -rng.exponential(1e-6, size=(n_w, n_t)) * (rng.uniform(size=(n_w, n_t)) < 0.3)
    w = np.ones(n_w)
    clean, _, _ = surface_envelope(ell, w, rho2_over_D_hi=1e5)
    assert clean == pytest.approx(1e5)
    rippled = ell.copy(); rippled[:2, 100] = 5e-4                    # two walkers in 20,000 rise by 5e-4
    edge, err, floor = surface_envelope(rippled, w, rho2_over_D_hi=1e5)
    assert 0 < edge < 1e5 and err <= 2.0 * floor
    gain = lambda rd: 2 / n_w * np.expm1(rd * 5e-4)
    assert gain(edge) <= floor * 1.0001                             # the gain at the edge is within its floor


def test_the_band_is_judged_over_the_saves_an_acquisition_spans(packs):
    """``waveform_band`` on a pack in windows reads the windows an acquisition reaches, not the whole walk: a
    one-window PGSE is judged exactly as on the first window alone (the same projection, the same bands and the
    same dropped phase), a two-window PGSE as on the first two, and the bands allowed grow with the windows
    (``K`` per window) while the frequency they resolve does not."""
    parent, seg, _ = packs
    wf = _waveforms()
    assert seg._span(_duration(wf["pgse_1"])) == (int(seg.segments["n_t"]), 1)
    assert seg._span(_duration(wf["pgse_2"])) == (2 * (int(seg.segments["n_t"]) - 1) + 1, 2)
    assert seg._span(_duration(wf["train_4"])) == (int(seg.n_t), 4)
    assert parent._span(_duration(wf["pgse_1"])) == (int(parent.n_t), 1)
    one, two = seg.truncate(1), seg.truncate(2)
    for name, pk in (("pgse_1", one), ("pgse_2", two)):
        hz_s, k_s, err_s = seg.waveform_band(wf[name])
        hz_p, k_p, err_p = pk.waveform_band(wf[name])
        assert (hz_s, k_s) == (hz_p, k_p) and err_s == pytest.approx(err_p, rel=1e-12), name
    assert seg.waveform_band(wf["pgse_1"])[0] == pytest.approx(seg.waveform_band(wf["pgse_2"])[0], rel=0.5)
    assert np.isfinite(seg.replay(wf["pgse_2"])).all()                             # within K per window over two windows


def test_resegment_states_that_the_contact_channel_is_inherited(packs):
    """``provenance.resegmented.contact``: the parent's contact band and walk duration, marked inherited, beside
    the pack's own ``replay_envelope`` that bounds where the windowed contact holds (#528)."""
    parent, seg, _ = packs
    c = seg.meta["provenance"]["resegmented"]["contact"]
    assert c["inherited"] is True
    assert c["parent_K"] == parent.meta["compression"]["channels"]["boundary_local_time"]["K"] == 8
    assert c["parent_T"] == pytest.approx(T)
    assert "rho2_over_D_max" in c["note"] and seg.meta["replay_envelope"]["tissue"]["rho2_over_D_max"] > 0


def _worst_window_surface_ratio(parent_pack, raw_dlog_b, w, D, env, tmp_path, steps, stamp):
    """Resegment ``parent_pack`` into windows of ``steps`` saves and return the worst window's
    ``err_surface / floor_surface`` against the window's own TRUE raw contact (``raw_dlog_b``, the ground truth
    only a test has): what #528 found resegment's own certificate (measured against the PARENT's decoded window,
    not the raw walk) cannot see."""
    p = tmp_path / f"{stamp}.rpk"
    parent_pack.save(p)
    seg = read_rpk(p).resegment(bank.SEGMENT_T, out_path=tmp_path / f"{stamp}-seg.rpk", id=f"t/{stamp}-reseg")
    worst = 0.0
    for i in range(seg.n_segments):
        lo = i * steps
        raw_win = np.array(raw_dlog_b[:, lo:lo + steps + 1], copy=True)
        if i:
            raw_win[:, 0] = 0.0                                 # the window's own first save ends no step of it
        s = seg.segment(i)
        cf = bank._surface_fidelity(dict(dlog_b=raw_win, w=w, D_intra=D), s.arrays,
                                    s.meta["compression"]["channels"]["boundary_local_time"], env)
        worst = max(worst, cf["err"] / cf["floor"] if cf["floor"] else float("inf"))
    return worst


def test_resegment_of_a_band_floored_one_window_pack_passes_the_windows_own_certificate(tmp_path):
    """#528's proper fix, checked end to end: a one-window pack built AT the new band floor (16 bands per
    storage-rule window) resegments into windows that reproduce the TRUE raw window's contact within tol x its
    split-half floor -- the thing #528 found failing for a pack built below the floor (here, one pinned at the
    plain ladder's K=8, as the historical one-window packs were), measured the same way against the same raw
    walk the directly-segmented build (#528's `_build_segmented` route) is measured against."""
    m = _c2_slab_master()
    env = _lean_env()
    T_walk = (C2_N_T - 1) * C2_DT
    steps = int(round(bank.SEGMENT_T / C2_DT))
    kw = dict(method="bridge_dst", envelope=env, K=64, license="CC-BY-4.0", citation="test", segment_T=T_walk)
    good = build_replay_pack(m, id="t/c2-good", **kw)                             # the new floor: K >= 16/window
    bad = build_replay_pack(m, id="t/c2-bad", blt_temporal_K=8, **kw)             # pinned at the historical K=8
    assert good.meta["compression"]["channels"]["boundary_local_time"]["K"] >= 16 * 10
    assert bad.meta["compression"]["channels"]["boundary_local_time"]["K"] == 8
    good_ratio = _worst_window_surface_ratio(good, m["dlog_b"], m["w"], m["D_intra"], env, tmp_path, steps, "good")
    bad_ratio = _worst_window_surface_ratio(bad, m["dlog_b"], m["w"], m["D_intra"], env, tmp_path, steps, "bad")
    assert good_ratio <= 2.0, good_ratio
    assert bad_ratio > 2.0, bad_ratio
