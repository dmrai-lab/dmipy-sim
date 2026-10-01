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
from dmipy_sim.spec.tissue import Tissue
from tests.test_bank import _lean_env, _susc_master

N_T, DT = 197, 5e-4                      # 98 ms of walk: four windows of 49 steps (24.5 ms)
T = (N_T - 1) * DT
TIS = Tissue(T2={"extra": 0.05, "intra": 0.02}, T1={"extra": 1.0, "intra": 0.5}, rho=2e-5, chi_iso=1e-7)
KNOBS = {"bare": dict(), "relaxation": dict(tissue=Tissue(T2={"extra": 0.05, "intra": 0.02}, T1={"extra": 1.0, "intra": 0.5})),
         "contact": dict(tissue=Tissue(rho=2e-5)), "field": dict(tissue=Tissue(chi_iso=1e-7), scanner=3.0),
         "all": dict(tissue=TIS, scanner=3.0)}


def _master(n_t=N_T):
    """The slab walk with C1 (a third of the walkers cross into pool 1 in the third window), C2 and the C3 grid."""
    m = _susc_master()
    for k in ("traj", "comp", "dlog_b"):
        m[k] = np.asarray(m[k])[:, :n_t]
    m["T_max"] = (n_t - 1) * DT
    comp = np.zeros((m["traj"].shape[0], n_t), np.int8); comp[:1000, 120:] = 1
    m["comp"] = comp
    m["substrate"] = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6).spec.to_dict()   # names for the two pools the walk labels
    return m


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
