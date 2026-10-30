"""``ReplayPack.rewindow`` (#525): a one-window pack stored in windows of a shorter duration replays as its parent
on every tier and every route, to the rewindowed pack's certified floor, whatever window an acquisition spans; a
prefix of its windows is a range of it; a pose expansion reads the windows an acquisition reaches and no more.
Every window is a closed-form re-projection of the parent's stored functions (the bridge and the field modes by
their window maps, the events sliced), never a decode to saves."""
import time

import numpy as np
import numpy.testing as npt
import pytest

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.replay import ReplayPack, read_rpk
from dmipy_replay.spec.tissue import Tissue
from tests.test_bank import _lean_env
from tests.test_storage_windows import DT, N_T, T, _kw, _walk, _float_pack

TIS = Tissue(T2={"extra": 0.05, "intra": 0.02}, T1={"extra": 1.0, "intra": 0.5}, rho2=2e-5, chi_iso=1e-7)
KNOBS = {"bare": dict(), "relaxation": dict(tissue=Tissue(T2={"extra": 0.05, "intra": 0.02}, T1={"extra": 1.0, "intra": 0.5})),
         "contact": dict(tissue=Tissue(rho2=2e-5)), "field": dict(tissue=Tissue(chi_iso=1e-7), scanner=3.0),
         "all": dict(tissue=TIS, scanner=3.0)}


def _parent(T_max=T):
    """The cell walk as ONE window (K = 38 over 20 ms): C1 events, the contact bridge and the path channel."""
    K = int(round(38 * T_max / T))
    kw = _kw()
    kw["envelope"] = dict(kw["envelope"], theta_deg=[0, 90], max_refocus_pulses=8)
    return _float_pack(_walk(T_max, K, T_max=T_max), id="t/parent", K=K, blt_temporal_K=K, susc_path_K=16, window_T=T_max, **kw)


@pytest.fixture(scope="module")
def packs(tmp_path_factory):
    parent = _parent()
    p = tmp_path_factory.mktemp("reseg")
    parent.save(p / "parent.rpk")
    parent = read_rpk(str(p / "parent.rpk"))
    seg = parent.rewindow(T / 4, out_path=p / "seg.rpk", id="t/seg")
    return parent, seg, p


def _dirs(n, seed=0):
    u = np.random.default_rng(seed).normal(size=(n, 3))
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def _waveforms():
    """100 diffusion measurements and a train: PGSE inside window 0, PGSE across two windows, PGSTE across three,
    an eight-echo train across all four."""
    dirs = _dirs(50)
    two = np.concatenate([dirs, dirs]); b = lambda b1, b2: [b1] * 50 + [b2] * 50
    return {"pgse_1": sequences.pgse(two, 1e-3, 3e-3, bvalues=b(0.5e9, 1e9), TE=5e-3, n_t=200, slew_rate=np.inf),
            "pgse_2": sequences.pgse(two, 2e-3, 6e-3, bvalues=b(1e9, 2e9), TE=10e-3, n_t=400, slew_rate=np.inf),
            "pgste_3": sequences.pgste(two, 1.5e-3, 10e-3, bvalues=b(0.5e9, 1e9), TE=15e-3, n_t=600, slew_rate=np.inf),
            "train_4": sequences.cpmg(8, 2.4e-3, gradient_directions=[[1.0, 0, 0]], bvalues=[0.0], n_t_per_echo=40)}


def test_the_windows_and_their_record(packs):
    parent, seg, p = packs
    assert parent.n_windows == 1 and seg.n_windows == 4
    assert seg.T_window == pytest.approx(T / 4) and seg.windows["T"] == pytest.approx(T / 4) and seg.n_t == N_T
    assert seg.K == 10                                                       # ceil(38 / 4): the bands per second kept
    assert seg.temporal_bandwidth_hz == pytest.approx(parent.temporal_bandwidth_hz, rel=0.06)
    r = seg.meta["provenance"]["rewindowed"]
    assert r["parent_id"] == "t/parent" and r["n_seg"] == 4 and r["time_dropped"] == 0 and r["T_seg"] == pytest.approx(T / 4)
    assert len(r["parent_sha256"]) == 64 and r["K"] == 10 and r["parent_K"] == 38
    f = seg.fidelity
    assert f["combined"] == "bound_over_windows" and len(f["windows"]) == 4
    for w in f["windows"]:
        assert w["err_reprojection"] <= 2.0 * w["floor_max"] and "reprojection" in w["terms"] and "err_surface" in w
    ch = seg.meta["compression"]["channels"]
    assert ch["susceptibility_grid"] == parent.meta["compression"]["channels"]["susceptibility_grid"]
    assert ch["compartment"]["columns"][0]["kind"] == "events"               # the walk crosses: events in every window
    per_window = {k[3:] for k in seg.arrays if k.startswith("s3/")}
    assert {"pos_x", "blt_bridge_dst", "comp_events_t", "susc_path_dct"} <= per_window
    np.testing.assert_array_equal(seg.spin_weights, parent.spin_weights)
    back = read_rpk(str(p / "seg.rpk"))
    assert set(back.arrays) == set(seg.arrays)
    for k in seg.arrays:
        npt.assert_array_equal(np.asarray(back.arrays[k]), np.asarray(seg.arrays[k]), err_msg=k)


def test_the_window_ends_are_the_parents_bridge(packs):
    """Every window's endpoints are the parent's stored bridge read at its ends, to float32 (the window map is
    closed-form; the parent's block ends are the raw subset's positions only to the parent's truncation)."""
    parent, seg, _ = packs
    from dmipy_replay.replay.compression import positions_at
    X = positions_at(parent.arrays, dict(T=parent.T_window), np.arange(5) * T / 4)        # the bridge at the window ends
    for i in range(4):
        Ci = seg.window(i).position_coeffs
        npt.assert_allclose(Ci[:, 0, :], X[:, i, :], rtol=0, atol=1e-6 * np.abs(X).max())
        npt.assert_allclose(Ci[:, 0, :] + Ci[:, 1, :], X[:, i + 1, :], rtol=0, atol=1e-6 * np.abs(X).max())
    npt.assert_allclose(seg.window(0).r0, parent.r0, atol=0)


@pytest.mark.parametrize("name", ["pgse_1", "pgse_2", "pgste_3", "train_4"])
@pytest.mark.parametrize("tier", list(KNOBS))
def test_the_windows_replay_as_the_parent(packs, name, tier):
    parent, seg, _ = packs
    seq = _waveforms()[name]
    a = parent.replay(seq, complex_signal=True, **KNOBS[tier])
    b = seg.replay(seq, complex_signal=True, **KNOBS[tier])
    assert np.abs(a - b).max() <= 2.0 * seg.fidelity["floor_max"]


@pytest.mark.parametrize("name", ["pgse_1", "pgse_2", "pgste_3"])
def test_the_pose_expansion_is_the_parents(packs, name):
    parent, seg, _ = packs
    seq = _waveforms()[name]
    rp = parent.pose_response(seq, tissue=TIS, scanner=3.0, keep=(8, 0))
    rs = seg.pose_response(seq, tissue=TIS, scanner=3.0, keep=(8, 0))
    floor = 2.0 * seg.fidelity["floor_max"]
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


def test_a_truncated_rewindowed_pack_is_its_first_window(packs, tmp_path):
    parent, seg, _ = packs
    one = seg.truncate(1, out_path=tmp_path / "one.rpk")
    assert one.n_windows == 1 and one.n_t == 11
    seq = _waveforms()["pgse_1"]
    for knobs in KNOBS.values():
        npt.assert_array_equal(one.replay(seq, complex_signal=True, **knobs), seg.replay(seq, complex_signal=True, **knobs))
    npt.assert_array_equal(one.pose_response(seq, tissue=TIS, scanner=3.0, keep=(8, 0)).coeffs,
                           seg.pose_response(seq, tissue=TIS, scanner=3.0, keep=(8, 0)).coeffs)


def test_the_trailing_remainder_is_dropped_within_its_bound_and_refused_beyond(packs, tmp_path):
    """Windows of 5.5 ms leave 3.5 ms of the 20 ms walk: refused at the default bound, dropped and recorded at 0.2."""
    parent, seg, _ = packs
    with pytest.raises(ValueError, match="uncovered"):
        parent.rewindow(5.5e-3, out_path=tmp_path / "x.rpk")
    s = parent.rewindow(5.5e-3, out_path=tmp_path / "seg.rpk", max_dropped_fraction=0.2)
    assert s.n_windows == 3 and s.meta["provenance"]["rewindowed"]["time_dropped"] == pytest.approx(3.5e-3)
    seq = _waveforms()["pgse_2"]
    a = parent.replay(seq, complex_signal=True, **KNOBS["all"])
    assert np.abs(a - s.replay(seq, complex_signal=True, **KNOBS["all"])).max() <= 2.0 * s.fidelity["floor_max"]
    with pytest.raises(ValueError, match="already stores"):
        seg.rewindow(T / 8, out_path=tmp_path / "y.rpk")


def test_the_window_maps_are_the_closed_form_projection():
    """The bridge and mode window matrices (``bank.bridge_window_matrix`` / ``mode_window_matrix``) reproduce a dense
    quadrature of the window's own projection of the parent's stored function."""
    from dmipy_sim.replay.bank import bridge_window_matrix, mode_window_matrix
    rng = np.random.default_rng(0)
    T_p, K, t0, Tn, Kn = 0.01, 12, 0.003, 0.004, 8
    x0, dx = 1.0, 0.5
    beta = rng.normal(size=K) * 1e-1
    k = np.arange(1, K + 1)
    I = lambda t: x0 + dx * t / T_p + (beta[None, :] * np.sqrt(2 / T_p) * np.sin(np.pi * np.outer(t, k) / T_p)).sum(1)
    tau = np.linspace(0, Tn, 200001)
    It = I(t0 + tau)
    u = It - It[0] - (It[-1] - It[0]) * tau / Tn
    j = np.arange(1, Kn + 1)
    bn = np.trapezoid(u[None, :] * np.sqrt(2 / Tn) * np.sin(np.pi * np.outer(j, tau) / Tn), tau, axis=1)
    out = bridge_window_matrix(T_p, K, t0, Tn, Kn) @ np.concatenate([[x0, dx], beta])
    assert abs(out[0] - It[0]) < 1e-12 and abs(out[1] - (It[-1] - It[0])) < 1e-12
    assert np.abs(out[2:] - bn).max() < 1e-12 * np.abs(bn).max()
    c = rng.normal(size=K)
    e = np.sqrt(2 / T_p) * np.cos(np.pi * np.outer(t0 + tau, np.arange(K)) / T_p); e[:, 0] = 1 / np.sqrt(T_p)
    bt = e @ c
    E = np.sqrt(2 / Tn) * np.cos(np.pi * np.outer(np.arange(Kn), tau) / Tn); E[0] = 1 / np.sqrt(Tn)
    cn = np.trapezoid(bt[None, :] * E, tau, axis=1)
    assert np.abs(mode_window_matrix(T_p, K, t0, Tn, Kn) @ c - cn).max() < 1e-9 * np.abs(cn).max()


def _timing_packs(tmp_path):
    """A free walk long in blocks and wide in band (2000 blocks, K = 1000), and the same walk in four windows: the
    cost of a pose expansion is then the band contraction of the windows the acquisition spans."""
    g = d.FreeDiffusion()
    walk = d.simulate_trajectories(4000, 2e-9, g, 0.1, 5e-5, seed=2, require_gpu=False, K=1000, window_T=0.1, tiers=())
    env = dict(_lean_env(), ogse_periods=[], shortd_deltas_frac=[])
    parent = _float_pack(walk, id="t/long", license="x", citation="x", envelope=env, K=1000, window_T=0.1)
    parent.save(tmp_path / "long.rpk")
    parent = read_rpk(str(tmp_path / "long.rpk"))
    return parent, parent.rewindow(0.025, out_path=tmp_path / "long_seg.rpk", envelope=env)


def test_a_pose_expansion_costs_the_windows_it_spans(tmp_path):
    """The one-window acquisition on the rewindowed pack reads one window of 250 bands where the parent reads its
    whole walk of 1000, and the expansion is the parent's to the floor. Its time is held to a loose ordering only,
    the best of five on a shared machine."""
    parent, seg = _timing_packs(tmp_path)
    assert seg.n_windows == 4 and seg.K == 250
    seq = sequences.pgse(_dirs(30), 4e-3, 10e-3, bvalues=[1e9] * 30, TE=20e-3, n_t=800, slew_rate=np.inf)
    spanned = lambda pk: [int(w.K) for w, _, _ in pk._prepare(seq, tissue=None, scanner=None, orientation=None, compartment=None)["windows"]]
    assert spanned(parent) == [1000] and spanned(seg) == [250]

    def best(pk):
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); r = pk.pose_response(seq, keep=(8, 0)); ts.append(time.perf_counter() - t0)
        return min(ts), r

    t_parent, rp = best(parent)
    t_seg, rs = best(seg)
    assert np.abs(rp.coeffs - rs.coeffs).max() <= 2.0 * seg.fidelity["floor_max"]
    assert t_seg < t_parent, (t_seg, t_parent)


def test_the_contact_envelope_is_stated_and_held(packs):
    """The pack states the largest rho2 / D its contact tier serves (RPK.md 8.7); a relaxivity inside replays as the
    parent within the floor, one beyond is refused by name on every route."""
    parent, seg, _ = packs
    top = seg.rho2_over_D_max
    assert top is not None and top > 0 and seg.meta["replay_envelope"]["tissue"]["rho2_over_D_max"] == top
    r = seg.meta["provenance"]["surface_envelope_restated"]
    assert r["rho2_over_D_max"] == top == min(r["per_window"]) and r["reference"] == "t/parent"
    for f in seg.fidelity["windows"]:
        assert f["surface_rho2_over_D_max"] == top and f["err_surface"] <= 2.0 * f["floor_surface"]
    D = seg.diffusivity
    seq = _waveforms()["pgse_2"]
    inside = Tissue(rho2=0.9 * top * D)
    a = parent.replay(seq, complex_signal=True, tissue=inside)
    assert np.abs(a - seg.replay(seq, complex_signal=True, tissue=inside)).max() <= 2.0 * seg.fidelity["floor_max"]
    beyond = Tissue(rho2=1.5 * top * D)
    for route in (lambda: seg.replay(seq, tissue=beyond), lambda: seg.walker_primitives(seq).signals(beyond),
                  lambda: seg.pose_response(seq, tissue=beyond, keep=(8, 0))):
        with pytest.raises(ValueError, match="contact envelope"):
            route()


def test_an_unknown_replay_envelope_tissue_key_is_refused():
    """A pack's ``replay_envelope.tissue`` knows only ``rho2_over_D_max``; any other key there is refused
    by name at every ``rpk_schema_version``, recorded or not -- there is no version-gated acceptance of
    another spelling."""
    def _pk(key, version=None):
        meta = {"replay_envelope": {"tissue": {key: 42.0}}}
        if version is not None:
            meta["rpk_schema_version"] = version
        return ReplayPack({"pos_x_ends": np.zeros((1, 2), np.float32)}, meta)

    for key in ("rho_over_D_max", "rho_2_over_D_max", "foo"):
        for version in (None, "0.4", "0.5", "0.6", "0.5.0-draft"):
            with pytest.raises(ValueError, match=rf"declares \['{key}'\]"):
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


def test_the_band_is_judged_over_the_windows_an_acquisition_spans(packs):
    """``waveform_band`` on a pack in windows reads the windows an acquisition reaches, not the whole walk: a
    one-window PGSE is judged exactly as on the first window alone (the same projection, the same bands and the
    same dropped phase), a two-window PGSE as on the first two, and the bands allowed grow with the windows
    (``K`` per window) while the frequency they resolve does not."""
    from dmipy_replay.replay.replay import _duration
    parent, seg, _ = packs
    wf = _waveforms()
    assert seg._span(_duration(wf["pgse_1"])) == (pytest.approx(seg.T_window), 1)
    assert seg._span(_duration(wf["pgse_2"])) == (pytest.approx(2 * seg.T_window), 2)
    assert seg._span(_duration(wf["train_4"])) == (pytest.approx(4 * seg.T_window), 4)
    assert parent._span(_duration(wf["pgse_1"])) == (pytest.approx(parent.T_window), 1)
    one, two = seg.truncate(1), seg.truncate(2)
    for name, pk in (("pgse_1", one), ("pgse_2", two)):
        t = float(seg.fidelity["floor_max"])                                      # one tolerance: the spans compared
        hz_s, k_s, err_s = seg.waveform_band(wf[name], tol=t)
        hz_p, k_p, err_p = pk.waveform_band(wf[name], tol=t)
        assert (hz_s, k_s) == (hz_p, k_p) and err_s == pytest.approx(err_p, rel=1e-12), name
    assert np.isfinite(seg.replay(wf["pgse_2"])).all()                             # within K per window over two windows
