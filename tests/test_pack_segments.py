"""A pack stores its walk in segments of one duration (RPK.md 4.3), and a replay across them is the sum of the windows'
contractions: a walk stored in two windows replays as the same walk stored in one, on every tier and every scalar
route, to the containers' rounding; a window is a pack of its own; a prefix of whole windows is a range of the
file; a partial prefix re-encodes the decoded windows; the builder refuses a walk that is not a whole number of
windows."""
import numpy as np
import numpy.testing as npt
import pytest

import dmipy_sim as d
from dmipy_sim import build_replay_pack
from dmipy_sim.replay.bank import combine_segment_fidelity, segment_plan
from dmipy_sim.replay.study import walker_primitives
from dmipy_sim.spec.tissue import Tissue
from tests.test_bank import _lean_env, _susc_master

N_T, DT, N_W = 41, 5e-4, 600                 # 20 ms of walk; two windows of 10 ms
TIS = Tissue(T2=[0.05, 0.02], T1=[1.0, 0.5], rho=2e-5, chi_iso=1e-7)


def _master():
    m = _susc_master(n_w=N_W)
    for k in ("traj", "comp", "dlog_b"):
        m[k] = np.asarray(m[k])[:, :N_T]
    m["T_max"] = (N_T - 1) * DT
    comp = np.zeros((N_W, N_T), np.int8); comp[:200, 25:] = 1          # a third of the walkers cross in window 1
    m["comp"] = comp
    return m


@pytest.fixture(scope="module")
def packs():
    """The same walk stored in one window and in two, both lossless in every channel (the bridge at K = n - 2, the
    field's cosine basis at K = n) in float32 containers, so the two differ by rounding alone."""
    m = _master()
    env = dict(_lean_env(), B0_list=[3.0], theta_deg=[0])
    kw = dict(license="CC-BY-4.0", citation="test", envelope=env, blt_dtype=np.float32, susc_path_bits=16)
    one = build_replay_pack(m, id="t/one", K=N_T - 2, blt_temporal_K=N_T - 2, susc_path_K=N_T, segment_T=(N_T - 1) * DT, **kw)
    two = build_replay_pack(m, id="t/two", K=19, blt_temporal_K=19, susc_path_K=21, segment_T=0.01, **kw)
    return m, one, two


def _seq(n_t=N_T, TE=None):
    return d.pgse([[1, 0, 0], [0, 0, 1]], 0.004, 0.012, gradient_strengths=0.3, n_t=n_t, slew_rate=np.inf, TE=TE)


def test_the_layout_and_the_table(packs):
    m, one, two = packs
    assert one.n_segments == 1 and one.segments == dict(n=1, n_t=N_T, T=pytest.approx((N_T - 1) * DT), walks=[dict(first=0, last=0, seed=0)])
    assert two.n_segments == 2 and two.segments["n_t"] == 21 and two.segments["T"] == pytest.approx(0.01)
    assert two.n_t == N_T and two.meta["compression"]["n_t"] == 21 and two.K == 19
    per_window = {k[3:] for k in two.arrays if k.startswith("s1/")}
    assert {"pos_x", "pos_y", "pos_z", "blt_bridge_dst", "blt_start", "blt_endpoint", "comp_rle_vals", "comp_rle_lens",
            "comp_rle_counts", "susc_path_dct", "susc_path_scale"} <= per_window
    shared = {k for k in two.arrays if "/" not in k and k not in per_window}
    assert shared == {"spin_weights", "susc_grid_iso_local", "susc_grid_iso_P"}
    assert "comp_static" not in two.arrays                                  # a walk that crosses anywhere stores runs everywhere
    f = two.fidelity
    assert f["certified"] == "measured" and len(f["segments"]) == 2 and f["within_2x_floor"]


def test_a_window_is_the_pack_a_fresh_walk_of_it_would_be(packs):
    m, one, two = packs
    from dmipy_sim.replay.bank import _window_master
    env = dict(_lean_env(), B0_list=[3.0], theta_deg=[0])
    w1 = build_replay_pack(_window_master(m, 20, 40), id="t/two", K=19, blt_temporal_K=19, susc_path_K=21, segment_T=0.01,
                           license="CC-BY-4.0", citation="test", envelope=env, blt_dtype=np.float32, susc_path_bits=16,
                           _occupancy_runs=True)
    s1 = two.segment(1)
    assert s1.n_t == 21 and s1.n_segments == 1 and s1.segments["T"] == pytest.approx(0.01)
    for k in w1.arrays:
        npt.assert_array_equal(s1.arrays[k], w1.arrays[k], err_msg=k)
    npt.assert_allclose(two.positions(), np.asarray(m["traj"]), atol=1e-11)


@pytest.mark.parametrize("knobs", [dict(), dict(tissue=TIS), dict(tissue=TIS, scanner=3.0)])
def test_two_windows_replay_as_one_on_every_tier(packs, knobs):
    m, one, two = packs
    seq = _seq()
    npt.assert_allclose(two.replay(seq, complex_signal=True, **knobs), one.replay(seq, complex_signal=True, **knobs), rtol=1e-5, atol=1e-6)
    w1, ew1, E1 = one.walker_signals(seq, **knobs); w2, ew2, E2 = two.walker_signals(seq, **knobs)
    npt.assert_allclose(ew2, ew1, rtol=1e-5); npt.assert_allclose(E2, E1, atol=1e-5)


def test_an_acquisition_off_the_save_grid_splits_exactly(packs):
    m, one, two = packs
    seq = d.pgse([[1, 0, 0]], 0.0042, 0.0123, gradient_strengths=0.3, n_t=67, slew_rate=np.inf, TE=0.0198)
    npt.assert_allclose(two.replay(seq, tissue=TIS, scanner=3.0, complex_signal=True),
                        one.replay(seq, tissue=TIS, scanner=3.0, complex_signal=True), rtol=1e-5, atol=1e-6)


def test_the_primitives_and_the_pose_expansion_sum_over_windows(packs):
    m, one, two = packs
    seq = _seq()
    p1, p2 = walker_primitives(one, seq), walker_primitives(two, seq)
    npt.assert_allclose(p2.phi, p1.phi, atol=1e-6)
    npt.assert_allclose(p2.exposure_t2, p1.exposure_t2, atol=1e-12); npt.assert_allclose(p2.exposure_t1, p1.exposure_t1, atol=1e-12)
    npt.assert_allclose(p2.contact, p1.contact, rtol=1e-4, atol=1e-9)   # float32 endpoints of the contact bridge
    npt.assert_allclose(p2.field_iso, p1.field_iso, atol=1e-3 * np.abs(p1.field_iso).max())   # the int16 container of the path channel
    r1 = one.pose_response(seq, tissue=TIS, scanner=3.0); r2 = two.pose_response(seq, tissue=TIS, scanner=3.0)
    npt.assert_allclose(r2.coeffs, r1.coeffs, atol=1e-5)
    w, ew, E = two.walker_signals(seq, tissue=TIS, scanner=3.0)
    w_, ew_, E_ = p2.signals(TIS, 3.0)
    npt.assert_allclose(ew_, ew, rtol=1e-10); npt.assert_allclose(E_, E, atol=1e-10)


def test_a_short_acquisition_reads_the_first_window_alone(packs):
    m, one, two = packs
    short = d.pgse([[1, 0, 0]], 0.002, 0.006, gradient_strengths=0.3, n_t=21, slew_rate=np.inf)   # TE = 10 ms
    blind = d.replay.ReplayPack({k: (np.full_like(v, np.nan) if k.startswith("s1/") and np.issubdtype(np.asarray(v).dtype, np.floating) else v)
                                 for k, v in two.arrays.items()}, two.meta)
    a = blind.replay(short, tissue=TIS, scanner=3.0, complex_signal=True)
    assert np.all(np.isfinite(a))
    npt.assert_array_equal(a, two.segment(0).replay(short, tissue=TIS, scanner=3.0, complex_signal=True))


def test_a_prefix_of_whole_windows_is_a_range_and_a_partial_one_re_encodes(packs):
    m, one, two = packs
    pr = two.prefix(0.01)
    assert pr.n_segments == 1 and pr.n_t == 21 and set(pr.arrays) == set(two.segment(0).arrays)
    for k in pr.arrays:
        npt.assert_array_equal(pr.arrays[k], two.segment(0).arrays[k])
    assert pr.fidelity["certified"] == "bounded" and pr.meta["provenance"]["truncated"]["segments_kept"] == 1
    pp = two.prefix(0.015)
    assert pp.n_segments == 1 and pp.n_t == 31 and pp.fidelity["within_2x_floor"]
    seq = d.pgse([[1, 0, 0]], 0.003, 0.009, gradient_strengths=0.3, n_t=31, slew_rate=np.inf)   # TE = 15 ms
    npt.assert_allclose(pp.replay(seq, tissue=TIS, complex_signal=True), two.replay(seq, tissue=TIS, complex_signal=True), atol=3e-3)


def test_the_plan_refuses_a_walk_that_is_not_whole_windows():
    assert segment_plan(41, 5e-4, 0.01) == (2, 21) and segment_plan(41, 5e-4, 0.02) == (1, 41) and segment_plan(41, 5e-4, 0.1) == (1, 41)
    with pytest.raises(ValueError, match="segment_T"):
        segment_plan(41, 5e-4, None)
    with pytest.raises(ValueError, match="not a whole number"):
        segment_plan(41, 5e-4, 0.015)
    with pytest.raises(ValueError, match="not a whole number of saves"):
        segment_plan(41, 5e-4, 0.0101)


def test_the_bound_over_segments():
    a = dict(metric="m", err_max=0.01, floor_max=0.05, err_surface=0.002, floor_surface=0.04, per_family={"PGSE": dict(err_max=0.01, floor_max=0.05)})
    b = dict(metric="m", err_max=0.02, floor_max=0.03, err_surface=0.001, floor_surface=0.06, per_family={"PGSE": dict(err_max=0.02, floor_max=0.03)})
    c = combine_segment_fidelity([a, b])
    assert c["err_max"] == pytest.approx(0.03) and c["floor_max"] == pytest.approx(0.05) and c["err_surface"] == pytest.approx(0.003)
    assert c["floor_surface"] == pytest.approx(0.06) and c["per_family"]["PGSE"] == dict(err_max=pytest.approx(0.03), floor_max=pytest.approx(0.05))
    assert c["certified"] == "bounded" and c["within_2x_floor"] and len(c["segments"]) == 2


def test_a_pack_without_the_table_is_refused():
    with pytest.raises(ValueError, match="walk_params.segments"):
        d.replay.ReplayPack({"pos_x": np.zeros((2, 4), np.float32)}, {"walk_params": {"n_t": 3}}).n_segments
