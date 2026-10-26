"""A pack stores its walk in windows of one duration (RPK.md 4.3), and a replay across them is the sum of the windows'
contractions: a walk accumulated in two windows replays as the same walk accumulated in one, on every tier and
every scalar route, within the two packs' certified truncation (the bands per second are the same, the band
truncation of each window its own); a window is a pack of its own; a prefix of whole windows is a range of the
file; a partial prefix re-projects the windows it cuts; the producer refuses a walk that is not a whole number of
windows."""
import dataclasses

import numpy as np
import numpy.testing as npt
import pytest

import dmipy_sim as d
from dmipy_sim import build_replay_pack
from dmipy_sim.replay import bank
from dmipy_sim.replay.bank import WINDOW_T, combine_window_fidelity
from dmipy_sim.replay.study import walker_primitives
from dmipy_replay.spec.substrate import Susceptibility
from dmipy_replay.spec.tissue import Tissue
from tests.test_bank import _lean_env, _slab_grid

N_T, DT, N_W = 41, 5e-4, 600                 # 20 ms of walk in 40 blocks; two windows of 10 ms
T = (N_T - 1) * DT
TIS = Tissue(T2={"extra": 0.05, "intra": 0.02}, T1={"extra": 1.0, "intra": 0.5}, rho2=2e-5, chi_iso=1e-7)
TIS0 = TIS.replace(chi_iso=None)              # the same material read without a field: no scanner, no chi (#593)


def _geometry():
    """One permeable cylinder in a periodic cell: walkers cross between the two named pools (events in C1)."""
    return d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6, permeability=2e-5)


def _magnetic(spec):
    """The cell's spec with its lumen a field source and the field tier declared (the synthetic grid stands in)."""
    return spec.replace(susceptibility={"intra": Susceptibility(None, None, "none")},
                        validity=dataclasses.replace(spec.validity, tiers=list(spec.validity.tiers) + ["field"]))


def _walk(window_T, K, *, n_w=N_W, seed=0, T_max=T):
    """The cell walked with C1 (events), C2 and C3 (the slab grid read in the walk, 16 modes), every walker's
    block ends kept, in windows of ``window_T`` at ``K`` bands each."""
    w = d.simulate_trajectories(n_w, 2e-9, _geometry(), T_max, DT, seed=seed, require_gpu=False, K=K, window_T=window_T,
                                field_basis=_slab_grid(), K_c3=16, raw_subset=n_w)
    return dataclasses.replace(w, spec=_magnetic(w.spec))


def _kw(**more):
    env = dict(_lean_env(), B0_list=[3.0], theta_deg=[0])
    kw = dict(license="CC-BY-4.0", citation="test", envelope=env, blt_dtype=np.float32, field_container=((None, 16),), fidelity="measured")
    kw.update(more)
    return kw


@pytest.fixture(scope="module")
def packs():
    """The same walk accumulated in one window (K = 38) and in two (K = 19 each): the same bands per second, every
    channel in float32 containers, so the two differ by each window's band truncation and the containers' rounding."""
    w1, w2 = _walk(T, 38), _walk(0.01, 19)
    npt.assert_array_equal(w1.raw_positions, w2.raw_positions)                 # one random stream: one walk
    one = _float_pack(w1, id="t/one", K=38, blt_temporal_K=38, susc_path_K=16, window_T=T, **_kw())
    two = _float_pack(w2, id="t/two", K=19, blt_temporal_K=19, susc_path_K=16, window_T=0.01, **_kw())
    return w1, w2, one, two


def _bound(*packs):
    """What two packs of one walk may differ by: the sum of their walk-free bounds (the terms of RPK.md 9.4), which
    hold for every acquisition within the envelope, not only the battery the measured certificate replayed."""
    return sum(float(p.fidelity.get("err_derived", p.fidelity["err_max"])) for p in packs) + 1e-9


def _seq(n_t=N_T, TE=None):
    return d.pgse([[1, 0, 0], [0, 0, 1]], 0.004, 0.012, gradient_strengths=0.3, n_t=n_t, slew_rate=np.inf, TE=TE)


def test_the_layout_and_the_table(packs):
    w1, w2, one, two = packs
    assert one.n_windows == 1 and one.windows == dict(n=1, T=pytest.approx(T), walks=[dict(first=0, last=0, seed=0)])
    assert two.n_windows == 2 and two.windows["T"] == pytest.approx(0.01) and two.T_window == pytest.approx(0.01)
    assert two.n_t == N_T and "n_t" not in two.meta["compression"] and two.K == 19
    assert {k: two.block[k] for k in ("h", "m", "precision", "n_blocks", "rule")} == dict(
        h=pytest.approx(DT), m=w2.m, precision="float32", n_blocks=40, rule="given")
    assert set(two.meta["walk_params"]["qv"]) == {"positions", "local_time", "field", "field_increments"} and "terms" in two.fidelity
    assert two.meta["walk_params"]["field_read_interval"] == pytest.approx(w2.field_read_interval)
    per_window = {k[3:] for k in two.arrays if k.startswith("s1/")}
    assert {"pos_x", "pos_y", "pos_z", "blt_bridge_dst", "blt_start", "blt_endpoint", "comp_events_offsets", "comp_events_t",
            "comp_events_label", "susc_path_dct", "susc_path_scale"} <= per_window
    shared = {k for k in two.arrays if "/" not in k and k not in per_window}
    assert shared == set()                                                  # the field is the path channel's: no grid; both pools weigh 1
    assert "comp_static" not in two.arrays                                  # a walk that crosses stores events
    assert two.meta["compression"]["channels"]["compartment"]["columns"][0]["kind"] == "events"
    f = two.fidelity
    assert f["certified"] == "measured" and len(f["windows"]) == 2 and f["within_2x_floor"]
    assert all(w["measured_by"] == "raw_subset" and "err_surface" in w and "err_susc_path" in w for w in f["windows"])


def test_a_window_is_the_pack_a_fresh_walk_of_it_would_be(packs):
    w1, w2, one, two = packs
    s1 = two.window(1)
    assert s1.n_t == 21 and s1.n_windows == 1 and s1.windows["T"] == pytest.approx(0.01)
    npt.assert_array_equal(s1.r0, np.asarray(w2.raw_positions)[:, 20])            # the window opens on the exact position
    npt.assert_array_equal(two.window(0).r0, np.asarray(w2.raw_positions)[:, 0])
    # the window's own walk object packs to the same tensors
    p1 = _float_pack(w2.window(1), id="t/two", K=19, blt_temporal_K=19, susc_path_K=16, window_T=0.01, **_kw())
    for k in p1.arrays:
        npt.assert_array_equal(s1.arrays[k], p1.arrays[k], err_msg=k)
    # the events of window 1 are the walk's within it, timed on the walk's clock, opened by the label at its start
    from dmipy_replay.replay.events import events_within
    ev, lab0 = events_within(w2.comp_events, w2.compartment, 0.01, 0.02)
    from dmipy_replay.replay.compression import occupancy_events
    start, got = occupancy_events(s1.arrays, s1.meta["compression"]["channels"]["compartment"]["columns"][0])
    npt.assert_array_equal(start, lab0)
    assert got == ev


@pytest.mark.parametrize("knobs", [dict(), dict(tissue=TIS0), dict(tissue=TIS, scanner=3.0)])
def test_two_windows_replay_as_one_on_every_tier(packs, knobs):
    w1, w2, one, two = packs
    for seq in (_seq(), _seq(TE=T)):                                               # the echo inside the last window, and at the walk's end
        npt.assert_allclose(two.replay(seq, complex_signal=True, **knobs), one.replay(seq, complex_signal=True, **knobs), atol=_bound(one, two))
        w_1, ew1, E1 = one.walker_signals(seq, **knobs); w_2, ew2, E2 = two.walker_signals(seq, **knobs)
        npt.assert_allclose(E2.mean(0), E1.mean(0), atol=_bound(one, two))
    npt.assert_allclose(ew2, ew1, rtol=1e-4)      # at the walk's end: relaxation on the events exact, the contact the float32 endpoints


def test_an_acquisition_off_the_block_grid_splits_exactly(packs):
    w1, w2, one, two = packs
    seq = d.pgse([[1, 0, 0]], 0.0042, 0.0123, gradient_strengths=0.3, n_t=67, slew_rate=np.inf, TE=0.0198)
    npt.assert_allclose(two.replay(seq, tissue=TIS, scanner=3.0, complex_signal=True),
                        one.replay(seq, tissue=TIS, scanner=3.0, complex_signal=True), atol=_bound(one, two))


def test_the_primitives_sum_over_windows(packs):
    w1, w2, one, two = packs
    seq = _seq(TE=T)                                                               # the gate spans the walk: the contact is the endpoints
    p1, p2 = walker_primitives(one, seq), walker_primitives(two, seq)
    npt.assert_allclose(p2.exposure_t2, p1.exposure_t2, atol=1e-12); npt.assert_allclose(p2.exposure_t1, p1.exposure_t1, atol=1e-12)
    npt.assert_allclose(p2.contact, p1.contact, rtol=1e-4, atol=1e-9)   # float32 endpoints of the contact bridge
    npt.assert_allclose(np.exp(1j * p2.phi).mean(0), np.exp(1j * p1.phi).mean(0), atol=_bound(one, two))
    npt.assert_allclose(p2.signal(TIS, 3.0), p1.signal(TIS, 3.0), atol=_bound(one, two))
    w, ew, E = two.walker_signals(seq, tissue=TIS, scanner=3.0)
    w_, ew_, E_ = p2.signals(TIS, 3.0)
    npt.assert_allclose(ew_, ew, rtol=1e-10); npt.assert_allclose(E_, E, atol=1e-10)


def test_a_short_acquisition_reads_the_first_window_alone(packs):
    w1, w2, one, two = packs
    short = d.pgse([[1, 0, 0]], 0.002, 0.006, gradient_strengths=0.3, n_t=21, slew_rate=np.inf)   # TE = 10 ms
    blind = d.replay.ReplayPack({k: (np.full_like(v, np.nan) if k.startswith("s1/") and np.issubdtype(np.asarray(v).dtype, np.floating) else v)
                                 for k, v in two.arrays.items()}, two.meta)
    a = blind.replay(short, tissue=TIS, scanner=3.0, complex_signal=True)
    assert np.all(np.isfinite(a))
    npt.assert_array_equal(a, two.window(0).replay(short, tissue=TIS, scanner=3.0, complex_signal=True))


def test_a_prefix_of_whole_windows_is_a_range_and_a_partial_one_re_projects(packs):
    w1, w2, one, two = packs
    pr = two.prefix(0.01)
    assert pr.n_windows == 1 and pr.n_t == 21 and set(pr.arrays) == set(two.window(0).arrays)
    for k in pr.arrays:
        npt.assert_array_equal(pr.arrays[k], two.window(0).arrays[k])
    assert pr.fidelity["combined"] == "bound_over_windows" and pr.meta["provenance"]["truncated"]["windows_kept"] == 1
    with pytest.raises(ValueError, match="window"):                       # a prefix inside a later window: no re-projection across windows
        two.prefix(0.015)
    pp = one.prefix(0.015)
    assert pp.n_windows == 1 and pp.n_t == 31 and pp.fidelity["within_2x_floor"] and "err_reprojection" in pp.fidelity
    seq = d.pgse([[1, 0, 0]], 0.003, 0.009, gradient_strengths=0.3, n_t=31, slew_rate=np.inf)   # TE = 15 ms
    npt.assert_allclose(pp.replay(seq, tissue=TIS0, complex_signal=True), one.replay(seq, tissue=TIS0, complex_signal=True),
                        atol=2.0 * pp.fidelity["floor_max"])


def test_the_producer_refuses_a_walk_that_is_not_whole_windows():
    from dmipy_sim.engine.accumulate import window_count
    assert window_count(0.02, 0.01) == (2, 0.01) and window_count(0.02, 0.02) == (1, 0.02) and window_count(0.02, 0.1)[0] == 1
    with pytest.raises(ValueError, match="whole number"):
        window_count(0.015, 0.01)
    with pytest.raises(ValueError, match="whole number"):
        d.simulate_trajectories(8, 2e-9, _geometry(), 0.015, DT, seed=0, require_gpu=False, K=8, window_T=0.01)


def test_the_bound_over_windows():
    a = dict(metric="m", err_max=0.01, floor_max=0.05, err_surface=0.002, floor_surface=0.04, per_family={"PGSE": dict(err_max=0.01, floor_max=0.05)})
    b = dict(metric="m", err_max=0.02, floor_max=0.03, err_surface=0.001, floor_surface=0.06, per_family={"PGSE": dict(err_max=0.02, floor_max=0.03)})
    c = combine_window_fidelity([a, b])
    assert c["err_max"] == pytest.approx(0.03) and c["floor_max"] == pytest.approx(0.05) and c["err_surface"] == pytest.approx(0.003)
    assert c["floor_surface"] == pytest.approx(0.06) and c["per_family"]["PGSE"] == dict(err_max=pytest.approx(0.03), floor_max=pytest.approx(0.05))
    assert c["combined"] == "bound_over_windows" and c["within_2x_floor"] and len(c["windows"]) == 2


def test_a_pack_without_the_table_is_refused():
    with pytest.raises(ValueError, match="walk_params.windows"):
        d.replay.ReplayPack({"pos_x": np.zeros((2, 4), np.float32)}, {"walk_params": {"block": {"h": 1e-3}}}).n_windows


def test_a_pack_with_the_earlier_table_is_refused(packs, tmp_path):
    """A header that names its window table ``walk_params.segments`` declares no window table, and is refused by name."""
    import copy
    from dmipy_sim.replay.replay import read_rpk, write_rpk
    w1, w2, one, two = packs
    meta = copy.deepcopy(two.meta)
    meta["walk_params"]["segments"] = meta["walk_params"].pop("windows")
    write_rpk(tmp_path / "earlier.rpk", two.arrays, meta)
    with pytest.raises(ValueError, match="walk_params.windows"):
        read_rpk(tmp_path / "earlier.rpk").n_windows


def _crushed_train(n_echoes=8, beta=120.0):
    """A refocusing train of 2 ms echoes over 16 ms with a crusher pair about every pulse: echoes in both windows."""
    from dmipy_sim import sequences
    s = sequences.cpmg(n_echoes, 2e-3, beta_deg=beta, gradient_directions=[[1.0, 0, 0]], bvalues=[0.0], n_t_per_echo=40)
    wins = [(float(e.t_s) - 0.5e-3, float(e.t_s) + 0.5e-3) for e in s.rf if e.label in ("refocus", "refocusing")]
    return dataclasses.replace(s, crusher={"windows_s": wins, "n_cycles": 16.0})


@pytest.mark.parametrize("engine", ["numpy", "jax"])
@pytest.mark.parametrize("knobs", [dict(), dict(tissue=TIS, scanner=3.0), dict(tissue=TIS0, b1_scale=0.9), dict(off_resonance_T=2e-7)])
def test_the_bloch_route_hands_the_state_across_windows(packs, knobs, engine):
    """The vector-Bloch propagation over two windows, the refocusing pulse exactly on the join, equals the one-window
    propagation within the two packs' band truncation (the route reads the stored path at the block ends: a
    sampling of the same function the scalar route integrates); the state every walker leaves a window in is the
    state it starts the next in."""
    w1, w2, one, two = packs
    seq = _seq()                                                                  # the 180 at 10 ms: the boundary block end
    a = one.replay_bloch(seq, complex_signal=True, backend=engine, **knobs)
    b = two.replay_bloch(seq, complex_signal=True, backend=engine, **knobs)
    npt.assert_allclose(b, a, atol=_bound(one, two))


def test_a_crushed_train_spans_the_windows(packs):
    w1, w2, one, two = packs
    train = _crushed_train()
    a = one.replay_bloch(train, tissue=TIS0, complex_signal=True, backend="jax")
    b = two.replay_bloch(train, tissue=TIS0, complex_signal=True, backend="jax")
    assert a.shape == (1, 8)
    npt.assert_allclose(b, a, atol=_bound(one, two))


def test_the_bloch_route_reads_at_the_acquisitions_readout(packs):
    """A short acquisition on a long pack is read at its own readout, not at the walk's end: the two-window pack,
    the one-window pack and the first window alone agree, and with an ideal pulse the scalar route agrees too, to
    what the stored function holds between its block ends (the Bloch route reads it at the block ends, the scalar
    route integrates it)."""
    w1, w2, one, two = packs
    short = d.pgse([[1, 0, 0]], 0.002, 0.006, gradient_strengths=0.3, n_t=21, slew_rate=np.inf)   # TE = 10 ms of 20
    a = one.replay_bloch(short, tissue=TIS0, complex_signal=True)
    b = two.replay_bloch(short, tissue=TIS0, complex_signal=True)
    c = two.window(0).replay_bloch(short, tissue=TIS0, complex_signal=True)
    npt.assert_allclose(b, a, atol=_bound(one, two)); npt.assert_allclose(c, b, atol=1e-8)
    npt.assert_allclose(np.abs(a), one.replay(short, tissue=TIS0), rtol=2e-3)


def test_a_walk_continues_from_its_end_and_appends_as_windows():
    """A 10 ms pack extended by 10 ms with a fresh seed has the layout of a 20 ms walk stored in two windows, its
    second window starts exactly where the first ended, its table records both walks, its certificate is the bound
    over the windows, and it replays a 20 ms acquisition as the one-walk pack does within the floor."""
    from dmipy_sim.replay.continuation import append_windows, continue_walk, end_state
    g = d.Cylinder(2e-6, (0, 0, 1))
    kw = dict(license="x", citation="x", blt_temporal_K=16, window_T=0.01)
    walk = dict(seed=0, require_gpu=False, K=16, window_T=0.01)
    two = _float_pack(d.simulate_trajectories(2000, 2e-9, g, 0.02, 5e-4, **walk), id="t/two", **kw)
    one = _float_pack(d.simulate_trajectories(2000, 2e-9, g, 0.01, 5e-4, **walk), id="t/one", **kw)
    r_end, pool = end_state(one)
    assert pool is not None and np.all(pool == 1)                                   # every walker in the lumen
    cont = continue_walk(one, 0.01, seed=7, require_gpu=False)
    npt.assert_array_equal(np.asarray(cont.position_coeffs[:, 0, 0, :], np.float32), r_end.astype(np.float32))
    ext = append_windows(one, cont, seed=7)
    assert sorted(ext.arrays) == sorted(two.arrays) and ext.n_t == two.n_t == 41 and ext.n_windows == 2
    assert ext.windows["walks"] == [dict(first=0, last=0, seed=0), dict(first=1, last=1, seed=7)]
    npt.assert_array_equal(ext.window(1).r0, r_end.astype(np.float32))
    assert ext.fidelity["combined"] == "bound_over_windows" and len(ext.fidelity["windows"]) == 2
    assert ext.meta["provenance"]["continuations"][0]["seed"] == 7
    seq = _seq()
    tis = Tissue(T2={"extra": 0.05, "intra": 0.02}, rho2=1e-5)
    a, b = two.replay(seq, tissue=tis), ext.replay(seq, tissue=tis)
    assert np.abs(a - b).max() < 2.0 * max(two.fidelity["floor_max"], ext.fidelity["floor_max"])
    with pytest.raises(ValueError, match="whole number"):
        continue_walk(one, 0.015, seed=1, require_gpu=False)
    longer = ext.extend(0.02, seed=9, require_gpu=False)
    assert longer.n_windows == 4 and longer.n_t == 81 and longer.windows["walks"][-1] == dict(first=2, last=3, seed=9)
    long_seq = d.pgse([[1, 0, 0]], 0.005, 0.02, gradient_strengths=0.2, n_t=61, slew_rate=np.inf)   # 30 ms: three windows
    assert 0.0 < float(longer.replay(long_seq)[0]) < 1.0


def test_shards_of_a_windowed_walk_merge(packs):
    """Two shards of the same substrate, each stored in two windows with a field path channel, merge into one pack
    of two windows whose every window's scale table carries a block axis and whose replay is the weight-combined shards'."""
    from dmipy_sim.replay.bank import merge_packs
    w1, w2, one, two = packs
    b = _float_pack(_walk(0.01, 19, seed=5), id="t/b", K=19, blt_temporal_K=19, susc_path_K=16, window_T=0.01, **_kw())
    merged = merge_packs([two, b], id="t/merged")
    assert merged.n_windows == 2 and merged.n_walkers == 2 * N_W and merged.arrays["s1/susc_path_scale"].shape[0] == 2
    seq = _seq()
    Sa, Sb = two.replay(seq, tissue=TIS, scanner=3.0, complex_signal=True), b.replay(seq, tissue=TIS, scanner=3.0, complex_signal=True)
    npt.assert_allclose(merged.replay(seq, tissue=TIS, scanner=3.0, complex_signal=True), (Sa + Sb) / 2.0, atol=1e-6)


def _float_pack(*a, **k):
    """``build_replay_pack`` in the float containers: this module compares packs and replays to the bit, which the
    derived integer default does not promise."""
    from dmipy_sim.replay.bank import build_replay_pack
    k.setdefault("position_container", None); k.setdefault("blt_container", None); k.setdefault("field_container", None)
    return build_replay_pack(*a, **k)
