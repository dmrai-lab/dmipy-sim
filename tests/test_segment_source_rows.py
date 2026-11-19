"""Time segments declared to the walk: per walker and segment the walk accumulates ``int_a^b (x(t) - x(0)) dt`` beside
its bands (``segments=`` of ``_walk_geometry``, ``CoefficientWalk.segment_rows``).

* The row IS the trapezoid of the walker's positions at the walk's own step: on a walk of one step per block whose
  every position is kept, the accumulated row equals the trapezoid of ``raw_positions`` walker by walker, to ``1e-12``
  of ``|row| + sum |w dx|`` in float64 accumulation and to the summation depth's float32 rounding in float32.
* Paired on the same walkers, the segment contracted from the first ``k`` bands approaches the row as ``k`` rises.
* The declaration adds rows and changes nothing else: the walk's every array is the same bits with and without it.
* Each refusal is named, before any step.
* The time-segment layout writer reads the rows as a source of its own: a partition of declared segments takes
  ``gamma`` times the rows, and when every partition does, the certificate's terms carry no band truncation.
"""
import dataclasses

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine.accumulate import RING
from dmipy_sim.engine.core import _walk_geometry, walk_cost

D = 2e-9
RADIUS = 5e-6
T = 0.01
H = 2.5e-5                                      # the block, equal to the step (sub_steps=1): 400 blocks
SEGMENTS = ((1e-3, 3e-3), (6e-3, 8e-3))
N = 2000
K = 16


def _geometry():
    return d.Cylinder(RADIUS, (0, 0, 1))


def _walk(dtype, n=N, segments=SEGMENTS, **kw):
    kw = dict(dict(seed=11, window_T=T, block_h=H, sub_steps=1, K=K, tiers=(), raw_subset=n, require_gpu=False), **kw)
    return _walk_geometry(n, D, _geometry(), T, segments=segments, accumulation_dtype=dtype, **kw)


@pytest.fixture(scope="module")
def walk32():
    return _walk("float32")


def _trapezoid(w):
    """``(n, n_seg, 3)`` float64: ``int_a^b (x(t) - x(0)) dt`` with ``x`` linear between the kept positions (every step's
    here), and ``sum_i |w(t_i) dx_i|`` over each segment's weight, both formed outside the walk."""
    x = np.asarray(w.raw_positions, np.float64)
    x = x - x[:, :1]
    h = float(w.h)
    dx = np.diff(x, axis=1)
    tm = (np.arange(dx.shape[1]) + 0.5) * h
    trap, scale = [], []
    for a, b in SEGMENTS:
        ia, ib = int(round(a / h)), int(round(b / h))
        trap.append(h * (x[:, ia + 1:ib].sum(axis=1) + 0.5 * (x[:, ia] + x[:, ib])))
        wt = np.clip(b - np.maximum(a, tm), 0.0, None)
        scale.append(np.abs(wt[None, :, None] * dx).sum(axis=1))
    return np.stack(trap, axis=1), np.stack(scale, axis=1)


def _float32_depth(w):
    """The roundings on the deepest path of one step's term into a float32 row: the increment (1), the block's moment
    sum over its ``L`` steps (``L - 1``), the table's product (1), a ring's product of ``2 RING`` terms (``2 RING -
    1``), the chain over the rings (one add per ring), the sum over the windows (``n_windows - 1``), the block length
    in float32 and its product (2)."""
    n_bw = int(w.n_blocks) // int(w.n_windows)
    n_rings = -(-n_bw // RING)
    L = int(w.sub_steps)
    return 1 + (L - 1) + 1 + (2 * RING - 1) + n_rings + (int(w.n_windows) - 1) + 2


def test_the_row_is_the_trapezoid_of_the_raw_positions_in_float32(walk32):
    w = walk32
    assert w.sub_steps == 1 and w.raw_positions.shape == (N, w.n_blocks + 1, 3)
    rows = np.asarray(w.segment_rows)
    assert rows.dtype == np.float32 and rows.shape == (N, len(SEGMENTS), 3)
    trap, scale = _trapezoid(w)
    u = np.finfo(np.float32).eps / 2.0
    c = _float32_depth(w)
    bound = c * u * scale + 1e-12 * (np.abs(trap) + scale)
    err = np.abs(rows.astype(np.float64) - trap)
    assert np.all(err <= bound), f"worst error / bound {np.max(err / bound):.3g} (depth {c})"
    assert np.max(err / scale) > 0.0                                  # float32 rounding is there, within its depth


def test_a_segment_in_a_later_window_is_referred_to_the_walks_start():
    """Over two storage windows the second segment lies in the second window; its row is still ``int (x(t) - x(0))
    dt`` from the walk's start, the first window's displacement entering through the weight's constant."""
    w = _walk("float32", n=500, window_T=T / 2)
    assert w.n_windows == 2
    trap, scale = _trapezoid(w)
    err = np.abs(np.asarray(w.segment_rows, np.float64) - trap)
    bound = _float32_depth(w) * np.finfo(np.float32).eps / 2.0 * scale + 1e-12 * (np.abs(trap) + scale)
    assert np.all(err <= bound), f"worst error / bound {np.max(err / bound):.3g}"


def test_a_window_keeps_the_rows_where_they_are_referred_to_its_start(walk32):
    """A one-window walk's window 0 is the walk, rows included; a window of a walk of several carries none."""
    assert walk32.window(0).segment_rows is walk32.segment_rows
    w = _walk("float32", n=50, window_T=T / 2)
    assert w.segment_rows is not None and w.window(0).segment_rows is None and w.window(1).segments is None


def test_the_row_is_the_trapezoid_of_the_raw_positions_in_float64(x64):
    w = _walk("float64")
    rows = np.asarray(w.segment_rows)
    assert rows.dtype == np.float64
    trap, scale = _trapezoid(w)
    err = np.abs(rows - trap)
    assert np.all(err <= 1e-12 * (np.abs(rows) + scale)), f"worst {np.max(err / (np.abs(rows) + scale)):.3g}"
    assert w.segments["precision"] == "float64"


def test_the_contracted_bands_approach_the_row_on_the_same_walkers(walk32):
    """Per walker, the segment read from the first ``k`` bands differs from the row by the bands' truncation, a
    deterministic difference on the same walkers that falls as ``k`` rises."""
    from dmipy_replay.replay.compression import segment_projection
    w = walk32
    rows = np.asarray(w.segment_rows, np.float64)
    C = np.array(np.asarray(w.position_coeffs, np.float64)[:, 0])
    C[:, 0, :] = 0.0
    rms = []
    for k in (4, 8, 16):
        m = np.einsum("wjc,sj->wsc", C[:, :k + 2], segment_projection(np.asarray(SEGMENTS), T, k))
        rms.append(float(np.sqrt(np.mean((m - rows) ** 2))))
    assert rms[0] > rms[1] > rms[2] > 0.0, rms


@pytest.mark.parametrize("raw", [0, 40])
def test_the_declaration_changes_no_other_bit(raw):
    """The bands, the endpoints, the quadratic variation, the contact, the occupancy, the raw subset and the block
    record of a walk with segments are those of the same walk without them, to the bit."""
    kw = dict(seed=5, window_T=T, block_h=1e-4, K=K, tiers="all", raw_subset=raw, require_gpu=False)
    a = _walk_geometry(200, D, _geometry(), T, **kw)
    b = _walk_geometry(200, D, _geometry(), T, segments=SEGMENTS, **kw)
    assert a.segment_rows is None and a.segments is None and b.segment_rows.shape == (200, 2, 3)
    for f in dataclasses.fields(a):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if f.name in ("segment_rows", "segments", "geometry", "run", "spec", "field_basis"):
            continue
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            assert va is not None and vb is not None and np.array_equal(va, vb), f.name
        elif f.name == "work":
            assert {k: v for k, v in va.items() if "second" not in k} == {k: v for k, v in vb.items() if "second" not in k}
        else:
            assert va == vb, f.name


class _Backend:
    """A backend object that would walk anything: what refuses the rows is the producer."""
    name = "elsewhere"

    def refuses(self, request):
        return None

    def walk_batch(self, request, r0, keys):
        raise AssertionError("a refused walk walks nothing")


@pytest.mark.parametrize("segments, match", [
    (((1e-3, 3.01e-3),), "segment 0 ends at 0.00301 s, off the walk's block grid"),
    (((4e-3, 6e-3),), r"crosses the boundary of the storage window at 0.005 s"),
    (((3e-3, 3e-3),), "is empty or reversed"),
    (((3e-3, 1e-3),), "is empty or reversed"),
    (((1e-3, 3e-3), (2e-3, 4e-3)), "segment edges are not increasing"),
    (((8e-3, 1.1e-2),), "lies outside the walk"),
    ((), "a non-empty sequence of"),
])
def test_each_refusal_is_named_before_any_step(segments, match):
    with pytest.raises(ValueError, match=match):
        _walk_geometry(8, D, _geometry(), T, window_T=T / 2, block_h=H, K=K, tiers=(), require_gpu=False,
                       segments=segments)


def test_order_zero_and_a_backend_are_refused_by_name():
    with pytest.raises(ValueError, match="a block order of at least 1"):
        _walk_geometry(8, D, _geometry(), T, block_h=H, K=K, block_order=0, tiers=(), require_gpu=False, segments=SEGMENTS)
    for call in (_walk_geometry, walk_cost):
        with pytest.raises(ValueError, match=r"backend 'elsewhere': this walk uses the time segments' rows \(segments=\)"):
            call(8, D, _geometry(), T, block_h=H, K=K, tiers=(), backend=_Backend(), segments=SEGMENTS)


def test_the_cost_counts_the_rows_accumulators():
    with_rows = walk_cost(8, D, _geometry(), T, block_h=H, K=K, tiers=(), segments=SEGMENTS)
    without = walk_cost(8, D, _geometry(), T, block_h=H, K=K, tiers=())
    assert with_rows.accumulator_bytes_per_walker - without.accumulator_bytes_per_walker == len(SEGMENTS) * 3 * 4


def test_the_rows_travel_with_the_walk(walk32, tmp_path):
    """Saved and loaded, taken and joined, the rows and their declaration stay with their walkers."""
    from dmipy_sim.coefficient_walk import CoefficientWalk, join_walks
    w = walk32.take(np.arange(300))
    w.save(tmp_path / "w.safetensors")
    back = CoefficientWalk.load(tmp_path / "w.safetensors")
    assert np.array_equal(back.segment_rows, w.segment_rows) and back.segments == w.segments
    head = w.take(np.arange(100))
    tail = dataclasses.replace(w.take(np.arange(100, 300)), first_walker=100, raw_positions=None)
    assert np.array_equal(join_walks([head, tail]).segment_rows, w.segment_rows)


def test_the_layout_writer_reads_the_rows_as_their_own_source(tmp_path):
    """A partition of declared constant segments takes ``gamma`` times the walk's rows (no band, no band check); with
    every partition so, the source is ``walk_segments`` and the certificate's terms carry no truncation. A partition the
    walk did not declare is read from the bands."""
    import dmipy_replay as dr
    from dmipy_analytic.constants import GAMMA
    from dmipy_replay.phantom.grid import Grid
    from dmipy_replay.replay.time_segments import Partition
    from dmipy_sim.replay.bank import build_replay_pack
    from dmipy_sim.replay.time_segments import segment_rows, write_time_segment_layout
    from tests._envelope import stated
    TE, n_t = 0.02, 201
    se = dr.pgse([[0, 0, 1]], 4e-3, 10e-3, gradient_strengths=0.1, TE=TE, n_t=n_t, slew_rate=np.inf)
    part = Partition.of("pgse", se)
    other = Partition.of("other", dr.pgse([[0, 0, 1]], 3e-3, 10e-3, gradient_strengths=0.1, TE=TE, n_t=n_t, slew_rate=np.inf))
    w = _walk_geometry(400, D, _geometry(), TE, seed=7, window_T=TE, block_h=1e-4, require_gpu=False,
                       segments=part.segments)
    w = dataclasses.replace(w, envelope=stated(of=w)["envelope"])
    r0 = np.asarray(w.position_coeffs)[:, 0, 0, :]
    lo, hi = r0.min(0), r0.max(0)
    vs = tuple((hi - lo) * 1.001 + 1e-9)
    grid = Grid(shape=(1, 1, 1), voxel_size_m=vs, origin_m=tuple((lo + hi) / 2))
    pk = build_replay_pack(w, id="t/segment-rows", license="x", citation="x", sigma_star=0.02, planned_floor=0.02,
                           voxel_grid=grid)
    rec = segment_rows(w, {"pgse": part}, str(tmp_path / "rows"), grid=grid, pack=pk, g_max=1.0)
    assert rec["source"]["kind"] == "walk_segments" and rec["partitions"]["pgse"]["vectors"] == "accumulated"
    terms, pk_terms = rec["certificate"]["terms"], pk.meta["fidelity"]["terms"]
    assert "truncation" in pk_terms and "positions" in pk_terms["quadrature"]
    assert "truncation" not in terms and "positions" not in terms.get("quadrature", {})
    assert terms["rows"]["carries"] == ["step_quadrature", "rounding"] and terms["rounding"] == pk_terms["rounding"]
    reach = np.asarray(rec["partitions"]["pgse"]["reach"])         # by hand: gamma^2 reach^2 (b - a) D dt^2 / 6, halved
    want = 0.5 * sum(GAMMA ** 2 * r ** 2 * (b - a) * D * w.dt_sim ** 2 / 6.0 for r, (a, b) in zip(reach, part.segments))
    assert terms["step_quadrature"] == pytest.approx(want, rel=1e-12) and want > 0.0
    stored = np.load(tmp_path / "rows" / "p_pgse.npy")
    order = np.lexsort((np.zeros(w.n_walkers, np.uint8), np.zeros(w.n_walkers, np.int64)))
    assert np.array_equal(stored, (GAMMA * np.asarray(w.segment_rows, np.float64)).astype(stored.dtype)[order])
    with pytest.raises(ValueError, match="a layout from segment rows with background=True is refused"):
        segment_rows(w, {"pgse": part}, str(tmp_path / "bg"), grid=grid, pack=pk, g_max=1.0, background=True)
    assert "systematic" in terms["rows"]["step_quadrature"]["nature"]
    man = write_time_segment_layout([str(tmp_path / "rows")], str(tmp_path / "lay"))
    assert man["source"]["kind"] == "walk_segments" and "truncation" not in man["certificate"]["terms"]
    mixed = segment_rows(w, {"pgse": part, "other": other}, str(tmp_path / "mixed"), grid=grid, pack=pk, g_max=1.0)
    assert mixed["source"]["kind"] == "walk"
    assert mixed["partitions"]["pgse"]["vectors"] == "accumulated" and mixed["partitions"]["other"]["vectors"] == "bands"


def test_the_step_quadrature_is_the_free_walks_signal_excess():
    """The constant and the power of :func:`step_quadrature_term`, against free diffusion: two adjacent square lobes of
    one step each (``delta = Delta = dt``) along x, walked at the step ``dt`` and at ``dt / 4``. The walk's phase is the
    trapezoid of its positions, so its variance is the continuous path's less ``V`` and its signal is the exact
    ``exp(-b D)`` times ``exp(V / 2)``, with ``V / 2`` the term at the lobes' level (``V`` falls as ``dt^2``: ``V / 32``
    at ``dt / 4``). The two walks cannot be paired (a finer step draws other increments), so each is held to the exact
    free signal times its predicted excess, and their difference to ``S (exp(V / 2) - exp(V / 32))``, within three
    standard errors (the walkers' spread of ``cos phi``; the step law's matched moments leave the non-Gaussian part of a
    step's phase below 1e-4 here). A term off by a factor two (a wrong ``/ 6`` or power of ``dt``) moves the
    difference by 0.024, ten standard errors."""
    from types import SimpleNamespace
    from dmipy_analytic.constants import GAMMA
    from dmipy_sim.replay.time_segments import step_quadrature_term
    dt, G, n = 1e-3, None, 20000
    segs = ((0.0, dt), (dt, 2 * dt))
    b_unit = GAMMA ** 2 * dt ** 3 * 2.0 / 3.0                       # b / G^2 of adjacent square lobes, Delta = delta
    G = float(np.sqrt(0.25 / (b_unit * D)))                          # b D = 0.25
    S_exact = float(np.exp(-b_unit * G ** 2 * D))
    sig = {}
    for h in (dt, dt / 4):
        w = _walk_geometry(n, D, d.FreeDiffusion(), 2 * dt, seed=3, window_T=2 * dt, block_h=h, sub_steps=1, K=4,
                           tiers=(), require_gpu=False, segments=segs, accumulation_dtype="float32")
        r = np.asarray(w.segment_rows, np.float64)
        c = np.cos(GAMMA * G * (r[:, 0, 0] - r[:, 1, 0]))
        V = 2.0 * step_quadrature_term(SimpleNamespace(segments=segs), [G, G], D, w.dt_sim)
        sig[h] = (c.mean(), c.std(ddof=1) / np.sqrt(n), V)
    (s1, e1, V1), (s4, e4, V4) = sig[dt], sig[dt / 4]
    assert V4 == pytest.approx(V1 / 16, rel=1e-12) and V1 / 2 == pytest.approx(0.25 / 4, rel=1e-9)
    assert abs(s1 - S_exact * np.exp(V1 / 2)) <= 3 * e1
    assert abs(s4 - S_exact * np.exp(V4 / 2)) <= 3 * e4
    assert abs((s1 - s4) - S_exact * (np.exp(V1 / 2) - np.exp(V4 / 2))) <= 3 * np.hypot(e1, e4)
    assert abs((s1 - s4) - S_exact * (np.exp(V1 / 4) - np.exp(V4 / 4))) > 3 * np.hypot(e1, e4)   # half the term fails
