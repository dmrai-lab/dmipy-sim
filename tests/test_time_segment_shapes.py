"""Shaped terms of the time-segment layout (dmipy-sim#807): a partition is terms, each an interval and a shape psi
(a constant segment is psi = 1; a sampled shape is a trapezoid with its ramps, a cosine, a designed waveform; terms may
overlap, a tensor encoding is three shapes on one interval). The vector of a term is the bands against psi 1_[a,b]
(exact), a sequence fits when its effective gradient is sum_s psi_s(t) a_s with a residual inside the layout's
rounding, and the image of every such sequence is the sibling pack's replay within the layout's rounding term."""
from __future__ import annotations
import dmipy_replay.replay.compression as _r_compression
import json
import os
import shutil

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine.core import _walk_geometry
from dmipy_replay.acquisition.waveforms import calc_b
from dmipy_replay.constants import GAMMA
from dmipy_sim.replay import compression as cx
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_replay.replay.time_segments import Partition, Shape, _RTOL
from dmipy_sim.replay.time_segments import (
    TimeSegmentLayout,
    _walk_view,
    level_bound,
    rounding_eps,
    segment_rows,
    segment_vectors,
    write_time_segment_layout,
)

D0 = 2e-9
G_MAX = 1.0                                                          # T/m: the envelope's gradient (the walk names no class)
N_T, TE = 501, 0.05                                                  # a 0.1 ms grid over a 50 ms echo
BOX, TRAP, COS = "box", "trap", "cos"


# ------------------------------------------------------------------ the sequences
def _templates():
    """A square PGSE, a PGSE with 0.25 ms ramps, a cosine OGSE (one 50 Hz period each side of the 180, its edges ramped)
    and a tensor encoding's three shapes (three whole sines on one interval before the 180, one per row along x)."""
    box = d.pgse([[0, 0, 1]], 4e-3, 10e-3, gradient_strengths=0.1, TE=TE, n_t=N_T, slew_rate=np.inf)
    trap = d.pgse([[0, 0, 1]], 8e-3, 20e-3, gradient_strengths=0.05, TE=TE, n_t=N_T, slew_rate=200.0)
    cos = d.ogse([[0, 0, 1]], 50.0, 0.02, shape="cosine", gradient_strengths=0.05, TE=TE, n_t=N_T, slew_rate=200.0)
    i0, i1 = 60, 210
    k = np.arange(i1 - i0)
    G = np.zeros((3, N_T, 3), np.float32)
    for j in range(3):
        G[j, i0:i1, 0] = 0.05 * np.sin(2.0 * np.pi * (j + 1) * k / (i1 - i0))
    tensor = d.ScannerSequence(G=G, dt=box.dt, rf=box.rf, readout=box.readout)
    return {BOX: box, TRAP: trap, COS: cos, "tensor": tensor}


def _partitions():
    t = _templates()
    i0, i1 = 60 * t["tensor"].dt, 210 * t["tensor"].dt
    return {BOX: Partition.of(BOX, t[BOX]), TRAP: Partition.from_sequence(TRAP, t[TRAP]),
            COS: Partition.from_sequence(COS, t[COS]),
            "tensor": Partition.from_sequence("tensor", t["tensor"], terms=[(i0, i1)] * 3, rows=[0, 1, 2])}


def _rows(template, G):
    """A sequence of ``template``'s grid, RF schedule and readout playing the rows ``G`` ``(n_meas, n_t, 3)``."""
    return d.ScannerSequence(G=G, dt=template.dt, rf=template.rf, readout=template.readout)


def _play(template, directions, bvalues):
    """``template`` (one row, one direction) played along each of ``directions`` at the amplitude that gives its b: the
    one shape at another amplitude and direction (a ramp of fixed duration), as a class plays it."""
    G = np.asarray(template.G, np.float64)[0]
    prof = G[:, int(np.abs(G).max(axis=0).argmax())]
    prof = prof / np.abs(prof).max()                                  # the unit shape; the amplitude is set by b
    u = np.asarray(directions, np.float64); u = u / np.linalg.norm(u, axis=1)[:, None]
    unit = float(calc_b(template)[0]) / float(np.abs(G).max()) ** 2
    amp = np.sqrt(np.asarray(bvalues, np.float64) / unit)
    return _rows(template, np.stack([prof[:, None] * (a * v)[None, :] for a, v in zip(amp, u)]))


def _tensor(template, R, weights):
    """The tensor encoding of weights ``a_k`` along the axes ``R[:, k]``: ``G(t) = sum_k a_k psi_k(t) R e_k``."""
    G = np.asarray(template.G, np.float64)[:, :, 0]                  # (3, n_t): psi_k
    return _rows(template, np.stack([np.einsum("k,kt,ck->tc", w, G, r) for r, w in zip(R, weights)]))


# ------------------------------------------------------------------ the fixture: a small walk, its pack and the layout
@pytest.fixture(scope="module")
def small(tmp_path_factory):
    """A cylinder walk (600 walkers, 50 ms), its pack on a 2 x 2 voxel grid, the segment rows of the four partitions from
    the walk and the layout."""
    from dmipy_replay.phantom.grid import Grid
    tmp = tmp_path_factory.mktemp("shapes")
    w = _walk_geometry(600, D0, d.Cylinder(radius=4e-6, orientation=(0, 0, 1)), TE, seed=3, window_T=TE, require_gpu=False)
    r0 = np.asarray(w.position_coeffs)[:, 0, 0, :]
    lo, hi = r0.min(0), r0.max(0)
    vs = np.array([(hi[0] - lo[0]) / 2 * 1.001, (hi[1] - lo[1]) / 2 * 1.001, 1e-5])
    grid = Grid(shape=(2, 2, 1), voxel_size_m=tuple(vs), origin_m=(lo[0] - 1e-9 + vs[0] / 2, lo[1] - 1e-9 + vs[1] / 2, 0.0))
    pk = build_replay_pack(w, id="t/shapes", license="x", citation="x", sigma_star=0.02, planned_floor=0.02, voxel_grid=grid)
    parts = _partitions()
    rec = segment_rows(w, parts, str(tmp / "rows"), grid=grid, pack=pk, g_max=G_MAX)
    man = write_time_segment_layout([str(tmp / "rows")], str(tmp / "layout"))
    return dict(walk=w, pack=pk, view=_walk_view(w), grid=grid, parts=parts, rec=rec, man=man, tmp=tmp,
                lay=TimeSegmentLayout(str(tmp / "layout")), tmpl=_templates())


def _per_voxel(src, grid, seq):
    """``(n_vox, n_meas)``: the per-voxel ensemble magnitude of ``src``'s replay (walker by walker, start voxels)."""
    w, ew, E = src.walker_signals(seq, tol=None)
    ijk, _ = grid.bin(src.r0); v = np.ravel_multi_index(ijk.T, grid.shape)
    num = np.stack([np.bincount(v, (ew * E[:, k]).real, minlength=grid.n_voxels)
                    + 1j * np.bincount(v, (ew * E[:, k]).imag, minlength=grid.n_voxels) for k in range(E.shape[1])], 1)
    return np.abs(num / np.bincount(v, w, minlength=grid.n_voxels)[:, None])


def _term(sigma2, floor):
    """The rounding's term on a voxel's signal: ``3 sigma`` times the Monte-Carlo floor plus the bias ``sigma^2 / 2``."""
    return 3.0 * np.sqrt(sigma2) * floor + 0.5 * sigma2


def _assert_image(M, name, seq):
    """The layout's image of ``seq`` on partition ``name`` is the walk's own float replay within the partition's
    rounding term, and the pack's within it plus the pack's quantisation (zero from the walk), well inside the floor."""
    S, floor = M["lay"].image(seq, partition=name)
    exact = _per_voxel(M["view"], M["grid"], seq)
    ref = _per_voxel(M["pack"], M["grid"], seq)
    S = S.reshape(exact.shape); fl = float(np.nanmax(floor))
    sigma2 = M["man"]["partitions"][name]["rounding"]["variance"]
    assert np.abs(S - exact).max() < _term(sigma2, fl), name
    assert np.abs(S - ref).max() < 0.25 * fl, name
    return np.abs(S - exact).max()


# ------------------------------------------------------------------ the projection and the fit, without a walk
def test_a_shaped_projection_is_the_bands_of_the_sample_and_hold_waveform():
    """``segment_projection`` of a term with a shape is exactly :func:`band_projection` of the waveform ``psi`` held on
    the grid inside the term and 0 outside: no quadrature stands between the stored shape and the stored vector. A
    segment with no shape is the indicator (the same rows as before), also inside a window that clips the term."""
    rng = np.random.default_rng(1)
    dt, n_t, K, T = 1e-4, 300, 24, 0.03
    psi = rng.normal(size=120); psi /= np.abs(psi).max()
    wf = np.zeros((1, n_t, 1)); wf[0, 40:160, 0] = psi
    ref = _r_compression.band_projection(wf, dt, T, K)[0, :, 0]
    got = _r_compression.segment_projection([(40 * dt, 160 * dt)], T, K, shapes=[psi])[0]
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-13 * np.abs(ref).max())
    ind = _r_compression.segment_projection([(40 * dt, 160 * dt), (0.0, 1e-3)], T, K)
    assert np.array_equal(ind, _r_compression.segment_projection([(40 * dt, 160 * dt), (0.0, 1e-3)], T, K, shapes=[None, None]))
    part = _r_compression.segment_projection([(40 * dt, 160 * dt)], 0.01, K, t0=0.005, shapes=[psi])[0]       # a window clips the term
    np.testing.assert_allclose(part, _r_compression.band_projection(wf, dt, 0.01, K, t0=0.005)[0, :, 0], rtol=0, atol=1e-13)


def test_a_shape_is_a_stated_normalised_record_and_a_term_has_one():
    """A shape reads and writes its record; its values are normalised (``max|psi| = 1``), its area is zero for whole
    periods and its weight ``int|psi|`` is the duration for a constant; a record without ``kind`` or a field of its
    kind, an unknown kind, an unnormalised or a mis-sized record are each refused naming what is wrong."""
    s = Shape.sampled([2.0, -4.0, 1.0, 4.0])
    assert max(abs(v) for v in s.values) == 1.0 and Shape.from_meta(s.to_meta(), "t") == s
    assert Shape.from_meta(Shape.constant().to_meta(), "t") == Shape.constant()
    k = np.arange(100)
    sine = Shape.sampled(np.sin(2 * np.pi * 2 * k / 100))
    assert sine.area(0.01) == 0.0 and sine.weight(0.01) == pytest.approx(0.01 * np.abs(np.sin(2 * np.pi * 2 * k / 100)).mean() / np.abs(np.sin(2 * np.pi * 2 * k / 100)).max())
    assert Shape.constant().area(0.01) == Shape.constant().weight(0.01) == 0.01
    assert Shape.of_samples([0.5, 0.5, 0.5]).is_constant and not Shape.of_samples([0.5, 0.25]).is_constant
    for rec, field in ((None, "shape record"), ({}, "shape record"), ({"kind": "ramp"}, "'ramp'"),
                       ({"kind": "sampled", "values": [1.0]}, "'cells'"), ({"kind": "sampled", "cells": 1}, "'values'"),
                       ({"kind": "sampled", "cells": 2, "values": [0.5, 0.25]}, "normalised"),
                       ({"kind": "sampled", "cells": 3, "values": [1.0, 0.5]}, "normalised")):
        with pytest.raises(ValueError, match=field):
            Shape.from_meta(rec, "term 0")


def test_a_sequence_fits_its_shapes_to_the_float_rounding_the_constant_path_is_held_to():
    """The least squares fit of shaped terms is held to the tolerance of the constant path (``_RTOL`` of the peak): a
    residual off the shapes at half of it fits and leaves the levels, at twice of it is refused naming the partition,
    the row, the term and the residual's relative norm; the same tolerance refuses a constant segment that is not
    constant."""
    p = _partitions()[TRAP]
    tmpl = _templates()[TRAP]
    G = np.asarray(tmpl.G, np.float64)
    design = p.design(tmpl.dt, tmpl.n_t)[0]
    i0 = int(round(p.segments[0][0] / tmpl.dt))
    bump = np.zeros(tmpl.n_t); bump[i0 + 3:i0 + 20] = 1.0
    bump -= design @ np.linalg.lstsq(design, bump, rcond=None)[0]        # off the shapes: the levels (and the net moment) stay
    r0 = np.zeros_like(G); r0[0, :, 2] = bump / np.abs(bump).max()
    peak = np.abs(G).max()
    L0 = p.levels(tmpl)
    np.testing.assert_allclose(p.levels(tmpl.with_gradient(G + 0.5 * _RTOL * peak * r0)), L0, atol=1e-6)
    with pytest.raises(ValueError, match=rf"partition '{TRAP}': row 0 leaves a residual of relative norm .* term 0 \["):
        p.levels(tmpl.with_gradient(G + 2.0 * _RTOL * peak * r0))
    box = _partitions()[BOX]
    bt = _templates()[BOX]
    g = np.asarray(bt.G, np.float64); i = int(round(box.segments[0][0] / bt.dt)) + 5
    g[0, i, 2] *= 1.0 - 2.0 * _RTOL
    with pytest.raises(ValueError, match="not constant on segment 0"):
        box.levels(bt.with_gradient(g))


def test_zero_area_terms_leave_no_moment_and_the_class_is_the_box():
    """A cosine over whole periods leaves no moment (area zero), so its levels are free of the refocusing constraint:
    the refocused basis is the identity, the class vertices are the box's, and a constant segment's keep their plane."""
    ps = _partitions()
    zero = Partition.from_sequence("z", _templates()["tensor"], terms=[(6e-3, 21e-3)] * 3, rows=[0, 1, 2])
    assert not zero.areas.any() and zero.weights.min() > 0
    idx, N = zero.refocused_basis()
    np.testing.assert_array_equal(N, np.eye(3))
    idx, V = zero.class_vertices(G_MAX)
    assert V.shape == (8, 3) and np.abs(V).min() == G_MAX
    np.testing.assert_allclose(zero.reach(G_MAX), np.sqrt(3.0) * G_MAX)
    box = ps[BOX]
    np.testing.assert_array_equal(box.areas, box.durations); np.testing.assert_array_equal(box.weights, box.durations)
    idx, V = box.class_vertices(G_MAX)
    assert np.allclose(V @ box.durations[idx], 0.0)
    # b_matrix: symmetric, positive definite, its diagonal gamma^2 int Q_k^2 of each shape's running integral
    B = zero.b_matrix()
    np.testing.assert_allclose(B, B.T, rtol=1e-12)
    assert np.linalg.eigvalsh(B).min() > 0
    k = np.arange(150)
    Q1 = np.concatenate([[0.0], np.cumsum(np.asarray(zero.shapes[0].values))]) * 1e-4          # psi_1's running integral at the samples
    want = GAMMA ** 2 * 1e-4 * ((Q1[:-1] ** 2 + Q1[:-1] * Q1[1:] + Q1[1:] ** 2) / 3.0).sum()
    assert B[0, 0] == pytest.approx(want, rel=1e-9)
    rho = level_bound(zero, np.stack([np.eye(3)] * 3), dict(G_max=G_MAX, b_max=None))
    assert rho == pytest.approx(3 * 3 * G_MAX ** 2)                   # three axes, each the box's worst: sum of l_s^2 = 3 G^2


# ------------------------------------------------------------------ the layout
def test_a_constant_partition_is_bit_identical_whichever_way_it_is_built(small):
    """The vectors of a constant partition are the bands against the indicator, bit for bit what the segment projection
    gave before shapes (``gamma sum C W`` with ``W = segment_projection(S)``); :meth:`Partition.from_sequence` of a
    plateau lobe is the constant term :meth:`Partition.of` reads, and its rows, words and image are the same bytes."""
    M = small
    p = M["parts"][BOX]
    assert p.is_constant and p.is_indicator and p == Partition.of(BOX, _templates()[BOX])
    same = Partition.from_sequence(BOX, _templates()[BOX])
    assert same == p and same.to_meta() == p.to_meta() and all(s.is_constant for s in same.shapes)
    v = M["view"]
    C = _r_compression.read_position_coeffs(v.arrays, dtype=np.float64, cached=False)   # modified in place below: a decode of its own
    x0 = C[:, 0, :].copy(); C[:, 0, :] -= x0
    old = GAMMA * np.einsum("wjc,sj->wsc", C, _r_compression.segment_projection(np.asarray(p.segments), M["walk"].T_window, C.shape[1] - 2))
    new = segment_vectors(v, p)
    assert np.array_equal(old, new)
    rows = str(M["tmp"] / "rows_from_sequence")
    segment_rows(M["walk"], {BOX: same}, rows, grid=M["grid"], pack=M["pack"], g_max=G_MAX)
    one = str(M["tmp"] / "rows_of")
    segment_rows(M["walk"], {BOX: p}, one, grid=M["grid"], pack=M["pack"], g_max=G_MAX)
    for f in sorted(os.listdir(one)):
        if f != "rows.json":
            assert open(os.path.join(one, f), "rb").read() == open(os.path.join(rows, f), "rb").read(), f
    a, b = (json.load(open(os.path.join(r, "rows.json"))) for r in (one, rows))
    for r in (a, b):
        r.pop("created")
    assert a == b
    la, lb = (write_time_segment_layout([r], str(M["tmp"] / n)) for r, n in ((one, "lay_of"), (rows, "lay_fs")))
    for f in ("w.npy", "tiles.npy", f"p_{BOX}.npy"):
        assert open(os.path.join(M["tmp"], "lay_of", f), "rb").read() == open(os.path.join(M["tmp"], "lay_fs", f), "rb").read()
    seq = _play(_templates()[BOX], [[0, 0, 1], [1, 0, 0], [1, 1, 1]], [1e9, 2e9, 3e9])
    Sa, fa = TimeSegmentLayout(str(M["tmp"] / "lay_of")).image(seq)
    Sb, fb = TimeSegmentLayout(str(M["tmp"] / "lay_fs")).image(seq)
    assert np.array_equal(Sa, Sb, equal_nan=True) and np.array_equal(fa, fb, equal_nan=True)
    assert [s["shape"] for s in M["man"]["partitions"][BOX]["segments"]] == [{"kind": "constant"}] * 2


def test_the_image_of_a_trapezoid_class_is_the_replay(small):
    """A PGSE with 0.25 ms ramps, its shape stored per lobe: at several b-values and directions (the one shape at
    another amplitude, so the ramp keeps its duration) the layout's image is the walk's replay within the rounding."""
    M = small
    p = M["lay"].partitions[TRAP]
    assert not p.is_constant and [s.cells for s in p.shapes] == [s.cells for s in M["parts"][TRAP].shapes]
    rng = np.random.default_rng(0)
    u = rng.normal(size=(5, 3))
    seq = _play(M["tmpl"][TRAP], np.concatenate([u, u, [[0, 0, 1]]]), np.r_[np.full(5, 1e9), np.full(5, 2.5e9), 0.0])
    name, L = M["lay"].fit(seq)
    assert name == TRAP
    b = np.einsum("msc,st,mtc->m", L, p.b_matrix(), L)
    np.testing.assert_allclose(b, calc_b(seq), rtol=3e-3, atol=1e5)
    assert _assert_image(M, TRAP, seq) > 0.0


def test_the_image_of_a_cosine_ogse_class_is_the_replay(small):
    """One 50 Hz period each side of the refocusing, its shape stored per block: the image at several b-values and
    directions is the walk's replay within the rounding; the two blocks' levels are the fit of the cosine."""
    M = small
    rng = np.random.default_rng(5)
    u = rng.normal(size=(4, 3))
    seq = _play(M["tmpl"][COS], np.concatenate([u, u, [[1, 0, 0]]]), np.r_[np.full(4, 0.5e9), np.full(4, 2e9), 1e9])
    name, L = M["lay"].fit(seq)
    assert name == COS and L.shape == (9, 2, 3)
    np.testing.assert_allclose(np.einsum("msc,s->mc", L, M["lay"].partitions[COS].areas), 0.0, atol=1e-6)
    b = np.einsum("msc,st,mtc->m", L, M["lay"].partitions[COS].b_matrix(), L)
    np.testing.assert_allclose(b, calc_b(seq), rtol=3e-3)
    _assert_image(M, COS, seq)


def test_a_tensor_encoding_is_three_shapes_on_one_interval(small):
    """Three shapes on one interval, three vectors per walker: linear, planar and spherical weightings at random
    rotations all read the same stored words, and the image equals the walk's replay within the rounding; the b of
    the levels through the layout's b matrix is the b of the sequence."""
    from scipy.spatial.transform import Rotation
    M = small
    p = M["lay"].partitions["tensor"]
    assert len(p.segments) == 3 and len(set(p.segments)) == 1 and not p.is_constant and not p.areas.any()
    R = Rotation.random(9, random_state=4).as_matrix()
    wts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0.5, 0.25], [0.3, 1, 0.7]], float)
    seq = _tensor(M["tmpl"]["tensor"], R, wts)
    b0 = calc_b(seq)
    seq = _rows(M["tmpl"]["tensor"], np.asarray(seq.G, np.float64) * np.sqrt(1.5e9 / b0.max()))
    name, L = M["lay"].fit(seq)
    assert name == "tensor" and L.shape == (9, 3, 3)
    scale = np.sqrt(1.5e9 / b0.max())
    G0 = np.asarray(M["tmpl"]["tensor"].G, np.float64)[:, :, 0]
    peak = G0[np.arange(3), np.abs(G0).argmax(axis=1)]                # a shape is normalised: its level carries the amplitude
    for m in range(9):                                                # the levels are a_k R e_k
        np.testing.assert_allclose(L[m], scale * (wts[m] * peak)[:, None] * R[m].T, atol=2e-6)
    b = np.einsum("msc,st,mtc->m", L, p.b_matrix(), L)
    np.testing.assert_allclose(b, calc_b(seq), rtol=3e-3)
    _assert_image(M, "tensor", seq)


def test_a_sequence_that_matches_no_term_is_refused_by_name(small):
    """Another ramp duration (the support changes), a gradient that is the trapezoid tapered (the shape is not the stored
    one), another cosine frequency, a ramp that differs by a fraction of a sample (0.29 ms for 0.25 ms) and a delivery on
    shaped terms are each refused naming the partition."""
    M = small
    trap = M["tmpl"][TRAP]
    for slew in (100.0, 130.0):
        other = d.pgse([[0, 0, 1]], 8e-3, 20e-3, gradient_strengths=0.05, TE=TE, n_t=N_T, slew_rate=slew)
        with pytest.raises(ValueError, match=rf"{TRAP}: the sequence plays a gradient outside the segments of partition '{TRAP}'"):
            M["lay"].image(_play(other, [[0, 0, 1]], [3e9]), partition=TRAP)
        with pytest.raises(ValueError, match="fits no stored partition"):
            M["lay"].image(_play(other, [[0, 0, 1]], [3e9]))
    tapered = _play(trap, [[0, 0, 1]], [3e9])
    taper = 1.0 + 0.4 * np.cos(2.0 * np.pi * np.arange(N_T) * trap.dt / 0.03)
    tapered = _rows(trap, np.asarray(tapered.G, np.float64) * taper[None, :, None])
    with pytest.raises(ValueError, match=rf"{TRAP}: the sequence's effective gradient does not fit the shapes of partition "
                                         rf"'{TRAP}': row 0 leaves a residual of relative norm .* term [01] "):
        M["lay"].image(tapered, partition=TRAP)
    near = d.pgse([[0, 0, 1]], 8e-3, 20e-3, gradient_strengths=0.05, TE=TE, n_t=N_T, slew_rate=170.0)   # a ramp 0.04 ms longer
    with pytest.raises(ValueError, match=rf"{TRAP}: .*does not fit the shapes of partition '{TRAP}': row 0 leaves a residual of relative norm"):
        M["lay"].image(_play(near, [[0, 0, 1]], [3e9]), partition=TRAP)
    faster = d.ogse([[0, 0, 1]], 100.0, 0.02, shape="cosine", gradient_strengths=0.05, TE=TE, n_t=N_T, slew_rate=200.0)
    with pytest.raises(ValueError, match=rf"{COS}: .*partition '{COS}'"):
        M["lay"].image(_play(faster, [[0, 0, 1]], [2e9]), partition=COS)
    with pytest.raises(ValueError, match="shaped terms"):
        M["lay"].delivery(_play(trap, [[0, 0, 1]], [1e9]), None, partition=TRAP)


def test_a_manifest_term_without_a_shape_record_is_refused_by_name(small, tmp_path):
    """A manifest whose term has no ``shape`` is not read (there is no default shape); nor is one whose shape names no
    kind; rows without ``shapes`` are not consolidated; blocks that disagree on a shape are not either."""
    M = small
    old = str(tmp_path / "old")
    shutil.copytree(M["lay"].path, old)
    path = os.path.join(old, "manifest.json")
    man = json.load(open(path))
    del man["partitions"][COS]["segments"][1]["shape"]
    json.dump(man, open(path, "w"))
    with pytest.raises(ValueError, match=rf"term 1 of partition '{COS}' in the manifest has no shape record"):
        TimeSegmentLayout(old)
    man["partitions"][COS]["segments"][1]["shape"] = {"cells": 3}
    json.dump(man, open(path, "w"))
    with pytest.raises(ValueError, match="no shape record"):
        TimeSegmentLayout(old)
    rows = str(tmp_path / "rows")
    shutil.copytree(str(M["tmp"] / "rows"), rows)
    rec = json.load(open(os.path.join(rows, "rows.json")))
    del rec["partitions"][TRAP]["shapes"]
    json.dump(rec, open(os.path.join(rows, "rows.json"), "w"))
    with pytest.raises(ValueError, match=rf"partition '{TRAP}' has no 'shapes' record"):
        write_time_segment_layout([rows], str(tmp_path / "lay"))
    rows2 = str(tmp_path / "rows2")
    shutil.copytree(str(M["tmp"] / "rows"), rows2)
    rec = json.load(open(os.path.join(rows2, "rows.json")))
    rec["partitions"][TRAP]["shapes"][0] = {"kind": "constant"}
    json.dump(rec, open(os.path.join(rows2, "rows.json"), "w"))
    with pytest.raises(ValueError, match="disagree on the partitions"):
        write_time_segment_layout([str(M["tmp"] / "rows"), rows2], str(tmp_path / "lay2"))


def test_the_manifest_states_each_terms_shape_and_the_level_bound_uses_its_weight(small):
    """Every term of the manifest records its shape; the shape reads back as the partition's; the widths of shaped
    terms are the rule's (``step = sqrt(12 eps^2 / n) / reach``) at the reach the shaped b matrix gives, and the net
    moment tolerance of a fit scales with ``int |psi|``; """
    M = small
    man = M["man"]
    assert "diffusivity" not in man["source"] and man["format"] == "time_segments/2"
    eps = rounding_eps(0.02)
    for name, p in man["partitions"].items():
        for s, seg in enumerate(p["segments"]):
            assert Shape.from_meta(seg["shape"], "t") == M["lay"].partitions[name].shapes[s]
        n = sum(1 for seg in p["segments"] if seg["q_max"] > 0)
        for seg in p["segments"]:
            assert seg["step"] == pytest.approx(np.sqrt(12 * p["rounding"]["budget"] / n) / seg["q_max"])
    assert man["partitions"][COS]["segments"][0]["shape"]["kind"] == "sampled"
    assert man["partitions"][COS]["segments"][0]["shape"]["cells"] == 200
    assert man["partitions"][BOX]["segments"][0]["shape"] == {"kind": "constant"}
    # the reach of a shaped class from the b matrix: the largest level within b_max is sqrt(b_max / b_unit) on one term
    cos = M["lay"].partitions[COS]
    N = cos.refocused_basis()[1]
    Bm = N.T @ cos.b_matrix() @ N
    want = np.sqrt(3e9 * np.diag(N @ np.linalg.solve(Bm, N.T)))
    np.testing.assert_allclose(np.minimum(want, G_MAX), man["partitions"][COS]["segments"][0]["q_max"] * np.ones(2), rtol=1e-9)
