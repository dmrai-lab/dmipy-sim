"""The shape-moment layout (dmipy-sim#505): a columnar pack contracted once against a waveform's shape replays
every amplitude and direction of that shape as the columnar image does, with the same split-half floor; a shape
is one direction with amplitude-invariant ramps, and a b-value maps to an amplitude through the recorded unit b."""
from __future__ import annotations
import json
import os

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.columnar import ColumnarPack
from dmipy_sim.replay.shape_moments import ShapeMoments, write_shape_moments, _profile
from dmipy_sim.replay.study import Acquisition, Protocol, Study
from dmipy_sim.fill.consolidate import consolidate
from dmipy_sim.spec import walk_spec, StratifiedByVoxel


@pytest.fixture(scope="module")
def layout(spec_grid):
    """Two blocks of the three-strand spec consolidated into columns, and their n_t."""
    spec, grid, tmp = spec_grid
    os.makedirs(str(tmp / "sm_shards"), exist_ok=True)
    n_t = None
    for b in (0, 1):
        want = np.zeros(grid.shape, np.int64); want[b] = 12
        w = walk_spec(spec, T_max=8e-4, dt_save=2e-4, seed=11 + b, require_gpu=False, field=False,
                      seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}))
        pk = build_replay_pack(w, id=f"t/sm{b}", license="x", citation="x", K=3, voxel_grid=grid,
                               out_path=str(tmp / "sm_shards" / f"block-000{b}.p1.rpk"))
        n_t = pk.n_t
    consolidate(str(tmp / "sm_shards"), str(tmp / "sm_layout"), blocks=[0, 1], id="t/sm-columns")
    return ColumnarPack(str(tmp / "sm_layout")), grid, n_t, tmp


def _shape(n_t, delta=0.2e-3, Delta=0.5e-3, g=0.05, direction=(0.0, 0.0, 1.0)):
    return d.pgse([list(direction)], delta, Delta, gradient_strengths=g, n_t=n_t, slew_rate=np.inf)


def test_the_moments_replay_every_amplitude_and_direction_of_the_shape(layout):
    """Six directions at three amplitudes of one shape: the layout's image equals the columnar image of the
    same acquisitions (built as one sequence at those b-values) to float32 arithmetic, its floor the study's
    split-half floor, and NaN sits exactly where the columns have no rows."""
    col, grid, n_t, tmp = layout
    out = str(tmp / "sm_moments")
    manifest = write_shape_moments(col, {"a": _shape(n_t), "c": _shape(n_t, delta=0.2e-3, Delta=0.4e-3, g=0.02)}, out,
                                   tol=1e-9, chunk_rows=7)                  # two shapes on one grid: one contraction
    assert manifest["n_rows"] == col.n_rows and manifest["K"] == col.K and manifest["source"]["n_rows"] == col.n_rows
    sm = ShapeMoments(out)
    assert sm.shapes == ["a", "c"] and sm.moments("a").shape == (col.n_rows, 3) and sm.moments("a").dtype == np.float32
    rng = np.random.default_rng(3)
    u = rng.normal(size=(6, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    dirs = np.concatenate([u, u, u, [[0, 0, 1]]]); b = np.r_[np.full(6, 2e8), np.full(6, 1e9), np.full(6, 3e9), 0.0]
    S, floor = sm.image("a", b, dirs)
    seq = d.pgse(dirs.tolist(), 0.2e-3, 0.5e-3, bvalues=b, n_t=n_t, slew_rate=np.inf)
    S_col, _, plan = col.image(seq, tol=1e-9, chunk_rows=5)
    assert S.shape == S_col.shape == tuple(grid.shape) + (19,)
    np.testing.assert_array_equal(np.isnan(S), np.isnan(S_col))
    err = np.nanmax(np.abs(S - S_col))
    assert err < 2e-6, err                    # float32 phases and complex64 sums against the float64 image: 2.4e-7 measured
    np.testing.assert_allclose(np.nan_to_num(S[..., -1]), np.nan_to_num(S_col[..., -1]) , atol=1e-6)   # b = 0 is 1
    _, floor_study, _ = col.image(Study(Protocol([Acquisition(seq)])), tol=1e-9, chunk_rows=5)
    np.testing.assert_array_equal(np.isnan(floor), np.isnan(floor_study[0]))
    assert np.nanmax(np.abs(floor - floor_study[0])) < 2e-6
    S_c, _ = sm.image("c", b[:6], dirs[:6])                                   # the second shape of the shared pass
    seq_c = d.pgse(dirs[:6].tolist(), 0.2e-3, 0.4e-3, bvalues=b[:6], n_t=n_t, slew_rate=np.inf)
    S_col_c, _, _ = col.image(seq_c, tol=1e-9, chunk_rows=5)
    assert np.nanmax(np.abs(S_c - S_col_c)) < 2e-6 and np.nanmax(np.abs(S_c - S[..., :6])) > 1e-3


def test_the_amplitude_of_a_b_value_is_the_shapes_own(layout):
    """``g = sqrt(b / b_unit)``: the recorded unit b makes the builder's own b at any amplitude."""
    col, grid, n_t, tmp = layout
    sm = ShapeMoments(str(tmp / "sm_moments"))
    seq = _shape(n_t, g=0.123)
    g = sm.amplitude("a", d.calc_b(seq))
    np.testing.assert_allclose(g, 0.123, rtol=1e-6)
    np.testing.assert_array_equal(sm.amplitude("a", [0.0]), [0.0])
    with pytest.raises(ValueError, match="negative"):
        sm.amplitude("a", [-1.0])
    m = json.load(open(os.path.join(str(tmp / "sm_moments"), "manifest.json")))
    assert m["shapes"]["a"]["build_spec"][0] == "pgse" and m["shapes"]["a"]["encoding"]["delta"] == [0.2e-3]
    assert m["code"]["package"] == "dmipy-sim" and m["source"]["manifest_sha256"]


def test_a_shape_is_one_direction_with_amplitude_invariant_ramps(layout):
    col, grid, n_t, tmp = layout
    s, u, g = _profile(_shape(n_t, g=0.07, direction=(0.6, 0.0, 0.8)))
    assert abs(s).max() == pytest.approx(1.0) and g == pytest.approx(0.07) and np.allclose(u, [0.6, 0, 0.8])
    with pytest.raises(ValueError, match="one-row"):
        _profile(d.pgse([[1, 0, 0], [0, 1, 0]], 0.2e-3, 0.5e-3, gradient_strengths=0.05, n_t=n_t, slew_rate=np.inf))
    with pytest.raises(ValueError, match="slew_rate=np.inf"):
        _profile(d.pgse([[0, 0, 1]], 1e-3, 2e-3, gradient_strengths=0.05, n_t=n_t, slew_rate=200.0))
    with pytest.raises(ValueError, match="one direction"):
        _profile(d.pte([0, 0, 1], 0.4e-3, bvalues=[1e9], n_t=n_t, slew_rate=np.inf))
    with pytest.raises(ValueError, match="at least one"):
        write_shape_moments(col, {}, str(tmp / "sm_none"))


def test_directions_must_match_and_be_unit(layout):
    col, grid, n_t, tmp = layout
    sm = ShapeMoments(str(tmp / "sm_moments"))
    with pytest.raises(ValueError, match="match"):
        sm.image("a", [1e9, 1e9], [[0, 0, 1]])
    with pytest.raises(ValueError, match="unit"):
        sm.image("a", [1e9], [[0, 0, 2]])
    with pytest.raises(KeyError):
        sm.moments("b")


def test_the_bands_are_judged_at_the_shapes_own_amplitude_and_the_device_rows_can_be_released(layout):
    """A shape built at a tenth of the amplitude needs no more bands than one built at the full amplitude (the
    truncation error scales with the amplitude squared, and the manifest records the amplitude judged at); after
    ``release`` the device holds the shared rows only."""
    col, grid, n_t, tmp = layout
    out = str(tmp / "sm_amp")
    lo = write_shape_moments(col, {"lo": _shape(n_t, g=0.005)}, out, tol=1e-9, chunk_rows=7)
    assert lo["shapes"]["lo"]["amplitude_built"] == pytest.approx(0.005) and lo["K"] <= col.K
    sm = ShapeMoments(str(tmp / "sm_moments"))
    sm.image("a", [1e9], [[0, 0, 1]]); sm.image("c", [1e9], [[0, 0, 1]])
    assert set(sm._device) == {"w", "seg", "a", "c"}
    sm.release(keep=["c"])
    assert set(sm._device) == {"w", "seg", "c"}
