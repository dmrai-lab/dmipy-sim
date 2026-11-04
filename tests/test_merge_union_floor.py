"""The floor of a merged pack: the shards that partition one walk's walkers combine as the pooled standard error of
the union (it falls as 1 / sqrt(N)), disjoint voxel blocks keep their largest, and the combination is stated."""
from __future__ import annotations

import numpy as np
import pytest

from dmipy_sim import Cylinder
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.replay.bank import build_replay_pack, merge_packs, pool_standard_errors
from tests._envelope import TEST_BAND_HZ, battery_reach, stated, walk_envelope

N = 240


def _pack(first, n, id, **kw):
    walk = _walk_geometry(n, 2e-9, Cylinder(4e-6, (0, 0, 1)), T_max=0.02, block_h=2e-4, seed=611, first_walker=first, require_gpu=False)
    return build_replay_pack(walk, id=id, license="x", citation="x", K=16, position_container=None, blt_container=None, field_container=None,
                             envelope=walk_envelope(G_max=battery_reach(walk.T_window), band_hz=TEST_BAND_HZ, of=walk), **kw)


@pytest.fixture(scope="module")
def whole():
    return _pack(0, N, "t/whole")


def test_pool_standard_errors_is_the_standard_error_of_the_pooled_mean():
    assert pool_standard_errors([0.01, 0.01], [100, 100]) == pytest.approx(0.01 / np.sqrt(2))
    assert pool_standard_errors([0.02, 0.01], [1, 3]) == pytest.approx(np.sqrt((0.25 * 0.02) ** 2 + (0.75 * 0.01) ** 2))
    with pytest.raises(ValueError, match="positive weight"):
        pool_standard_errors([0.01], [0])


@pytest.mark.parametrize("parts", [2, 4])
def test_walker_range_shards_merge_to_the_floor_of_the_whole_walk(whole, parts):
    """The same walkers walked whole and as ``parts`` ranges of one seed: the merged floor is the whole pack's within a few
    percent (the pooled mean of independent shards omits the between-shard-mean term of the union's own
    split, second order in the shard size), where the largest-over-shards floor sits ``sqrt(parts)`` above."""
    n = N // parts
    shards = [_pack(i * n, n, f"t/s{i}") for i in range(parts)]
    m = merge_packs(shards, id="t/merged")
    f_whole, f_merged = whole.meta["fidelity"]["floor_max"], m.meta["fidelity"]["floor_max"]
    f_shards = [s.meta["fidelity"]["floor_max"] for s in shards]
    print(f"parts={parts} whole={f_whole:.6g} merged={f_merged:.6g} shards={max(f_shards):.6g}")
    assert f_merged == pytest.approx(f_whole, rel=0.01) and f_merged >= f_whole * (1 - 1e-3)
    assert max(f_shards) > 1.3 * f_merged                                    # the old combination
    fid = m.meta["fidelity"]
    assert fid["floor_combined"] == "pooled_shards"
    assert fid["noise_floor"] == pytest.approx(1 / np.sqrt(N), rel=1e-5)
    assert fid["within_2x_floor"] == bool(fid["err_max"] <= 2 * fid["floor_max"])
    assert m.meta["compression"]["precision_tiers"]["floor_at_full_n"] == pytest.approx(f_merged)


def test_the_target_statements_are_restated_from_the_merged_floor(whole):
    """Shards planned for a floor they miss alone: the union meets it, and the statement says so."""
    n = N // 2
    planned = 1.05 * whole.meta["fidelity"]["floor_max"]
    shards = [_pack(i * n, n, f"t/p{i}", sigma_star=0.5, planned_floor=planned) for i in range(2)]
    assert not any(s.meta["fidelity"]["meets_planned_floor"] for s in shards)
    fid = merge_packs(shards, id="t/merged").meta["fidelity"]
    assert fid["meets_planned_floor"] and fid["meets_target"] == fid["meets_numerical_cap"]


def test_one_walker_counted_twice_is_refused(whole):
    n = N // 2
    with pytest.raises(ValueError, match="counts each walker once"):
        merge_packs([_pack(0, n, "t/a"), _pack(n - 10, n, "t/b")], id="t/dup")




def test_a_pack_of_one_shard_assembles_as_that_shard(whole):
    """A pack walked as a single shard is assembled by the same merge: the union of one shard is that shard, its
    walkers' tensors unchanged, its floor and error as the shard states them, under the pack's own id."""
    m = merge_packs([whole], id="t/one")
    assert m.meta["id"] == "t/one"
    assert sorted(m.arrays) == sorted(whole.arrays)
    for k in whole.arrays:
        a, b = m.arrays[k], whole.arrays[k]
        a = np.concatenate(a) if isinstance(a, list) else np.asarray(a)
        b = np.concatenate(b) if isinstance(b, list) else np.asarray(b)
        assert np.array_equal(a, b), k
    f, f0 = m.meta["fidelity"], whole.meta["fidelity"]
    assert f["floor_max"] == pytest.approx(f0["floor_max"], rel=1e-12) and f["err_max"] == f0["err_max"]
    assert f["floor_combined"] == "pooled_shards"
    with pytest.raises(ValueError, match="at least one shard"):
        merge_packs([], id="t/none")
