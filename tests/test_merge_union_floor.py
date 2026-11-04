"""The floor of a merged pack is the floor of the union's own walkers (``fidelity.floor_combined = "exact_union"``): the
same estimator a built pack states (:func:`compression.measure_floor_coded`), read from five additive per-measurement
sums of the shards' walkers with the merged pack's final weights, in every mode."""
from __future__ import annotations

import numpy as np
import pytest

from dmipy_sim import Cylinder, walk
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.replay import compression as _cx
from dmipy_sim.replay.bank import build_replay_pack, merge_packs, pool_standard_errors
from dmipy_sim.spec import StratifiedByVoxel
from tests._envelope import TEST_BAND_HZ, TEST_G_MAX, battery_reach, shard_containers, walk_envelope
from tests.conftest import spec_without_source

N = 240
RTOL = 2e-5       # float32 coefficient storage and the certificate's 6 significant digits


def _pack(first, n, id, **kw):
    walk_ = _walk_geometry(n, 2e-9, Cylinder(4e-6, (0, 0, 1)), T_max=0.02, block_h=2e-4, seed=611, first_walker=first, require_gpu=False)
    return build_replay_pack(walk_, id=id, license="x", citation="x", K=16, position_container=None, blt_container=None, field_container=None,
                             envelope=walk_envelope(G_max=battery_reach(walk_.T_window), band_hz=TEST_BAND_HZ, of=walk_), **kw)


@pytest.fixture(scope="module")
def whole():
    return _pack(0, N, "t/whole")


def _stored_floor(pack):
    """The floor a build reads, :func:`compression.measure_floor_coded`, over the walkers a pack stores with its weights."""
    from dmipy_replay.replay import compression as _rc
    from dmipy_sim.replay.reencode import stated_envelope
    C = _rc.read_position_coeffs(pack.arrays, dtype=np.float64)
    return _cx.measure_floor_coded(C, float(pack.dt), int(pack.n_t), stated_envelope(pack), w=np.asarray(pack.spin_weights, np.float64))


def test_the_floor_is_a_function_of_five_additive_sums():
    """The sums of disjoint walker sets add, and the floor read from their total is the floor of the union."""
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(50, 6)) * 0.7; w = rng.uniform(0.2, 2.0, 50); lw = rng.normal(size=50) * 0.1
    meta = [dict(fam="A")] * 3 + [dict(fam="B")] * 3
    whole_ = _cx.floor_from_sums(_cx.floor_sums(phi, w, lw), meta)
    parts = _cx.add_floor_sums(_cx.floor_sums(phi[:13], w[:13], lw[:13]), _cx.floor_sums(phi[13:], w[13:], lw[13:]))
    f = _cx.floor_from_sums(parts, meta)
    s = np.exp(lw[:, None] + 1j * phi); S = (w[:, None] * s).sum(0) / w.sum()
    direct = np.sqrt((w[:, None] ** 2 * np.abs(s - S) ** 2).sum(0)) / w.sum()           # the definition, written out
    assert whole_["floor_max"] == pytest.approx(direct.max(), rel=1e-12) and f["floor_max"] == pytest.approx(direct.max(), rel=1e-12)
    assert f["per_family"]["A"] == pytest.approx(direct[:3].max(), rel=1e-12) and f["per_family"]["B"] == pytest.approx(direct[3:].max(), rel=1e-12)
    assert f["n_eff"] == pytest.approx(w.sum() ** 2 / (w ** 2).sum(), rel=1e-12) and f["noise_floor"] == pytest.approx(1 / np.sqrt(50))


def test_pool_standard_errors_is_the_planners_prediction():
    assert pool_standard_errors([0.01, 0.01], [100, 100]) == pytest.approx(0.01 / np.sqrt(2))
    assert pool_standard_errors([0.02, 0.01], [1, 3]) == pytest.approx(np.sqrt((0.25 * 0.02) ** 2 + (0.75 * 0.01) ** 2))
    with pytest.raises(ValueError, match="positive weight"):
        pool_standard_errors([0.01], [0])


@pytest.mark.parametrize("parts", [2, 4])
def test_walker_range_shards_merge_to_the_floor_of_the_whole_walk(whole, parts):
    """The same walkers walked whole and as ``parts`` ranges of one seed: the merged ``floor_max``, ``per_family`` and
    ``noise_floor`` are the whole pack's to rounding -- the between-shard term of the union is in them."""
    n = N // parts
    shards = [_pack(i * n, n, f"t/s{i}") for i in range(parts)]
    m = merge_packs(shards, id="t/merged")
    fw, fm = whole.meta["fidelity"], m.meta["fidelity"]
    print(f"parts={parts} whole={fw['floor_max']:.10g} merged={fm['floor_max']:.10g} rel={abs(fm['floor_max'] / fw['floor_max'] - 1):.2e}")
    assert fm["floor_max"] == pytest.approx(fw["floor_max"], rel=RTOL)
    assert fm["floor_combined"] == "exact_union"
    assert set(fm["per_family"]) == set(fw["per_family"])
    for fam in fw["per_family"]:
        assert fm["per_family"][fam]["floor_max"] == pytest.approx(fw["per_family"][fam]["floor_max"], rel=RTOL), fam
    assert fm["noise_floor"] == pytest.approx(fw["noise_floor"], rel=RTOL)
    assert fm["floor_quantisation"] == pytest.approx(fw["floor_quantisation"], rel=RTOL, abs=1e-12)
    assert fm["within_2x_floor"] == bool(fm["err_max"] <= 2 * fm["floor_max"])
    assert m.meta["compression"]["precision_tiers"]["floor_at_full_n"] == pytest.approx(fm["floor_max"])    # the tiers read the stated floor


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
    assert f["floor_max"] == pytest.approx(f0["floor_max"], rel=RTOL) and f["err_max"] == f0["err_max"]
    assert f["floor_combined"] == "exact_union"
    with pytest.raises(ValueError, match="at least one shard"):
        merge_packs([], id="t/none")


# ---------------------------------------------------------------- voxel blocks and recertified passes (per-voxel certificate)
def _block_pack(spec, grid, tmp, id, seed, want, containers):
    w = walk(spec, T_max=8e-4, block_h=2e-4, seed=seed, require_gpu=False, K=3,
             seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}), envelope=walk_envelope(of=spec, G_max=TEST_G_MAX))
    pk = build_replay_pack(w, id=id, license="x", citation="x", K=3, voxel_grid=grid, out_path=str(tmp / f"{id.split('/')[-1]}.rpk"),
                           envelope=w.envelope.replace(band_hz=TEST_BAND_HZ), **containers)
    return w, pk


def _whole_floor(walks, packs, weights):
    """What a build of these walkers as ONE pack reads: :func:`compression.measure_floor_coded` over their float
    coefficients (every shard's, in order) with ``weights``, on the shards' battery."""
    from dmipy_sim.replay.reencode import stated_envelope
    K = packs[0].meta["compression"]["K"]
    C = np.concatenate([np.asarray(w.position_coeffs, np.float64)[:, 0, :K + 2, :] for w in walks])
    return _cx.measure_floor_coded(C, float(packs[0].dt), int(packs[0].n_t), stated_envelope(packs[0]), w=np.asarray(weights, np.float64))


def test_voxel_blocks_merge_to_the_floor_of_the_walkers_built_whole(spec_grid):
    """Disjoint voxel blocks under a per-voxel certificate: the merged pack-level ``floor_max`` and ``per_family`` are the
    floor of all the walkers read as one population, not the largest block's."""
    spec, grid, tmp = spec_grid
    spec = spec_without_source(spec)
    walks, packs = [], []
    for b in (0, 1):
        want = np.zeros(grid.shape, np.int64); want[b] = 12
        w, pk = _block_pack(spec, grid, tmp, f"t/blk-{b}", 7 + b, want, shard_containers(packs))
        walks.append(w); packs.append(pk)
    m = merge_packs(packs, id="t/blocks")
    ref = _whole_floor(walks, packs, np.concatenate([np.asarray(p.spin_weights, np.float64) for p in packs]))
    fm = m.meta["fidelity"]
    shard_max = max(p.meta["fidelity"]["floor_max"] for p in packs)
    print(f"blocks whole={ref['floor_max']:.10g} merged={fm['floor_max']:.10g} rel={abs(fm['floor_max'] / ref['floor_max'] - 1):.2e} largest shard={shard_max:.6g}")
    assert fm["floor_combined"] == "exact_union"
    assert fm["floor_max"] == pytest.approx(ref["floor_max"], rel=RTOL)
    for fam, v in ref["per_family"].items():
        assert fm["per_family"][fam]["floor_max"] == pytest.approx(v, rel=RTOL), fam
    assert fm["floor_max"] < shard_max                                                  # the union's walkers are more than a block's
    assert fm["within_2x_floor"] == bool(fm["err_max"] <= 2 * fm["floor_max"])


def test_a_recertified_16_84_pass_pair_merges_to_the_floor_of_the_whole_pack_at_the_union_weights(spec_grid):
    """A 16 % first pass and its 84 % top-up on one block, merged with census weights (:func:`union_weights`): the pack's
    ``floor_max`` is that of the walkers read as one pack WITH THE UNION'S WEIGHTS (the merged pack's own), not the
    as-is concatenation's."""
    spec, grid, tmp = spec_grid
    spec = spec_without_source(spec)
    walks, packs = [], []
    for p_, per in ((1, 4), (2, 21)):
        want = np.zeros(grid.shape, np.int64); want[0] = per
        w, pk = _block_pack(spec, grid, tmp, f"t/pass-{p_}", 100 * p_, want, shard_containers(packs))
        walks.append(w); packs.append(pk)
    m = merge_packs(packs, id="t/union", overlap="recertify")
    wm = np.asarray(m.spin_weights, np.float64)
    ref = _whole_floor(walks, packs, wm)
    asis = _whole_floor(walks, packs, np.concatenate([np.asarray(p.spin_weights, np.float64) for p in packs]))
    fm = m.meta["fidelity"]
    print(f"recertify whole(union weights)={ref['floor_max']:.10g} merged={fm['floor_max']:.10g} "
          f"rel={abs(fm['floor_max'] / ref['floor_max'] - 1):.2e} as-is={asis['floor_max']:.6g}")
    assert fm["floor_combined"] == "exact_union"
    assert fm["floor_max"] == pytest.approx(ref["floor_max"], rel=RTOL)
    for fam, v in ref["per_family"].items():
        assert fm["per_family"][fam]["floor_max"] == pytest.approx(v, rel=RTOL), fam
    assert abs(asis["floor_max"] / ref["floor_max"] - 1) > 1e-3                        # the as-is weights are a different (worse) population
    assert fm["noise_floor"] == pytest.approx(ref["noise_floor"], rel=RTOL)


def test_a_shard_that_states_no_battery_is_refused_by_name():
    """The floor of a union is read on the battery the shards state; a shard built without that statement is refused,
    naming it, unless the caller states the settings."""
    import copy
    from dmipy_replay import ReplayPack
    a, b = _pack(0, N // 2, "t/a"), _pack(N // 2, N // 2, "t/old")
    meta = copy.deepcopy(b.meta); del meta["fidelity"]["battery"]
    with pytest.raises(ValueError, match=r"shard 1 \('t/old'\).*fidelity.battery.*Rebuild"):
        merge_packs([a, ReplayPack(dict(b.arrays), meta)], id="t/refused")
