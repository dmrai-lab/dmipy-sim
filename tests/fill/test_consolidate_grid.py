"""Shards a walker wrote (dmipy-sim#756): packs with no per-voxel certificate and their walkers' own ``voxel_ijk``
(n, 3), from several rounds that read the field at their own intervals. The grid is the recipe's: the consolidator
certifies them on it, and the layout it writes is the layout of shards packed with ``voxel_grid=``."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from dmipy_sim.fill.consolidate import consolidate
from dmipy_sim.fill.recipe import manifest_grid
from dmipy_sim.phantom import Grid
from dmipy_sim.replay.bank import build_replay_pack, merge_packs, voxel_certificate
from dmipy_sim.replay.replay import ReplayPack, read_rpk, write_rpk
from dmipy_sim.spec import StratifiedByVoxel, walk_spec
from tests.conftest import spec_without_source
from tests.test_field_read_interval import BLOCK, FINE


def as_written_by_a_walker(pack, path):
    """``pack`` as a walker writes a shard: no per-voxel certificate, each walker's own voxel."""
    ijk, _ = Grid.from_meta(pack.meta["fidelity"]["per_voxel"]["grid"]).bin(pack.r0)
    arrays = {k: v for k, v in pack.arrays.items() if k != "voxel_certificate"}
    arrays["voxel_ijk"] = ijk.astype(np.int32)
    meta = json.loads(json.dumps(pack.meta)); del meta["fidelity"]["per_voxel"]
    write_rpk(path, arrays, meta)
    return read_rpk(path)


def certified_table(out):
    pack = ReplayPack.open(out)
    return pack, pack.columns_over(["voxel_ijk", "voxel_certificate"], [(0, 0)])


# ------------------------------------------------------------------------------------- blocks of rounds, no field
@pytest.fixture(scope="module")
def rounds(spec_grid):
    """Block 0 walked in two rounds of different counts (4 and 12 walkers per voxel and pool) over the same voxels and
    block 1 in one: ``(grid, certified shards dir, walker-written shards dir, certified packs, written packs)``."""
    spec, grid, tmp = spec_grid
    spec = spec_without_source(spec)
    tmp = tmp / "walker"; os.makedirs(str(tmp / "certified"), exist_ok=True); os.makedirs(str(tmp / "written"), exist_ok=True)
    plan = [("block-0000.p1.a.r0", 0, 4, 100), ("block-0000.p1.a.r1", 0, 12, 200), ("block-0001.p1", 1, 12, 300)]
    cert, written = [], []
    for name, block, per, seed in plan:
        want = np.zeros(grid.shape, np.int64); want[block] = per
        w = walk_spec(spec, T_max=8e-4, block_h=2e-4, seed=seed, require_gpu=False, K=3,
                      seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}))
        pk = build_replay_pack(w, id=f"t/{name}", license="x", citation="x", K=3, voxel_grid=grid, out_path=str(tmp / "certified" / f"{name}.rpk"))
        cert.append(pk); written.append(as_written_by_a_walker(pk, str(tmp / "written" / f"{name}.rpk")))
    return grid, str(tmp / "certified"), str(tmp / "written"), cert, written, tmp


def test_a_walkers_shards_merge_with_the_union_weights_of_their_voxels(rounds):
    """The rounds of a block share (voxel, pool) cells: the merge renormalises their weights to the union by the walkers'
    own ``voxel_ijk``, as it does by the certificate's grid for packed shards, and keeps ``voxel_ijk`` walker-leading."""
    grid, _, _, cert, written, _ = rounds
    a = merge_packs(cert[:2], id="t/a", overlap="recertify")
    b = merge_packs(written[:2], id="t/b", overlap="recertify")
    np.testing.assert_allclose(np.asarray(b.spin_weights), np.asarray(a.spin_weights), rtol=1e-6)
    assert not np.allclose(np.asarray(b.spin_weights), np.concatenate([np.asarray(p.spin_weights) for p in written[:2]]))
    assert np.asarray(b.arrays["voxel_ijk"]).shape == (b.n_walkers, 3) and "per_voxel" not in b.meta["fidelity"]
    np.testing.assert_array_equal(np.asarray(b.arrays["voxel_ijk"]), np.concatenate([np.asarray(p.arrays["voxel_ijk"]) for p in written[:2]]))


def test_the_pack_certificate_is_the_one_a_build_with_the_grid_measures(rounds):
    grid, _, _, cert, written, _ = rounds
    arrays, pv = voxel_certificate(written[2], grid)
    np.testing.assert_array_equal(arrays["voxel_ijk"], cert[2].arrays["voxel_ijk"])
    np.testing.assert_allclose(arrays["voxel_certificate"], cert[2].arrays["voxel_certificate"], rtol=1e-6, equal_nan=True)
    for k in ("n_voxels", "walkers_min", "floor_max", "floor_median", "thin_voxels", "pools"):
        assert pv[k] == pytest.approx(cert[2].meta["fidelity"]["per_voxel"][k])


def test_a_layout_of_walkers_shards_is_the_layout_of_certified_shards(rounds, tmp_path):
    grid, cdir, wdir, cert, written, tmp = rounds
    ref, ref_idx = consolidate(cdir, str(tmp_path / "ref"), blocks=[0, 1], id="t/layout")
    man, idx = consolidate(wdir, str(tmp_path / "out"), blocks=[0, 1], id="t/layout", grid=grid)
    assert idx == ref_idx
    pv, pv_ref = man["meta"]["fidelity"]["per_voxel"], ref["meta"]["fidelity"]["per_voxel"]
    assert pv["n_voxels"] == pv_ref["n_voxels"] == 8 and pv["shards"] == pv_ref["shards"] == 2
    assert pv["within_2x_floor_fraction"] is None and pv["err_max"] is None            # the error is the pack's, not a voxel's
    for k in ("walkers_min", "floor_max", "floor_median", "thin_voxels", "pools", "grid"):
        assert pv[k] == pytest.approx(pv_ref[k])
    pack, table = certified_table(str(tmp_path / "out")); _, table_ref = certified_table(str(tmp_path / "ref"))
    np.testing.assert_array_equal(table["voxel_ijk"], table_ref["voxel_ijk"])
    np.testing.assert_allclose(table["voxel_certificate"], table_ref["voxel_certificate"], rtol=1e-5, equal_nan=True)
    assert pack.meta["fidelity"]["per_voxel"]["grid"] == grid.to_meta() and pack.n_rows == sum(p.n_walkers for p in cert)
    # the rows of the block of two rounds are the union's: one weight per (voxel, pool) cell, summing to the cell's fraction
    pack_ref = ReplayPack.open(str(tmp_path / "ref"))
    np.testing.assert_allclose(np.asarray(pack.view(contact=True).spin_weights), np.asarray(pack_ref.view(contact=True).spin_weights), rtol=1e-5)


def test_the_grid_is_the_recipes_and_nothing_else_stands_in_for_it(rounds, tmp_path):
    grid, cdir, wdir, *_ = rounds
    with pytest.raises(ValueError, match="no per-voxel certificate .* no voxel grid is given"):
        consolidate(wdir, str(tmp_path / "none"), blocks=[1], id="t/none")
    other = Grid(shape=(2, 2, 2), voxel_size_m=(10e-6,) * 3, origin_m=(7e-6,) * 3)
    with pytest.raises(ValueError, match="is not the recipe's"):
        consolidate(cdir, str(tmp_path / "other"), blocks=[1], id="t/other", grid=other)


def test_the_grid_of_a_manifest():
    g = manifest_grid(dict(grid=dict(shape=[2, 3, 4], voxel_size_m=[1e-5, 2e-5, 3e-5], origin_m=[0.0, 1e-6, 2e-6], attach="substrate")))
    assert g.shape == (2, 3, 4) and g.voxel_size_m == (1e-5, 2e-5, 3e-5) and g.origin_m == (0.0, 1e-6, 2e-6)


# ----------------------------------------------------------------- two blocks that read the strand field at their own intervals
@pytest.fixture(scope="module")
def field_blocks(tmp_path_factory):
    """Block 0 reads the field at the step, block 1 once per block; each one slab of the grid, walked with the strand
    field by the adaptive producer: ``(grid, directory of shards written by a walker, packs)``."""
    from dmipy_sim.io.strands import concat_centerlines, write_tck
    from dmipy_sim.spec import disco_spec
    tmp = tmp_path_factory.mktemp("fieldblocks"); os.makedirs(str(tmp / "w"))
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp / "t.tck"), str(tmp / "d.txt")
    write_tck(tck, *concat_centerlines(cls_), coordinate_unit_m=25e-6)
    np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    spec = disco_spec(tck, dia, side_m=20e-6)
    grid = Grid(shape=(2, 2, 2), voxel_size_m=(10e-6,) * 3, origin_m=(5e-6,) * 3)
    packs = []
    for block, interval in ((0, FINE), (1, BLOCK)):
        want = np.zeros(grid.shape, np.int64); want[block] = 6
        w = walk_spec(spec, None, 8e-4, BLOCK, seed=3 + block, n_probe=20_000, require_gpu=False, adaptive_steps=True, K=4,
                      field_cutoff_max_m=25e-6, seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}),
                      **({} if interval is None else dict(field_read_interval=interval)))
        pk = build_replay_pack(w, id=f"t/b{block}", license="x", citation="x", K=4, susc_path_K=4, device="numpy", voxel_grid=grid,
                               position_container=None, blt_container=None)
        packs.append(as_written_by_a_walker(pk, str(tmp / "w" / f"block-{block:04d}.p1.rpk")))
    return grid, str(tmp / "w"), packs


def test_a_layout_of_blocks_reads_the_field_at_the_coarsest(field_blocks, tmp_path):
    grid, wdir, (b0, b1) = field_blocks
    assert b0.meta["walk_params"]["field_read_interval"] < b1.meta["walk_params"]["field_read_interval"]
    man, idx = consolidate(wdir, str(tmp_path / "out"), blocks=[0, 1], id="t/field", grid=grid)
    meta = man["meta"]
    assert meta["walk_params"]["field_read_interval"] == b1.meta["walk_params"]["field_read_interval"] == pytest.approx(BLOCK)
    env, env1 = meta["replay_envelope"]["field_read"], b1.meta["replay_envelope"]["field_read"]
    assert {k: v for k, v in env.items() if k != "B0_chi_max_T"} == {k: v for k, v in env1.items() if k != "B0_chi_max_T"}
    # the term at the coarsest interval on the blocks' read series together (walker-weighted)
    from dmipy_sim.engine.accumulate import field_read_term
    n = np.asarray([b.meta["walk_params"]["n_walkers"] for b in (b0, b1)], np.float64)
    qv = (n[:, None] * np.asarray([b.meta["walk_params"]["qv"]["field_increments"] for b in (b0, b1)])).sum(0) / n.sum()
    consts = np.max([b.meta["compression"]["channels"]["susc_field_basis"]["read"]["constants"] for b in (b0, b1)], axis=0)
    at = env1["term_at"]
    assert meta["fidelity"]["terms"]["field_read"] == pytest.approx(field_read_term(BLOCK, qv, at["B0"], at["chi"], consts))
    assert meta["fidelity"]["terms"]["field_read"] > b0.meta["fidelity"]["terms"]["field_read"]
    assert [s["field_read_interval"] for s in meta["provenance"]["shards"]] == [b0.meta["walk_params"]["field_read_interval"], BLOCK]
    t = meta["fidelity"]["terms"]
    assert meta["fidelity"]["err_max"] >= t["truncation"] + sum(t["quadrature"].values()) + t["field_read"] + t["rounding"] - 1e-15
    pack = ReplayPack.open(str(tmp_path / "out"))
    pv = pack.meta["fidelity"]["per_voxel"]
    assert pv["n_voxels"] == 8 and pv["floor_median"] > 0 and pv["grid"] == grid.to_meta()
