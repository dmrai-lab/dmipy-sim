"""The columnar layout (dmipy-sim#290): shards consolidated into columns replay as the merged pack does, a view
reads the rows and bands asked for and nothing else, the image loop reproduces the per-voxel replay and shares one
pass over the rows between settings, and a later pass of a block appends with the union's weights."""
from __future__ import annotations
import os

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.io.strands import write_tck
from dmipy_sim.phantom import Grid
from dmipy_sim.replay.bank import build_replay_pack, merge_packs
from dmipy_sim.replay.replay import ReplayPack
from dmipy_sim.fill.consolidate import consolidate, append, band_groups_of
from dmipy_sim.spec import disco_spec, walk_spec, StratifiedByVoxel
from dmipy_sim.spec.tissue import Tissue


@pytest.fixture(scope="module")
def spec_grid(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("columnar")
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp / "t.tck"), str(tmp / "d.txt")
    write_tck(tck, cls_, coordinate_unit_m=25e-6); np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    spec = disco_spec(tck, dia, side_m=20e-6)
    grid = Grid(shape=(2, 2, 2), voxel_size_m=(10e-6,) * 3, origin_m=(5e-6,) * 3)
    return spec, grid, tmp


def _shard(spec, grid, tmp, name, block, per, seed):
    want = np.zeros(grid.shape, np.int64); want[block] = per
    w = walk_spec(spec, T_max=8e-4, dt_save=2e-4, seed=seed, require_gpu=False, field=False,
                  seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}))
    os.makedirs(str(tmp / "shards"), exist_ok=True)
    return build_replay_pack(w, id=f"t/{name}", license="x", citation="x", K=3, voxel_grid=grid, out_path=str(tmp / "shards" / f"{name}.rpk"))


@pytest.fixture(scope="module")
def layout(spec_grid):
    spec, grid, tmp = spec_grid
    a = _shard(spec, grid, tmp, "block-0000.p1", 0, 12, 7); b = _shard(spec, grid, tmp, "block-0001.p1", 1, 12, 8)
    out = str(tmp / "layout")
    manifest, index = consolidate(str(tmp / "shards"), out, blocks=[0, 1], id="t/columns")
    merged = merge_packs([a, b], id="t/merged")
    return out, manifest, index, (a, b), merged, grid, spec, tmp


def _per_voxel(pack, seq, grid, **kw):
    w, ew, E = pack.walker_signals(seq, **kw)
    ijk, _ = grid.bin(pack.r0); v = np.ravel_multi_index(ijk.T, grid.shape); keys, inv = np.unique(v, return_inverse=True)
    num = np.zeros((len(keys), E.shape[1]), complex); den = np.zeros(len(keys))
    np.add.at(num, inv, ew[:, None] * E); np.add.at(den, inv, w)
    return keys, np.abs(num / den[:, None])


def _seq(n_t):
    return d.set_b(d.pgse([[1, 0, 0], [0, 0, 1]], 0.2e-3, 0.5e-3, gradient_strengths=0.1, n_t=n_t, slew_rate=np.inf), [1e9, 1e9])


def test_the_layout_holds_the_merged_pack_bit_for_bit(layout):
    out, manifest, index, (a, b), merged, grid, spec, tmp = layout
    pack = ReplayPack.open(out)
    assert pack.n_rows == merged.n_walkers == index["n_rows"] and pack.K == 3 and pack.band_groups == list(band_groups_of(3, merged.meta["compression"].get("container") or []))
    assert manifest["meta"]["fidelity"]["per_voxel"]["n_voxels"] == merged.meta["fidelity"]["per_voxel"]["n_voxels"] == 8
    full = pack.view(contact=True)                                         # every row, every band, every tier: the merged pack
    seq = _seq(merged.n_t)
    np.testing.assert_allclose(full.replay(seq), merged.replay(seq), rtol=1e-9)
    np.testing.assert_allclose(full.replay(seq, tissue=Tissue(rho=1e-5, D=1.7e-9)), merged.replay(seq, tissue=Tissue(rho=1e-5, D=1.7e-9)), rtol=1e-9)
    keys_m, S_m = _per_voxel(merged, seq, grid); keys_c, S_c = _per_voxel(full, seq, grid)
    np.testing.assert_array_equal(keys_m, keys_c); np.testing.assert_allclose(S_c, S_m, rtol=1e-9)


def test_a_view_reads_its_voxels_rows_and_nothing_else(layout):
    out, manifest, index, (a, b), merged, grid, spec, tmp = layout
    pack = ReplayPack.open(out); seq = _seq(merged.n_t)
    vox = [tuple(r["ijk"]) for r in index["rows"]][:1]
    rows = sum(e - s for s, e in pack.rows_of(vox))
    pack.src.bytes_read = pack.src.requests = 0
    v = pack.view(voxels=vox)
    assert v.n_walkers == rows and rows < merged.n_walkers
    per_row = pack.plan(seq, voxels=vox)["bytes_per_row"]
    assert pack.src.bytes_read < 2 * (per_row * rows + sum(c["parts"][0]["nbytes"] for c in manifest["columns"].values() if not c["per_row"]))
    keys_m, S_m = _per_voxel(merged, seq, grid); keys_v, S_v = _per_voxel(v, seq, grid)
    i = int(np.flatnonzero(keys_m == keys_v[0])[0])
    np.testing.assert_allclose(S_v[0], S_m[i], rtol=1e-9)


def test_the_image_is_the_per_voxel_replay_and_settings_share_the_pass(layout):
    out, manifest, index, (a, b), merged, grid, spec, tmp = layout
    pack = ReplayPack.open(out); seq = _seq(merged.n_t)
    t = Tissue(rho=1e-5, D=1.7e-9)
    S, floor, plan = pack.image([seq], settings=[(None, None), (t, None)], tol=1e-9, chunk_rows=5)
    assert S.shape == (2,) + tuple(grid.shape) + (2,) and plan["K"] == 3 and plan["rows"] == merged.n_walkers
    for si, kw in enumerate(({}, {"tissue": t})):
        keys, S_m = _per_voxel(merged, seq, grid, **kw)
        got = S[si].reshape(-1, 2)[keys]
        np.testing.assert_allclose(got, S_m, rtol=1e-9, atol=1e-12)
    S1, _, _ = pack.image(seq, tissue=t, tol=1e-9)
    np.testing.assert_allclose(S1, S[1], rtol=0, atol=1e-13)
    from dmipy_sim.replay.bank import voxel_fidelity_volumes
    g, floors, counts = voxel_fidelity_volumes(merged)
    worst = np.nanmax(np.stack([floors[n] for n in floors]), axis=0)
    np.testing.assert_allclose(np.nan_to_num(floor, nan=-1), np.nan_to_num(np.where(sum(counts.values()) > 0, worst, np.nan), nan=-1), rtol=1e-6)


def test_a_later_pass_appends_with_the_union_weights(spec_grid):
    """A first pass of 4 walkers per voxel and pool and a top-up of 12 on the same block: consolidated then appended,
    the layout replays as the recertifying merge of the two (the union's weights: every walker of a (voxel, pool)
    alike, the mean of the passes' fractions over the union's count), its index holds both row ranges, and the
    certificate counts the union."""
    spec, grid, tmp = spec_grid
    tmp = tmp / "passes"; os.makedirs(str(tmp), exist_ok=True)
    p1 = _shard(spec, grid, tmp, "block-0000.p1", 0, 4, 100); p2 = _shard(spec, grid, tmp, "block-0000.p2", 0, 12, 200)
    out = str(tmp / "layout")
    consolidate(str(tmp / "shards"), out, blocks=[0], id="t/p1", pass_=1)
    manifest, index = append(str(tmp / "shards"), out, id="t/p1+p2", pass_=2)
    m = merge_packs([p1, p2], id="t/union", overlap="recertify")
    pack = ReplayPack.open(out); seq = _seq(m.n_t)
    assert pack.n_rows == m.n_walkers == p1.n_walkers + p2.n_walkers
    from collections import Counter
    two = [k for k, c in Counter((tuple(r["ijk"]), r["pool"]) for r in index["rows"]).items() if c == 2]
    assert len(two) == 4 * 2                                                # every (voxel, pool) of the block's four voxels has both ranges
    full = pack.view(contact=True)
    np.testing.assert_allclose(np.sort(np.asarray(full.spin_weights)), np.sort(np.asarray(m.spin_weights)), rtol=1e-6)
    np.testing.assert_allclose(full.replay(seq), m.replay(seq), rtol=1e-6)
    S, floor, plan = pack.image(seq, tol=1e-9, chunk_rows=7)
    keys, S_m = _per_voxel(m, seq, grid)
    np.testing.assert_allclose(S.reshape(-1, 2)[keys], S_m, rtol=1e-6)
    cert = pack.columns_over(["voxel_certificate"], [(0, 0)])["voxel_certificate"]
    assert (cert[:, :, 0] == 16).all() and manifest["meta"]["columnar"]["repair"]["voxels"] == 4
