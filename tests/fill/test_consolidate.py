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
from tests.conftest import spec_without_source


def _shard(spec, grid, tmp, name, block, per, seed):
    want = np.zeros(grid.shape, np.int64); want[block] = per
    w = walk_spec(spec, T_max=8e-4, block_h=2e-4, seed=seed, require_gpu=False, K=3,
                  seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}))
    os.makedirs(str(tmp / "shards"), exist_ok=True)
    return build_replay_pack(w, id=f"t/{name}", license="x", citation="x", K=3, voxel_grid=grid, out_path=str(tmp / "shards" / f"{name}.rpk"))


@pytest.fixture(scope="module")
def layout(spec_grid):
    spec, grid, tmp = spec_grid
    spec = spec_without_source(spec)      # consolidation mechanics, not the field tier
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
    np.testing.assert_allclose(full.replay(seq, tissue=Tissue(rho2=1e-5)), merged.replay(seq, tissue=Tissue(rho2=1e-5)), rtol=1e-9)   # the surface term at the walked D
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
    t = Tissue(rho2=1e-5)
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
    spec = spec_without_source(spec)      # consolidation mechanics, not the field tier
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


# ------------------------------------------------------------ packs in a derived curve (positions and contact bit-packed)

def _curve_shard(spec, grid, tmp, name, block, per, seed, K):
    from dmipy_sim.replay.compression import band_curve
    want = np.zeros(grid.shape, np.int64); want[block] = per
    w = walk_spec(spec, T_max=6e-3, block_h=2e-4, seed=seed, require_gpu=False, K=K,
                  seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}))
    os.makedirs(str(tmp / "cshards"), exist_ok=True)
    return build_replay_pack(w, id=f"t/{name}", license="x", citation="x", K=K, voxel_grid=grid, blt_temporal_K=K,
                             position_container=band_curve(K, 7), blt_container=band_curve(K, 7),
                             out_path=str(tmp / "cshards" / f"{name}.rpk"))


@pytest.fixture(scope="module")
def curve_layout(spec_grid):
    spec, grid, tmp = spec_grid
    spec = spec_without_source(spec)
    K = 24
    a = _curve_shard(spec, grid, tmp, "block-0000.p1", 0, 12, 7, K); b = _curve_shard(spec, grid, tmp, "block-0001.p1", 1, 12, 8, K)
    out = str(tmp / "clayout")
    consolidate(str(tmp / "cshards"), out, blocks=[0, 1], id="t/ccolumns")
    return out, (a, b), merge_packs([a, b], id="t/cmerged"), K


def test_a_curve_pack_keeps_its_ranges_as_column_groups_and_replays_as_the_merged_pack(curve_layout):
    out, (a, b), merged, K = curve_layout
    cont = merged.meta["compression"]["container"]
    assert any(r["bits"] not in (8, 16) for r in cont)                                  # a packed range is in play
    pack = ReplayPack.open(out)
    assert pack.band_groups == [r["bands"][1] for r in cont] == list(band_groups_of(K, cont))
    names = set(pack.columns)
    assert {f"pos_x_b{i}" + ("" if r["bits"] in (8, 16) else f"u{r['bits']}") for i, r in enumerate(cont)} <= names
    full = pack.view(contact=True)
    seq = d.set_b(d.pgse([[1, 0, 0], [0, 0, 1]], 0.2e-3, 0.5e-3, gradient_strengths=0.1, n_t=merged.n_t, slew_rate=np.inf), [1e9, 1e9])
    np.testing.assert_allclose(full.replay(seq), merged.replay(seq), rtol=1e-9)
    np.testing.assert_allclose(full.replay(seq, tissue=Tissue(rho2=1e-5)), merged.replay(seq, tissue=Tissue(rho2=1e-5)), rtol=1e-9)


def test_a_band_prefix_view_of_a_curve_pack_is_the_prefix_of_its_ranges(curve_layout):
    out, (a, b), merged, K = curve_layout
    pack = ReplayPack.open(out)
    k1 = pack.band_groups[1]
    v = pack.view(K=k1)
    assert v.K == k1 and len(v.meta["compression"]["container"]) == 2
    from dmipy_sim.replay.compression import read_position_coeffs
    np.testing.assert_array_equal(read_position_coeffs(v.arrays)[:, :, :], read_position_coeffs(pack.view().arrays)[:, :k1 + 2, :])



def test_a_path_container_pack_consolidates_and_reads_back_as_the_merged_pack(spec_grid):
    """The field path channel in a container of two ranges (the second bit-packed): the mode groups are the ranges, the
    ranges are columns as stored, and a view reads them back to the merged pack's replay."""
    spec, grid, tmp = spec_grid
    cont = [{"modes": [0, 2], "bits": 12}, {"modes": [2, 4], "bits": 6}]
    packs = []
    for blk, seed in ((0, 7), (1, 8)):
        want = np.zeros(grid.shape, np.int64); want[blk] = 12
        w = walk_spec(spec, T_max=8e-4, block_h=2e-4, seed=seed, require_gpu=False, K=3, K_c3=4, field_cutoff_max_m=25e-6, n_probe=20_000, adaptive_steps=True,
                      seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}))
        os.makedirs(str(tmp / "pshards"), exist_ok=True)
        packs.append(build_replay_pack(w, id=f"t/p{blk}", license="x", citation="x", K=3, voxel_grid=grid, susc_path_K=4, field_container=cont, position_container=((None, 8),), blt_container=((None, 8),),
                                       device="numpy", out_path=str(tmp / "pshards" / f"block-000{blk}.p1.rpk")))
    out = str(tmp / "players")
    consolidate(str(tmp / "pshards"), out, blocks=[0, 1], id="t/pcolumns")
    merged = merge_packs(packs, id="t/pmerged")
    pack = ReplayPack.open(out)
    assert pack.path_groups == [2, 4] and "susc_path_dct_b1u6" in pack.columns and "susc_path_dct" not in pack.columns
    v = pack.view(modes=4)
    pm = v.meta["compression"]["channels"]["susc_field_basis"]
    assert pm["container"] == cont and pm["K"] == 4
    v2 = pack.view(modes=2)
    assert v2.meta["compression"]["channels"]["susc_field_basis"]["container"] == cont[:1]
    from dmipy_sim.replay.bank import susc_path_coeffs
    C_full = susc_path_coeffs(v.arrays, pm)[0]; C_2 = susc_path_coeffs(v2.arrays, v2.meta["compression"]["channels"]["susc_field_basis"])[0]
    np.testing.assert_array_equal(C_2, C_full[:, :, :2])
    C_m = susc_path_coeffs(merged.arrays, merged.meta["compression"]["channels"]["susc_field_basis"])[0]
    assert sorted(np.round(np.abs(C_m).sum(axis=(1, 2)), 12)) == sorted(np.round(np.abs(C_full).sum(axis=(1, 2)), 12))


def _shard_pools(spec, grid, tmp, name, block, per, seed, *, pools, out_dir):
    """A shard of ``block`` that walks only ``pools`` (the others get no walkers), in the container every shard of a
    fill shares (the certifying pack's, which the walker writes; shards built apart would each derive their own)."""
    want = np.zeros(grid.shape, np.int64); want[block] = per
    w = walk_spec(spec, T_max=8e-4, block_h=2e-4, seed=seed, require_gpu=False, K=3,
                  seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={p: (want if p in pools else 0 * want) for p in ("extra", "intra")}))
    os.makedirs(out_dir, exist_ok=True)
    return build_replay_pack(w, id=f"t/{name}", license="x", citation="x", K=3, voxel_grid=grid, out_path=os.path.join(out_dir, f"{name}.rpk"),
                             position_container=((16, 16), (None, 8)), blt_container=((16, 16), (None, 8)))


def _layout_bytes(out, manifest, index):
    cols = {}
    for k, c in manifest["columns"].items():
        cols[k] = b"".join(open(os.path.join(out, p["file"]), "rb").read()[p["data_offset"]:p["data_offset"] + p["nbytes"]] for p in c["parts"])
    return cols, index


def test_shards_prepared_apart_and_appended_in_order_write_the_streaming_passs_layout(layout, tmp_path):
    """Each shard prepared on its own (:func:`prepare_block`, the later one first), saved and loaded back, then
    appended in block order across a checkpoint and a resume, writes the layout ``consolidate`` writes: the same
    column bytes, index and manifest."""
    import json
    from dmipy_sim.fill.consolidate import Consolidator, PreparedBlock, prepare_block
    from dmipy_sim.replay.replay import read_rpk
    out, manifest, index, _, _, _, _, tmp = layout
    names = ["block-0000.p1.rpk", "block-0001.p1.rpk"]
    for nm in reversed(names):
        block = prepare_block(nm, read_rpk(str(tmp / "shards" / nm))).save(str(tmp_path / "prepared" / nm))
        assert block.directory == str(tmp_path / "prepared" / nm) and not os.path.exists(block.directory + ".part")
    con = Consolidator(str(tmp_path / "out"))
    con.append(PreparedBlock.load(str(tmp_path / "prepared" / names[0]))); con.checkpoint()
    con = Consolidator(str(tmp_path / "out")); con.resume()
    con.append(PreparedBlock.load(str(tmp_path / "prepared" / names[1]))); con.checkpoint()
    m2, i2 = con.finish("t/columns")
    assert i2 == index
    assert _layout_bytes(str(tmp_path / "out"), m2, i2)[0] == _layout_bytes(out, manifest, index)[0]
    assert json.load(open(tmp_path / "out" / "manifest.json")) == json.load(open(os.path.join(out, "manifest.json")))


def test_a_shard_prepared_on_another_grid_is_refused(layout, tmp_path):
    """A shard whose rows were sorted on a grid other than the layout's does not append."""
    from dmipy_sim.fill.consolidate import Consolidator, prepare_block
    from dmipy_sim.replay.replay import read_rpk
    _, _, _, _, _, grid, _, tmp = layout
    con = Consolidator(str(tmp_path / "out"))
    con.append(prepare_block("block-0000.p1.rpk", read_rpk(str(tmp / "shards" / "block-0000.p1.rpk"))))
    other = dict(grid.to_meta()); other["origin_m"] = [float(x) + 1e-15 for x in other["origin_m"]]
    block = prepare_block("block-0001.p1.rpk", read_rpk(str(tmp / "shards" / "block-0001.p1.rpk")), layout_grid=other)
    with pytest.raises(ValueError, match="prepared on the grid"):
        con.append(block)


@pytest.mark.parametrize("split", ["pools", "rounds", "dealt"])
def test_a_block_given_as_round_shards_consolidates_as_the_block_given_whole(spec_grid, split):
    """A block's round shards (``block-NNNN.<pool>.r<r>.rpk``: one pool's walkers, or one round of a pool's) or dealt
    shards (``block-NNNN.s<idx>.<pool>.r<r>.rpk`` named by the shard's ``.walk.json``)
    consolidate to the same index and column bytes as the block's merged pack given as its one shard."""
    spec, grid, tmp = spec_grid
    spec = spec_without_source(spec)
    tmp = tmp / f"rounds-{split}"
    if split == "pools":
        parts = [_shard_pools(spec, grid, tmp, "block-0000.intra.r0", 0, 6, 31, pools=("intra",), out_dir=str(tmp / "rounds")),
                 _shard_pools(spec, grid, tmp, "block-0000.extra.r0", 0, 24, 32, pools=("extra",), out_dir=str(tmp / "rounds"))]
    elif split == "dealt":              # block-NNNN.s<idx>.<pool>.rpk: the shards a node walks of a dealt block
        parts = [_shard_pools(spec, grid, tmp, "block-0000.s0.intra.r0", 0, 6, 31, pools=("intra",), out_dir=str(tmp / "rounds")),
                 _shard_pools(spec, grid, tmp, "block-0000.s1.intra.r0", 0, 6, 33, pools=("intra",), out_dir=str(tmp / "rounds"))]
        import json   # each shard's summary names its files; an orphan round of a failed shard has none and is ignored
        for idx in (0, 1):
            json.dump({"files": [f"blocks/t/block-0000.s{idx}.intra.r0.rpk"]}, open(str(tmp / "rounds" / f"block-0000.s{idx}.walk.json"), "w"))
        _shard_pools(spec, grid, tmp, "block-0000.s2.extra.r0", 0, 6, 35, pools=("extra",), out_dir=str(tmp / "rounds"))
    else:
        parts = [_shard_pools(spec, grid, tmp, "block-0000.intra.r0", 0, 6, 31, pools=("intra",), out_dir=str(tmp / "rounds")),
                 _shard_pools(spec, grid, tmp, "block-0000.intra.r1", 0, 6, 33, pools=("intra",), out_dir=str(tmp / "rounds"))]
    os.makedirs(str(tmp / "whole"), exist_ok=True)
    merge_packs(sorted(parts, key=lambda p: p.meta["id"]), id="t/block-0000", overlap="recertify", out_path=str(tmp / "whole" / "block-0000.rpk"))
    m_r, i_r = consolidate(str(tmp / "rounds"), str(tmp / "out-rounds"), blocks=[0], id="t/columns")
    m_w, i_w = consolidate(str(tmp / "whole"), str(tmp / "out-whole"), blocks=[0], id="t/columns")
    cols_r, _ = _layout_bytes(str(tmp / "out-rounds"), m_r, i_r)
    cols_w, _ = _layout_bytes(str(tmp / "out-whole"), m_w, i_w)
    assert i_r == i_w and i_r["n_rows"] > 0
    assert cols_r.keys() == cols_w.keys()
    for k in cols_r:
        assert cols_r[k] == cols_w[k], k
    assert m_r["meta"]["fidelity"] == m_w["meta"]["fidelity"]


def test_a_block_consolidates_alike_whatever_walker_ranges_it_is_read_in(spec_grid, monkeypatch):
    """A block's pack read a few walkers at a time (``compression.SLICE_BYTES`` for five walkers' coefficients, so the
    band statistics and the row gathers run over many ranges) consolidates to the same column bytes, index and meta as
    read in one range; the band and mode variance tables agree to the rounding of sums over other ranges."""
    import dmipy_sim.replay.compression as cx
    spec, grid, tmp = spec_grid
    spec = spec_without_source(spec)
    tmp = tmp / "ranges"
    _shard_pools(spec, grid, tmp, "block-0000.intra.r0", 0, 6, 41, pools=("intra",), out_dir=str(tmp / "rounds"))
    _shard_pools(spec, grid, tmp, "block-0000.extra.r0", 0, 9, 42, pools=("extra",), out_dir=str(tmp / "rounds"))
    out = {}
    for name, sb in (("one", 1 << 40), ("few", 8 * 3 * 5 * 5)):
        monkeypatch.setattr(cx, "SLICE_BYTES", sb)
        m, i = consolidate(str(tmp / "rounds"), str(tmp / f"out-{name}"), blocks=[0], id="t/columns")
        out[name] = (m, i, _layout_bytes(str(tmp / f"out-{name}"), m, i)[0])
    (m1, i1, c1), (m2, i2, c2) = out["one"], out["few"]
    assert i1 == i2 and i1["n_rows"] > 0 and c1.keys() == c2.keys()
    for k in c1:
        assert c1[k] == c2[k], k
    tables = ("band_variance", "band_variance_max", "path_mode_variance")
    strip = lambda m: dict(m, columnar={k: v for k, v in m["columnar"].items() if k not in tables})
    assert strip(m1["meta"]) == strip(m2["meta"])
    for k in tables:
        a, b = m1["meta"]["columnar"].get(k), m2["meta"]["columnar"].get(k)
        assert (a is None) == (b is None)
        if a is not None:
            np.testing.assert_allclose(np.asarray(b, float), np.asarray(a, float), rtol=1e-12, err_msg=k)
