"""The block layout (dmipy-sim#505): :func:`dmipy_sim.fill.blocks.write_blocks` re-tiles a columnar layout into
4x4x4 voxel blocks without touching a single coefficient, so :func:`~dmipy_sim.replay.columnar.open_columnar`
reads it with the same code as the original layout and every view it returns carries the same walkers, in the
same per-(voxel, pool) order, as the source's.

A block layout's row order is GLOBALLY different from the source's by design (rows are grouped by block, not by
plain voxel lexicographic order), so a whole-pack ``view(voxels=None)`` array is a PERMUTATION of the source's,
not an equal array -- the two are only meaningfully comparable per (voxel, pool) group. Every equality check here
is therefore either a single-voxel view (whose own row order is unaffected: `pool` is always the second, ascending
sort key) or a full-pack view decomposed back into its (voxel, pool) row ranges before comparison.

``consumer_poc`` (SubstrateCommons/disco-replay, 546,853 rows, 137 voxels) is pulled once with::

    HF_HUB_CACHE=/home/rutger/.cache/hf-session python -c "
    from huggingface_hub import snapshot_download
    snapshot_download('SubstrateCommons/disco-replay', repo_type='dataset', allow_patterns=['consumer_poc/*'])"

and the tests that need it skip (never fetch it themselves) when it is not already cached."""
from __future__ import annotations
import json
import os
import random
import struct

import numpy as np
import pytest
from safetensors.numpy import save_file

from dmipy_sim.fill.blocks import write_blocks, _morton3
from dmipy_sim.phantom.grid import Grid
from dmipy_sim.replay.columnar import open_columnar
from dmipy_sim import sequences

REPO = "SubstrateCommons/disco-replay"


def _eq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return np.array_equal(a, b, equal_nan=True) if np.issubdtype(a.dtype, np.floating) else np.array_equal(a, b)


# ---- consumer_poc: pulled once, out of band; skip (never fetch) if it is not there
@pytest.fixture(scope="module")
def consumer_poc_dir():
    from huggingface_hub import snapshot_download
    try:
        root = snapshot_download(REPO, repo_type="dataset", allow_patterns=["consumer_poc/*"], local_files_only=True)
    except Exception:
        pytest.skip(f"consumer_poc is not cached locally; snapshot_download({REPO!r}, repo_type='dataset', "
                     "allow_patterns=['consumer_poc/*']) it once first")
    d = os.path.join(root, "consumer_poc")
    if not os.path.exists(os.path.join(d, "manifest.json")):
        pytest.skip(f"{d} has no manifest.json")
    return d


@pytest.fixture(scope="module")
def blocks_dir(consumer_poc_dir, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("consumer_poc_blocks"))
    write_blocks(consumer_poc_dir, out, workers=4)
    return out


@pytest.fixture(scope="module")
def packs(consumer_poc_dir, blocks_dir):
    return open_columnar(consumer_poc_dir), open_columnar(blocks_dir)


@pytest.fixture(scope="module")
def all_voxels(consumer_poc_dir):
    index = json.load(open(os.path.join(consumer_poc_dir, "index.json")))
    return sorted({tuple(r["ijk"]) for r in index["rows"]})


@pytest.fixture(scope="module")
def random_voxels(all_voxels):
    rng = random.Random(0)
    return rng.sample(all_voxels, 20)


# ---- (a) per-voxel and whole-pack content, bit-identical
def _assert_single_voxel_views_agree(src, blk, voxels, K=32, **kw):
    for v in voxels:
        va = src.view(K=K, voxels=[v], **kw); vb = blk.view(K=K, voxels=[v], **kw)
        assert set(va.arrays) == set(vb.arrays)
        for k in va.arrays:
            assert _eq(va.arrays[k], vb.arrays[k]), f"voxel {v}, column {k!r} differs"


def test_random_voxels_bit_identical(packs, random_voxels):
    src, blk = packs
    _assert_single_voxel_views_agree(src, blk, random_voxels)


def test_random_voxels_bit_identical_K64(packs, random_voxels):
    src, blk = packs
    _assert_single_voxel_views_agree(src, blk, random_voxels, K=64)


def test_random_voxels_bit_identical_modes8(packs, random_voxels):
    src, blk = packs
    _assert_single_voxel_views_agree(src, blk, random_voxels, modes=8)


def test_random_voxels_bit_identical_contact(packs, random_voxels):
    src, blk = packs
    _assert_single_voxel_views_agree(src, blk, random_voxels, contact=True)


def test_every_voxel_bit_identical(packs, all_voxels):
    src, blk = packs
    _assert_single_voxel_views_agree(src, blk, all_voxels)


def _row_ranges(pack):
    """``{ijk: {pool: (start, end)}}`` of a pack's OWN row numbering."""
    d = {}
    for r in pack.index["rows"]:
        d.setdefault(tuple(r["ijk"]), {})[r["pool"]] = (r["start"], r["end"])
    return d


def test_voxels_none_bit_identical(packs, all_voxels):
    """``view(voxels=None)`` on both layouts: same total rows and per-array dtype/shape (the layouts' global row
    order differs by design), and -- decomposed back to (voxel, pool) row ranges -- the same content."""
    src, blk = packs
    for kw in ({"K": 32}, {"K": 64}, {"K": 32, "modes": 8}, {"K": 32, "contact": True}):
        va = src.view(voxels=None, **kw); vb = blk.view(voxels=None, **kw)
        assert set(va.arrays) == set(vb.arrays)
        assert va.n_walkers == vb.n_walkers == src.n_rows
        ra, rb = _row_ranges(src), _row_ranges(blk)
        n_rows = src.n_rows                                      # a VIEW's arrays: per-row ones have this many rows
        per_row_names = {k for k in va.arrays if np.asarray(va.arrays[k]).shape[:1] == (n_rows,)}
        table_names = set(va.arrays) - per_row_names
        for k in table_names:                                  # whole-pack tables: never row-permuted
            assert _eq(va.arrays[k], vb.arrays[k]), f"table column {k!r} differs"
        for v in all_voxels:
            for pool in set(ra[v]) | set(rb[v]):
                sa, ea = ra[v][pool]; sb, eb = rb[v][pool]
                for k in per_row_names:
                    a = np.asarray(va.arrays[k])[sa:ea]; b = np.asarray(vb.arrays[k])[sb:eb]
                    assert _eq(a, b), f"voxel {v}, pool {pool}, column {k!r} differs"


# ---- (b) the block-major index and manifest
def test_index_covers_every_source_entry_once(consumer_poc_dir, blocks_dir):
    src_index = json.load(open(os.path.join(consumer_poc_dir, "index.json")))
    blk_index = json.load(open(os.path.join(blocks_dir, "index.json")))
    assert blk_index["n_rows"] == src_index["n_rows"]

    def counts(index):
        return sorted((tuple(r["ijk"]), r["pool"], r["end"] - r["start"]) for r in index["rows"])
    assert counts(blk_index) == counts(src_index)

    # blocks: 4x4x4, Morton-ordered, rows cover [0, n_rows) with no gap/overlap, voxel counts match membership
    assert blk_index["block_shape"] == [4, 4, 4] and blk_index["block_order"] == "morton"
    blocks = blk_index["blocks"]
    prev_end = 0
    for b in blocks:
        s, e = b["rows"]
        assert s == prev_end, "a gap or overlap between consecutive blocks' row ranges"
        prev_end = e
    assert prev_end == blk_index["n_rows"]
    codes = [_morton3(*b["ijk"]) for b in blocks]
    assert codes == sorted(codes), "blocks are not in Morton order"
    for b in blocks:
        bx, by, bz = b["ijk"]
        voxels_in_block = {tuple(r["ijk"]) for r in blk_index["rows"]
                            if r["start"] >= b["rows"][0] and r["end"] <= b["rows"][1]
                            and (r["ijk"][0] // 4, r["ijk"][1] // 4, r["ijk"][2] // 4) == (bx, by, bz)}
        assert len(voxels_in_block) == b["voxels"]


def test_manifest_parts_tile_without_gaps_or_overlaps(blocks_dir):
    manifest = json.load(open(os.path.join(blocks_dir, "manifest.json")))
    n_rows = json.load(open(os.path.join(blocks_dir, "index.json")))["n_rows"]
    for name, c in manifest["columns"].items():
        if not c["per_row"]:
            continue
        parts = sorted(c["parts"], key=lambda p: p["rows"][0])
        prev = 0
        for p in parts:
            s, e = p["rows"]
            assert s == prev, f"column {name!r}: gap or overlap before row {s} (part {p})"
            prev = e
        assert prev == n_rows, f"column {name!r}: parts cover {prev} rows, not {n_rows}"


def test_meta_is_the_source_plus_two_columnar_keys(consumer_poc_dir, blocks_dir):
    src_meta = json.load(open(os.path.join(consumer_poc_dir, "manifest.json")))["meta"]
    blk_meta = json.load(open(os.path.join(blocks_dir, "manifest.json")))["meta"]
    assert blk_meta["columnar"]["layout"] == "blocks"
    src_bytes = open(os.path.join(consumer_poc_dir, "manifest.json"), "rb").read()
    index_bytes = open(os.path.join(consumer_poc_dir, "index.json"), "rb").read()
    import hashlib
    assert blk_meta["columnar"]["source"]["manifest_sha256"] == hashlib.sha256(src_bytes).hexdigest()
    assert blk_meta["columnar"]["source"]["index_sha256"] == hashlib.sha256(index_bytes).hexdigest()
    src_c = dict(src_meta["columnar"]); blk_c = dict(blk_meta["columnar"])
    del blk_c["layout"], blk_c["source"]
    assert blk_c == src_c
    for k in src_meta:
        if k != "columnar":
            assert blk_meta[k] == src_meta[k]


# ---- (c) plan() agrees
def test_plan_agrees(packs):
    src, blk = packs
    seq = sequences.pgse([[1, 0, 0], [0, 0, 1]], 0.0102, 0.0167, bvalues=[1e9, 1e9], TE=0.0535)
    assert src.plan(seq) == blk.plan(seq)


# ---- existing columnar tests still pass (informational: run separately, see the report)


# ---- (d) iter_views: one view per block, in block order -- needs UNIFORM block sizes, so a small synthetic
# source is built by hand (no walk engine involved: this exercises the layout mechanics, not physics)
def _write_synthetic_source(out_dir, grid_shape=(8, 4, 4), rows_per_voxel=5, seed=0):
    rng = np.random.default_rng(seed)
    grid = Grid(shape=grid_shape, voxel_size_m=(1e-5,) * 3)
    cols = {k: [] for k in ("band_block", "comp_static", "spin_weights",
                             "pos_x_ends", "pos_y_ends", "pos_z_ends",
                             "pos_x_b0", "pos_y_b0", "pos_z_b0", "pos_x_b1", "pos_y_b1", "pos_z_b1")}
    rows_meta = []; row = 0
    for ijk in np.ndindex(*grid_shape):                          # plain (x, y, z) lexicographic source order
        n = rows_per_voxel
        cols["band_block"].append(rng.integers(0, 4, size=n).astype(np.uint16))
        cols["comp_static"].append(np.zeros(n, np.int8))
        cols["spin_weights"].append(np.full(n, 1.0 / n, np.float32))
        for a in "xyz":
            cols[f"pos_{a}_ends"].append(rng.normal(size=(n, 2)).astype(np.float32))
            cols[f"pos_{a}_b0"].append(rng.integers(-1000, 1000, size=(n, 16)).astype(np.int16))
            cols[f"pos_{a}_b1"].append(rng.integers(-100, 100, size=(n, 16)).astype(np.int8))
        rows_meta.append({"ijk": [int(x) for x in ijk], "pool": 0, "start": row, "end": row + n})
        row += n
    n_rows = row
    arrays = {k: np.concatenate(v, axis=0) for k, v in cols.items()}
    os.makedirs(os.path.join(out_dir, "columns"), exist_ok=True)
    data_path = os.path.join(out_dir, "columns", "data.safetensors")
    save_file(arrays, data_path)
    with open(data_path, "rb") as fh:
        hdr_n = struct.unpack("<Q", fh.read(8))[0]; header = json.loads(fh.read(hdr_n))
    columns = {}
    for name, arr in arrays.items():
        off0, off1 = header[name]["data_offsets"]
        columns[name] = {"dtype": str(arr.dtype), "shape": list(arr.shape), "per_row": True,
                          "file": "columns/data.safetensors", "data_offset": 8 + hdr_n + off0, "nbytes": off1 - off0}
    meta = {"compression": {"K": 32, "method": "bridge_dst", "channels": {}},
            "walk_params": {"n_t": 10, "dt_traj": 1e-4, "n_walkers": n_rows, "seed": seed},
            "fidelity": {"per_voxel": {"floor_median": 0.01, "grid": grid.to_meta()}},
            "columnar": {"band_groups": [16, 32], "path_groups": [], "sorted_by": ["voxel", "pool"]}}
    manifest = {"meta": meta, "columns": columns}
    index = {"grid": grid.to_meta(), "n_rows": n_rows, "band_groups": [16, 32], "path_groups": [], "rows": rows_meta}
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"))
    json.dump(index, open(os.path.join(out_dir, "index.json"), "w"))
    return n_rows


@pytest.fixture(scope="module")
def uniform_blocks(tmp_path_factory):
    """Two 4x4x4 blocks of equal size (an 8x4x4 grid, 5 rows/voxel: 320 rows each)."""
    src = str(tmp_path_factory.mktemp("uniform_src"))
    n_rows = _write_synthetic_source(src, grid_shape=(8, 4, 4), rows_per_voxel=5)
    out = str(tmp_path_factory.mktemp("uniform_blocks"))
    manifest, index = write_blocks(src, out, workers=1)
    return out, index, n_rows


def test_iter_views_one_per_block_in_block_order(uniform_blocks):
    out, index, n_rows = uniform_blocks
    blocks = index["blocks"]
    assert len(blocks) == 2
    sizes = {b["rows"][1] - b["rows"][0] for b in blocks}
    assert sizes == {320}                                        # both blocks the same size, by construction

    pack = open_columnar(out)
    views = list(pack.iter_views(chunk_rows=320, K=32))
    assert len(views) == len(blocks)
    row = 0
    for view, b in zip(views, blocks):
        s, e = b["rows"]
        assert view.n_walkers == e - s
        row = e
    assert row == n_rows


# ---- refusals
def test_refuses_a_missing_required_column(tmp_path_factory):
    src = str(tmp_path_factory.mktemp("missing_col_src"))
    _write_synthetic_source(src, grid_shape=(4, 4, 4), rows_per_voxel=2)
    manifest = json.load(open(os.path.join(src, "manifest.json")))
    del manifest["columns"]["pos_y_b1"]
    json.dump(manifest, open(os.path.join(src, "manifest.json"), "w"))
    out = str(tmp_path_factory.mktemp("missing_col_out"))
    with pytest.raises(ValueError, match="pos_y_b1"):
        write_blocks(src, out, workers=1)


def test_refuses_a_row_count_mismatch(tmp_path_factory):
    src = str(tmp_path_factory.mktemp("bad_rows_src"))
    _write_synthetic_source(src, grid_shape=(4, 4, 4), rows_per_voxel=2)
    index = json.load(open(os.path.join(src, "index.json")))
    index["n_rows"] = index["n_rows"] + 1
    json.dump(index, open(os.path.join(src, "index.json"), "w"))
    out = str(tmp_path_factory.mktemp("bad_rows_out"))
    with pytest.raises(ValueError, match="rows"):
        write_blocks(src, out, workers=1)


def test_a_table_part_with_no_row_span_is_read_whole(tmp_path):
    """The full DiSCo manifest writes a table's single part with ``rows: None``; the gather spans the table."""
    import numpy as np, json
    from safetensors.numpy import save_file
    from dmipy_sim.fill.blocks import _read_whole
    arr = np.arange(12, dtype=np.float32).reshape(4, 3)
    save_file({"voxel_certificate": arr}, str(tmp_path / "tables.safetensors"))
    n = int.from_bytes(open(tmp_path / "tables.safetensors", "rb").read(8), "little")
    hdr = json.loads(open(tmp_path / "tables.safetensors", "rb").read()[8:8 + n])
    off0, off1 = hdr["voxel_certificate"]["data_offsets"]
    col = {"dtype": "float32", "shape": [4, 3], "per_row": False,
           "parts": [{"file": "tables.safetensors", "rows": None, "data_offset": 8 + n + off0, "nbytes": off1 - off0}]}
    assert np.array_equal(_read_whole(col, str(tmp_path)), arr)
