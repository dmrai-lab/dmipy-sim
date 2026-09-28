"""The columnar layout of :mod:`dmipy_sim.replay.columnar`, re-tiled into voxel blocks (dmipy-sim#505): the same
``manifest.json``/``index.json`` schema, so :func:`~dmipy_sim.replay.columnar.open_columnar` reads it unchanged --
only the row order and the file arrangement differ. Rows are grouped into 4x4x4 voxel blocks (block ijk = voxel
ijk // 4), the blocks ordered by the Morton (Z-order) code of the block ijk, and within a block rows keep the
source's (voxel ijk lexicographic, pool) order; global row numbers are re-assigned in this block-major order. Every
per-row column is split into five TIERS (``t0``, ``t1``, ``t2``, ``contact``, ``modes``, cheapest first), each
tier's columns of one block stored as named tensors in ONE safetensors file (``<tier>/b%04d.safetensors``), so a
consumer copying the layout to local disk (a Space's start-up copy) can fetch a tier whole and a z-slab of it as
one contiguous read per tier file instead of thousands of scattered ranges. The five whole-pack tables
(``voxel_ijk``, ``voxel_certificate``, ``pos_band_scale``, ``blt_band_scale``, ``susc_path_scale``) move to one
``tables.safetensors`` at the layout root, copied once.

Every tensor written is a byte-identical gather of the source's rows (``np.memmap`` on the source's column parts,
sliced and concatenated; nothing is decoded or re-encoded). The manifest is assembled from each block's own
safetensors header -- the offsets are never computed by hand.

    python -m dmipy_sim.fill.blocks SOURCE_DIR OUT_DIR [--workers N]
    write_blocks(source_dir, out_dir, workers=8)

``SOURCE_DIR`` is a local columnar layout as :mod:`dmipy_sim.fill.consolidate` writes it (``manifest.json``,
``index.json``, column files) -- not the Hub; pull it to local disk first (``huggingface_hub.snapshot_download``).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import multiprocessing
import os
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from safetensors.numpy import save_file

_MMAP_CACHE: dict = {}


def _memmap(path):
    mm = _MMAP_CACHE.get(path)
    if mm is None:
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        _MMAP_CACHE[path] = mm
    return mm


def _morton3(x, y, z):
    """The Morton (Z-order) code of one 3-D integer coordinate (bit-interleaved, 21 bits/axis)."""
    def part1by2(n):
        n = int(n) & 0x1fffff
        n = (n | (n << 32)) & 0x1f00000000ffff
        n = (n | (n << 16)) & 0x1f0000ff0000ff
        n = (n | (n << 8)) & 0x100f00f00f00f00f
        n = (n | (n << 4)) & 0x10c30c30c30c30c3
        n = (n | (n << 2)) & 0x1249249249249249
        return n
    return part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2)


def _read_header(path):
    """``(header_len, header)`` of a safetensors file: the length prefix and the parsed JSON header."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return n, header


def _read_range(col, source_dir, s, e):
    """Rows ``[s, e)`` of one manifest column, gathered from its parts by ``np.memmap`` -- a byte-identical
    slice, never decoded."""
    dt = np.dtype(col["dtype"]); shape = col["shape"]
    parts = col["parts"] if "parts" in col else [{"file": col["file"], "rows": [0, shape[0]],
                                                    "data_offset": col["data_offset"], "nbytes": col["nbytes"]}]
    rowbytes = int(np.prod(shape[1:], dtype=np.int64)) * dt.itemsize
    chunks = []
    for p in parts:
        ps, pe = (0, shape[0]) if p.get("rows") is None else p["rows"]     # a table's one part spans it whole
        lo, hi = max(s, ps), min(e, pe)
        if lo < hi:
            mm = _memmap(os.path.join(source_dir, p["file"]))
            off = p["data_offset"] + (lo - ps) * rowbytes
            arr = np.frombuffer(mm, dtype=dt, count=(hi - lo) * int(np.prod(shape[1:], dtype=np.int64)), offset=off)
            chunks.append((lo, arr.reshape((hi - lo,) + tuple(shape[1:]))))
    if not chunks:
        raise ValueError(f"rows [{s}, {e}) are not covered by any part of a column")
    chunks.sort(key=lambda c: c[0])
    return chunks[0][1] if len(chunks) == 1 else np.concatenate([a for _, a in chunks], axis=0)


def _read_whole(col, source_dir):
    return _read_range(col, source_dir, 0, col["shape"][0])


# ---- tiers: which per-row column goes where
_T0_SCALARS = ("band_block", "comp_static", "spin_weights")
_TIER_ORDER = ("t0", "t1", "t2", "contact", "modes")


def _tier_of(name):
    if name in _T0_SCALARS:
        return "t0"
    if name.startswith("pos_"):
        suf = name.rsplit("_", 1)[-1]
        if suf in ("ends", "b0", "b1"):
            return "t0"
        if suf == "b2":
            return "t1"
        if suf in ("b3", "b4", "b5"):
            return "t2"
    if name.startswith("blt_"):
        return "contact"
    if name.startswith("susc_path_m"):
        return "modes"
    raise ValueError(f"the block layout writer has no tier for the per-row column {name!r}")


def _classify_tiers(per_row_cols):
    """The per-row columns grouped into tiers, refusing a source that carries part but not all of a group a
    tier needs (e.g. ``pos_x_b2`` without ``pos_y_b2``/``pos_z_b2``, or one of the three ``blt_`` columns without
    the other two)."""
    have = set(per_row_cols)
    for req in ("band_block", "comp_static", "spin_weights", "pos_x_ends", "pos_y_ends", "pos_z_ends"):
        if req not in have:
            raise ValueError(f"the source is missing the required column {req!r}")
    for suf in ("ends", "b0", "b1", "b2", "b3", "b4", "b5"):
        names = {f"pos_{a}_{suf}" for a in "xyz"}
        present = names & have
        if present and present != names:
            raise ValueError(f"the source is missing {sorted(names - present)} (pos_*_{suf} must be present on all three axes)")
    blt = {"blt_b0", "blt_endpoint", "blt_start"}
    present = blt & have
    if present and present != blt:
        raise ValueError(f"the source is missing {sorted(blt - present)} (the contact tier needs all three)")
    tiers = {t: [] for t in _TIER_ORDER}
    for name in per_row_cols:
        tiers[_tier_of(name)].append(name)
    return tiers


def _write_block(job):
    source_dir, out_dir, bid, src_ranges, tier_cols, columns = job
    nbytes = 0
    for tier, cols in tier_cols.items():
        if not cols:
            continue
        tensors = {}
        for name in cols:
            col = columns[name]
            arrs = [_read_range(col, source_dir, s, e) for s, e in src_ranges]
            tensors[name] = arrs[0] if len(arrs) == 1 else np.concatenate(arrs, axis=0)
        path = os.path.join(out_dir, tier, f"b{bid:04d}.safetensors")
        save_file(tensors, path)
        nbytes += os.path.getsize(path)
    return bid, nbytes


def _progress(done, total, nbytes, t0, force=False):
    now = time.time()
    if not force and done < total and now - _progress._last < 2.0:
        return
    _progress._last = now
    dt = now - t0
    rate = (nbytes / dt / 1e9) if dt > 0 else 0.0
    print(f"[fill.blocks] {done}/{total} blocks, {nbytes / 1e9:.3f} GB written, {rate:.3f} GB/s", file=sys.stderr)


_progress._last = 0.0


def write_blocks(source_dir, out_dir, *, block_shape=(4, 4, 4), workers=8):
    """The columnar layout at ``source_dir`` (a local directory) re-tiled into 4x4x4 voxel blocks under
    ``out_dir``. Returns ``(manifest, index)`` -- the same objects written to ``out_dir/manifest.json`` and
    ``out_dir/index.json``. Refuses a source missing a required per-row column (:func:`_classify_tiers`), a
    column whose block parts do not tile ``[0, n_rows)`` without gaps or overlaps, or a regrouping whose row
    count disagrees with the source's."""
    t0 = time.time()
    manifest_path = os.path.join(source_dir, "manifest.json")
    index_path = os.path.join(source_dir, "index.json")
    manifest_bytes = open(manifest_path, "rb").read(); index_bytes = open(index_path, "rb").read()
    manifest = json.loads(manifest_bytes); index = json.loads(index_bytes)
    columns = manifest["columns"]; n_rows_src = int(index["n_rows"])

    bx, by, bz = block_shape

    def block_of(ijk):
        return (int(ijk[0]) // bx, int(ijk[1]) // by, int(ijk[2]) // bz)

    groups: dict = {}
    for r in index["rows"]:
        groups.setdefault(block_of(r["ijk"]), []).append(r)
    block_ijks = sorted(groups, key=lambda b: _morton3(*b))

    new_rows, block_records, src_ranges_by_block = [], [], []
    row = 0
    for bid, bijk in enumerate(block_ijks):
        entries = sorted(groups[bijk], key=lambda r: (tuple(r["ijk"]), r["pool"]))
        start = row; ranges = []; voxels = set()
        for e in entries:
            s, e_ = int(e["start"]), int(e["end"]); n = e_ - s
            ranges.append((s, e_)); voxels.add(tuple(int(x) for x in e["ijk"]))
            new_rows.append({"ijk": [int(x) for x in e["ijk"]], "pool": int(e["pool"]), "start": row, "end": row + n})
            row += n
        block_records.append({"id": bid, "ijk": list(bijk), "rows": [start, row], "voxels": len(voxels)})
        src_ranges_by_block.append(ranges)
    n_rows = row
    if n_rows != n_rows_src:
        raise ValueError(f"the block regrouping covers {n_rows} rows, the source has {n_rows_src}")
    n_blocks = len(block_ijks)

    per_row_cols = [n for n, c in columns.items() if c["per_row"]]
    table_cols = [n for n, c in columns.items() if not c["per_row"]]
    tier_cols = _classify_tiers(per_row_cols)

    os.makedirs(out_dir, exist_ok=True)
    for tier, cols in tier_cols.items():
        if cols:
            os.makedirs(os.path.join(out_dir, tier), exist_ok=True)

    jobs = [(source_dir, out_dir, bid, src_ranges_by_block[bid], tier_cols, columns) for bid in range(n_blocks)]
    done = 0; bytes_written = 0
    if workers and workers > 1 and n_blocks > 1:
        # spawn, not fork: dmipy_sim (hence this module, once pickled function references are unpickled in the
        # worker) imports JAX, and JAX warns that os.fork() with its threads risks a deadlock
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            futs = [ex.submit(_write_block, j) for j in jobs]
            for fut in as_completed(futs):
                _bid, nbytes = fut.result(); done += 1; bytes_written += nbytes
                _progress(done, n_blocks, bytes_written, t0)
    else:
        for j in jobs:
            _bid, nbytes = _write_block(j); done += 1; bytes_written += nbytes
            _progress(done, n_blocks, bytes_written, t0)
    _progress(done, n_blocks, bytes_written, t0, force=True)

    # the five tables, once, at the root
    table_arrays = {name: _read_whole(columns[name], source_dir) for name in table_cols}
    tables_path = os.path.join(out_dir, "tables.safetensors")
    save_file(table_arrays, tables_path)
    hdr_n, header = _read_header(tables_path)
    manifests = {}
    for name in table_cols:
        entry = header[name]; off0, off1 = entry["data_offsets"]
        manifests[name] = {"dtype": columns[name]["dtype"], "shape": columns[name]["shape"], "per_row": False,
                            "parts": [{"file": "tables.safetensors", "rows": None,
                                       "data_offset": 8 + hdr_n + off0, "nbytes": off1 - off0}]}

    header_cache: dict = {}
    for name in per_row_cols:
        tier = _tier_of(name); c = columns[name]; parts = []
        for bid in range(n_blocks):
            key = (tier, bid)
            if key not in header_cache:
                header_cache[key] = _read_header(os.path.join(out_dir, tier, f"b{bid:04d}.safetensors"))
            hdr_n, header = header_cache[key]
            entry = header[name]; off0, off1 = entry["data_offsets"]
            s, e = block_records[bid]["rows"]
            if parts and parts[-1]["rows"][1] != s:
                raise ValueError(f"column {name!r}: block {bid}'s rows [{s}, {e}) do not follow the previous "
                                  f"part's end {parts[-1]['rows'][1]} (a gap or an overlap)")
            parts.append({"file": f"{tier}/b{bid:04d}.safetensors", "rows": [s, e],
                          "data_offset": 8 + hdr_n + off0, "nbytes": off1 - off0})
        covered = sum(p["rows"][1] - p["rows"][0] for p in parts)
        if covered != n_rows:
            raise ValueError(f"column {name!r}: parts cover {covered} rows, expected {n_rows}")
        manifests[name] = {"dtype": c["dtype"], "shape": [n_rows] + list(c["shape"][1:]), "per_row": True, "parts": parts}

    out_columns = {name: manifests[name] for name in columns}   # the source's own column order (plan()'s reads it)
    meta = json.loads(json.dumps(manifest["meta"]))
    meta["columnar"] = dict(meta["columnar"], layout="blocks",
                             source={"prefix": os.path.abspath(source_dir),
                                     "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                                     "index_sha256": hashlib.sha256(index_bytes).hexdigest()})
    out_manifest = {"meta": meta, "columns": out_columns}
    out_index = {"grid": index["grid"], "n_rows": n_rows, "band_groups": index["band_groups"],
                 "path_groups": index.get("path_groups", []), "rows": new_rows,
                 "blocks": block_records, "block_shape": list(block_shape), "block_order": "morton"}
    json.dump(out_manifest, open(os.path.join(out_dir, "manifest.json"), "w"))
    json.dump(out_index, open(os.path.join(out_dir, "index.json"), "w"))
    print(f"[fill.blocks] done: {n_rows} rows, {n_blocks} blocks, {len(out_columns)} columns, "
          f"{bytes_written / 1e9:.3f} GB, {time.time() - t0:.0f} s", file=sys.stderr)
    return out_manifest, out_index


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dmipy_sim.fill.blocks", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source_dir", help="a local columnar layout (manifest.json, index.json, column files)")
    ap.add_argument("out_dir", help="where the block layout is written")
    ap.add_argument("--workers", type=int, default=8, help="block-writing process pool size")
    a = ap.parse_args(argv)
    write_blocks(a.source_dir, a.out_dir, workers=a.workers)


if __name__ == "__main__":
    main()
