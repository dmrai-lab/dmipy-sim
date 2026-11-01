"""The shards of one dataset consolidated into the columnar layout of :mod:`dmipy_sim.replay.columnar` (dmipy-sim#290), in
one streaming pass: every shard is fetched, split into its columns and appended to the open part of each column,
then deleted; a part that reaches ``cap_bytes`` is closed and uploaded while the pass goes on, so the machine that
consolidates never holds more than the open parts. Rows are ordered by (block, voxel, pool), so the row index is
exact after every shard and no shard is read twice; nothing is requantised. The pass checkpoints after every
shard and resumes from the checkpoint. A later pass of a block (a top-up, a repair) is APPENDED: its rows after
the existing ones, its (voxel, pool) ranges beside the old ones in the index, the weights of both renormalised
to the union as a recertifying merge does (:func:`~dmipy_sim.replay.bank.union_weights`), the old rows patched
in the weights part they sit in, and a repaired pool's certificate row the union's count with the new pass's
floor.

The voxel grid is the recipe's, not a shard's: a shard that carries no per-voxel certificate (one a walker wrote,
which measures no floor) is certified here on the recipe's grid (``--manifest``, else the repository's
``manifest.json``; ``grid=`` from Python): its walkers binned by start voxel and pool, the split-half floor of each
(voxel, pool) read from its coefficients (:func:`~dmipy_sim.replay.bank.voxel_certificate`). The shards of the
layout read the field at the intervals of their own rounds; the layout records the coarsest and its term
(:func:`~dmipy_sim.replay.bank.charge_field_read`).

    python -m dmipy_sim.replay.consolidate --repo OWNER/NAME --shards blocks/disco --out DIR --upload disco
    python -m dmipy_sim.replay.consolidate --shards DIR_OF_SHARDS --manifest manifest.json --out DIR --blocks 0-3 --id NAME
    python -m dmipy_sim.replay.consolidate --repo OWNER/NAME --out DIR --upload disco --append DIR_OF_SHARDS --pass 3
    consolidate(shards, out_dir, blocks=..., grid=...)          # from Python, a directory of shards, nothing uploaded

The layout: ``manifest.json`` (the merged meta as :func:`~dmipy_sim.replay.bank.merge_packs` writes it, plus
``columnar`` -- the band and mode groups, the per-band variance tables the read predicates use -- and the column
table: every column a list of parts ``{file, rows, data_offset, nbytes}``), ``index.json`` (the row range of every
(voxel, pool)), ``columns/<name>.p<NNN>.safetensors`` (a part is a safetensors file whose header is written with a
fixed padding and completed at close) and the small tables as one file each.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import queue
import struct
import sys
import threading
import time

import numpy as np

BAND_GROUPS = (16, 32, 64, 128, 192, 256)                  # cumulative band counts of the position groups (the first at 16 bit)
PATH_GROUPS = (8, 16, 32)                                  # cumulative mode counts of the field path groups
POOL_KEY = "comp_static"
HEADER_PAD = 256
ST_DTYPE = {"float16": "F16", "float32": "F32", "float64": "F64", "int8": "I8", "int16": "I16", "int32": "I32", "int64": "I64", "uint8": "U8", "uint16": "U16", "bool": "BOOL"}
log = logging.getLogger("dmipy_sim.replay.consolidate")


def shard_name(block, P):
    """``block-NNNN`` for the whole block, ``block-NNNN.pK`` for pass K of it (``P`` carries ``pass``, None for the whole block)."""
    return f"block-{block:04d}" + (f".p{P['pass']}" if P.get("pass") is not None else "")


def parse_shard(f):
    """``(block, pass)`` of a shard or claim path (``.../block-0012.p1.rpk``, ``.../block-0012.host.json``);
    ``pass`` None for the whole block."""
    rest = f.split("block-")[1]; block = int(rest[:4]); tag = rest[4:].split(".")
    p = tag[1] if len(tag) > 1 else ""
    return block, (int(p[1:]) if p.startswith("p") and p[1:].isdigit() else None)


def manifest_grid(man):
    """The voxel grid a manifest states (``grid``: ``shape``, ``voxel_size_m``, ``origin_m``), attached to the substrate."""
    from ..phantom import Grid
    G = man["grid"]
    return Grid(shape=tuple(G["shape"]), voxel_size_m=tuple(G["voxel_size_m"]), origin_m=tuple(G["origin_m"]))


def band_groups_of(K, container):
    """The stored band groups of a pack of ``K`` bands: with the banded ``container`` (the pack's own ranges, which
    :func:`~dmipy_sim.replay.compression.band_curve` or any explicit container wrote) one group per range, each a
    stored tensor kept whole; without one (the float coefficients) the groups are :data:`BAND_GROUPS` alone. The
    last group ends at ``K``."""
    if container:
        groups = sorted({min(int(r["bands"][1]), K) for r in container})
    else:
        groups = [g for g in BAND_GROUPS if g < K]
    if not groups or groups[-1] != K:
        groups.append(K)
    return tuple(groups)


def path_groups_of(K_path, container=None):
    """The stored mode groups of a path channel of ``K_path`` modes: one per range of its container, else the
    single-depth tensor split at :data:`PATH_GROUPS`."""
    if container:
        return tuple(sorted({min(int(r["modes"][1]), K_path) for r in container}))
    return tuple([g for g in PATH_GROUPS if g < K_path] + [K_path])


class Part:
    """One open safetensors file of one column: rows appended, the header (padded to a fixed size) completed at close."""

    def __init__(self, path, name, dtype, row_shape, rows=0):
        self.path, self.name, self.dtype, self.row_shape = path, name, np.dtype(dtype), tuple(int(x) for x in row_shape)
        self.rowbytes = int(np.prod(self.row_shape, dtype=np.int64)) * self.dtype.itemsize
        self.rows = rows
        if rows == 0:
            self.fh = open(path, "wb"); self.fh.write(self._header(0))
        else:                                              # a resume: the file cut back to the checkpoint's rows
            self.fh = open(path, "r+b"); self.fh.truncate(8 + HEADER_PAD + rows * self.rowbytes); self.fh.seek(0, 2)

    def _header(self, rows):
        h = json.dumps({self.name: {"dtype": ST_DTYPE[self.dtype.name], "shape": [rows, *self.row_shape], "data_offsets": [0, rows * self.rowbytes]}}).encode()
        assert len(h) <= HEADER_PAD
        return struct.pack("<Q", HEADER_PAD) + h + b" " * (HEADER_PAD - len(h))

    @property
    def nbytes(self):
        return self.rows * self.rowbytes

    def append(self, arr):
        arr = np.ascontiguousarray(arr, dtype=self.dtype)
        assert arr.shape[1:] == self.row_shape, (arr.shape, self.row_shape)
        self.fh.write(arr.tobytes()); self.rows += int(arr.shape[0])

    def flush(self):
        self.fh.flush(); os.fsync(self.fh.fileno())

    def close(self):
        self.fh.seek(0); self.fh.write(self._header(self.rows)); self.flush(); self.fh.close()


class Column:
    """A per-row column in capped parts: ``append`` rows, ``close_part`` when the open part is full."""

    def __init__(self, name, dtype, row_shape, out_dir, cap_bytes, on_close):
        self.name, self.dtype, self.row_shape, self.out_dir, self.cap, self.on_close = name, np.dtype(dtype), tuple(row_shape), out_dir, cap_bytes, on_close
        self.parts = []                                    # closed: {file, rows: [start, end], data_offset, nbytes}
        self.cur = None; self.row = 0; self.start = 0

    def _path(self, i):
        return os.path.join(self.out_dir, "columns", f"{self.name}.p{i:03d}.safetensors")

    def append(self, arr):
        if self.cur is None:
            self.cur = Part(self._path(len(self.parts)), self.name, self.dtype, self.row_shape); self.start = self.row
        self.cur.append(arr); self.row += int(arr.shape[0])
        if self.cur.nbytes >= self.cap:
            self.close_part()

    def close_part(self):
        if self.cur is None or self.cur.rows == 0:
            return
        self.cur.close()
        rec = {"file": f"columns/{os.path.basename(self.cur.path)}", "rows": [self.start, self.row], "data_offset": 8 + HEADER_PAD, "nbytes": self.cur.nbytes}
        self.parts.append(rec); self.on_close(self.cur.path, rec); self.cur = None

    def state(self):
        return {"parts": self.parts, "row": self.row, "open": None if self.cur is None else {"start": self.start, "rows": self.cur.rows}}

    def restore(self, st, uploaded):
        self.parts, self.row = list(st["parts"]), int(st["row"])
        if st["open"] is not None:
            self.start = int(st["open"]["start"]); rows = int(st["open"]["rows"]); path = self._path(len(self.parts))
            rec = {"file": f"columns/{os.path.basename(path)}", "rows": [self.start, self.start + rows], "data_offset": 8 + HEADER_PAD,
                   "nbytes": rows * int(np.prod(self.row_shape, dtype=np.int64)) * self.dtype.itemsize}
            if not os.path.exists(path):                       # closed and uploaded by a finish that ended after its close
                if rec["file"] not in uploaded:
                    raise FileNotFoundError(f"{path}: the open part is gone and not on the hub's upload list")
                self.parts.append(rec); return
            self.cur = Part(path, self.name, self.dtype, self.row_shape, rows=rows)

    def manifest(self, n_rows):
        return {"dtype": self.dtype.name, "shape": [n_rows, *self.row_shape], "per_row": True, "parts": self.parts}


class Uploader:
    """Closed parts uploaded to the hub in the background, several per commit, at most ``depth`` waiting (the
    writer blocks beyond that, which bounds the disk), each deleted after its commit; a failed commit is retried
    with backoff. Without a ``prefix`` nothing is uploaded and nothing deleted."""

    def __init__(self, repo, prefix, depth=6, batch=6):
        self.repo, self.prefix, self.batch = repo, prefix, batch
        self.enabled = prefix is not None
        self.q = queue.Queue(maxsize=depth); self.done = []; self.failed = None; self.lock = threading.Lock()
        if self.enabled:
            from huggingface_hub import HfApi
            self.api = HfApi(); self.thread = threading.Thread(target=self._run, daemon=True); self.thread.start()

    def put(self, path, rel):
        if self.enabled:
            self.q.put((path, rel))

    def _commit(self, items):
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(path_in_repo=f"{self.prefix}/{rel}", path_or_fileobj=path) for path, rel in items]
        mb = sum(os.path.getsize(p) for p, _ in items) / 1e6
        for attempt in range(8):
            try:
                t0 = time.time()
                self.api.create_commit(repo_id=self.repo, repo_type="dataset", operations=ops, commit_message=f"consolidate: {', '.join(r for _, r in items)}")
                log.info("uploaded %d files (%.0f MB, %.0f s): %s", len(items), mb, time.time() - t0, ", ".join(r for _, r in items))
                for path, rel in items:
                    os.remove(path)
                    with self.lock:
                        self.done.append(rel)
                return True
            except Exception as e:
                log.warning("commit of %d files failed (%s): retry %d", len(items), e, attempt + 1); time.sleep(min(600, 15 * 2 ** attempt))
        self.failed = items[0][1]; log.error("commit of %s gave up", self.failed)
        return False

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            items = [item]; t0 = time.time()
            while len(items) < self.batch and time.time() - t0 < 30:
                try:
                    nxt = self.q.get(timeout=5)
                except queue.Empty:
                    continue
                if nxt is None:
                    self._commit(items); return
                items.append(nxt)
            if not self._commit(items):
                return

    def finish(self):
        if self.enabled:
            self.q.put(None); self.thread.join()

    def hub_files(self):
        """The layout's files on the hub (a finish that ended after uploading kept no local record of them)."""
        if not self.enabled:
            return set()
        p = self.prefix + "/"
        return {f[len(p):] for f in self.api.list_repo_files(self.repo, repo_type="dataset") if f.startswith(p)}


class Shards:
    """Where the shards are: a directory of ``.rpk`` files, or a prefix of the hub's repository; ``name in shards``
    tests presence, ``fetch(name, scratch)`` gives a local path (a hub shard is downloaded and then deleted)."""

    def __init__(self, where, repo=None):
        self.remote = not os.path.isdir(where); self.uri = where; self.repo = repo
        if self.remote:
            from huggingface_hub import HfApi
            listed = {f.split("/")[-1] for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith(where + "/")}
        else:
            listed = set(os.listdir(where))
        self.names = {f for f in listed if f.endswith(".rpk")}
        self.summaries = {f for f in listed if f.endswith(".walk.json")}      # a dealt shard's summary, ``block-NNNN[.pK].s<idx>.walk.json``

    def read_json(self, name):
        """The parsed JSON file ``name`` of the shards' location."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            return json.load(open(self.fetch(name, tmp)))

    def __contains__(self, name):
        return name in self.names

    def fetch(self, name, scratch):
        if not self.remote:
            return os.path.join(self.uri, name)
        from huggingface_hub import hf_hub_download
        for attempt in range(8):
            try:
                return hf_hub_download(self.repo, f"{self.uri}/{name}", repo_type="dataset", local_dir=scratch)
            except Exception as e:
                log.warning("fetch of %s failed (%s): retry %d", name, e, attempt + 1); time.sleep(min(600, 15 * 2 ** attempt))
        raise IOError(f"cannot fetch {name}")


def _no_grid(name):
    return ValueError(f"shard {name} carries no per-voxel certificate (fidelity.per_voxel) and no voxel grid is given: the grid is "
                      "the recipe's (consolidate(grid=...), --manifest), or the shards are packed with voxel_grid=")


def certify_voxels(name, pk, grid):
    """``pk.meta`` of a shard on the recipe's ``grid``: a shard with a per-voxel certificate must be on that grid; one
    without is certified here (:func:`~dmipy_sim.replay.bank.voxel_certificate`), and the voxel its walker states (a
    writer's walker-leading ``voxel_ijk``) is checked against the binning of its start."""
    from .bank import _agree, voxel_certificate
    from dmipy_replay.phantom.grid import Grid
    meta = pk.meta; pv = (meta.get("fidelity") or {}).get("per_voxel")
    if pv is not None:
        if grid is not None and not _agree(Grid.from_meta(pv["grid"]).to_meta(), grid.to_meta()):
            raise ValueError(f"shard {name}: its per-voxel grid {pv['grid']!r} is not the recipe's {grid.to_meta()!r}")
        return meta
    if grid is None:
        raise _no_grid(name)
    stated = pk.arrays.get("voxel_ijk")
    arrays, pv = voxel_certificate(pk, grid)
    if stated is not None and np.asarray(stated).shape == (int(meta["walk_params"]["n_walkers"]), 3):
        ijk, _ = grid.bin(pk.r0)
        elsewhere = int((np.asarray(stated, np.int64) != ijk).any(axis=1).sum())
        if elsewhere:                  # a seed on a voxel face, binned into the neighbour once its start is stored
            log.warning("%s: %d of %d walkers start in another voxel than the one they were seeded for", name, elsewhere, len(ijk))
        pv["binned_off_stratum"] = elsewhere
    pk.arrays = {**pk.arrays, **arrays}
    meta = json.loads(json.dumps(meta)); meta["fidelity"] = dict(meta.get("fidelity") or {}, per_voxel=pv); pk.meta = meta
    return meta


def columns_of(k, x, band_groups, path_groups):
    """The layout's columns of rows ``x`` of tensor ``k``: the float positions as their two ends and their
    ``band_groups``, the float field modes as their ``path_groups``, any other tensor as itself."""
    if k in ("pos_x", "pos_y", "pos_z"):
        lo, out = 0, {f"{k}_ends": x[:, :2]}
        for g, hi in enumerate(band_groups):
            out[f"{k}_b{g}"] = x[:, 2 + lo:2 + hi]; lo = hi
        return out
    if k == "susc_path_dct":
        lo, out = 0, {}
        for g, hi in enumerate(path_groups):
            out[f"susc_path_m{g}"] = x[:, :, lo:hi]; lo = hi
        return out
    return {k: x}


def _canonical(x):
    return json.dumps(x, sort_keys=True)


class PreparedBlock:
    """One shard's share of the layout, all :meth:`Consolidator.append` needs and nothing the layout accumulates: the
    shard's identity and certified meta, the grid its rows are sorted on, its band and mode sums, its scale tables
    and band blocks, its (voxel, pool) row ranges, its certificate rows and its columns in (voxel, pool) order. Made
    by :func:`prepare_block` from a pack, whose columns are gathered from the pack as :meth:`chunks` reads them, or by
    :meth:`load` from what :meth:`save` wrote, whose columns are read from the saved files a range of rows at a time:
    the two append to the same bytes."""

    RECORD = "block.pkl"

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def chunks(self):
        """The shard's columns in row order, ``(column, rows)``, a range of rows at a time: the ranges a pass over the
        pack's tensors reads (:data:`~dmipy_sim.replay.compression.SLICE_BYTES` of each tensor's rows)."""
        if self.directory is None:
            for k, rows, _ in self.groups:
                v = self.tensors[k]
                for a in range(0, self.n, rows):
                    yield from columns_of(k, np.asarray(v[self.order[a:a + rows]]), self.band_groups, self.path_groups).items()
            return
        mm = {c: np.load(os.path.join(self.directory, "columns", f"{c}.npy"), mmap_mode="r") for c, _, _ in self.columns}
        for _, rows, cols in self.groups:
            for a in range(0, self.n, rows):
                for c in cols:
                    yield c, mm[c][a:a + rows]

    def save(self, path):
        """The block written under ``path`` (its record and one ``.npy`` file per column), atomically: the directory
        appears complete or not at all. Returns the saved block, read back with :meth:`load`."""
        import pickle
        import shutil
        part = path + ".part"
        shutil.rmtree(part, ignore_errors=True); os.makedirs(os.path.join(part, "columns"))
        fh, dts = {}, {c: np.dtype(dt) for c, dt, _ in self.columns}
        for c, dt, shape in self.columns:
            fh[c] = open(os.path.join(part, "columns", f"{c}.npy"), "wb")
            np.lib.format.write_array_header_1_0(fh[c], {"descr": np.lib.format.dtype_to_descr(dts[c]), "fortran_order": False,
                                                         "shape": (self.n, *shape)})
        for c, x in self.chunks():
            fh[c].write(np.ascontiguousarray(x, dtype=dts[c]).tobytes())
        for f in fh.values():
            f.flush(); os.fsync(f.fileno()); f.close()
        record = {k: v for k, v in self.__dict__.items() if k not in ("tensors", "order", "directory")}
        with open(os.path.join(part, self.RECORD), "wb") as f:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL); f.flush(); os.fsync(f.fileno())
        if os.path.exists(path):
            shutil.rmtree(path)
        os.replace(part, path)
        return PreparedBlock.load(path)

    @classmethod
    def load(cls, path):
        """The block :meth:`save` wrote under ``path``."""
        import pickle
        with open(os.path.join(path, cls.RECORD), "rb") as f:
            record = pickle.load(f)
        return cls(**record, tensors=None, order=None, directory=path)


def shard_identity(pk, grid=None):
    """What every shard of one layout shares: its walk (block and window table), codec, substrate, envelope, the
    grid it is certified on (its own per-voxel certificate's, else the recipe's ``grid``; None when neither) and the
    names of its tensors."""
    from .bank import _codec_signature, _spec_identity, _walk_identity, _envelope_identity
    meta = pk.meta
    return dict(walk_params=_walk_identity(meta["walk_params"]), compression=_codec_signature(meta["compression"]),
                substrate=_spec_identity(meta.get("substrate")), replay_envelope=_envelope_identity(meta.get("replay_envelope")),
                grid=(meta["fidelity"].get("per_voxel") or {}).get("grid") or (None if grid is None else grid.to_meta()),
                names=sorted(k for k in pk.arrays if k not in ("band_block", "voxel_ijk", "voxel_certificate")))


def prepare_block(name, pk, grid=None, layout_grid=None):
    """Shard ``pk`` (named ``name``) prepared for :meth:`Consolidator.append`: certified on the recipe's ``grid`` when
    it carries no per-voxel certificate (:func:`certify_voxels`), its walkers binned by start on ``layout_grid`` (a
    grid's meta: the layout's; None: the shard's own) and ordered by (voxel, pool), its band and mode variances summed
    over the walkers and maximised over its populated voxels, a walker range at a time. Everything here is the
    shard's own: shards prepare independently and in any order, and append in the layout's."""
    from .bank import field_read_record, _MEASURED_CHANNEL_KEYS
    from .compression import SLICE_BYTES, slice_walkers
    from dmipy_replay.replay.compression import read_position_coeffs, walker_rows
    from dmipy_replay.phantom.grid import Grid
    meta = pk.meta; n = int(meta["walk_params"]["n_walkers"])
    ident = shard_identity(pk, grid)
    if ident["grid"] is None:
        raise _no_grid(name)
    meta = certify_voxels(name, pk, grid)
    layout_grid = ident["grid"] if layout_grid is None else layout_grid
    band_groups = band_groups_of(int(meta["compression"]["K"]), meta["compression"].get("container") or [])
    pm = (meta["compression"].get("channels") or {}).get("susc_field_basis")
    path_groups = path_groups_of(int(pm["K"]), pm.get("container")) if pm else ()
    bins = Grid.from_meta(layout_grid)
    ijk, inside = bins.bin(pk.r0)
    if not inside.all():
        raise ValueError(f"{name}: {int((~inside).sum())} walkers start outside the certificate grid")
    pool = np.asarray(pk.arrays[POOL_KEY]).astype(np.int64) if POOL_KEY in pk.arrays else np.zeros(n, np.int64)
    vox = np.ravel_multi_index(ijk.T, bins.shape)
    order = np.lexsort((pool, vox)); vox, pool = vox[order], pool[order]
    # the band and mode variances the read predicates use: the walker sum, the worst populated voxel, read a walker
    # range at a time
    vs = np.flatnonzero(np.r_[True, vox[1:] != vox[:-1]]); counts = np.diff(np.r_[vs, n])
    cell = np.empty(n, np.int64); cell[order] = np.repeat(np.arange(len(vs)), counts)    # each walker's voxel
    band_sq, per_vox = None, None
    step = slice_walkers(pk.arrays)
    for a in range(0, n, step):
        C2 = read_position_coeffs(walker_rows(pk.arrays, a, a + step), dtype=np.float64)[:, 2:, :] ** 2
        band_sq = C2.sum(axis=0) if band_sq is None else band_sq + C2.sum(axis=0)
        o = np.argsort(cell[a:a + step], kind="stable"); c_ = cell[a:a + step][o]
        first = np.flatnonzero(np.r_[True, c_[1:] != c_[:-1]])
        if per_vox is None:
            per_vox = np.zeros((len(vs),) + C2.shape[1:])
        per_vox[c_[first]] += np.add.reduceat(C2[o], first, axis=0)
        del C2
    del cell
    per_vox /= counts[:, None, None]
    band_max = np.max(per_vox[counts >= 100], axis=0) if (counts >= 100).any() else None
    del per_vox
    from dmipy_replay.replay.field import has_susc_path
    path_sq = None
    if has_susc_path(pk.arrays):
        from dmipy_replay.replay.field import susc_path_coeffs
        for a in range(0, n, step):
            s_ = (np.asarray(susc_path_coeffs(walker_rows(pk.arrays, a, a + step), pm)[0], np.float64) ** 2).sum(axis=0)
            path_sq = s_ if path_sq is None else path_sq + s_
    # the scale tables with their block axis; every row's block within the shard
    scale_keys = [k for k in pk.arrays if k.endswith("_band_scale") or k.split("/")[-1] == "susc_path_scale"]
    table = lambda k: (np.asarray(pk.arrays[k])[None] if (k.split("/")[-1] == "susc_path_scale" and np.asarray(pk.arrays[k]).ndim == 2) else np.asarray(pk.arrays[k]))
    scales = {k: table(k) for k in scale_keys}
    nb = int(scales[scale_keys[0]].shape[0]) if scale_keys else 0
    blk = (np.asarray(pk.arrays["band_block"], np.int64) if "band_block" in pk.arrays else np.zeros(n, np.int64))[order]
    # the per-row columns, gathered in the row order a range of rows at a time
    tensors = {k: v for k, v in pk.arrays.items() if not (k in ("voxel_ijk", "voxel_certificate", "band_block") or k in scale_keys)}
    for k, v in tensors.items():
        if np.shape(v)[:1] != (n,):
            raise ValueError(f"{name}: array {k!r} is not walker-leading")
    groups, columns = [], []
    for k, v in tensors.items():
        probe = columns_of(k, np.asarray(v[:1]), band_groups, path_groups)
        groups.append((k, max(1, int(SLICE_BYTES // max(1, np.asarray(v[:1]).nbytes))), list(probe)))
        columns += [(c, x.dtype.str, tuple(x.shape[1:])) for c, x in probe.items()]
    key = vox * 8 + pool
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]); ends = np.r_[starts[1:], n]
    ranges = [(int(vox[s]), int(pool[s]), int(s), int(e)) for s, e in zip(starts, ends)]
    seed = meta["walk_params"]["seed"]; read = field_read_record(meta)
    return PreparedBlock(
        name=name, n=n, ident=ident, meta=meta, layout_grid=layout_grid, band_groups=band_groups, path_groups=path_groups,
        band_sq=band_sq, band_max=band_max, path_sq=path_sq, scales=scales, n_blocks=nb, band_block=blk, ranges=ranges,
        cert=(np.asarray(pk.arrays["voxel_ijk"], np.int64), np.asarray(pk.arrays["voxel_certificate"], np.float64),
              [int(p) for p in meta["fidelity"]["per_voxel"]["pools"]]),
        fid=dict(meta.get("fidelity") or {}),
        chan={c: {k: m.get(k) for k in _MEASURED_CHANNEL_KEYS} for c, m in (meta["compression"].get("channels") or {}).items() if isinstance(m, dict)},
        seed=[int(x) for x in seed] if isinstance(seed, (list, tuple)) else int(seed), read=read,
        prov=dict(id=meta.get("id"), n_walkers=n, shard=name, rounds=(meta.get("provenance") or {}).get("shards"),
                  **({} if read is None else dict(field_read_interval=read["interval"]))),
        groups=groups, columns=columns, tensors=tensors, order=order, directory=None)


class Consolidator:
    """The streaming pass: ``add(name, pack)`` per shard in block order -- or each shard prepared apart
    (:func:`prepare_block`, in any order and any process) and ``append``-ed in block order -- ``checkpoint`` after
    each, ``finish(id)`` at the end (the tables, the index, the manifest); ``resume`` picks a checkpoint up."""

    def __init__(self, out_dir, cap_bytes=1e9, uploader=None, scratch=None, grid=None):
        self.out, self.cap = out_dir, cap_bytes
        self.recipe_grid = grid                           # the recipe's voxel grid: certifies a shard that carries none
        self.up = uploader or Uploader(None, None)
        self.scratch = scratch or os.path.join(out_dir, "scratch")
        os.makedirs(os.path.join(out_dir, "columns"), exist_ok=True); os.makedirs(self.scratch, exist_ok=True)
        self.columns = {}; self.scales = {}; self.certs = []; self.index = []; self.done = []
        self.n_rows = 0; self.n_blocks = 0
        self.meta0 = None; self.ident0 = None; self.fid = []; self.chan = []; self.reads = []; self.seeds = []; self.prov = []
        self.acc = {"band_sq": None, "band_n": 0, "band_max": None, "path_sq": None, "path_n": 0}
        self.grid = None; self.strays = 0; self.band_groups = None; self.path_groups = ()
        self.repaired = None

    # ---- a shard
    def add(self, name, pk):
        """Shard ``pk`` appended: prepared on the layout's grid (:func:`prepare_block`) and appended (:meth:`append`)."""
        self._refuse_other(name, shard_identity(pk, self.recipe_grid))
        self.append(prepare_block(name, pk, self.recipe_grid, None if self.ident0 is None else self.ident0["grid"]))

    def _refuse_other(self, name, ident):
        """Refused when shard ``name``'s :func:`shard_identity` differs from the layout's first, naming the key."""
        from .bank import _agree
        if self.meta0 is None:
            return
        for k, v in ident.items():
            if not _agree(v, self.ident0[k]):
                where = k
                if isinstance(v, dict) and isinstance(self.ident0[k], dict):      # name the key that differs
                    diff = sorted(set(v) ^ set(self.ident0[k]) | {j for j in set(v) & set(self.ident0[k])
                                                                   if not _agree(v[j], self.ident0[k][j])})
                    where = ", ".join(f"{k}.{j}" for j in diff) or k
                raise ValueError(f"shard {name} differs in {where}: {self.ident0[k]!r} vs {v!r}; the shards of one "
                                 "layout are one walk's, on one block and one window table")

    def append(self, block):
        """A :class:`PreparedBlock` appended after the rows so far: refused unless it is the same walk, codec,
        substrate, envelope, grid and tensors as the first; its band and mode sums added, its scale tables and band
        blocks after the earlier ones, its columns and (voxel, pool) row ranges after the earlier rows."""
        from dmipy_replay.phantom.grid import Grid
        name, ident, n = block.name, block.ident, block.n
        # the walk first (its block and window table name the refusal), then the codec, the substrate, the grid (the
        # shard's own, else the recipe's)
        if self.meta0 is not None:
            self._refuse_other(name, ident)
            if _canonical(block.layout_grid) != _canonical(self.ident0["grid"]):
                raise ValueError(f"shard {name} was prepared on the grid {block.layout_grid!r}, the layout's is {self.ident0['grid']!r}")
            if (tuple(block.band_groups), tuple(block.path_groups)) != (tuple(self.band_groups), tuple(self.path_groups)):
                raise ValueError(f"shard {name}: band groups {block.band_groups} and path groups {block.path_groups} differ from "
                                 f"the layout's {self.band_groups} and {self.path_groups}")
        else:
            self.meta0, self.ident0 = json.loads(json.dumps(block.meta)), ident
            self.grid = Grid.from_meta(ident["grid"])
            self.band_groups, self.path_groups = tuple(block.band_groups), tuple(block.path_groups)
        self.acc["band_sq"] = block.band_sq if self.acc["band_sq"] is None else self.acc["band_sq"] + block.band_sq
        self.acc["band_n"] += n
        if block.band_max is not None:
            self.acc["band_max"] = block.band_max if self.acc["band_max"] is None else np.maximum(self.acc["band_max"], block.band_max)
        if block.path_sq is not None:
            self.acc["path_sq"] = block.path_sq if self.acc["path_sq"] is None else self.acc["path_sq"] + block.path_sq
            self.acc["path_n"] += n
        for k, t in block.scales.items():
            self.scales.setdefault(k, []).append(t)
        blk = block.band_block + self.n_blocks
        self.n_blocks += block.n_blocks
        names = sorted(["band_block", *(c for c, _, _ in block.columns)])
        if self.columns and names != sorted(self.columns):
            raise ValueError(f"{name}: columns {names} differ from the first shard's {sorted(self.columns)}")
        for c, dt, shape in [("band_block", np.dtype(np.uint16).str, ()), *block.columns]:
            if c not in self.columns:
                self.columns[c] = Column(c, np.dtype(dt), shape, self.out, self.cap, lambda p, rec: self.up.put(p, rec["file"]))
        self.columns["band_block"].append(blk.astype(np.uint16))
        for c, x in block.chunks():
            self.columns[c].append(x)
        self.index += [(v, p, self.n_rows + s, self.n_rows + e) for v, p, s, e in block.ranges]
        self.n_rows += n
        self.certs.append(block.cert)
        self.fid.append(block.fid)
        self.chan.append(block.chan)
        self.seeds.append(block.seed)
        self.reads.append(block.read)
        self.prov.append(block.prov)
        self.done.append(name)

    # ---- checkpoint and resume
    def checkpoint(self):
        for c in self.columns.values():
            if c.cur is not None:
                c.cur.flush()
        st = dict(done=self.done, n_rows=self.n_rows, n_blocks=self.n_blocks, columns={k: c.state() for k, c in self.columns.items()},
                  column_defs={k: [c.dtype.name, list(c.row_shape)] for k, c in self.columns.items()},
                  index=self.index, seeds=self.seeds, prov=self.prov, fid=self.fid, chan=self.chan, reads=self.reads, meta0=self.meta0, ident0=self.ident0,
                  uploaded=list(self.up.done), band_groups=list(self.band_groups), path_groups=list(self.path_groups),
                  repaired=self.repaired)
        np.savez(os.path.join(self.out, "state.npz"), **{k: (np.asarray(v) if v is not None else np.zeros(0)) for k, v in self.acc.items()},
                 **{f"scale__{k}__{i}": a for k, arrs in self.scales.items() for i, a in enumerate(arrs)},
                 **{f"cert__{i}__ijk": c[0] for i, c in enumerate(self.certs)}, **{f"cert__{i}__cert": c[1] for i, c in enumerate(self.certs)},
                 **{f"cert__{i}__pools": np.asarray(c[2]) for i, c in enumerate(self.certs)})
        tmp = os.path.join(self.out, "state.json.tmp")
        json.dump(st, open(tmp, "w")); os.replace(tmp, os.path.join(self.out, "state.json"))

    def resume(self):
        from dmipy_replay.phantom.grid import Grid
        st = json.load(open(os.path.join(self.out, "state.json"))); z = np.load(os.path.join(self.out, "state.npz"))
        self.done, self.n_rows, self.n_blocks = st["done"], st["n_rows"], st["n_blocks"]
        self.index = [tuple(r) for r in st["index"]]; self.seeds, self.prov, self.fid, self.chan, self.reads = st["seeds"], st["prov"], st["fid"], st["chan"], st["reads"]
        self.meta0, self.ident0 = st["meta0"], st["ident0"]; self.grid = Grid.from_meta(self.ident0["grid"])
        if self.recipe_grid is None:                          # a pass appended to a layout is certified on the layout's grid
            self.recipe_grid = self.grid
        self.band_groups, self.path_groups = tuple(st["band_groups"]), tuple(st["path_groups"]); self.repaired = st.get("repaired")
        on_hub = set(st["uploaded"]) | self.up.hub_files()
        for k, (dt, shape) in st["column_defs"].items():
            c = Column(k, dt, shape, self.out, self.cap, lambda p, rec: self.up.put(p, rec["file"])); c.restore(st["columns"][k], on_hub); self.columns[k] = c
            for rec in c.parts:
                if rec["file"] not in on_hub and os.path.exists(os.path.join(self.out, rec["file"])):
                    self.up.put(os.path.join(self.out, rec["file"]), rec["file"])
        self.up.done = list(on_hub)
        for k in self.acc:
            a = z[k]; self.acc[k] = (int(a) if k.endswith("_n") else (None if a.size == 0 else a))
        for f in z.files:
            if f.startswith("scale__"):
                _, k, i = f.split("__"); self.scales.setdefault(k, {})[int(i)] = z[f]
        self.scales = {k: [v[i] for i in sorted(v)] for k, v in self.scales.items()}
        n_c = len({f.split("__")[1] for f in z.files if f.startswith("cert__")})
        self.certs = [(z[f"cert__{i}__ijk"], z[f"cert__{i}__cert"], [int(p) for p in z[f"cert__{i}__pools"]]) for i in range(n_c)]
        log.info("resumed after %d shards, %d rows", len(self.done), self.n_rows)

    # ---- the end
    def finish(self, id):
        from safetensors.numpy import save_file
        from .bank import charge_field_read, per_voxel_summary
        for c in self.columns.values():
            c.close_part()
        tables = {k: np.concatenate(v) for k, v in self.scales.items()}
        pools = sorted(set().union(*[set(p) for _, _, p in self.certs]))

        def aligned(c_, own):
            out = np.full((c_.shape[0], len(pools), 3), np.nan); out[:, :, 0] = 0.0
            for j_, p_ in enumerate(own):
                out[:, pools.index(p_)] = c_[:, j_]
            return out
        ijk = np.concatenate([c[0] for c in self.certs]); cert = np.concatenate([aligned(c[1], c[2]) for c in self.certs])
        keys_all = np.ravel_multi_index(tuple(ijk.T), tuple(self.grid.shape))
        from .bank import held_voxels
        held = held_voxels(cert)
        rep = self.repaired
        if rep:                                                # a repaired voxel's row is the repair's
            first = sum(len(c_[0]) for c_ in self.certs[:rep["first_shard"]])
            held &= ~((np.arange(len(ijk)) < first) & np.isin(keys_all, np.asarray(rep["voxels"])))
        key = keys_all[held]
        uk, cnt = np.unique(key, return_counts=True)
        if (cnt > 1).any():
            raise ValueError(f"two shards certify voxel {np.unravel_index(uk[cnt > 1][0], self.grid.shape)}")
        order, strays = {}, 0
        for r_, (i_, c_, h_) in enumerate(zip(map(tuple, ijk), cert, held)):
            if i_ not in order or h_:
                if i_ in order and cert[order[i_]][:, 0].sum() > 0:
                    strays += 1
                order[i_] = r_
            elif c_[:, 0].sum() > 0:
                strays += 1
        self.strays = strays
        rows = np.array(sorted(order.values()))
        tables["voxel_ijk"] = ijk[rows].astype(np.int32); c = cert[rows].copy()
        if rep:                                                # a pool the repair did not walk keeps the earlier row's entry;
            first = sum(len(c_[0]) for c_ in self.certs[:rep["first_shard"]])      # a pool it walked counts the union
            keys_kept = keys_all[rows]; old_of = {}
            for r_ in np.flatnonzero((np.arange(len(ijk)) < first) & (cert[:, :, 0].sum(1) > 0) & np.isin(keys_all, np.asarray(rep["voxels"]))):
                old_of[int(keys_all[r_])] = r_
            for j_, kk in enumerate(keys_kept):
                if int(kk) in old_of and rows[j_] >= first:
                    old = cert[old_of[int(kk)]]; empty = c[j_][:, 0] == 0
                    c[j_][empty] = old[empty]; c[j_][~empty, 0] += old[~empty, 0]
        tables["voxel_certificate"] = c.astype(np.float32)
        fid = dict(self.fid[0])
        for k, agg in (("err_max", max), ("floor_max", max), ("noise_floor", max), ("err_surface", max), ("floor_surface", max)):
            vals = [f.get(k) for f in self.fid]
            if all(v is not None for v in vals):
                fid[k] = float(agg(vals))
        if all("within_2x_floor" in f for f in self.fid):
            fid["within_2x_floor"] = bool(all(f["within_2x_floor"] for f in self.fid))
        fid.pop("per_family", None)
        fid["per_voxel"] = per_voxel_summary(self.ident0["grid"], pools, c, shards=len(self.done), recertified=bool(rep))
        comp_meta = json.loads(json.dumps(self.meta0["compression"]))
        for c_, m in (comp_meta.get("channels") or {}).items():
            if isinstance(m, dict):
                for k in ("trace_residual",):
                    vals = [ch.get(c_, {}).get(k) for ch in self.chan]
                    if all(v is not None for v in vals):
                        m[k] = float(max(vals))
        if comp_meta.get("walker_preserving"):
            from .bank import _precision_tiers
            shapes = {k: np.broadcast_to(np.zeros((), c_.dtype), (self.n_rows, *c_.row_shape)) for k, c_ in self.columns.items()}
            shapes.update(tables)
            comp_meta["precision_tiers"] = _precision_tiers(shapes, self.n_rows, float(fid.get("floor_max") or 0.0), False)
        meta = dict(self.meta0)
        meta.update(id=id, compression=comp_meta, fidelity=fid, walk_params=dict(self.meta0["walk_params"], n_walkers=self.n_rows, seed=self.seeds),
                    provenance=dict(self.meta0.get("provenance") or {}, shards=self.prov))
        charge_field_read(meta, self.reads)                    # the blocks' rounds read the field at their own intervals: the layout states the coarsest
        meta["columnar"] = {"band_groups": list(self.band_groups), "path_groups": list(self.path_groups), "sorted_by": ["block", "voxel", "pool"],
                            "band_variance": (self.acc["band_sq"] / self.acc["band_n"]).T.tolist(),
                            "band_variance_max": (None if self.acc["band_max"] is None else self.acc["band_max"].T.tolist()),
                            "path_mode_variance": (None if self.acc["path_sq"] is None else (self.acc["path_sq"] / self.acc["path_n"]).tolist()),
                            "stray_certificate_rows": self.strays,
                            "repair": (None if not rep else dict(voxels=len(rep["voxels"]), shards=len(self.done) - rep["first_shard"]))}
        columns = {k: c_.manifest(self.n_rows) for k, c_ in self.columns.items()}
        for k, v in tables.items():
            rel = f"columns/{k}.safetensors"; f = os.path.join(self.out, rel); save_file({k: v}, f)
            with open(f, "rb") as fh:
                hn = struct.unpack("<Q", fh.read(8))[0]
            columns[k] = {"dtype": v.dtype.name, "shape": list(v.shape), "per_row": False,
                          "parts": [{"file": rel, "rows": None, "data_offset": 8 + hn, "nbytes": int(v.nbytes)}]}
            self.up.put(f, rel)
        index = {"grid": self.ident0["grid"], "n_rows": int(self.n_rows), "band_groups": list(self.band_groups), "path_groups": list(self.path_groups),
                 "rows": [{"ijk": [int(x) for x in np.unravel_index(v, self.grid.shape)], "pool": p, "start": s, "end": e} for v, p, s, e in self.index]}
        from dmipy_replay.replay.replay import declared_susceptibility_field
        declared_susceptibility_field(meta, has_field=bool(self.path_groups))   # the layout holds the equivalence
        manifest = {"meta": meta, "columns": columns}
        json.dump(index, open(os.path.join(self.out, "index.json"), "w"))
        json.dump(manifest, open(os.path.join(self.out, "manifest.json"), "w"))
        self.up.put(os.path.join(self.out, "index.json"), "index.json"); self.up.put(os.path.join(self.out, "manifest.json"), "manifest.json")
        self.up.finish()
        if self.up.failed:
            raise IOError(f"upload of {self.up.failed} failed; resume")
        log.info("done: %d rows, %d index rows, %d columns, %d parts", self.n_rows, len(self.index), len(columns), sum(len(c_["parts"]) for c_ in columns.values()))
        return manifest, index


def _shard_names(b, shards, pass_):
    """The shard files of block ``b``, in the order they are merged: its pass file when the pass wrote one, else the
    whole block's single file; else the pass's shards -- the round shards (``block-NNNN.pK.<pool>.r<r>.rpk``, sorted
    by (pool, round)) and the dealt shards -- else the same of the whole block. A dealt shard is the ``.rpk`` files
    named in the ``files`` of each ``block-NNNN[.pK].s<idx>.walk.json`` summary present, sorted by (idx, pool, round)
    (``block-NNNN[.pK].s<idx>.<pool>.r<r>.rpk``); a shard without a summary (a walk that failed) contributes
    nothing. Empty when the block has none."""
    import re
    for P in ({"pass": pass_}, {}):
        one = shard_name(b, P) + ".rpk"
        if one in shards:
            return [one]
        stem = re.escape(shard_name(b, P))
        rounds = re.compile(stem + r"\.(?P<pool>[^.]+)\.r(?P<r>\d+)\.rpk")
        summary = re.compile(stem + r"\.s(?P<idx>\d+)\.walk\.json")
        member = re.compile(stem + r"\.s(?P<idx>\d+)\.(?P<pool>[^.]+)\.r(?P<r>\d+)\.rpk")
        by_round = [((m["pool"], int(m["r"])), n) for n in shards.names if (m := rounds.fullmatch(n))]
        by_deal = []
        for sname in shards.summaries:
            if not summary.fullmatch(sname):
                continue
            for f in shards.read_json(sname)["files"]:
                n = os.path.basename(f)
                if n.endswith(".rpk") and (m := member.fullmatch(n)):
                    by_deal.append(((int(m["idx"]), m["pool"], int(m["r"])), n))
        found = [n for _, n in sorted(by_round)] + [n for _, n in sorted(set(by_deal))]
        if found:
            return found
    return []


def _block_name(b, names, pass_):
    """The name a block's rows are recorded under: its single shard's, else the block's own (the pass file's when
    the round shards are the pass's)."""
    if len(names) == 1:
        return names[0]
    return shard_name(b, {"pass": pass_} if f".p{pass_}." in names[0] else {}) + ".rpk"


def _fetch_block(b, names, sh, scratch, pass_, read_rpk):
    """``(pack, paths)`` of block ``b``: its one shard, or its shards merged into the block's pack (held in memory;
    the fetched files are the caller's to delete)."""
    paths = [sh.fetch(n, scratch) for n in names]
    if len(paths) == 1:
        return read_rpk(paths[0]), paths
    from .bank import merge_packs
    return merge_packs(paths, id=_block_name(b, names, pass_), overlap="recertify"), paths


def consolidate(shards, out_dir, *, blocks, id, pass_=1, repo=None, upload=None, cap_bytes=1e9, resume=False, grid=None):
    """The shards of ``blocks`` (their indices; ``block-NNNN.p<pass>.rpk`` where it exists, else the whole block,
    else the block's round shards merged into the block's pack) from ``shards`` (a directory, or a prefix of
    ``repo``) into the layout under ``out_dir``, published under ``upload`` on ``repo`` when given. ``grid`` is the
    recipe's voxel grid (:func:`manifest_grid`): the per-voxel certificate of every shard that
    carries none is measured on it. Returns ``(manifest, index)``."""
    from .replay import read_rpk
    sh = Shards(shards, repo); up = Uploader(repo, upload)
    con = Consolidator(out_dir, cap_bytes, up, grid=grid)
    if resume:
        con.resume()
    names_of = {b: _shard_names(b, sh, pass_) for b in blocks}
    missing = [b for b in blocks if not names_of[b]]
    if missing:
        raise FileNotFoundError(f"{len(missing)} blocks have no shard (e.g. {missing[:5]})")
    t0 = time.time()
    for i, b in enumerate(blocks):
        names = names_of[b]; name = _block_name(b, names, pass_)
        if name in con.done:
            continue
        t1 = time.time(); pk, paths = _fetch_block(b, names, sh, con.scratch, pass_, read_rpk)
        con.add(name, pk); del pk
        if sh.remote:
            for path in paths:
                os.remove(path)
        con.checkpoint()
        log.info("block %4d %s: %d rows total, %.1f s (%.0f s elapsed, %d/%d shards)", b, name, con.n_rows, time.time() - t1, time.time() - t0, i + 1, len(blocks))
        if up.failed:
            raise IOError(f"upload of {up.failed} failed; resume")
    return con.finish(id)


def append(shards_dir, out_dir, *, id, pass_, repo=None, upload=None, cap_bytes=1e9):
    """A later pass of some blocks (``block-NNNN.p<pass>.rpk`` in ``shards_dir``, or the block's round shards
    ``block-NNNN.p<pass>.<pool>.r<r>.rpk``, merged) appended to the layout under
    ``out_dir`` (its checkpoint resumed): the union weights on both sides, the old rows patched in their weights
    part, the repaired pools' certificate rows the union's. Returns ``(manifest, index)``."""
    from .replay import read_rpk
    up = Uploader(repo, upload); con = Consolidator(out_dir, cap_bytes, up); con.resume()
    n_before = len(con.done); boundary = con.n_rows
    old_ranges = {}
    for v, p, s, e in con.index:
        old_ranges.setdefault((v, p), []).append((s, e))
    sh = Shards(shards_dir)
    blocks = sorted({b for b, p in map(parse_shard, sh.names) if p == pass_})
    union = {}; repaired = set()
    for b in blocks:
        names = _shard_names(b, sh, pass_); name = _block_name(b, names, pass_)
        pk, _ = _fetch_block(b, names, sh, con.scratch, pass_, read_rpk); n = pk.n_walkers
        ijk, _ = con.grid.bin(pk.r0); vox = np.ravel_multi_index(ijk.T, con.grid.shape)
        pool = np.asarray(pk.arrays[POOL_KEY]).astype(np.int64) if POOL_KEY in pk.arrays else np.zeros(n, np.int64)
        w = np.asarray(pk.arrays["spin_weights"], np.float64).copy()
        for key in set(zip(vox.tolist(), pool.tolist())):
            sel = (vox == key[0]) & (pool == key[1]); n_new = int(sel.sum()); n_old = sum(e - s for s, e in old_ranges.get(key, []))
            f_new = float(w[sel].sum())
            union[key] = dict(n_old=n_old, n_new=n_new, f_new=f_new)
            if n_old:                                          # the union's weight is set once the old fraction is read (below)
                continue
            w[sel] = f_new / n_new
        pk.arrays["spin_weights"] = w.astype(np.float32)
        repaired |= {int(k) for k in np.unique(vox)}
        con._pending = getattr(con, "_pending", []) + [(name, pk)]
    # the old fractions, read from the weights parts once, then every new row's weight set to the union's
    man_old = _manifest_of(con, repo, upload)
    old_w = _read_weights(con, man_old, [(s, e) for k in union for (s, e) in old_ranges.get(k, [])], repo, upload)
    for key, u in union.items():
        if u["n_old"]:
            u["f_old"] = float(sum(old_w[(s, e)].sum() for s, e in old_ranges[key]))
            u["f_union"] = 0.5 * (u["f_old"] + u["f_new"])       # the mean of the two censuses, as union_weights
            u["w_union"] = u["f_union"] / (u["n_old"] + u["n_new"])
    for name, pk in con._pending:
        ijk, _ = con.grid.bin(pk.r0); vox = np.ravel_multi_index(ijk.T, con.grid.shape)
        pool = np.asarray(pk.arrays[POOL_KEY]).astype(np.int64) if POOL_KEY in pk.arrays else np.zeros(pk.n_walkers, np.int64)
        w = np.asarray(pk.arrays["spin_weights"], np.float64).copy()
        for key, u in union.items():
            if u["n_old"]:
                w[(vox == key[0]) & (pool == key[1])] = u["w_union"]
        pk.arrays["spin_weights"] = w.astype(np.float32)
        con.add(name, pk); con.checkpoint()
        log.info("append %s: %d rows (%d total)", name, pk.n_walkers, con.n_rows)
    con.repaired = dict(first_shard=n_before, voxels=sorted(repaired))
    manifest, index = con.finish(id)
    # the old rows' weights, patched in the parts they sit in
    parts = manifest["columns"]["spin_weights"]["parts"]; touched = {}
    for key, u in union.items():
        if not u["n_old"]:
            continue
        for s, e in old_ranges[key]:
            for p in parts:
                ps, pe = p["rows"]; lo, hi = max(s, ps), min(e, pe)
                if lo < hi:
                    touched.setdefault(p["file"], []).append((lo - ps, hi - ps, u["w_union"]))
    for rel, edits in touched.items():
        path = _local_part(con, rel, repo, upload)
        off = next(p["data_offset"] for p in parts if p["file"] == rel)
        with open(path, "r+b") as fh:
            for lo, hi, w_union in edits:
                fh.seek(off + 4 * lo); fh.write(np.full(hi - lo, w_union, np.float32).tobytes())
        if upload:
            from huggingface_hub import HfApi
            HfApi().upload_file(path_or_fileobj=path, path_in_repo=f"{upload}/{rel}", repo_id=repo, repo_type="dataset",
                                commit_message=f"append: the union weights of {len(edits)} old row ranges in {rel}")
        log.info("patched %d old row ranges in %s", len(edits), rel)
    return manifest, index


def _manifest_of(con, repo, upload):
    p = os.path.join(con.out, "manifest.json")
    if os.path.exists(p):
        return json.load(open(p))
    import requests
    from huggingface_hub import hf_hub_url, get_token
    tok = get_token()
    return json.loads(requests.get(hf_hub_url(repo, f"{upload}/manifest.json", repo_type="dataset"), headers={"Authorization": f"Bearer {tok}"} if tok else {}).text)


def _local_part(con, rel, repo, upload):
    p = os.path.join(con.out, rel)
    if os.path.exists(p):
        return p
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, f"{upload}/{rel}", repo_type="dataset", local_dir=os.path.join(con.out, "patch"))


def _read_weights(con, manifest, ranges, repo, upload):
    parts = manifest["columns"]["spin_weights"]["parts"]; out = {}
    for s, e in ranges:
        vals = []
        for p in parts:
            ps, pe = p["rows"]; lo, hi = max(s, ps), min(e, pe)
            if lo < hi:
                with open(_local_part(con, p["file"], repo, upload), "rb") as fh:
                    fh.seek(p["data_offset"] + 4 * (lo - ps)); vals.append(np.frombuffer(fh.read(4 * (hi - lo)), np.float32))
        out[(s, e)] = np.concatenate(vals) if vals else np.zeros(0, np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="the dataset repository (owner/name) the shards and the layout live on")
    ap.add_argument("--shards", default=None, help="a directory of shards, or a prefix of the repository (e.g. blocks/disco)")
    ap.add_argument("--out", required=True); ap.add_argument("--upload", default=None, help="the prefix the layout is published under")
    ap.add_argument("--blocks", default=None, help="'a-b' or a comma list of block indices (default: 0..n-blocks-1)"); ap.add_argument("--n-blocks", type=int, default=None)
    ap.add_argument("--pass", dest="pass_", type=int, default=1); ap.add_argument("--part-bytes", type=float, default=1e9)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--id", required=True)
    ap.add_argument("--manifest", default=None, help="the recipe's manifest.json (its voxel grid certifies the shards that carry none); "
                    "default with --repo: the repository's")
    ap.add_argument("--append", default=None, help="a directory of a later pass's shards to append to the layout under --out")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(a.out, "consolidate.log"))])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if a.append:
        append(a.append, a.out, id=a.id, pass_=a.pass_, repo=a.repo, upload=a.upload, cap_bytes=a.part_bytes); return
    grid = None
    if a.manifest or a.repo:
        path = a.manifest
        if path is None:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(a.repo, "manifest.json", repo_type="dataset")
        grid = manifest_grid(json.load(open(path)))
    if a.blocks is None:
        blocks = list(range(a.n_blocks))
    elif "-" in a.blocks:
        lo, hi = a.blocks.split("-"); blocks = list(range(int(lo), int(hi) + 1))
    else:
        blocks = [int(x) for x in a.blocks.split(",")]
    consolidate(a.shards, a.out, blocks=blocks, id=a.id, pass_=a.pass_, repo=a.repo, upload=a.upload, cap_bytes=a.part_bytes, resume=a.resume,
                grid=grid)


if __name__ == "__main__":
    main()
