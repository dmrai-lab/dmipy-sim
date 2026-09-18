"""A replay pack in columns, read by reference (dmipy-sim#290): rows ordered by (block, voxel, pool), one file per
stored tensor in parts, the position bands and the field modes in groups, the pack's meta once in ``manifest.json``
with every column's byte layout, and ``index.json`` with the row range of every (voxel, pool). A consumer opens the
layout where it is -- a directory, or ``hf://owner/name/prefix`` on the Hugging Face Hub, read by HTTP range -- and
gets :class:`~dmipy_sim.replay.replay.ReplayPack` VIEWS of the rows and bands an acquisition needs; nothing is
downloaded whole, and every byte fetched is one the replay uses.

    pack = open_columnar("hf://SubstrateCommons/disco-replay/disco")      # or ReplayPack.open(...)
    plan = pack.plan(seq, tissue=..., scanner=...)                          # bands, modes, tiers, bytes: before any transfer
    view = pack.view(K=32, voxels=[(20, 20, 20)])                           # one voxel's rows at 32 bands: an ordinary ReplayPack
    S, floor, plan = pack.image([seq], settings=[(None, None), (tissue, 3.0)])   # every setting's volume from one pass

The layout is written by :mod:`dmipy_sim.fill.consolidate`.
"""
from __future__ import annotations
import functools
import json
import os
import threading
import time

import numpy as np

POOL_KEY = "comp_static"


def _grid_of(meta):
    from ..phantom.grid import Grid
    return Grid.from_meta(meta["fidelity"]["per_voxel"]["grid"])


class Source:
    """Bytes of a column file by range: a directory, or the Hub through HTTP range requests -- one keep-alive
    session per thread, the CDN location of every file resolved once (afresh after a failed read), the ranges
    of a view fetched concurrently (``workers`` requests in flight), a failed range asked for again."""

    def __init__(self, uri, workers=8):
        self.uri = uri; self.bytes_read = 0; self.requests = 0; self.workers = workers
        self._tl = threading.local(); self._resolved = {}; self._lock = threading.Lock()
        self.remote = uri.startswith("hf://")
        if self.remote:
            from huggingface_hub import hf_hub_url, get_token
            owner, name, prefix = uri[5:].split("/", 2)                       # hf://owner/name/prefix
            self.repo, self.prefix = f"{owner}/{name}", prefix
            self.url = lambda rel: hf_hub_url(self.repo, f"{self.prefix}/{rel}", repo_type="dataset")
            tok = get_token(); self.headers = {"Authorization": f"Bearer {tok}"} if tok else {}

    def _session(self):
        import requests
        s = getattr(self._tl, "session", None)
        if s is None:
            s = self._tl.session = requests.Session()
        return s

    def _location(self, rel):
        with self._lock:
            loc = self._resolved.get(rel)
        if loc is None:
            r = self._session().head(self.url(rel), headers=self.headers, allow_redirects=True, timeout=60)
            loc = r.url
            with self._lock:
                self._resolved[rel] = loc
        return loc

    def read(self, rel, start, length):
        with self._lock:
            self.requests += 1; self.bytes_read += length
        if not self.remote:
            with open(os.path.join(self.uri, rel), "rb") as fh:
                fh.seek(start); return fh.read(length)
        err = None
        for attempt in range(6):
            try:
                loc = self._location(rel)
                r = self._session().get(loc, headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=120)
                if r.status_code == 206 and len(r.content) == length:
                    return r.content
                err = f"status {r.status_code}, {len(r.content)} of {length} bytes"
            except Exception as e:
                err = repr(e)
            with self._lock:
                self._resolved.pop(rel, None)
            time.sleep(min(60, 2 * 2 ** attempt))
        raise IOError(f"range read of {rel} [{start}, +{length}) failed after 6 attempts: {err}")

    def read_many(self, jobs):
        """``[(rel, start, length), ...]`` fetched concurrently, in order."""
        from concurrent.futures import ThreadPoolExecutor
        if not self.remote or len(jobs) == 1:
            return [self.read(*j) for j in jobs]
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(lambda j: self.read(*j), jobs))

    def text(self, rel):
        if not self.remote:
            return open(os.path.join(self.uri, rel)).read()
        r = self._session().get(self.url(rel), headers=self.headers, allow_redirects=True, timeout=120); r.raise_for_status(); return r.text


class ColumnarPack:
    """A pack in the columnar layout, open by reference. ``meta`` is the merged pack's header, ``columns`` the byte
    layout of every column (its parts), ``index`` the row range of every (voxel, pool), ``grid`` the certificate's
    grid, ``floor`` the certificate's median per-voxel floor."""

    def __init__(self, uri, workers=8):
        self.uri = uri; self.src = Source(uri, workers=workers)
        m = json.loads(self.src.text("manifest.json")); self.meta, self.columns = m["meta"], m["columns"]
        self.index = json.loads(self.src.text("index.json"))
        self.n_rows = int(self.index["n_rows"]); self.band_groups = list(self.index["band_groups"]); self.path_groups = list(self.index.get("path_groups", []))
        self.grid = _grid_of(self.meta)
        self.floor = float(self.meta["fidelity"]["per_voxel"]["floor_median"])
        self.K = int(self.meta["compression"]["K"])

    # ---- the plan: what an acquisition needs, before any byte moves
    def bands_for(self, seq, tol=0.25):
        """``(K', error)``: the fewest stored band groups whose truncation costs less than ``tol`` x the pack's median
        floor on every measurement -- the phase variance the dropped bands carry under the waveform's projection
        on the pack grid, ``(gamma dt)^2 sum_{k>K'} sum_d var_{k,d} W_{m,k,d}^2``, halved (a dropped independent
        phase under-attenuates the signal by that much), from the manifest's per-band variance of the worst
        populated voxel. The walk is not read."""
        from .replay import _compile_effective, GAMMA
        from ._replay_kernel import effective_gradient
        K = self.K; n_t = int(self.meta["walk_params"]["n_t"]); dt = float(self.meta["walk_params"]["dt_traj"])
        G = np.asarray(seq.G_eff, np.float64)
        Geff = effective_gradient(G, float(seq.dt), n_t, dt)
        W = (_compile_effective(Geff, dt, K, n_t) / (GAMMA * dt)).T.reshape(-1, K + 2, 3)[:, 2:, :]
        var = np.asarray(self.meta["columnar"].get("band_variance_max") or self.meta["columnar"]["band_variance"], np.float64).T   # (K, 3)
        per_band = (GAMMA * dt) ** 2 * (W ** 2 * var[None]).sum(axis=2)
        tail = np.cumsum(per_band[:, ::-1], axis=1)[:, ::-1]
        for Kp in self.band_groups:
            err = 0.5 * (tail[:, Kp] if Kp < K else np.zeros(per_band.shape[0]))
            if err.max() <= tol * self.floor:
                return int(Kp), float(err.max())
        return K, 0.0

    def modes_for(self, seq, scanner, tissue, tol=0.25):
        """``(M, error)``: the fewest stored field modes for a replay at the scanner's field with the tissue's chi (0
        without either): the dropped modes' phase variance ``(gamma dt_f B0 chi)^2 sum_{k>M} gate_hat_k^2 sum_ch
        var_{ch,k}``, halved, against ``tol`` x the floor; the channel sum is the worst case over field directions."""
        from .replay import GAMMA, _field_strength, _path_grid
        from ._replay_kernel import field_gate
        from scipy.fft import dct
        B0 = _field_strength(scanner); chi = None if tissue is None else tissue.chi_iso
        if B0 is None or chi is None or not self.path_groups:
            return 0, 0.0
        pm = self.meta["compression"]["channels"]["susceptibility_path"]
        n_t = int(self.meta["walk_params"]["n_t"]); dt = float(self.meta["walk_params"]["dt_traj"])
        n_tf, dt_f = _path_grid(pm, n_t, dt)
        g = dct(field_gate(seq, n_tf, dt_f), type=2, norm="ortho")[:self.path_groups[-1]]
        var = np.asarray(self.meta["columnar"]["path_mode_variance"], np.float64).sum(axis=0)
        per_mode = (GAMMA * dt_f * B0 * (abs(chi) + abs(tissue.chi_aniso or 0.0))) ** 2 * g ** 2 * var[:len(g)]
        tail = np.cumsum(per_mode[::-1])[::-1]
        for M in self.path_groups:
            err = 0.5 * (tail[M] if M < len(per_mode) else 0.0)
            if err <= tol * self.floor:
                return int(M), float(err)
        return int(self.path_groups[-1]), 0.0

    def plan(self, seq, *, tissue=None, scanner=None, tol=0.25, voxels=None):
        """What a replay of ``seq`` reads: bands, field modes, tiers, rows and bytes -- before any byte moves."""
        K, eK = self.bands_for(seq, tol); M, eM = self.modes_for(seq, scanner, tissue, tol)
        contact = tissue is not None and tissue.rho is not None
        relax = tissue is not None and (tissue.T2 is not None or tissue.T1 is not None)
        ranges = self.rows_of(voxels); rows = sum(e - s for s, e in ranges)
        names = self._names(K, M, contact, relax)
        per_row = sum(int(np.prod(self.columns[n]["shape"][1:], dtype=np.int64)) * np.dtype(self.columns[n]["dtype"]).itemsize for n in names)
        return dict(K=K, band_error=eK, modes=M, mode_error=eM, contact=contact, relaxation=relax, rows=rows,
                    bytes=per_row * rows, bytes_per_row=per_row, columns=names)

    def _names(self, K, M, contact, relaxation):
        """The per-row columns a replay at ``K`` bands, ``M`` field modes and the tiers asked for reads: every
        column of the manifest except the band groups beyond ``K``, the mode groups beyond ``M``, and the contact
        channel unless asked for."""
        n_b = len([g for g in self.band_groups if g <= K]); n_m = len([g for g in self.path_groups if g <= M]) if M else 0
        names = []
        for n, c in self.columns.items():
            if not c["per_row"]:
                continue
            if n[:4] == "pos_" and "_b" in n[5:]:
                if int(n.rsplit("_b", 1)[1]) >= n_b:
                    continue
            elif n.startswith("susc_path_m"):
                if int(n[len("susc_path_m"):]) >= n_m:
                    continue
            elif n.startswith("blt_") and not contact:
                continue
            names.append(n)
        return names

    # ---- rows
    def rows_of(self, voxels=None, pools=None):
        """Contiguous row ranges ``(start, end)`` of the voxels (ijk triples; all when None) and pools asked for,
        adjacent ranges merged; a (voxel, pool) walked in two passes has two."""
        rs = []
        want = None if voxels is None else {tuple(int(x) for x in v) for v in voxels}
        for r in self.index["rows"]:
            if (want is None or tuple(r["ijk"]) in want) and (pools is None or r["pool"] in pools):
                if rs and rs[-1][1] == r["start"]:
                    rs[-1] = (rs[-1][0], r["end"])
                else:
                    rs.append((r["start"], r["end"]))
        return rs

    def _parts(self, c):
        if "parts" in c:
            return c["parts"]
        return [{"file": c["file"], "rows": ([0, c["shape"][0]] if c["per_row"] else None), "data_offset": c["data_offset"], "nbytes": c["nbytes"]}]

    def _jobs(self, name, ranges):
        c = self.columns[name]; dt = np.dtype(c["dtype"]); shape = c["shape"]; parts = self._parts(c)
        if not c["per_row"]:
            p = parts[0]
            return [(p["file"], p["data_offset"], p["nbytes"])], dt, shape, [None]
        rowbytes = int(np.prod(shape[1:], dtype=np.int64)) * dt.itemsize
        jobs, rs = [], []
        for s, e in ranges:
            for p in parts:
                ps, pe = p["rows"]; lo, hi = max(s, ps), min(e, pe)
                if lo < hi:
                    jobs.append((p["file"], p["data_offset"] + (lo - ps) * rowbytes, (hi - lo) * rowbytes)); rs.append((lo, hi))
        return jobs, dt, shape, rs

    def columns_over(self, names, ranges):
        """The columns as arrays over ``ranges``, every byte range of every column fetched concurrently."""
        plan = {n: self._jobs(n, ranges) for n in names}
        jobs = [j for n in names for j in plan[n][0]]
        raws = iter(self.src.read_many(jobs)); out = {}
        for n in names:
            js, dt, shape, rs = plan[n]
            parts = []
            for j, r in zip(js, rs):
                raw = next(raws)
                parts.append(np.frombuffer(raw, dt).reshape(shape if r is None else (r[1] - r[0],) + tuple(shape[1:])))
            out[n] = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
        return out

    # ---- views
    def view(self, *, K=None, modes=0, voxels=None, pools=None, contact=False, relaxation=True, ranges=None):
        """A :class:`~dmipy_sim.replay.replay.ReplayPack` of the rows and bands asked for: ``K`` bands (rounded up to a
        stored group; the pack's K when None), ``modes`` field modes (0: the field tier left out), the contact tier
        with ``contact``."""
        from .replay import ReplayPack
        ranges = self.rows_of(voxels, pools) if ranges is None else ranges
        K = self.K if K is None else int(K)
        K_read = next((g for g in self.band_groups if g >= K), self.band_groups[-1])
        M_read = 0 if not modes else next((g for g in self.path_groups if g >= modes), self.path_groups[-1])
        tables = [n for n, c in self.columns.items() if not c["per_row"] and (contact or not n.startswith("blt_")) and (M_read or n != "susc_path_scale")]
        names = self._names(K_read, M_read, contact, relaxation) + tables
        arrays = self.columns_over(names, ranges)
        if "pos_band_scale" in arrays:
            arrays["pos_band_scale"] = np.ascontiguousarray(arrays["pos_band_scale"][..., :K_read])
        else:                                                              # the float coefficients: the ends and the groups as one array
            n_b = len([g for g in self.band_groups if g <= K_read])
            for a in "xyz":
                arrays[f"pos_{a}"] = np.concatenate([arrays.pop(f"pos_{a}_ends")] + [arrays.pop(f"pos_{a}_b{i}") for i in range(n_b)], axis=1)
        meta = json.loads(json.dumps(self.meta)); meta["compression"]["K"] = int(K_read)
        if M_read:
            arrays["susc_path_dct"] = np.concatenate([arrays.pop(f"susc_path_m{i}") for i in range(len([g for g in self.path_groups if g <= M_read]))], axis=2)
            arrays["susc_path_scale"] = np.ascontiguousarray(arrays["susc_path_scale"][..., :M_read])
            meta["compression"]["channels"]["susceptibility_path"]["K"] = int(M_read)
        else:
            meta["compression"]["channels"].pop("susceptibility_path", None); meta["compression"]["channels"].pop("susceptibility_grid", None)
        if not contact:
            meta["compression"]["channels"].pop("boundary_local_time", None)
        return ReplayPack(arrays, meta)

    def iter_views(self, chunk_rows=2_000_000, **kw):
        """The pack in row groups of about ``chunk_rows`` on (voxel, pool) boundaries, each a view (``**kw`` as
        :meth:`view`), the next group's bytes fetched while the current one is contracted; a failed fetch reaches
        the consumer."""
        import queue
        groups, cur = [], []
        for r in self.index["rows"]:
            if cur and cur[-1][1] == r["start"]:
                cur[-1] = (cur[-1][0], r["end"])
            else:
                cur.append((r["start"], r["end"]))
            if sum(e - s for s, e in cur) >= chunk_rows:
                groups.append(cur); cur = []
        if cur:
            groups.append(cur)
        q = queue.Queue(maxsize=1)

        def fetch():
            try:
                for g in groups:
                    q.put(self.view(ranges=g, **kw))
                q.put(None)
            except BaseException as e:
                q.put(e)
        threading.Thread(target=fetch, daemon=True).start()
        while True:
            v = q.get()
            if v is None:
                return
            if isinstance(v, BaseException):
                raise v
            yield v

    # ---- the image loop
    def image(self, seq, *, tissue=None, scanner=None, settings=None, tol=0.25, chunk_rows=2_000_000, progress=None):
        """``(S, floor, plan)``: the signal of every voxel of the grid under ``seq`` (one sequence or a list, their
        measurements side by side; NaN where the pack has none), the certificate's floor beside it (the worst pool's),
        and the plan with what was read. One pass over the rows: the host fetches, dequantises and forms each
        walker's phase (:meth:`ReplayPack.walker_phases`); the device forms the exponential and the sum over each
        voxel's rows in double precision. With ``settings`` (a list of ``(tissue, scanner)``) every setting's volume
        comes from the same pass -- the settings differ only in what is applied to the rows -- and ``S`` gains a
        leading axis. ``progress(rows, bytes, seconds)`` is called after every row group."""
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        seqs = list(seq) if isinstance(seq, (list, tuple)) else [seq]
        multi = settings is not None; settings = list(settings) if multi else [(tissue, scanner)]
        t0 = time.time(); plans = [self.plan(s, tissue=t_, scanner=B_, tol=tol) for s in seqs for t_, B_ in settings]
        plan = max(plans, key=lambda p: (p["K"], p["modes"]))
        plan = dict(plan, K=max(p["K"] for p in plans), modes=max(p["modes"] for p in plans), contact=any(p["contact"] for p in plans), settings=len(settings))
        self.src.bytes_read = self.src.requests = 0
        n_meas = sum(int(np.asarray(s.G_eff).shape[0]) for s in seqs); n_vox = int(np.prod(self.grid.shape))
        num = np.zeros((len(settings), n_vox, n_meas), complex); den = np.zeros(n_vox); rows = 0
        ROWS, SEGS = 1 << 16, 256                                            # padded to multiples: a few compiled shapes

        @functools.partial(jax.jit, static_argnums=3)
        def voxel_sums(phi, ew, seg, n_seg):
            return jax.ops.segment_sum(jnp.exp(1j * phi) * ew[:, None], seg, num_segments=n_seg)
        for pk in self.iter_views(chunk_rows=chunk_rows, K=plan["K"], modes=plan["modes"], contact=plan["contact"]):
            n = pk.n_walkers
            ijk, _ = self.grid.bin(pk.r0); v = np.ravel_multi_index(ijk.T, self.grid.shape)
            starts = np.flatnonzero(np.r_[True, v[1:] != v[:-1]]); n_seg = len(starts)   # runs of one voxel
            seg = np.repeat(np.arange(n_seg), np.diff(np.r_[starts, n]))
            n_pad = -(-n // ROWS) * ROWS; s_pad = -(-(n_seg + 1) // SEGS) * SEGS  # one spare segment takes the padding rows
            seg_p = np.full(n_pad, n_seg, np.int32); seg_p[:n] = seg; w = None
            for si, (t_, B_) in enumerate(settings):
                sums = []
                for s in seqs:                                                # each sequence's own relaxation and contact weights
                    w, ew, p = pk.walker_phases(s, tissue=t_, scanner=B_)
                    phi = np.zeros((n_pad, p.shape[1])); phi[:n] = p; ew_p = np.zeros(n_pad); ew_p[:n] = ew
                    sums.append(np.asarray(voxel_sums(jnp.asarray(phi), jnp.asarray(ew_p), jnp.asarray(seg_p), s_pad))[:n_seg])
                np.add.at(num[si], v[starts], np.concatenate(sums, axis=1))
            np.add.at(den, v[starts], np.add.reduceat(np.asarray(w), starts))   # a voxel may sit in two row groups
            rows += n
            if progress:
                progress(rows, self.src.bytes_read, time.time() - t0)
        S = np.full((len(settings), n_vox, n_meas), np.nan); m = den > 0; S[:, m] = np.abs(num[:, m] / den[m][None, :, None])
        floor = np.full(n_vox, np.nan); cert = self.columns_over(["voxel_ijk", "voxel_certificate"], [(0, 0)])
        vi = np.ravel_multi_index(np.asarray(cert["voxel_ijk"]).T, self.grid.shape)
        fl = np.asarray(cert["voxel_certificate"])[:, :, 1]; known = np.isfinite(fl).any(axis=1)
        floor[vi[known]] = np.nanmax(fl[known], axis=1)
        plan.update(bytes_read=self.src.bytes_read, requests=self.src.requests, seconds=time.time() - t0, rows=rows)
        S = S.reshape((len(settings),) + tuple(self.grid.shape) + (n_meas,))
        return (S if multi else S[0]), floor.reshape(self.grid.shape), plan


def open_columnar(uri, workers=8):
    """The columnar layout at ``uri`` (a directory, or ``hf://owner/name/prefix``) as a :class:`ColumnarPack`."""
    return ColumnarPack(uri, workers=workers)
