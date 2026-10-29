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
import re
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
        from .replay import _refuse_predating
        _refuse_predating(self.meta, uri, tensors=list(self.columns))
        self.index = json.loads(self.src.text("index.json"))
        self.n_rows = int(self.index["n_rows"]); self.band_groups = list(self.index["band_groups"]); self.path_groups = list(self.index.get("path_groups", []))
        self.grid = _grid_of(self.meta)
        self.floor = float(self.meta["fidelity"]["per_voxel"]["floor_median"])
        self.K = int(self.meta["compression"]["K"])
        self.susceptibility_field

    @property
    def dt(self):
        """The block length ``h`` (s) of the walk the layout stores (``walk_params.block``)."""
        return float(self.meta["walk_params"]["block"]["h"])

    @property
    def n_t(self):
        """The block ends of the walk, ``T_max / h + 1``: the grid a waveform built for the layout is sampled on."""
        return int(round(float(self.meta["walk_params"]["T_max"]) / self.dt)) + 1

    @property
    def T_window(self):
        """One storage window's duration (s): the interval the bands are on."""
        return float(self.meta["walk_params"]["windows"]["T"])

    @property
    def susceptibility_field(self):
        """The layout's susceptibility field, ``"present"`` or ``"absent"``, from its embedded spec's pools with its
        stored field modes as the field channel (:func:`~dmipy_sim.replay.replay.declared_susceptibility_field`);
        a layout without an embedded spec, or one that breaks the equivalence, is refused when it is opened."""
        from .replay import declared_susceptibility_field
        return declared_susceptibility_field(self.meta, has_field=bool(self.path_groups))

    # ---- the plan: what an acquisition needs, before any byte moves
    @property
    def truncation_budget(self):
        """What a waveform may cost this layout in band truncation (:func:`~dmipy_sim.replay.replay.truncation_budget`):
        its ``fidelity.target_floor`` less its other certified terms; a layout that states no cap is held to its median
        per-voxel floor."""
        from .replay import truncation_budget
        return truncation_budget(self.meta, floor=self.floor, floor_source="fidelity.per_voxel.floor_median")

    def bands_for(self, seq, tol=0.25):
        """``(K', error)``: the fewest stored band groups whose truncation costs less than ``tol`` x the pack's median
        floor on every measurement, and that truncation's bound, by the rule of :mod:`~dmipy_sim.replay.planning`
        (:func:`~dmipy_sim.replay.planning.band_tail` and :func:`~dmipy_sim.replay.planning.fewest`): the stored bands
        between ``K'`` and ``K`` at the manifest's per-band variance of the worst populated voxel, and the bands beyond
        ``K`` (never stored) at the free bridge's variance at the walk's diffusivity, in closed form to infinity
        (:func:`~dmipy_sim.replay.compression.band_phase_variances`), which bounds every restricted walk's. A waveform
        whose cost at the full ``K`` exceeds :attr:`truncation_budget` is refused by name; the walk is not read."""
        from .planning import band_tail, fewest
        from .compression import band_phase_variances
        K = self.K
        G = np.asarray(seq.G_eff, np.float64)
        D = self.meta["walk_params"].get("diffusivity")
        if D is None:
            raise ValueError("the layout records no diffusivity, so the truncation beyond its stored bands cannot be bounded")
        pb, total = band_phase_variances(G, float(seq.dt), self.T_window, K, float(D))
        beyond = 0.5 * np.clip(total - pb.sum(axis=1), 0.0, None)                  # the bands never stored, per measurement
        budget = self.truncation_budget
        if beyond.max() > budget["budget"]:
            raise ValueError(f"the waveform's gradient reaches beyond this layout's temporal band: the {K} stored bands "
                             f"({K / (2.0 * self.T_window):.4g} Hz) leave up to {beyond.max():.2e} of the signal in phase the layout "
                             f"does not carry, against the {budget['budget']:.2e} it may cost (its cap {budget['cap']:.2e} from "
                             f"{budget['cap_source']} less {budget['other_terms']:.2e} of other certified terms)")
        var = np.asarray(self.meta["columnar"].get("band_variance_max") or self.meta["columnar"]["band_variance"], np.float64).T   # (K, 3)
        tail = (band_tail(G, seq.dt, [(self.T_window, None)], var) + beyond[:, None]).max(axis=0)
        return fewest(tail, self.band_groups, tol * self.floor)

    def modes_for(self, seq, scanner, tissue, tol=0.25):
        """``(M, error)``: the fewest stored field modes for a replay at the scanner's field with the tissue's chi (0
        when :func:`~dmipy_sim.replay.replay.field_term` resolves the pair to no field term), by the rule of
        :mod:`~dmipy_sim.replay.planning` (:func:`~dmipy_sim.replay.planning.mode_tail`) on the manifest's per-mode variance."""
        from .replay import field_term
        from .planning import mode_tail, fewest
        f = field_term(self.susceptibility_field, tissue, scanner)
        if not f.active:
            return 0, 0.0
        if not self.path_groups:
            raise ValueError("this pair has a field term and the layout stores no field modes (C3 path channel)")
        gate = seq.effective_gate if seq.gate is None else seq.gate
        var = np.asarray(self.meta["columnar"]["path_mode_variance"], np.float64).sum(axis=0)
        tail = mode_tail(gate, seq.dt, [(self.T_window, None)], self.path_groups[-1], var, f.B0, abs(f.chi_iso) + abs(f.chi_aniso))
        return fewest(tail, self.path_groups, tol * self.floor)

    def plan(self, seq, *, tissue=None, scanner=None, tol=0.25, voxels=None):
        """What a replay of ``seq`` reads: bands, field modes, tiers, rows and bytes -- before any byte moves."""
        K, eK = self.bands_for(seq, tol); M, eM = self.modes_for(seq, scanner, tissue, tol)
        contact = tissue is not None and (tissue.rho2 is not None or tissue.rho1 is not None)
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
            mb = re.fullmatch(r"pos_[xyz]_b(\d+)(?:u\d+)?", n)
            mp = re.fullmatch(r"susc_path_dct_b(\d+)(?:u\d+)?", n)
            if mb:
                if int(mb.group(1)) >= n_b:
                    continue
            elif mp:
                if int(mp.group(1)) >= n_m:
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
        if meta["compression"].get("container"):                           # the ranges read: a prefix of the container
            meta["compression"]["container"] = meta["compression"]["container"][:len([g for g in self.band_groups if g <= K_read])]
        pmeta = meta["compression"]["channels"].get("susc_field_basis") or {}
        if M_read:
            n_m = len([g for g in self.path_groups if g <= M_read])
            if pmeta.get("container"):                                      # the container's ranges, kept as stored: a prefix of them
                pmeta["container"] = pmeta["container"][:n_m]
            else:
                arrays["susc_path_dct"] = np.concatenate([arrays.pop(f"susc_path_m{i}") for i in range(n_m)], axis=2)
            arrays["susc_path_scale"] = np.ascontiguousarray(arrays["susc_path_scale"][..., :M_read])
            meta["compression"]["channels"]["susc_field_basis"]["K"] = int(M_read)
        else:
            meta["compression"]["channels"].pop("susc_field_basis", None); meta["compression"]["channels"].pop("susceptibility_grid", None)
            if self.path_groups:
                meta["view"] = dict(field="unread")                    # the layout holds C3; this view reads none of it
        if not contact:
            meta["compression"]["channels"].pop("boundary_local_time", None)
        return ReplayPack(arrays, meta)

    def row_groups(self, chunk_rows=2_000_000):
        """The index rows in groups of about ``chunk_rows`` rows, cut on (voxel, pool) boundaries: the groups
        :meth:`iter_views` reads, each a list of the index's ``{ijk, pool, start, end}`` rows in row order."""
        groups, cur, n = [], [], 0
        for r in self.index["rows"]:
            cur.append(r); n += r["end"] - r["start"]
            if n >= chunk_rows:
                groups.append(cur); cur, n = [], 0
        if cur:
            groups.append(cur)
        return groups

    @staticmethod
    def group_ranges(rows):
        """The contiguous row ranges ``(start, end)`` of a group of index rows, adjacent ranges merged."""
        rs = []
        for r in rows:
            if rs and rs[-1][1] == r["start"]:
                rs[-1] = (rs[-1][0], r["end"])
            else:
                rs.append((r["start"], r["end"]))
        return rs

    def iter_views(self, chunk_rows=2_000_000, **kw):
        """The pack in row groups of about ``chunk_rows`` on (voxel, pool) boundaries (:meth:`row_groups`), each a
        view (``**kw`` as :meth:`view`), the next group's bytes fetched while the current one is contracted; a
        failed fetch reaches the consumer."""
        import queue
        groups = [self.group_ranges(g) for g in self.row_groups(chunk_rows)]
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
        from .study import Study
        if isinstance(seq, Study):
            return self.image_study(seq, tol=tol, chunk_rows=chunk_rows, progress=progress)
        return self._image_settings(seq, tissue=tissue, scanner=scanner, settings=settings, tol=tol, chunk_rows=chunk_rows, progress=progress)

    def image_study(self, study, *, tol=0.25, chunk_rows=2_000_000, progress=None):
        """``(S, floor, plan)`` for a :class:`~dmipy_sim.replay.study.Study` (dmipy-sim#297): ``S`` is
        ``(pairs, *grid, n_meas)``, ``floor`` ``(pairs, *grid)`` the split-half floor of each volume (the rows of a
        voxel in two halves, ``max_m |S_a - S_b| / 2`` on the S0-normalised signal), ``plan`` what was read. One pass
        over the rows: per row group the primitives of every acquisition once (the bands and the path channel
        contracted, the exposures and the contact read), on the device; every (tissue, scanner) pair then an
        elementwise term and two segment sums."""
        import functools
        import jax
        import jax.numpy as jnp
        was_x64 = bool(jax.config.read("jax_enable_x64"))
        jax.config.update("jax_enable_x64", True)
        try:
            pairs = [study.resolved(k) for k in range(len(study))]
            plans = [self.plan(a.waveform, tissue=t_, scanner=s_, tol=tol) for a in study.protocol for t_, s_ in pairs]
            plan = dict(plans[0], K=max(p["K"] for p in plans), modes=max(p["modes"] for p in plans), contact=any(p["contact"] for p in plans),
                        relaxation=any(p["relaxation"] for p in plans), settings=len(pairs), study=study.to_meta())
            self.src.bytes_read = self.src.requests = 0; t0 = time.time()
            n_meas = study.protocol.n_meas; n_vox = int(np.prod(self.grid.shape))
            num = np.zeros((len(pairs), 2, n_vox, n_meas), complex); den = np.zeros((2, n_vox)); rows = 0
            ROWS, SEGS = 1 << 16, 256

            @functools.partial(jax.jit, static_argnums=(10,))
            def pair_sums(phi, fi, fa, et2, et1, ct, ct1, w, seg2, terms, n_seg2):
                """``Primitives.signals`` on the device: `(ew[:, None] * E)` summed over each voxel's rows, with every
                term taken from :meth:`~dmipy_sim.replay.study.Primitives.reduction_terms` and none re-derived here."""
                a_i, a_a, rho_D, rho1_D, invT2, invT1, amp, vox = terms
                logw = -(et2 @ invT2) - (et1 @ invT1) + rho_D * ct + rho1_D * ct1
                ph = phi + (a_i * fi + a_a * fa)[:, None]
                E = jnp.exp(1j * ph) * vox[None, :]                       # replay._signal_factor
                return jax.ops.segment_sum(E * (amp * w * jnp.exp(logw))[:, None], seg2, num_segments=n_seg2)
            for pk in self.iter_views(chunk_rows=chunk_rows, K=plan["K"], modes=plan["modes"], contact=plan["contact"]):
                n = pk.n_walkers
                ijk, _ = self.grid.bin(pk.r0); v = np.ravel_multi_index(ijk.T, self.grid.shape)
                starts = np.flatnonzero(np.r_[True, v[1:] != v[:-1]]); n_seg = len(starts)
                seg = np.repeat(np.arange(n_seg), np.diff(np.r_[starts, n])); half = (np.arange(n) - starts[seg]) % 2
                n_pad = -(-n // ROWS) * ROWS; s_pad = -(-(n_seg + 1) // SEGS) * SEGS
                seg2 = np.full(n_pad, 2 * n_seg, np.int32); seg2[:n] = 2 * seg + half
                w_all = None
                for a, sl in zip(study.protocol, study.protocol.slices):
                    prim = pk.walker_primitives(a, tol=None); w_all = prim.w
                    pad = lambda x, shape: (np.zeros((n_pad,) + shape) if x is None else np.concatenate([np.asarray(x, np.float64), np.zeros((n_pad - n,) + shape)]))
                    n_pools = prim.n_pools or 1
                    phi = pad(prim.phi, (prim.phi.shape[1],)); fi = pad(prim.field_iso, ()); fa = pad(prim.field_aniso, ())
                    et2 = pad(prim.exposure_t2, (n_pools,)); et1 = pad(prim.exposure_t1, (n_pools,))
                    ct = pad(prim.contact, ()); ct1 = pad(prim.contact_t1, ()); w = pad(prim.w, ())
                    dev = [jnp.asarray(x) for x in (phi, fi, fa, et2, et1, ct, ct1, w)] + [jnp.asarray(seg2)]
                    for k, (t_, s_) in enumerate(pairs):
                        rt = prim.reduction_terms(t_, s_)                        # every per-pair term, resolved once
                        terms = (jnp.float64(rt["a_iso"]), jnp.float64(rt["a_aniso"]), jnp.float64(rt["rho2_over_D"]),
                                 jnp.float64(rt["rho1_over_D"]), jnp.asarray(rt["invT2"]), jnp.asarray(rt["invT1"]),
                                 jnp.float64(rt["amplitude"]), jnp.asarray(rt["voxel"], jnp.float64))
                        sums = np.asarray(pair_sums(*dev, terms, 2 * s_pad))[:2 * n_seg].reshape(n_seg, 2, -1)
                        np.add.at(num[k, 0, :, sl], v[starts], sums[:, 0]); np.add.at(num[k, 1, :, sl], v[starts], sums[:, 1])
                wh = np.zeros((2, n)); wh[half, np.arange(n)] = w_all
                np.add.at(den[0], v[starts], np.add.reduceat(wh[0], starts)); np.add.at(den[1], v[starts], np.add.reduceat(wh[1], starts))
                rows += n
                if progress:
                    progress(rows, self.src.bytes_read, time.time() - t0)
            S = np.full((len(pairs), n_vox, n_meas), np.nan); floor = np.full((len(pairs), n_vox), np.nan)
            both = (den > 0).all(0); any_ = den.sum(0) > 0
            tot = num.sum(1); S[:, any_] = np.abs(tot[:, any_] / den.sum(0)[any_][None, :, None])
            Sa = np.abs(num[:, 0][:, both] / den[0][both][None, :, None]); Sb = np.abs(num[:, 1][:, both] / den[1][both][None, :, None])
            floor[:, both] = 0.5 * np.abs(Sa - Sb).max(-1)
            plan.update(bytes_read=self.src.bytes_read, requests=self.src.requests, seconds=time.time() - t0, rows=rows)
            return S.reshape((len(pairs),) + tuple(self.grid.shape) + (n_meas,)), floor.reshape((len(pairs),) + tuple(self.grid.shape)), plan
        finally:
            jax.config.update("jax_enable_x64", was_x64)

    def _image_settings(self, seq, *, tissue=None, scanner=None, settings=None, tol=0.25, chunk_rows=2_000_000, progress=None):
        """``(S, floor, plan)``: the signal of every voxel of the grid under ``seq`` (one sequence or a list, their
        measurements side by side; NaN where the pack has none), the certificate's floor beside it (the worst pool's),
        and the plan with what was read. One pass over the rows: the host fetches, dequantises and forms each
        walker's phase (:meth:`ReplayPack.walker_phases`); the device forms the exponential and the sum over each
        voxel's rows in double precision. With ``settings`` (a list of ``(tissue, scanner)``) every setting's volume
        comes from the same pass -- the settings differ only in what is applied to the rows -- and ``S`` gains a
        leading axis. ``progress(rows, bytes, seconds)`` is called after every row group."""
        import jax
        import jax.numpy as jnp
        was_x64 = bool(jax.config.read("jax_enable_x64"))
        jax.config.update("jax_enable_x64", True)
        try:
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
                        w, ew, p = pk.walker_phases(s, tissue=t_, scanner=B_, tol=None)
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
        finally:
            jax.config.update("jax_enable_x64", was_x64)


def open_columnar(uri, workers=8):
    """The columnar layout at ``uri`` (a directory, or ``hf://owner/name/prefix``) as a :class:`ColumnarPack`."""
    return ColumnarPack(uri, workers=workers)
