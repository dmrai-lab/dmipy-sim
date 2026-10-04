"""A replay pack contracted against the shape of a gradient waveform, once, so that every acquisition that plays
that shape -- at any amplitude, in any direction -- is a three-term phase per walker (dmipy-sim#505: the whole
DiSCo grid replayed while a user waits).

A single-direction waveform is ``G(t) = g s(t) u``: ``s`` its shape (``max |s| = 1``), ``g`` its amplitude (T/m)
and ``u`` its unit direction. The band contraction of a replay (:func:`~dmipy_sim.replay.replay._band_phase`) is
linear in the waveform, so the gradient phase of walker ``w`` is ``g (u . m_w)`` with the **shape moment**
``m_w = gamma int s(t) r_w(t) dt`` (rad per T/m): the pack's bands contracted against the shape's projection on
the three axes, once per shape, through the same ``_prepare`` / ``_band_phase`` route every replay takes.

The layout is **tiled**: rows are stored in tiles of ``TILE`` rows, every tile the rows of one segment (a voxel and
a split half), the last tile of a segment padded with zero weight, so that a voxel's sum is a fused reduction over
whole tiles and the sum over tiles a scatter with as many collisions as a segment has tiles. A scatter over rows
serialises on a voxel's thousands of contiguous rows (measured on the L40S: 6.4 s per 2^19 rows against 1 ms for
the tiled reduction). Per shape ``m_<name>.npy`` holds ``(n_tiles, TILE, 3)`` float32; once, ``w.npy``
``(n_tiles, TILE)`` (the walkers' weights, 0 on padding), ``tiles.npy`` ``(n_tiles,)`` (``2 voxel + half``), and
``manifest.json`` naming the source layout (its manifest's sha256), the bands read and their truncation error,
and every shape's profile with the b-value it encodes at unit amplitude, so that an amplitude follows from a
b-value as ``g = sqrt(b / b_unit)``.

With ``tiers=True`` the layout also holds each walker's **primitives** as :func:`~dmipy_sim.replay.study.walker_primitives`
gives them, gated by the shape's coherence pathway: once, the walker's pool (``pool.npy``, uint8: a walker that never
crosses has one exposure, its pool's); per **sequence group** (the shapes that share a grid, an RF schedule and a
readout -- every PGSE class at one echo time is one group, a stimulated-echo class another, since its magnetisation
is longitudinal during the mixing time and its field integral, its transverse exposure and its T1 exposure differ),
the gated boundary local time (``contact_<g>.npy``) and the gated path-field channel integrals
(``field_<g>.npy``, the 7 or 13 channels :func:`~dmipy_sim.fields.hollow_cylinder.field_terms` contracts under a
field direction), with the group's transverse and longitudinal exposure times and its pathway amplitude, the embedded
substrate spec and the walk's diffusivity in the manifest. :meth:`ShapeMoments.image` then takes ``tissue=``,
``scanner=`` and ``b0_direction=`` (the field's direction in the substrate frame: the magnitude and the direction of
B0 are both knobs) and applies exactly :meth:`~dmipy_sim.replay.study.Primitives.reduction_terms`: the per-row
weight ``amp exp(-tau_2 / T2_pool - tau_1 / T1_pool + (rho_2 / D) contact)`` and the phase ``B0 (chi_iso A + chi_aniso
B)`` with ``A, B`` contracted from the channels for that direction on the host per call (dmipy-sim#514); a tier asked
for that the layout does not carry is refused by name, a tier left at None is inactive and the image is the bare one. A shape's pathway amplitude (a stimulated echo's ``0.5 sin a1 sin a2 sin a3``) scales
its image in every case.

A shape is a one-row single-direction :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence` whose
ramps do not scale with its amplitude (built with ``slew_rate=np.inf``): a finite slew makes the shape a
function of ``g`` and the moment would not be one vector. The image of an acquisition on the layout is
``|sum_w w exp(i g u . m_w)| / sum_w w`` per voxel with its split-half floor, formed on the device from
device-resident rows: float32 phases with the three products written out (a float32 matmul on CUDA is TF32),
the exponential in complex64, the tile sums and every sum after them in complex128. Two backends run that
arithmetic, ``"jax"`` (the reference) and ``"torch"`` (dmipy-sim#510: Hugging Face's shared GPU pool runs PyTorch
only, and drops the device between calls, so the torch path can rebuild its tensors from the memory-mapped tiles
on every call); the tiling, the manifest, the amplitudes, the weights and the NaN / floor bookkeeping are the one
class's, the backend chooses only where ``exp``, the tile reduce and the scatter run.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

MANIFEST = "manifest.json"
FORMAT = "shape_moments/2"
TILE = 128                                                            # rows per tile
_AXES = np.eye(3)


def _profile(seq):
    """``(s, u, g)`` of a one-row single-direction sequence: its shape (``max |s| = 1``), unit direction and amplitude
    (T/m); a refusal for anything else, and for ramps that scale with the amplitude."""
    G = np.asarray(seq.G, np.float64)
    if G.ndim != 3 or G.shape[0] != 1:
        raise ValueError(f"a shape is a one-row sequence, got {G.shape[0]} measurements")
    g_t = G[0]                                                          # (n_t, 3)
    peak = g_t[np.argmax(np.linalg.norm(g_t, axis=1))]
    g = float(np.linalg.norm(peak))
    if g == 0.0:
        raise ValueError("a shape needs a non-zero gradient; build it at any amplitude")
    u = peak / g
    s = g_t @ u
    off = np.linalg.norm(g_t - s[:, None] * u[None, :], axis=1).max()
    if off > 1e-6 * g:
        raise ValueError("a shape plays one direction; this waveform's gradient turns (a b-tensor or rotating "
                         "encoding is not one shape at many amplitudes)")
    ramp = getattr(seq.encoding, "ramp_time", None) if seq.encoding is not None else None
    if ramp is not None and np.any(np.asarray(ramp, np.float64) > 0):
        raise ValueError("this shape's ramps scale with its amplitude (a finite slew rate), so it is a different "
                         "shape at every b-value; build it with slew_rate=np.inf")
    return s / g, u, g


def _axis_sequence(seq, s):
    """The shape at unit amplitude along x, y and z as one three-row sequence: its phases are a walker's moment."""
    from ..acquisition.waveforms import tile_waveform
    return tile_waveform(seq, 3).with_gradient(np.stack([s[:, None] * e[None, :] for e in _AXES]))


def _contractions(shapes, prof):
    """The sequences whose band contraction gives every shape's moment, as few as the shapes allow: shapes on one
    grid with one RF schedule and readout share a sequence (their rows side by side, one contraction per row
    group); ``[(sequence, [(name, first_row), ...]), ...]``."""
    from ..acquisition.waveforms import tile_waveform
    groups = {}
    for name, seq in shapes.items():
        key = (float(seq.dt), int(np.shape(seq.G)[1]), seq.rf, seq.readout, seq.timing is None or seq.timing,
               json.dumps(_jsonable(seq.crusher), sort_keys=True))
        groups.setdefault(key, []).append(name)
    out = []
    for names in groups.values():
        base = shapes[names[0]]
        G = np.concatenate([np.stack([prof[n][0][:, None] * e[None, :] for e in _AXES]) for n in names])
        out.append((tile_waveform(base, len(G)).with_gradient(G), [(n, 3 * i) for i, n in enumerate(names)]))
    return out


def _plan(col, chunk_rows):
    """The tiling, from the index alone: per row group of the pass, its voxel runs ``(voxel, start, n, tile0,
    tile1)`` -- the run's rows within the group, and the first tile of each half -- and every tile's segment."""
    groups, tiles, t = [], [], 0
    for rows in col.row_groups(chunk_rows):
        runs, off = [], 0
        for r in rows:
            v = int(np.ravel_multi_index(tuple(r["ijk"]), col.grid.shape)); n = r["end"] - r["start"]
            if runs and runs[-1][0] == v:
                runs[-1][2] += n
            else:
                runs.append([v, off, n])
            off += n
        planned = []
        for v, start, n in runs:
            n0, n1 = (n + 1) // 2, n // 2
            k0, k1 = -(-n0 // TILE), -(-n1 // TILE)
            planned.append((v, start, n, t, t + k0)); tiles += [2 * v] * k0 + [2 * v + 1] * k1; t += k0 + k1
        groups.append(planned)
    return groups, np.asarray(tiles, np.int32)


def write_shape_moments(source, shapes, out_dir, *, tol=0.25, chunk_rows=2_000_000, workers=8, progress=None, tiers=False):
    """The shape-moment layout of the columnar pack at ``source`` (a
    :class:`~dmipy_sim.replay.columnar.ColumnarPack`, a directory or ``hf://owner/name/prefix``) for ``shapes``,
    a ``{name: sequence}`` of one-row single-direction sequences, written to ``out_dir`` in one pass over the rows
    at the fewest band groups whose truncation error stays under ``tol`` times the pack's median floor on every
    axis of every shape **at the amplitude the shape is built at** (the error grows with the amplitude squared:
    build a shape at the largest amplitude it will be replayed at). ``progress(rows, bytes, seconds)`` is called
    after every row group. With ``tiers`` the walkers' primitives at the shapes' common echo time are stored too (the
    field modes and the contact tier read in the same pass). Returns the manifest."""
    from .columnar import ColumnarPack
    from .replay import _band_phase
    from .study import Acquisition
    from ..acquisition.waveforms import calc_b
    col = source if isinstance(source, ColumnarPack) else ColumnarPack(source, workers=workers)
    if not shapes:
        raise ValueError("no shapes: give at least one {name: sequence}")
    prof = {name: _profile(seq) for name, seq in shapes.items()}
    contractions = _contractions(shapes, prof)
    plans = [col.bands_for(_axis_sequence(shapes[name], prof[name][0] * prof[name][2]), tol=tol) for name in shapes]
    K = max(k for k, _ in plans); band_error = max(e for _, e in plans)
    b_unit = {name: float(calc_b(_axis_sequence(shapes[name], prof[name][0]))[0]) for name in shapes}
    groups, tiles = _plan(col, chunk_rows)
    n_tiles = len(tiles); n_rows = col.n_rows; n_vox = int(np.prod(col.grid.shape))
    os.makedirs(out_dir, exist_ok=True)
    m = {name: np.lib.format.open_memmap(os.path.join(out_dir, f"m_{name}.npy"), mode="w+", dtype=np.float32, shape=(n_tiles, TILE, 3))
         for name in shapes}
    w = np.lib.format.open_memmap(os.path.join(out_dir, "w.npy"), mode="w+", dtype=np.float32, shape=(n_tiles, TILE))
    np.save(os.path.join(out_dir, "tiles.npy"), tiles)
    view_kw = dict(K=K)
    tier_cols = {}; tier_groups = {}
    from ..acquisition.epg import pathway_weight
    pathway = {name: float(pathway_weight(seq)) for name, seq in shapes.items()}
    if tiers:
        modes = col.path_groups[-1] if col.path_groups else 0
        view_kw = dict(K=K, modes=modes, contact=True)
        tier_cols["pool"] = np.lib.format.open_memmap(os.path.join(out_dir, "pool.npy"), mode="w+", dtype=np.uint8, shape=(n_tiles, TILE))
        for gi, (a, names) in enumerate(contractions):
            g = f"g{gi}"
            tier_cols[f"contact_{g}"] = np.lib.format.open_memmap(os.path.join(out_dir, f"contact_{g}.npy"), mode="w+", dtype=np.float32, shape=(n_tiles, TILE))
            tier_groups[g] = dict(shapes=[n_ for n_, _ in names], acquisition=Acquisition(shapes[names[0][0]]), tau_t2=None, tau_t1=None,
                                  pathway=pathway[names[0][0]], contact=False, field=False, field_channels=None, has_aniso=False,
                                  columns={"contact": f"contact_{g}.npy"})
        tier_meta = dict(modes=int(modes), pools=None, D_walk=None, substrate=col.meta.get("substrate"), relaxation=False, exposure="pool")
    col.src.bytes_read = col.src.requests = 0; t0 = time.time(); row = 0
    for pk, runs in zip(col.iter_views(chunk_rows=chunk_rows, **view_kw), groups):
        n = pk.n_walkers
        ijk, _ = col.grid.bin(pk.r0); v = np.ravel_multi_index(ijk.T, col.grid.shape)
        expect = np.concatenate([np.full(n_, v_, v.dtype) for v_, _, n_, _, _ in runs])
        if len(expect) != n or not np.array_equal(v, expect):
            raise RuntimeError("the rows' start voxels disagree with the index's (voxel, pool) ranges")
        phi = {}
        for a, names in contractions:
            P = pk._prepare(a, tissue=None, scanner=None, orientation=None, compartment=None)
            if np.any(P["voxel"] != 1.0):
                raise ValueError(f"shapes {[n_ for n_, _ in names]} leave an amplitude per measurement (an unbalanced "
                                 "encoding); a moment carries a phase only")
            ph = _band_phase(P)                                     # (n, 3 per shape): the one band contraction
            for name, first in names:
                phi[name] = ph[:, first:first + 3]
            w_rows = P["w"]
        tier_rows = {}
        if tiers:
            for gi, (a, names) in enumerate(contractions):
                g = f"g{gi}"; grp = tier_groups[g]
                prim = pk.walker_primitives(grp["acquisition"])         # the tiers at this group's gate, the study image's route
                if prim.exposure_t2 is not None:
                    e2 = np.asarray(prim.exposure_t2); e1 = np.asarray(prim.exposure_t1)
                    if not np.allclose(e2.max(1), e2.sum(1)):
                        raise ValueError("a walker was in more than one pool before the echo; this layout stores one pool "
                                         "per walker and needs a walk that does not cross")
                    pool = np.argmax(e2, axis=1).astype(np.uint8)
                    t2 = e2.max(1); t1 = e1.max(1)
                    for tau, key in ((t2, "tau_t2"), (t1, "tau_t1")):
                        if tau.size and not np.allclose(tau, tau[0], atol=1e-9):
                            raise ValueError(f"the {key} exposure differs between walkers of one gate; a non-crossing walk has one")
                        grp[key] = float(tau[0]) if tau.size else 0.0
                    tier_meta.update(relaxation=True, pools=int(e2.shape[1]))
                    tier_rows["pool"] = pool
                else:
                    tier_rows.setdefault("pool", np.zeros(n, np.uint8))
                tier_rows[f"contact_{g}"] = np.zeros(n, np.float32) if prim.contact is None else np.asarray(prim.contact, np.float32)
                grp["contact"] = grp["contact"] or prim.contact is not None
                if prim.field_iso is not None:                          # the channels themselves, so the field direction stays a knob
                    from .replay import _path_field_channels, _has_aniso
                    acq = grp["acquisition"]
                    Pg = pk._prepare(acq.waveform, tissue=None, scanner=None, orientation=acq.orientation, compartment=None)
                    Psi, ch_names = _path_field_channels(Pg, acq.waveform)
                    Psi = np.asarray(Psi, np.float32)
                    if f"field_{g}" not in tier_cols:
                        tier_cols[f"field_{g}"] = np.lib.format.open_memmap(os.path.join(out_dir, f"field_{g}.npy"), mode="w+", dtype=np.float32, shape=(n_tiles, TILE, Psi.shape[1]))
                        grp["columns"]["field"] = f"field_{g}.npy"; grp["field"] = True
                        grp["field_channels"] = list(ch_names); grp["has_aniso"] = bool(_has_aniso(pk.meta["compression"]["channels"].get("susceptibility_grid"), ch_names))
                    tier_rows[f"field_{g}"] = Psi
                if prim.D_walk is not None:
                    tier_meta["D_walk"] = float(prim.D_walk)
        for v_, start, n_, t0_, t1_ in runs:
            for half, t_ in enumerate((t0_, t1_)):
                k = (n_ + 1 - half) // 2
                if k == 0:
                    continue
                sl = slice(start + half, start + n_, 2); flat = slice(t_ * TILE, t_ * TILE + k)
                for name in shapes:
                    m[name].reshape(-1, 3)[flat] = phi[name][sl]
                w.reshape(-1)[flat] = w_rows[sl]
                if tiers:
                    for name, arr in tier_cols.items():
                        if arr.ndim == 3:
                            arr.reshape(-1, arr.shape[-1])[flat] = tier_rows[name][sl]
                        else:
                            arr.reshape(-1)[flat] = tier_rows[name][sl]
        row += n
        if progress:
            progress(row, col.src.bytes_read, time.time() - t0)
    if row != n_rows:
        raise RuntimeError(f"the pass read {row} rows of {n_rows}")
    for a in list(m.values()) + [w] + list(tier_cols.values()):
        a.flush()
    manifest = dict(
        format=FORMAT, n_rows=int(n_rows), n_tiles=int(n_tiles), tile=TILE, K=int(K), band_error=float(band_error), tol=float(tol),
        tiers=(dict(tier_meta, pool_column="pool.npy",
                    groups={g: {k: v for k, v in grp.items() if k != "acquisition"} for g, grp in tier_groups.items()}) if tiers else None),
        source=dict(uri=col.uri, manifest_sha256=hashlib.sha256(col.src.text(MANIFEST).encode()).hexdigest(),
                    K=int(col.K), n_rows=int(col.n_rows), floor=float(col.floor),
                    grid=col.meta["fidelity"]["per_voxel"]["grid"], pack=col.meta.get("id")),
        n_voxels=int(n_vox), columns=dict(w="w.npy", tiles="tiles.npy", **{name: f"m_{name}.npy" for name in shapes}),
        shapes={name: dict(profile=prof[name][0].tolist(), dt=float(shapes[name].dt), n_t=int(len(prof[name][0])),
                           b_unit=b_unit[name], amplitude_built=prof[name][2], pathway=pathway[name], family=getattr(shapes[name], "family", None),
                           build_spec=_jsonable(shapes[name].build_spec),
                           encoding={k: _jsonable(getattr(shapes[name].encoding, k, None))
                                     for k in ("delta", "Delta", "TE") if shapes[name].encoding is not None})
                for name in shapes},
        pass_=dict(bytes_read=int(col.src.bytes_read), requests=int(col.src.requests), seconds=time.time() - t0,
                   chunk_rows=int(chunk_rows)),
        code=_code(), created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(out_dir, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def _code():
    """The dmipy-sim that wrote a layout: the package version and its git commit when the tree is a checkout."""
    import subprocess
    from importlib.metadata import version, PackageNotFoundError
    from .. import __file__ as pkg
    root = os.path.dirname(os.path.dirname(os.path.abspath(pkg)))
    try:
        ver = version("dmipy-sim")
    except PackageNotFoundError:
        ver = None
    try:
        r = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        commit = r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        commit = None
    return dict(package="dmipy-sim", version=ver, commit=commit)


def _jsonable(x):
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return repr(x)


class ShapeMoments:
    """The shape-moment layout at ``path``, tiles memory-mapped from disk: ``manifest``, ``grid``, ``shapes`` (the
    names), and :meth:`image` -- a whole grid under any amplitudes and directions of one shape, on the device.
    Open a Hub layout with :meth:`open`."""

    TILES = 4096                                                        # tiles per device chunk (2^19 rows)
    MEAS = 64                                                           # measurements padded to a multiple

    def __init__(self, path):
        from ..phantom.grid import Grid
        self.path = path
        with open(os.path.join(path, MANIFEST)) as f:
            self.manifest = json.load(f)
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} layout ({self.manifest.get('format')!r})")
        self.grid = Grid.from_meta(self.manifest["source"]["grid"])
        self.n_rows = int(self.manifest["n_rows"]); self.n_tiles = int(self.manifest["n_tiles"]); self.tile = int(self.manifest["tile"])
        self.n_vox = int(np.prod(self.grid.shape))
        self.shapes = [n for n in self.manifest["columns"] if n not in ("w", "tiles")]
        self.tiers = self.manifest.get("tiers")
        self._m = {}; self._device = {}; self._host = {}

    @staticmethod
    def open(uri, *, shapes=None, revision=None, workers=8):
        """The layout at ``uri``: a directory, or ``hf://owner/name/prefix`` fetched into the Hub cache (the
        manifest, the shared tiles and the moments of ``shapes``, every shape when None) at ``revision``."""
        if not uri.startswith("hf://"):
            return ShapeMoments(uri)
        from huggingface_hub import snapshot_download, hf_hub_download
        owner, name, prefix = uri[5:].split("/", 2); repo = f"{owner}/{name}"
        mf = hf_hub_download(repo, f"{prefix}/{MANIFEST}", repo_type="dataset", revision=revision)
        with open(mf) as f:
            manifest = json.load(f)
        local = snapshot_download(repo, repo_type="dataset", revision=revision, max_workers=workers,
                                  allow_patterns=[f"{prefix}/{f}" for f in ShapeMoments.files(manifest, shapes)])
        return ShapeMoments(os.path.join(local, prefix))

    @staticmethod
    def columns(manifest):
        """Every column of a layout by name -> file: ``w``, ``tiles``, one per shape, and with tiers ``pool`` and
        ``contact_<g>`` / ``field_<g>`` per sequence group."""
        cols = dict(manifest["columns"])
        tiers = manifest.get("tiers")
        if tiers:
            cols["pool"] = tiers["pool_column"]
            for g, grp in tiers["groups"].items():
                cols.update({f"{c}_{g}": f for c, f in grp["columns"].items()})
        return cols

    @staticmethod
    def files(manifest, shapes=None):
        """The files a layout needs for ``shapes`` (every shape when None): the manifest, the shared columns, the
        tier columns and the moments of those shapes."""
        cols = ShapeMoments.columns(manifest)
        keep = set(manifest["columns"]) - {"w", "tiles"} if shapes is None else set(shapes)
        return [MANIFEST] + [f for name, f in cols.items() if name in ("w", "tiles") or name not in manifest["columns"] or name in keep]

    def _column(self, name):
        if name not in self._m:
            self._m[name] = np.load(os.path.join(self.path, self.columns(self.manifest)[name]), mmap_mode="r")
        return self._m[name]

    def _tier_group(self, shape):
        """The sequence group of ``shape`` (its gate): ``(g, group)``."""
        for g, grp in self.tiers["groups"].items():
            if shape in grp["shapes"]:
                return g, grp
        raise KeyError(f"no tiers for shape {shape!r}")

    def _tier_columns(self, shape):
        """The device-side tier columns of ``shape``'s group: the pool once, the group's contact, and its field
        terms for the current field direction (``field_iso_<g>@<dir>``, ``field_aniso_<g>@<dir>``, contracted from
        the stored channels: on the host when preloaded there, else on the device by the torch backend)."""
        g, grp = self._tier_group(shape)
        d = self._b0_key
        return ("pool", f"contact_{g}", f"field_iso_{g}@{d}", f"field_aniso_{g}@{d}")

    def _is_tier_column(self, name):
        return name == "pool" or name.startswith(("contact_", "field_iso_", "field_aniso_", "field_"))

    @staticmethod
    def _direction_key(b0_direction):
        """The one spelling of a field direction in the tier columns' names (``field_iso_<g>@<key>``)."""
        b0 = np.asarray(b0_direction, np.float64)
        if b0.shape != (3,) or abs(np.linalg.norm(b0) - 1.0) > 1e-6:
            raise ValueError("b0_direction is a unit vector")
        return ",".join(f"{x:.6f}" for x in b0)

    _b0_key = _direction_key((0.0, 0.0, 1.0))      # the frame's z, the pose the tiers were gated under

    def _set_b0_direction(self, b0_direction):
        """The field direction (a unit vector in the substrate frame) the tier columns are contracted for."""
        self._b0_key = self._direction_key(b0_direction)

    def tier_maps(self, shape, tissue=None, scanner=None, *, backend="jax", device=None, resident=True, b0_direction=(0.0, 0.0, 1.0)):
        """Per-voxel means over the walkers, at the walk's weights, of what each tier multiplies into :meth:`image`
        on ``shape``'s gate, each on the grid (NaN where the layout has no rows): ``pool`` (the weight fraction per
        pool, ``{name: grid}``), ``contact`` (the stored term of the gated boundary local time, the exponent's ``-l``;
        None without the contact tier), ``contact_weight`` (``exp(rho_2_over_D contact)`` at the pair's rho_2, the
        contact tier's factor; None without a tissue), ``phase`` and ``phase_std`` (the mean and the spread of the
        sheath field's phase offset ``a_iso iso + a_aniso aniso`` at the pair's field and ``b0_direction``, radians;
        None without a scanner). The same device columns as the tiered image (kept across calls when ``resident``),
        one pass over the tiles; ``backend`` and ``device`` as in :meth:`image`."""
        if backend not in ("jax", "torch"):
            raise ValueError(f"backend is 'jax' or 'torch', got {backend!r}")
        if not self.tiers:
            raise ValueError("this layout holds the bare diffusion phase only (written without tiers=True)")
        tm = self.terms(shape, tissue, scanner)
        self._set_b0_direction(b0_direction)
        g, grp = self._tier_group(shape)
        pools = [(q["name"], int(q["id"])) for q in self.tiers["substrate"]["pools"]]
        if backend == "torch" and device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _, w, tiles, pool, contact, fiso, faniso = self._resident(shape, backend, device, resident, names=("w", "tiles") + self._tier_columns(shape))
        rho_D, a_iso, a_aniso = (0.0, 0.0, 0.0) if tm is None else (float(tm[1]), float(tm[2]), float(tm[3]))
        names = [("pool", name, pid) for name, pid in pools]
        if grp.get("contact"):
            names.append(("contact",))
            if tm is not None:
                names.append(("contact_weight",))
        if grp.get("field") and tm is not None:
            names += [("phase",), ("phase_sq",)]

        def rows(sl, xp):
            out = []
            for q in names:
                if q[0] == "pool":
                    out.append((pool[sl] == q[2]).to(w.dtype) if backend == "torch" else (pool[sl] == q[2]).astype(w.dtype))
                elif q[0] == "contact":
                    out.append(contact[sl])
                elif q[0] == "contact_weight":
                    out.append(xp.exp(rho_D * contact[sl]))
                else:
                    ph = a_iso * fiso[sl] + a_aniso * faniso[sl]
                    out.append(ph if q[0] == "phase" else ph * ph)
            return out

        n_seg = 2 * self.n_vox + 1
        acc = (_voxel_sums_torch if backend == "torch" else _voxel_sums_jax)(rows, len(names), w, tiles, n_seg, self.TILES)
        num = acc[:, :2 * self.n_vox].reshape(len(names), self.n_vox, 2).sum(2)
        den = self.weights.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(den > 0, num / den, np.nan).reshape((len(names),) + tuple(self.grid.shape))
        by = {q[0] if q[0] != "pool" else ("pool", q[1]): m for q, m in zip(names, means)}
        maps = dict(pool={name: by[("pool", name)] for name, _ in pools}, contact=by.get("contact"), contact_weight=by.get("contact_weight"),
                    phase=by.get("phase"), phase_std=None)
        if "phase" in by:
            maps["phase_std"] = np.sqrt(np.maximum(by["phase_sq"] - by["phase"] ** 2, 0.0))
        return maps

    FIELD_CHUNK_TILES = 65_536                       # tiles per contraction chunk: 65,536 x 128 x 13 doubles = 0.9 GB of temporaries

    def _field_terms(self, g, b0_direction, device=None):
        """``(iso, aniso)`` per row of group ``g`` for ``b0_direction`` (a unit vector in the substrate frame), from
        the stored channels through the one contraction, zeros when the group carries no field. The memory-mapped
        channel column is contracted in chunks of ``FIELD_CHUNK_TILES`` tiles: on the host into two float32 arrays
        (one chunk's float64 temporaries at a time, never a copy of the whole column), or, given a torch ``device``,
        chunk by chunk uploaded and contracted there into two device tensors (the column crosses once, at the
        transfer's rate; nothing of it stays on the host)."""
        from ..fields.hollow_cylinder import field_terms
        grp = self.tiers["groups"][g]
        T = self.TILES; n_pad = -(-self.n_tiles // T) * T
        if device is None:
            iso = np.zeros((n_pad, self.tile), np.float32); aniso = np.zeros((n_pad, self.tile), np.float32)
            up = np.asarray
        else:
            import torch
            iso = torch.zeros((n_pad, self.tile), dtype=torch.float32, device=device); aniso = torch.zeros((n_pad, self.tile), dtype=torch.float32, device=device)
            up = lambda a: torch.as_tensor(np.asarray(a), device=device)
        if grp.get("field"):
            chans = self._column(f"field_{g}")
            b0 = np.asarray(b0_direction, np.float64)
            for s in range(0, self.n_tiles, self.FIELD_CHUNK_TILES):
                e = min(s + self.FIELD_CHUNK_TILES, self.n_tiles)
                a, b = field_terms(up(chans[s:e]), b0)
                iso[s:e] = a if device is None else a.to(torch.float32)
                if grp.get("has_aniso") and b is not None:
                    aniso[s:e] = b if device is None else b.to(torch.float32)
        return iso, aniso

    def terms(self, shape, tissue=None, scanner=None):
        """The reduction terms of a (tissue, scanner) pair on ``shape``'s gate -- exactly
        :meth:`~dmipy_sim.replay.study.Primitives.reduction_terms` on a descriptor of what the layout carries:
        ``None`` for the bare image (no tissue, no scanner), else ``(logw_pool (n_pools,), rho_2_over_D, a_iso, a_aniso,
        amplitude)`` with ``logw_pool = -tau_2 / T2 - tau_1 / T1`` per pool at the group's exposure times. A tier the
        pair needs that the layout lacks is refused by name."""
        if tissue is None and scanner is None:
            return None
        if not self.tiers:
            raise ValueError("this layout holds the bare diffusion phase only (written without tiers=True); a tissue "
                             "or a scanner needs the tiers")
        from .study import Primitives
        from ..spec.substrate import SubstrateSpec
        from ..spec.tissue import _by_pool_id
        t = self.tiers; g, grp = self._tier_group(shape)
        spec = SubstrateSpec.from_dict(t["substrate"]) if t.get("substrate") else None

        def by_pool(values, what):
            if values is None:
                return None
            if spec is None:
                raise ValueError(f"{what} was given by pool but the layout embeds no substrate spec")
            if not isinstance(values, dict):
                raise ValueError(f"{what} is {{pool name: seconds}} over every pool of the spec {[p.name for p in spec.pools]}")
            return [float(v) for v in _by_pool_id(spec, values, what)]
        n_pools = int(t["pools"]) if t.get("relaxation") else None
        desc = Primitives(w=np.ones(1), phi=np.zeros((1, 1)),
                          field_iso=np.zeros(1) if grp.get("field") else None, field_aniso=np.zeros(1) if grp.get("field") else None,
                          exposure_t2=np.zeros((1, n_pools)) if n_pools else None, exposure_t1=np.zeros((1, n_pools)) if n_pools else None,
                          contact=np.zeros(1) if grp.get("contact") else None, D_walk=t.get("D_walk"), voxel=np.ones(1),
                          pathway=float(grp["pathway"]), by_pool=by_pool)
        rt = desc.reduction_terms(tissue, scanner)
        logw = -(float(grp["tau_t2"] or 0.0) * np.asarray(rt["invT2"], np.float64) + float(grp["tau_t1"] or 0.0) * np.asarray(rt["invT1"], np.float64))
        return (logw, float(rt["rho_2_over_D"]), float(rt["a_iso"]), float(rt["a_aniso"]), float(rt["amplitude"]))

    def moments(self, shape):
        """``(n_tiles, tile, 3)`` float32, memory-mapped: the moment of every row under ``shape`` (0 on padding)."""
        if shape not in self.shapes:
            raise KeyError(f"no shape {shape!r} in this layout; it holds {self.shapes}")
        return self._column(shape)

    def amplitude(self, shape, bvalues):
        """The amplitudes (T/m) at which ``shape`` encodes ``bvalues`` (s/m^2): ``sqrt(b / b_unit)``."""
        b = np.asarray(bvalues, np.float64)
        if np.any(b < 0):
            raise ValueError("a b-value is not negative")
        return np.sqrt(b / float(self.manifest["shapes"][shape]["b_unit"]))

    @property
    def weights(self):
        """The per-voxel weight sums ``(n_vox, 2)`` (the two split halves), the denominator of every image."""
        w = np.asarray(self._column("w"), np.float64).sum(1); tiles = np.asarray(self._column("tiles"))
        return np.bincount(tiles, w, minlength=2 * self.n_vox).reshape(self.n_vox, 2)

    def release(self, keep=()):
        """Drop the device copies of every shape's tiles but ``keep``'s (the shared tiles stay), on every backend."""
        for name in [n for n in self._device if n[1] not in ("w", "tiles") and not self._is_tier_column(n[1]) and n[1] not in keep]:
            del self._device[name]

    def preload(self, shapes=None):
        """The padded host arrays of ``shapes`` (every shape when None) and the shared tiles kept in this process's
        memory (about 1.9 GB per shape for DiSCo), so that a device copy built later -- in a forked worker of a
        shared pool, which inherits them for free -- costs a host-to-device transfer and not a read of the layout
        (measured on Hugging Face's ZeroGPU: the mounted bucket reads at 80 MB/s, the transfer at 8 GB/s)."""
        names = ["w", "tiles"] + list(self.shapes if shapes is None else shapes)
        if self.tiers:
            for sh in (self.shapes if shapes is None else shapes):
                names += list(self._tier_columns(sh))
        for name in dict.fromkeys(names):
            if name not in self._host:
                self._host[name] = self._padded(name)

    def _padded(self, name):
        """The column ``name`` (a shape, ``"w"`` or ``"tiles"``) on the host, padded to whole chunks of ``TILES``
        (padding rows weigh nothing and scatter into the dump segment); the preloaded copy when there is one."""
        if name in self._host:
            return self._host[name]
        T = self.TILES; n_pad = -(-self.n_tiles // T) * T
        if name.startswith(("field_iso_", "field_aniso_")):
            g, d = name.split("_", 2)[2].split("@")
            iso, aniso = self._field_terms(g, [float(x) for x in d.split(",")])
            self._host[f"field_iso_{g}@{d}"], self._host[f"field_aniso_{g}@{d}"] = iso, aniso
            return self._host[name]
        if name == "w" or self._is_tier_column(name):
            a = np.zeros((n_pad, self.tile), np.uint8 if name == "pool" else np.float32); a[:self.n_tiles] = self._column(name)
        elif name == "tiles":
            a = np.full(n_pad, 2 * self.n_vox, np.int32); a[:self.n_tiles] = self._column("tiles")
        else:
            a = np.zeros((n_pad, self.tile, 3), np.float32); a[:self.n_tiles] = self.moments(name)
        return a

    def _resident(self, shape, backend, device, resident, names=("w", "tiles")):
        """``(m, *names)`` for ``shape`` on the backend's device, kept across calls when ``resident`` (the shared
        tiles once per backend), else built afresh from the memory-mapped tiles. Field terms for a direction the
        host has not contracted are contracted on the device (torch), the channel column streamed there once."""
        if backend == "jax":
            import jax.numpy as jnp
            put = jnp.asarray
        else:
            import torch
            put = lambda a: torch.as_tensor(a if a.dtype != np.int32 else a.astype(np.int64), device=device)
        out = []; made = {}
        for name in (shape,) + tuple(names):
            key = (backend, name, str(device))
            if key not in self._device:
                if backend == "torch" and name.startswith(("field_iso_", "field_aniso_")) and name not in self._host:
                    if name not in made:                     # a direction the host has not contracted: contract on the device
                        g, d = name.split("_", 2)[2].split("@")
                        iso, aniso = self._field_terms(g, [float(x) for x in d.split(",")], device=device)
                        made.update({f"field_iso_{g}@{d}": iso, f"field_aniso_{g}@{d}": aniso})
                    arr = made[name]
                else:
                    arr = put(self._padded(name))
                if not resident:
                    out.append(arr); continue
                self._device[key] = arr
            out.append(self._device[key])
        return tuple(out)

    def image(self, shape, bvalues, directions, *, backend="jax", device=None, resident=True, tissue=None, scanner=None,
              b0_direction=(0.0, 0.0, 1.0)):
        """``(S, floor)`` of the grid under ``shape`` at ``bvalues`` (s/m^2) along ``directions`` (unit vectors):
        ``S`` is ``grid.shape + (n_meas,)``, the weighted ensemble magnitude per voxel (NaN where the layout has no
        rows), ``floor`` ``grid.shape`` its split-half floor ``max_m |S_a - S_b| / 2``. A b = 0 row's direction
        is immaterial. ``backend`` runs the arithmetic on JAX (its default device) or torch (``device``, the current
        CUDA device when None and one exists, else the CPU); ``resident=False`` uploads the tiles for this call only
        (a shared pool that drops the device between calls). Run-to-run identity of the scatter needs
        ``XLA_FLAGS=--xla_gpu_deterministic_ops=true`` on JAX and ``torch.use_deterministic_algorithms(True)`` on
        torch; the caller sets them. ``tissue`` and ``scanner`` (a :class:`~dmipy_sim.spec.tissue.Tissue`, a
        :class:`~dmipy_sim.acquisition.scanners.ScannerLimits` or tesla) turn the stored tiers into the weights and
        the field phase of :meth:`terms`; both None is the bare image. ``b0_direction`` is the field's direction in
        the substrate frame (a unit vector; the default is the frame's z, the pose the tiers were gated under), the
        channels contracted for it when a scanner is given: on the host when preloaded there, else on the device by the
        torch backend)."""
        if backend not in ("jax", "torch"):
            raise ValueError(f"backend is 'jax' or 'torch', got {backend!r}")
        tm = self.terms(shape, tissue, scanner)
        self._set_b0_direction(b0_direction)
        g = self.amplitude(shape, bvalues)
        u = np.asarray(directions, np.float64)
        if u.shape != (len(g), 3):
            raise ValueError(f"directions must be ({len(g)}, 3) to match {len(g)} b-values, got {u.shape}")
        norm = np.linalg.norm(u, axis=1); nz = g > 0
        if np.any(np.abs(norm[nz] - 1.0) > 1e-6):
            raise ValueError("directions are unit vectors")
        u = np.where(nz[:, None], u, 0.0)
        M = len(g); M_pad = -(-M // self.MEAS) * self.MEAS
        g_p = np.zeros(M_pad, np.float32); g_p[:M] = g
        u_p = np.zeros((M_pad, 3), np.float32); u_p[:M] = u
        n_seg = 2 * self.n_vox + 1
        if backend == "torch" and device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if tm is None:
            m, w, tiles = self._resident(shape, backend, device, resident)
            acc = (_sums_jax if backend == "jax" else _sums_torch)(m, w, tiles, g_p, u_p, n_seg, self.TILES)
        else:
            logw_pool, rho_D, a_iso, a_aniso, amp = tm
            m, w, tiles, pool, contact, fiso, faniso = self._resident(shape, backend, device, resident, names=("w", "tiles") + self._tier_columns(shape))
            logw_pool = np.asarray(logw_pool, np.float32)
            acc = (_sums_jax_tiers if backend == "jax" else _sums_torch_tiers)(
                m, w, tiles, pool, contact, fiso, faniso, g_p, u_p, logw_pool, np.float32(rho_D), np.float32(a_iso),
                np.float32(a_aniso), np.float32(amp), n_seg, self.TILES)
        num = acc[:2 * self.n_vox, :M].reshape(self.n_vox, 2, M)
        den = self.weights
        S = np.full((self.n_vox, M), np.nan); floor = np.full(self.n_vox, np.nan)
        any_ = den.sum(1) > 0; both = (den > 0).all(1)
        pw = float(self.manifest["shapes"][shape].get("pathway", 1.0)) if tm is None else 1.0   # the tiered path applied it as amp
        S[any_] = pw * np.abs(num[any_].sum(1) / den[any_].sum(1)[:, None])
        Sa = pw * np.abs(num[both, 0] / den[both, 0][:, None]); Sb = pw * np.abs(num[both, 1] / den[both, 1][:, None])
        floor[both] = 0.5 * np.abs(Sa - Sb).max(1)
        return S.reshape(tuple(self.grid.shape) + (M,)), floor.reshape(self.grid.shape)


def _sums_jax(m, w, tiles, g_p, u_p, n_seg, chunk):
    """The per-segment sums ``(n_seg, M_pad)`` complex128 on JAX: the compiled chunk kernel over the tiles."""
    import jax.numpy as jnp
    acc = jnp.zeros((n_seg, g_p.shape[0]), jnp.complex128); kernel = _compiled(n_seg)
    g_d, u_d = jnp.asarray(g_p), jnp.asarray(u_p)
    for i in range(0, m.shape[0], chunk):
        acc = kernel(m[i:i + chunk], w[i:i + chunk], tiles[i:i + chunk], g_d, u_d, acc)
    return np.asarray(acc)


def _tile_sums_torch(mc, wc, g, u):
    """One chunk's tile sums on torch, ``(chunk, M_pad)`` complex128: the phase as three products (a float32 matmul
    on CUDA is TF32 unless ``allow_tf32`` is off, so no matmul), ``w exp(i ph)`` in complex64, the reduce over the
    tile's rows."""
    import torch
    ph = g[None, None, :] * (mc[:, :, 0:1] * u[None, None, :, 0] + mc[:, :, 1:2] * u[None, None, :, 1] + mc[:, :, 2:3] * u[None, None, :, 2])
    return torch.polar(wc[:, :, None].expand_as(ph), ph).sum(1).to(torch.complex128)


_TORCH_KERNEL = {}


def _torch_kernel():
    """The chunk function fused by ``torch.compile`` when the compiler is available (measured on the L40S, DiSCo's
    layout, 184 measurements: eager 3.3 s per image, compiled 1.44 s, JAX 0.97 s; the first compiled call 3.6 s),
    the eager function otherwise."""
    if "fn" not in _TORCH_KERNEL:
        import torch
        try:
            _TORCH_KERNEL["fn"] = torch.compile(_tile_sums_torch, dynamic=False)
        except Exception:                                     # no compiler on this platform: eager
            _TORCH_KERNEL["fn"] = _tile_sums_torch
    return _TORCH_KERNEL["fn"]


def _sums_torch(m, w, tiles, g_p, u_p, n_seg, chunk):
    """The same sums on torch: the tile sums of every chunk (:func:`_tile_sums_torch`, compiled when possible)
    ``index_add_``-ed into a complex128 accumulator (deterministic under
    ``torch.use_deterministic_algorithms(True)``)."""
    import torch
    dev = m.device
    g = torch.as_tensor(g_p, device=dev); u = torch.as_tensor(u_p, device=dev)
    acc = torch.zeros((n_seg, g_p.shape[0]), dtype=torch.complex128, device=dev)
    fn = _torch_kernel()
    with torch.no_grad():
        for i in range(0, m.shape[0], chunk):
            try:
                ts = fn(m[i:i + chunk], w[i:i + chunk], g, u)
            except Exception:                                  # the compiled path failed at run time: eager, once for all
                _TORCH_KERNEL["fn"] = fn = _tile_sums_torch
                ts = fn(m[i:i + chunk], w[i:i + chunk], g, u)
            acc.index_add_(0, tiles[i:i + chunk], ts)
    return acc.cpu().numpy()


def _voxel_sums_torch(rows, n_q, w, tiles, n_seg, chunk):
    """The weighted per-segment sums ``(n_q, n_seg)`` float64 on torch of the ``n_q`` per-row quantities ``rows(sl,
    torch)`` gives for a chunk of tiles: each times the walk's weight, summed over the tile's rows, ``index_add_``-ed
    into the segments."""
    import torch
    acc = torch.zeros((n_q, n_seg), dtype=torch.float64, device=w.device)
    with torch.no_grad():
        for i in range(0, w.shape[0], chunk):
            sl = slice(i, i + chunk)
            ts = torch.stack([(q * w[sl]).sum(1) for q in rows(sl, torch)]).to(torch.float64)
            acc.index_add_(1, tiles[sl], ts)
    return acc.cpu().numpy()


def _voxel_sums_jax(rows, n_q, w, tiles, n_seg, chunk):
    """:func:`_voxel_sums_torch` on JAX (``segment_sum`` into the segments)."""
    import jax
    import jax.numpy as jnp
    acc = jnp.zeros((n_seg, n_q), jnp.float64)
    for i in range(0, w.shape[0], chunk):
        sl = slice(i, i + chunk)
        ts = jnp.stack([(q * w[sl]).sum(1) for q in rows(sl, jnp)], axis=1).astype(jnp.float64)
        acc = acc + jax.ops.segment_sum(ts, tiles[sl], num_segments=n_seg)
    return np.asarray(acc).T


def _sums_jax_tiers(m, w, tiles, pool, contact, fiso, faniso, g_p, u_p, logw_pool, rho_D, a_iso, a_aniso, amp, n_seg, chunk):
    """:func:`_sums_jax` with the tiers: each row's weight ``w amp exp(logw_pool[pool] + rho_D contact)`` and its
    phase offset ``a_iso field_iso + a_aniso field_aniso`` (:meth:`Primitives.signals` on the device)."""
    import jax.numpy as jnp
    acc = jnp.zeros((n_seg, g_p.shape[0]), jnp.complex128); kernel = _compiled_tiers(n_seg)
    g_d, u_d = jnp.asarray(g_p), jnp.asarray(u_p); lw = jnp.asarray(logw_pool)
    sc = tuple(jnp.float32(x) for x in (rho_D, a_iso, a_aniso, amp))
    for i in range(0, m.shape[0], chunk):
        sl = slice(i, i + chunk)
        acc = kernel(m[sl], w[sl], tiles[sl], pool[sl], contact[sl], fiso[sl], faniso[sl], g_d, u_d, lw, *sc, acc)
    return np.asarray(acc)


def _sums_torch_tiers(m, w, tiles, pool, contact, fiso, faniso, g_p, u_p, logw_pool, rho_D, a_iso, a_aniso, amp, n_seg, chunk):
    """:func:`_sums_torch` with the tiers (eager: the weight and the phase offset per row, then the chunk kernel)."""
    import torch
    dev = m.device
    g = torch.as_tensor(g_p, device=dev); u = torch.as_tensor(u_p, device=dev); lw = torch.as_tensor(logw_pool, device=dev)
    acc = torch.zeros((n_seg, g_p.shape[0]), dtype=torch.complex128, device=dev)
    fn = _torch_kernel()
    with torch.no_grad():
        for i in range(0, m.shape[0], chunk):
            sl = slice(i, i + chunk)
            wf = w[sl] * float(amp) * torch.exp(lw[pool[sl].long()] + float(rho_D) * contact[sl])
            off = float(a_iso) * fiso[sl] + float(a_aniso) * faniso[sl]
            try:
                ts = _tile_sums_torch_offset(m[sl], wf, off, g, u)
            except Exception:
                ts = _tile_sums_torch_offset(m[sl], wf, off, g, u)
            acc.index_add_(0, tiles[sl], ts)
    return acc.cpu().numpy()


def _tile_sums_torch_offset(mc, wc, off, g, u):
    """:func:`_tile_sums_torch` with a per-row phase offset ``off (chunk, tile)``."""
    import torch
    ph = g[None, None, :] * (mc[:, :, 0:1] * u[None, None, :, 0] + mc[:, :, 1:2] * u[None, None, :, 1] + mc[:, :, 2:3] * u[None, None, :, 2]) + off[:, :, None]
    return torch.polar(wc[:, :, None].expand_as(ph), ph).sum(1).to(torch.complex128)


_KERNELS = {}


def _compiled_tiers(n_seg):
    """The chunk kernel with the tiers: the per-row weight and phase offset folded into the fused reduction."""
    import jax
    import jax.numpy as jnp
    key = ("tiers", n_seg)
    if key not in _KERNELS:
        @jax.jit
        def kernel(m, w, tiles, pool, contact, fiso, faniso, g, u, logw_pool, rho_D, a_iso, a_aniso, amp, acc):
            ph = g[None, None, :] * (m[:, :, 0:1] * u[None, None, :, 0] + m[:, :, 1:2] * u[None, None, :, 1] + m[:, :, 2:3] * u[None, None, :, 2])
            ph = ph + (a_iso * fiso + a_aniso * faniso)[:, :, None]
            wf = w * amp * jnp.exp(logw_pool[pool.astype(jnp.int32)] + rho_D * contact)
            E = (jnp.exp(1j * ph) * wf[:, :, None]).astype(jnp.complex128)
            return acc + jax.ops.segment_sum(E.sum(1), tiles, num_segments=n_seg)
        _KERNELS[key] = kernel
    return _KERNELS[key]


def _compiled(n_seg):
    """The chunk kernel for ``n_seg`` segments: ``exp(i g (u . m)) w`` summed over each tile's rows (a fused
    reduction), the tile sums scattered into their segments. The three products are written out: a float32 matmul
    on CUDA is TF32 and moves a 30 rad phase by 0.03. The tile sums and the accumulator are complex128; the cast
    costs nothing measurable and the sums are then exact to the float32 phases."""
    import jax
    import jax.numpy as jnp
    if n_seg not in _KERNELS:
        @jax.jit
        def kernel(m, w, tiles, g, u, acc):
            ph = g[None, None, :] * (m[:, :, 0:1] * u[None, None, :, 0] + m[:, :, 1:2] * u[None, None, :, 1] + m[:, :, 2:3] * u[None, None, :, 2])
            E = (jnp.exp(1j * ph) * w[:, :, None]).astype(jnp.complex128)
            return acc + jax.ops.segment_sum(E.sum(1), tiles, num_segments=n_seg)
        _KERNELS[n_seg] = kernel
    return _KERNELS[n_seg]
