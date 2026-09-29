"""A replay pack contracted against the shape of a gradient waveform, once, so that every acquisition that plays
that shape -- at any amplitude, in any direction -- is a three-term phase per walker (dmipy-sim#505: the whole
DiSCo grid replayed while a user waits).

A single-direction waveform is ``G(t) = g s(t) u``: ``s`` its shape (``max |s| = 1``), ``g`` its amplitude (T/m)
and ``u`` its unit direction. The band contraction of a replay (:func:`~dmipy_sim.replay.replay._band_phase`) is
linear in the waveform, so the gradient phase of walker ``w`` is ``g (u . m_w)`` with the **shape moment**
``m_w = gamma int s(t) r_w(t) dt`` (rad per T/m): the pack's bands contracted against the shape's projection on
the three axes, once per shape, through the same ``_prepare`` / ``_band_phase`` route every replay takes. The
layout holds, per shape, ``m`` as ``(n_rows, 3)`` float32; once, the walkers' weights and the voxel of every row
with its split half; and ``manifest.json`` naming the source layout (its manifest's sha256), the bands read and
their truncation error, and every shape's profile with the b-value it encodes at unit amplitude, so that an
amplitude follows from a b-value as ``g = sqrt(b / b_unit)``.

A shape is a one-row single-direction :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence` whose
ramps do not scale with its amplitude (built with ``slew_rate=np.inf``): a finite slew makes the shape a
function of ``g`` and the moment would not be one vector. The image of an acquisition on the layout is
``|sum_w w exp(i g u . m_w)| / sum_w w`` per voxel with its split-half floor, formed on the device in single
precision from device-resident rows: the phase's three products are written out (a float32 matmul on CUDA is
TF32), the exponential and the per-voxel sums are complex64, the voxel totals are combined in double precision
on the host.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

MANIFEST = "manifest.json"
FORMAT = "shape_moments/1"
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


def write_shape_moments(source, shapes, out_dir, *, tol=0.25, chunk_rows=2_000_000, workers=8, progress=None):
    """The shape-moment layout of the columnar pack at ``source`` (a
    :class:`~dmipy_sim.replay.columnar.ColumnarPack`, a directory or ``hf://owner/name/prefix``) for ``shapes``,
    a ``{name: sequence}`` of one-row single-direction sequences, written to ``out_dir`` in one pass over the rows
    at the fewest band groups whose truncation error stays under ``tol`` times the pack's median floor on every
    axis of every shape. ``progress(rows, bytes, seconds)`` is called after every row group. Returns the manifest."""
    from .columnar import ColumnarPack
    from .replay import _band_phase
    from ..acquisition.waveforms import calc_b
    col = source if isinstance(source, ColumnarPack) else ColumnarPack(source, workers=workers)
    if not shapes:
        raise ValueError("no shapes: give at least one {name: sequence}")
    prof = {name: _profile(seq) for name, seq in shapes.items()}
    contractions = _contractions(shapes, prof)
    plans = [col.bands_for(a, tol=tol) for a, _ in contractions]
    K = max(k for k, _ in plans); band_error = max(e for _, e in plans)
    b_unit = {name: float(calc_b(_axis_sequence(shapes[name], prof[name][0]))[0]) for name in shapes}
    os.makedirs(out_dir, exist_ok=True)
    n_rows = col.n_rows; n_vox = int(np.prod(col.grid.shape))
    m = {name: np.lib.format.open_memmap(os.path.join(out_dir, f"m_{name}.npy"), mode="w+", dtype=np.float32, shape=(n_rows, 3))
         for name in shapes}
    w = np.lib.format.open_memmap(os.path.join(out_dir, "w.npy"), mode="w+", dtype=np.float32, shape=(n_rows,))
    seg = np.lib.format.open_memmap(os.path.join(out_dir, "seg.npy"), mode="w+", dtype=np.int32, shape=(n_rows,))
    col.src.bytes_read = col.src.requests = 0; t0 = time.time(); row = 0
    for pk in col.iter_views(chunk_rows=chunk_rows, K=K):
        n = pk.n_walkers
        ijk, _ = col.grid.bin(pk.r0); v = np.ravel_multi_index(ijk.T, col.grid.shape)
        starts = np.flatnonzero(np.r_[True, v[1:] != v[:-1]])
        run = np.repeat(starts, np.diff(np.r_[starts, n]))
        seg[row:row + n] = 2 * v + (np.arange(n) - run) % 2
        for a, names in contractions:
            P = pk._prepare(a, tissue=None, scanner=None, orientation=None, compartment=None)
            if np.any(P["voxel"] != 1.0) or P["pathway"] != 1.0:
                raise ValueError(f"shapes {[n for n, _ in names]} leave an amplitude per measurement (an unbalanced "
                                 "encoding or a stimulated echo); a moment carries a phase only")
            phi = _band_phase(P)                                    # (n, 3 per shape): the one band contraction
            for name, first in names:
                m[name][row:row + n] = phi[:, first:first + 3]
            w[row:row + n] = P["w"]
        row += n
        if progress:
            progress(row, col.src.bytes_read, time.time() - t0)
    if row != n_rows:
        raise RuntimeError(f"the pass read {row} rows of {n_rows}")
    for a in list(m.values()) + [w, seg]:
        a.flush()
    manifest = dict(
        format=FORMAT, n_rows=int(n_rows), K=int(K), band_error=float(band_error), tol=float(tol),
        source=dict(uri=col.uri, manifest_sha256=hashlib.sha256(col.src.text(MANIFEST).encode()).hexdigest(),
                    K=int(col.K), n_rows=int(col.n_rows), floor=float(col.floor),
                    grid=col.meta["fidelity"]["per_voxel"]["grid"], pack=col.meta.get("id")),
        n_voxels=int(n_vox), columns=dict(w="w.npy", seg="seg.npy", **{name: f"m_{name}.npy" for name in shapes}),
        shapes={name: dict(profile=prof[name][0].tolist(), dt=float(shapes[name].dt), n_t=int(len(prof[name][0])),
                           b_unit=b_unit[name], family=getattr(shapes[name], "family", None),
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
    """The shape-moment layout at ``path``, rows memory-mapped from disk: ``manifest``, ``grid``, ``shapes`` (the
    names), and :meth:`image` -- a whole grid under any amplitudes and directions of one shape, on the device.
    Open a Hub layout with :meth:`open`."""

    ROWS = 1 << 19                                                      # rows per device chunk
    MEAS = 64                                                           # measurements padded to a multiple

    def __init__(self, path):
        from ..phantom.grid import Grid
        self.path = path
        with open(os.path.join(path, MANIFEST)) as f:
            self.manifest = json.load(f)
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"{path} is not a shape-moment layout ({self.manifest.get('format')!r})")
        self.grid = Grid.from_meta(self.manifest["source"]["grid"])
        self.n_rows = int(self.manifest["n_rows"]); self.n_vox = int(np.prod(self.grid.shape))
        self.shapes = [n for n in self.manifest["columns"] if n not in ("w", "seg")]
        self._m = {}; self._device = {}

    @staticmethod
    def open(uri, *, shapes=None, revision=None, workers=8):
        """The layout at ``uri``: a directory, or ``hf://owner/name/prefix`` fetched into the Hub cache (the
        manifest, the shared rows and the moments of ``shapes``, every shape when None) at ``revision``."""
        if not uri.startswith("hf://"):
            return ShapeMoments(uri)
        from huggingface_hub import snapshot_download, hf_hub_download
        owner, name, prefix = uri[5:].split("/", 2); repo = f"{owner}/{name}"
        mf = hf_hub_download(repo, f"{prefix}/{MANIFEST}", repo_type="dataset", revision=revision)
        with open(mf) as f:
            cols = json.load(f)["columns"]
        want = [cols["w"], cols["seg"], MANIFEST] + [cols[s] for s in (shapes if shapes is not None else cols) if s not in ("w", "seg")]
        local = snapshot_download(repo, repo_type="dataset", revision=revision, max_workers=workers,
                                  allow_patterns=[f"{prefix}/{f}" for f in want])
        return ShapeMoments(os.path.join(local, prefix))

    def _column(self, name):
        if name not in self._m:
            self._m[name] = np.load(os.path.join(self.path, self.manifest["columns"][name]), mmap_mode="r")
        return self._m[name]

    def moments(self, shape):
        """``(n_rows, 3)`` float32, memory-mapped: the moment of every row under ``shape``."""
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
        w = np.asarray(self._column("w"), np.float64); seg = np.asarray(self._column("seg"))
        return np.bincount(seg, w, minlength=2 * self.n_vox).reshape(self.n_vox, 2)

    def _resident(self, shape):
        """The rows on the device, padded to whole chunks: ``(m, w, seg)`` for ``shape`` (shared rows once)."""
        import jax.numpy as jnp
        C = self.ROWS; n_pad = -(-self.n_rows // C) * C
        if "w" not in self._device:
            w = np.zeros(n_pad, np.float32); w[:self.n_rows] = self._column("w")
            seg = np.full(n_pad, 2 * self.n_vox, np.int32); seg[:self.n_rows] = self._column("seg")
            self._device["w"], self._device["seg"] = jnp.asarray(w), jnp.asarray(seg)
        if shape not in self._device:
            m = np.zeros((n_pad, 3), np.float32); m[:self.n_rows] = self.moments(shape)
            self._device[shape] = jnp.asarray(m)
        return self._device[shape], self._device["w"], self._device["seg"]

    def image(self, shape, bvalues, directions):
        """``(S, floor)`` of the grid under ``shape`` at ``bvalues`` (s/m^2) along ``directions`` (unit vectors):
        ``S`` is ``grid.shape + (n_meas,)``, the weighted ensemble magnitude per voxel (NaN where the layout has no
        rows), ``floor`` ``grid.shape`` its split-half floor ``max_m |S_a - S_b| / 2``. A b = 0 row's direction
        is immaterial."""
        import jax
        import jax.numpy as jnp
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
        m, w, seg = self._resident(shape)
        n_seg = 2 * self.n_vox + 1
        acc = jnp.zeros((n_seg, M_pad), jnp.complex64); kernel = _compiled(n_seg)
        g_d, u_d = jnp.asarray(g_p), jnp.asarray(u_p)
        for i in range(0, m.shape[0], self.ROWS):
            acc = kernel(m[i:i + self.ROWS], w[i:i + self.ROWS], seg[i:i + self.ROWS], g_d, u_d, acc)
        num = np.asarray(acc, np.complex128)[:2 * self.n_vox, :M].reshape(self.n_vox, 2, M)
        den = self.weights
        S = np.full((self.n_vox, M), np.nan); floor = np.full(self.n_vox, np.nan)
        any_ = den.sum(1) > 0; both = (den > 0).all(1)
        S[any_] = np.abs(num[any_].sum(1) / den[any_].sum(1)[:, None])
        Sa = np.abs(num[both, 0] / den[both, 0][:, None]); Sb = np.abs(num[both, 1] / den[both, 1][:, None])
        floor[both] = 0.5 * np.abs(Sa - Sb).max(1)
        return S.reshape(tuple(self.grid.shape) + (M,)), floor.reshape(self.grid.shape)


_KERNELS = {}


def _compiled(n_seg):
    """The chunk kernel for ``n_seg`` segments: ``exp(i g (u . m)) w`` summed into each row's segment, added to the
    accumulator. The three products are written out: a float32 matmul on CUDA is TF32 and moves a 30 rad phase by
    0.03."""
    import jax
    import jax.numpy as jnp
    if n_seg not in _KERNELS:
        @jax.jit
        def kernel(m, w, seg, g, u, acc):
            ph = g[None, :] * (m[:, 0:1] * u[None, :, 0] + m[:, 1:2] * u[None, :, 1] + m[:, 2:3] * u[None, :, 2])
            E = jnp.exp(1j * ph) * w[:, None]
            return acc + jax.ops.segment_sum(E, seg, num_segments=n_seg)
        _KERNELS[n_seg] = kernel
    return _KERNELS[n_seg]
