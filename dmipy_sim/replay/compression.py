"""IR-basis compression of a master walk — the replay representation, compressed.

A Monte-Carlo master walk is a large object (N_w walkers x N_t steps x 3). Every
replay observable is a functional of the stored walk, and the walk compresses along
two physical redundancies:

  * TEMPORAL — a deliverable gradient waveform is band-limited, so only the low-order
    temporal content of a path is ever probed. ``bridge_dst`` -- the only C0 representation --
    splits each path into its two exact endpoints and a residual pinned at both, keeping the
    lowest ``K`` sine bands of the residual: lossless for any acquisition inside that band, and
    it PRESERVES walker identity (so the per-walker relaxation / boundary / MT channels stay
    aligned and replay too).
  * ENSEMBLE — walkers are exchangeable samples. ``lowrank`` keeps ``K`` Karhunen-
    Loeve modes + exact per-walker coefficients (walker-preserving, storage ~ N_w).
    ``gaussian`` / ``marginal`` store only a coefficient DISTRIBUTION and resample at
    replay (walker-count-independent, sub-MB) — exact in the Gaussian limit, gradient
    replay only (they do not preserve walker identity).

The physics channels get their own codecs (positions are smooth+low-rank; these are not):

  * ``compartment`` (C1) — named occupancy columns, piecewise-constant per walker -> row
    RLE. The MT bound pool is one of these columns, not a channel of its own.
  * ``boundary_local_time`` (surface relaxivity ell(t), rho/D=1) — two codecs:
      - ``boundary_local_time`` : DENSITY-AWARE sparse-CSR / dense-int8 of the *per-step*
        signal. Lossless to the value quant; but the per-step signal is dense at small R
        (every step hits a wall) so it caps at ~2x there.
      - ``boundary_dct`` (this port's addition): DCT of the *cumulative* boundary time
        B(t)=cumsum(ell). B is an integral -> smooth+monotone -> a handful of modes
        reproduce it (and hence any TE truncation / gate) losslessly to the MC floor,
        at a ratio that GROWS with n_t. This is the codec that makes surface-relaxivity
        replay memory-viable at the fine, high-n_t fidelity we walk once and keep.

``mode_space_signal`` computes the replay signal directly from the compressed basis,
never reconstructing the trajectory. ``auto_select_modes`` picks the smallest K that
meets a fidelity tolerance against the uncompressed walk.

Ported from the private replay-pack pipeline; the fidelity scorer here is self-contained
numpy (phase integral + separable log-weights) so this module depends only on numpy+scipy.
"""
from __future__ import annotations
import logging

log = logging.getLogger(__name__)

import re
import functools
import numpy as np

try:
    from scipy.fft import dst as _dst, idst as _idst
except ImportError as _e:  # pragma: no cover
    raise ImportError("dmipy_sim.replay.compression needs scipy (pip install scipy)") from _e

from ..constants import GAMMA

POSITION_METHOD = "bridge_dst"          # the only C0 representation; see encode_bridge_dst
WALKER_PRESERVING = (POSITION_METHOD,)
ALL_METHODS = WALKER_PRESERVING

_RETIRED = {
    "temporal_dct": "cosine bands of the whole path",
    "lowrank": "KL modes over the flattened path",
    "gaussian": "a fitted coefficient distribution",
    "marginal": "coefficient quantiles",
}

# C2 shares C0's fate: the cumulative local time moved from detrended DCT-II bands
# (``blt_dct_coeffs``) to the bridge form (``blt_bridge_dst`` + ``blt_start``). The array NAME
# changed deliberately -- a stale pack then fails to find its channel instead of decoding sine
# bands as cosine ones and returning a plausible wrong attenuation.
_RETIRED_BOUNDARY = {"dct": "detrended cosine bands of the cumulative local time"}


def require_position_method(method):
    """Raise unless ``method`` is the one C0 representation this build reads.

    There is deliberately no fallback. A retired codec stores different quantities under the
    same tensor names -- ``bridge_dst`` puts the two endpoints where a band codec puts its two
    lowest bands -- so decoding one as the other yields plausible numbers rather than an error.
    Refusing is the only safe response; the pack must be re-encoded from its master.
    """
    if method == POSITION_METHOD:
        return method
    if method in _RETIRED:
        raise ValueError(
            f"position codec {method!r} ({_RETIRED[method]}) is no longer read. Positions are "
            f"stored as {POSITION_METHOD!r}: two exact endpoints followed by sine bands of the "
            f"pinned residual. The first two coefficients per axis are NOT bands, so decoding "
            f"this pack with the current reader would return plausible but wrong values. "
            f"Re-encode the pack from its master with build_replay_pack().")
    if method is None:
        raise ValueError(
            f"pack declares no position codec. It must declare "
            f"compression.method = {POSITION_METHOD!r}; refusing to assume it, since a pack "
            f"written by an older build stores different quantities under the same names.")
    raise ValueError(f"unknown position codec {method!r}; expected {POSITION_METHOD!r}")
_F16, _F32 = 2, 4


# --------------------------------------------------------------------- encoders

#: The default band container of the bridge channels: the first 16 bands at 16 bits, the rest at 8. A bridge's sine
#: coefficients fall as 1/k, so a per-band scale keeps the quantisation step proportional to the band's own range:
#: at 8 bits a band contributes a step of ~1/127 of its range, which for band 17 and beyond of a 100 ms walk is
#: nanometres; the first bands carry the walker's excursion and keep 16 bits.
BAND_CONTAINER = ((16, 16), (None, 8))


def _band_ranges(container, K):
    """``[(k0, k1, bits), ...]`` covering bands ``0 .. K-1`` from a container ``((upto, bits), ...)``; ``upto`` None
    is the end. Refuses a gap, an overlap, or a width other than 8 or 16."""
    out, k0 = [], 0
    for upto, bits in container:
        k1 = int(K) if upto is None else min(int(upto), int(K))
        if int(bits) not in (8, 16):
            raise ValueError(f"band container bits must be 8 or 16 (got {bits})")
        if k1 > k0:
            out.append((k0, k1, int(bits)))
        k0 = max(k0, k1)
    if k0 < int(K):
        raise ValueError(f"band container covers bands up to {k0} of {K}; end it with (None, bits)")
    return out


def quantise_bands(B, container, key, scale_key):
    """The integer container of sine bands ``B`` (``(N_w, K)`` or ``(N_w, K, n_axes)``): per band (and axis) a
    scale ``max |B| / (2^(bits-1) - 1)`` and the rounded integers, one tensor per band range at its width, under
    ``{key}_b{i}``; the scales under ``scale_key`` as ``(1, K)`` or ``(1, n_axes, K)`` float32 -- the leading axis is
    the walker BLOCK: a merged pack stacks its shards' scales there and carries ``band_block`` ``(N_w,)``, the
    block of each walker (absent: block 0). Returns ``(arrays, meta)`` with ``meta["container"]`` the ranges."""
    B = np.asarray(B, np.float64)
    K = B.shape[1]
    ranges = _band_ranges(container, K)
    amax = np.abs(B).max(axis=0)                                      # (K,) or (K, n_axes)
    arrays, cont = {}, []
    scale = np.zeros(amax.shape, np.float64)
    for i, (k0, k1, bits) in enumerate(ranges):
        lim = 2 ** (bits - 1) - 1
        scale[k0:k1] = np.maximum(amax[k0:k1], 1e-300) / lim
        q = np.rint(B[:, k0:k1] / scale[None, k0:k1]).astype(np.int16 if bits == 16 else np.int8)
        arrays[f"{key}_b{i}"] = np.ascontiguousarray(q)
        cont.append({"bands": [int(k0), int(k1)], "bits": int(bits)})
    arrays[scale_key] = np.ascontiguousarray((scale.T if scale.ndim == 2 else scale)[None]).astype(np.float32)
    return arrays, {"container": cont}


def band_scales(arrays, scale_key, n_w):
    """The per-walker scale of every band, ``(N_w, K)`` or ``(N_w, n_axes, K)``: the block table indexed by each
    walker's block (``band_block``, absent: one block)."""
    table = np.asarray(arrays[scale_key], np.float64)
    if "band_block" in arrays:
        return table[np.asarray(arrays["band_block"], np.int64)]
    if table.shape[0] != 1:
        raise KeyError(f"{scale_key} holds {table.shape[0]} blocks but the pack carries no band_block")
    return np.broadcast_to(table[0], (int(n_w),) + table.shape[1:])


def dequantise_bands(arrays, container, key, scale_key, dtype=np.float64):
    """The bands back as ``(N_w, K)`` or ``(N_w, K, n_axes)`` from :func:`quantise_bands`' tensors."""
    q0 = np.asarray(arrays[f"{key}_b0"])
    scale = band_scales(arrays, scale_key, q0.shape[0])               # (N_w, K) or (N_w, n_axes, K)
    scale = np.swapaxes(scale, 1, 2) if scale.ndim == 3 else scale    # -> (N_w, K[, n_axes])
    parts = [np.asarray(arrays[f"{key}_b{i}"], np.float64) * scale[:, r["bands"][0]:r["bands"][1]]
             for i, r in enumerate(container)]
    return np.concatenate(parts, axis=1).astype(dtype)


DEVICES = ("auto", "numpy", "jax")


def _jax_has_gpu():
    try:
        import jax
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:                                                # no jax, or no backend
        return False


def resolve_device(device):
    """``"numpy"`` or ``"jax"`` from a device word: ``"auto"`` is the JAX device when one is a GPU, else numpy
    (a CPU JAX transform is no faster than scipy's)."""
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, not {device!r}")
    if device == "auto":
        return "jax" if _jax_has_gpu() else "numpy"
    return device


CHUNK_BYTES = 1 << 30
"""Bytes of the walk a host pass holds at once: every codec and certificate pass over the walkers runs in chunks
of at most this many bytes of positions, so the builder's peak memory is the walk plus this, not a multiple of
the walk."""


def _chunk(chunk_bytes):
    return int(CHUNK_BYTES if chunk_bytes is None else chunk_bytes)


def dst_bands(u, K, *, device="auto", chunk_bytes=None):
    """The lowest ``K`` orthonormal DST-I bands along axis 1 of ``u`` ``(N_w, N, ...)``, as float64: scipy on the
    host, or on the JAX device as one matmul against the sine matrix at full precision (``Precision.HIGHEST``:
    a float32 ``@`` on a CUDA device is TF32 otherwise), in walker chunks of ``chunk_bytes``."""
    K = int(K)
    if resolve_device(device) == "numpy":
        return np.asarray(_dst(np.asarray(u, np.float64), axis=1, type=1, norm="ortho")[:, :K], np.float64)
    import jax
    import jax.numpy as jnp
    N = int(u.shape[1])
    n = np.arange(1, N + 1)[:, None]; k = np.arange(1, K + 1)[None, :]
    S = jnp.asarray(np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * n * k / (N + 1)), jnp.float32)          # (N, K)
    f = jax.jit(lambda x: jnp.einsum("wn...,nk->wk...", x, S, precision=jax.lax.Precision.HIGHEST))
    rows = max(1, int(_chunk(chunk_bytes) // max(int(np.prod(u.shape[1:])) * 4, 1)))
    out = np.empty((u.shape[0], K) + tuple(u.shape[2:]), np.float64)
    for i in range(0, u.shape[0], rows):
        out[i:i + rows] = np.asarray(f(jnp.asarray(np.asarray(u[i:i + rows], np.float32))), np.float64)
    return out


def dct_bands(u, K, *, device="auto", chunk_bytes=None):
    """The lowest ``K`` orthonormal DCT-II bands along axis 1 of ``u`` ``(N_w, N, ...)``, as float64 -- the cosine
    twin of :func:`dst_bands` for the path field channel: scipy on the host, or one full-precision matmul against
    the cosine matrix on the JAX device, in walker chunks."""
    K = int(K)
    if resolve_device(device) == "numpy":
        from scipy.fft import dct
        return np.asarray(dct(np.asarray(u, np.float64), type=2, norm="ortho", axis=1)[:, :K], np.float64)
    import jax
    import jax.numpy as jnp
    N = int(u.shape[1])
    n = np.arange(N)[:, None]; k = np.arange(K)[None, :]
    Cm = np.sqrt(2.0 / N) * np.cos(np.pi * (2 * n + 1) * k / (2 * N)); Cm[:, 0] /= np.sqrt(2.0)
    Cd = jnp.asarray(Cm, jnp.float32)                                                                 # (N, K)
    f = jax.jit(lambda x: jnp.einsum("wn...,nk->wk...", x, Cd, precision=jax.lax.Precision.HIGHEST))
    rows = max(1, int(_chunk(chunk_bytes) // max(int(np.prod(u.shape[1:])) * 4, 1)))
    out = np.empty((u.shape[0], K) + tuple(u.shape[2:]), np.float64)
    for i in range(0, u.shape[0], rows):
        out[i:i + rows] = np.asarray(f(jnp.asarray(np.asarray(u[i:i + rows], np.float32))), np.float64)
    return out


def coded_phases(C, dt, G, n_t, *, device="auto", chunk_bytes=None):
    """``(N_w, n_meas)`` gradient phase of every walker under the waveforms ``G`` ``(n_meas, n_t, 3)`` from the
    bridge coefficients ``C`` ``(N_w, K+2, 3)`` alone: ``gamma dt sum C W`` with ``W`` the bridge projection of the
    waveforms' exact per-save weights (:func:`_replay_kernel.effective_gradient`) -- the replay's own reading of a
    pack, no path decoded (host float64, or the device in float32 at full precision)."""
    from ._replay_kernel import effective_gradient
    C = np.asarray(C); K = int(C.shape[1]) - 2
    Geff = effective_gradient(np.asarray(G, np.float64), float(dt), int(n_t), float(dt))              # per-save weights
    W = bridge_projection(Geff, int(n_t), K)                                                         # (n_meas, K+2, 3)
    W2 = (GAMMA * float(dt)) * W.reshape(W.shape[0], -1).T                                           # ((K+2)*3, n_meas)
    Cf = C.reshape(C.shape[0], -1)
    if resolve_device(device) == "numpy":
        return np.asarray(Cf, np.float64) @ W2
    import jax
    import jax.numpy as jnp
    Wd = jnp.asarray(W2, jnp.float32)
    f = jax.jit(lambda x: jnp.matmul(x, Wd, precision=jax.lax.Precision.HIGHEST))
    rows = max(1, int(_chunk(chunk_bytes) // max(Cf.shape[1] * 4, 1)))
    out = np.empty((Cf.shape[0], W2.shape[1]), np.float64)
    for i in range(0, Cf.shape[0], rows):
        out[i:i + rows] = np.asarray(f(jnp.asarray(np.asarray(Cf[i:i + rows], np.float32))), np.float64)
    return out


def measure_floor_coded(C, dt, n_t, env=None, *, w=None, logw=None, device="auto"):
    """The split-half Monte-Carlo floor of a walk over the envelope's battery, read from its bridge coefficients
    (:func:`coded_phases`): the same ensemble split :func:`measure_fidelity` uses, no path decoded. Returns
    ``floor_max``, ``noise_floor`` and the floor per family."""
    env = env or default_envelope()
    G, meta = acquisition_battery(int(n_t), float(dt), env)
    phi = coded_phases(C, dt, G, n_t, device=device)
    nw = phi.shape[0]
    ww = np.ones(nw) if w is None else np.asarray(w, float)
    lw = np.zeros(nw) if logw is None else np.asarray(logw, float)
    e = np.exp(lw[:, None] + 1j * phi) * ww[:, None]
    idx = np.random.default_rng(0).permutation(nw); h = nw // 2
    ia, ib = idx[:h], idx[h:]
    Sa = e[ia].sum(0) / ww[ia].sum(); Sb = e[ib].sum(0) / ww[ib].sum()
    floor = np.abs(Sa - Sb) / 2.0
    fams = sorted({m["fam"] for m in meta})
    per_fam = {f: float(floor[[i for i, m in enumerate(meta) if m["fam"] == f]].max()) for f in fams}
    return dict(floor_max=float(floor.max()), noise_floor=float(1 / np.sqrt(nw)), per_family=per_fam)


def encode_bridge_dst(X, K, container=None, *, device="auto"):
    """Endpoints plus a Brownian bridge, expanded on the sine basis (per axis).

    Splits each path the way the gradient phase reads it -- into the two endpoints and a
    residual pinned at both -- and keeps the lowest ``K`` DST-I bands of the residual:

        r(t) = r(0) + (t/T)[r(T) - r(0)] + u(t),    u(0) = u(T) = 0

    Stored per axis as ``[r(0), r(T)-r(0), beta_1..beta_K]``, so the layout stays one tensor per
    spatial axis and the first two entries are the coefficients the gradient moments pair with.

    Three properties, none of which is a compression claim. The gradient phase becomes
    ``r(0).M0 + (r(T)-r(0)).M1/T + sum_k beta_k Ghat_k``, so refocusing and velocity
    compensation annihilate the first two columns exactly.  Both endpoints are held exactly
    rather than to the truncation error, so a stored walk can be continued from where it ended.
    And the sine basis is the variance-optimal one for what remains: the discrete Brownian
    bridge has covariance ``min(m,n) - mn/N`` whose inverse is the Dirichlet Laplacian, so its
    Karhunen-Loeve eigenvectors are exactly the DST-I vectors.

    Accuracy is indistinguishable from ``temporal_dct``, and that is a theorem rather than a
    coincidence: the difference operator maps the cosine basis onto the sine basis exactly,
    ``c_k(n) - c_k(n-1) = -2 sin(pi k / 2N) s_{k-1}(n)``, so a cosine expansion of the path is a
    sine expansion of its increments and the two truncate to the same subspaces.
    """
    X = np.asarray(X)
    Nw, Nt, _ = X.shape
    a = np.asarray(X[:, 0, :], np.float64)
    v = np.asarray(X[:, -1, :], np.float64) - a
    tau = np.arange(Nt) / (Nt - 1.0)
    K = int(min(K, Nt - 2))
    dev = "numpy" if resolve_device(device) == "numpy" else "jax"
    B = np.empty((Nw, K, 3), np.float64); rows = max(1, int(CHUNK_BYTES // max(Nt * 3 * 4, 1)))
    for i in range(0, Nw, rows):                                     # the residual per walker chunk, never the whole walk again
        sl = slice(i, i + rows)
        if dev == "numpy":
            u = np.asarray(X[sl], np.float64) - (a[sl, None, :] + v[sl, None, :] * tau[None, :, None])
        else:
            u = np.asarray(X[sl], np.float32) - np.asarray(a[sl, None, :] + v[sl, None, :] * tau[None, :, None], np.float32)
        B[sl] = dst_bands(u[:, 1:-1, :], K, device=dev)
    C = np.concatenate([a[:, None, :], v[:, None, :], B], axis=1)   # (Nw, K+2, 3)
    meta = {"method": "bridge_dst", "K": K, "n_t": int(Nt)}
    if container is None:                                            # the float32 container, one tensor per axis
        arrays = pack_position_arrays(C, np.float32)
        return arrays, meta, Nw * 3 * (K + 2) * _F32
    arrays = pack_position_arrays(C, np.float32, container=container)
    meta["container"] = [{"bands": [k0, k1], "bits": b} for k0, k1, b in _band_ranges(container, K)]
    return arrays, meta, int(sum(int(np.asarray(v).nbytes) for v in arrays.values()))


def _bridge_positions(C, Nt):
    """Positions ``(n, Nt, 3)`` of the bridge coefficients ``C`` ``(n, K+2, 3)``."""
    a, v, B = C[:, 0, :], C[:, 1, :], C[:, 2:, :]
    tau = np.arange(Nt) / (Nt - 1.0)
    u = np.zeros((C.shape[0], Nt, 3))
    if B.shape[1]:
        u[:, 1:-1, :] = _idst(B, axis=1, type=1, norm="ortho", n=Nt - 2)
    return a[:, None, :] + v[:, None, :] * tau[None, :, None] + u


def decode_bridge_dst(arrays, meta, walkers=None):
    """Reconstruct positions from endpoints plus sine bands, of every walker or of the slice ``walkers``."""
    C = read_position_coeffs(arrays, dtype=np.float64)
    return _bridge_positions(C if walkers is None else C[walkers], int(meta["n_t"]))


def decoder(arrays, meta):
    """``(lo, hi) -> positions (hi - lo, n_t, 3)`` of walkers ``lo:hi``, the coefficients read once: what the
    certificate reads instead of a decoded copy of the whole walk."""
    require_position_method(meta["method"])
    C = read_position_coeffs(arrays, dtype=np.float64); Nt = int(meta["n_t"])
    return lambda lo, hi: _bridge_positions(C[lo:hi], Nt)


def bridge_moment_rows(G, n_t):
    """Sequence-side rows the two endpoint coefficients pair with: ``(M0, M1/T)``.

    ``M0 = sum_n G_n`` is the refocusing condition and ``M1 = sum_n tau_n G_n`` the
    velocity-compensation one, so a motion-compensated waveform makes both vanish.
    """
    G = np.asarray(G, np.float64)
    tau = np.arange(n_t) / (n_t - 1.0)
    return G.sum(1), (G * tau[None, :, None]).sum(1)


def bridge_projection(G, n_t, K):
    """The waveform's projection on the bridge basis, ``(n_meas, K+2, 3)``: the two gradient
    moments (:func:`bridge_moment_rows`) then the ``K`` sine bands ``DST-I(G[1:-1])``. The
    gradient phase of a walk stored as ``C`` (:func:`encode_bridge_dst`) is
    ``gamma dt sum_{k,d} C[w,k,d] W[m,k,d]``; :func:`mode_space_phi` and
    :func:`dmipy_sim.replay.replay.compile_scheme` both read this one projection."""
    G = np.asarray(G, np.float64)
    n_t = int(n_t)
    if G.shape[1] != n_t:
        raise ValueError(f"waveform has {G.shape[1]} samples, pack walk has n_t={n_t}")
    M0, M1 = bridge_moment_rows(G, n_t)
    Ghat = _dst(G[:, 1:-1, :], axis=1, type=1, norm="ortho")[:, :int(K), :]
    return np.concatenate([M0[:, None, :], M1[:, None, :], Ghat], axis=1)






ENCODERS = {POSITION_METHOD: encode_bridge_dst}


# ------------------------------------------------------- run-length row coding
def rle_encode_rows(A):
    """Run-length encode a piecewise-constant integer matrix (Nw, Nt) row-wise into
    three flat arrays (safetensors-friendly)."""
    A = np.asarray(A)
    vals, lens, counts = [], [], np.empty(A.shape[0], np.int32)
    for i, row in enumerate(A):
        chg = np.flatnonzero(np.diff(row)) + 1
        starts = np.concatenate(([0], chg))
        ends = np.concatenate((chg, [row.size]))
        vals.append(row[starts]); lens.append(ends - starts); counts[i] = starts.size
    return (np.concatenate(vals).astype(np.int16),
            np.concatenate(lens).astype(np.int32), counts, int(A.shape[1]))


def rle_decode_rows(vals, lens, counts, n_t):
    out = np.empty((counts.size, n_t), np.int16); p = 0
    for i, c in enumerate(counts):
        out[i] = np.repeat(vals[p:p + c], lens[p:p + c]); p += c
    return out


# ------------------------------------------- per-walker physics-channel codecs
def encode_boundary_local_time(dlog, nlevels=4096):
    """DENSITY-AWARE per-step codec: sparse CSR (isolated fibres) or dense int8 (packed
    WM). Lossless to the value quant. NOTE: caps at ~2x when contacts are dense (small R);
    prefer ``encode_boundary_bridge`` for the smooth cumulative representation."""
    A = np.asarray(dlog, np.float64)
    nw, nt = A.shape
    scale = float(np.abs(A).max()) + 1e-30
    nnz = int(np.count_nonzero(A))
    sparse_bytes = nnz * 4 + nw * 4
    dense_bytes = nw * nt
    if sparse_bytes <= dense_bytes:
        counts = np.empty(nw, np.int32); cols = []; qvals = []
        for i, row in enumerate(A):
            nz = np.flatnonzero(row)
            counts[i] = nz.size; cols.append(nz)
            qvals.append(np.rint(row[nz] / scale * (nlevels - 1)))
        arrays = {
            "blt_counts": counts,
            "blt_cols": (np.concatenate(cols) if len(cols) else np.zeros(0)).astype(
                np.int16 if nt <= 32000 else np.int32),
            "blt_qvals": (np.concatenate(qvals) if len(qvals) else np.zeros(0)).astype(
                np.int16 if nlevels <= 32000 else np.int32),
        }
        return arrays, {"channel": "boundary_local_time", "mode": "sparse",
                        "n_t": int(nt), "scale": scale, "nlevels": int(nlevels)}
    q = np.rint(np.clip(A / scale, -1.0, 1.0) * 127.0).astype(np.int8)
    return {"blt_dense_q": q}, {"channel": "boundary_local_time", "mode": "dense",
                                "n_t": int(nt), "scale": scale, "nlevels": 127}


def decode_boundary_local_time(arrays, meta, walkers=None):
    """Per-save ell(t) of every walker, or of the slice ``walkers``, from the quantised container."""
    nt = int(meta["n_t"]); scale = float(meta["scale"]); nl = int(meta["nlevels"])
    sl = slice(None) if walkers is None else walkers
    if meta.get("mode") == "dense" or "blt_dense_q" in arrays:
        q = np.asarray(np.asarray(arrays["blt_dense_q"])[sl], np.float64)
        return (q * scale / nl).astype(np.float32)
    counts = np.asarray(arrays["blt_counts"]); cols = np.asarray(arrays["blt_cols"])
    qvals = np.asarray(arrays["blt_qvals"], np.float64)
    lo, hi, _ = sl.indices(counts.size)
    out = np.zeros((hi - lo, nt), np.float32); p = int(counts[:lo].sum())
    for i, c in enumerate(counts[lo:hi]):
        c = int(c)
        out[i, cols[p:p + c]] = (qvals[p:p + c] * scale / (nl - 1)).astype(np.float32)
        p += c
    return out


def encode_boundary_bridge(dlog, K=16, dtype=np.float32, container=None, *, device="auto"):
    """Bridge codec for the boundary-local-time channel: the CUMULATIVE local time B(t)=cumsum(ell)
    stored as its two exact endpoints plus SINE bands of the pinned residual -- the same form C0
    uses for positions, on the same segment grid.

    ell(t) is spiky and dense; its running integral B(t) is smooth and ~linear (roughly constant
    contact rate). Detrending by the chord B(0) + tau*(B(T)-B(0)) leaves a residual vanishing at
    BOTH ends, which is what makes DST-I the basis rather than DCT-II.

    **The basis is chosen for exactness, not for error.** Measured on a reflecting slab (4000
    walkers, n_t=1024), worst-over-t error in B(t) is a wash -- DST-I beats DCT-II by only 1.0-1.3x
    at K=4..64, and the coefficient slopes are indistinguishable (-0.96 vs -0.99). What differs is
    the endpoint. A truncated DCT-II residual does NOT vanish at t=T, so the reconstruction misses
    the stored total by 1.6%-6.4% of B(T) at EVERY K; the sine form is identically zero there by
    construction, so B(0) and B(T) come back exact at every K:

        K                  4        8        16       32       64
        DCT-II endpoint    6.4e-2   4.5e-2   3.1e-2   2.5e-2   1.6e-2
        DST-I  endpoint    0        0        0        0        0

    That is the property the segment algebra needs. Splitting into S segments and chaining their
    endpoints drifts by ~1.5e-2 under DCT-II regardless of S, and by <1e-16 under the sine form --
    so a pack can be cut at a segment boundary, or two segments merged, without the surface channel
    accumulating error the way a band codec does. ``blt_endpoint`` remains the exact total B(T) that
    the ungated rho attenuation reads directly.

    Stores (N_w, K) sine bands + two floats per walker; ratio ~ n_t/K, which GROWS with walk length.
    Lossless at K = n_t - 2 (the interior dimension), NOT at K = n_t.
    """
    A = np.asarray(dlog)
    nw, nt = A.shape
    K = int(min(K, nt - 2))
    tau = np.linspace(0.0, 1.0, nt)[None, :]
    a = np.empty(nw); endpoint = np.empty(nw); C = np.empty((nw, K), np.float64)
    rows = max(1, int(CHUNK_BYTES // max(nt * 8, 1)))
    for i in range(0, nw, rows):                               # the cumulative time per chunk (float64), its bands
        sl = slice(i, i + rows)
        B = np.cumsum(np.asarray(A[sl], np.float64), axis=1)   # (rows, n_t) smooth
        a[sl] = B[:, 0]                                        # exact B(0)
        endpoint[sl] = B[:, -1]                                # exact total local time B(T)
        resid = B - (a[sl, None] + (endpoint[sl] - a[sl])[:, None] * tau)   # exactly 0 at BOTH ends
        C[sl] = dst_bands(resid[:, 1:-1], K, device=device)
    # ``dtype`` sets the band precision; packs pass f16 via build_replay_pack's ``blt_dtype``. The
    # two ENDPOINTS are always f32 -- they are the exact quantities the rho attenuation and the
    # segment chaining read, where f16's ~3 significant digits would be a real error, not a rounding.
    meta = {"channel": "boundary_local_time", "mode": "bridge_dst", "n_t": int(nt), "K": int(K),
            "dtype": np.dtype(dtype).name}
    arrays = {"blt_start": a.astype(np.float32), "blt_endpoint": endpoint.astype(np.float32)}
    if container is None:
        arrays["blt_bridge_dst"] = C.astype(dtype)
    else:                                                          # the integer container: bands per range, a scale per band
        q, qm = quantise_bands(C, container, "blt", "blt_band_scale")
        arrays.update(q); meta["container"] = qm["container"]; meta["dtype"] = "bands"
    return arrays, meta


# --------------------------------------------------------- the bridge on the device
@functools.lru_cache(maxsize=None)
def _device_bridge(n_t, K):
    """The jitted bridge of a device batch of ``n_t`` saves at ``K`` bands: ``(positions (b, n_t, 3)) -> (b, K+2, 3)``
    and ``(local time (b, n_t)) -> (start (b,), endpoint (b,), bands (b, K))``, both float32, the bands as one
    matmul against the DST-I matrix at ``Precision.HIGHEST`` (a float32 ``@`` on a CUDA device is TF32 otherwise)."""
    import jax
    import jax.numpy as jnp
    N = n_t - 2
    n = np.arange(1, N + 1)[:, None]; k = np.arange(1, K + 1)[None, :]
    S = jnp.asarray(np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * n * k / (N + 1)), jnp.float32)      # (N, K)
    tau = jnp.asarray(np.arange(n_t) / (n_t - 1.0), jnp.float32)
    hi = jax.lax.Precision.HIGHEST

    @jax.jit
    def positions(pos):
        pos = pos.astype(jnp.float32)
        a = pos[:, 0, :]; v = pos[:, -1, :] - a
        u = pos - (a[:, None, :] + v[:, None, :] * tau[None, :, None])                          # 0 at both ends
        bands = jnp.einsum("wnd,nk->wkd", u[:, 1:-1, :], S, precision=hi)
        return jnp.concatenate([a[:, None, :], v[:, None, :], bands], axis=1)

    @jax.jit
    def local_time(dlog):
        B = jnp.cumsum(dlog.astype(jnp.float32), axis=1)                                          # a tree scan: no drift
        a = B[:, 0]; e = B[:, -1]
        u = B - (a[:, None] + (e - a)[:, None] * tau[None, :])
        return a, e, jnp.matmul(u[:, 1:-1], S, precision=hi)

    return positions, local_time


def bridge_coefficients_device(pos, K):
    """The C0 coefficients ``(b, K+2, 3)`` float32 of a batch of positions that is on the device: what
    :func:`encode_bridge_dst` computes, formed where the batch is so that only the coefficients cross to the
    host (dmrai-lab/dmipy-sim#446). ``K`` is capped at ``n_t - 2`` as the host encoder caps it."""
    n_t = int(pos.shape[1]); K = int(min(K, n_t - 2))
    return np.asarray(_device_bridge(n_t, K)[0](pos), np.float32), K


def boundary_coefficients_device(dlog, K):
    """The C2 coefficients of a batch of per-save local time on the device: ``(start (b,), endpoint (b,),
    bands (b, K))`` float32, what :func:`encode_boundary_bridge` computes, formed where the batch is."""
    n_t = int(dlog.shape[1]); K = int(min(K, n_t - 2))
    a, e, bands = _device_bridge(n_t, K)[1](dlog)
    return np.asarray(a, np.float32), np.asarray(e, np.float32), np.asarray(bands, np.float32)


def has_c2(arrays):
    """Whether the arrays carry the C2 bridge in either container."""
    return "blt_bridge_dst" in arrays or "blt_b0" in arrays


def has_c1(arrays):
    """Whether the arrays carry the C1 ``comp`` column in either form (runs or the static label)."""
    return "comp_rle_vals" in arrays or "comp_static" in arrays


def c2_bands_K(arrays, meta):
    """The number of C2 bands stored, from the metadata or the tensors."""
    if meta and meta.get("K") is not None:
        return int(meta["K"])
    if "blt_bridge_dst" in arrays:
        return int(np.asarray(arrays["blt_bridge_dst"]).shape[1])
    return int(sum(np.asarray(arrays[k]).shape[1] for k in arrays if re.fullmatch(r"blt_b\d+", k)))


def bridge_bands(arrays, meta, key="blt", scale_key="blt_band_scale", dtype=np.float64):
    """The C2 bands ``(N_w, K)`` from either container: the float tensor ``blt_bridge_dst`` or the integer ranges
    ``blt_b<i>`` with ``blt_band_scale`` (``meta["container"]``)."""
    if "blt_bridge_dst" in arrays:
        return np.asarray(arrays["blt_bridge_dst"], dtype)
    if meta.get("container") is None:
        raise KeyError("this pack's C2 channel is in the integer container but its metadata carries no 'container'")
    return dequantise_bands(arrays, meta["container"], key, scale_key, dtype)


def decode_boundary_bridge(arrays, meta, walkers=None):
    """Reconstruct per-save ell(t) = diff(B) from the two endpoints + the pinned sine bands, of every walker
    or of the slice ``walkers``."""
    nt = int(meta["n_t"])
    sl = slice(None) if walkers is None else walkers
    C = bridge_bands(arrays, meta)[sl]
    a = np.asarray(arrays["blt_start"], np.float64)[sl]
    endpoint = np.asarray(arrays["blt_endpoint"], np.float64)[sl]
    tau = np.linspace(0.0, 1.0, nt)[None, :]
    u = np.zeros((C.shape[0], nt), np.float64)
    u[:, 1:-1] = _idst(C, axis=1, type=1, norm="ortho", n=nt - 2)
    B = u + (a[:, None] + (endpoint - a)[:, None] * tau)
    ell = np.diff(B, axis=1, prepend=B[:, :1] * 0.0)          # per-save increments
    return ell.astype(np.float32)


# C1 is a set of named OCCUPANCY COLUMNS, not a single label track. At each save a walker has an
# occupancy over the declared pools summing to 1, and replay weights the per-pool rates by it:
# R(t) = sum_c f_c(t) R_c. Every case is that one form -- an integer compartment label is one-hot,
# a permeable crossing is a fraction on the geometric axis, and MT binding is a fraction on an
# INDEPENDENT axis (a walker can be intra-axonal and bound). Storing the axes as separate columns
# is the compact encoding of the joint simplex: f_bound = b, f_geom = (1-b)*[comp == geom].
#
# This is why magnetization transfer needs no channel of its own. The bound pool is a column here;
# what makes MT a distinct capability tier is the REPLAY side -- RF as vector-Bloch rotations, the
# bound-pool knobs, and the equilibrium-start requirement -- not the storage. Compare C0 and C2,
# which now share a byte-identical bridge layout and remain separate tiers for the same reason.
_EXCLUSIVE_COLUMN = "comp"          # the geometric axis: mutually exclusive, may be an int label


def _encode_occupancy_column(x, name, Q, force_runs=False):
    """One C1 column -> RLE arrays under ``{name}_rle_*`` + its descriptor. ``force_runs`` stores a track no walker
    crosses in as runs rather than the static label: a window of a walk whose other windows cross.

    Integer labels on the exclusive axis are RLE'd losslessly; everything else is an occupancy in
    [0, scale] quantized to Q levels then RLE'd (integer RLE would lossily int-cast the fractions).
    Occupancy is piecewise constant with long runs either way, which is what makes RLE the codec --
    and the run lengths are physically meaningful: on the bound column they ARE the dwell times, so
    C1 carries the realised exchange statistics rather than a summarising mean rate.
    """
    A = np.asarray(x)
    if name == _EXCLUSIVE_COLUMN and not force_runs and np.array_equal(A, np.round(A)) and A.ndim == 2 and (A == A[:, :1]).all():
        # nothing crosses: one label per walker, the static column (an impermeable substrate)
        return ({f"{name}_static": A[:, 0].astype(np.int8)}, {"name": name, "kind": "static", "n_t": int(A.shape[1])})
    if name == _EXCLUSIVE_COLUMN and np.array_equal(A, np.round(A)):
        vals, lens, counts, n_t = rle_encode_rows(A.astype(np.int32))
        return ({f"{name}_rle_vals": vals.astype(np.int16),
                 f"{name}_rle_lens": lens.astype(np.int32),
                 f"{name}_rle_counts": np.asarray(counts, np.int32)},
                {"name": name, "kind": "label", "n_t": int(n_t)})
    scale = float(np.nanmax(np.abs(A))) or 1.0
    q = np.rint(np.clip(A, 0.0, scale) / scale * (Q - 1)).astype(np.int32)
    vals, lens, counts, n_t = rle_encode_rows(q)
    return ({f"{name}_rle_vals": vals.astype(np.uint8 if Q <= 256 else np.int32),
             f"{name}_rle_lens": lens.astype(np.uint16 if n_t < 65535 else np.int32),
             f"{name}_rle_counts": np.asarray(counts, np.int32)},
            {"name": name, "kind": "fraction", "Q": int(Q), "scale": scale, "n_t": int(n_t)})


def _decode_occupancy_column(arrays, d):
    name = d["name"]
    if d["kind"] == "static":
        return np.repeat(np.asarray(arrays[f"{name}_static"], np.int16)[:, None], int(d["n_t"]), axis=1)
    q = rle_decode_rows(np.asarray(arrays[f"{name}_rle_vals"]),
                        np.asarray(arrays[f"{name}_rle_lens"]),
                        np.asarray(arrays[f"{name}_rle_counts"]), int(d["n_t"]))
    if d["kind"] == "label":
        return q.astype(np.int16)
    return (q.astype(np.float32) / float(d["Q"] - 1)) * float(d.get("scale", 1.0))


def encode_occupancy(columns, Q=256, force_runs=False):
    """C1 codec. ``columns`` maps a pool-axis name to its (N_w, N_t) track.

    ``comp`` is the exclusive geometric axis (integer labels, or a fraction for a permeable
    crossing); any other name is an occupancy in [0,1] on an independent axis -- ``bound`` for the
    MT macromolecular pool. Column order is not significant; the descriptors carry the names.
    """
    if _EXCLUSIVE_COLUMN not in columns:
        raise ValueError(f"C1 needs the {_EXCLUSIVE_COLUMN!r} column; got {sorted(columns)}")
    arrays, cols = {}, []
    for name, x in columns.items():
        a, d = _encode_occupancy_column(x, name, Q, force_runs=force_runs)
        arrays.update(a); cols.append(d)
    n_t = {d["n_t"] for d in cols}
    if len(n_t) != 1:
        raise ValueError(f"C1 columns disagree on n_t: {[(d['name'], d['n_t']) for d in cols]}")
    return arrays, {"channel": "compartment", "columns": cols, "n_t": cols[0]["n_t"]}


def decode_occupancy(arrays, meta):
    """-> {name: track}. ``comp`` is int16 labels (or a float fraction); others are float [0,1]."""
    if "columns" not in meta:
        raise ValueError(
            "this pack's C1 metadata predates the occupancy-column layout (no 'columns' key; it "
            "carries the single-track form with 'fractional'/'Q' at the top level). C1 is now a set "
            "of named columns -- the geometric axis under 'comp' plus, for an MT pack, the bound "
            "pool under 'bound' (retiring the separate bound_fraction channel and its bfrac_rle_* "
            "keys). The stored arrays are readable but their meaning is declared differently, so "
            "the pack must be re-encoded from its master rather than reinterpreted here.")
    return {d["name"]: _decode_occupancy_column(arrays, d) for d in meta["columns"]}


def is_current_c1(meta):
    """True if a pack's C1 channel metadata is in the occupancy-column layout."""
    return isinstance(meta, dict) and "columns" in meta


CHANNEL_ENCODERS = {POSITION_METHOD: encode_bridge_dst}
CHANNEL_DECODERS = {"boundary_local_time": decode_boundary_local_time,
                    "boundary_bridge": decode_boundary_bridge,
                    "compartment": decode_occupancy}


def encode(X, method=POSITION_METHOD, K=32, **kw):
    if method not in ENCODERS:
        raise ValueError(f"unknown method {method!r}; choose from {list(ENCODERS)}")
    return ENCODERS[method](X, K, **kw)


# --------------------------------------------------------------------- decoder

# ---------------------------------------------------------------- position layout (axis-addressable)
# Positions are stored as ONE TENSOR PER SPATIAL AXIS -- pos_x/pos_y/pos_z, each (n_walkers, K) --
# rather than a single (n_walkers, K, 3). safetensors gives every tensor its own byte range, so this
# makes an axis subset a contiguous read that COMPOSES with the walker-prefix read used for precision
# tiers: (axes you need) x (walkers you need) = one contiguous range per axis.
#
# The old (n_w, K, 3) layout cannot express this: the spatial axis has stride 1, so selecting one axis
# is a strided gather. Measured on a 174 MiB pack, `get_slice("dct_coeffs")[:, :, 0:1]` took 169.8 ms
# against 16.3 ms to read the WHOLE tensor -- 10x slower than not optimising at all.
#
# Which axes a consumer actually needs is set by the geometry: a slab restricts one direction (the other
# two are free and analytic), a cylinder two, a sphere three. Nothing is deleted -- the archive keeps all
# three, so b-tensor / rotating-waveform users fetch everything and lose nothing.
POSITION_AXES = ("pos_x", "pos_y", "pos_z")


def pack_position_arrays(C, dtype=np.float32, container=None):
    """(n_w, K+2, 3) coefficients -> the position tensors: the float container ``pos_x``, ``pos_y``, ``pos_z`` (each
    ``(n_w, K+2)``), or with ``container`` the integer one -- the two exact endpoints per axis under ``pos_x_ends``
    etc. (float32) and the bands quantised per band under ``pos_x_b<i>`` with the scales in ``pos_band_scale``
    (:func:`quantise_bands`)."""
    C = np.asarray(C)
    if container is None:
        return {POSITION_AXES[i]: np.ascontiguousarray(C[:, :, i]).astype(dtype) for i in range(C.shape[2])}
    out = {f"{POSITION_AXES[i]}_ends": np.ascontiguousarray(C[:, :2, i]).astype(np.float32) for i in range(C.shape[2])}
    q, _ = quantise_bands(C[:, 2:, :], container, "pos", "pos_band_scale")
    for i in range(C.shape[2]):
        for k in [k for k in q if re.fullmatch(r"pos_b\d+", k)]:
            out[f"{POSITION_AXES[i]}{k[3:]}"] = np.ascontiguousarray(q[k][:, :, i])
    out["pos_band_scale"] = q["pos_band_scale"]
    return out


def position_container(arrays):
    """The container the position tensors are in: ``"float"`` (``pos_x`` ...) or ``"bands"`` (``pos_x_ends`` ...)."""
    if POSITION_AXES[0] in arrays:
        return "float"
    if f"{POSITION_AXES[0]}_ends" in arrays:
        return "bands"
    return None


def has_axis_layout(arrays):
    return position_container(arrays) is not None


def read_position_coeffs(arrays, axes=None, dtype=np.float64):
    """Coefficients as (n_w, K, n_axes) from either layout.

    ``axes`` selects spatial components by index (default all present). Reading a subset is the point of
    the layout: pass e.g. ``axes=(0,)`` for a slab or ``(0, 1)`` for a cylinder's transverse plane.
    """
    kind = position_container(arrays)
    if kind is not None:
        suffix = "" if kind == "float" else "_ends"
        present = [i for i, k in enumerate(POSITION_AXES) if f"{k}{suffix}" in arrays]
        want = list(present if axes is None else axes)
        missing = [i for i in want if i not in present]
        if missing:
            raise KeyError(f"pack does not carry position axes {missing} "
                           f"(has {[POSITION_AXES[i] for i in present]}); it was written with a reduced "
                           f"axis set and cannot serve this encoding")
        if kind == "float":
            return np.stack([np.asarray(arrays[POSITION_AXES[i]], dtype) for i in want], axis=2)
        n_w = int(np.asarray(arrays[f"{POSITION_AXES[present[0]]}_ends"]).shape[0])
        scale = band_scales(arrays, "pos_band_scale", n_w)                          # (N_w, n_axes, K)
        ranges = sorted({int(k.split("_b")[1]) for k in arrays if re.fullmatch(f"{POSITION_AXES[present[0]]}_b\\d+", k)})
        cols = []
        for i in want:
            ends = np.asarray(arrays[f"{POSITION_AXES[i]}_ends"], np.float64)
            bands, k0 = [], 0
            for r in ranges:
                q = np.asarray(arrays[f"{POSITION_AXES[i]}_b{r}"], np.float64)
                bands.append(q * scale[:, i, k0:k0 + q.shape[1]]); k0 += q.shape[1]
            cols.append(np.concatenate([ends] + bands, axis=1))
        return np.stack(cols, axis=2).astype(dtype)
    # No legacy (n_w, K, 3) fallback by design. A dataset with two position layouts forces every
    # consumer to carry a compatibility branch, and that branch is where silent errors live -- a reader
    # that guesses the wrong convention returns plausible numbers. Fail loudly instead; convert the pack.
    raise KeyError(
        f"pack has no position axes {POSITION_AXES}; found {sorted(arrays)}. Packs written before the "
        f"axis-per-tensor layout store a single (n_walkers, K, 3) 'dct_coeffs'; re-encode them with "
        f"pack_position_arrays(). This reader does not accept mixed layouts -- a dataset with two "
        f"position layouts forces every consumer to carry a compatibility branch, and that branch is "
        f"where silent errors live.")


def decode(arrays, meta, n_walkers=None, seed=0, walkers=None):
    """Reconstruct positions (n_walkers, n_t, 3) from a pack's stored arrays, or those of the slice ``walkers``."""
    require_position_method(meta["method"])
    return decode_bridge_dst(arrays, meta, walkers)


def rank_of(method=POSITION_METHOD, n_t=None):
    """Number of coefficients per axis at which ``method`` is an exact rewrite of the walk.

``n_t - 2``: ``bridge_dst`` spends two coefficients on the endpoints and expands the pinned residual on the
    remaining ``n_t - 2`` interior samples -- so the representation is exactly rank-preserving,
    not merely close to it.
    """
    require_position_method(method)
    return int(n_t) - 2


def is_lossless_at(method, K, n_t):
    """Whether ``method`` at ``K`` bands reproduces the walk exactly (to storage precision)."""
    return method == POSITION_METHOD and int(K) >= rank_of(method, n_t)


def is_walker_preserving(method):
    return method == POSITION_METHOD


# ------------------------------------------------------------ mode-space replay
def mode_space_phi(arrays, meta, G, dt, n_walkers=None, seed=0, *, dt_wf=None):
    """Gradient phase phi_i (N_w, n_meas) in the compressed basis, WITHOUT reconstructing the trajectory:
    the exact integral of the waveform ``G`` (on its own grid ``dt_wf``, default the pack's ``dt``) against
    the piecewise-linear path through the saves, read through the per-save weights of
    :func:`_replay_kernel.effective_gradient` and the bridge projection. Linear in position, so it commutes
    with the codec: phi = gamma*dt * [r(0).M0 + (r(T)-r(0)).M1 + sum_k beta_k Ghat_k], and a
    motion-compensated waveform zeroes the first two columns exactly.
    """
    from ._replay_kernel import effective_gradient
    require_position_method(meta.get("method"))
    C = read_position_coeffs(arrays, dtype=np.float64)              # (N_w, K+2, n_axes)
    n_t = int(meta["n_t"])
    Geff = effective_gradient(G, dt if dt_wf is None else dt_wf, n_t, dt)
    W = bridge_projection(Geff, n_t, C.shape[1] - 2)                # (n_meas, K+2, 3)
    return (GAMMA * dt) * np.einsum("wkd,mkd->wm", C, W)


def mode_space_signal(arrays, meta, G, dt, logw=None, weights=None,
                      n_walkers=None, seed=0):
    """Mode-space signal S = <w exp(logw + i phi)> (n_meas,) complex, computing phi
    (mode_space_phi) and the exp reduction, never reconstructing the trajectory. `logw`
    (N_w,) carries the separable relaxation/surface log-weight (see *_logweight)."""
    phi = mode_space_phi(arrays, meta, G, dt, n_walkers=n_walkers, seed=seed)
    n_w = phi.shape[0]
    lw = np.zeros(n_w) if logw is None else np.asarray(logw, float)
    w = np.ones(n_w) if weights is None else np.asarray(weights, float)
    return (np.exp(lw[:, None] + 1j * phi) * (w / w.sum())[:, None]).sum(0)


def relaxation_logweight(comp, T2_per_comp, T1_per_comp, dt, chi=None, active=None):
    """Per-walker relaxation log-weight from the compartment channel -- O(N_w N_t), no trajectory. ``comp`` is
    integer labels OR fractional 2-compartment occupancy, one per save; a save's occupancy is accumulated over the
    step that ends at it, so the first save (which ends no step) relaxes nothing and the whole walk relaxes over
    ``(n_t - 1) dt``. With ``chi`` (:func:`~dmipy_sim.replay._replay_kernel.bin_gate`, 0 at the first save) the
    transverse periods relax at T2 and the longitudinal ones at T1, the longitudinal periods being ``active - chi``
    with ``active`` the acquisition's own accumulation gate (its ones through ``bin_gate``): a stored period of a
    stimulated echo relaxes at T1, the walk beyond the echo relaxes at nothing; a gate without its extent is
    refused, since nothing in ``chi`` says where the acquisition ends."""
    comp = np.asarray(comp)
    invT2 = np.where(np.asarray(T2_per_comp) > 0, 1.0 / np.maximum(np.asarray(T2_per_comp, float), 1e-30), 0.0)
    invT1 = np.where(np.asarray(T1_per_comp) > 0, 1.0 / np.maximum(np.asarray(T1_per_comp, float), 1e-30), 0.0)
    if np.issubdtype(comp.dtype, np.floating) and not np.array_equal(comp, np.round(comp)):
        f = np.clip(comp, 0.0, 1.0)
        r2 = (1.0 - f) * invT2[0] + f * invT2[1]
        r1 = (1.0 - f) * invT1[0] + f * invT1[1]
    else:
        ci = comp.astype(np.int64); r2 = invT2[ci]; r1 = invT1[ci]
    if chi is None:
        return -dt * r2[:, 1:].sum(1)
    if active is None:
        raise ValueError("relaxation_logweight: a gate `chi` needs the acquisition's own extent `active` (bin_gate of its "
                         "ones) -- the longitudinal periods are `active - chi`; without it T1 would act over the walk "
                         "beyond the echo")
    chi = np.asarray(chi, float)[None, :]
    lon = np.asarray(active, float).reshape(1, -1) - chi
    return -dt * (chi * r2 + np.clip(lon, 0.0, None) * r1).sum(1)


def _run_bounds(lens, counts, n_t):
    """The walker and the save interval ``[k0, k1)`` of every run of a walker-major RLE, from the run lengths and
    the runs per walker (every walker's runs cover its ``n_t`` saves)."""
    lens = np.asarray(lens, np.int64); counts = np.asarray(counts, np.int64)
    run_w = np.repeat(np.arange(counts.size), counts)
    k1 = np.cumsum(lens) - run_w * int(n_t)
    return run_w, k1 - lens, k1


def relaxation_logweight_runs(arrays, column, T2_per_comp, T1_per_comp, dt, chi=None, active=None):
    """:func:`relaxation_logweight` on the stored runs, never on a decoded track: the per-walker log-weight is a
    sum over the walker's runs of the run's rate times the gate's on-time within it, ``dt * r * (Cg[k1] - Cg[k0])``
    with ``Cg`` the gate's prefix sum over the saves. O(runs) -- one run per walker on an impermeable substrate.
    ``column`` is the ``comp`` column's descriptor (``kind`` label or fraction); the gates as in
    :func:`relaxation_logweight`: without ``chi`` the walk relaxes over its ``(n_t - 1) dt`` at T2, and with it the
    longitudinal periods are ``active - chi``."""
    name = column["name"]; n_t = int(column["n_t"])
    if column["kind"] == "static":                                                 # one run per walker
        lab = np.asarray(arrays[f"{name}_static"])
        vals, lens, counts = lab, np.full(lab.shape[0], n_t, np.int64), np.ones(lab.shape[0], np.int64)
    else:
        vals = np.asarray(arrays[f"{name}_rle_vals"]); lens = np.asarray(arrays[f"{name}_rle_lens"]); counts = np.asarray(arrays[f"{name}_rle_counts"])
    invT2 = np.where(np.asarray(T2_per_comp) > 0, 1.0 / np.maximum(np.asarray(T2_per_comp, float), 1e-30), 0.0)
    invT1 = np.where(np.asarray(T1_per_comp) > 0, 1.0 / np.maximum(np.asarray(T1_per_comp, float), 1e-30), 0.0)
    if column["kind"] in ("label", "static"):
        v = vals.astype(np.int64); r2 = invT2[v]; r1 = invT1[v]
    else:
        f = np.clip(vals.astype(np.float64) / float(column["Q"] - 1) * float(column.get("scale", 1.0)), 0.0, 1.0)
        r2 = (1.0 - f) * invT2[0] + f * invT2[1]; r1 = (1.0 - f) * invT1[0] + f * invT1[1]
    if chi is None:
        chi = np.ones(n_t); chi[0] = 0.0                                          # the first save ends no step
        lon = np.zeros(n_t)
    else:
        if active is None:
            raise ValueError("relaxation_logweight_runs: a gate `chi` needs the acquisition's own extent `active`")
        chi = np.asarray(chi, np.float64).reshape(-1)[:n_t]
        lon = np.clip(np.asarray(active, np.float64).reshape(-1)[:n_t] - chi, 0.0, None)
    Cchi = np.concatenate([[0.0], np.cumsum(chi)]); Clon = np.concatenate([[0.0], np.cumsum(lon)])
    run_w, k0, k1 = _run_bounds(lens, counts, n_t)
    per_run = r2 * (Cchi[k1] - Cchi[k0]) + r1 * (Clon[k1] - Clon[k0])
    return -float(dt) * np.bincount(run_w, weights=per_run, minlength=counts.size)


def surface_logweight_bridge(arrays, meta, rho_over_D, chi):
    """The gated surface log-weight ``(rho/D) sum_t chi_t ell_t`` from the bridge form itself, without the per-save
    series: with ``ell = diff(B)`` (``ell_0 = B_0``, the start), summation by parts gives ``sum_t d_t B_t`` with
    ``d_t = chi_t - chi_{t+1}`` (``chi_{n_t} = 0``), and on the bridge ``B = a + (e - a) tau + u``, ``u = idst(C)``
    (DST-I, orthonormal, its own inverse), that is ``a sum d + (e - a) sum d tau + C . dst(d[1:-1])``: two scalars
    and one ``(n_w, K)`` product, equal to the decoded sum to rounding."""
    from scipy.fft import dst
    nt = int(meta["n_t"])
    C = bridge_bands(arrays, meta); K = C.shape[1]
    a = np.asarray(arrays["blt_start"], np.float64); e = np.asarray(arrays["blt_endpoint"], np.float64)
    chi = np.asarray(chi, np.float64).reshape(-1)[:nt]
    if chi.shape[0] < nt:
        chi = np.concatenate([chi, np.zeros(nt - chi.shape[0])])
    d = chi - np.concatenate([chi[1:], [0.0]])
    tau = np.linspace(0.0, 1.0, nt)
    dhat = dst(d[1:-1], type=1, norm="ortho")[:K]
    return float(rho_over_D) * (a * d.sum() + (e - a) * (d * tau).sum() + C @ dhat)


def surface_logweight_series(blt, rho_over_D, chi=None):
    """Per-walker surface-relaxivity log-weight ``(rho/D) sum_k chi_k ell_i(t_k)`` from the
    per-save boundary local-time SERIES ``blt`` (``(n_walkers, n_t)``, stored at ``rho/D = 1``,
    ``<= 0``). The pack-level entry point, which reads the C2 channel and its exact endpoint, is
    :func:`dmipy_sim.replay.replay.surface_logweight`."""
    blt = np.asarray(blt, np.float64)
    s = blt.sum(1) if chi is None else (np.asarray(chi, float)[None, :] * blt).sum(1)
    return float(rho_over_D) * s


# --------------------------------------------------------- envelope & fidelity
def default_envelope():
    """The default certified acquisition envelope for a pack (b in s/m^2)."""
    return dict(bvals=[0.0, 0.5e9, 1e9, 2e9, 3e9],
                dirs=[[0, 0, 1], [1, 0, 0], [1, 0, 1]],
                ogse_periods=[1, 2, 3, 5], shortd_b=1e9,
                shortd_deltas_frac=[0.2, 0.1, 0.05, 0.025],
                delta_frac=0.2, Delta_frac=0.5)


def _bipolar(t, delta, Delta):
    return ((t < delta).astype(float) - ((t >= Delta) & (t < Delta + delta)).astype(float))


def acquisition_battery(n_t, dt, env):
    """Effective (refocused) waveforms G (M,n_t,3) + per-measurement meta, from an envelope."""
    t = np.arange(n_t) * dt; T = n_t * dt
    G, meta = [], []

    def add(prof, fam, d, b, **extra):
        q = GAMMA * dt * np.cumsum(prof); b_unit = dt * np.sum(q ** 2)
        amp = 0.0 if b <= 0 else np.sqrt(b / b_unit)
        dv = np.asarray(d, float); dv = dv / np.linalg.norm(dv)
        G.append(amp * prof[:, None] * dv[None, :])
        meta.append(dict(fam=fam, b=float(b), Gpeak=abs(amp), **extra))

    pgse = _bipolar(t, env["delta_frac"] * T, env["Delta_frac"] * T)
    for d in env["dirs"]:
        for b in env["bvals"]:
            add(pgse, "PGSE", d, b, f_hz=1.0 / (2 * env["delta_frac"] * T))
    for nper in env["ogse_periods"]:
        prof = np.cos(2 * np.pi * nper * t / T)
        for d in env["dirs"]:
            for b in env["bvals"]:
                add(prof, f"OGSE{nper}", d, b, f_hz=nper / T)
    for fr in env["shortd_deltas_frac"]:
        delta = fr * T; Delta = min(0.55 * T, T - delta - dt)
        prof = _bipolar(t, delta, Delta)
        for d in env["dirs"]:
            add(prof, "SHORTD", d, env["shortd_b"], f_hz=1.0 / (2 * delta))
    return np.asarray(G, np.float32), meta


def _walker_phases(pos, dt, G):
    """``(N_w, n_meas)`` phase of every walker under every waveform of ``G`` ``(n_meas, n_t, 3)``: the exact
    integral of the on-grid waveform against the piecewise-linear path through the saves (per-save weights of
    `effective_gradient`)."""
    from ._replay_kernel import effective_gradient
    pos = np.asarray(pos, np.float64); G = np.asarray(G, np.float64)
    return (GAMMA * dt) * np.einsum("mtd,ntd->nm", effective_gradient(G, dt, pos.shape[1], dt), pos)


def _phases_by_chunk(pos, dt, G, chunk_bytes=None, shape=None):
    """:func:`_walker_phases` of every walker, ``(N_w, n_meas)`` float64, taken in walker chunks so that no more
    than ``chunk_bytes`` of the positions is held in float64 at once: the phases are 40 MB where the trajectory
    is 12 GB, and the certificate needs only the phases. ``pos`` is the positions ``(N_w, n_t, 3)`` or a callable
    ``(lo, hi) -> positions of walkers lo:hi`` (a :func:`decoder`), whose ``shape`` ``(N_w, n_t)`` is then given."""
    if callable(pos):
        n_w, n_t = shape; chunk = pos
    else:
        pos = np.asarray(pos); n_w, n_t = pos.shape[0], pos.shape[1]; chunk = lambda lo, hi: pos[lo:hi]
    step = max(1, int(_chunk(chunk_bytes) // (8 * 3 * n_t)))
    out = np.empty((n_w, G.shape[0]), np.float64)
    for lo in range(0, n_w, step):
        out[lo:lo + step] = _walker_phases(chunk(lo, min(lo + step, n_w)), dt, G)
    return out


def _mean_signal(phi, w=None, logw=None):
    """``<w exp(logw) exp(i phi)>`` over the walkers of ``phi`` ``(N_w, n_meas)``, weights normalised."""
    nw = phi.shape[0]
    ww = np.ones(nw) if w is None else np.asarray(w, float)
    lw = np.zeros(nw) if logw is None else np.asarray(logw, float)
    return (np.exp(lw[:, None] + 1j * phi) * (ww / ww.sum())[:, None]).sum(0)


def _replay_complex_np(pos, dt, G, *, w=None, logw=None):
    """Self-contained numpy replay <w exp(logw) exp(i phi)> over :func:`_walker_phases`. Ground truth for the
    fidelity scorer."""
    return _mean_signal(_phases_by_chunk(pos, dt, G), w, logw)


def measure_fidelity(traj, dt_traj, decoded_pos, env=None, w=None, logw=None, chunk_bytes=None):
    """Max complex replay error per acquisition family, decoded vs raw positions, against
    a split-half Monte-Carlo floor. `logw` (optional) applies the same separable weight to
    both so surface/relaxation packs are scored with their physics on. The positions are read
    in walker chunks of at most ``chunk_bytes`` (default :data:`CHUNK_BYTES`) in float64; ``decoded_pos`` may be a
    :func:`decoder` so that the decoded walk is never held whole; when it is ``traj`` itself
    (a raw walk's floor) the phases are read once."""
    env = env or default_envelope()
    r = np.asarray(traj); dt = float(dt_traj)                                 # the walk as stored; float64 per chunk below
    G, meta = acquisition_battery(r.shape[1], dt, env)
    phi_raw = _phases_by_chunk(r, dt, G, chunk_bytes)                          # (N_w, n_meas): all the certificate reads
    S_raw = _mean_signal(phi_raw, w, logw)
    S_dec = S_raw if decoded_pos is traj else _mean_signal(
        _phases_by_chunk(decoded_pos if callable(decoded_pos) else np.asarray(decoded_pos), dt, G, chunk_bytes,
                         shape=r.shape[:2]), w, logw)
    idx = np.random.default_rng(0).permutation(r.shape[0]); h = r.shape[0] // 2
    ia, ib = idx[:h], idx[h:]
    la = None if logw is None else np.asarray(logw)[ia]
    lb = None if logw is None else np.asarray(logw)[ib]
    wa = None if w is None else np.asarray(w)[ia]
    wb = None if w is None else np.asarray(w)[ib]
    Sa = _mean_signal(phi_raw[ia], wa, la)
    Sb = _mean_signal(phi_raw[ib], wb, lb)
    floor = np.abs(Sa - Sb) / 2.0
    fams = sorted({m["fam"] for m in meta})
    per_fam = {}
    for f in fams:
        j = np.array([i for i, m in enumerate(meta) if m["fam"] == f])
        per_fam[f] = dict(err_max=float(np.abs(S_dec[j] - S_raw[j]).max()),
                          floor_max=float(floor[j].max()))
    err_all = float(np.abs(S_dec - S_raw).max())
    floor_all = float(floor.max())
    return dict(metric="max_abs_complex_signal_error", err_max=err_all,
                floor_max=floor_all, noise_floor=float(1 / np.sqrt(r.shape[0])),
                within_2x_floor=bool(err_all <= 2 * floor_all), per_family=per_fam)


def auto_select_modes(X, traj, dt_traj, method=POSITION_METHOD, env=None, tol=2.0,
                      err_target=None, K_grid=(8, 16, 32, 48, 64, 96, 128),
                      w=None, logw=None, verbose=False):
    """Smallest K whose decoded replay error meets the target (err_target absolute, else
    tol * split-half floor). Returns (K, fidelity_report); falls back to the largest K."""
    env = env or default_envelope()
    require_position_method(method)
    cache = None
    best = None
    for K in K_grid:
        arrays, meta, _ = (ENCODERS[method](X, K, _cache=cache) if cache
                           else ENCODERS[method](X, K))
        pos = decode(arrays, meta, n_walkers=(X.shape[0] if is_walker_preserving(method) else None))
        fid = measure_fidelity(traj, dt_traj, pos, env, w=w, logw=logw)
        if verbose:
            log.info(f"  K={K}: err={fid['err_max']:.4f} floor={fid['floor_max']:.4f} "
                  f"{'OK' if fid['within_2x_floor'] else '>'}")
        best = (K, fid)
        thresh = err_target if err_target is not None else tol * fid["floor_max"]
        if fid["err_max"] <= thresh:
            return K, fid
    return best
