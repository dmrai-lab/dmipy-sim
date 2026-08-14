"""IR-basis compression of a master walk — the replay representation, compressed.

A Monte-Carlo master walk is a large object (N_w walkers x N_t steps x 3). Every
replay observable is a functional of the stored walk, and the walk compresses along
two physical redundancies:

  * TEMPORAL — a deliverable gradient waveform is band-limited, so only the low-order
    temporal content of a path is ever probed. ``temporal_dct`` keeps the lowest ``K``
    orthonormal DCT bands of each path: lossless for any acquisition inside that band,
    and it PRESERVES walker identity (so the per-walker relaxation / boundary / MT
    channels stay aligned and replay too).
  * ENSEMBLE — walkers are exchangeable samples. ``lowrank`` keeps ``K`` Karhunen-
    Loeve modes + exact per-walker coefficients (walker-preserving, storage ~ N_w).
    ``gaussian`` / ``marginal`` store only a coefficient DISTRIBUTION and resample at
    replay (walker-count-independent, sub-MB) — exact in the Gaussian limit, gradient
    replay only (they do not preserve walker identity).

The physics channels get their own codecs (positions are smooth+low-rank; these are not):

  * ``compartment`` / ``bound_fraction`` — piecewise-constant per walker -> row RLE.
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

import numpy as np

try:
    from scipy.fft import dct as _dct, idct as _idct
except ImportError as _e:  # pragma: no cover
    raise ImportError("dmipy_sim.compression needs scipy (pip install scipy)") from _e

from .constants import GAMMA

WALKER_PRESERVING = ("temporal_dct", "lowrank")
DISTRIBUTIONAL = ("gaussian", "marginal")
ALL_METHODS = WALKER_PRESERVING + DISTRIBUTIONAL
_F16, _F32 = 2, 4


# --------------------------------------------------------------------- encoders
def encode_temporal_dct(X, K):
    """Keep the lowest K orthonormal DCT-II temporal bands of each path (per axis)."""
    Nw, Nt, _ = X.shape
    C = _dct(np.asarray(X, np.float64), axis=1, type=2, norm="ortho")[:, :K, :]
    arrays = pack_position_arrays(C, np.float32)
    meta = {"method": "temporal_dct", "K": int(K), "n_t": int(Nt)}
    nbytes = Nw * 3 * K * _F16
    return arrays, meta, nbytes


def _kl(X):
    Nw, Nt, _ = X.shape
    Xf = np.asarray(X, np.float64).reshape(Nw, 3 * Nt)
    mean = Xf.mean(0, keepdims=True)
    U, s, Vt = np.linalg.svd(Xf - mean, full_matrices=False)
    return Xf, mean, U, s, Vt, Nt


def encode_lowrank(X, K, _cache=None):
    Xf, mean, U, s, Vt, Nt = _cache or _kl(X)
    K = min(K, s.shape[0])
    A = (U[:, :K] * s[:K])
    arrays = {"modes": Vt[:K].astype(np.float32), "mean": mean[0].astype(np.float32),
              "coeffs": A.astype(np.float16)}
    meta = {"method": "lowrank", "K": int(K), "n_t": int(Nt)}
    nbytes = X.shape[0] * K * _F16 + K * 3 * Nt * _F16 + 3 * Nt * _F16
    return arrays, meta, nbytes


def encode_gaussian(X, K, _cache=None):
    Xf, mean, U, s, Vt, Nt = _cache or _kl(X)
    K = min(K, s.shape[0])
    A = (U[:, :K] * s[:K])
    cov = np.cov(A, rowvar=False) if K > 1 else np.array([[A.var()]])
    arrays = {"modes": Vt[:K].astype(np.float32), "mean": mean[0].astype(np.float32),
              "coeff_mean": A.mean(0).astype(np.float32), "coeff_cov": cov.astype(np.float32)}
    meta = {"method": "gaussian", "K": int(K), "n_t": int(Nt),
            "n_walkers": int(X.shape[0])}
    nbytes = (K * 3 * Nt + K * K + 3 * Nt) * _F32
    return arrays, meta, nbytes


def encode_marginal(X, K, Q=256, _cache=None):
    Xf, mean, U, s, Vt, Nt = _cache or _kl(X)
    K = min(K, s.shape[0])
    A = (U[:, :K] * s[:K])
    qs = np.linspace(0, 1, Q)
    arrays = {"modes": Vt[:K].astype(np.float32), "mean": mean[0].astype(np.float32),
              "coeff_quantiles": np.quantile(A, qs, axis=0).astype(np.float32)}
    meta = {"method": "marginal", "K": int(K), "n_t": int(Nt), "Q": int(Q),
            "n_walkers": int(X.shape[0])}
    nbytes = (K * 3 * Nt + K * Q + 3 * Nt) * _F32
    return arrays, meta, nbytes


ENCODERS = {"temporal_dct": encode_temporal_dct, "lowrank": encode_lowrank,
            "gaussian": encode_gaussian, "marginal": encode_marginal}


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
def encode_bound_fraction(bfrac, Q=256):
    """(arrays, meta) for the bound_fraction channel: quantize [0,1] -> {0..Q-1}, RLE rows."""
    b = np.clip(np.asarray(bfrac, np.float64), 0.0, 1.0)
    q = np.rint(b * (Q - 1)).astype(np.int32)
    vals, lens, counts, n_t = rle_encode_rows(q)
    val_dtype = np.uint8 if Q <= 256 else np.int32
    len_dtype = np.uint16 if n_t < 65535 else np.int32
    arrays = {"bfrac_rle_vals": vals.astype(val_dtype),
              "bfrac_rle_lens": lens.astype(len_dtype),
              "bfrac_rle_counts": np.asarray(counts, np.int32)}
    return arrays, {"channel": "bound_fraction", "Q": int(Q), "n_t": int(n_t)}


def decode_bound_fraction(arrays, meta):
    q = rle_decode_rows(np.asarray(arrays["bfrac_rle_vals"]),
                        np.asarray(arrays["bfrac_rle_lens"]),
                        np.asarray(arrays["bfrac_rle_counts"]), int(meta["n_t"]))
    return (q.astype(np.float32) / float(meta["Q"] - 1))


def encode_boundary_local_time(dlog, nlevels=4096):
    """DENSITY-AWARE per-step codec: sparse CSR (isolated fibres) or dense int8 (packed
    WM). Lossless to the value quant. NOTE: caps at ~2x when contacts are dense (small R);
    prefer ``encode_boundary_dct`` for the smooth cumulative representation."""
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


def decode_boundary_local_time(arrays, meta):
    nt = int(meta["n_t"]); scale = float(meta["scale"]); nl = int(meta["nlevels"])
    if meta.get("mode") == "dense" or "blt_dense_q" in arrays:
        q = np.asarray(arrays["blt_dense_q"], np.float64)
        return (q * scale / nl).astype(np.float32)
    counts = np.asarray(arrays["blt_counts"]); cols = np.asarray(arrays["blt_cols"])
    qvals = np.asarray(arrays["blt_qvals"], np.float64)
    out = np.zeros((counts.size, nt), np.float32); p = 0
    for i, c in enumerate(counts):
        c = int(c)
        out[i, cols[p:p + c]] = (qvals[p:p + c] * scale / (nl - 1)).astype(np.float32)
        p += c
    return out


def encode_boundary_dct(dlog, K=16, dtype=np.float32):
    """IR-basis codec for the boundary-local-time channel: DETREND-then-DCT of the CUMULATIVE
    local time B(t)=cumsum(ell). ell(t) is spiky+dense, but its running integral B(t) is
    smooth. B is ~linear (roughly constant contact rate), and a bare DCT of a ramp has a
    Gibbs endpoint error that exceeds the MC floor at small K -- fatal for the longest-TE
    truncation. So store the EXACT endpoint B(T) (1 float/walker) and DCT only the residual
    B(t) - (t/T)*B(T), which vanishes at both ends and is tiny+smooth: K~8 bands take every
    TE truncation / per-save gate (ell = diff(B)) to ~100x below the MC floor. Stores
    (N_w, K+1) float32; ratio ~ n_t/K, which GROWS with walk length -- this is what makes
    surface-relaxivity replay memory-viable at high n_t (record once, sub-micron)."""
    A = np.asarray(dlog, np.float64)
    nw, nt = A.shape
    B = np.cumsum(A, axis=1)                                   # (N_w, n_t) smooth
    endpoint = B[:, -1].copy()                                 # exact total local time
    ramp = np.linspace(0.0, 1.0, nt)[None, :]
    resid = B - endpoint[:, None] * ramp                      # 0 at both ends, small+smooth
    C = _dct(resid, axis=1, type=2, norm="ortho")[:, :K]      # (N_w, K)
    # ``dtype`` sets the coefficient precision. It defaults to f32 so the codec keeps its
    # lossless-at-K=n_t contract; packs pass f16 (halves the tier at measurably identical ensemble
    # error) via build_replay_pack's ``blt_dtype``. The ENDPOINT is always f32 -- it is the exact
    # cumulative total the rho attenuation reads directly, where f16's ~3 significant digits would be
    # a real error rather than a rounding one.
    arrays = {"blt_dct_coeffs": C.astype(dtype),
              "blt_endpoint": endpoint.astype(np.float32)}
    meta = {"channel": "boundary_local_time", "mode": "dct", "n_t": int(nt), "K": int(K),
            "dtype": np.dtype(dtype).name}
    return arrays, meta


def decode_boundary_dct(arrays, meta):
    """Reconstruct per-save ell(t) = diff(B) from the stored endpoint + residual DCT bands."""
    nt = int(meta["n_t"])
    C = np.asarray(arrays["blt_dct_coeffs"], np.float64)
    endpoint = np.asarray(arrays["blt_endpoint"], np.float64)
    ramp = np.linspace(0.0, 1.0, nt)[None, :]
    B = _idct(C, axis=1, type=2, norm="ortho", n=nt) + endpoint[:, None] * ramp
    ell = np.diff(B, axis=1, prepend=B[:, :1] * 0.0)          # per-save increments
    return ell.astype(np.float32)


def encode_compartment(comp, Q=256):
    """Compartment codec. Integer labels -> lossless row RLE; fractional per-save occupancy
    (permeable) -> quantize to Q levels over [0, scale] and RLE (faithful to fractions)."""
    A = np.asarray(comp)
    is_frac = not np.array_equal(A, np.round(A))
    if not is_frac:
        vals, lens, counts, n_t = rle_encode_rows(A.astype(np.int32))
        arrays = {"comp_rle_vals": vals.astype(np.int16),
                  "comp_rle_lens": lens.astype(np.int32),
                  "comp_rle_counts": np.asarray(counts, np.int32)}
        return arrays, {"channel": "compartment", "fractional": False, "n_t": int(n_t)}
    scale = float(np.nanmax(np.abs(A))) or 1.0
    q = np.rint(np.clip(A, 0.0, scale) / scale * (Q - 1)).astype(np.int32)
    vals, lens, counts, n_t = rle_encode_rows(q)
    arrays = {"comp_rle_vals": vals.astype(np.uint8 if Q <= 256 else np.int32),
              "comp_rle_lens": lens.astype(np.uint16 if n_t < 65535 else np.int32),
              "comp_rle_counts": np.asarray(counts, np.int32)}
    return arrays, {"channel": "compartment", "fractional": True, "Q": int(Q),
                    "scale": scale, "n_t": int(n_t)}


def decode_compartment(arrays, meta):
    q = rle_decode_rows(np.asarray(arrays["comp_rle_vals"]),
                        np.asarray(arrays["comp_rle_lens"]),
                        np.asarray(arrays["comp_rle_counts"]), int(meta["n_t"]))
    if meta.get("fractional"):
        return (q.astype(np.float32) / float(meta["Q"] - 1)) * float(meta.get("scale", 1.0))
    return q.astype(np.int16)


CHANNEL_ENCODERS = {"bound_fraction": encode_bound_fraction,
                    "boundary_local_time": encode_boundary_local_time,
                    "boundary_dct": encode_boundary_dct,
                    "compartment": encode_compartment}
CHANNEL_DECODERS = {"bound_fraction": decode_bound_fraction,
                    "boundary_local_time": decode_boundary_local_time,
                    "boundary_dct": decode_boundary_dct,
                    "compartment": decode_compartment}


def encode(X, method="temporal_dct", K=32, **kw):
    if method not in ENCODERS:
        raise ValueError(f"unknown method {method!r}; choose from {list(ENCODERS)}")
    return ENCODERS[method](X, K, **kw)


# --------------------------------------------------------------------- decoder
def _sample_coeffs(arrays, meta, n_walkers=None, seed=0):
    method = meta["method"]; rng = np.random.default_rng(seed)
    if method == "lowrank":
        A = np.asarray(arrays["coeffs"], np.float64)
        if n_walkers is not None and n_walkers != A.shape[0]:
            raise ValueError("lowrank preserves its stored walker count only")
        return A
    K = np.asarray(arrays["modes"]).shape[0]
    n = n_walkers or int(meta.get("n_walkers", 100_000))
    if method == "gaussian":
        L = np.linalg.cholesky(np.asarray(arrays["coeff_cov"], np.float64) + 1e-30 * np.eye(K))
        return rng.standard_normal((n, K)) @ L.T + np.asarray(arrays["coeff_mean"], np.float64)
    if method == "marginal":
        quant = np.asarray(arrays["coeff_quantiles"], np.float64)
        qs = np.linspace(0, 1, quant.shape[0]); u = rng.random((n, K))
        return np.stack([np.interp(u[:, k], qs, quant[:, k]) for k in range(K)], axis=1)
    raise ValueError(f"unknown method {method!r}")


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


def pack_position_arrays(C, dtype=np.float32):
    """(n_w, K, 3) coefficients -> {'pos_x','pos_y','pos_z'}, each (n_w, K)."""
    C = np.asarray(C)
    return {POSITION_AXES[i]: np.ascontiguousarray(C[:, :, i]).astype(dtype)
            for i in range(C.shape[2])}


def has_axis_layout(arrays):
    return POSITION_AXES[0] in arrays


def read_position_coeffs(arrays, axes=None, dtype=np.float64):
    """Coefficients as (n_w, K, n_axes) from either layout.

    ``axes`` selects spatial components by index (default all present). Reading a subset is the point of
    the layout: pass e.g. ``axes=(0,)`` for a slab or ``(0, 1)`` for a cylinder's transverse plane.
    """
    if has_axis_layout(arrays):
        present = [i for i, k in enumerate(POSITION_AXES) if k in arrays]
        want = list(present if axes is None else axes)
        missing = [i for i in want if i not in present]
        if missing:
            raise KeyError(f"pack does not carry position axes {missing} "
                           f"(has {[POSITION_AXES[i] for i in present]}); it was written with a reduced "
                           f"axis set and cannot serve this encoding")
        return np.stack([np.asarray(arrays[POSITION_AXES[i]], dtype) for i in want], axis=2)
    # No legacy (n_w, K, 3) fallback by design. A dataset with two position layouts forces every
    # consumer to carry a compatibility branch, and that branch is where silent errors live -- a reader
    # that guesses the wrong convention returns plausible numbers. Fail loudly instead; convert the pack.
    raise KeyError(
        f"pack has no position axes {POSITION_AXES}; found {sorted(arrays)}. Packs written before the "
        f"axis-per-tensor layout store a single (n_walkers, K, 3) 'dct_coeffs'; re-encode them with "
        f"pack_position_arrays(). This reader does not accept mixed layouts -- a dataset with two "
        f"position layouts forces every consumer to carry a compatibility branch, and that branch is "
        f"where silent errors live.")


def decode(arrays, meta, n_walkers=None, seed=0):
    """Reconstruct positions (n_walkers, n_t, 3) from a pack's stored arrays."""
    method = meta["method"]; Nt = int(meta["n_t"])
    if method == "temporal_dct":
        C = read_position_coeffs(arrays)
        return _idct(C, axis=1, type=2, norm="ortho", n=Nt).astype(np.float64)
    modes = np.asarray(arrays["modes"], np.float64)
    mean = np.asarray(arrays.get("mean", np.zeros(modes.shape[1])), np.float64)
    A = _sample_coeffs(arrays, meta, n_walkers, seed)
    X = A @ modes + mean
    return X.reshape(X.shape[0], Nt, 3)


def is_walker_preserving(method):
    return method in WALKER_PRESERVING


# ------------------------------------------------------------ mode-space replay
def mode_space_phi(arrays, meta, G, dt, n_walkers=None, seed=0):
    """Gradient phase phi_i (N_w, n_meas) in the compressed basis, WITHOUT reconstructing
    the trajectory. Linear in position, so it commutes with every position codec:

    * ``temporal_dct`` (orthonormal DCT bands C): phi = gamma*dt * sum_{k<K,d} C_{k,d} Ghat_{k,d},
      Ghat = DCT(G) truncated to the same K bands.
    * ``lowrank``/``gaussian``/``marginal`` (KL modes V): phi = A.c(G) + phi0, with
      c(G) = gamma*dt (V.vec(G)) a K-vector per measurement.
    """
    method = meta.get("method", "temporal_dct")
    G = np.asarray(G, np.float64); n_meas = G.shape[0]
    if method == "temporal_dct":
        C = read_position_coeffs(arrays, dtype=np.float64)         # (N_w, K, n_axes)
        K = C.shape[1]
        Ghat = _dct(G, axis=1, type=2, norm="ortho")[:, :K, :]    # (n_meas, K, 3)
        return (GAMMA * dt) * np.einsum("wkd,mkd->wm", C, Ghat)
    Gf = G.reshape(n_meas, -1)
    modes = np.asarray(arrays["modes"], np.float64)
    mean = np.asarray(arrays.get("mean", np.zeros(modes.shape[1])), np.float64)
    A = _sample_coeffs(arrays, meta, n_walkers, seed)
    c = (GAMMA * dt) * (modes @ Gf.T)
    phi0 = (GAMMA * dt) * (Gf @ mean)
    return A @ c + phi0[None, :]


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


def relaxation_logweight(comp, T2_per_comp, T1_per_comp, dt, chi=None):
    """Per-walker relaxation log-weight from the compartment channel — O(N_w N_t), no
    trajectory. `comp` is integer labels OR fractional 2-compartment occupancy."""
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
        return -dt * r2.sum(1)
    chi = np.asarray(chi, float)[None, :]
    return -dt * (chi * r2 + (1.0 - chi) * r1).sum(1)


def surface_logweight(blt, rho_over_D, chi=None):
    """Per-walker surface-relaxivity log-weight = (rho/D) sum_k chi_k ell_i(t_k) (ell stored
    at rho/D=1, <=0). Additive over the boundary channel; no trajectory."""
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


def _replay_complex_np(pos, dt, G, *, w=None, logw=None):
    """Self-contained numpy replay <w exp(logw) exp(i phi)>, phi = gamma dt sum_t G.r.
    Ground truth for the fidelity scorer (no engine dependency)."""
    pos = np.asarray(pos, np.float64); G = np.asarray(G, np.float64)
    nw = pos.shape[0]
    phi = (GAMMA * dt) * np.einsum("mtd,ntd->nm", G, pos)     # (N_w, n_meas)
    ww = np.ones(nw) if w is None else np.asarray(w, float)
    lw = np.zeros(nw) if logw is None else np.asarray(logw, float)
    return (np.exp(lw[:, None] + 1j * phi) * (ww / ww.sum())[:, None]).sum(0)


def measure_fidelity(traj, dt_traj, decoded_pos, env=None, w=None, logw=None):
    """Max complex replay error per acquisition family, decoded vs raw positions, against
    a split-half Monte-Carlo floor. `logw` (optional) applies the same separable weight to
    both so surface/relaxation packs are scored with their physics on."""
    env = env or default_envelope()
    r = np.asarray(traj, np.float64); dt = float(dt_traj)
    G, meta = acquisition_battery(r.shape[1], dt, env)
    S_raw = _replay_complex_np(r, dt, G, w=w, logw=logw)
    S_dec = _replay_complex_np(np.asarray(decoded_pos, np.float64), dt, G, w=w, logw=logw)
    idx = np.random.default_rng(0).permutation(r.shape[0]); h = r.shape[0] // 2
    ia, ib = idx[:h], idx[h:]
    la = None if logw is None else np.asarray(logw)[ia]
    lb = None if logw is None else np.asarray(logw)[ib]
    wa = None if w is None else np.asarray(w)[ia]
    wb = None if w is None else np.asarray(w)[ib]
    Sa = _replay_complex_np(r[ia], dt, G, w=wa, logw=la)
    Sb = _replay_complex_np(r[ib], dt, G, w=wb, logw=lb)
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


def auto_select_modes(X, traj, dt_traj, method="temporal_dct", env=None, tol=2.0,
                      err_target=None, K_grid=(8, 16, 32, 48, 64, 96, 128),
                      w=None, logw=None, verbose=False):
    """Smallest K whose decoded replay error meets the target (err_target absolute, else
    tol * split-half floor). Returns (K, fidelity_report); falls back to the largest K."""
    env = env or default_envelope()
    cache = _kl(X) if method in ("lowrank", "gaussian", "marginal") else None
    best = None
    for K in K_grid:
        arrays, meta, _ = (ENCODERS[method](X, K, _cache=cache) if cache
                           else ENCODERS[method](X, K))
        pos = decode(arrays, meta, n_walkers=(X.shape[0] if is_walker_preserving(method) else None))
        fid = measure_fidelity(traj, dt_traj, pos, env, w=w, logw=logw)
        if verbose:
            print(f"  K={K}: err={fid['err_max']:.4f} floor={fid['floor_max']:.4f} "
                  f"{'OK' if fid['within_2x_floor'] else '>'}", flush=True)
        best = (K, fid)
        thresh = err_target if err_target is not None else tol * fid["floor_max"]
        if fid["err_max"] <= thresh:
            return K, fid
    return best
