"""Master-walk compression — the algorithm behind replay-pack generation.

A Monte-Carlo master walk is a large object (N_w walkers x N_t steps x 3). Every
replay observable is a functional of the stored POSITIONS, so the walk compresses
along two physical redundancies:

  * TEMPORAL — a deliverable gradient waveform is band-limited, so only the low-order
    temporal content of the path is ever probed. ``temporal_dct`` keeps the lowest
    ``K`` DCT bands of each path: lossless for any acquisition inside that band, and
    it PRESERVES walker identity (so the per-walker relaxation / boundary / MT
    channels stay aligned and replay too).
  * ENSEMBLE — walkers are exchangeable samples. ``lowrank`` keeps ``K`` Karhunen-
    Loeve modes + exact per-walker coefficients (walker-preserving, storage ~ N_w).
    ``gaussian`` / ``marginal`` store only a coefficient DISTRIBUTION and resample at
    replay (walker-count-independent, sub-MB) — exact in the Gaussian limit, but they
    do NOT preserve walker identity, so they serve gradient/EAP replay only.

Walker-preserving methods (``temporal_dct``, ``lowrank``) are the lossless tier that
carries the whole multi-physics button space; ``gaussian``/``marginal`` are the
aggressive, gradient-only tier. Each method (a) ENCODES positions to a small dict of
named arrays (safetensors-ready), (b) DECODES back to positions, and reports its byte
footprint. ``auto_select_modes`` picks the smallest K that meets a fidelity tolerance
against the uncompressed walk, and ``measure_fidelity`` produces the self-certifying
report stored in the pack.

Susceptibility note (public): fidelity/replay here are gradient + relaxation + surface.
Off-resonance/susceptibility replay is PROVIDER-driven in public dmipy-sim (a pack stores
positions; a :mod:`dmipy_sim.susceptibility` provider's ``delta_bz_fn`` is applied at
replay), so this module carries no packed field-map codecs — unlike the private engine,
which precomputed Phi_C/Phi_S/Phi_0 field maps.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.fft import dct as _dct, idct as _idct
except ImportError as _e:  # pragma: no cover
    raise ImportError("dmipy_sim.compression needs scipy (pip install scipy)") from _e

WALKER_PRESERVING = ("temporal_dct", "lowrank")
DISTRIBUTIONAL = ("gaussian", "marginal")
ALL_METHODS = WALKER_PRESERVING + DISTRIBUTIONAL
_F16, _F32 = 2, 4


# --------------------------------------------------------------------- encoders
def encode_temporal_dct(X, K):
    """Keep the lowest K orthonormal DCT-II temporal bands of each path (per axis)."""
    Nw, Nt, _ = X.shape
    C = _dct(np.asarray(X, np.float64), axis=1, type=2, norm="ortho")[:, :K, :]
    arrays = {"dct_coeffs": C.astype(np.float32)}
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
    three flat arrays (safetensors-friendly). Compartment tags change ~never per
    walker (impermeable myelin), so this is a huge, lossless reduction."""
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
# The surface-relaxivity (boundary_local_time) and MT (bound_fraction) channels are
# dense (Nw, Nt) per-walker arrays, but with very different structure from positions,
# so they get their own codecs (positions are smooth+low-rank; these are not):
#
#  * bound_fraction  -- occupancy in [0,1], ~binary with long dwell/free runs
#    (measured: ~86% exactly 0, ~10% exactly 1, ~4% fractional). QUANTIZE to Q levels
#    and run-length encode the rows: ~8x at Q=256 (MAE ~4e-5), lossless to quant.
#
#  * boundary_local_time -- per-save wall-contact local time (rho/D=1), ~85% zeros
#    (isolated contacts) and NOT low-rank (idiosyncratic per walker: low-rank needs
#    K>32 and still misses the noise floor). Store SPARSE (nonzero cols + quantized
#    values + per-row count, i.e. CSR): ~3-5x, lossless to the value quantization,
#    which is set fine enough that the surface-relaxivity log-weight stays within the
#    MC floor. The replay uses Sum_t chi(t)*dlog(t), so per-save values are kept (not
#    just the endpoint) to remain valid for any sequence gate (SPEC §6.6).

def encode_bound_fraction(bfrac, Q=256):
    """(arrays, meta) for the bound_fraction channel: quantize [0,1] -> {0..Q-1}, RLE rows."""
    b = np.clip(np.asarray(bfrac, np.float64), 0.0, 1.0)
    q = np.rint(b * (Q - 1)).astype(np.int32)
    vals, lens, counts, n_t = rle_encode_rows(q)
    val_dtype = np.uint8 if Q <= 256 else np.int32   # Q-1 levels; run lens <= n_t
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
    """(arrays, meta) for the boundary_local_time channel — DENSITY-AWARE.

    Wall-contact density varies hugely by substrate: a lone fibre is ~15% nonzero
    (sparse wins big), but packed white matter is ~55% (walls hit constantly). Sparse
    CSR costs ~4 B/nonzero, so above ~50% density it is *larger* than a dense array;
    there a dense int8 quantization (1 B/entry, density-independent, half of raw float16)
    wins. Pick whichever is smaller per pack. Both are lossless to the value quant step,
    fine enough that the surface-relaxivity log-weight stays within the MC floor."""
    A = np.asarray(dlog, np.float64)
    nw, nt = A.shape
    scale = float(np.abs(A).max()) + 1e-30
    nnz = int(np.count_nonzero(A))
    sparse_bytes = nnz * 4 + nw * 4            # int16 col + int16 val per nz, int32 counts
    dense_bytes = nw * nt                      # int8 per entry
    if sparse_bytes <= dense_bytes:            # SPARSE (isolated fibres, low density)
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
    # DENSE int8 (packed WM, high density): signed 127-level quantization, 1 B/entry
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


def encode_compartment(comp, Q=256):
    """Compartment-channel codec. Integer labels (impermeable) -> lossless row RLE.
    FRACTIONAL per-save occupancy (permeable: a walker crossing a membrane mid-save has
    fractional time-in-compartment) is near-binary with long runs (measured ~84% exactly
    0, ~16% exactly 1) -- structurally like bound_fraction -- so quantize to Q levels over
    [0, scale] and RLE. Integer RLE would int-cast the fractions (lossy); this is faithful."""
    A = np.asarray(comp)
    is_frac = not np.array_equal(A, np.round(A))
    if not is_frac:
        vals, lens, counts, n_t = rle_encode_rows(A.astype(np.int32))
        arrays = {"comp_rle_vals": vals.astype(np.int16),
                  "comp_rle_lens": lens.astype(np.int32),
                  "comp_rle_counts": np.asarray(counts, np.int32)}
        return arrays, {"channel": "compartment", "fractional": False, "n_t": int(n_t)}
    scale = float(np.nanmax(np.abs(A))) or 1.0        # 1.0 for 2-comp occupancy; >1 = multi-comp averaged id
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
                    "compartment": encode_compartment}
CHANNEL_DECODERS = {"bound_fraction": decode_bound_fraction,
                    "boundary_local_time": decode_boundary_local_time,
                    "compartment": decode_compartment}


def encode(X, method="lowrank", K=32, **kw):
    if method not in ENCODERS:
        raise ValueError(f"unknown method {method!r}; choose from {list(ENCODERS)}")
    return ENCODERS[method](X, K, **kw)


# --------------------------------------------------------------------- decoder
def _sample_coeffs(arrays, meta, n_walkers=None, seed=0):
    """KL coefficients A (n, K): stored per-walker for `lowrank`; sampled from the stored
    distribution for `gaussian`/`marginal` (any n)."""
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


def decode(arrays, meta, n_walkers=None, seed=0):
    """Reconstruct positions (n_walkers, n_t, 3) from a pack's stored arrays."""
    method = meta["method"]; Nt = int(meta["n_t"])
    if method == "temporal_dct":
        C = np.asarray(arrays["dct_coeffs"], np.float64)
        return _idct(C, axis=1, type=2, norm="ortho", n=Nt).astype(np.float64)
    modes = np.asarray(arrays["modes"], np.float64)
    mean = np.asarray(arrays.get("mean", np.zeros(modes.shape[1])), np.float64)
    A = _sample_coeffs(arrays, meta, n_walkers, seed)
    X = A @ modes + mean
    return X.reshape(X.shape[0], Nt, 3)


def is_walker_preserving(method):
    return method in WALKER_PRESERVING


def replay_gradient_lowrank(arrays, meta, G, dt, weights=None, n_walkers=None, seed=0):
    """Mode-space gradient replay — WITHOUT reconstructing the trajectory (all position
    codecs; see mode_space_phi). The gradient phase is *linear* in position, and positions
    are stored as ``r = A·V + μ`` (coeffs A, K modes V, mean μ), so

        φ_i = γ·dt·Σ_{k,d} G[k,d]·r_i[k,d] = A_i·c(G) + φ0,
        c(G) = γ·dt·(V · vec(G))   (a K-vector, computed ONCE per measurement),
        φ0   = γ·dt·(μ · vec(G)).

    This is an exact algebraic identity (matches the dense contract to machine precision),
    costs ``N_w·K`` per measurement instead of ``N_w·N_t·3``, and never materialises the
    ``(N_w, N_t, 3)`` positions. Distributional codecs (gaussian/marginal) use the same
    formula after sampling A. ``G`` is ``(n_meas, N_t, 3)`` in T/m; returns complex signal
    ``(n_meas,)`` = weighted ``⟨exp(iφ)⟩``.
    """
    phi = mode_space_phi(arrays, meta, G, dt, n_walkers=n_walkers, seed=seed)   # (N_w, n_meas)
    w = np.ones(phi.shape[0]) if weights is None else np.asarray(weights, np.float64)
    return (np.exp(1j * phi) * (w / w.sum())[:, None]).sum(0)


def mode_space_phi(arrays, meta, G, dt, n_walkers=None, seed=0):
    """Gradient phase φ_i (N_w, n_meas) in the compressed basis, WITHOUT reconstructing the
    trajectory. The phase is linear in position, so it commutes with every position codec:

    * ``lowrank``/``gaussian``/``marginal`` (KL modes ``V``): φ = A·c(G) + φ0, with
      c(G) = γΔt·(V·vec(G)) a K-vector per measurement. A is the stored coeffs (lowrank)
      or freshly sampled (distributional, `n_walkers` walkers).
    * ``temporal_dct`` (orthonormal DCT-II bands ``C``): φ = γΔt·Σ_{k<K,d} C_{k,d}·Ĝ_{k,d},
      where Ĝ = DCT(G) — transform the waveform into the same K bands and contract.

    Reused by relaxation/surface-weighted replay, which multiply a separable per-walker
    log-weight onto exp(iφ)."""
    from .constants import GAMMA
    method = meta.get("method", "lowrank")
    G = np.asarray(G, np.float64); n_meas = G.shape[0]
    if method == "temporal_dct":
        C = np.asarray(arrays["dct_coeffs"], np.float64)          # (N_w, K, 3)
        K = C.shape[1]
        Ghat = _dct(G, axis=1, type=2, norm="ortho")[:, :K, :]    # (n_meas, K, 3)
        return (GAMMA * dt) * np.einsum("wkd,mkd->wm", C, Ghat)
    Gf = G.reshape(n_meas, -1)                       # (n_meas, N_t*3)
    modes = np.asarray(arrays["modes"], np.float64)  # (K, N_t*3)
    mean = np.asarray(arrays.get("mean", np.zeros(modes.shape[1])), np.float64)
    A = _sample_coeffs(arrays, meta, n_walkers, seed)  # (N_w, K)
    c = (GAMMA * dt) * (modes @ Gf.T)                # (K, n_meas)
    phi0 = (GAMMA * dt) * (Gf @ mean)                # (n_meas,)
    return A @ c + phi0[None, :]


_JAX_KERNELS = {}


def _jax_signal_kernels():
    """Lazily build + cache the jitted device kernels (compiled once, per input shape/dtype;
    keeps the matmul AND the N_w·n_meas exp-reduce on the GPU)."""
    if "lr" not in _JAX_KERNELS:
        import jax, jax.numpy as jnp

        @jax.jit
        def lr(A, modes, mean, Gf, gdt, lw, w):        # lowrank / distributional
            phi = A @ (gdt * (modes @ Gf.T)) + (gdt * (Gf @ mean))[None, :]
            return (jnp.exp(lw[:, None] + 1j * phi) * (w / w.sum())[:, None]).sum(0)

        @jax.jit
        def dct(C, Ghat, gdt, lw, w):                  # temporal_dct
            phi = gdt * jnp.einsum("wkd,mkd->wm", C, Ghat)
            return (jnp.exp(lw[:, None] + 1j * phi) * (w / w.sum())[:, None]).sum(0)

        _JAX_KERNELS["lr"], _JAX_KERNELS["dct"] = lr, dct
    return _JAX_KERNELS["lr"], _JAX_KERNELS["dct"]


def mode_space_signal(arrays, meta, G, dt, logw=None, weights=None,
                      backend="numpy", precision="float64", n_walkers=None, seed=0):
    """Mode-space gradient signal S = ⟨w·exp(logw + iφ)⟩ (n_meas,) complex, computing the
    phase (§mode_space_phi) AND the ``exp(iφ)`` reduction — the cost floor — on the chosen
    backend/precision, never reconstructing the trajectory.

    backend='numpy' (default, CPU) or 'jax' (device; GPU makes the N_w·n_meas exp-reduce
    ~free — the whole battery is near-instant). precision='float64' (default) or 'float32'
    (~1.6x faster on CPU; error stays within the MC floor for typical b, but the phase can
    be large at high b, so it is opt-in, not the default). `logw` (N_w,) and `weights` (N_w,)
    default to 0 and uniform."""
    from .constants import GAMMA
    method = meta.get("method", "lowrank")
    G = np.asarray(G, np.float64); n_meas = G.shape[0]
    gdt = float(GAMMA * dt)
    if backend == "jax":
        import jax.numpy as jnp
        fdt = jnp.float32 if precision == "float32" else jnp.float64
        klr, kdct = _jax_signal_kernels()          # jitted (matmul + exp-reduce on device)
        if method == "temporal_dct":
            C = jnp.asarray(arrays["dct_coeffs"], fdt)                # (N_w, K, 3)
            Ghat = jnp.asarray(_dct(G, axis=1, type=2, norm="ortho")[:, :C.shape[1], :], fdt)
            n_w = C.shape[0]
            lw = jnp.zeros(n_w, fdt) if logw is None else jnp.asarray(logw, fdt)
            w = jnp.ones(n_w, fdt) if weights is None else jnp.asarray(weights, fdt)
            return np.asarray(kdct(C, Ghat, jnp.asarray(gdt, fdt), lw, w))
        A = jnp.asarray(_sample_coeffs(arrays, meta, n_walkers, seed), fdt)
        modes = jnp.asarray(arrays["modes"], fdt)
        mean = jnp.asarray(arrays.get("mean", np.zeros(np.asarray(arrays["modes"]).shape[1])), fdt)
        Gf = jnp.asarray(G.reshape(n_meas, -1), fdt)
        n_w = A.shape[0]
        lw = jnp.zeros(n_w, fdt) if logw is None else jnp.asarray(logw, fdt)
        w = jnp.ones(n_w, fdt) if weights is None else jnp.asarray(weights, fdt)
        return np.asarray(klr(A, modes, mean, Gf, jnp.asarray(gdt, fdt), lw, w))
    # numpy backend
    phi = mode_space_phi(arrays, meta, G, dt, n_walkers=n_walkers, seed=seed)
    if precision == "float32":
        phi = phi.astype(np.float32)
    n_w = phi.shape[0]
    lw = np.zeros(n_w) if logw is None else np.asarray(logw)
    w = np.ones(n_w) if weights is None else np.asarray(weights, float)
    if precision == "float32":
        lw = lw.astype(np.float32); w = w.astype(np.float32)
    return (np.exp(lw[:, None] + 1j * phi) * (w / w.sum())[:, None]).sum(0)


def relaxation_logweight(comp, T2_per_comp, T1_per_comp, dt, chi=None):
    """Per-walker relaxation log-weight from the compartment channel — O(N_w·N_t), no
    trajectory. `comp` is integer labels OR fractional 2-compartment occupancy (permeable);
    fractional blends the two pools' rates. `chi` is the per-step transverse gate (default
    all-ones = spin-echo, T2 only); otherwise `chi`·(1/T2) + (1-`chi`)·(1/T1)."""
    comp = np.asarray(comp)
    invT2 = np.where(np.asarray(T2_per_comp) > 0, 1.0 / np.maximum(np.asarray(T2_per_comp, float), 1e-30), 0.0)
    # A pack may carry T2 but no T1 (transverse-only) -> T1_per_comp is None; treat as no
    # longitudinal relaxation (rate 0), matching the T2 compartment shape.
    if T1_per_comp is None:
        invT1 = np.zeros_like(invT2)
    else:
        invT1 = np.where(np.asarray(T1_per_comp) > 0, 1.0 / np.maximum(np.asarray(T1_per_comp, float), 1e-30), 0.0)
    if np.issubdtype(comp.dtype, np.floating) and not np.array_equal(comp, np.round(comp)):
        f = np.clip(comp, 0.0, 1.0)                  # occupancy of compartment 1 (2-comp)
        r2 = (1.0 - f) * invT2[0] + f * invT2[1]
        r1 = (1.0 - f) * invT1[0] + f * invT1[1]
    else:
        ci = comp.astype(np.int64); r2 = invT2[ci]; r1 = invT1[ci]
    if chi is None:
        return -dt * r2.sum(1)
    chi = np.asarray(chi, float)[None, :]
    return -dt * (chi * r2 + (1.0 - chi) * r1).sum(1)


def surface_logweight(blt, rho_over_D, chi=None):
    """Per-walker surface-relaxivity log-weight = (ρ/D)·Σ_k χ_k·ℓ_i(t_k) (ℓ stored at ρ/D=1,
    ≤0). Additive sum over the (already small) surface channel; no trajectory."""
    blt = np.asarray(blt, np.float64)
    s = blt.sum(1) if chi is None else (np.asarray(chi, float)[None, :] * blt).sum(1)
    return float(rho_over_D) * s


# --------------------------------------------------------- envelope & fidelity
def default_envelope():
    """The default certified acquisition envelope for a pack. b in s/m^2; OGSE up to
    a moderate frequency (higher needs more modes)."""
    return dict(bvals=[0.0, 0.5e9, 1e9, 2e9, 3e9],
                dirs=[[0, 0, 1], [1, 0, 0], [1, 0, 1]],
                ogse_periods=[1, 2, 3, 5], shortd_b=1e9,
                shortd_deltas_frac=[0.2, 0.1, 0.05, 0.025],
                B0_list=[], theta_deg=[0, 90],
                delta_frac=0.2, Delta_frac=0.5)


def _bipolar(t, delta, Delta):
    return ((t < delta).astype(float) - ((t >= Delta) & (t < Delta + delta)).astype(float))


def acquisition_battery(n_t, dt, env):
    """Effective (refocused) waveforms G (M,n_t,3) + per-measurement meta for the
    fidelity scorer, from an envelope dict."""
    from .constants import GAMMA
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


def _replay_complex(pos, dt, G, *, w=None, susceptibility=None, eps_P=None,
                    comp=None, T2pc=None, T1pc=None, dlog_b=None, rho=0.0, D=None, chunk=5000):
    """Complex signal <w e^{logw} e^{i phi}> through the public engine, chunked over walkers.

    Public susceptibility is PROVIDER-driven: pass a :mod:`dmipy_sim.susceptibility`
    provider (or a bare ``r -> ΔBz`` callable) as ``susceptibility=`` and it is applied via
    :func:`dmipy_sim.trajectories.replay` (no packed field maps)."""
    from .trajectories import replay as _replay
    Nt = pos.shape[1]; nw = pos.shape[0]
    ww = np.ones(nw) if w is None else np.asarray(w, float)
    num = np.zeros(G.shape[0], np.complex128); den = float(ww.sum())
    for s in range(0, nw, chunk):
        e = min(s + chunk, nw)
        kw = dict(chi_perp=np.ones(Nt, np.float32), return_walker_signals=True)
        if comp is not None and T2pc is not None:
            kw.update(comp_traj=comp[s:e], T2_per_comp=T2pc, T1_per_comp=T1pc)
        if dlog_b is not None and rho:
            kw.update(dlog_boundary_unit=dlog_b[s:e], surface_relaxivity=float(rho), D=float(D))
        if susceptibility is not None:
            kw.update(susceptibility=susceptibility, eps_P=eps_P)
        phi, logw, _ = _replay(pos[s:e].astype(np.float32), dt, G, dt, **kw)
        phi = np.asarray(phi); logw = np.asarray(logw)
        if logw.shape[-1] == 1:
            logw = np.broadcast_to(logw, phi.shape)
        num += (np.exp(logw) * np.exp(1j * phi) * ww[s:e][None, :]).sum(axis=1)
    return num / den


def measure_fidelity(master, decoded_pos, env=None, chunk=5000):
    """Max complex replay error per acquisition family, decoded vs raw positions,
    against a split-half Monte-Carlo floor. `master` is a dict with traj + channels."""
    env = env or default_envelope()
    r = np.asarray(master["traj"], np.float64); dt = float(master["dt_traj"])
    Nt = r.shape[1]; w = master.get("w")
    G, meta = acquisition_battery(Nt, dt, env)
    S_raw = _replay_complex(r, dt, G, w=w, chunk=chunk)
    S_dec = _replay_complex(np.asarray(decoded_pos, np.float64), dt, G,
                            w=(w if decoded_pos.shape[0] == r.shape[0] else None), chunk=chunk)
    # Split-half MC floor over a RANDOM permutation of walkers: master walkers are
    # seeded in compartment order, so a contiguous split mixes compartment fractions
    # and inflates the floor (which would let auto-K accept a too-coarse pack).
    idx = np.random.default_rng(0).permutation(r.shape[0]); h = r.shape[0] // 2
    ia, ib = idx[:h], idx[h:]
    wa = None if w is None else np.asarray(w)[ia]
    wb = None if w is None else np.asarray(w)[ib]
    Sa = _replay_complex(r[ia], dt, G, w=wa, chunk=chunk)
    Sb = _replay_complex(r[ib], dt, G, w=wb, chunk=chunk)
    floor = np.abs(Sa - Sb) / 2.0
    fams = sorted({m["fam"] for m in meta})
    per_fam = {}
    for f in fams:
        idx = np.array([i for i, m in enumerate(meta) if m["fam"] == f])
        per_fam[f] = dict(err_max=float(np.abs(S_dec[idx] - S_raw[idx]).max()),
                          floor_max=float(floor[idx].max()))
    err_all = float(np.abs(S_dec - S_raw).max())
    floor_all = float(floor.max())
    return dict(metric="max_abs_complex_signal_error", err_max=err_all,
                floor_max=floor_all, noise_floor=float(1 / np.sqrt(r.shape[0])),
                within_2x_floor=bool(err_all <= 2 * floor_all), per_family=per_fam)


def measure_floor(master, env=None, chunk=5000):
    """Split-half Monte-Carlo floor (max over the acquisition battery) of the ensemble
    signal — the irreducible precision of a finite-N walk. Scales as ~1/sqrt(N); used to
    size the walker count to a target precision sigma* (see bank.build_to_floor)."""
    env = env or default_envelope()
    r = np.asarray(master["traj"], np.float64); dt = float(master["dt_traj"]); w = master.get("w")
    G, meta = acquisition_battery(r.shape[1], dt, env)
    idx = np.random.default_rng(0).permutation(r.shape[0]); h = r.shape[0] // 2
    ia, ib = idx[:h], idx[h:]
    wa = None if w is None else np.asarray(w)[ia]; wb = None if w is None else np.asarray(w)[ib]
    Sa = _replay_complex(r[ia], dt, G, w=wa, chunk=chunk)
    Sb = _replay_complex(r[ib], dt, G, w=wb, chunk=chunk)
    return float((np.abs(Sa - Sb) / 2.0).max())


def auto_select_modes(X, master, method="lowrank", env=None, tol=2.0, err_target=None,
                      K_grid=(8, 16, 32, 48, 64, 96, 128), chunk=5000, verbose=False):
    """Smallest K whose decoded replay error meets the target. If `err_target` is given
    (absolute, e.g. a precision sigma*), accept K when err_max <= err_target; otherwise
    accept when err_max <= tol * split-half floor. Returns (K, fidelity_report); falls
    back to the largest K if none meet the target."""
    env = env or default_envelope()
    cache = _kl(X) if method in ("lowrank", "gaussian", "marginal") else None
    best = None
    for K in K_grid:
        arrays, meta, _ = ENCODERS[method](X, K, _cache=cache) if cache else ENCODERS[method](X, K)
        pos = decode(arrays, meta, n_walkers=(X.shape[0] if is_walker_preserving(method) else None))
        fid = measure_fidelity(master, pos, env, chunk=chunk)
        if verbose:
            print(f"  K={K}: err={fid['err_max']:.4f} floor={fid['floor_max']:.4f} "
                  f"{'OK' if fid['within_2x_floor'] else '>'}", flush=True)
        best = (K, fid)
        thresh = err_target if err_target is not None else tol * fid["floor_max"]
        if fid["err_max"] <= thresh:
            return K, fid
    return best  # none met target: largest K + its (failing) report
