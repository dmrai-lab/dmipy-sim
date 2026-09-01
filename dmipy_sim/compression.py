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
    from scipy.fft import dct as _dct, idct as _idct, dst as _dst, idst as _idst
except ImportError as _e:  # pragma: no cover
    raise ImportError("dmipy_sim.compression needs scipy (pip install scipy)") from _e

from .constants import GAMMA

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

def encode_bridge_dst(X, K):
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
    X = np.asarray(X, np.float64)
    Nw, Nt, _ = X.shape
    a = X[:, 0, :]
    v = X[:, -1, :] - X[:, 0, :]
    tau = np.arange(Nt) / (Nt - 1.0)
    u = X - (a[:, None, :] + v[:, None, :] * tau[None, :, None])
    K = int(min(K, Nt - 2))
    B = _dst(u[:, 1:-1, :], axis=1, type=1, norm="ortho")[:, :K, :]
    C = np.concatenate([a[:, None, :], v[:, None, :], B], axis=1)   # (Nw, K+2, 3)
    arrays = pack_position_arrays(C, np.float32)
    meta = {"method": "bridge_dst", "K": K, "n_t": int(Nt)}
    return arrays, meta, Nw * 3 * (K + 2) * _F16


def decode_bridge_dst(arrays, meta):
    """Reconstruct positions from endpoints plus sine bands."""
    Nt = int(meta["n_t"])
    C = read_position_coeffs(arrays, dtype=np.float64)
    a, v, B = C[:, 0, :], C[:, 1, :], C[:, 2:, :]
    tau = np.arange(Nt) / (Nt - 1.0)
    u = np.zeros((C.shape[0], Nt, 3))
    if B.shape[1]:
        u[:, 1:-1, :] = _idst(B, axis=1, type=1, norm="ortho", n=Nt - 2)
    return a[:, None, :] + v[:, None, :] * tau[None, :, None] + u


def bridge_moment_rows(G, n_t):
    """Sequence-side rows the two endpoint coefficients pair with: ``(M0, M1/T)``.

    ``M0 = sum_n G_n`` is the refocusing condition and ``M1 = sum_n tau_n G_n`` the
    velocity-compensation one, so a motion-compensated waveform makes both vanish.
    """
    G = np.asarray(G, np.float64)
    tau = np.arange(n_t) / (n_t - 1.0)
    return G.sum(1), (G * tau[None, :, None]).sum(1)






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


def encode_boundary_bridge(dlog, K=16, dtype=np.float32):
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
    A = np.asarray(dlog, np.float64)
    nw, nt = A.shape
    B = np.cumsum(A, axis=1)                                   # (N_w, n_t) smooth
    a = B[:, 0].copy()                                         # exact B(0)
    endpoint = B[:, -1].copy()                                 # exact total local time B(T)
    tau = np.linspace(0.0, 1.0, nt)[None, :]
    resid = B - (a[:, None] + (endpoint - a)[:, None] * tau)   # exactly 0 at BOTH ends
    K = int(min(K, nt - 2))
    C = _dst(resid[:, 1:-1], axis=1, type=1, norm="ortho")[:, :K]
    # ``dtype`` sets the band precision; packs pass f16 via build_replay_pack's ``blt_dtype``. The
    # two ENDPOINTS are always f32 -- they are the exact quantities the rho attenuation and the
    # segment chaining read, where f16's ~3 significant digits would be a real error, not a rounding.
    arrays = {"blt_bridge_dst": C.astype(dtype),
              "blt_start": a.astype(np.float32),
              "blt_endpoint": endpoint.astype(np.float32)}
    meta = {"channel": "boundary_local_time", "mode": "bridge_dst", "n_t": int(nt), "K": int(K),
            "dtype": np.dtype(dtype).name}
    return arrays, meta


def decode_boundary_bridge(arrays, meta):
    """Reconstruct per-save ell(t) = diff(B) from the two endpoints + the pinned sine bands."""
    nt = int(meta["n_t"])
    C = np.asarray(arrays["blt_bridge_dst"], np.float64)
    a = np.asarray(arrays["blt_start"], np.float64)
    endpoint = np.asarray(arrays["blt_endpoint"], np.float64)
    tau = np.linspace(0.0, 1.0, nt)[None, :]
    u = np.zeros((C.shape[0], nt), np.float64)
    u[:, 1:-1] = _idst(C, axis=1, type=1, norm="ortho", n=nt - 2)
    B = u + (a[:, None] + (endpoint - a)[:, None] * tau)
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


CHANNEL_ENCODERS = {POSITION_METHOD: encode_bridge_dst}
CHANNEL_DECODERS = {"bound_fraction": decode_bound_fraction,
                    "boundary_local_time": decode_boundary_local_time,
                    "boundary_bridge": decode_boundary_bridge,
                    "compartment": decode_compartment}


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
    require_position_method(meta["method"])
    return decode_bridge_dst(arrays, meta)


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
def mode_space_phi(arrays, meta, G, dt, n_walkers=None, seed=0):
    """Gradient phase phi_i (N_w, n_meas) in the compressed basis, WITHOUT reconstructing
    the trajectory. Linear in position, so it commutes with every position codec:

    The first two coefficients pair with the gradient moments,
    phi = gamma*dt * [r(0).M0 + (r(T)-r(0)).M1 + sum_k beta_k Ghat_k] with Ghat the sine bands
    of G, so a motion-compensated waveform zeroes the first two columns exactly.
      c(G) = gamma*dt (V.vec(G)) a K-vector per measurement.
    """
    require_position_method(meta.get("method"))
    G = np.asarray(G, np.float64)
    C = read_position_coeffs(arrays, dtype=np.float64)              # (N_w, K+2, n_axes)
    n_t = int(meta["n_t"]); K = C.shape[1] - 2
    if G.shape[1] != n_t:
        raise ValueError(f"waveform has {G.shape[1]} samples, pack walk has n_t={n_t}")
    M0, M1 = bridge_moment_rows(G, n_t)                             # (n_meas, 3) each
    Ghat = _dst(G[:, 1:-1, :], axis=1, type=1, norm="ortho")[:, :K, :]
    W = np.concatenate([M0[:, None, :], M1[:, None, :], Ghat], axis=1)
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
            print(f"  K={K}: err={fid['err_max']:.4f} floor={fid['floor_max']:.4f} "
                  f"{'OK' if fid['within_2x_floor'] else '>'}", flush=True)
        best = (K, fid)
        thresh = err_target if err_target is not None else tol * fid["floor_max"]
        if fid["err_max"] <= thresh:
            return K, fid
    return best
