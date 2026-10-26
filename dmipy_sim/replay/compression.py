"""The codecs a replay pack stores its channels in, and the readers of each.

Every channel of a running integral is stored as bands on ONE basis, the continuous one of the storage window
(RPK.md 9.5), accumulated by the walk while it runs (:mod:`dmipy_sim.engine.accumulate`); nothing is encoded from
samples on a grid:

  * positions (C0) -- ``bridge_dst``, the one representation: per axis the path's two exact endpoints and the lowest
    ``K`` sine bands of the residual pinned at both, ``beta_k = int u(t) s_k(t) dt`` with ``s_k = sqrt(2 / T) sin(pi
    k t / T)``; a deliverable gradient is band-limited, so the replay is exact for every acquisition inside the band.
    Any other codec name is refused generically (:func:`require_position_method`). The readers:
    :func:`band_projection` (the waveform side, exact per sample), :func:`bridge_at` / :func:`positions_at` (the
    stored function at instants a reader names), the free band variances (:func:`band_variances`), the block table the walk
    accumulates against (:func:`block_table`, :func:`cosine_band_scale`) and its quadrature (:func:`quadrature_rel`).
  * boundary local time (C2, rho2/D = 1) -- the cumulative local time ``B(t)`` in the same bridge form as the
    positions (:func:`encode_boundary_bands`): its two endpoints exact, its residual as sine bands; the gated
    contact read on the bridge itself (:func:`surface_logweight_bridge`), ``B(t)`` at instants by
    :func:`local_time_at`.
  * the field (C3) -- the cosine modes ``c_k = int e_k b dt`` of the field read along the path
    (:func:`cosine_mode_scale`, :func:`cosine_projection` the gate's side; the tensors in :mod:`dmipy_sim.replay.bank`).
  * compartment occupancy (C1) -- named columns, piecewise constant per walker, stored EXACTLY IN TIME: one static
    label per walker, or the events ``(t, label)`` of every change (:func:`encode_events_occupancy`, RPK.md 9.5.7);
    relaxation is the sum over the runs between events of the rate times the gate's on-time, no grid
    (:func:`relaxation_logweight_runs`). The MT bound pool is one of these columns, not a channel of its own.

The integer band containers (:func:`quantise_bands`, :func:`derive_container`) are a choice of bytes, not of meaning.
"""
from __future__ import annotations
import logging
from dmipy_replay.replay.compression import CHUNK_BYTES, DEVICES, MIN_BITS, OCCUPANCY_COLUMNS, POSITION_AXES, _EXCLUSIVE_COLUMN, _PACK_ROWS, _RANGE_RE, _band_ranges, band_projection, range_name, read_position_coeffs, require_position_method

log = logging.getLogger(__name__)

import re
import math
import functools
import numpy as np

try:
    from scipy.fft import dst as _scipy_dst, idst as _scipy_idst
except ImportError as _e:  # pragma: no cover
    raise ImportError("dmipy_sim.replay.compression needs scipy (pip install scipy)") from _e
import os as _os

#: the threads pocketfft runs the sine transforms on: every walker's transform is independent, so this changes no
#: bit; ``DMIPY_SIM_FFT_WORKERS`` overrides (a shared box), the default is half the cores up to eight (#642)
FFT_WORKERS = int(_os.environ.get("DMIPY_SIM_FFT_WORKERS", 0) or max(1, min(8, (_os.cpu_count() or 2) // 2)))


def _dst(x, **kw):
    return _scipy_dst(x, workers=FFT_WORKERS, **kw)


def _idst(x, **kw):
    return _scipy_idst(x, workers=FFT_WORKERS, **kw)

from dmipy_replay.constants import GAMMA


_F16, _F32 = 2, 4








def block_table(T, n_blocks, K, m, *, blocks=None, xp=np):
    """The basis table of an accumulated walk, ``(n, K + 1, m + 1)``: for each block ``b`` of ``blocks`` (default every
    one of the ``n_blocks`` blocks of length ``h = T / n_blocks`` the window ``T`` is cut into) and each mode
    ``k = 0..K``, the Taylor coefficients of ``cos(pi k t / T)`` about the block's centre ``t_b = (b + 1/2) h`` in
    the block's own unit ``tau = (t - t_b) / h``:

        cos(omega_k t) = sum_{j <= m} table[b, k, j] tau^j + O(tau^(m+1)),
        table[b, k, j] = (omega_k h)^j / j! * cos^(j)(omega_k t_b),      omega_k h = pi k / n_blocks.

    A block adds ``sum_j table[b, k, j] M_j`` to the walk's cosine projection ``c_k``, with ``M_j = sum_i tau_i^j
    dI_i`` the moments of a channel's increments over the block's steps (``tau_i`` each step's midpoint), and
    :func:`cosine_band_scale` turns the projections into the bridge's bands. The table is the same for every
    walker, it depends on ``T`` only through the block count, and the phase ``omega_k t_b = pi r / (2 n_blocks)``
    is reduced exactly in integers, ``r = k (2 b + 1) mod 4 n_blocks``, so a float32 table is as accurate at the
    last block as at the first. ``xp`` is ``numpy`` (float64) or ``jax.numpy`` (float32, ``blocks`` traced): the
    one definition the JAX kernels evaluate in the walk and every other reader evaluates on the host."""
    if not float(T) > 0.0 or int(n_blocks) < 1 or int(K) < 0 or int(m) < 0:
        raise ValueError(f"block_table needs T > 0, n_blocks >= 1, K >= 0 and m >= 0; got {T!r}, {n_blocks!r}, {K!r}, {m!r}")
    N, K, m = int(n_blocks), int(K), int(m)
    host = xp is np
    if not host and K * (2 * N) >= 2 ** 31:
        raise ValueError(f"block_table's integer phase K (2 n_blocks) = {K * 2 * N} exceeds int32; fewer blocks or bands")
    it = np.int64 if host else xp.int32
    ft = np.float64 if host else xp.float32
    b = xp.arange(N, dtype=it) if blocks is None else xp.asarray(blocks).astype(it)
    k = xp.arange(K + 1, dtype=it)
    r = (k[None, :] * (2 * b[:, None] + 1)) % (4 * N)
    theta = r.astype(ft) * ft(np.pi / (2.0 * N))
    c, s = xp.cos(theta), xp.sin(theta)
    wh = k.astype(ft) * ft(np.pi / N)                                              # omega_k h
    cols, fact = [], 1.0
    for j in range(m + 1):
        if j:
            fact *= j
        deriv = (c, -s, -c, s)[j % 4]
        cols.append((wh[None, :] ** j) * deriv / ft(fact))
    return xp.stack(cols, axis=-1)


def quadrature_rel(m, h, omega):
    """The relative variance error the order-``m`` block table (:func:`block_table`) adds to the cosine
    projection of a free path at angular frequency ``omega`` (rad/s) over blocks of ``h`` seconds:

        rel(m, h) = (omega h / 2)^(2 (m + 1)) / ((m + 1)!^2 (2 m + 3)).

    The table's error on block ``b`` is the next Taylor term, ``omega^(m+1) / (m+1)! cos^(m+1)(omega t_b) int tau^(m+1)
    dx``; free increments give it the variance ``2 D int_{-h/2}^{h/2} tau^(2m+2) dtau = 2 D h^(2m+3) / (2^(2m+2)
    (2m + 3))``, summed over the ``T / h`` blocks at the mean ``1/2`` of the squared derivative, against the mode's
    own ``2 D T / 2`` -- 1/12, 1/320 and 1/16128 of ``(omega h)^2``, ``(omega h)^4`` and ``(omega h)^6`` for m = 0, 1,
    2. As a statement on a replay it is ABSOLUTE: the accumulation adds an independent phase of variance at most
    ``rel(m, h) 2 b D_free`` per measurement at the band's top frequency, whatever the substrate, since a restricted
    walk's increments are no larger than a free walk's while its modes' own variances collapse."""
    m = int(m)
    return (np.asarray(omega, np.float64) * float(h) / 2.0) ** (2 * (m + 1)) / (math.factorial(m + 1) ** 2 * (2 * m + 3))






















# --------------------------------------------------------------------- encoders

#: The first container of the registry (0.6): the first 16 bands at 16 bits, the rest at 8. Packs written with it stay
#: readable; it is no longer the default.
TWO_LEVEL_CONTAINER = ((16, 16), (None, 8))


class BandCurve:
    """A container given as a rule rather than as ranges: :func:`band_curve` at ``b1`` and ``taper``, resolved to ranges
    against the K of the pack it is applied to (:func:`_band_ranges`), so one spec serves every band count."""
    __slots__ = ("b1", "taper")

    def __init__(self, b1, taper=1.0):
        self.b1, self.taper = float(b1), float(taper)

    def ranges(self, K):
        return band_curve(K, self.b1, self.taper)

    def __repr__(self):
        return f"BandCurve(b1={self.b1:g}, taper={self.taper:g})"

    def __eq__(self, other):
        return isinstance(other, BandCurve) and (self.b1, self.taper) == (other.b1, other.taper)

    def __hash__(self):
        return hash((self.b1, self.taper))


BUDGET = 0.1
"""The share of the ordered floor the container may spend: the certificate's error against the float pack of the same
walk is derived to sit at ``BUDGET`` x floor."""
KAPPA = 1.0
"""Calibration (radians^-1): the random-phase error of the mean signal, in units of the walkers' split-half floor, per
radian of rms quantisation phase noise (:func:`derive_container`)."""


class DerivedContainer:
    """The default container, derived per pack by :func:`derive_container` from the band scales the encoder measures
    and the floor the pack is ordered at. ``floor`` None is filled by the builder (``sigma_star``, else the walk's own
    measured floor); ``dt`` the battery grid's interval and ``env`` the acquisition envelope the phase weights are read on;
    ``weight`` the contact channel's ``rho2 / D`` ceiling (None: positions, weighted by the envelope's waveforms)."""

    def __init__(self, floor=None, dt=None, env=None, budget=BUDGET, weight=None):
        self.floor, self.dt, self.env, self.budget, self.weight = floor, dt, env, float(budget), weight

    def bound(self, **kw):
        d = dict(floor=self.floor, dt=self.dt, env=self.env, budget=self.budget, weight=self.weight); d.update(kw)
        return DerivedContainer(**d)

    def ranges(self, K, scales, n_t):
        if self.floor is None or (self.weight is None and self.dt is None):
            raise ValueError("a derived container needs the pack's floor target and battery grid; the builder supplies them")
        return derive_container(scales, K, self.floor, n_t=n_t, dt=self.dt, env=self.env, budget=self.budget, weight=self.weight)

    def __repr__(self):
        return f"DerivedContainer(floor={self.floor}, budget={self.budget:g})"


BAND_CONTAINER = DerivedContainer()
"""The default container of the bridge channels: derived per pack (:class:`DerivedContainer`)."""


@functools.lru_cache(maxsize=32)
def _phase_weight(K, n_t, dt, env_key):
    """``max over the envelope's measurements of gamma^2 sum_{k,d} W[m,k,d]^2`` for the lowest ``K`` bands, ``W`` the
    battery's :func:`band_projection` over the window of ``n_t`` blocks of ``dt``: the phase variance per unit
    quantisation variance of one band, summed over bands, for the worst waveform."""
    env = dict(env_key) if env_key else default_envelope()
    G, _ = acquisition_battery(int(n_t), float(dt), env)
    W = band_projection(G, float(dt), (int(n_t) - 1) * float(dt), int(K))[:, 2:, :]
    return float((GAMMA ** 2 * (W ** 2).sum(axis=(1, 2))).max())


def container_eps(floor_target, *, budget=BUDGET, kappa=KAPPA):
    """The rms quantisation phase (radians) a container may add at ``floor_target``: the larger of it costs the mean
    signal ``kappa eps`` floors at random and ``eps^2 / 2`` as a bias, each held at its share of ``budget`` x
    ``floor_target`` so that the two sum in quadrature to at most the budget::

        eps = min(budget / (sqrt(2) kappa), sqrt(budget floor_target))."""
    return min(float(budget) / (np.sqrt(2.0) * float(kappa)), float(np.sqrt(budget * float(floor_target))))


def derive_container(scales, K, floor_target, *, n_t=None, dt=None, env=None, budget=BUDGET, kappa=KAPPA, b_min=3,
                     b_max=16, weight=None, phase=None):
    """The ranges ``[{"bands": [k0, k1], "bits": b}, ...]`` of one common quantisation step across the bands.

    A band quantised at ``b`` bits against its scale ``a_k`` (the block's maximum) has step ``a_k / (2^(b-1) - 1)``
    and noise ``step^2 / 12``. The noise is a random phase per walker with variance
    ``eps^2 = S step^2 / 12`` for one step common to the bands, ``S`` the phase variance per unit step variance of
    the envelope's worst waveform (:func:`_phase_weight`; a contact channel: ``weight^2 K``; a channel that states its
    own, such as the field's ``gamma^2 T``: ``phase``). It costs the mean signal
    two ways: a random term, ``kappa * eps`` times the walkers' split-half floor, which does not depend on the number
    of walkers, and a bias ``eps^2 / 2``, which does not fall with them while the floor does. Holding both under
    ``budget`` x ``floor_target`` (each at its share, so that the two sum in quadrature to at most ``budget``)::

        eps  = min(budget / (sqrt(2) kappa), sqrt(budget floor_target)),      step = sqrt(12) eps / sqrt(S),
        b_k  = ceil(log2(a_k / step)) + 1,    clipped to [b_min, b_max].

    A band below the step keeps ``b_min`` (an empty range is not carried). Equal-bit runs form ranges grown forward
    to whole bytes, the last taking the remainder, as :func:`band_curve`."""
    scales = np.asarray(scales, np.float64).ravel()[:int(K)]
    eps = container_eps(floor_target, budget=budget, kappa=kappa)
    S = (float(phase) if phase is not None else float(weight) ** 2 * int(K) if weight is not None
         else _phase_weight(int(K), int(n_t), float(dt), _env_key(env)))
    step = float(np.sqrt(12.0) * eps / np.sqrt(S))
    bits = np.clip(np.ceil(np.log2(np.maximum(scales, 1e-300) / step)) + 1, b_min, b_max).astype(int)
    return _group_bits(bits)


def _env_key(env):
    return None if env is None else tuple(sorted((k, tuple(map(tuple, v)) if k == "dirs" else (tuple(v) if isinstance(v, (list, tuple)) else v))
                                                 for k, v in env.items()))


def _group_bits(bits):
    """Ranges from per-band bits: equal-bit runs grown forward to whole bytes (``len * bits`` a multiple of 8), the last
    range taking the remainder."""
    K = len(bits)
    out, pos = [], 0
    while pos < K:
        b = int(bits[pos]); end = pos
        while end < K and bits[end] == b:
            end += 1
        step = 8 // math.gcd(8, b)
        end = pos + -(-(end - pos) // step) * step
        end = K if end >= K else end
        if out and out[-1]["bits"] == b:
            out[-1]["bands"][1] = end
        else:
            out.append({"bands": [pos, end], "bits": b})
        pos = end
    return out


def resolve_container(container, B, n_t=None):
    """``container`` with a :class:`DerivedContainer` replaced by its ranges for the bands ``B`` ``(N_w, K[, n_axes])``
    (``n_t`` the battery's grid of a positions container, which the envelope's waveforms are read on)."""
    if isinstance(container, DerivedContainer):
        a = np.abs(np.asarray(B)).max(axis=0)
        return container.ranges(B.shape[1], a.max(axis=1) if a.ndim == 2 else a, n_t)
    return container



def band_curve(K, b1, taper=1.0, b_min=3, b_max=16):
    """The band container of ``K`` bands as a curve: band ``k`` (1-based) holds ``round(b1 - taper * log2(k))``
    bits, clipped to ``[b_min, b_max]``. Contiguous bands of equal bits form one range, and a range's length is
    grown forward to the next multiple of ``8 / gcd(8, bits)`` bands so that ``len * bits`` is a whole number of
    bytes; the last range takes the remainder. Returns ``[{"bands": [k0, k1], "bits": b}, ...]``, the form a pack's
    metadata records and the builder's ``position_container`` accepts."""
    K = int(K)
    if not MIN_BITS <= int(b_min) <= int(b_max) <= 16:
        raise ValueError(f"band_curve needs {MIN_BITS} <= b_min <= b_max <= 16 (got {b_min}, {b_max})")
    bits = np.clip(np.rint(float(b1) - float(taper) * np.log2(np.arange(1, K + 1))), b_min, b_max).astype(int)
    return _group_bits(bits)








def pack_bits(u, bits):
    """Unsigned integers ``u`` ``(N, m)`` below ``2**bits`` as ``(N, ceil(m*bits/8))`` ``uint8``: each row's values
    side by side, MSB first, the last byte zero-padded."""
    u = np.asarray(u)
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint32)
    out = np.empty((u.shape[0], -(-u.shape[1] * bits // 8)), np.uint8)
    for i in range(0, u.shape[0], _PACK_ROWS):
        c = u[i:i + _PACK_ROWS].astype(np.uint32)
        out[i:i + _PACK_ROWS] = np.packbits(((c[:, :, None] >> shifts) & 1).astype(np.uint8).reshape(c.shape[0], -1), axis=1)
    return out




def _store_range(q, bits):
    """The stored tensor of integer band values ``q`` ``(N, m[, n_axes])`` in ``[-lim, lim]`` at ``bits``: native
    ``int8`` / ``int16``, or offset-binary (``q + 2**(bits-1)``) bit-packed along the band axis."""
    if bits == 8:
        return np.ascontiguousarray(q.astype(np.int8))
    if bits == 16:
        return np.ascontiguousarray(q.astype(np.int16))
    u = q.astype(np.int64) + (1 << (bits - 1))
    if q.ndim == 2:
        return pack_bits(u, bits)
    return np.stack([pack_bits(u[..., a], bits) for a in range(q.shape[2])], axis=2)




def quantise_bands(B, container, key, scale_key):
    """The integer container of sine bands ``B`` (``(N_w, K)`` or ``(N_w, K, n_axes)``): per band (and axis) a
    scale ``max |B| / (2^(bits-1) - 1)`` and the rounded integers, one tensor per band range at its depth, under
    :func:`range_name` (a depth other than 8 or 16 is bit-packed, ``(N_w, ceil(m*bits/8)[, n_axes])`` ``uint8``);
    the scales under ``scale_key`` as ``(1, K)`` or ``(1, n_axes, K)`` float32 -- the leading axis is
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
        q = np.rint(B[:, k0:k1] / scale[None, k0:k1])
        arrays[range_name(key, i, bits)] = _store_range(q, bits)
        cont.append({"bands": [int(k0), int(k1)], "bits": int(bits)})
    arrays[scale_key] = np.ascontiguousarray((scale.T if scale.ndim == 2 else scale)[None]).astype(np.float32)
    return arrays, {"container": cont}








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




def _chunk(chunk_bytes):
    return int(CHUNK_BYTES if chunk_bytes is None else chunk_bytes)


def _dst_matrix(N, K):
    """The orthonormal DST-I matrix ``(N, K)``: ``sqrt(2 / (N + 1)) sin(pi n k / (N + 1))`` for ``n = 1..N`` and
    ``k = 1..K``, as scipy's ``dst(type=1, norm="ortho")`` reads it. The one sine matrix every device band, bridge
    and prefix decode contracts against."""
    n = np.arange(1, int(N) + 1)[:, None]; k = np.arange(1, int(K) + 1)[None, :]
    return np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * n * k / (N + 1))


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


@functools.lru_cache(maxsize=None)
def _device_dct_series(n_t, K):
    """The jitted DCT-II bands of a device chunk of a field series, ``(rows, n_t, n_ch) -> (rows, n_ch, K)``
    float32: one ``einsum`` against the cosine matrix at ``Precision.HIGHEST`` (a float32 ``@`` on a CUDA device
    is TF32 otherwise) -- the field channel's own layout, cached by ``(n_t, K)`` so a walker-chunked call
    traces and compiles ONCE rather than once per chunk (what a fresh ``jax.jit`` closure built inside the
    caller's loop body does not give; the cosine twin of :func:`_device_bridge`)."""
    import jax
    import jax.numpy as jnp
    n = np.arange(int(n_t))[:, None]; k = np.arange(int(K))[None, :]
    Cm = np.sqrt(2.0 / n_t) * np.cos(np.pi * (2 * n + 1) * k / (2 * n_t)); Cm[:, 0] /= np.sqrt(2.0)
    Cd = jnp.asarray(Cm, jnp.float32)                                                                 # (n_t, K)
    hi = jax.lax.Precision.HIGHEST

    @jax.jit
    def f(x):
        b = jnp.einsum("wnc,nk->wkc", x.astype(jnp.float32), Cd, precision=hi)
        return jnp.transpose(b, (0, 2, 1))                                                             # (rows, n_ch, K)

    return f


def dct_bands_series(x, K, *, device="auto"):
    """The lowest ``K`` orthonormal DCT-II bands of a field-series chunk ``(rows, n_t, n_ch)``, as ``(rows, n_ch,
    K)`` -- the layout :func:`dmipy_sim.replay.bank.susc_path_encode_series` stores: scipy on the host (float64,
    the reference), or the cached device matmul (:func:`_device_dct_series`, float32, ``Precision.HIGHEST``)."""
    K = int(K)
    if resolve_device(device) == "numpy":
        from scipy.fft import dct
        b = dct(np.asarray(x, np.float64), type=2, norm="ortho", axis=1)[:, :K]
        return np.ascontiguousarray(np.transpose(b, (0, 2, 1)))
    import jax.numpy as jnp
    return np.asarray(_device_dct_series(int(x.shape[1]), K)(jnp.asarray(x)))


def coded_phases(C, dt, G, n_t, *, device="auto", chunk_bytes=None):
    """``(N_w, n_meas)`` gradient phase of every walker under the waveforms ``G`` ``(n_meas, n_t, 3)`` (on the pack's
    block grid of ``n_t`` block ends ``dt`` apart) from the bridge coefficients ``C`` ``(N_w, K+2, 3)`` alone:
    ``gamma sum C W`` with ``W`` the waveforms' :func:`band_projection` -- the replay's own reading of a pack, no
    path decoded (host float64, or the device in float32 at full precision)."""
    C = np.asarray(C); K = int(C.shape[1]) - 2
    W = band_projection(np.asarray(G, np.float64), float(dt), (int(n_t) - 1) * float(dt), K)
    W2 = GAMMA * W.reshape(W.shape[0], -1).T                                                         # ((K+2)*3, n_meas)
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
    (:func:`coded_phases`): the walkers split into two fixed halves, no path decoded. Returns
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










# ------------------------------------------- per-walker physics-channel codecs
def encode_boundary_bands(start, endpoint, bands, *, T, dtype=np.float32, container=None):
    """The C2 channel's tensors and metadata from its bridge already formed: the cumulative local time's exact
    endpoints ``start`` / ``endpoint`` ``(N_w,)`` (always float32) and its continuous ``bands`` ``(N_w, K)`` over the
    window ``T``, the bands at ``dtype`` or in the integer ``container``."""
    C = np.asarray(bands, np.float64)
    K = int(C.shape[1])
    meta = {"channel": "boundary_local_time", "mode": "bridge_dst", "T": float(T), "K": K,
            "dtype": np.dtype(dtype).name}
    arrays = {"blt_start": np.asarray(start, np.float32), "blt_endpoint": np.asarray(endpoint, np.float32)}
    if container is None:
        arrays["blt_bridge_dst"] = C.astype(dtype)
    else:                                                          # the integer container: bands per range, a scale per band
        q, qm = quantise_bands(C, resolve_container(container, C), "blt", "blt_band_scale")
        arrays.update(q); meta["container"] = qm["container"]; meta["dtype"] = "bands"
    return arrays, meta








# C1 is a set of named OCCUPANCY COLUMNS, not a single label track. At every instant a walker has an
# occupancy over the declared pools summing to 1, and replay weights the per-pool rates by it:
# R(t) = sum_c f_c(t) R_c. Every case is that one form -- an integer compartment label is one-hot, and MT binding
# is a label on an INDEPENDENT axis (a walker can be intra-axonal and bound). Storing the axes as separate columns
# is the compact encoding of the joint simplex: f_bound = b, f_geom = (1-b)*[comp == geom].
#
# A column is stored EXACTLY IN TIME (RPK.md 9.5.7): one label per walker when no walker changes it (the static
# form, which a producer MUST write when it applies), else the EVENTS (t, label) of every change, the first at t = 0
# with the initial label -- never a track on a grid. A reader takes the label at an instant as that of the last event
# at or before it, and the integral of a rate over an interval as the sum over the runs between events, exact.
#


def encode_static_occupancy(labels):
    """The C1 channel of a walk no walker of which leaves its pool: the one label per walker (``comp_static`` int8)."""
    return ({"comp_static": np.asarray(labels).astype(np.int8)},
            {"channel": "compartment", "columns": [{"name": "comp", "kind": "static"}]})


def encode_events_occupancy(columns, *, t0=0.0):
    """The C1 channel's tensors and descriptors from its columns' events: ``columns`` maps a column name (``comp``,
    the geometric axis; ``bound``, the MT pool) to ``(start, events)`` -- the label every walker holds at ``t0``
    (the walk's start, or the window's) and an :class:`~dmipy_sim.coefficient_walk.EventList` of its changes after
    it, or ``None`` for a column no walker changes. A changing column is written as ``<c>_events_offsets``
    ``(N_w + 1,)`` int64, ``<c>_events_t`` ``(N_ev,)`` float64 and ``<c>_events_label`` ``(N_ev,)`` int8 (int16 past
    127), every walker's list opening with its label at ``t0`` (RPK.md 9.5.7); a static one as ``<c>_static``."""
    if _EXCLUSIVE_COLUMN not in columns:
        raise ValueError(f"C1 needs the {_EXCLUSIVE_COLUMN!r} column; got {sorted(columns)}")
    arrays, cols = {}, []
    for name, (start, events) in columns.items():
        if name not in OCCUPANCY_COLUMNS:
            raise ValueError(f"a C1 column is one of {OCCUPANCY_COLUMNS}; got {name!r}")
        start = np.asarray(start).astype(np.int64).reshape(-1)
        if events is None:
            arrays[f"{name}_static"] = start.astype(np.int8)
            cols.append({"name": name, "kind": "static"})
            continue
        n_w = start.shape[0]
        if events.n_walkers != n_w:
            raise ValueError(f"column {name!r}: {events.n_walkers} walkers' events for {n_w} start labels")
        counts = events.counts + 1
        off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        t = np.empty(int(off[-1]), np.float64); lab = np.empty(int(off[-1]), np.int64)
        heads = off[:-1]
        t[heads] = float(t0); lab[heads] = start
        mask = np.ones(int(off[-1]), bool); mask[heads] = False
        t[mask] = events.t; lab[mask] = events.label
        itype = np.int8 if lab.size == 0 or lab.max() <= 127 else np.int16
        arrays[f"{name}_events_offsets"] = off
        arrays[f"{name}_events_t"] = t
        arrays[f"{name}_events_label"] = lab.astype(itype)
        cols.append({"name": name, "kind": "events"})
    return arrays, {"channel": "compartment", "columns": cols}












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
        for k in [k for k in q if re.fullmatch("pos" + _RANGE_RE, k)]:
            out[f"{POSITION_AXES[i]}{k[3:]}"] = np.ascontiguousarray(q[k][:, :, i])
    out["pos_band_scale"] = q["pos_band_scale"]
    return out










def mode_space_phi(arrays, meta, G, dt, *, dt_wf=None):
    """Gradient phase phi_i (N_w, n_meas) in the compressed basis, WITHOUT reconstructing the trajectory:
    the exact integral of the waveform ``G`` (on its own grid ``dt_wf``, default ``dt``) against the stored path over
    the window ``meta["T"]``, through :func:`band_projection`. Linear in position, so it
    commutes with the codec: phi = gamma [r(0).M0 + (r(T)-r(0)).M1 + sum_k beta_k W_k], and a
    motion-compensated waveform zeroes the first two columns exactly.
    """
    require_position_method(meta.get("method"))
    C = read_position_coeffs(arrays, dtype=np.float64)              # (N_w, K+2, n_axes)
    W = band_projection(G, dt if dt_wf is None else dt_wf, float(meta["T"]), C.shape[1] - 2)
    return GAMMA * np.einsum("wkd,mkd->wm", C, W)


def mode_space_signal(arrays, meta, G, dt, logw=None, weights=None):
    """Mode-space signal S = <w exp(logw + i phi)> (n_meas,) complex, computing phi
    (mode_space_phi) and the exp reduction, never reconstructing the trajectory. `logw`
    (N_w,) carries the separable relaxation/surface log-weight (see *_logweight)."""
    phi = mode_space_phi(arrays, meta, G, dt)
    n_w = phi.shape[0]
    lw = np.zeros(n_w) if logw is None else np.asarray(logw, float)
    w = np.ones(n_w) if weights is None else np.asarray(weights, float)
    return (np.exp(lw[:, None] + 1j * phi) * (w / w.sum())[:, None]).sum(0)










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
    integral of the on-grid waveform against the piecewise-linear path through a raw subset's block ends ``dt`` apart
    (the per-block-end weights of `effective_gradient`): the dense oracle a measured certificate reads."""
    from dmipy_replay.replay._replay_kernel import effective_gradient
    pos = np.asarray(pos, np.float64); G = np.asarray(G, np.float64)
    W = effective_gradient(G, dt, pos.shape[1], dt)                           # (n_meas, n_t, 3)
    # one GEMM over the flattened (block end, axis) index, on BLAS's threads
    return (GAMMA * dt) * (pos.reshape(pos.shape[0], -1) @ W.reshape(W.shape[0], -1).T)


def _phases_by_chunk(pos, dt, G, shape=None):
    """:func:`_walker_phases` of every walker, ``(N_w, n_meas)`` float64, taken in walker chunks so that no more
    than :data:`CHUNK_BYTES` of the positions is held in float64 at once: the certificate needs only the phases. ``pos`` is the positions ``(N_w, n_t, 3)``, a
    or a callable ``(lo, hi) -> positions of walkers lo:hi``, whose ``shape`` ``(N_w, n_t)`` is then given."""
    if callable(pos):
        n_w, n_t = shape; chunk = pos
    else:
        pos = np.asarray(pos); n_w, n_t = pos.shape[0], pos.shape[1]; chunk = lambda lo, hi: pos[lo:hi]
    step = max(1, int(_chunk(None) // (8 * 3 * n_t)))
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
