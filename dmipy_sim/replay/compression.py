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

from ..constants import GAMMA

POSITION_METHOD = "bridge_dst"          # the only C0 representation: start, displacement and sine bands (RPK.md 9.5.3)
ALL_METHODS = (POSITION_METHOD,)

def require_position_method(method):
    """Raise unless ``method`` is the one C0 representation this build reads.

    There is deliberately no fallback: any other codec stores different quantities under the same
    tensor names -- ``bridge_dst`` puts the two endpoints where a band codec puts its two lowest
    bands -- so decoding one as the other yields plausible numbers rather than an error. Refusing
    is the only safe response, by name, for whatever codec a pack declares; the pack must be
    re-encoded from its master with ``build_replay_pack()``.
    """
    if method == POSITION_METHOD:
        return method
    if method is None:
        raise ValueError(
            f"pack declares no position codec. It must declare "
            f"compression.method = {POSITION_METHOD!r}; refusing to assume it, since a pack "
            f"written by an older build stores different quantities under the same names.")
    raise ValueError(
        f"position codec {method!r} is not read by this build. Positions are stored as "
        f"{POSITION_METHOD!r}: two exact endpoints per axis followed by sine bands of the pinned "
        f"residual. Re-encode the pack from its master with build_replay_pack().")
_F16, _F32 = 2, 4


# ------------------------------------------------------------------ the continuous basis
def cosine_band_scale(K, T):
    """Per band ``k = 1..K`` the factor taking the cosine projection of a path's increments, ``c_k = int cos(pi k t /
    T) dx(t)``, to the bridge's band on the continuous basis, ``beta_k = sqrt(2 T) / (pi k) c_k``, ``(K,)`` float64.

    The bridge of a path is ``u(t) = x(t) - x(0) - (t / T) (x(T) - x(0))``, pinned at both ends, so by parts
    ``int u s_k dt = -int S_k du`` with ``S_k = -sqrt(2 / T) (T / pi k) cos(pi k t / T)`` an antiderivative of ``s_k``,
    and ``du = dx - (x(T) - x(0)) dt / T`` adds nothing because the cosine integrates to zero over the window (the
    block table's midpoint sum of it is zero too, ``sum_b cos(pi k (2b + 1) / 2N) = 0``)."""
    k = np.arange(1, int(K) + 1, dtype=np.float64)
    return np.sqrt(2.0 * float(T)) / (np.pi * k)


def cosine_mode_scale(K, T):
    """Per mode ``k = 0..K`` the factor taking the accumulated projection ``sum cos(pi k t / T) b dt`` to the mode on the
    orthonormal cosine basis of the window, ``c_k = int e_k b dt`` with ``e_0 = T^-1/2`` and ``e_k = sqrt(2 / T)
    cos(pi k t / T)`` (RPK.md 9.5.3): ``(K + 1,)`` float64, ``[1 / sqrt(T), sqrt(2 / T), ...]``."""
    out = np.full(int(K) + 1, np.sqrt(2.0 / float(T)))
    out[0] = 1.0 / np.sqrt(float(T))
    return out


def cosine_projection(G, dt_wf, T, K, *, t0=None):
    """THE projection of a sample-and-hold course on a window's orthonormal cosine modes, ``(n_meas, K, n_c)``: ``g_k =
    int G(t) e_k(t - t0) dt`` over the window ``[t0, t0 + T]``, ``k = 0..K-1``, piece by piece in closed form (``int_a^b
    e_k dt = sqrt(2 / T) (sin(omega b) - sin(omega a)) / omega``, ``(b - a) / sqrt(T)`` at ``k = 0``). The field phase
    of a walk stored as the modes ``c_k`` of ``b(t)`` is then ``gamma sum_k g_k c_k`` (Parseval, RPK.md 9.5.4), the
    cosine twin of :func:`band_projection`."""
    T = float(T)
    Gl, a, b, _ = _window_pieces(G, dt_wf, T, t0)
    n_meas, _, n_c = Gl.shape
    out = np.zeros((n_meas, int(K), n_c))
    if int(K) == 0 or a.size == 0:
        return out
    out[:, 0, :] = np.einsum("mnc,n->mc", Gl, b - a) / np.sqrt(T)
    if int(K) > 1:
        k = np.arange(1, int(K))
        om = np.pi * k / T
        W = np.sqrt(2.0 / T) * (np.sin(np.outer(b, om)) - np.sin(np.outer(a, om))) / om[None, :]        # (n_pieces, K-1)
        out[:, 1:, :] = np.einsum("mnc,nk->mkc", Gl, W)
    return out


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


def band_variances(K, T, D):
    """The free bridge's band variances on the continuous basis, ``(K,)``: the eigenvalues of the bridge covariance
    ``2 D (min(s, t) - s t / T)`` per axis, ``2 D T^2 / (pi k)^2`` -- the limit as ``dt -> 0`` of the discrete
    sine transform's ``D dt / (2 sin^2(pi k / 2N))`` on ``N`` intervals of ``dt``, read in the continuous
    normalisation."""
    k = np.arange(1, int(K) + 1, dtype=np.float64)
    return 2.0 * float(D) * float(T) ** 2 / (np.pi * k) ** 2


#: bytes of float64 a continuous projection holds per chunk of bands (the pieces x the bands of one chunk)
_PROJECTION_CHUNK = 1 << 25


def _window_pieces(G, dt_wf, T, t0):
    """The sample-and-hold pieces of ``G`` ``(n_meas, n_wf, n_c)`` within the window ``[t0, t0 + T]``: ``(G_live,
    a, b, live)`` with the pieces' ends on the window's clock, clipped to it, the pieces outside dropped (``live`` the
    samples kept). Without ``t0`` a non-zero sample starting beyond ``T`` is refused (the stored path ends there)."""
    G = np.asarray(G, np.float64)
    e = np.arange(G.shape[1] + 1) * float(dt_wf)
    start = 0.0 if t0 is None else float(t0)
    if t0 is None:
        beyond = e[:-1] > float(T) * (1.0 + 1e-9)
        if beyond.any() and np.any(G[:, beyond, :] != 0.0):
            raise ValueError(f"the waveform has a non-zero gradient beyond the pack's T_max = {float(T):.6g} s "
                             f"(its own extent is {e[-1]:.6g} s); the stored path ends there")
    a = np.clip(e[:-1] - start, 0.0, float(T)); b = np.clip(e[1:] - start, 0.0, float(T))
    live = b > a
    return G[:, live, :], a[live], b[live], live


def _piece_weights(a, b, T, k):
    """``(n_pieces, len(k))``: the integral of ``s_k(t)`` over each piece ``[a, b]`` of the window, ``sqrt(2 / T) (cos(omega
    a) - cos(omega b)) / omega``, written ``2 sin(omega (a + b) / 2) sin(omega (b - a) / 2) / omega`` (no cancellation)."""
    om = np.pi * np.asarray(k, np.float64) / float(T)
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    return np.sqrt(2.0 / float(T)) * 2.0 * np.sin(np.outer(mid, om)) * np.sin(np.outer(half, om)) / om[None, :]


def band_projection(G, dt_wf, T, K, *, t0=None):
    """THE projection of a waveform on a pack's bands, ``(n_meas, K + 2, n_c)``: ``M0 = int G``, ``M1 =
    int G (t - t0) / T`` and ``W_k = int G(t) s_k(t - t0) dt`` over the window ``[t0, t0 + T]``, exactly for the
    sample-and-hold waveform ``G`` ``(n_meas, n_wf, n_c)`` on its own grid ``dt_wf`` (each sample held over its
    interval, as :func:`~dmipy_sim.replay._replay_kernel.effective_gradient` reads it), piece by piece
    (:func:`_piece_weights`). The gradient phase of a walk stored as ``C = [x(0), x(T) - x(0), beta_1..beta_K]`` is
    ``gamma sum C W``; :func:`band_projection_matrix` is the same map as a matrix on the samples."""
    T = float(T)
    Gl, a, b, _ = _window_pieces(G, dt_wf, T, t0)
    n_meas, _, n_c = Gl.shape
    out = np.zeros((n_meas, int(K) + 2, n_c))
    out[:, 0, :] = np.einsum("mnc,n->mc", Gl, b - a)
    out[:, 1, :] = np.einsum("mnc,n->mc", Gl, (b * b - a * a) / (2.0 * T))
    if int(K) == 0 or a.size == 0:
        return out
    step = max(1, int(_PROJECTION_CHUNK // max(8 * a.size, 1)))
    for k0 in range(1, int(K) + 1, step):
        k = np.arange(k0, min(k0 + step, int(K) + 1))
        out[:, 1 + k0:1 + k0 + k.size, :] = np.einsum("mnc,nk->mkc", Gl, _piece_weights(a, b, T, k))
    return out


def segment_projection(segments, T, K, *, t0=None):
    """THE projection of time segments on a pack's bands, ``(n_seg, K + 2)``: per segment ``[a, b]`` (s, on the walk's
    clock) its indicator's ``M0``, ``M1`` and ``W_k`` over the window ``[t0, t0 + T]`` (the segment clipped to it), in
    the closed form :func:`band_projection` sums piece by piece (:func:`_piece_weights`) -- what that projection gives a
    waveform equal to 1 on the segment and 0 elsewhere, wherever the segment's edges fall. A walk stored as ``C =
    [x(0), x(T) - x(0), beta_1..beta_K]`` has ``int_a^b x dt = sum C W`` per axis."""
    T = float(T)
    start = 0.0 if t0 is None else float(t0)
    S = np.asarray(segments, np.float64).reshape(-1, 2)
    a = np.clip(S[:, 0] - start, 0.0, T); b = np.clip(S[:, 1] - start, 0.0, T)
    b = np.maximum(a, b)
    out = np.zeros((S.shape[0], int(K) + 2))
    out[:, 0] = b - a
    out[:, 1] = (b * b - a * a) / (2.0 * T)
    if int(K):
        out[:, 2:] = _piece_weights(a, b, T, np.arange(1, int(K) + 1))
    return out


def band_projection_matrix(n_wf, dt_wf, T, K, *, t0=None):
    """``(K + 2, n_wf)``: :func:`band_projection` as a matrix on a waveform's ``n_wf`` samples (one axis) -- column ``n``
    the projection of the unit sample ``n`` held over its interval, zero for a sample outside the window."""
    T = float(T)
    e = np.arange(int(n_wf) + 1) * float(dt_wf)
    start = 0.0 if t0 is None else float(t0)
    a = np.clip(e[:-1] - start, 0.0, T); b = np.clip(e[1:] - start, 0.0, T)
    out = np.zeros((int(K) + 2, int(n_wf)))
    out[0] = b - a
    out[1] = (b * b - a * a) / (2.0 * T)
    if int(K):
        out[2:] = _piece_weights(a, b, T, np.arange(1, int(K) + 1)).T
    return out


def bridge_phase_variance(G, dt_wf, T, D, *, t0=None):
    """``(n_meas,)``: the variance of ``int G . u dt`` over the window ``[t0, t0 + T]`` for the free bridge ``u`` of a
    walk at diffusivity ``D`` (``gamma^2`` not applied), every band at once -- what the continuous bands carry
    together, so that the phase the bands beyond ``K`` would carry is this less the retained bands' sum. With
    ``q(r) = int_r^T G`` and the bridge covariance ``2 D (min(s, t) - s t / T)`` per axis it is ``2 D (int q^2 - (int
    q)^2 / T)``, ``q`` piecewise linear for a sample-and-hold ``G``, so both integrals are exact."""
    from ._replay_kernel import _cumulative_at
    T = float(T)
    start = 0.0 if t0 is None else float(t0)
    G = np.asarray(G, np.float64)
    e = np.arange(G.shape[1] + 1) * float(dt_wf) - start
    p = np.unique(np.concatenate([[0.0, T], e[(e > 0.0) & (e < T)]]))
    Q0, _ = _cumulative_at(G, dt_wf, start + p)                                  # (n_meas, n_p, n_c)
    q = Q0[:, -1:, :] - Q0                                                        # int_r^T G at the breakpoints
    L = np.diff(p)[None, :, None]
    qa, qb = q[:, :-1, :], q[:, 1:, :]
    int_q2 = (L * (qa * qa + qa * qb + qb * qb) / 3.0).sum(axis=1)
    int_q = (L * (qa + qb) / 2.0).sum(axis=1)
    return 2.0 * float(D) * (int_q2 - int_q * int_q / T).sum(axis=1)


def band_phase_variances(G, dt_wf, T, K, D, *, t0=None):
    """``(per_band (n_meas, K), total (n_meas,))``: the phase variance (rad^2) each of the lowest ``K`` bands carries for
    the waveform ``G`` on a free walk at ``D`` over the window ``T`` -- ``gamma^2 sum_c W_kc^2 var_k`` with
    :func:`band_projection` and :func:`band_variances` -- and what every band of the bridge carries together, the
    closed form :func:`bridge_phase_variance`. The phase the bands beyond ``j`` would carry is ``total -
    per_band[:, :j].sum(1)``, which a dropped independent phase costs the signal half of: the truncation term of a
    pack's band, walk-free."""
    W = band_projection(G, dt_wf, T, int(K), t0=t0)[:, 2:, :]
    pb = GAMMA ** 2 * (W ** 2).sum(axis=2) * band_variances(int(K), T, D)[None, :]
    return pb, GAMMA ** 2 * bridge_phase_variance(G, dt_wf, T, D, t0=t0)


def rescale_bands(arrays, factor):
    """The arrays with every bridge band (positions and contact, any container, any window) multiplied by
    ``factor`` and the exact endpoints untouched: a float container's band columns, an integer container's scale
    table. The field channel's modes ``c_k = int e_k b dt`` carry the same ``sqrt(2 / T)`` and are scaled alike: the
    float tensor ``susc_path_dct``, or the integer ranges' ``susc_path_scale``. What a continuous-basis pack read on a
    grid divided by ``a`` takes (``factor = a^(-1/2)``: the basis's normalisation ``sqrt(2 / T)`` follows ``T / a``)."""
    out = {}
    pos = set(POSITION_AXES)
    for key, v in arrays.items():
        name = key.rsplit("/", 1)[-1]
        if v is None:
            out[key] = v
        elif name in pos:
            a = np.array(v, np.float64, copy=True); a[:, 2:] *= float(factor); out[key] = a
        elif name in ("pos_band_scale", "blt_band_scale", "blt_bridge_dst"):
            out[key] = np.asarray(v, np.float64) * float(factor)
        elif name == "susc_path_scale":
            out[key] = np.asarray(v, np.float64) * float(factor)
        elif name == "susc_path_dct" and np.asarray(v).dtype.kind == "f":      # the float container: no scale table
            out[key] = (np.asarray(v, np.float64) * float(factor)).astype(np.asarray(v).dtype)
        else:
            out[key] = v
    return out


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

NATIVE_BITS = (8, 16)
"""The depths stored as a native integer tensor (``int8``, ``int16``); every other depth in ``MIN_BITS .. 15`` is
bit-packed."""
MIN_BITS = 3


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


def _band_ranges(container, K):
    """``[(k0, k1, bits), ...]`` covering bands ``0 .. K-1`` from a container: ``((upto, bits), ...)`` with ``upto``
    None the end, or the ``[{"bands": [k0, k1], "bits": b}, ...]`` a pack's metadata records and :func:`band_curve`
    returns (the path channel's ranges name ``"modes"`` for ``"bands"``). Refuses a gap, an overlap, or a depth outside ``MIN_BITS .. 16``."""
    out, k0 = [], 0
    if isinstance(container, BandCurve):
        container = container.ranges(K)
    if isinstance(container, DerivedContainer):
        raise ValueError("a derived container is resolved against the measured band scales (resolve_container) before its ranges are read")
    for item in container:
        if isinstance(item, dict):
            upto, bits = int(item["bands"][1] if "bands" in item else item["modes"][1]), int(item["bits"])
        else:
            upto, bits = item
        k1 = int(K) if upto is None else min(int(upto), int(K))
        if not MIN_BITS <= int(bits) <= 16:
            raise ValueError(f"band container bits must be {MIN_BITS}..16 (got {bits})")
        if k1 > k0:
            out.append((k0, k1, int(bits)))
        k0 = max(k0, k1)
    if k0 < int(K):
        raise ValueError(f"band container covers bands up to {k0} of {K}; end it with (None, bits)")
    return out


def range_name(key, i, bits):
    """The tensor name of band range ``i`` at ``bits``: ``{key}_b{i}`` for a native depth, ``{key}_b{i}u{bits}`` for a
    bit-packed one -- the depth is part of the name so a reader without the metadata can unpack."""
    return f"{key}_b{i}" if int(bits) in NATIVE_BITS else f"{key}_b{i}u{int(bits)}"


_RANGE_RE = r"_b(\d+)(?:u(\d+))?"
_PACK_ROWS = 1 << 15


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


def unpack_bits(packed, bits, m):
    """The ``m`` unsigned integers of ``bits`` each per row of ``packed`` ``(N, nbytes)``: the inverse of
    :func:`pack_bits`, ``(N, m)`` ``uint32``."""
    packed = np.asarray(packed)
    w = (1 << np.arange(bits - 1, -1, -1)).astype(np.uint32)
    out = np.empty((packed.shape[0], m), np.uint32)
    for i in range(0, packed.shape[0], _PACK_ROWS):
        b = np.unpackbits(packed[i:i + _PACK_ROWS], axis=1, count=m * bits).reshape(-1, m, bits)
        out[i:i + _PACK_ROWS] = (b.astype(np.uint32) * w).sum(axis=2, dtype=np.uint32)
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


def _load_range(t, bits, m):
    """The integer band values ``(N, m[, n_axes])`` of a stored tensor: :func:`_store_range` inverted."""
    t = np.asarray(t)
    if bits in NATIVE_BITS:
        return t
    off = 1 << (bits - 1)
    if t.ndim == 2:
        return unpack_bits(t, bits, m).astype(np.int32) - off
    return np.stack([unpack_bits(t[..., a], bits, m).astype(np.int32) - off for a in range(t.shape[2])], axis=2)


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
    r0 = container[0]
    t0 = np.asarray(arrays[range_name(key, 0, r0["bits"])])
    scale = band_scales(arrays, scale_key, t0.shape[0])               # (N_w, K) or (N_w, n_axes, K)
    scale = np.swapaxes(scale, 1, 2) if scale.ndim == 3 else scale    # -> (N_w, K[, n_axes])
    parts = []
    for i, r in enumerate(container):
        k0, k1 = r["bands"]
        q = _load_range(arrays[range_name(key, i, r["bits"])], int(r["bits"]), int(k1 - k0))
        parts.append(np.asarray(q, np.float64) * scale[:, k0:k1])
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


def bridge_basis(times, K, T):
    """The bridge's basis at the instants ``times`` (s from the window's start) of a window of ``T`` seconds:
    ``(S (K, n), tau (n,))``, ``S[k, n] = s_k(t_n) = sqrt(2 / T) sin(pi k t_n / T)`` and the chord's fraction ``tau =
    t / T``; ``s_k`` is zero at both ends of the window, where the bridge is pinned."""
    t = np.asarray(times, np.float64).reshape(-1)
    tau = t / float(T)
    k = np.arange(1, int(K) + 1, dtype=np.float64)
    return np.sqrt(2.0 / float(T)) * np.sin(np.pi * np.outer(k, tau)), tau


def bridge_at(C, times, T):
    """The function a stored bridge vector holds, at the instants ``times`` (s from the window's start): ``C`` is
    ``(n_w, K + 2, ...)``, the start, the displacement and the ``K`` bands of each walker (per axis for the positions,
    one channel for the contact), and the value is the chord plus the bands' series, ``(n_w, n, ...)`` float64."""
    C = np.asarray(C, np.float64)
    S, tau = bridge_basis(times, C.shape[1] - 2, T)
    tail = (1,) * (C.ndim - 2)
    out = C[:, None, 0] + C[:, None, 1] * tau.reshape((1, -1) + tail)
    if C.shape[1] > 2:
        out = out + np.einsum("wk...,kn->wn...", C[:, 2:], S)
    return out


def positions_at(arrays, meta, times, walkers=None):
    """Every walker's position at the instants ``times`` (s from the window's start) of a window of ``meta["T"]``
    seconds, ``(n_w, n, 3)`` float64, from the stored endpoints and bands (of the slice ``walkers`` when given)."""
    C = read_position_coeffs(arrays, dtype=np.float64)
    return bridge_at(C if walkers is None else C[walkers], times, float(meta["T"]))


def local_time_at(arrays, meta, times, walkers=None):
    """The cumulative boundary local time ``B(t)`` (at ``rho2 / D = 1``, non-positive) of every walker at the instants
    ``times`` (s from the window's start), ``(n_w, n)`` float64, from the contact channel's exact endpoints and its
    bands (:func:`bridge_bands`; of the slice ``walkers`` when given). The contact over an interval is the
    difference of two of these."""
    sl = slice(None) if walkers is None else walkers
    a = np.asarray(arrays["blt_start"], np.float64)[sl]
    e = np.asarray(arrays["blt_endpoint"], np.float64)[sl]
    vec = np.concatenate([a[:, None], (e - a)[:, None], bridge_bands(arrays, meta)[sl]], axis=1)
    return bridge_at(vec, times, float(meta["T"]))


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


def has_c2(arrays):
    """Whether the arrays carry the C2 bridge in either container."""
    return "blt_bridge_dst" in arrays or any(k.startswith("blt_b0") for k in arrays)


def c2_bands_K(arrays, meta):
    """The number of C2 bands stored, from the metadata or the tensors."""
    if meta and meta.get("K") is not None:
        return int(meta["K"])
    if "blt_bridge_dst" in arrays:
        return int(np.asarray(arrays["blt_bridge_dst"]).shape[1])
    return int(np.asarray(arrays["blt_band_scale"]).shape[-1])


def bridge_bands(arrays, meta, key="blt", scale_key="blt_band_scale", dtype=np.float64):
    """The C2 bands ``(N_w, K)`` from either container: the float tensor ``blt_bridge_dst`` or the integer ranges
    ``blt_b<i>`` with ``blt_band_scale`` (``meta["container"]``)."""
    if "blt_bridge_dst" in arrays:
        return np.asarray(arrays["blt_bridge_dst"], dtype)
    if meta.get("container") is None:
        raise KeyError("this pack's C2 channel is in the integer container but its metadata carries no 'container'")
    return dequantise_bands(arrays, meta["container"], key, scale_key, dtype)


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
# This is why magnetization transfer needs no channel of its own. The bound pool is a column here;
# what makes MT a distinct capability tier is the REPLAY side -- RF as vector-Bloch rotations, the
# bound-pool knobs, and the equilibrium-start requirement -- not the storage.
_EXCLUSIVE_COLUMN = "comp"          # the geometric axis: mutually exclusive, an int label
OCCUPANCY_COLUMNS = ("comp", "bound")


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


def occupancy_events(arrays, column):
    """A column's stored events as ``(start (N_w,), EventList)``: the label every walker holds at its first event
    (the window's start) and its changes after it (:class:`~dmipy_sim.coefficient_walk.EventList`); a static column
    as ``(labels, None)``."""
    from ..coefficient_walk import EventList
    name = column["name"]
    if column["kind"] == "static":
        return np.asarray(arrays[f"{name}_static"]).astype(np.int64), None
    if column["kind"] != "events":
        raise ValueError(f"C1 column {name!r} is stored as {column['kind']!r}, which this build does not read: a column is "
                         "'static' or 'events' (RPK.md 9.5.7)")
    off = np.asarray(arrays[f"{name}_events_offsets"], np.int64)
    t = np.asarray(arrays[f"{name}_events_t"], np.float64)
    lab = np.asarray(arrays[f"{name}_events_label"]).astype(np.int64)
    if np.any(np.diff(off) < 1):
        raise ValueError(f"C1 column {name!r}: every walker's list opens with its label at the window's start; a walker has none")
    heads = off[:-1]
    mask = np.ones(t.shape[0], bool); mask[heads] = False
    counts = np.diff(off) - 1
    return lab[heads], EventList(np.concatenate([[0], np.cumsum(counts)]), t[mask], lab[mask])


def decode_occupancy(arrays, meta):
    """-> ``{name: (start, events)}`` per column (:func:`occupancy_events`): the labels at the window's start and the
    changes after it; a static column's events are ``None``. A pack whose C1 metadata predates the column layout is
    refused by name."""
    if "columns" not in meta:
        raise ValueError(
            "this pack's C1 metadata predates the occupancy-column layout (no 'columns' key); the pack is read by no "
            "revision of this reader. A pack is never converted: re-walk it with a conformant producer.")
    return {d["name"]: occupancy_events(arrays, d) for d in meta["columns"]}


def occupancy_at(arrays, column, times):
    """The labels of every walker at each of ``times`` (s, on the walk's clock), ``(N_w, len(times))``: a column
    evaluated on a reader's instants (the dense oracle, the Bloch route's per-block reading) -- never a stored form."""
    from ..coefficient_walk import events_at_instants
    start, ev = occupancy_events(arrays, column)
    return events_at_instants(ev, start, times)


def is_current_c1(meta):
    """True if a pack's C1 channel metadata is in the occupancy-column layout."""
    return isinstance(meta, dict) and "columns" in meta


def has_c1(arrays):
    """Whether the arrays carry the C1 ``comp`` column in either form (events or the static label)."""
    return "comp_events_offsets" in arrays or "comp_static" in arrays


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
        for k in [k for k in q if re.fullmatch("pos" + _RANGE_RE, k)]:
            out[f"{POSITION_AXES[i]}{k[3:]}"] = np.ascontiguousarray(q[k][:, :, i])
    out["pos_band_scale"] = q["pos_band_scale"]
    return out


def position_ranges(arrays, name):
    """``[(i, bits), ...]`` the band ranges stored under ``{name}_b<i>[u<bits>]``, in band order: the depth of a
    native tensor from its dtype, of a bit-packed one from its name."""
    out = []
    for k in arrays:
        m = re.fullmatch(re.escape(name) + _RANGE_RE, k)
        if m:
            out.append((int(m.group(1)), int(m.group(2)) if m.group(2) else 8 * np.asarray(arrays[k]).dtype.itemsize))
    return sorted(out)


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
        ranges = position_ranges(arrays, POSITION_AXES[present[0]])
        K_b = int(scale.shape[2])
        cols = []
        for i in want:
            ends = np.asarray(arrays[f"{POSITION_AXES[i]}_ends"], np.float64)
            bands, k0 = [], 0
            for r, bits in ranges:
                t = arrays[range_name(POSITION_AXES[i], r, bits)]
                m = int(t.shape[1]) if bits in NATIVE_BITS else min(int(t.shape[1]) * 8 // bits, K_b - k0)
                q = np.asarray(_load_range(t, bits, m), np.float64)
                bands.append(q * scale[:, i, k0:k0 + m]); k0 += m
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


def _rates(times):
    """Per-pool rates ``1 / T`` of a list of times by pool id: ``inf`` is no decay; a time that is not positive is
    refused, since a zero would be an infinite rate and an identically dark signal, not "no decay"."""
    T = np.asarray(times, np.float64).reshape(-1)
    if not (T > 0).all():
        raise ValueError(f"a relaxation time by pool must be positive, float('inf') for no decay; got {T.tolist()}")
    return 1.0 / T


class CumulativeGate:
    """A sample-and-hold gate on its own grid as its running integral: ``gate(t) = int_0^t chi(t') dt'`` for any ``t``
    (s), piecewise linear between the grid's knots, exact for the held samples, and flat beyond the gate's extent
    (its last sample is the readout, held over nothing, RPK.md 6.6) and before ``t = 0``. What a run between two
    events is weighed by: ``gate(t1) - gate(t0)`` is the gate's on-time over the run, with no grid of the pack in
    it."""
    __slots__ = ("knots", "C", "T")

    def __init__(self, chi, dt_wf):
        chi = np.asarray(chi, np.float64).reshape(-1)
        if chi.size < 2:
            raise ValueError(f"a gate needs at least two samples (the last is the readout, held over nothing); got {chi.size}")
        self.knots = np.arange(chi.size) * float(dt_wf)
        self.C = np.concatenate([[0.0], np.cumsum(chi[:-1]) * float(dt_wf)])
        self.T = float(self.knots[-1])

    def __call__(self, t):
        return np.interp(np.asarray(t, np.float64), self.knots, self.C, left=0.0, right=float(self.C[-1]))

    @property
    def total(self):
        return float(self.C[-1])


def relaxation_logweight_runs(arrays, column, T2_per_comp, T1_per_comp, chi, active, *, t0=0.0, t1):
    """The per-walker relaxation log-weight over the window ``[t0, t1]`` from a C1 column's stored form, exact in time
    (RPK.md 9.5.7): a static column is one run per walker, an events column a run per entry from its instant to the
    next entry's (the last to ``t1``), and the log-weight is ``-sum_runs (r2(label) [chi(t_b) - chi(t_a)] +
    r1(label) [lon(t_b) - lon(t_a)])`` with ``chi`` the transverse gate's running integral (:class:`CumulativeGate`)
    and ``lon = active - chi`` the longitudinal periods' (``active`` the acquisition's own extent: a stored period of
    a stimulated echo relaxes at T1, the walk beyond the echo relaxes at nothing). O(events), one run per walker
    when nothing crosses; no grid, no ``dt``."""
    name = column["name"]
    invT2, invT1 = _rates(T2_per_comp), _rates(T1_per_comp)
    lon = lambda t: active(t) - chi(t)
    if column["kind"] == "static":
        lab = np.asarray(arrays[f"{name}_static"]).astype(np.int64)
        on = float(chi(t1) - chi(t0)); off = float(lon(t1) - lon(t0))
        return -(invT2[lab] * on + invT1[lab] * off)
    if column["kind"] != "events":
        raise ValueError(f"C1 column {name!r} is stored as {column['kind']!r}, which this build does not read")
    off_ = np.asarray(arrays[f"{name}_events_offsets"], np.int64)
    t = np.asarray(arrays[f"{name}_events_t"], np.float64)
    lab = np.asarray(arrays[f"{name}_events_label"]).astype(np.int64)
    n_w = off_.shape[0] - 1
    w = np.repeat(np.arange(n_w), np.diff(off_))
    t_a = np.maximum(t, float(t0))
    t_b = np.empty_like(t); t_b[:-1] = t[1:]; t_b[-1] = t1
    t_b[off_[1:] - 1] = float(t1)
    t_b = np.minimum(t_b, float(t1))
    per = invT2[lab] * (chi(t_b) - chi(t_a)) + invT1[lab] * (lon(t_b) - lon(t_a))
    return -np.bincount(w, weights=per, minlength=n_w)


def surface_logweight_bridge(arrays, meta, rho2_over_D, gate=None, dt_gate=None, *, t0=0.0):
    """The gated surface log-weight ``(rho2 / D) int chi(t) dB(t)`` over the window ``[t0, t0 + T]`` from the contact
    channel's continuous bridge ``B(t) = a + (e - a)(t - t0) / T + sum_k beta_k s_k(t - t0)``: the gate ``chi`` held
    sample by sample on its own grid ``dt_gate`` (its last sample the readout, held over nothing), so

        int chi dB = (e - a) <chi>_T + sum_k beta_k sqrt(2 / T) omega_k int chi cos(omega_k (t - t0)) dt,

    each piece's cosine integral in closed form, no series decoded. Without a gate the exact total ``(rho2 / D) (e -
    a)``."""
    a = np.asarray(arrays["blt_start"], np.float64); e = np.asarray(arrays["blt_endpoint"], np.float64)
    if gate is None:
        return float(rho2_over_D) * (e - a)
    T = float(meta["T"])
    chi = np.asarray(gate, np.float64).reshape(-1)
    if chi.size < 2:
        raise ValueError(f"a gate needs at least two samples (the last is the readout, held over nothing); got {chi.size}")
    Gc, lo, hi, _ = _window_pieces(chi[None, :-1, None], dt_gate, T, float(t0))
    g = Gc[0, :, 0]
    C = bridge_bands(arrays, meta); K = C.shape[1]
    om = np.pi * np.arange(1, K + 1) / T
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    Wc = np.sqrt(2.0 / T) * (g[:, None] * 2.0 * np.cos(np.outer(mid, om)) * np.sin(np.outer(half, om))).sum(axis=0)
    return float(rho2_over_D) * ((e - a) * float((g * (hi - lo)).sum()) / T + C @ Wc)


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
    from ._replay_kernel import effective_gradient
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
