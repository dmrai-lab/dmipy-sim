"""Compression of a master walk: the codecs a replay pack stores its channels in.

A Monte-Carlo master walk is large (N_w walkers x N_t saves x 3), and every replay observable is a
functional of it. The codecs keep walker identity, so the per-walker channels stay aligned and replay
together:

  * positions (C0) -- ``bridge_dst``, the one representation (:func:`encode_bridge_dst`): per axis the
    path's two exact endpoints and the lowest ``K`` sine bands of the residual pinned at both. A
    deliverable gradient is band-limited, so the replay is exact for every acquisition inside the band.
    Any other codec name is refused generically (:func:`require_position_method`). The bands are the
    projections ``beta_k = int u(t) s_k(t) dt`` of the bridge ``u`` on the continuous basis ``s_k = sqrt(2 / T)
    sin(pi k t / T)`` of the window, which a walk accumulates while it runs (:func:`block_table`,
    :func:`cosine_band_scale`); a walk kept at its saves is read the same way, its block moments the increments
    between saves (:func:`encode_bridge_dst`). The readers: :func:`band_projection` (the waveform side, exact per
    sample), :func:`saves_basis` (the bands at instants), the gated contact (:func:`surface_logweight_bridge`) and
    the free band variances (:func:`band_variances`).
  * compartment occupancy (C1) -- named columns, piecewise-constant per walker, stored as row run
    lengths (:func:`rle_encode_rows`) or one static label per walker. The MT bound pool is one of these
    columns, not a channel of its own.
  * boundary local time (C2, rho2/D = 1) -- the cumulative local time ``B(t) = cumsum(ell)`` in the same
    bridge form as the positions (:func:`encode_boundary_bridge`): its two endpoints exact, its residual as
    sine bands. A pack whose arrays use another representation's names (its ``blt_dct_coeffs`` where this reader
    looks for ``blt_bridge_dst``) simply has no channel this reader finds.

:func:`mode_space_signal` computes the replay signal from the coefficients without reconstructing the
trajectory. :func:`auto_select_modes` finds the smallest ``K`` that meets a fidelity tolerance against the
uncompressed walk -- a DIAGNOSTIC (how many bands does this envelope need), called by no producer: a pack's band
is the scanner class's by the storage rule (:func:`dmipy_sim.replay.bank.storage_band_hz`, #643). The band transforms run in numpy/scipy on the host and, on a GPU, as full-precision
JAX matmuls (:func:`dst_bands`).
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

POSITION_METHOD = "bridge_dst"          # the only C0 representation; see encode_bridge_dst
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


def _dst1_to_continuous(K, n_t, T):
    """Per band the factor taking the orthonormal DST-I band of a bridge sampled at ``n_t`` saves to its continuous
    band at block order 0: summation by parts gives the DST-I band ``sqrt(2 / N) / (2 sin(pi k / 2N)) c_k`` with
    ``c_k`` the cosine projection of the increments between saves at their midpoints (``N = n_t - 1``), exactly; the
    continuous band of the same ``c_k`` is :func:`cosine_band_scale`. ``sqrt(dt)`` times a factor that tends to 1
    as ``k / N -> 0``."""
    N = int(n_t) - 1
    if int(K) > N - 1:
        raise ValueError(f"the walk's {int(n_t)} saves hold {N - 1} bands; {int(K)} were asked for")
    k = np.arange(1, int(K) + 1, dtype=np.float64)
    return cosine_band_scale(K, T) * 2.0 * np.sin(np.pi * k / (2.0 * N)) / np.sqrt(2.0 / N)


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
    ``2 D (min(s, t) - s t / T)`` per axis, ``2 D T^2 / (pi k)^2`` -- the limit as ``dt -> 0`` of the DST-I's
    ``D dt / (2 sin^2(pi k / 2N))`` on ``N`` intervals of ``dt``, read in the continuous normalisation (``beta =
    sqrt(dt) beta_dst1``)."""
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
    table. What a continuous-basis pack read on a grid divided by ``a`` takes (``factor = a^(-1/2)``: the basis's
    normalisation ``sqrt(2 / T)`` follows ``T / a``)."""
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


def derive_container(scales, K, floor_target, *, n_t, dt, env=None, budget=BUDGET, kappa=KAPPA, b_min=3,
                     b_max=16, weight=None):
    """The ranges ``[{"bands": [k0, k1], "bits": b}, ...]`` of one common quantisation step across the bands.

    A band quantised at ``b`` bits against its scale ``a_k`` (the block's maximum) has step ``a_k / (2^(b-1) - 1)``
    and noise ``step^2 / 12``. The noise is a random phase per walker with variance
    ``eps^2 = S step^2 / 12`` for one step common to the bands, ``S`` the phase variance per unit step variance of
    the envelope's worst waveform (:func:`_phase_weight`; a contact channel: ``weight^2 K``). It costs the mean signal
    two ways: a random term, ``kappa * eps`` times the walkers' split-half floor, which does not depend on the number
    of walkers, and a bias ``eps^2 / 2``, which does not fall with them while the floor does. Holding both under
    ``budget`` x ``floor_target`` (each at its share, so that the two sum in quadrature to at most ``budget``)::

        eps  = min(budget / (sqrt(2) kappa), sqrt(budget floor_target)),      step = sqrt(12) eps / sqrt(S),
        b_k  = ceil(log2(a_k / step)) + 1,    clipped to [b_min, b_max].

    A band below the step keeps ``b_min`` (an empty range is not carried). Equal-bit runs form ranges grown forward
    to whole bytes, the last taking the remainder, as :func:`band_curve`."""
    scales = np.asarray(scales, np.float64).ravel()[:int(K)]
    eps = container_eps(floor_target, budget=budget, kappa=kappa)
    S = float(weight) ** 2 * int(K) if weight is not None else _phase_weight(int(K), int(n_t), float(dt), _env_key(env))
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


def dst_bands(u, K, *, device="auto", chunk_bytes=None):
    """The lowest ``K`` orthonormal DST-I bands along axis 1 of ``u`` ``(N_w, N, ...)``, as float64: scipy on the
    host, or on the JAX device as one matmul against the sine matrix at full precision (``Precision.HIGHEST``:
    a float32 ``@`` on a CUDA device is TF32 otherwise), in walker chunks of ``chunk_bytes``."""
    K = int(K)
    if resolve_device(device) == "numpy":
        return np.asarray(_dst(np.asarray(u, np.float64), axis=1, type=1, norm="ortho")[:, :K], np.float64)
    import jax
    import jax.numpy as jnp
    S = jnp.asarray(_dst_matrix(int(u.shape[1]), K), jnp.float32)                                    # (N, K)
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


@functools.lru_cache(maxsize=None)
def _device_dct_series(n_t, K, keep):
    """The jitted DCT-II bands of a device chunk of a field series, ``(rows, n_t, n_ch) -> (rows, n_ch', K)``
    float32: one ``einsum`` against the cosine matrix at ``Precision.HIGHEST`` (a float32 ``@`` on a CUDA device
    is TF32 otherwise) -- the field channel's own layout, cached by ``(n_t, K, keep)`` so a walker-chunked call
    traces and compiles ONCE rather than once per chunk (what a fresh ``jax.jit`` closure built inside the
    caller's loop body does not give; the cosine twin of :func:`_device_bridge`). ``keep`` is the static tuple of
    channels to read (``None``: every one) -- a dropped channel (e.g. ``iso_P_zz``) is selected where the device
    already copies host to device, so it costs no separate host copy."""
    import jax
    import jax.numpy as jnp
    n = np.arange(int(n_t))[:, None]; k = np.arange(int(K))[None, :]
    Cm = np.sqrt(2.0 / n_t) * np.cos(np.pi * (2 * n + 1) * k / (2 * n_t)); Cm[:, 0] /= np.sqrt(2.0)
    Cd = jnp.asarray(Cm, jnp.float32)                                                                 # (n_t, K)
    hi = jax.lax.Precision.HIGHEST
    idx = None if keep is None else jnp.asarray(keep, jnp.int32)

    @jax.jit
    def f(x):
        x = x.astype(jnp.float32)
        if idx is not None:
            x = jnp.take(x, idx, axis=2)
        b = jnp.einsum("wnc,nk->wkc", x, Cd, precision=hi)
        return jnp.transpose(b, (0, 2, 1))                                                             # (rows, n_ch', K)

    return f


def dct_bands_series(x, K, *, device="auto", keep=None):
    """The lowest ``K`` orthonormal DCT-II bands of a field-series chunk ``(rows, n_t, n_ch)``, as ``(rows, n_ch',
    K)`` -- the layout :func:`dmipy_sim.replay.bank.susc_path_encode_series` stores: scipy on the host (float64,
    the reference), or the cached device matmul (:func:`_device_dct_series`, float32, ``Precision.HIGHEST``).
    ``keep`` selects channels along axis 2 (``None``: every one); the device route selects AFTER the host-to-
    device copy it makes anyway, so a dropped channel costs no separate host copy -- the numpy route selects
    with one fancy index, same as every other host copy here."""
    K = int(K)
    if resolve_device(device) == "numpy":
        from scipy.fft import dct
        xs = x if keep is None else x[:, :, keep]
        b = dct(np.asarray(xs, np.float64), type=2, norm="ortho", axis=1)[:, :K]
        return np.ascontiguousarray(np.transpose(b, (0, 2, 1)))
    import jax.numpy as jnp
    f = _device_dct_series(int(x.shape[1]), K, None if keep is None else tuple(int(i) for i in keep))
    return np.asarray(f(jnp.asarray(x)))


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


def encode_bridge_dst(X, K, dt, container=None, *, device="auto"):
    """A walk kept at its saves ``X`` ``(N_w, n_t, 3)`` (``dt`` apart) in the pack's form: per axis its two exact
    endpoints and the lowest ``K`` continuous bands of the bridge between them,

        r(t) = r(0) + (t/T)[r(T) - r(0)] + u(t),    u(0) = u(T) = 0,    beta_k = int u s_k dt,

    stored as ``[r(0), r(T)-r(0), beta_1..beta_K]`` per axis. The saves are read as a walk accumulated at block order
    0 over blocks of ``dt``: the cosine projection of the increments between saves at their midpoints, which the
    orthonormal DST-I of the saved bridge holds exactly up to a factor per band (:func:`_dst1_to_continuous`), so the
    bands are formed as that DST-I (scipy's transform, or one full-precision matmul per walker chunk on the device)
    times the factor. The gradient phase is then ``gamma (r(0).M0 + (r(T)-r(0)).M1 + sum_k beta_k W_k)``
    (:func:`band_projection`): refocusing and velocity compensation annihilate the first two columns exactly, both
    endpoints are held exactly, and the sine basis is the variance-optimal one for the bridge (its covariance's
    eigenfunctions). The block order 0 reading costs the replay ``quadrature_rel(0, dt, omega_k)`` of each band."""
    X = X if is_lazy(X) else np.asarray(X)
    Nw, Nt, _ = X.shape
    K = int(min(K, Nt - 2))
    T = (Nt - 1) * float(dt)
    dev = "numpy" if resolve_device(device) == "numpy" else "jax"
    rows = max(1, int(CHUNK_BYTES // max(Nt * 3 * 4, 1)))
    if dev == "jax":
        # the bridge formed on the device per walker chunk: the host forms no line and no residual
        import jax.numpy as jnp
        positions, _ = _device_bridge(Nt, K)
        C = np.empty((Nw, K + 2, 3), np.float64)
        for i in range(0, Nw, rows):
            sl = slice(i, i + rows)
            C[sl] = np.asarray(positions(jnp.asarray(np.asarray(X[sl], np.float32))), np.float64)
    else:
        a = np.asarray(X[:, 0, :], np.float64)
        v = np.asarray(X[:, -1, :], np.float64) - a
        tau = np.arange(Nt) / (Nt - 1.0)
        B = np.empty((Nw, K, 3), np.float64)
        for i in range(0, Nw, rows):                                 # the residual per walker chunk, never the whole walk again
            sl = slice(i, i + rows)
            u = np.asarray(X[sl], np.float64) - (a[sl, None, :] + v[sl, None, :] * tau[None, :, None])
            B[sl] = dst_bands(u[:, 1:-1, :], K, device=dev)
        C = np.concatenate([a[:, None, :], v[:, None, :], B], axis=1)   # (Nw, K+2, 3)
    C[:, 2:, :] *= _dst1_to_continuous(K, Nt, T)[None, :, None]
    meta = {"method": "bridge_dst", "K": K, "T": T}
    if container is None:                                            # the float32 container, one tensor per axis
        arrays = pack_position_arrays(C, np.float32)
        return arrays, meta, Nw * 3 * (K + 2) * _F32
    container = resolve_container(container, C[:, 2:, :], Nt)
    arrays = pack_position_arrays(C, np.float32, container=container)
    meta["container"] = [{"bands": [k0, k1], "bits": b} for k0, k1, b in _band_ranges(container, K)]
    return arrays, meta, int(sum(int(np.asarray(v).nbytes) for v in arrays.values()))


def _bridge_positions(C, Nt, T):
    """Positions ``(n, Nt, 3)`` at the ``Nt`` block ends of a window of ``T`` seconds from the coefficients ``C`` ``(n,
    K+2, 3)``: the chord plus the bands' series at those instants."""
    a, v, B = C[:, 0, :], C[:, 1, :], C[:, 2:, :]
    tau = np.arange(Nt) / (Nt - 1.0)
    u = np.zeros((C.shape[0], Nt, 3))
    if B.shape[1]:
        S, _ = saves_basis(Nt, 0, Nt, B.shape[1], T=T)
        u = np.einsum("wkd,kn->wnd", B, S)
    return a[:, None, :] + v[:, None, :] * tau[None, :, None] + u


def decode_bridge_dst(arrays, meta, walkers=None, *, n_t):
    """Positions at ``n_t`` evenly spaced instants of the window ``meta["T"]`` (both ends included: the reader's grid,
    a pack's block ends by default) from the endpoints and the bands, of every walker or of the slice ``walkers``."""
    C = read_position_coeffs(arrays, dtype=np.float64)
    return _bridge_positions(C if walkers is None else C[walkers], int(n_t), float(meta["T"]))


def decoder(arrays, meta, *, n_t):
    """``(lo, hi) -> positions (hi - lo, n_t, 3)`` of walkers ``lo:hi`` at ``n_t`` evenly spaced instants of the
    window, the coefficients read once: what the certificate reads instead of a decoded copy of the whole walk."""
    require_position_method(meta["method"])
    C = read_position_coeffs(arrays, dtype=np.float64); Nt = int(n_t); T = float(meta["T"])
    return lambda lo, hi: _bridge_positions(C[lo:hi], Nt, T)


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


def encode_boundary_bridge(dlog, K, dt, dtype=np.float32, container=None, *, device="auto"):
    """Bridge codec for the boundary-local-time channel kept at the saves (``dlog`` ``(N_w, n_t)``, the contact of
    the step ending at each save, ``dt`` apart): the CUMULATIVE local time B(t)=cumsum(ell) stored as its two exact
    endpoints plus the continuous bands of the pinned residual -- the same form C0 uses for positions, read at block
    order 0 over the saves as :func:`encode_bridge_dst` reads them.

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

    That is the property the window algebra needs. Splitting into S windows and chaining their
    endpoints drifts by ~1.5e-2 under DCT-II regardless of S, and by <1e-16 under the sine form --
    so a pack can be cut at a window boundary, or two windows merged, without the surface channel
    accumulating error the way a band codec does. ``blt_endpoint`` remains the exact total B(T) that
    the ungated rho2 attenuation reads directly.

    Stores (N_w, K) sine bands + two floats per walker; ratio ~ n_t/K, which GROWS with walk length.
    """
    A = np.asarray(dlog)
    nw, nt = A.shape
    K = int(min(K, nt - 2))
    a = np.empty(nw); endpoint = np.empty(nw); C = np.empty((nw, K), np.float64)
    rows = max(1, int(CHUNK_BYTES // max(nt * 8, 1)))
    if resolve_device(device) != "numpy":
        # the cumulative time, the chord and the bands on the device per chunk (the compressed master's route, #446)
        import jax.numpy as jnp
        _, local_time = _device_bridge(nt, K)
        for i in range(0, nw, rows):
            sl = slice(i, i + rows)
            A64 = np.asarray(A[sl], np.float64)
            a[sl] = A64[:, 0]; endpoint[sl] = A64.sum(axis=1)          # the exact endpoints, float64 on the host as the numpy route
            _, _, b_ = local_time(jnp.asarray(A64.astype(np.float32)))  # the bands on the device
            C[sl] = np.asarray(b_)
    else:
        tau = np.linspace(0.0, 1.0, nt)[None, :]
        for i in range(0, nw, rows):                           # the cumulative time per chunk (float64), its bands
            sl = slice(i, i + rows)
            B = np.cumsum(np.asarray(A[sl], np.float64), axis=1)   # (rows, n_t) smooth
            a[sl] = B[:, 0]                                    # exact B(0)
            endpoint[sl] = B[:, -1]                            # exact total local time B(T)
            resid = B - (a[sl, None] + (endpoint[sl] - a[sl])[:, None] * tau)   # exactly 0 at BOTH ends
            C[sl] = dst_bands(resid[:, 1:-1], K, device=device)
    # ``dtype`` sets the band precision; packs pass f16 via build_replay_pack's ``blt_dtype``. The
    # two ENDPOINTS are always f32 -- they are the exact quantities the rho2 attenuation and the
    # window chaining read, where f16's ~3 significant digits would be a real error, not a rounding.
    T = (nt - 1) * float(dt)
    C *= _dst1_to_continuous(K, nt, T)[None, :]
    return encode_boundary_bands(a, endpoint, C, T=T, dtype=dtype, container=container)


# --------------------------------------------------------- the bridge on the device
@functools.lru_cache(maxsize=None)
def _device_bridge(n_t, K):
    """The jitted bridge of a device batch of ``n_t`` saves at ``K`` bands: ``(positions (b, n_t, 3)) -> (b, K+2, 3)``
    and ``(local time (b, n_t)) -> (start (b,), endpoint (b,), bands (b, K))``, both float32, the bands as one
    matmul against the DST-I matrix at ``Precision.HIGHEST`` (a float32 ``@`` on a CUDA device is TF32 otherwise)."""
    import jax
    import jax.numpy as jnp
    S = jnp.asarray(_dst_matrix(n_t - 2, K), jnp.float32)                                         # (N, K)
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


@functools.lru_cache(maxsize=8)
def _saves_decode_kernel(n_t, start, stop, K, T):
    """``C (c, K+2, 3) -> positions (c, stop - start, 3)`` float32: block ends ``start .. stop - 1`` of a window of ``n_t``
    block ends over ``T`` seconds, the bands evaluated only where they are read."""
    import jax
    import jax.numpy as jnp
    hi = jax.lax.Precision.HIGHEST
    S_part, tau = saves_basis(n_t, start, stop, K, T=T)
    S_part = jnp.asarray(S_part, jnp.float32); tau = jnp.asarray(tau, jnp.float32)

    @jax.jit
    def kernel(C):
        a, v, B = C[:, 0, :], C[:, 1, :], C[:, 2:, :]
        u = jnp.einsum("wkd,kn->wnd", B, S_part, precision=hi)                  # (c, stop - start, 3)
        return a[:, None, :] + v[:, None, :] * tau[None, :, None] + u

    return kernel


def saves_basis(n_t, start, stop, K, *, T):
    """The bands and the chord at the instants ``start .. stop - 1`` of ``n_t`` evenly spaced over a window of ``T``
    seconds (a pack's block ends): ``(S (K, n), tau (n,))``, ``S[k, n] = s_k(t_n) = sqrt(2 / T) sin(pi k t_n / T)``,
    zero at both ends of the window, where the bridge is pinned."""
    s = np.arange(int(start), int(stop))
    tau = s / (n_t - 1.0)
    k = np.arange(1, int(K) + 1, dtype=np.float64)
    return np.sqrt(2.0 / float(T)) * np.sin(np.pi * np.outer(k, tau)), tau


def decode_saves(C, n_t, start, stop, *, T, device="auto", chunk_bytes=1 << 30):
    """The positions ``(n_w, stop - start, 3)`` float32 at the block ends ``start .. stop - 1`` of a window of ``n_t``
    block ends over ``T`` seconds from its coefficients ``C`` ``(n_w, K+2, 3)``: a prefix (``start = 0``) or a stretch
    of the window, the bands evaluated at those instants alone as one product per walker chunk -- on the device, or by
    the host's BLAS."""
    C = np.asarray(C, np.float64)
    n_w, K = C.shape[0], C.shape[1] - 2
    start, stop, n_t = int(start), int(stop), int(n_t)
    if not 0 <= start < stop <= n_t:
        raise ValueError(f"saves {start} .. {stop - 1} are not within a walk of {n_t}")
    out = np.empty((n_w, stop - start, 3), np.float32)
    if resolve_device(device) == "numpy":
        S, tau = saves_basis(n_t, start, stop, K, T=T)
        step = max(1, int(chunk_bytes // (8 * 3 * (stop - start + K + 2))))
        for lo in range(0, n_w, step):
            c = C[lo:lo + step]
            u = np.matmul(np.transpose(c[:, 2:, :], (0, 2, 1)), S)                    # (c, 3, stop - start)
            out[lo:lo + step] = c[:, None, 0, :] + c[:, None, 1, :] * tau[None, :, None] + np.transpose(u, (0, 2, 1))
        return out
    import jax.numpy as jnp
    kernel = _saves_decode_kernel(n_t, start, stop, int(K), float(T))
    step = max(1, int(chunk_bytes // (4 * 3 * (stop - start + K + 2))))
    for lo in range(0, n_w, step):
        sl = slice(lo, min(lo + step, n_w))
        out[sl] = np.asarray(kernel(jnp.asarray(C[sl], jnp.float32)), np.float32)
    return out


def has_c2(arrays):
    """Whether the arrays carry the C2 bridge in either container."""
    return "blt_bridge_dst" in arrays or any(k.startswith("blt_b0") for k in arrays)


def has_c1(arrays):
    """Whether the arrays carry the C1 ``comp`` column in either form (runs or the static label)."""
    return "comp_rle_vals" in arrays or "comp_static" in arrays


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


def decode_boundary_bridge(arrays, meta, walkers=None, n_cut=None, start=0, *, n_t):
    """The contact per interval ``ell = diff(B)`` at ``n_t`` evenly spaced instants of the window ``meta["T"]`` (the
    reader's grid, a pack's block ends by default) from the two exact endpoints and the continuous bands of the
    cumulative local time ``B``, of every walker or of the slice ``walkers``; instants ``start .. n_cut - 1`` only
    when asked (a prefix, or a stretch of the window: ``ell`` at ``start`` is ``B(start) - B(start - 1)``), the bands
    evaluated there alone as one product."""
    nt = int(n_t)
    sl = slice(None) if walkers is None else walkers
    C = bridge_bands(arrays, meta)[sl]
    a = np.asarray(arrays["blt_start"], np.float64)[sl]
    endpoint = np.asarray(arrays["blt_endpoint"], np.float64)[sl]
    stop = nt if n_cut is None else min(int(n_cut), nt)
    start = int(start)
    lo = max(start - 1, 0)                                                                # the block end before the first read
    S, tau = saves_basis(nt, lo, stop, C.shape[1], T=float(meta["T"]))
    B = C @ S + (a[:, None] + (endpoint - a)[:, None] * tau[None, :])
    ell = np.diff(B, axis=1, prepend=B[:, :1] * 0.0) if start == 0 else np.diff(B, axis=1)
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
        return ({f"{name}_static": A[:, 0].astype(np.int8)}, {"name": name, "kind": "static"})
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


def _decode_occupancy_column(arrays, d, n_t=None):
    name = d["name"]
    if d["kind"] == "static":
        lab = np.asarray(arrays[f"{name}_static"], np.int16)
        return lab if n_t is None else np.repeat(lab[:, None], int(n_t), axis=1)
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
    n_t = {d["n_t"] for d in cols if "n_t" in d}
    if len(n_t) > 1:
        raise ValueError(f"C1 columns disagree on the track length: {[(d['name'], d.get('n_t')) for d in cols]}")
    out = {"channel": "compartment", "columns": cols}
    if n_t:
        out["n_t"] = n_t.pop()
    return arrays, out


def encode_static_occupancy(labels):
    """The C1 channel of a walk no walker of which leaves its pool: the one label per walker (``comp_static`` int8)."""
    return ({"comp_static": np.asarray(labels).astype(np.int8)},
            {"channel": "compartment", "columns": [{"name": "comp", "kind": "static"}]})


def decode_occupancy(arrays, meta, *, n_t=None):
    """-> {name: track}. ``comp`` is int16 labels (or a float fraction); others are float [0,1]. A static column is
    the label per walker, broadcast over ``n_t`` instants when given; a track is its stored length."""
    if "columns" not in meta:
        raise ValueError(
            "this pack's C1 metadata predates the occupancy-column layout (no 'columns' key; it "
            "carries the single-track form with 'fractional'/'Q' at the top level). C1 is now a set "
            "of named columns -- the geometric axis under 'comp' plus, for an MT pack, the bound "
            "pool under 'bound' (retiring the separate bound_fraction channel and its bfrac_rle_* "
            "keys). The stored arrays are readable but their meaning is declared differently, so "
            "the pack must be re-encoded from its master rather than reinterpreted here.")
    return {d["name"]: _decode_occupancy_column(arrays, d, n_t) for d in meta["columns"]}


def is_current_c1(meta):
    """True if a pack's C1 channel metadata is in the occupancy-column layout."""
    return isinstance(meta, dict) and "columns" in meta


CHANNEL_DECODERS = {"boundary_bridge": decode_boundary_bridge,
                    "compartment": decode_occupancy}


def encode(X, method, K, dt, **kw):
    """The positions ``X`` kept at their saves ``dt`` apart in the pack's form by ``method``."""
    if method not in ENCODERS:
        raise ValueError(f"unknown method {method!r}; choose from {list(ENCODERS)}")
    return ENCODERS[method](X, K, dt, **kw)


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


def decode(arrays, meta, walkers=None, *, n_t):
    """Positions ``(n_walkers, n_t, 3)`` at ``n_t`` evenly spaced instants of the window ``meta["T"]`` from a pack's
    stored arrays, or those of the slice ``walkers``."""
    require_position_method(meta["method"])
    return decode_bridge_dst(arrays, meta, walkers, n_t=n_t)


# ------------------------------------------------------------ mode-space replay
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
    invT2, invT1 = _rates(T2_per_comp), _rates(T1_per_comp)
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


def relaxation_logweight_runs(arrays, column, T2_per_comp, T1_per_comp, dt, chi=None, active=None, *, n_t=None):
    """:func:`relaxation_logweight` on the stored runs, never on a decoded track: the per-walker log-weight is a
    sum over the walker's runs of the run's rate times the gate's on-time within it, ``dt * r * (Cg[k1] - Cg[k0])``
    with ``Cg`` the gate's prefix sum over the saves. O(runs) -- one run per walker on an impermeable substrate.
    ``column`` is the ``comp`` column's descriptor (``kind`` label or fraction); the gates as in
    :func:`relaxation_logweight`: without ``chi`` the walk relaxes over its ``(n_t - 1) dt`` at T2, and with it the
    longitudinal periods are ``active - chi``. A static column (one label per walker) spans ``n_t`` instants, the
    gate's length when one is given."""
    name = column["name"]
    if column["kind"] == "static":
        n_t = int(len(np.asarray(chi).reshape(-1)) if (n_t is None and chi is not None) else n_t)
    else:
        n_t = int(column["n_t"])
    if column["kind"] == "static":                                                 # one run per walker
        lab = np.asarray(arrays[f"{name}_static"])
        vals, lens, counts = lab, np.full(lab.shape[0], n_t, np.int64), np.ones(lab.shape[0], np.int64)
    else:
        vals = np.asarray(arrays[f"{name}_rle_vals"]); lens = np.asarray(arrays[f"{name}_rle_lens"]); counts = np.asarray(arrays[f"{name}_rle_counts"])
    invT2, invT1 = _rates(T2_per_comp), _rates(T1_per_comp)
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


def surface_logweight_series(blt, rho2_over_D, chi=None):
    """Per-walker surface-relaxivity log-weight ``(rho2/D) sum_k chi_k ell_i(t_k)`` from the
    per-save boundary local-time SERIES ``blt`` (``(n_walkers, n_t)``, stored at ``rho2/D = 1``,
    ``<= 0``). The pack-level entry point, which reads the C2 channel and its exact endpoint, is
    :func:`dmipy_sim.replay.replay.surface_logweight`."""
    blt = np.asarray(blt, np.float64)
    s = blt.sum(1) if chi is None else (np.asarray(chi, float)[None, :] * blt).sum(1)
    return float(rho2_over_D) * s


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
    W = effective_gradient(G, dt, pos.shape[1], dt)                           # (n_meas, n_t, 3)
    # one GEMM over the flattened (save, axis) index: the same contraction the einsum spelled, on BLAS's threads
    # (the einsum ran c_einsum single-threaded: 7.8 s of a 15 s pack on 20k walkers x 873 saves, #642)
    return (GAMMA * dt) * (pos.reshape(pos.shape[0], -1) @ W.reshape(W.shape[0], -1).T)


class LazyWalk:
    """A walk read per walker range from a decoder rather than held whole: ``shape`` ``(n_w, n_t, 3)`` and
    ``walk[lo:hi]`` (a walker slice) and ``walk[:, i, :]`` (one save of every walker), each decoded on demand. What the pack builder, its encoder and its certificate read when the walk is the decoded
    prefix of a pack (dmrai-lab/dmipy-sim#449 item 3): none of them needs the walk whole, and this refuses to
    become one (``np.asarray`` raises) so that no caller does so by accident."""

    def __init__(self, decoder, shape, dtype=np.float32, chunk_bytes=None):
        self._decoder = decoder; self.shape = tuple(int(x) for x in shape); self.dtype = np.dtype(dtype)
        self.chunk_bytes = _chunk(chunk_bytes)

    def __getitem__(self, key):
        if isinstance(key, slice):
            lo, hi, step = key.indices(self.shape[0])
            if step != 1:
                raise IndexError("a lazy walk is read in contiguous walker ranges")
            return np.asarray(self._decoder(int(lo), int(hi)), self.dtype)
        if isinstance(key, tuple) and len(key) == 3 and key[0] == slice(None) and key[2] == slice(None) and np.ndim(key[1]) == 0:
            i = int(key[1]) % self.shape[1]                            # one save of every walker, per chunk
            out = np.empty((self.shape[0], 3), self.dtype)
            step = max(1, int(self.chunk_bytes // (4 * 3 * self.shape[1])))
            for lo in range(0, self.shape[0], step):
                hi = min(lo + step, self.shape[0])
                out[lo:hi] = self[lo:hi][:, i, :]
            return out
        raise IndexError("a lazy walk is read as walk[lo:hi] (walkers) or walk[:, i, :] (one save)")

    def __array__(self, dtype=None, copy=None):
        raise TypeError(f"this walk of {self.shape} is read per walker range, not held whole (LazyWalk)")


def is_lazy(x):
    return isinstance(x, LazyWalk)


def _phases_by_chunk(pos, dt, G, shape=None):
    """:func:`_walker_phases` of every walker, ``(N_w, n_meas)`` float64, taken in walker chunks so that no more
    than :data:`CHUNK_BYTES` of the positions is held in float64 at once: the phases are 40 MB where the trajectory
    is 12 GB, and the certificate needs only the phases. ``pos`` is the positions ``(N_w, n_t, 3)``, a
    :class:`LazyWalk` (read per walker range), or a callable ``(lo, hi) -> positions of walkers lo:hi`` (a
    :func:`decoder`), whose ``shape`` ``(N_w, n_t)`` is then given."""
    if callable(pos) and not is_lazy(pos):
        n_w, n_t = shape; chunk = pos
    else:
        pos = pos if is_lazy(pos) else np.asarray(pos); n_w, n_t = pos.shape[0], pos.shape[1]; chunk = lambda lo, hi: pos[lo:hi]
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


def measure_fidelity(traj, dt_traj, decoded_pos=None, env=None, w=None, logw=None, *, phases=None):
    """Max complex replay error per acquisition family, the stored walk against the raw positions, against a
    split-half Monte-Carlo floor. `logw` (optional) applies the same separable weight to both so surface/relaxation
    packs are scored with their physics on. The stored walk is read as the replay reads it, ``phases(G)`` -> its
    phases ``(N_w, n_meas)`` under the battery's waveforms ``G`` (:func:`coded_phases`, the bands' projection), or
    as positions at the saves (``decoded_pos``: an array or a :func:`decoder`, never held whole); ``decoded_pos``
    ``traj`` itself is a raw walk's floor alone. The raw positions are read in walker chunks of at most
    :data:`CHUNK_BYTES` in float64."""
    env = env or default_envelope()
    r = traj if is_lazy(traj) else np.asarray(traj); dt = float(dt_traj)      # the walk as stored; float64 per chunk below
    G, meta = acquisition_battery(r.shape[1], dt, env)
    phi_raw = _phases_by_chunk(r, dt, G)                          # (N_w, n_meas): all the certificate reads
    S_raw = _mean_signal(phi_raw, w, logw)
    if phases is not None:
        S_dec = _mean_signal(phases(G), w, logw)
    elif decoded_pos is traj:
        S_dec = S_raw
    else:
        S_dec = _mean_signal(_phases_by_chunk(decoded_pos if callable(decoded_pos) else np.asarray(decoded_pos), dt, G,
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
    best = None
    for K in K_grid:
        arrays, meta, _ = ENCODERS[method](X, K, dt_traj)
        pos = decode(arrays, meta)
        fid = measure_fidelity(traj, dt_traj, pos, env, w=w, logw=logw)
        if verbose:
            log.info(f"  K={K}: err={fid['err_max']:.4f} floor={fid['floor_max']:.4f} "
                  f"{'OK' if fid['within_2x_floor'] else '>'}")
        best = (K, fid)
        thresh = err_target if err_target is not None else tol * fid["floor_max"]
        if fid["err_max"] <= thresh:
            return K, fid
    return best
