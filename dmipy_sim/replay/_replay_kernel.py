"""The replay primitives: one gradient phase, one resampler, one spin-echo gate.

Every route that turns a stored walk into a signal -- the raw numpy replay, its JAX twin, the
compressed mode-space replay, the pre-pulse azimuth, the two vector-Bloch replays, the bank's
susceptibility replay and the SH responder -- integrates the same quantity,
``phi = gamma * dt * sum_t G(t) . r(t)``, on the same time grid. They read it from here.

* :func:`effective_gradient` gives a waveform's exact per-save weights against the piecewise-linear path
  through the saves, from the waveform's own grid (:func:`waveform_moments`); no resampling of ``G``.
* :func:`gradient_phase` is the total phase per (measurement, walker); :func:`phase_increments`
  the per-step increments one measurement contributes, for a propagator that applies them in
  order.
* :func:`se_gate` is the transverse-phase sign of a spin echo: ``+1`` before the 180 at
  ``refocus_time``, ``-1`` after, balanced to sum to zero so a static field refocuses exactly.

Each has a JAX twin (``*_jax``) where a consumer differentiates through it.
"""
import numpy as np

from ..constants import GAMMA

try:
    import jax
    import jax.numpy as jnp
    _JAX = True
except ImportError:  # pragma: no cover
    _JAX = False

_FULL = jax.lax.Precision.HIGHEST if _JAX else None     # no TF32 in a phase contraction

_SAME_GRID_RTOL = 1e-9


def _same_grid(dt_wf, dt_traj):
    return abs(float(dt_wf) - float(dt_traj)) / max(abs(float(dt_traj)), 1e-30) <= _SAME_GRID_RTOL


def waveform_moments(G, dt_wf, n_t, dt_pack):
    """Zeroth and first moments of the sample-and-hold waveform ``G`` (``(n_meas, n_wf, 3)``, its own step
    ``dt_wf``) over the walk's save intervals ``[t_k, t_{k+1}]``, ``t_k = k dt_pack``:

        A0[k] = int G dt,    A1[k] = int G (t - t_k) dt,        k = 0 .. n_t - 2.

    Exact (closed form per constant piece), whatever the two grids and wherever the edges fall. A waveform
    may end at ``T = (n_t - 1) dt_pack`` (its last sample sits there and integrates to nothing); one with a
    non-zero sample starting beyond ``T`` is refused, because the stored path ends there.
    """
    G = np.asarray(G, np.float64)
    n_meas, n_wf, _ = G.shape
    dt_wf, dt_pack = float(dt_wf), float(dt_pack)
    T = (int(n_t) - 1) * dt_pack
    e = np.arange(n_wf + 1) * dt_wf                                   # the waveform's interval edges
    beyond = e[:-1] > T * (1.0 + 1e-9)
    if beyond.any() and np.any(G[:, beyond, :] != 0.0):
        raise ValueError(f"the waveform has a non-zero gradient beyond the pack's T_max = {T:.6g} s "
                         f"(its own extent is {e[-1]:.6g} s); the stored path ends there")
    Q0 = np.concatenate([np.zeros((n_meas, 1, 3)), np.cumsum(G, axis=1) * dt_wf], axis=1)          # int_0^t G
    Q1 = np.concatenate([np.zeros((n_meas, 1, 3)),
                         np.cumsum(G * ((e[1:] ** 2 - e[:-1] ** 2) / 2.0)[None, :, None], axis=1)], axis=1)   # int_0^t G t
    tp = np.arange(int(n_t)) * dt_pack
    tc = np.clip(tp, 0.0, e[-1])
    j = np.clip(np.searchsorted(e, tc, side="right") - 1, 0, n_wf - 1)
    Q0p = Q0[:, j, :] + G[:, j, :] * (tc - e[j])[None, :, None]
    Q1p = Q1[:, j, :] + G[:, j, :] * ((tc ** 2 - e[j] ** 2) / 2.0)[None, :, None]
    A0 = np.diff(Q0p, axis=1)
    A1 = np.diff(Q1p, axis=1) - tp[:-1][None, :, None] * A0
    return A0, A1


def effective_gradient(G, dt_wf, n_t, dt_pack):
    """The per-save weights of a waveform against the path: ``(n_meas, n_t, 3)`` such that
    ``gamma dt_pack sum_k Geff[k] . r[k]`` is **exactly** ``gamma int G(t) . r(t) dt`` for the piecewise-linear
    path through the saves (the conditional mean of a Brownian path given its samples) and the
    sample-and-hold waveform on its own grid:

        Geff[k] dt_pack = (A0[k] - A1[k] / dt_pack) + A1[k-1] / dt_pack .

    On the pack's own grid this is the trapezoid rule; off it there is no interpolation of ``G`` at all, so
    an edge between two saves carries exactly its b. Every replay route reads this one function.
    """
    A0, A1 = waveform_moments(G, dt_wf, n_t, dt_pack)
    n_meas = A0.shape[0]
    W = np.zeros((n_meas, int(n_t), 3))
    W[:, :-1, :] += A0 - A1 / float(dt_pack)
    W[:, 1:, :] += A1 / float(dt_pack)
    return W / float(dt_pack)


def effective_gradient_jax(G, dt_wf, n_t, dt_pack):
    """:func:`effective_gradient` for a traced ``G`` (differentiable in ``G``), float32 output."""
    n_meas, n_wf, _ = G.shape
    n_t = int(n_t)
    dt_wf, dt_pack = float(dt_wf), float(dt_pack)
    G = G.astype(jnp.float32)
    e = np.arange(n_wf + 1) * dt_wf
    tp = np.arange(n_t) * dt_pack
    tc = np.clip(tp, 0.0, e[-1])
    j = np.clip(np.searchsorted(e, tc, side="right") - 1, 0, n_wf - 1)           # static: the grids are known
    w1 = jnp.asarray((e[1:] ** 2 - e[:-1] ** 2) / 2.0, jnp.float32)
    Q0 = jnp.concatenate([jnp.zeros((n_meas, 1, 3), jnp.float32), jnp.cumsum(G, axis=1) * jnp.float32(dt_wf)], axis=1)
    Q1 = jnp.concatenate([jnp.zeros((n_meas, 1, 3), jnp.float32), jnp.cumsum(G * w1[None, :, None], axis=1)], axis=1)
    Q0p = Q0[:, j, :] + G[:, j, :] * jnp.asarray(tc - e[j], jnp.float32)[None, :, None]
    Q1p = Q1[:, j, :] + G[:, j, :] * jnp.asarray((tc ** 2 - e[j] ** 2) / 2.0, jnp.float32)[None, :, None]
    A0 = jnp.diff(Q0p, axis=1)
    A1 = jnp.diff(Q1p, axis=1) - jnp.asarray(tp[:-1], jnp.float32)[None, :, None] * A0
    z = jnp.zeros((n_meas, 1, 3), jnp.float32)
    W = jnp.concatenate([A0 - A1 / dt_pack, z], axis=1) + jnp.concatenate([z, A1 / dt_pack], axis=1)
    return W / jnp.float32(dt_pack)


_PHASE_CHUNK_BYTES = 256 * 2 ** 20     # float64 working copy of the trajectory per chunk


def gradient_phase(G_traj, traj, dt, chunk_bytes=_PHASE_CHUNK_BYTES):
    """``gamma * dt * sum_t G[m, t] . r[w, t]`` -> ``(n_meas, n_walkers)`` float64, with ``G_traj`` the
    waveform's per-save weights (:func:`effective_gradient`): the exact integral against the path.

    The contraction runs in float64 over walker chunks of at most ``chunk_bytes`` each, so a stored
    walk of 1e5 walkers by 1e3 steps (2.4 GB as float64) is never materialised whole. Each
    walker's phase is its own dot product, so the chunking changes nothing beyond float64 rounding.
    It is an ``einsum`` and not a matmul because the operands are a handful of measurement rows
    against a transposed walk: BLAS gemm on that shape ran 30x slower than einsum's own kernel here.
    """
    G_flat = np.asarray(G_traj, np.float64).reshape(np.shape(G_traj)[0], -1)
    n_w = traj.shape[0]
    row = int(np.prod(traj.shape[1:]))
    out = np.empty((G_flat.shape[0], n_w))
    step = max(1, int(chunk_bytes // (8 * row)))
    for a in range(0, n_w, step):                      # one float64 chunk alive at a time
        out[:, a:a + step] = np.einsum('mr,wr->mw', G_flat,
                                       np.asarray(traj[a:a + step], np.float64).reshape(-1, row))
    return (GAMMA * float(dt)) * out


def gradient_phase_jax(G_traj, traj, dt):
    """:func:`gradient_phase` for traced operands (float32).

    The contraction is pinned to full float32 precision: on a GPU the default lets XLA run a
    float32 matmul at TF32 (10 mantissa bits), which on a phase of hundreds of radians is an
    error of order 0.1 rad per walker and biases the signal by 10-20%.
    """
    return (float(GAMMA) * float(dt)) * jnp.einsum('mtx,wtx->mw', G_traj.astype(jnp.float32),
                                                   traj.astype(jnp.float32), precision=_FULL)


def phase_increments(G_m, traj, dt):
    """Per-step increments ``gamma * dt * G_m[t] . r[w, t]`` of ONE measurement -> ``(n_t, n_walkers)``,
    for a propagator that rotates the magnetisation step by step (the vector-Bloch replays)."""
    G_m = np.asarray(G_m, np.float64)
    traj = np.asarray(traj, np.float64)
    return (GAMMA * float(dt)) * np.einsum('td,wtd->tw', G_m, traj)


def phase_increment(g_t, r_t, dt):
    """One step of :func:`phase_increments`: ``gamma * dt * r[w] . g`` -> ``(n_walkers,)``."""
    return (GAMMA * float(dt)) * (np.asarray(r_t, np.float64) @ np.asarray(g_t, np.float64))


def phase_increments_jax(G_m, traj, dt):
    """:func:`phase_increments` for traced operands, at full float32 precision (see
    :func:`gradient_phase_jax`)."""
    return (float(GAMMA) * float(dt)) * jnp.einsum('td,wtd->tw', jnp.asarray(G_m), jnp.asarray(traj),
                                                   precision=_FULL)


def se_gate(n_t, dt, refocus_time):
    """The transverse-phase gate of a spin echo as per-save weights: the sign function ``s(t) = +1`` before the
    180 at ``refocus_time`` (s) and ``-1`` after, integrated exactly against the piecewise-linear path through
    the saves (the same reading as :func:`effective_gradient`), so a 180 at ANY instant is exact and a 180 at
    ``T/2`` refocuses a static field to the bit (``sum s = 0``). ``None`` is a gradient echo (``s == +1``).
    Multiply per-save values by ``dt`` and these weights to integrate them over the echo."""
    n_t = int(n_t); dt = float(dt)
    T = (n_t - 1) * dt
    if refocus_time is None:
        lp = np.full(n_t - 1, dt)                                   # +1 everywhere
    else:
        tr = float(np.clip(refocus_time, 0.0, T))
        lp = np.clip(tr - np.arange(n_t - 1) * dt, 0.0, dt)         # the +1 part of each interval
    A0 = lp - (dt - lp)
    A1 = lp ** 2 / 2.0 - (dt ** 2 - lp ** 2) / 2.0
    W = np.zeros(n_t)
    W[:-1] += A0 - A1 / dt
    W[1:] += A1 / dt
    return W / dt