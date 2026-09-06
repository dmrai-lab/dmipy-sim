"""The replay primitives: one gradient phase, one resampler, one spin-echo gate.

Every route that turns a stored walk into a signal -- the raw numpy replay, its JAX twin, the
compressed mode-space replay, the pre-pulse azimuth, the two vector-Bloch replays, the bank's
susceptibility replay and the SH responder -- integrates the same quantity,
``phi = gamma * dt * sum_t G(t) . r(t)``, on the same time grid. They read it from here.

* :func:`resample_gradient` puts a waveform on the walk's save grid: linear interpolation of
  each gradient component, zero outside the waveform's window, and plain length alignment when
  the two grids share a step.
* :func:`gradient_phase` is the total phase per (measurement, walker); :func:`phase_increments`
  the per-step increments one measurement contributes, for a propagator that applies them in
  order.
* :func:`se_gate` is the transverse-phase sign of a spin echo: ``+1`` before the 180 at
  ``refocus_time``, ``-1`` after, balanced to sum to zero so a static field refocuses exactly.

Each has a JAX twin (``*_jax``) where a consumer differentiates through it.
"""
import numpy as np

from .constants import GAMMA

try:
    import jax
    import jax.numpy as jnp
    _JAX = True
except ImportError:  # pragma: no cover
    _JAX = False

_SAME_GRID_RTOL = 1e-9


def _same_grid(dt_wf, dt_traj):
    return abs(float(dt_wf) - float(dt_traj)) / max(abs(float(dt_traj)), 1e-30) <= _SAME_GRID_RTOL


def resample_gradient(G, dt_wf, dt_traj, n_t_traj):
    """Waveform ``G`` (``(n_meas, n_t_wf, 3)``, step ``dt_wf``) on the walk grid (``n_t_traj``
    samples of ``dt_traj``): linear interpolation per component, zero outside the waveform.

    On the same grid the waveform is truncated or zero-padded to ``n_t_traj`` samples.
    Returns float64 ``(n_meas, n_t_traj, 3)``.
    """
    G = np.asarray(G, np.float64)
    n_meas, n_t_wf, _ = G.shape
    n_t_traj = int(n_t_traj)
    if _same_grid(dt_wf, dt_traj):
        if n_t_wf == n_t_traj:
            return G
        if n_t_wf > n_t_traj:
            return G[:, :n_t_traj, :]
        out = np.zeros((n_meas, n_t_traj, 3))
        out[:, :n_t_wf, :] = G
        return out
    t_wf = np.arange(n_t_wf) * float(dt_wf)
    t_traj = np.arange(n_t_traj) * float(dt_traj)
    out = np.zeros((n_meas, n_t_traj, 3))
    for m in range(n_meas):
        for ax in range(3):
            out[m, :, ax] = np.interp(t_traj, t_wf, G[m, :, ax], left=0.0, right=0.0)
    return out


def resample_gradient_jax(G, dt_wf, dt_traj, n_t_traj):
    """:func:`resample_gradient` for a traced ``G`` (differentiable), float32 output."""
    n_meas, n_t_wf, _ = G.shape
    n_t_traj = int(n_t_traj)
    G = G.astype(jnp.float32)
    if dt_wf is None or _same_grid(dt_wf, dt_traj):
        if n_t_wf == n_t_traj:
            return G
        if n_t_wf > n_t_traj:
            return G[:, :n_t_traj, :]
        return jnp.concatenate([G, jnp.zeros((n_meas, n_t_traj - n_t_wf, 3), jnp.float32)], axis=1)
    t_wf = jnp.arange(n_t_wf, dtype=jnp.float32) * float(dt_wf)
    t_traj = jnp.arange(n_t_traj, dtype=jnp.float32) * float(dt_traj)

    def one(g_1d):
        return jnp.interp(t_traj, t_wf, g_1d, left=0.0, right=0.0)
    G_t = jax.vmap(jax.vmap(one))(G.transpose(0, 2, 1))          # (n_meas, 3, n_t_traj)
    return G_t.transpose(0, 2, 1)


def gradient_phase(G_traj, traj, dt):
    """``gamma * dt * sum_t G[m, t] . r[w, t]`` -> ``(n_meas, n_walkers)`` float64, with ``G_traj``
    already on the walk grid (:func:`resample_gradient`)."""
    G_traj = np.asarray(G_traj, np.float64)
    traj = np.asarray(traj, np.float64)
    n_meas = G_traj.shape[0]
    n_w = traj.shape[0]
    return (GAMMA * float(dt)) * (G_traj.reshape(n_meas, -1) @ traj.reshape(n_w, -1).T)


def gradient_phase_jax(G_traj, traj, dt):
    """:func:`gradient_phase` for traced operands (float32)."""
    return (float(GAMMA) * float(dt)) * jnp.einsum('mtx,wtx->mw', G_traj.astype(jnp.float32),
                                                   traj.astype(jnp.float32))


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
    """:func:`phase_increments` for traced operands."""
    return (float(GAMMA) * float(dt)) * jnp.einsum('td,wtd->tw', jnp.asarray(G_m), jnp.asarray(traj))


def se_gate(n_t, dt, refocus_time):
    """Transverse-phase gate ``s(t)`` of a spin echo: ``+1`` before the 180 at ``refocus_time``
    (s), ``-1`` after, balanced so that ``sum s = 0`` and a static field refocuses exactly.
    ``None`` is a gradient echo (``s == +1``)."""
    if refocus_time is None:
        return np.ones(int(n_t))
    t = np.arange(int(n_t)) * float(dt)
    s = np.sign(float(refocus_time) - t).astype(float)
    d = int(round(s.sum()))
    if d != 0:
        side = np.where(s == np.sign(d))[0]
        s[side[np.argsort(-np.abs(t[side] - float(refocus_time)))[:abs(d)]]] = 0.0
    return s
