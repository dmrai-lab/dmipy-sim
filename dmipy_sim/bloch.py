"""Forward vector-Bloch Monte-Carlo engine (B1 / RF, single pass, no replay).

Alongside the scalar ``cos(phi)`` engine (``core.simulate``), this carries each
walker's full magnetization vector ``M = (Mx, My, Mz)`` through the *actual*
sequence operators in a single forward ``jax.lax.scan`` -- no trajectory storage,
no replay.  Each step applies, in order:

  * the diffusive move + boundary reflection (the same walk as ``core.simulate``);
  * RF pulses (``rf_events``): a Rodrigues rotation about the in-plane B1 axis.  A
    finite pulse is spread over its real duration as partial rotations, so the free
    precession acting between them makes off-resonant / imperfect flips EMERGE;
  * free precession: the PHYSICAL gradient phase ``d phi = gamma * G(t).r(t) * dt``
    rotates ``Mxy`` about z, plus an optional global off-resonance and a per-pulse
    carrier offset (``offset_hz``);
  * relaxation: ``Mxy *= exp(-dt/T2)`` and ``Mz -> M0 + (Mz - M0) exp(-dt/T1)``
    (proper longitudinal recovery toward equilibrium).

Spin echoes and CPMG refocusing are EMERGENT: the 180 rotation conjugates the
accumulated phase, so the echo forms by itself -- there is no ``eps_P`` sign, no
coherence gate, no pathway enumeration.  **Pass the PHYSICAL (same-sign-lobe)
gradient** (``Waveform.G``); the RF rotations do the refocusing, so the
bipolar/effective convention must NOT be used here.

Scope: B1 + relaxation + gradient, single forward pass.  No susceptibility and no
replay (both deliberately out of scope for this public engine).  A magnetization-
transfer bound pool blends onto this same per-step update in later work.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .constants import GAMMA
from .gpu import gpu_available

__all__ = ["simulate_bloch"]


# ── per-step RF rotation (Rodrigues about the in-plane B1 axis) ──────────────────
def _rf_increment_jax(M, flip, ax):
    """Rotate ``M`` (n_meas, 3) by ``flip`` rad about the in-plane axis ``ax`` rad.

    Rotation axis ``u = (cos ax, sin ax, 0)`` (ax = 0 -> +x, ax = pi/2 -> +y).
    ``flip = 0`` is the identity, so a step with no pulse leaves M untouched.
    """
    ux, uy = jnp.cos(ax), jnp.sin(ax)
    c, s = jnp.cos(flip), jnp.sin(flip)
    omc = 1.0 - c
    Mx, My, Mz = M[:, 0], M[:, 1], M[:, 2]
    Mx2 = (c + ux * ux * omc) * Mx + (ux * uy * omc) * My + (uy * s) * Mz
    My2 = (ux * uy * omc) * Mx + (c + uy * uy * omc) * My + (-ux * s) * Mz
    Mz2 = (-uy * s) * Mx + (ux * s) * My + c * Mz
    return jnp.stack([Mx2, My2, Mz2], axis=1)


def _build_rf_schedule(rf_events, dt, n_t):
    """Rasterise ``rf_events`` onto the ``n_t``-step grid.

    Returns per-step arrays ``(dflip, axis, carrier_dphi)``: the incremental flip
    (rad) applied at that step, its B1 axis (rad), and the off-resonance carrier
    phase (rad) accrued over that step.  A finite pulse (``duration_s`` > 0) is
    centred on its nominal time ``t_s`` and spread over ``round(duration_s/dt)``
    steps of equal flip (the excitation at ``t_s = 0`` runs forward from step 0);
    ``duration_s = 0`` is the instantaneous hard-pulse limit (one step).  Distinct
    pulses are not expected to overlap a step.
    """
    dflip = np.zeros(n_t, dtype=np.float64)
    axis = np.zeros(n_t, dtype=np.float64)
    carrier = np.zeros(n_t, dtype=np.float64)
    for e in rf_events:
        i0 = int(round(float(e['t_s']) / dt))
        dur = float(e.get('duration_s', 0.0) or 0.0)
        nsub = max(1, int(round(dur / dt))) if dur > 0.0 else 1
        i_start = max(0, i0 - nsub // 2)
        total = np.deg2rad(float(e.get('flip_deg', 180.0)))
        ax = np.deg2rad(float(e.get('axis_deg', 0.0)))
        off_dphi = 2.0 * np.pi * float(e.get('offset_hz', 0.0) or 0.0) * dt
        per = total / nsub
        for j in range(nsub):
            i = min(i_start + j, n_t - 1)
            dflip[i] += per
            axis[i] = ax
            carrier[i] += off_dphi
    return dflip, axis, carrier


def _make_bloch_step_fn(geometry, D, dt, T2, T1, M0, off_resonance_hz):
    """Per-timestep forward Bloch scan body (per walker; M is (n_meas, 3)).

    Carry ``(r, M, key, u_crush)``; ``u_crush`` is the walker's fixed macroscopic
    voxel coordinate in [0,1), used only inside crusher windows (0 otherwise).
    """
    gamma_dt = jnp.float32(GAMMA * dt)
    step_len = jnp.float32(np.sqrt(6.0 * D * dt))
    E2 = jnp.float32(np.exp(-dt / T2)) if T2 is not None else jnp.float32(1.0)
    E1 = jnp.float32(np.exp(-dt / T1)) if T1 is not None else jnp.float32(1.0)
    M0f = jnp.float32(M0)
    global_carrier = jnp.float32(2.0 * np.pi * float(off_resonance_hz) * dt)
    reflect = geometry.reflect

    def step_fn(carry, inputs):
        r, M, key, uc = carry                               # r:(3,)  M:(n_meas,3)  uc:()
        g_t, rf_dflip, rf_axis, rf_carrier, crush_rate = inputs   # g_t:(n_meas,3)

        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (3,), dtype=jnp.float32)
        unit = noise / jnp.linalg.norm(noise)
        r_new = reflect(r, unit * step_len)

        M = _rf_increment_jax(M, rf_dflip, rf_axis)         # RF rotation (0 -> identity)

        # free-precession phase, per measurement: gradient + global + pulse carrier +
        # the emergent voxel-scale crusher (a per-walker macroscopic phase during a
        # crusher window; the ensemble spread over >> 2 pi dephases the transverse
        # residual while leaving the longitudinally-stored magnetisation untouched).
        dphi = (gamma_dt * (g_t @ r_new) + global_carrier + rf_carrier
                + crush_rate * uc)                          # (n_meas,)
        c, s = jnp.cos(dphi), jnp.sin(dphi)
        Mx = (c * M[:, 0] - s * M[:, 1]) * E2
        My = (s * M[:, 0] + c * M[:, 1]) * E2
        Mz = M0f + (M[:, 2] - M0f) * E1                     # recover toward equilibrium
        return (r_new, jnp.stack([Mx, My, Mz], axis=1), key, uc), (Mx + 1j * My)

    return step_fn


def _build_crusher(crusher, dt, n_t):
    """Per-step crusher phase rate (rad/step) for a per-walker u in [0,1).

    ``crusher`` = ``{'windows_s': [(t0,t1), ...], 'n_cycles': float}`` or None.  Over a
    window of ``n_win`` steps the rate is ``2 pi n_cycles / n_win``, so a walker at
    macroscopic coordinate ``u`` accrues ``2 pi n_cycles u`` across the window and the
    ensemble (u ~ U[0,1)) dephases over ``n_cycles`` turns -- the voxel-scale spoiler.
    """
    rate = np.zeros(n_t, dtype=np.float64)
    if crusher is None:
        return rate, False
    n_cycles = float(crusher.get('n_cycles', 16.0))
    for (t0, t1) in crusher.get('windows_s', []):
        i0, i1 = int(round(t0 / dt)), int(round(t1 / dt))
        i0, i1 = max(0, i0), min(n_t, i1)
        if i1 > i0:
            rate[i0:i1] = 2.0 * np.pi * n_cycles / (i1 - i0)
    return rate, True


def simulate_bloch(n_walkers, diffusivity, waveform, geometry, rf_events, *,
                   T2=None, T1=None, M0=1.0, off_resonance_hz=0.0, seed=0,
                   echo_steps=None, return_mz=False, crusher=None, require_gpu=None):
    """Forward vector-Bloch signal for a sequence of RF pulses on ``waveform.G``.

    Parameters
    ----------
    n_walkers, diffusivity, waveform, geometry : as ``core.simulate`` (``waveform``
        may be a ``Waveform`` or any object with a ``.waveform``; the PHYSICAL
        same-sign gradient is expected).
    rf_events : list of dict
        ``{'t_s', 'flip_deg', 'axis_deg', 'duration_s', 'offset_hz'}`` per pulse;
        the first is usually the excitation.  ``axis_deg`` is the B1 phase (0 = x,
        90 = y); ``duration_s = 0`` is an instantaneous hard pulse; ``offset_hz``
        gives an off-resonance carrier over the pulse (0 = on-resonance).
    T2, T1 : float or None
        Relaxation times (s).  None disables that channel (factor 1).
    M0 : float
        Equilibrium longitudinal magnetization (per walker; uniform here).
    off_resonance_hz : float
        Global spin off-resonance (Hz) applied every step (a voxel B0 offset).
    echo_steps : sequence of int, optional
        Step indices at which to record the (walker-mean) transverse signal, e.g.
        a CPMG echo train.  Returns ``(n_meas, n_echo)`` complex when given.
    return_mz : bool
        Also return the final walker-mean ``Mz`` (n_meas,).

    Returns
    -------
    signals : (n_meas,) complex        walker-mean ``Mx + i My`` at the last step
        (or ``(n_meas, n_echo)`` when ``echo_steps`` is given), optionally with
        ``Mz`` appended when ``return_mz``.
    """
    if require_gpu and not gpu_available():
        raise RuntimeError("simulate_bloch(require_gpu=True) but no CUDA device is "
                           "visible to JAX.")

    if hasattr(waveform, 'waveform'):
        waveform = waveform.waveform
    G = np.asarray(waveform.G, dtype=np.float64)            # (n_meas, n_t, 3)
    dt = float(waveform.dt)
    _orient_R = getattr(geometry, '_orient_R', None)
    if _orient_R is not None:
        G = G @ np.asarray(_orient_R, dtype=np.float64)
    n_meas, n_t, _ = G.shape

    dflip, axis, carrier = _build_rf_schedule(rf_events, dt, n_t)
    crush_rate, has_crush = _build_crusher(crusher, dt, n_t)
    G_scan = jnp.asarray(np.transpose(G, (1, 0, 2)), dtype=jnp.float32)   # (n_t,n_meas,3)
    scan_inputs = (G_scan,
                   jnp.asarray(dflip, dtype=jnp.float32),
                   jnp.asarray(axis, dtype=jnp.float32),
                   jnp.asarray(carrier, dtype=jnp.float32),
                   jnp.asarray(crush_rate, dtype=jnp.float32))

    # pos_key / walker_key keep the SAME 2-way split as core.simulate (identical walk
    # for parity); the crusher's per-walker macro coordinate is an independent stream.
    master_key = jax.random.PRNGKey(seed)
    pos_key, walker_key = jax.random.split(master_key)
    walker_keys = jax.random.split(walker_key, n_walkers)
    r0 = geometry.init_positions(n_walkers, pos_key)        # (n_walkers, 3)
    if has_crush:
        uw = jax.random.uniform(jax.random.fold_in(master_key, 0xC0FFEE), (n_walkers,),
                                dtype=jnp.float32)
    else:
        uw = jnp.zeros((n_walkers,), dtype=jnp.float32)

    step_fn = _make_bloch_step_fn(geometry, float(diffusivity), dt,
                                  T2, T1, float(M0), float(off_resonance_hz))
    M_init = jnp.zeros((n_meas, 3), dtype=jnp.float32).at[:, 2].set(jnp.float32(M0))

    want_echo = echo_steps is not None

    def simulate_walker(r0_w, key_w, uw_w):
        (r_f, M_f, _, _), xy_seq = jax.lax.scan(
            step_fn, (r0_w, M_init, key_w, uw_w), scan_inputs)
        return (M_f, xy_seq) if want_echo else M_f

    if want_echo:
        M_final, xy_seq = jax.vmap(simulate_walker, in_axes=(0, 0, 0))(r0, walker_keys, uw)
        # xy_seq: (n_walkers, n_t, n_meas) complex -> walker-mean at the echo steps
        echoes = jnp.mean(xy_seq, axis=0)[jnp.asarray(echo_steps, dtype=int)]   # (n_echo,n_meas)
        signals = np.asarray(echoes.T)                     # (n_meas, n_echo)
    else:
        M_final = jax.vmap(simulate_walker, in_axes=(0, 0, 0))(r0, walker_keys, uw)  # (n_w,n_meas,3)
        signals = np.asarray(jnp.mean(M_final[:, :, 0] + 1j * M_final[:, :, 1], axis=0))

    if return_mz:
        mz = np.asarray(jnp.mean(M_final[:, :, 2], axis=0))
        return signals, mz
    return signals
