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


def _make_bloch_step_fn(geometry, D, dt, T2, T1, M0, off_resonance_hz, rho=0.0):
    """Per-timestep forward Bloch scan body (per walker; M is (n_meas, 3)).

    Carry ``(r, M, key, u_crush)``; ``u_crush`` is the walker's fixed macroscopic
    voxel coordinate in [0,1), used only inside crusher windows (0 otherwise).
    ``rho`` (m/s) is the transverse surface relaxivity: at each wall contact ``Mxy``
    is attenuated by ``exp((rho/D) * dlog)`` with ``dlog`` the boundary local time
    (rho/D=1 channel) -- the walk is unchanged, so rho=0 keeps the plain path exact.
    """
    gamma_dt = jnp.float32(GAMMA * dt)
    step_len = jnp.float32(np.sqrt(6.0 * D * dt))
    E2 = jnp.float32(np.exp(-dt / T2)) if T2 is not None else jnp.float32(1.0)
    E1 = jnp.float32(np.exp(-dt / T1)) if T1 is not None else jnp.float32(1.0)
    M0f = jnp.float32(M0)
    global_carrier = jnp.float32(2.0 * np.pi * float(off_resonance_hz) * dt)
    rho_over_D = jnp.float32(rho / D)
    has_surf = rho > 0.0 and hasattr(geometry, 'reflect_with_log_weight')
    reflect = geometry.reflect
    reflect_lw = getattr(geometry, 'reflect_with_log_weight', None)

    def step_fn(carry, inputs):
        r, M, key, uc = carry                               # r:(3,)  M:(n_meas,3)  uc:()
        g_t, rf_dflip, rf_axis, rf_carrier, crush_rate = inputs   # g_t:(n_meas,3)

        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (3,), dtype=jnp.float32)
        unit = noise / jnp.linalg.norm(noise)
        if has_surf:
            r_new, dlog = reflect_lw(r, unit * step_len, jnp.float32(1.0))
            surf = jnp.exp(rho_over_D * dlog)               # transverse wall attenuation
        else:
            r_new = reflect(r, unit * step_len)
            surf = jnp.float32(1.0)

        M = _rf_increment_jax(M, rf_dflip, rf_axis)         # RF rotation (0 -> identity)

        # free-precession phase, per measurement: gradient + global + pulse carrier +
        # the emergent voxel-scale crusher (a per-walker macroscopic phase during a
        # crusher window; the ensemble spread over >> 2 pi dephases the transverse
        # residual while leaving the longitudinally-stored magnetisation untouched).
        dphi = (gamma_dt * (g_t @ r_new) + global_carrier + rf_carrier
                + crush_rate * uc)                          # (n_meas,)
        c, s = jnp.cos(dphi), jnp.sin(dphi)
        Mx = (c * M[:, 0] - s * M[:, 1]) * E2 * surf
        My = (s * M[:, 0] + c * M[:, 1]) * E2 * surf
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
                   echo_steps=None, return_mz=False, crusher=None,
                   surface_relaxivity=0.0,
                   kappa_MT=0.0, dwell_time=0.0, T2_bound=1e-5, T1_bound=1.0,
                   off_resonance_bound=0.0, sub_steps=None, return_bound_frac=False,
                   require_gpu=None):
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

    # Magnetization transfer: dispatch to the fused walk+Bloch path (binding at the
    # walls during the walk, bound-pool relaxation blended in).  kappa_MT == 0 keeps
    # the plain forward path below byte-identical.
    if kappa_MT > 0.0:
        return _simulate_bloch_mt(
            n_walkers, diffusivity, waveform, geometry, rf_events,
            T2=T2, T1=T1, M0=M0, off_resonance_hz=off_resonance_hz, seed=seed,
            echo_steps=echo_steps, return_mz=return_mz, crusher=crusher,
            surface_relaxivity=surface_relaxivity,
            kappa_MT=kappa_MT, dwell_time=dwell_time, T2_bound=T2_bound,
            T1_bound=T1_bound, off_resonance_bound=off_resonance_bound,
            sub_steps=sub_steps, return_bound_frac=return_bound_frac)

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
                                  T2, T1, float(M0), float(off_resonance_hz),
                                  rho=float(surface_relaxivity))
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


# ── magnetization transfer: fused forward walk + binding + Bloch ────────────────
def _geom_radius(geometry):
    """A characteristic feature radius (m) for the binding sub-step auto-tune."""
    for attr in ('radius', 'inner_radius', '_feature_radius'):
        v = getattr(geometry, attr, None)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def _make_bloch_mt_step_fn(geometry, D, dt, n_sub, T2, T1, M0, off_res_global,
                           kappa_MT, dwell_time, T2_bound, T1_bound, off_res_bound,
                           rho=0.0):
    """Per-timestep fused body: an ``n_sub`` binding sub-walk over ``dt``, evolving the
    walker's magnetization STATE-RESOLVED at the sub-step level.  Each sub-step the
    walker is free or bound; the RF nutates it either way (both pools feel B1, as in
    the two-pool Bloch--McConnell system), off-resonance precession and relaxation are
    applied with the walker's CURRENT-state rates (bound -> short-``T2_bound`` -> the
    tipped transverse dephases -> emergent saturation), and the single magnetization
    carries across bind/release = exact longitudinal + transverse exchange (no blend).
    Fine sub-stepping resolves the nutation/precession/relaxation Trotter split, so the
    ensemble converges to the exact two-pool ODE at the Monte-Carlo noise floor.  The
    gradient (diffusion-encoding) phase and crusher stay at the dt level (unchanged for
    the pure-diffusion path).  Carry ``(r, M, key, u_crush, bound_rem)``; scan output
    ``(Mxy, bf)``.  The final ``bound_rem`` flags which walkers are bound at readout, so
    the driver can report the observable FREE-pool signal (the bound pool is MR-dark).
    """
    dt_sub = dt / n_sub
    inv_nsub = jnp.float32(1.0 / n_sub)
    step_len = jnp.float32(np.sqrt(6.0 * D * dt_sub))
    kappa_over_D = jnp.float32(kappa_MT / D)
    dwell_steps_mean = jnp.float32(dwell_time / dt_sub)
    invT2_free = jnp.float32(1.0 / T2) if T2 is not None else jnp.float32(0.0)
    invT1_free = jnp.float32(1.0 / T1) if T1 is not None else jnp.float32(0.0)
    invT2_bound = jnp.float32(1.0 / T2_bound)
    invT1_bound = jnp.float32(1.0 / T1_bound)
    M0f = jnp.float32(M0)
    dt_sub_f = jnp.float32(dt_sub)
    gamma_dt = jnp.float32(GAMMA * dt)
    global_carrier = jnp.float32(2.0 * np.pi * float(off_res_global) * dt)
    bound_carrier = jnp.float32(2.0 * np.pi * float(off_res_bound) * dt)
    rho_over_D = jnp.float32(rho / D)
    reflect_lw = geometry.reflect_with_log_weight

    def step_fn(carry, inputs):
        r, M, key, uc, bound_rem = carry
        g_t, rf_dflip, rf_axis, rf_carrier, crush_rate = inputs
        flip_sub = rf_dflip * inv_nsub                          # RF flip per sub-step
        # off-resonance precession per sub-step (free pools + optional bound offset)
        carr_free_sub = (global_carrier + rf_carrier) * inv_nsub
        carr_bound_sub = bound_carrier * inv_nsub

        def inner(c, _):
            r, M, key, bound_rem = c
            key, step_key, stick_key, dwell_key = jax.random.split(key, 4)
            is_bound = bound_rem > jnp.float32(0.0)
            noise = jax.random.normal(step_key, (3,), dtype=jnp.float32)
            unit = noise / jnp.linalg.norm(noise)
            # rho_over_D = 1 -> dlog = -2 Sum d_perp; the binding local time is -dlog
            r_free, dlog = reflect_lw(r, unit * step_len, jnp.float32(1.0))
            local_time = -dlog
            p_stick = jnp.minimum(jnp.float32(1.0), kappa_over_D * local_time)
            newly = (~is_bound) & (jax.random.uniform(stick_key, dtype=jnp.float32) < p_stick)
            u_dwell = jax.random.uniform(dwell_key, dtype=jnp.float32)
            dwell_draw = -jnp.log(jnp.maximum(u_dwell, jnp.float32(1e-20))) * dwell_steps_mean
            r_next = jnp.where(is_bound, r, r_free)             # frozen while bound
            bound_rem_next = jnp.where(is_bound, bound_rem - jnp.float32(1.0),
                                       jnp.where(newly, dwell_draw, jnp.float32(0.0)))
            # --- spin evolution for THIS sub-step, resolved by state ---
            M = _rf_increment_jax(M, flip_sub, rf_axis)         # nutation (both pools)
            dphi = carr_free_sub + jnp.where(is_bound, carr_bound_sub, jnp.float32(0.0))
            c, s = jnp.cos(dphi), jnp.sin(dphi)
            invT2 = jnp.where(is_bound, invT2_bound, invT2_free)
            invT1 = jnp.where(is_bound, invT1_bound, invT1_free)
            E2, E1 = jnp.exp(-dt_sub_f * invT2), jnp.exp(-dt_sub_f * invT1)
            # surface relaxivity only on surviving FREE contacts (mutual exclusivity)
            surf = jnp.where(is_bound | newly, jnp.float32(1.0),
                             jnp.exp(rho_over_D * dlog))
            Mx = (c * M[:, 0] - s * M[:, 1]) * E2 * surf
            My = (s * M[:, 0] + c * M[:, 1]) * E2 * surf
            Mz = M0f + (M[:, 2] - M0f) * E1
            return (r_next, jnp.stack([Mx, My, Mz], axis=1), key, bound_rem_next), \
                   jnp.where(is_bound, jnp.float32(1.0), jnp.float32(0.0))

        (r_new, M_sub, key, bound_rem_new), bacc = jax.lax.scan(
            inner, (r, M, key, bound_rem), None, length=n_sub)
        bf = jnp.mean(bacc)                                     # bound occupancy this dt
        # dt-level diffusion-encoding gradient phase + crusher (unchanged path)
        dphi_g = gamma_dt * (g_t @ r_new) + crush_rate * uc
        cg, sg = jnp.cos(dphi_g), jnp.sin(dphi_g)
        Mx = cg * M_sub[:, 0] - sg * M_sub[:, 1]
        My = sg * M_sub[:, 0] + cg * M_sub[:, 1]
        M_out = jnp.stack([Mx, My, M_sub[:, 2]], axis=1)
        return (r_new, M_out, key, uc, bound_rem_new), (Mx + 1j * My, bf)

    return step_fn


def _simulate_bloch_mt(n_walkers, diffusivity, waveform, geometry, rf_events, *,
                       T2, T1, M0, off_resonance_hz, seed, echo_steps, return_mz,
                       crusher, surface_relaxivity, kappa_MT, dwell_time, T2_bound,
                       T1_bound, off_resonance_bound, sub_steps, return_bound_frac):
    """MT forward path (see ``simulate_bloch``).  Returns ``signals`` then, in order,
    ``mz`` (if ``return_mz``) and the walker-mean ``bound_frac`` time series (n_t,)
    (if ``return_bound_frac``)."""
    if dwell_time <= 0.0:
        raise ValueError("dwell_time must be > 0 when kappa_MT > 0 (MT on).")
    if not hasattr(geometry, 'reflect_with_log_weight'):
        raise TypeError(f"{type(geometry).__name__} has no reflect_with_log_weight; MT "
                        "binding needs the boundary-local-time channel "
                        "(Sphere / Cylinder / Box1D / Ellipsoid / Mesh).")
    D = float(diffusivity)
    if hasattr(waveform, 'waveform'):
        waveform = waveform.waveform
    G = np.asarray(waveform.G, dtype=np.float64)
    dt = float(waveform.dt)
    _orient_R = getattr(geometry, '_orient_R', None)
    if _orient_R is not None:
        G = G @ np.asarray(_orient_R, dtype=np.float64)
    n_meas, n_t, _ = G.shape

    # binding is trajectory-altering (freezes walkers), so it needs the finer R/25
    # sub-step (divisor 6*25^2 = 3750), not reflection's R/6.
    if sub_steps is None:
        R = _geom_radius(geometry)
        sub_steps = (1 if R is None else
                     max(1, int(np.ceil(dt / (R ** 2 / (3750.0 * D))))))

    dflip, axis, carrier = _build_rf_schedule(rf_events, dt, n_t)
    crush_rate, has_crush = _build_crusher(crusher, dt, n_t)
    G_scan = jnp.asarray(np.transpose(G, (1, 0, 2)), dtype=jnp.float32)
    scan_inputs = (G_scan,
                   jnp.asarray(dflip, dtype=jnp.float32),
                   jnp.asarray(axis, dtype=jnp.float32),
                   jnp.asarray(carrier, dtype=jnp.float32),
                   jnp.asarray(crush_rate, dtype=jnp.float32))

    master_key = jax.random.PRNGKey(seed)
    pos_key, walker_key = jax.random.split(master_key)
    walker_keys = jax.random.split(walker_key, n_walkers)
    r0 = geometry.init_positions(n_walkers, pos_key)
    uw = (jax.random.uniform(jax.random.fold_in(master_key, 0xC0FFEE), (n_walkers,),
                             dtype=jnp.float32) if has_crush
          else jnp.zeros((n_walkers,), dtype=jnp.float32))

    step_fn = _make_bloch_mt_step_fn(geometry, D, dt, int(sub_steps), T2, T1,
                                     float(M0), float(off_resonance_hz), float(kappa_MT),
                                     float(dwell_time), float(T2_bound), float(T1_bound),
                                     float(off_resonance_bound), rho=float(surface_relaxivity))
    M_init = jnp.zeros((n_meas, 3), dtype=jnp.float32).at[:, 2].set(jnp.float32(M0))

    def simulate_walker(r0_w, key_w, uw_w):
        (_, M_f, _, _, bound_rem_f), (xy_seq, bf_seq) = jax.lax.scan(
            step_fn, (r0_w, M_init, key_w, uw_w, jnp.float32(0.0)), scan_inputs)
        return M_f, xy_seq, bf_seq, bound_rem_f

    M_final, xy_seq, bf_seq, bound_rem_final = jax.vmap(
        simulate_walker, in_axes=(0, 0, 0))(r0, walker_keys, uw)
    # (n_w,n_meas,3) (n_w,n_t,n_meas) (n_w,n_t) (n_w,)

    if echo_steps is not None:
        echoes = jnp.mean(xy_seq, axis=0)[jnp.asarray(echo_steps, dtype=int)]
        signals = np.asarray(echoes.T)              # (n_meas, n_echo)
    else:
        signals = np.asarray(jnp.mean(M_final[:, :, 0] + 1j * M_final[:, :, 1], axis=0))

    out = [signals]
    if return_mz:
        # Ensemble-mean Mz over all walkers: each walker is a unit spin toggling between
        # the free and bound environments, so the mean over the whole population is the
        # observable longitudinal magnetization and matches the two-pool oracle's Mz_a to
        # the Monte-Carlo noise floor (the state-resolved sub-step evolution above supplies
        # the exact saturation; no occupancy blend).
        out.append(np.asarray(jnp.mean(M_final[:, :, 2], axis=0)))
    if return_bound_frac:
        out.append(np.asarray(jnp.mean(bf_seq, axis=0)))    # (n_t,)
    return out[0] if len(out) == 1 else tuple(out)
