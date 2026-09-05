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

import warnings

import numpy as np
import jax
import jax.numpy as jnp
from .geometry._boundary import bind_probability

from .constants import GAMMA
from .gpu import gpu_available
from .geometry import initial_positions
from .physics import (permeable_sub_steps, walk_sub_steps,
                      _warn_if_step_outruns_the_lookup)

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


def _make_bloch_step_fn(geometry, D, dt, T2, T1, M0, off_resonance_hz, rho=0.0,
                        field_fn=None, sub_steps=None):
    """Per-timestep forward Bloch scan body (per walker; M is (n_meas, 3)).

    Carry ``(r, M, key, u_crush)``; ``u_crush`` is the walker's fixed macroscopic
    voxel coordinate in [0,1), used only inside crusher windows (0 otherwise).
    ``rho`` (m/s) is the transverse surface relaxivity: at each wall contact ``Mxy``
    is attenuated by ``exp((rho/D) * dlog)`` with ``dlog`` the boundary local time
    (rho/D=1 channel) -- the walk is unchanged, so rho=0 keeps the plain path exact.
    ``field_fn`` (optional) maps the walker position ``r -> ΔBz`` (Tesla): a
    susceptibility-induced off-resonance field, added to the z-precession as
    ``gamma * ΔBz(r) * dt`` -- refocused by the sequence's own 180 pulse.  None keeps
    the plain path exact.
    """
    gamma_dt = jnp.float32(GAMMA * dt)
    E2 = jnp.float32(np.exp(-dt / T2)) if T2 is not None else jnp.float32(1.0)
    E1 = jnp.float32(np.exp(-dt / T1)) if T1 is not None else jnp.float32(1.0)
    M0f = jnp.float32(M0)
    global_carrier = jnp.float32(2.0 * np.pi * float(off_resonance_hz) * dt)
    rho_over_D = jnp.float32(rho / D)
    has_surf = rho > 0.0 and hasattr(geometry, 'reflect_with_log_weight')
    has_perm = (float(geometry.permeability or 0.0) > 0.0
                and hasattr(geometry, 'permeate'))
    reflect = geometry.reflect
    reflect_lw = getattr(geometry, 'reflect_with_log_weight', None)

    def _apply(r_new, phi_grad, surf, M, rf_dflip, rf_axis, rf_carrier, crush_rate, uc,
               phi_field=None):
        """RF rotation, then free-precession + relaxation (shared across walk variants).

        ``phi_grad`` (n_meas,) is the gradient phase accumulated over the step; the other
        carriers — global off-resonance, pulse carrier, the emergent voxel-scale crusher
        (a per-walker macroscopic phase that dephases the transverse residual while leaving
        the longitudinally-stored magnetisation untouched), and an optional susceptibility
        field — are added here at the dt grid."""
        M = _rf_increment_jax(M, rf_dflip, rf_axis)         # RF rotation (0 -> identity)
        dphi = phi_grad + global_carrier + rf_carrier + crush_rate * uc     # (n_meas,)
        if phi_field is not None:
            dphi = dphi + phi_field                         # accumulated per sub-step by the caller
        elif field_fn is not None:
            dphi = dphi + gamma_dt * field_fn(r_new)        # susceptibility off-resonance
        c, s = jnp.cos(dphi), jnp.sin(dphi)
        Mx = (c * M[:, 0] - s * M[:, 1]) * E2 * surf
        My = (s * M[:, 0] + c * M[:, 1]) * E2 * surf
        Mz = M0f + (M[:, 2] - M0f) * E1                     # recover toward equilibrium
        return jnp.stack([Mx, My, Mz], axis=1), (Mx + 1j * My)

    if has_perm:
        # Membrane crossing (Powles) is step-size sensitive, so sub-step the permeable
        # walk to step_l ~ R/25 within each waveform dt; the gradient phase accumulates
        # per fine sub-step. Same crossing physics as the scalar core.simulate walk, now
        # carried on the vector-Bloch M -- so a longitudinally-stored pool that EXCHANGES
        # across membranes during a mixing time is modelled correctly (e.g. FEXI).
        kappa_over_D = jnp.float32(geometry.permeability / D)
        permeate = geometry.permeate
        n_sub = permeable_sub_steps(geometry, float(D), dt)
        step_len_sub = jnp.float32(np.sqrt(6.0 * D * dt / n_sub))
        gamma_dt_sub = jnp.float32(GAMMA * dt / n_sub)

        def step_fn(carry, inputs):
            r, M, key, uc = carry                           # r:(3,)  M:(n_meas,3)  uc:()
            g_t, rf_dflip, rf_axis, rf_carrier, crush_rate = inputs

            def _sub(c, _):
                r, phi, logw, key = c
                key, sk_step, sk_perm = jax.random.split(key, 3)
                noise = jax.random.normal(sk_step, (3,), dtype=jnp.float32)
                unit = noise / jnp.linalg.norm(noise)
                r_new, dlog_w = permeate(r, unit * step_len_sub, kappa_over_D,
                                         rho_over_D, sk_perm)   # dlog_w already scaled by rho/D
                phi_new = phi + gamma_dt_sub * (g_t @ r_new)    # (n_meas,)
                return (r_new, phi_new, logw + dlog_w, key), None

            init = (r, jnp.zeros(M.shape[0], jnp.float32), jnp.float32(0.0), key)
            (r_new, phi_grad, logw, key), _ = jax.lax.scan(_sub, init, None, length=n_sub)
            surf = jnp.exp(logw)                            # transverse wall attenuation (1 if rho=0)
            M_new, xy = _apply(r_new, phi_grad, surf, M, rf_dflip, rf_axis, rf_carrier,
                               crush_rate, uc)
            return (r_new, M_new, key, uc), xy

        return step_fn

    # Sub-step the walk, by the SAME rule the scalar engine uses (`physics.make_step_fn`'s no-weight
    # branch), so `simulate_bloch` and `core.simulate` resolve the same collisions by construction
    # rather than by coincidence.
    #
    # This path took one displacement per waveform step no matter what `sub_steps` said -- the argument
    # was accepted, documented, and dropped. Analytic geometries did not care, because their reflect is
    # exact at any step length. A mesh cannot be: a step longer than the collision-lookup cell crosses
    # triangles that were never candidates and the walker simply leaves. Measured on a 2 um icosphere at
    # b=2e9 (narrow-pulse PGSE, square limit): this path returned 0.05052 where `core.simulate` on the
    # identical geometry and waveform gave 0.96305 and an analytic `Sphere` 0.96442 -- the
    # free-diffusion answer (0.01867), silently.
    #
    # `n_sub == 1` reproduces the old single-displacement path BIT-IDENTICALLY: the key is split once
    # per waveform step either way, and `gamma_dt_sub == gamma_dt`. So nothing that was already resolved
    # moves, and the susceptibility phase (now accumulated per sub-step, where it used to be evaluated
    # once at the step's end position) is likewise unchanged at n_sub == 1 and strictly better above it.
    n_sub = sub_steps if sub_steps else walk_sub_steps(geometry, float(D), dt)
    _warn_if_step_outruns_the_lookup(geometry, float(D), dt, n_sub, 'Bloch walk')
    step_len_sub = jnp.float32(np.sqrt(6.0 * D * dt / n_sub))
    gamma_dt_sub = jnp.float32(GAMMA * dt / n_sub)

    def step_fn(carry, inputs):
        r, M, key, uc = carry                               # r:(3,)  M:(n_meas,3)  uc:()
        g_t, rf_dflip, rf_axis, rf_carrier, crush_rate = inputs   # g_t:(n_meas,3)

        def _sub(c, _):
            r, phi, phi_f, logw, key = c
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, (3,), dtype=jnp.float32)
            unit = noise / jnp.linalg.norm(noise)
            if has_surf:
                r_new, dlog = reflect_lw(r, unit * step_len_sub, jnp.float32(1.0))
            else:
                r_new, dlog = reflect(r, unit * step_len_sub), jnp.float32(0.0)
            phi_f_new = (phi_f + gamma_dt_sub * field_fn(r_new)
                         if field_fn is not None else phi_f)
            return (r_new, phi + gamma_dt_sub * (g_t @ r_new), phi_f_new,
                    logw + dlog, key), None

        init = (r, jnp.zeros(M.shape[0], jnp.float32), jnp.float32(0.0), jnp.float32(0.0), key)
        (r_new, phi_grad, phi_field, logw, key), _ = jax.lax.scan(_sub, init, None, length=n_sub)
        # rho/D applied to the SUMMED local time, matching the single-step form exactly
        surf = jnp.exp(rho_over_D * logw) if has_surf else jnp.float32(1.0)
        M_new, xy = _apply(r_new, phi_grad, surf, M, rf_dflip, rf_axis, rf_carrier, crush_rate, uc,
                           phi_field=(phi_field if field_fn is not None else None))
        return (r_new, M_new, key, uc), xy

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
                   T2=None, T1=None, M0=1.0, off_resonance_hz=0.0, seed=0, r0=None,
                   echo_steps=None, return_mz=False, crusher=None,
                   surface_relaxivity=0.0,
                   kappa_MT=0.0, dwell_time=0.0, T2_bound=1e-5, T1_bound=1.0,
                   off_resonance_bound=0.0, sub_steps=None, return_bound_frac=False,
                   equilibrate_binding="auto", susceptibility=None, require_gpu=None):
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
    equilibrate_binding : {'auto', 'burnin', 'fast', 'off'}
        (MT only, ``kappa_MT > 0``.)  How to reach the thermal-equilibrium bound-pool
        occupancy ``k_f/(k_f+k_r)`` BEFORE the sequence -- the macromolecular pool exists
        before the pulse, so an all-free start under-fills it and biases the transfer
        whenever ``1/k_f`` is not ``<<`` the sequence duration.  ``'burnin'`` (the ``'auto'``
        default when MT is on) runs an adaptive RF-off burn-in until the occupancy plateaus
        (geometry-agnostic).  ``'fast'`` seeds the equilibrium occupancy directly (mid-air,
        no walk) -- allowed only when it is provably position-invariant (no gradient, or an
        MR-dark bound pool) with a known S/V, else it warns and falls back to ``'burnin'``.
        ``'off'`` keeps the legacy all-free start (correct only if you equilibrate yourself,
        e.g. a burn-in block inside the waveform).
    sub_steps : int, optional
        Fine sub-steps per waveform step; overrides the per-geometry auto-tune (which follows
        the scalar engine's rule -- ``walk_sub_steps``, or ``permeable_sub_steps`` on a
        permeable geometry).  A mesh NEEDS this: a sub-step longer than the collision-lookup
        cell crosses triangles that were never gathered as candidates, so walls are missed
        and the walker leaves.  Setting it to 1 on a mesh reproduces the pre-#69 behaviour,
        which returned the free-diffusion signal for a restricted pore; the runtime guard
        warns when the resulting step outruns the lookup.
    r0 : array-like of shape (n_walkers, 3), optional
        Explicit start positions in metres.  Default: ``geometry.init_positions(n, key)``,
        which on a mesh means ``intra=True`` -- INSIDE the surface.  Pass this whenever the
        pool you want is not the geometry's inside; a fibre bundle's extra-axonal pool is
        the case that occurs, and getting it wrong silently walks the intra pool instead.
        See :func:`dmipy_sim.geometries.initial_positions`.

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
            T2=T2, T1=T1, M0=M0, off_resonance_hz=off_resonance_hz, seed=seed, r0=r0,
            echo_steps=echo_steps, return_mz=return_mz, crusher=crusher,
            surface_relaxivity=surface_relaxivity,
            kappa_MT=kappa_MT, dwell_time=dwell_time, T2_bound=T2_bound,
            T1_bound=T1_bound, off_resonance_bound=off_resonance_bound,
            sub_steps=sub_steps, return_bound_frac=return_bound_frac,
            equilibrate_binding=equilibrate_binding, susceptibility=susceptibility)

    if hasattr(waveform, 'waveform'):
        waveform = waveform.waveform
    G = np.asarray(waveform.G, dtype=np.float64)            # (n_meas, n_t, 3)
    dt = float(waveform.dt)
    _orient_R = geometry._orient_R
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
    r0 = initial_positions(geometry, n_walkers, pos_key, r0)   # (n_walkers, 3)
    if has_crush:
        uw = jax.random.uniform(jax.random.fold_in(master_key, 0xC0FFEE), (n_walkers,),
                                dtype=jnp.float32)
    else:
        uw = jnp.zeros((n_walkers,), dtype=jnp.float32)

    field_fn = None
    if susceptibility is not None:
        field_fn = (susceptibility.delta_bz_fn()
                    if hasattr(susceptibility, "delta_bz_fn") else susceptibility)
    step_fn = _make_bloch_step_fn(geometry, float(diffusivity), dt,
                                  T2, T1, float(M0), float(off_resonance_hz),
                                  rho=float(surface_relaxivity), field_fn=field_fn,
                                  sub_steps=sub_steps)
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
    from .physics import _geometry_radius
    return _geometry_radius(geometry)


def _surface_to_volume(geometry):
    """Surface-to-volume ratio (1/m) used only for the fast (mid-air) equilibrium init's
    occupancy ``P_eq = k_f/(k_f+k_r)``, ``k_f = kappa_MT*(S/V)``.  Only closed analytic
    shapes are handled exactly; None otherwise, so the fast path is refused and the caller
    falls back to the geometry-agnostic burn-in (which needs no S/V)."""
    name = type(geometry).__name__
    R = _geom_radius(geometry)
    if R is not None and R > 0.0:
        if name == 'Sphere':
            return 3.0 / R
        if name == 'Cylinder':
            return 2.0 / R                      # infinite cylinder, lateral wall
    return None


def _resolve_equilibrate_mode(equilibrate_binding, G, T2, T2_bound, geometry):
    """Map the ``equilibrate_binding`` request to a concrete mode ('off'|'burnin'|'fast'),
    demoting 'fast' to 'burnin' (with a warning) whenever the mid-air bound positions would
    not be position-invariant.  Mid-air init is safe only when the bound spins never carry
    position-dependent signal: either no gradient at all (G==0), or an MR-dark bound pool
    (T2_bound << T2) so bound transverse dephases before any echo -- and S/V must be known
    for the occupancy."""
    eb = equilibrate_binding
    if eb in (None, False, 'off'):
        return 'off'
    if eb in ('auto', True, 'burnin'):
        return 'burnin'                         # safe, geometry-agnostic default when MT on
    if eb == 'fast':
        if _surface_to_volume(geometry) is None:
            warnings.warn("equilibrate_binding='fast' needs a known surface-to-volume "
                          "(analytic Sphere/Cylinder); falling back to 'burnin'.", stacklevel=3)
            return 'burnin'
        no_gradient = not bool(np.any(G))
        dark = (T2 is None) or (float(T2_bound) <= 0.02 * float(T2))
        if not (no_gradient or dark):
            warnings.warn("equilibrate_binding='fast' is unsafe: a gradient is present and "
                          "the bound pool is not MR-dark (T2_bound not << T2), so the mid-air "
                          "bound positions would bias the signal; falling back to 'burnin'.",
                          stacklevel=3)
            return 'burnin'
        return 'fast'
    raise ValueError(f"equilibrate_binding must be 'auto'|'burnin'|'fast'|'off', got {eb!r}")


def _equilibrate_burnin(step_fn, r0, walker_keys, uw, M_init, n_meas, dt, dwell_time,
                        tol=0.01, max_chunks=40):
    """Adaptive RF-off / gradient-off burn-in: evolve the walk in chunks until the bound-pool
    occupancy plateaus (relative change between chunks < ``tol``), so the binding reaches its
    thermal equilibrium before the sequence.  Geometry-agnostic (needs no S/V) and self-adapts
    to the slow long-dwell regime.  Returns equilibrated positions, per-walker residual bound
    counter (sub-steps), advanced keys, the plateau occupancy, and a converged flag."""
    n_walkers = r0.shape[0]
    n_chunk = int(max(4, round(float(dwell_time) / dt)))       # ~one dwell (turnover unit)
    z1 = jnp.zeros((n_chunk,), jnp.float32)
    burn_inputs = (jnp.zeros((n_chunk, n_meas, 3), jnp.float32), z1, z1, z1, z1)  # G,dflip,axis,carrier,crush

    def chunk_walker(r_w, key_w, uw_w, brem_w):
        (r_f, _, key_f, _, brem_f), (_, bf_seq) = jax.lax.scan(
            step_fn, (r_w, M_init, key_w, uw_w, brem_w), burn_inputs)
        return r_f, key_f, brem_f, jnp.mean(bf_seq)

    chunk = jax.jit(jax.vmap(chunk_walker, in_axes=(0, 0, 0, 0)))
    r, keys, brem = r0, walker_keys, jnp.zeros((n_walkers,), dtype=jnp.float32)
    occ_prev, converged = -1.0, False
    for _ in range(max_chunks):
        r, keys, brem, bf = chunk(r, keys, uw, brem)
        occ = float(jnp.mean(bf))
        if occ_prev >= 0.0 and abs(occ - occ_prev) <= tol * max(occ, 1e-6):
            occ_prev, converged = occ, True
            break
        occ_prev = occ
    return r, brem, keys, occ_prev, converged


def _make_bloch_mt_step_fn(geometry, D, dt, n_sub, T2, T1, M0, off_res_global,
                           kappa_MT, dwell_time, T2_bound, T1_bound, off_res_bound,
                           rho=0.0, field_fn=None):
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
    # Everything here is float32.  This used to be float64, which required flipping JAX's
    # `jax_enable_x64` PROCESS-WIDE (there is no other way to get float64 in JAX -- an explicitly
    # requested float64 array silently truncates to float32 without the flag) and never restoring
    # it, so every computation traced afterwards changed precision.  That broke `Mesh`, whose
    # bounce-loop carries were int32 while `argmin` returns int64 under x64, and made test outcomes
    # depend on run order.
    #
    # The stated reason was that the off-resonance phase reaches 10^2-10^3 rad over ~10^5-10^6
    # sub-steps and float32 loses it.  It does not: the phase is never accumulated into a growing
    # scalar, it is applied as a per-sub-step Rodrigues increment, so there is no large-plus-small
    # addition to lose.  Measured on the Z-spectrum saturation this rationale points at (CW off
    # resonance, t_sat 25 ms, offsets to 8 kHz = 1257 rad, 3 seeds, walk bit-identical between the
    # two because every RNG draw is explicitly float32):
    #
    #   offset      0 Hz    500    1000    3000    8000
    #   |f32-f64|  0.0020  0.0013  0.0014  0.0020  0.0014
    #   seed spread 0.0079  0.0048  0.0022  0.0020  0.0016
    #
    # The precision difference is under the Monte-Carlo noise at every offset and 4x under it at the
    # worst.  The Rodrigues coefficients below are already in cancellation-free half-angle form,
    # which is what actually keeps the small per-sub-step angles accurate.  See #70.
    _F = jnp.float32
    inv_nsub = _F(1.0 / n_sub)
    step_len = jnp.float32(np.sqrt(6.0 * D * dt_sub))
    kappa_over_D = jnp.float32(kappa_MT / D)
    dwell_steps_mean = jnp.float32(dwell_time / dt_sub)
    invT2_free = _F(1.0 / T2) if T2 is not None else _F(0.0)
    invT1_free = _F(1.0 / T1) if T1 is not None else _F(0.0)
    invT2_bound = _F(1.0 / T2_bound)
    invT1_bound = _F(1.0 / T1_bound)
    M0f = _F(M0)
    dt_sub_f = _F(dt_sub)
    gamma_dt = _F(GAMMA * dt)
    gamma_dt_sub = _F(GAMMA * dt_sub)                          # susceptibility precession / sub-step
    global_carrier = _F(2.0 * np.pi * float(off_res_global) * dt)
    bound_carrier = _F(2.0 * np.pi * float(off_res_bound) * dt)
    rho_over_D = _F(rho / D)
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
            p_stick = bind_probability(kappa_over_D, local_time)
            newly = (~is_bound) & (jax.random.uniform(stick_key, dtype=jnp.float32) < p_stick)
            u_dwell = jax.random.uniform(dwell_key, dtype=jnp.float32)
            dwell_draw = -jnp.log(jnp.maximum(u_dwell, jnp.float32(1e-20))) * dwell_steps_mean
            r_next = jnp.where(is_bound, r, r_free)             # frozen while bound
            bound_rem_next = jnp.where(is_bound, bound_rem - jnp.float32(1.0),
                                       jnp.where(newly, dwell_draw, jnp.float32(0.0)))
            # --- spin evolution for THIS sub-step, resolved by state ---
            # Strang split (symmetric, O(dt_sub^2)): half relaxation, one EXACT rotation
            # about the effective field, half relaxation.  The rotation solves
            # dM/dt = omega_eff x M exactly with omega_eff = (w1 cos ax, w1 sin ax, dw)
            # (rotation vector k = omega_eff*dt_sub), fusing RF nutation and off-resonance
            # precession; symmetrising the relaxation removes the rotation/relaxation split
            # error that matters for the fast bound pool (dt_sub*R2_bound ~ O(0.05)).
            invT2 = jnp.where(is_bound, invT2_bound, invT2_free)
            invT1 = jnp.where(is_bound, invT1_bound, invT1_free)
            E2h = jnp.exp(-0.5 * dt_sub_f * invT2)
            E1h = jnp.exp(-0.5 * dt_sub_f * invT1)
            mx = M[:, 0] * E2h                                   # first half-relaxation
            my = M[:, 1] * E2h
            mz = M0f + (M[:, 2] - M0f) * E1h
            dphi = carr_free_sub + jnp.where(is_bound, carr_bound_sub, _F(0.0))
            if field_fn is not None:                            # susceptibility off-resonance
                dphi = dphi + gamma_dt_sub * field_fn(r_next).astype(_F)  # at the spin's position
            kx = flip_sub * jnp.cos(rf_axis).astype(_F)
            ky = flip_sub * jnp.sin(rf_axis).astype(_F)
            kz = dphi
            th2 = kx * kx + ky * ky + kz * kz
            th = jnp.sqrt(th2)
            ca = jnp.cos(th)
            # Rodrigues coefficients in numerically STABLE form: the naive (1 - cos th)/th^2
            # suffers catastrophic cancellation for the small per-sub-step angles here, so use
            # the exact half-angle identity (1 - cos th)/th^2 = 1/2 (sin(th/2)/(th/2))^2 and
            # sinc for sin(th)/th -- both cancellation-free (evaluated in float64, _F).
            sb = jnp.where(th > _F(1e-12), jnp.sin(th) / th, _F(1.0))
            half = _F(0.5) * th
            sh = jnp.where(half > _F(1e-12), jnp.sin(half) / half, _F(1.0))
            cb = _F(0.5) * sh * sh
            kdotM = kx * mx + ky * my + kz * mz                  # (n_meas,)
            crx = ky * mz - kz * my                              # (k x M)
            cry = kz * mx - kx * mz
            crz = kx * my - ky * mx
            Rx = ca * mx + sb * crx + cb * kx * kdotM            # Rodrigues rotation
            Ry = ca * my + sb * cry + cb * ky * kdotM
            Rz = ca * mz + sb * crz + cb * kz * kdotM
            # second half-relaxation + surface relaxivity (discrete free-contact weight)
            surf = jnp.where(is_bound | newly, _F(1.0),
                             jnp.exp(rho_over_D * dlog))
            Mx = Rx * E2h * surf
            My = Ry * E2h * surf
            Mz = M0f + (Rz - M0f) * E1h
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
                       T2, T1, M0, off_resonance_hz, seed, r0, echo_steps, return_mz,
                       crusher, surface_relaxivity, kappa_MT, dwell_time, T2_bound,
                       T1_bound, off_resonance_bound, sub_steps, return_bound_frac,
                       equilibrate_binding="auto", susceptibility=None):
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
    _orient_R = geometry._orient_R
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
    r0 = initial_positions(geometry, n_walkers, pos_key, r0)
    uw = (jax.random.uniform(jax.random.fold_in(master_key, 0xC0FFEE), (n_walkers,),
                             dtype=jnp.float32) if has_crush
          else jnp.zeros((n_walkers,), dtype=jnp.float32))

    field_fn = None
    if susceptibility is not None:
        field_fn = (susceptibility.delta_bz_fn()
                    if hasattr(susceptibility, "delta_bz_fn") else susceptibility)
    step_fn = _make_bloch_mt_step_fn(geometry, D, dt, int(sub_steps), T2, T1,
                                     float(M0), float(off_resonance_hz), float(kappa_MT),
                                     float(dwell_time), float(T2_bound), float(T1_bound),
                                     float(off_resonance_bound), rho=float(surface_relaxivity),
                                     field_fn=field_fn)
    M_init = jnp.zeros((n_meas, 3), dtype=jnp.float32).at[:, 2].set(jnp.float32(M0))
    dwell_steps_mean = float(dwell_time) / (dt / int(sub_steps))   # residual dwell in sub-steps

    # ── bound-pool initial condition (the thermal-equilibrium occupancy k_f/(k_f+k_r)) ──
    mode = _resolve_equilibrate_mode(equilibrate_binding, G, T2, T2_bound, geometry)
    S_V = _surface_to_volume(geometry)
    P_eq = (kappa_MT * S_V) / (kappa_MT * S_V + 1.0 / dwell_time) if S_V is not None else None
    converged = True
    if mode == 'fast':
        ek = jax.random.fold_in(master_key, 0xB0117E)
        u_state, u_dwell = jax.random.uniform(ek, (2, n_walkers), dtype=jnp.float32)
        bound_rem0 = jnp.where(u_state < jnp.float32(P_eq),
                               -jnp.log(jnp.maximum(u_dwell, jnp.float32(1e-20)))
                               * jnp.float32(dwell_steps_mean), jnp.float32(0.0))
        run_keys = walker_keys
    elif mode == 'burnin':
        r0, bound_rem0, run_keys, occ_burn, converged = _equilibrate_burnin(
            step_fn, r0, walker_keys, uw, M_init, n_meas, dt, dwell_time)
    else:  # 'off' -- legacy all-free start
        bound_rem0 = jnp.zeros((n_walkers,), dtype=jnp.float32)
        run_keys = walker_keys

    def simulate_walker(r0_w, key_w, uw_w, brem0_w):
        (_, M_f, _, _, bound_rem_f), (xy_seq, bf_seq) = jax.lax.scan(
            step_fn, (r0_w, M_init, key_w, uw_w, brem0_w), scan_inputs)
        return M_f, xy_seq, bf_seq, bound_rem_f

    M_final, xy_seq, bf_seq, bound_rem_final = jax.vmap(
        simulate_walker, in_axes=(0, 0, 0, 0))(r0, run_keys, uw, bound_rem0)
    # (n_w,n_meas,3) (n_w,n_t,n_meas) (n_w,n_t) (n_w,)

    # ── equilibration self-check: warn rather than silently return a biased signal ──
    if mode != 'off':
        occ_run = float(jnp.mean(bf_seq))
        if mode == 'burnin' and not converged:
            warnings.warn("equilibrate_binding: bound-pool occupancy did not plateau within "
                          "the burn-in cap (long-dwell / heterogeneous substrate?); the MT "
                          "signal may be under-equilibrated.", stacklevel=2)
        if P_eq is not None and abs(occ_run - P_eq) > max(0.05 * P_eq, 0.01):
            warnings.warn(f"equilibrate_binding='{mode}': bound occupancy during the sequence "
                          f"({occ_run:.3f}) differs from equilibrium k_f/(k_f+k_r)={P_eq:.3f}; "
                          f"the MT bound pool may be off-equilibrium.", stacklevel=2)

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
