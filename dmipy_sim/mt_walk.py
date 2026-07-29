"""Magnetization-transfer random walk: the binding (sticking) physics.

This is the one genuinely new piece of *walker* physics MT needs.  It reuses the
engine's existing per-hit boundary-local-time channel -- ``reflect_with_log_weight
(r, step, rho_over_D=1)`` returns ``dlog = -2 * Sum d_perp`` -- so the stick
decision at each step is

    p_stick = min(1, (kappa_MT / D) * (-dlog)) = min(1, 2 (kappa_MT/D) Sum d_perp)

i.e. the impact-angle rule of :func:`dmipy_sim.mt.stick_probability`, consuming the
SAME local-time quantity the surface relaxivity uses (so it is timestep-independent
by construction, no Cottaar ``sqrt(dt)`` factor).  Because relaxivity gives
``-d ln S/dt = rho * (S/V)`` from that channel, the emergent free->bound rate is
``k_f = kappa_MT * (S/V)`` -- validated in ``tests/test_mt_binding.py``.

A stuck walker FREEZES (stops diffusing) for an exponentially-distributed dwell of
mean ``dwell_time`` (drawn as ``-log(u) * dwell_time / dt`` sub-steps, matching the
continuum residence time), then is released.  The bound state is carried in the
walker like a compartment id.  The walk records, per saved step, the fractional
bound occupancy ``bound_frac`` (time fraction spent bound) alongside the positions
and the free-pool boundary local time ``dlog_boundary_unit`` (for surface
relaxivity of the free pool at replay).

The bound spins' relaxation / RF (the actual MT saturation transfer) is applied at
REPLAY by :func:`dmipy_sim.trajectories.apply_waveform_bloch`, consuming
``bound_frac`` -- positions while bound are irrelevant to the MT observables (a
bound spin has ~zero transverse via its huge R2b).  This is the replay counterpart
of the fused forward path ``bloch.simulate_bloch(kappa_MT=...)``.

Targets closed analytic geometries with ``reflect_with_log_weight`` (Sphere,
Cylinder, Box1D, Ellipsoid) and Mesh (via ``reflect_with_binding`` when present).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .physics import _geometry_radius
from . import mt as _mt


def simulate_mt_trajectories(
    n_walkers: int,
    diffusivity: float,
    geometry,
    T_max: float,
    dt_save: float,
    kappa_MT: float,
    dwell_time: float,
    seed: int = 42,
    walker_batch_size: int = 50_000,
    sub_steps: int = None,
    equilibrate_binding="auto",
    require_gpu=None,
) -> tuple:
    """Random walk with MT surface binding.

    Parameters
    ----------
    n_walkers, diffusivity, geometry, T_max, dt_save, seed :
        As in :func:`dmipy_sim.core.simulate_trajectories`.  ``geometry`` must
        provide ``reflect_with_log_weight(r, step, rho_over_D)`` and
        ``init_positions(n, key)``.
    kappa_MT : float
        MT surface reactivity (m/s).  0 disables binding (pure diffusion walk).
    dwell_time : float
        Mean bound-pool residence time (s).  Must be > 0 when kappa_MT > 0.
    sub_steps : int, optional
        Fine sub-steps per saved step.  Default: auto to step_l ~ R/25 (binding
        is trajectory-altering, like permeability, so it needs the finer step,
        not reflection's R/6).
    equilibrate_binding : {'auto', 'burnin', 'fast', 'off'}
        How the bound pool reaches its thermal-equilibrium occupancy BEFORE t=0;
        see :func:`dmipy_sim.mt.resolve_equilibrate_mode`.

    Returns
    -------
    trajectories : (n_walkers, n_t, 3) float16   positions at each saved step.
    dt_actual : float                            saved time step.
    sub_steps : int                              fine sub-steps per saved step.
    dt_sim : float                               fine sub-step size.
    bound_frac : (n_walkers, n_t) float16        fractional bound occupancy per save.
    dlog_boundary_unit : (n_walkers, n_t) float16  free-pool boundary local time
        (rho/D=1), i.e. -2*Sum d_perp over the FREE sub-steps of each save.
    """
    from .gpu import check_gpu
    check_gpu(n_walkers, require_gpu, what="simulate_mt_trajectories")

    if kappa_MT > 0.0 and dwell_time <= 0.0:
        raise ValueError("dwell_time must be > 0 when kappa_MT > 0.")
    if not hasattr(geometry, "reflect_with_log_weight"):
        raise TypeError(
            f"{type(geometry).__name__} has no reflect_with_log_weight; MT binding "
            "needs the boundary-local-time channel (Sphere/Cylinder/Box1D/Ellipsoid/Mesh).")

    n_t = int(round(T_max / dt_save)) + 1
    dt_actual = T_max / (n_t - 1)

    # sub-step auto-tune: binding freezes walkers (trajectory-altering, like
    # permeability), so use the finer R/25 rule (divisor 6*25^2=3750), not R/6.
    R_geom = _geometry_radius(geometry)
    if sub_steps is None:
        if R_geom is None:
            sub_steps = 1
        else:
            dt_phys_max = float(R_geom) ** 2 / (3750.0 * diffusivity)
            sub_steps = max(1, int(np.ceil(dt_actual / dt_phys_max)))
    dt_sim = dt_actual / sub_steps
    step_l = jnp.float32(jnp.sqrt(6.0 * diffusivity * dt_sim))

    print(f"  [mt] sub_steps={sub_steps}, dt_sim={dt_sim*1e6:.3f} us, "
          f"step_l={float(step_l)*1e6:.4f} um"
          + (f", step_l/R={float(step_l)/float(R_geom):.4f}" if R_geom else ""),
          flush=True)

    kappa_over_D = jnp.float32(float(kappa_MT) / float(diffusivity))
    dwell_steps_mean = (jnp.float32(float(dwell_time) / dt_sim) if dwell_time > 0
                        else jnp.float32(0.0))
    # Side-dependent binding: a Mesh with mt_side exposes reflect_with_binding, which
    # returns the free-pool surface local time AND the side-weighted binding local
    # time in one pass.  Analytic geometries use reflect_with_log_weight (symmetric),
    # for which the binding local time is just -dlog (rho/D=1).
    _has_binding = hasattr(geometry, "reflect_with_binding")
    if _has_binding:
        _reflect_bind = geometry.reflect_with_binding

        def _move(r, step):
            r_free, dlog_free, blt = _reflect_bind(r, step, jnp.float32(1.0))
            return r_free, dlog_free, blt
    else:
        _reflect_lw = geometry.reflect_with_log_weight

        def _move(r, step):
            r_free, dlog_free = _reflect_lw(r, step, jnp.float32(1.0))
            return r_free, dlog_free, -dlog_free      # symmetric: binding LT = -dlog

    def inner_step(carry, _):
        r, key, bound_rem, dlog_acc, bound_acc = carry
        key, step_key, stick_key, dwell_key = jax.random.split(key, 4)
        is_bound = bound_rem > jnp.float32(0.0)

        # free-move proposal + per-step boundary local time (surface + binding)
        noise = jax.random.normal(step_key, (3,), dtype=jnp.float32)
        unit = noise / jnp.linalg.norm(noise)
        step = unit * step_l
        r_free, dlog_free, local_time = _move(r, step)   # dlog_free <= 0, local_time >= 0

        # stick decision (only if currently free); dwell drawn as exp residence
        p_stick = jnp.minimum(jnp.float32(1.0), kappa_over_D * local_time)
        u_stick = jax.random.uniform(stick_key, dtype=jnp.float32)
        newly = (~is_bound) & (u_stick < p_stick)
        u_dwell = jax.random.uniform(dwell_key, dtype=jnp.float32)
        dwell_draw = -jnp.log(jnp.maximum(u_dwell, jnp.float32(1e-20))) * dwell_steps_mean

        # advance: frozen if already bound, else the free move
        r_next = jnp.where(is_bound, r, r_free)
        bound_rem_next = jnp.where(
            is_bound, bound_rem - jnp.float32(1.0),
            jnp.where(newly, dwell_draw, jnp.float32(0.0)))
        # MUTUAL EXCLUSIVITY of binding and surface relaxivity at an encounter: a spin
        # that STICKS does not also relax at that wall contact (it left the free pool),
        # so its local time is removed from the rho-channel.  This keeps rho a pure
        # replay knob (the stick partition uses only kappa_MT).
        dlog_acc = dlog_acc + jnp.where(is_bound | newly, jnp.float32(0.0), dlog_free)
        bound_acc = bound_acc + jnp.where(is_bound, jnp.float32(1.0), jnp.float32(0.0))
        return (r_next, key, bound_rem_next, dlog_acc, bound_acc), None

    def outer_step(carry, _):
        r, key, bound_rem = carry
        init = (r, key, bound_rem, jnp.float32(0.0), jnp.float32(0.0))
        (r_f, key_f, bound_rem_f, dlog_acc, bound_acc), _ = jax.lax.scan(
            inner_step, init, None, length=sub_steps)
        bound_frac = bound_acc / jnp.float32(sub_steps)
        return (r_f, key_f, bound_rem_f), (r_f, dlog_acc, bound_frac)

    def one_walker(r0_w, key_w, brem0_w):
        (_, _, _), (pos, dlog, bfrac) = jax.lax.scan(
            outer_step, (r0_w, key_w, brem0_w), None, length=n_t)
        return pos, dlog, bfrac

    batch = jax.jit(jax.vmap(one_walker, in_axes=(0, 0, 0)))

    master_key = jax.random.PRNGKey(seed)
    pos_key, walker_key = jax.random.split(master_key)
    r0_all = geometry.init_positions(n_walkers, pos_key)
    walker_keys = jax.random.split(walker_key, n_walkers)

    # ── bound-pool equilibration ──
    # An all-free start under-fills the macromolecular pool; equilibrate to f_b BEFORE
    # t=0 and discard the preamble, so the saved bound_frac starts at equilibrium.
    brem0 = jnp.zeros((n_walkers,), jnp.float32)
    if kappa_MT > 0.0:
        mode = _mt.resolve_equilibrate_mode(equilibrate_binding, geometry)
        S_V = _mt.surface_to_volume(geometry)
        P_eq = _mt.bound_fraction(kappa_MT, dwell_time, S_V) if S_V is not None else None
        converged = True
        if mode == "fast":               # seed f_b walkers bound with a residual dwell
            ek = jax.random.fold_in(master_key, 0xB0117E)
            u_state, u_dwell = jax.random.uniform(ek, (2, n_walkers), dtype=jnp.float32)
            brem0 = jnp.where(u_state < jnp.float32(P_eq),
                              -jnp.log(jnp.maximum(u_dwell, jnp.float32(1e-20)))
                              * dwell_steps_mean, jnp.float32(0.0))
        elif mode == "burnin":           # adaptive occupancy-plateau burn-in
            n_chunk = max(4, int(round(float(dwell_time) / float(dt_sim))))

            def _burn_walker(r_w, key_w, brem_w):
                (r_f, key_f, brem_f, _, bacc), _ = jax.lax.scan(
                    inner_step, (r_w, key_w, brem_w, jnp.float32(0.0), jnp.float32(0.0)),
                    None, length=n_chunk)
                return r_f, key_f, brem_f, bacc / jnp.float32(n_chunk)
            _burn = jax.jit(jax.vmap(_burn_walker, in_axes=(0, 0, 0)))

            def _chunk_fn(r, keys, brem):
                r, keys, brem, bf = _burn(r, keys, brem)
                return r, keys, brem, jnp.mean(bf)
            r0_all, walker_keys, brem0, occ, converged = _mt.equilibrate_burnin_plateau(
                _chunk_fn, r0_all, walker_keys, brem0)
            print(f"  [mt] equilibrate '{mode}': <bound>={occ:.4f} "
                  + (f"(f_b={P_eq:.4f})" if P_eq is not None else ""), flush=True)
            if not converged:
                import warnings
                warnings.warn("equilibrate_binding: bound occupancy did not plateau within "
                              "the burn-in cap (long-dwell / heterogeneous substrate?); the "
                              "saved walk may be under-equilibrated.", stacklevel=2)

    pos_b, dlog_b, bfrac_b = [], [], []
    n_batches = (n_walkers + walker_batch_size - 1) // walker_batch_size
    for b in range(n_batches):
        s = b * walker_batch_size
        e = min(s + walker_batch_size, n_walkers)
        p, d, bf = batch(r0_all[s:e], walker_keys[s:e], brem0[s:e])
        pos_b.append(np.asarray(p).astype(np.float16))
        dlog_b.append(np.asarray(d).astype(np.float16))
        bfrac_b.append(np.asarray(bf).astype(np.float16))

    trajectories = np.concatenate(pos_b, axis=0)
    dlog_boundary_unit = np.concatenate(dlog_b, axis=0)
    bound_frac = np.concatenate(bfrac_b, axis=0)
    return trajectories, dt_actual, sub_steps, dt_sim, bound_frac, dlog_boundary_unit
