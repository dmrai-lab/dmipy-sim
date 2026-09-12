"""The trajectory producer with adaptive stepping: every walker steps at the rate its own situation needs.

The fused producer (:func:`dmipy_sim.engine.core.simulate_trajectories`) steps every walker at one ``dt_sim``,
resolved from the substrate's smallest feature (the R/6 rule of its smallest tube), so a walker in the extra-
axonal space of a strand substrate pays for the thinnest strand it will never touch. Here a save interval is
walked in ROUNDS, and at each round every walker is placed by the geometry's ``wall_scales``: its distance to
the nearest wall it can hit and the radius (curvature scale) of that wall.

- A walker farther from every wall than ``safety_sigma`` times the round's rms excursion takes ONE free
  Gaussian step for the round (truncated at its wall distance, so it cannot cross: the truncation removes a
  ``safety_sigma`` tail, 6e-8 of the mass at six sigma), accrues no contact and keeps its label.
- Every other walker runs the geometry's wall-aware kernel for the round at the step the R/6 rule gives for
  ITS nearest wall's radius, in radius classes doubling in step time (``dt_c = dt_min 2^c``, i.e. radius
  bounds ``R_min 2^(c/2)``): the finest class is the fused producer's own step and noise sequence.

Walkers are re-bucketed every round on the host (index arrays), the kernels are jitted per class and padded
walker count, and the positions stay on the device. The recorded channels are the fused producer's: positions
at every save, the boundary local time accumulated over each interval (the kernel's, at unit rho / D), the
compartment label (the geometries this serves are impermeable, so a label is constant). Illegal crossings are
counted the same way and the walk is refused if any occurred.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import jax
import jax.numpy as jnp

from ..persistent_walk import PersistentWalk

log = logging.getLogger("dmipy_sim")


def _pad_to(n, unit=4096):
    """The padded walker count a kernel is compiled for: powers of two from ``unit`` (a compile per size)."""
    return max(unit, 1 << int(math.ceil(math.log2(max(n, 1)))))


def simulate_trajectories_adaptive(n_walkers, diffusivity, geometry, T_max, dt_save, *, seed=0, r0=None,
                                   steps_per_round=16, safety_sigma=6.0, n_classes=4, sub_steps=None,
                                   walker_batch_size=100_000, require_gpu=None, storage_dtype=np.float32):
    """A :class:`~dmipy_sim.persistent_walk.PersistentWalk` of ``geometry`` with adaptive stepping (module
    docstring). ``geometry`` must offer ``wall_scales``, ``reflect_with_log_weight`` and ``classify_positions_exact``
    and be impermeable. ``steps_per_round`` is the finest class's steps per round (the round is
    ``steps_per_round`` finest steps long); ``n_classes`` the radius classes; ``sub_steps`` overrides the finest
    sub-step count per save (rounded up to a multiple of ``steps_per_round``)."""
    from .gpu import check_gpu
    from .physics import resolve_sub_steps, length_scales_of
    if not hasattr(geometry, "wall_scales"):
        raise TypeError(f"{type(geometry).__name__} offers no wall_scales; adaptive stepping needs the distance to the "
                        "nearest wall and its curvature scale")
    if geometry.permeability is not None:
        raise NotImplementedError("adaptive stepping is for impermeable substrates: a permeable wall needs the kernel at "
                                  "every step (the crossing decision is per contact)")
    check_gpu(n_walkers, require_gpu, what="simulate_trajectories_adaptive")
    D = float(diffusivity)
    n_t = int(round(T_max / dt_save)) + 1
    dt_actual = T_max / (n_t - 1)
    K = int(steps_per_round)
    n_min = resolve_sub_steps(geometry, D, dt_actual, surface=True, override=sub_steps)
    n_min = K * int(math.ceil(n_min / K))                          # the finest class divides the round
    n_rounds = n_min // K
    dt_min = dt_actual / n_min; dt_round = dt_actual / n_rounds
    R_min = float(length_scales_of(geometry).min_feature)
    sigma_round = math.sqrt(2.0 * D * dt_round)                    # per axis
    far_at = float(safety_sigma) * sigma_round                    # |step| > 6 sigma: 1.2e-7 of a 3-D Gaussian's mass
    n_classes = int(max(1, min(n_classes, int(math.log2(K)) + 1)))
    steps_c = [K >> c for c in range(n_classes)]                   # kernel steps per round, per class
    step_l_c = [jnp.float32(math.sqrt(6.0 * D * dt_round / m)) for m in steps_c]
    R_bounds = np.array([R_min * 2.0 ** (c / 2.0) for c in range(n_classes)])   # a class's lower radius bound
    log.info("adaptive walk: %d finest steps per save (dt_min %.3g us) in %d rounds of %d, %d radius classes from "
             "R_min %.2f um (steps per round %s), free step beyond %.2f um of a wall",
             n_min, dt_min * 1e6, n_rounds, K, n_classes, R_min * 1e6, steps_c, far_at * 1e6)

    reflect_lw = geometry.reflect_with_log_weight

    def _kernel(m, step_l):
        """One jitted dispatch per class and round: gather the class's walkers by index, step them ``m`` times,
        scatter them back; the padded tail of ``sel`` repeats a walker whose result is discarded."""
        def one(r, key):
            def body(carry, _):
                r, key, dlog = carry
                key, sub = jax.random.split(key)
                noise = jax.random.normal(sub, (3,), dtype=jnp.float32)
                step = noise / jnp.linalg.norm(noise) * step_l
                r_new, dlw = reflect_lw(r, step, jnp.float32(1.0))
                return (r_new, key, dlog + dlw), None
            (r_f, key_f, dlog_f), _ = jax.lax.scan(body, (r, key, jnp.float32(0.0)), None, length=m)
            return r_f, key_f, dlog_f
        stepped = jax.vmap(one)

        @jax.jit
        def run(r, keys, dlog, sel, n_real):
            r_c, key_c, dl_c = stepped(r[sel], keys[sel])
            real = jnp.arange(sel.shape[0]) < n_real
            sel_r = jnp.where(real, sel, r.shape[0])                      # out of bounds: dropped (-1 would wrap)
            r = r.at[sel_r].set(r_c, mode="drop"); keys = keys.at[sel_r].set(key_c, mode="drop")
            dlog = dlog.at[sel_r].add(dl_c, mode="drop")
            return r, keys, dlog
        return run

    kernels = [_kernel(m, l) for m, l in zip(steps_c, step_l_c)]

    scales_dev = geometry._wall_scales_device()

    @jax.jit
    def _bucket(d_wall, R_near):
        """Per walker: -1 for a free step, else its radius class."""
        far = d_wall > jnp.float32(far_at)
        cls = jnp.clip(jnp.floor(2.0 * jnp.log2(jnp.maximum(R_near, jnp.float32(R_min)) / jnp.float32(R_min))), 0, n_classes - 1)
        return jnp.where(far, -1, cls).astype(jnp.int8), far

    @jax.jit
    def _free_step(r, keys, d_wall, far):
        """One Gaussian step of the round for the far walkers, truncated at their wall distance; every walker's
        key advances (the far ones spend a draw), so the stream stays per walker."""
        def one(k):
            k, sub = jax.random.split(k)
            return k, jax.random.normal(sub, (3,), dtype=jnp.float32) * jnp.float32(sigma_round)
        keys_new, g = jax.vmap(one)(keys)
        norm = jnp.linalg.norm(g, axis=1, keepdims=True)
        cap = (jnp.minimum(d_wall, jnp.float32(far_at)) * jnp.float32(0.999))[:, None]
        g = jnp.where(norm > cap, g * cap / jnp.maximum(norm, 1e-30), g)
        return jnp.where(far[:, None], r + g, r), jnp.where(far[:, None], keys_new, keys)

    # seeds and keys, as the fused producer draws them
    from .core import initial_positions
    key = jax.random.PRNGKey(int(seed))
    pos_key, walker_key = jax.random.split(key)
    r0_all = np.asarray(initial_positions(geometry, n_walkers, pos_key, r0), np.float32)
    keys_all = jax.random.split(walker_key, n_walkers)
    comp_all = np.asarray(geometry.classify_positions_exact(jnp.asarray(r0_all)), np.int32)

    sdt = np.dtype(storage_dtype).type
    positions = np.empty((n_walkers, n_t, 3), sdt); dlog_all = np.zeros((n_walkers, n_t), sdt)
    positions[:, 0] = r0_all
    n_free = 0; n_kernel_steps = 0; n_illegal = 0
    n_batches = (n_walkers + walker_batch_size - 1) // walker_batch_size
    for b in range(n_batches):
        s, e = b * walker_batch_size, min((b + 1) * walker_batch_size, n_walkers)
        nb = e - s
        r = jnp.asarray(r0_all[s:e]); keys = keys_all[s:e]
        log.info("  adaptive: walkers %d-%d (%d%% done)...", s, e - 1, int(100 * e / n_walkers))
        for t in range(1, n_t):
            dlog_int = jnp.zeros(nb, jnp.float32)
            for _ in range(n_rounds):
                d_wall, R_near = scales_dev(r)
                bucket, far_j = _bucket(d_wall, R_near)
                r, keys = _free_step(r, keys, d_wall, far_j)
                bucket = np.asarray(bucket)                          # the one small host transfer per round
                n_free += int((bucket < 0).sum())
                for c in range(n_classes):
                    idx = np.flatnonzero(bucket == c)
                    if idx.size == 0:
                        continue
                    n_pad = _pad_to(idx.size)
                    sel = np.concatenate([idx, np.full(n_pad - idx.size, idx[0])]) if n_pad > idx.size else idx
                    r, keys, dlog_int = kernels[c](r, keys, dlog_int, jnp.asarray(sel), idx.size)
                    n_kernel_steps += idx.size * steps_c[c]
            positions[s:e, t] = np.asarray(r, sdt); dlog_all[s:e, t] = np.asarray(dlog_int, sdt)
        # the guarantee: nobody changed pool
        comp_end = np.asarray(geometry.classify_positions_exact(r), np.int32)
        n_illegal += int((comp_end != comp_all[s:e]).sum())
    if n_illegal:
        raise RuntimeError(f"adaptive walk: {n_illegal} walker(s) changed pool -- an illegal crossing; the walk is refused")
    total_steps = n_walkers * (n_t - 1) * n_min
    log.info("adaptive walk: %.1f%% of walker-rounds were free steps; kernel steps %.3g of the fused producer's %.3g "
             "(%.1fx fewer)", 100.0 * n_free / max(n_walkers * (n_t - 1) * n_rounds, 1), n_kernel_steps, total_steps,
             total_steps / max(n_kernel_steps, 1))
    comp = np.repeat(comp_all[:, None], n_t, axis=1).astype(np.int8)
    stepping = dict(rule="adaptive", free_fraction=n_free / max(n_walkers * (n_t - 1) * n_rounds, 1),
                    kernel_steps=int(n_kernel_steps), fused_steps=int(total_steps),
                    kernel_steps_ratio=total_steps / max(n_kernel_steps, 1), steps_per_round_by_class=steps_c,
                    radius_class_bounds_m=R_bounds.tolist(), far_at_m=far_at, safety_sigma=float(safety_sigma))
    return PersistentWalk(positions, float(dt_actual), int(n_min), float(dt_min), boundary_local_time=dlog_all,
                          compartment=comp, illegal_crossings=0, seed=int(seed), diffusivity=D, geometry=geometry,
                          stepping=stepping)
