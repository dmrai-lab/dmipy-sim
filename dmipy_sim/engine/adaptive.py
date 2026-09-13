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
                                   walker_batch_size=100_000, require_gpu=None, storage_dtype=np.float32,
                                   candidate_cache=True, candidate_k_start=64):
    """A :class:`~dmipy_sim.persistent_walk.PersistentWalk` of ``geometry`` with adaptive stepping (module
    docstring). ``geometry`` must offer ``wall_scales``, ``reflect_with_log_weight`` and ``classify_positions_exact``
    and be impermeable. ``steps_per_round`` is the finest class's steps per round (the round is
    ``steps_per_round`` finest steps long); ``n_classes`` the radius classes; ``sub_steps`` overrides the finest
    sub-step count per save (rounded up to a multiple of ``steps_per_round``). With ``candidate_cache`` (the
    default) each round gathers, per walker, the segments whose surface lies within the round's deterministic
    excursion once (``candidate_k_start`` entries, widened when a walker has more in reach) and steps against
    those instead of the 27-cell gather at every step."""
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

    cached = bool(candidate_cache) and hasattr(geometry, "reach_candidates") and hasattr(geometry, "_reflect_with")
    NUDGE = 1e-4 * R_min

    def _kernel(k_cand):
        """One jitted dispatch per round and class: gather the class's walkers by index, step them ``m`` times at
        ``step_l`` (both runtime scalars, so one program serves every class), scatter them back; the padded tail
        of ``sel`` repeats a walker whose result is discarded. With a candidate cache the segments a walker can
        meet in the round -- those whose surface lies within its deterministic excursion ``m step_l`` of its
        start -- are gathered once and the ``m`` steps test only them."""
        def one(r, key, m, step_l, reach):
            if cached:
                cand, valid, n_within = geometry.reach_candidates(r, reach, k_cand)

            def body(_, carry):
                r, key, dlog = carry
                key, sub = jax.random.split(key)
                noise = jax.random.normal(sub, (3,), dtype=jnp.float32)
                step = noise / jnp.linalg.norm(noise) * step_l
                if cached:
                    r_new, d_perp = geometry._step_with(r, step, cand, valid)
                    dlw = -2.0 * d_perp
                else:
                    r_new, dlw = reflect_lw(r, step, jnp.float32(1.0))
                return (r_new, key, dlog + dlw)
            r_f, key_f, dlog_f = jax.lax.fori_loop(0, m, body, (r, key, jnp.float32(0.0)))
            return r_f, key_f, dlog_f, (n_within if cached else jnp.int32(0))
        stepped = jax.vmap(one, in_axes=(0, 0, None, None, None))

        @jax.jit
        def run(r, keys, dlog, order, start, n_real, m, step_l, reach):
            """The class's walkers are ``order[start:start + n_real]`` (the walkers sorted by class, on the device);
            the window is ``n_pad`` wide, static, and its tail beyond ``n_real`` is stepped and dropped."""
            sel = jax.lax.dynamic_slice(order, (start,), (n_pad,))
            sel_g = jnp.minimum(sel, r.shape[0] - 1)
            r_c, key_c, dl_c, n_w = stepped(r[sel_g], keys[sel_g], m, step_l, reach)
            real = jnp.arange(n_pad) < n_real
            sel_r = jnp.where(real, sel, r.shape[0])                      # out of bounds: dropped (-1 would wrap)
            r = r.at[sel_r].set(r_c, mode="drop"); keys = keys.at[sel_r].set(key_c, mode="drop")
            dlog = dlog.at[sel_r].add(dl_c, mode="drop")
            return r, keys, dlog, jnp.where(real, n_w, 0).max()
        return run

    kernels = {}

    def kernel_for(k_cand, n_pad_):
        nonlocal n_pad
        n_pad = n_pad_
        if (k_cand, n_pad_) not in kernels:
            kernels[(k_cand, n_pad_)] = _kernel(k_cand)
        return kernels[(k_cand, n_pad_)]

    n_pad = 0

    @jax.jit
    def _order(bucket):
        """The walkers sorted by class (free first, as -1), padded by the widest window so any class window can be
        sliced from any start; the counts per class."""
        n = bucket.shape[0]
        order = jnp.argsort(bucket)
        pad = jnp.full(_pad_to(n), n, order.dtype)
        return jnp.concatenate([order, pad]), jnp.bincount(bucket.astype(jnp.int32) + 1, length=n_classes + 1)

    reach_c = [float(m) * float(l) + NUDGE for m, l in zip(steps_c, step_l_c)]
    k_cand = int(candidate_k_start)

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
    # the pool (0 free, 1 enclosed), not the object: a packed classify returns the tube's id, and a walker inside two
    # overlapping tubes is labelled by either
    comp_all = np.minimum(np.asarray(geometry.classify_positions_exact(jnp.asarray(r0_all)), np.int32), 1)

    if cached:                                                       # size the list from the start positions (the widest reach)
        sample = r0_all[np.random.default_rng(int(seed) + 3).choice(n_walkers, size=min(n_walkers, 20_000), replace=False)]
        counts = jax.jit(jax.vmap(lambda p: geometry.reach_candidates(p, jnp.float32(max(reach_c)), 1)[2]))(jnp.asarray(sample))
        k_cand = max(int(candidate_k_start), 1 << int(math.ceil(math.log2(1.5 * max(int(np.asarray(counts).max()), 1)))))
        log.info("adaptive: candidate list of %d (up to %d segments within %.2f um of a start position)", k_cand,
                 int(np.asarray(counts).max()), max(reach_c) * 1e6)
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
                order, counts = _order(bucket)
                counts = np.asarray(counts)                          # the one small host transfer per round
                n_free += int(counts[0]); start = int(counts[0])
                for c in range(n_classes):
                    n_c = int(counts[c + 1])
                    if n_c == 0:
                        continue
                    while True:                                          # the candidate list widens until it holds every segment in reach
                        r_try, keys_try, dlog_try, n_max = kernel_for(k_cand, _pad_to(n_c))(
                            r, keys, dlog_int, order, start, n_c, steps_c[c], step_l_c[c], jnp.float32(reach_c[c]))
                        if not cached or int(n_max) <= k_cand:
                            r, keys, dlog_int = r_try, keys_try, dlog_try
                            break
                        k_cand = 1 << int(math.ceil(math.log2(int(n_max))))
                        log.info("adaptive: candidate list widened to %d (a walker had %d segments in reach)", k_cand, int(n_max))
                    n_kernel_steps += n_c * steps_c[c]; start += n_c
            positions[s:e, t] = np.asarray(r, sdt); dlog_all[s:e, t] = np.asarray(dlog_int, sdt)
        # the guarantee: nobody changed pool
        comp_end = np.minimum(np.asarray(geometry.classify_positions_exact(r), np.int32), 1)
        n_illegal += int((comp_end != comp_all[s:e]).sum())
    if n_illegal:
        raise RuntimeError(f"adaptive walk: {n_illegal} walker(s) changed pool -- an illegal crossing; the walk is refused")
    total_steps = n_walkers * (n_t - 1) * n_min
    log.info("adaptive walk: %.1f%% of walker-rounds were free steps; kernel steps %.3g of the fused producer's %.3g "
             "(%.1fx fewer)", 100.0 * n_free / max(n_walkers * (n_t - 1) * n_rounds, 1), n_kernel_steps, total_steps,
             total_steps / max(n_kernel_steps, 1))
    comp = np.repeat(comp_all[:, None], n_t, axis=1).astype(np.int8)
    stepping = dict(rule="adaptive", candidate_cache=cached, candidate_k=int(k_cand),
                    free_fraction=n_free / max(n_walkers * (n_t - 1) * n_rounds, 1),
                    kernel_steps=int(n_kernel_steps), fused_steps=int(total_steps),
                    kernel_steps_ratio=total_steps / max(n_kernel_steps, 1), steps_per_round_by_class=steps_c,
                    radius_class_bounds_m=R_bounds.tolist(), far_at_m=far_at, safety_sigma=float(safety_sigma))
    return PersistentWalk(positions, float(dt_actual), int(n_min), float(dt_min), boundary_local_time=dlog_all,
                          compartment=comp, illegal_crossings=0, seed=int(seed), diffusivity=D, geometry=geometry,
                          stepping=stepping)
