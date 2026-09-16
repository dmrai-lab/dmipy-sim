"""The trajectory producer with adaptive stepping: every walker steps at the rate its own situation needs.

The fused producer (:func:`dmipy_sim.engine.core.simulate_trajectories`) steps every walker at one ``dt_sim``,
resolved from the substrate's smallest feature (the step rule of its smallest tube), so a walker in the extra-
axonal space of a strand substrate pays for the thinnest strand it will never touch. Here a save interval is
walked in ROUNDS, and at each round every walker is placed by the geometry's ``wall_scales``: its distance to
the nearest wall it can hit and the radius (curvature scale) of that wall.

- A walker farther from every wall than ``safety_sigma`` times the round's rms excursion takes ONE free
  Gaussian step for the round (truncated at its wall distance, so it cannot cross: the truncation removes a
  ``safety_sigma`` tail, 6e-8 of the mass at six sigma), accrues no contact and keeps its label.
- Every other walker runs the geometry's wall-aware kernel for the round at the step the family's rule gives for
  ITS nearest wall's radius, in radius classes doubling in step time (``dt_c = dt_min 2^c``, i.e. radius
  bounds ``R_min 2^(c/2)``): the finest class is the fused producer's own step and noise sequence.

Walkers are re-bucketed every round on the device (an argsort by class); the host reads the class counts once per round
(the padded class windows are static shapes) and nothing else until a chunk of CHUNK_SAVES saves is launched: the kernels'
overflow (a walker with more segments in reach than its candidate list) is a device-side maximum read once per chunk,
and the chunk is redone from its start with a wider list when it happened; the saves' positions and contact are stacked
on the device and copied to the host once per chunk. The kernels are jitted per class and padded
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
from .tables import jit_with_tables
from ..run import Run

log = logging.getLogger("dmipy_sim")


CHUNK_SAVES = 32   # saves launched per host sync and copied to the host together (32 x 100k walkers x 3 x 4 B = 38 MB)


def _pad_to(n, unit=4096):
    """The padded walker count a kernel is compiled for: powers of two from ``unit`` (a compile per size)."""
    return max(unit, 1 << int(math.ceil(math.log2(max(n, 1)))))


def simulate_trajectories_adaptive(n_walkers, diffusivity, geometry, T_max, dt_save, *, seed=0, r0=None,
                                   steps_per_round=16, safety_sigma=6.0, n_classes=4, sub_steps=None,
                                   walker_batch_size=100_000, require_gpu=None, storage_dtype=np.float32,
                                   candidate_cache=True, candidate_k_start=64, field_basis=None, field_reuse_intervals=4,
                                   field_sample_every=1, spec=None, spool=None):
    """A :class:`~dmipy_sim.persistent_walk.PersistentWalk` of ``geometry`` with adaptive stepping (module
    docstring). ``geometry`` must offer ``wall_scales``, ``reflect_with_log_weight`` and ``classify_positions_exact``
    and be impermeable. ``steps_per_round`` is the finest class's steps per round (the round is
    ``steps_per_round`` finest steps long); ``n_classes`` the radius classes; ``sub_steps`` overrides the finest
    sub-step count per save (rounded up to a multiple of ``steps_per_round``). With ``candidate_cache`` (the
    default) each round gathers, per walker, the segments whose surface lies within the round's deterministic
    excursion once (``candidate_k_start`` entries, widened when a walker has more in reach) and steps against
    those instead of the 27-cell gather at every step. With ``field_basis`` (a
    :class:`~dmipy_sim.fields.strand_field.StrandFieldBasis`) the walk samples the field's channels itself: once per
    save interval every walker gathers the nearest segment of each strand within the basis' cutoff, and at the end
    of every round the channels are evaluated against that list at the walker's position and accumulated; the
    interval's mean, the domain mean subtracted, is the sample the pack's path channel stores for that save
``field_sample_every`` reads the field at every that-many-th save only (the field channel keeps a few modes over
the walk, so it lives on its own, coarser grid than the positions; the sample of a read save is that save's
interval mean as before).
    (``PersistentWalk.field_samples``), so the tier costs one gather and a few fixed-list evaluations per
    walker-save rather than a field evaluation per stored point. The list is gathered every ``field_reuse_intervals``
    save intervals with the margin the walkers can travel in between (six sigma of that excursion) and masked by
    the true distance when evaluated, so reuse costs no strand within the cutoff.
    ``spec`` is the situation the walk records (the bundle spec ``walk_spec`` drives it by); without one the
    geometry writes its own, which for 12,196 cited centerlines is 100 MB of Python lists. ``spool`` names the
    walk's batches in the run's record (``"<spool>-batch-NNNN"``): every finished batch is written there at once
    (:meth:`dmipy_sim.run.Run.spool`), and a run resumed in that record reads the batches it finds instead of
    walking them again -- a killed walk costs the batch in progress; ``None`` spools nothing.
    """
    with Run("simulate_trajectories_adaptive", params=dict(n_walkers=n_walkers, diffusivity=diffusivity, geometry=type(geometry).__name__, T_max=T_max, dt_save=dt_save, walker_batch_size=walker_batch_size)) as run:
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

        @jax.jit
        def _counts(r):
            """The class counts of the walkers as they stand (free first): what sizes the windows of a chunk's saves."""
            d_wall, R_near = scales_dev(r)
            bucket, _ = _bucket(d_wall, R_near)
            return jnp.bincount(bucket.astype(jnp.int32) + 1, length=n_classes + 1)

        programs = {}

        def _save_program(k_cand, pads):
            """ONE jitted program for a whole save interval: every round's placement, free steps, bucketing and the
            class kernels, with the class windows of STATIC width ``pads`` (per class; 0 for a class the chunk has no
            walker in), so the host dispatches once per save and waits on nothing. A window narrower than its class
            or a candidate list shorter than a walker's reach raises the program's overflow flag, read once per
            chunk, and the chunk is redone with wider ones. The class kernel: gather the class's walkers by index,
            step them ``m`` times at ``step_l``, scatter them back; the window's tail beyond the class repeats a
            walker whose result is dropped. With a candidate cache the segments a walker can meet in the round --
            those whose surface lies within its deterministic excursion ``m step_l`` of its start -- are gathered
            once and the ``m`` steps test only them."""
            n_pad_max = max(pads) if pads else 0

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

            def save(r, keys, m_c, step_l_arr, reach_arr):
                """``m_c``, ``step_l_arr``, ``reach_arr``: the classes' step counts, step lengths and reaches as runtime
                scalars (as the per-class kernels took them), so the traced body and its rounding are theirs."""
                n = r.shape[0]
                dlog = jnp.zeros(n, jnp.float32); n_free_acc = jnp.int32(0); n_kern = jnp.int32(0); n_w_max = jnp.int32(0)
                over = jnp.bool_(False); cmax = jnp.zeros(n_classes, jnp.int32); r_rounds = []
                for _ in range(n_rounds):
                    d_wall, R_near = scales_dev(r)
                    bucket, far_j = _bucket(d_wall, R_near)
                    r, keys = _free_step(r, keys, d_wall, far_j)
                    order = jnp.argsort(bucket)
                    counts = jnp.bincount(bucket.astype(jnp.int32) + 1, length=n_classes + 1)
                    order_p = jnp.concatenate([order, jnp.full(max(n_pad_max, 1), n, order.dtype)])
                    n_free_acc = n_free_acc + counts[0]; start = counts[0]
                    for c in range(n_classes):
                        n_c = counts[c + 1]; cmax = cmax.at[c].max(n_c)
                        if pads[c] == 0:
                            over = over | (n_c > 0)
                            continue
                        over = over | (n_c > pads[c])

                        def run_class(op, c=c):
                            r, keys, dlog, start, n_c = op
                            sel = jax.lax.dynamic_slice(order_p, (start,), (pads[c],))
                            sel_g = jnp.minimum(sel, n - 1)
                            r_c, key_c, dl_c, n_w = stepped(r[sel_g], keys[sel_g], m_c[c], step_l_arr[c], reach_arr[c])
                            real = jnp.arange(pads[c]) < n_c
                            sel_r = jnp.where(real, sel, n)                          # out of bounds: dropped (-1 would wrap)
                            r = r.at[sel_r].set(r_c, mode="drop"); keys = keys.at[sel_r].set(key_c, mode="drop")
                            dlog = dlog.at[sel_r].add(dl_c, mode="drop")
                            return r, keys, dlog, jnp.where(real, n_w, 0).max()

                        def skip_class(op):
                            r, keys, dlog, start, n_c = op
                            return r, keys, dlog, jnp.int32(0)
                        # a class with no walker this round runs nothing (the host skipped its launch before)
                        r, keys, dlog, n_w_c = jax.lax.cond(n_c > 0, run_class, skip_class, (r, keys, dlog, start, n_c))
                        n_w_max = jnp.maximum(n_w_max, n_w_c)
                        n_kern = n_kern + n_c * m_c[c]
                        start = start + n_c
                    r_rounds.append(r)
                return r, keys, dlog, cmax, n_free_acc, n_kern, n_w_max, over, (jnp.stack(r_rounds) if sampling else None)
            return jit_with_tables(geometry, geometry.TABLES, save)

        def program_for(k_cand, pads):
            if (k_cand, pads) not in programs:
                programs[(k_cand, pads)] = _save_program(k_cand, pads)
            return programs[(k_cand, pads)]

        def pads_from(cmax):
            """The windows of the next chunk: each class's largest count seen, padded (a class within a tenth of its
            window's top takes the next width, so a class that grows a little does not redo a chunk); a window
            wider than the class steps its tail for nothing, so the width is the count's own padding wherever
            the headroom allows."""
            out = []
            for c_ in cmax:
                n = int(c_)
                if n == 0:
                    out.append(0); continue
                pad = _pad_to(n)
                out.append(_pad_to(pad + 1) if n > 0.9 * pad else pad)
            return tuple(out)

        reach_c = [float(m) * float(l) + NUDGE for m, l in zip(steps_c, step_l_c)]
        k_cand = int(candidate_k_start)

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
            counts = jit_with_tables(geometry, geometry.TABLES, jax.vmap(lambda p: geometry.reach_candidates(p, jnp.float32(max(reach_c)), 1)[2]))(jnp.asarray(sample))
            k_cand = max(int(candidate_k_start), 1 << int(math.ceil(math.log2(1.5 * max(int(np.asarray(counts).max()), 1)))))
            log.info("adaptive: candidate list of %d (up to %d segments within %.2f um of a start position)", k_cand,
                     int(np.asarray(counts).max()), max(reach_c) * 1e6)
        sampling = field_basis is not None
        if sampling:
            from ..fields.strand_field import StrandFieldBasis
            if not isinstance(field_basis, StrandFieldBasis):
                raise TypeError("field_basis must be a StrandFieldBasis: the walk samples the strand field along the path")
            f_reuse = max(1, int(field_reuse_intervals))
            f_every = max(1, int(field_sample_every))                    # the field's own save grid: every f_every saves
            f_margin = 6.0 * math.sqrt(2.0 * D * dt_actual * f_reuse)     # six sigma of the excursion over the reused intervals
            f_reach = float(field_basis.gather_radius_m)                 # the closed form's reach: the far switch with a far grid
            f_radius = min(f_reach + f_margin, 2.0 * f_reach)
            _at = field_basis.channels_at_device()
            f_mean = jnp.asarray(field_basis.mean, jnp.float32)
            n_tf = len(range(0, n_t, f_every))                          # the saves the field is read at
            field_all = np.empty((n_walkers, n_tf, 13), np.float32)
            f_chunk = 8192                                                # the gather's per-strand minima are n_strands per walker
            # size the list from the start positions at the gather radius
            sample = r0_all[np.random.default_rng(int(seed) + 5).choice(n_walkers, size=min(n_walkers, 20_000), replace=False)]
            n_probe = np.concatenate([np.asarray(field_basis.within_device(radius_m=f_radius, k=1)(jnp.asarray(sample[i:i + f_chunk]))[2])
                                      for i in range(0, sample.shape[0], f_chunk)])
            f_k = min(field_basis.segments_max + 1, 1 << int(math.ceil(math.log2(1.5 * max(int(n_probe.max()), 1)))))
            _list = dict(k=f_k, f=field_basis.within_device(radius_m=f_radius, k=f_k))   # widened when a walker outgrows it
            log.info("adaptive: field sampled in the walk: list of %d segments gathered every %d saves at %.1f um (cutoff %.0f um + "
                     "margin), up to %d in reach at the start", f_k, f_reuse, f_radius * 1e6, f_reach * 1e6, int(n_probe.max()))

            def within_dev(r):
                """The segments within reach of every walker of ``r``; the list widens (doubling, up to ``segments_max + 1``)
                when a walker has more segments in reach than it holds -- the start positions sized it, and a walker can
                drift into a denser neighbourhood."""
                while True:
                    parts = [_list["f"](r[i:i + f_chunk]) for i in range(0, r.shape[0], f_chunk)]
                    seg, keep, n_str = (jnp.concatenate([q[j] for q in parts]) for j in range(3))
                    n_max = int(n_str.max())
                    if n_max <= _list["k"]:
                        return seg, keep, n_str
                    k_new = min(field_basis.segments_max + 1, 1 << int(math.ceil(math.log2(n_max + 1))))
                    if k_new <= _list["k"]:
                        raise ValueError(f"a walker has {n_max} segments within {f_radius * 1e6:.0f} um, more than segments_max={field_basis.segments_max}")
                    log.info("adaptive: field segment list widened to %d (a walker had %d segments within %.0f um)", k_new, n_max, f_radius * 1e6)
                    _list["k"] = k_new; _list["f"] = field_basis.within_device(radius_m=f_radius, k=k_new)

            def at_dev(r, seg, keep):
                return jnp.concatenate([_at(r[i:i + f_chunk], seg[i:i + f_chunk], keep[i:i + f_chunk])
                                        for i in range(0, r.shape[0], f_chunk)])
        sdt = np.dtype(storage_dtype).type
        positions = np.empty((n_walkers, n_t, 3), sdt); dlog_all = np.zeros((n_walkers, n_t), sdt)
        positions[:, 0] = r0_all
        n_free = 0; n_kernel_steps = 0; n_illegal = 0
        for b, (s, e) in enumerate(run.batches(n_walkers, walker_batch_size, what="adaptive")):
            nb = e - s
            if spool is not None:
                got = run.spooled(f"{spool}-batch-{b:04d}")
                if got is not None:                                   # walked before the kill: read back, not redone
                    arr, hdr = got
                    positions[s:e] = arr["positions"]; dlog_all[s:e] = arr["boundary_local_time"]
                    if sampling:
                        field_all[s:e] = arr["field_samples"]
                    n_free += int(hdr["n_free"]); n_kernel_steps += int(hdr["n_kernel_steps"])
                    log.info("  adaptive: batch %d read from the spool (%d walkers)", b, nb)
                    continue
            r = jnp.asarray(r0_all[s:e]); keys = keys_all[s:e]
            n_free_0, n_kernel_0 = n_free, n_kernel_steps
            if sampling:
                seg, keep, n_str = within_dev(r)
                field_all[s:e, 0] = np.asarray(at_dev(r, seg, keep) - f_mean)
            pads = pads_from(np.asarray(_counts(r))[1:])                 # the first chunk's windows (the classes, not the free count): one sync per batch
            t = 1
            while t < n_t:
                # a chunk of saves, one dispatch each, launched without waiting on the device; the programs' overflow
                # (a class wider than its window, a walker with more segments in reach than the candidate list) is
                # read once per chunk, and the chunk is redone from its start with wider windows or list when it
                # happened; the saves' positions and contact are stacked on the device and copied to the host once
                t_end = min(t + CHUNK_SAVES, n_t)
                r_c0, keys_c0 = r, keys
                seg_c0 = (seg, keep, n_str) if sampling else None
                while True:
                    r, keys = r_c0, keys_c0
                    if sampling:
                        seg, keep, n_str = seg_c0
                    rs, dls, fs = [], [], []
                    cmax_c = jnp.zeros(n_classes, jnp.int32); free_c = jnp.int32(0); kern_c = jnp.int32(0); nw_c = jnp.int32(0); over_c = jnp.bool_(False)
                    prog = program_for(k_cand, pads)
                    m_arr = jnp.asarray(steps_c, jnp.int32); sl_arr = jnp.asarray([float(x) for x in step_l_c], jnp.float32); re_arr = jnp.asarray(reach_c, jnp.float32)
                    for tt in range(t, t_end):
                        read_field = sampling and (tt % f_every == 0)
                        if sampling and (tt - 1) % f_reuse == 0:                 # the list, reused over f_reuse intervals with its margin
                            seg, keep, n_str = within_dev(r)
                        r, keys, dlog_int, cmax_s, free_s, kern_s, nw_s, over_s, r_rounds = prog(r, keys, m_arr, sl_arr, re_arr)
                        cmax_c = jnp.maximum(cmax_c, cmax_s); free_c = free_c + free_s; kern_c = kern_c + kern_s
                        nw_c = jnp.maximum(nw_c, nw_s); over_c = over_c | over_s
                        rs.append(r); dls.append(dlog_int)
                        if read_field:                                            # the field at every round's positions, averaged
                            f_acc = jnp.zeros((nb, 13), jnp.float32)
                            for i_ in range(n_rounds):
                                f_acc = f_acc + at_dev(r_rounds[i_], seg, keep)
                            fs.append(f_acc / jnp.float32(n_rounds) - f_mean)
                    cmax_h = np.asarray(cmax_c); n_w = int(nw_c) if cached else 0; over = bool(over_c)   # the chunk's one sync
                    if not over and n_w <= k_cand:
                        break
                    if n_w > k_cand:
                        k_cand = 1 << int(math.ceil(math.log2(n_w)))
                        log.info("adaptive: candidate list widened to %d (a walker had %d segments in reach); saves %d-%d redone", k_cand, n_w, t, t_end - 1)
                    if over:
                        log.info("adaptive: class windows %s too narrow for counts %s; saves %d-%d redone", pads, cmax_h.tolist(), t, t_end - 1)
                    pads = tuple(max(p_, q_) for p_, q_ in zip(pads, pads_from(cmax_h)))
                n_free += int(free_c); n_kernel_steps += int(kern_c)
                pads = tuple(max(p_, q_) for p_, q_ in zip(pads, pads_from(cmax_h)))   # windows grow with the classes, never shrink
                positions[s:e, t:t_end] = np.asarray(jnp.stack(rs, axis=1), sdt)
                dlog_all[s:e, t:t_end] = np.asarray(jnp.stack(dls, axis=1), sdt)
                if fs:
                    field_all[s:e, [tt // f_every for tt in range(t, t_end) if tt % f_every == 0]] = np.asarray(jnp.stack(fs, axis=1))
                run.progress(s + nb * (t_end - 1) / n_t, n_walkers)              # the batch's fraction of its saves
                t = t_end
            # the guarantee: nobody changed pool
            comp_end = np.minimum(np.asarray(geometry.classify_positions_exact(r), np.int32), 1)
            n_illegal += int((comp_end != comp_all[s:e]).sum())
            if spool is not None and not n_illegal:
                arrays = dict(positions=positions[s:e], boundary_local_time=dlog_all[s:e])
                if sampling:
                    arrays["field_samples"] = field_all[s:e]
                run.spool(f"{spool}-batch-{b:04d}", arrays, dict(batch=b, start=s, end=e, n_free=n_free - n_free_0,
                                                                  n_kernel_steps=n_kernel_steps - n_kernel_0))
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
        walk = PersistentWalk(positions, float(dt_actual), int(n_min), float(dt_min), boundary_local_time=dlog_all,
                              compartment=comp, illegal_crossings=0, seed=int(seed), diffusivity=D, geometry=geometry, spec=spec,
                              stepping=stepping, field_basis=field_basis, field_samples=(field_all if sampling else None),
                              field_sample_every=(f_every if sampling else 1))
        object.__setattr__(walk, "run", run)
        return walk
