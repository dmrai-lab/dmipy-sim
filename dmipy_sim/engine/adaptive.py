"""The trajectory producer with adaptive stepping: every walker steps at the rate its own situation needs.

The fused producer (:func:`dmipy_sim.engine.core.simulate_trajectories`) steps every walker at one ``dt_sim``,
resolved from the substrate's smallest feature (the step rule of its smallest tube), so a walker in the extra-
axonal space of a strand substrate pays for the thinnest strand it will never touch. Here a block is
walked in ROUNDS, and at each round every walker is placed by the geometry's ``wall_scales``: its distance to
the nearest wall it can hit and the radius (curvature scale) of that wall.

- A walker farther from every wall than ``safety_sigma`` times the round's rms excursion takes ONE free
  Gaussian step for the round (truncated at its wall distance, so it cannot cross: the truncation removes a
  ``safety_sigma`` tail, 6e-8 of the mass at six sigma), accrues no contact and keeps its label.
- Every other walker runs the geometry's wall-aware kernel for the round at the step the family's rule gives for
  ITS nearest wall's radius, in radius classes doubling in step time (``dt_c = dt_min 2^c``, i.e. radius
  bounds ``R_min 2^(c/2)``): the finest class is the fused producer's own step and noise sequence.

Walkers are re-bucketed every round on the host (index arrays), the kernels are jitted per class and padded
walker count, and the positions stay on the device. The rule's numbers -- the finest step, the rounds, the
classes, the free-step threshold -- are one :class:`AdaptivePlan` (:func:`adaptive_plan`), which the rounds here
read and a backend (``backend=``, dmrai-lab/dmipy-sim#635) receives as ``WalkRequest.stepping`` to run the same
rule its own way. The recorded channels are the fused producer's, accumulated as its are (the bands of the
positions and of the contact, the field's modes, :mod:`dmipy_sim.engine.accumulate`), per round; the
compartment label is static (the geometries this serves are impermeable).

``illegal_crossings`` here is a coarser guard than the fused producer's: a walker's pool is compared once, at
the walk's end, against its start, never at every sub-step -- so a walker that crosses a wall and comes back
before the end is not seen (dmrai-lab/dmipy-sim#675). A void case with a per-save containment invariant
(``engine.backends.void_invariant``, the ``strands_void`` parity case) is what proves a backend walks the
starts it is actually given rather than, say, an empty candidate gather read as an unbounded one. The walk is
refused if any walker ended in the wrong pool.
"""
from __future__ import annotations

import logging
import math
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from .backends import JAX, WalkRequest, FieldSampling, resolve as resolve_backend
from .tables import jit_with_tables
from .physics import seed_walkers, pair_step
from ..math.step_law import LONGEST_STEP
from ..run import Run

log = logging.getLogger("dmipy_sim")


class AdaptivePlan(NamedTuple):
    """The adaptive rule's numbers for one walk: what the rounds step by, derived once from the substrate, the
    diffusivity and the block (:func:`adaptive_plan`), read by the JAX rounds and handed to a backend."""
    steps_per_round: int      #: K, the finest class's steps per round
    n_min: int                #: finest steps per block (a multiple of K)
    n_rounds: int             #: rounds per block, ``n_min // K``
    dt_min: float             #: the finest step's duration (s)
    dt_round: float           #: a round's duration (s)
    n_classes: int            #: radius classes
    safety_sigma: float       #: a walker farther than this many ``sigma_round`` from every wall steps freely
    sigma_round: float        #: the round's per-axis rms excursion, ``sqrt(2 D dt_round)`` (m)
    far_at: float             #: ``safety_sigma * sigma_round`` (m)
    R_min: float              #: the substrate's smallest feature (m), the finest class's radius
    R_bounds: tuple           #: a class's lower radius bound, ``R_min 2^(c/2)`` (m)
    steps_c: tuple            #: kernel steps per round per class, ``K >> c``
    step_l_c: tuple           #: the step length per class (m), ``sqrt(6 D dt_round / steps_c)``
    reach_c: tuple            #: the round's deterministic excursion per class (m), ``steps_c * LONGEST_STEP * step_l_c``: the candidate reach
    candidate_k: int          #: the candidate list's width (the JAX rounds widen it; a backend keeps its own)
    eps_D: float              #: the tolerance on the walk's diffusion spectrum every class's step is sized for
    f_band_hz: float          #: the band (Hz) the tolerance holds to: a stated one, else the block grid's Nyquist frequency
    dt_band_s: float          #: the band term's step (s), ``band_step(f_band_hz, eps_D)``: the ceiling of every class's step


def adaptive_plan(geometry, diffusivity, block_h, *, steps_per_round=16, safety_sigma=6.0, n_classes=4, sub_steps=None,
                  candidate_k_start=64, f_band=None):
    """The :class:`AdaptivePlan` of a walk of ``geometry`` at ``diffusivity`` accumulated in blocks of ``block_h``: the finest
    sub-step count from the step rule (:func:`~dmipy_sim.engine.physics.tolerance_sub_steps` and the rules beside it;
    ``sub_steps`` overrides it), rounded up to a multiple of ``steps_per_round``; the classes doubling in step time
    from the smallest feature, which is the rule's pore term at the class's radius, and none stepping longer than the
    rule's band term, which does not grow with the radius."""
    from .physics import resolve_sub_steps, length_scales_of, STEP_TOLERANCE
    from ..math.step_law import band_step
    D = float(diffusivity)
    K = int(steps_per_round)
    eps_D = float(STEP_TOLERANCE)
    f_band_hz = float(f_band) if f_band is not None else 0.5 / float(block_h)
    dt_band = float(band_step(f_band_hz, eps_D))
    n_min = resolve_sub_steps(geometry, D, block_h, surface=True, override=sub_steps, f_band=f_band)
    n_min = K * int(math.ceil(n_min / K))                          # the finest class divides the round
    n_rounds = n_min // K
    dt_min = block_h / n_min; dt_round = block_h / n_rounds
    R_min = float(length_scales_of(geometry).min_feature)
    sigma_round = math.sqrt(2.0 * D * dt_round)                    # per axis
    far_at = float(safety_sigma) * sigma_round                    # |step| > 6 sigma: 1.2e-7 of a 3-D Gaussian's mass
    n_classes = int(max(1, min(n_classes, int(math.log2(K)) + 1)))
    # kernel steps per round, per class: the doubling (the pore term at the class radius), and no class stepping
    # longer than the band term allows (a power of two, so a class's steps still divide the round)
    m_band = 1
    while m_band < K and dt_round / m_band > dt_band * (1.0 + 1e-9):
        m_band *= 2
    steps_c = tuple(max(K >> c, m_band) for c in range(n_classes))
    step_l_c = tuple(float(np.float32(math.sqrt(6.0 * D * dt_round / m))) for m in steps_c)
    R_bounds = tuple(R_min * 2.0 ** (c / 2.0) for c in range(n_classes))   # a class's lower radius bound
    NUDGE = 1e-4 * R_min
    reach_c = tuple(float(m) * LONGEST_STEP * l + NUDGE for m, l in zip(steps_c, step_l_c))   # the longest atom, m times
    return AdaptivePlan(K, int(n_min), int(n_rounds), float(dt_min), float(dt_round), n_classes, float(safety_sigma),
                        sigma_round, far_at, R_min, R_bounds, steps_c, step_l_c, reach_c, int(candidate_k_start),
                        eps_D, f_band_hz, dt_band)


def _pad_to(n, unit=4096):
    """The padded walker count a kernel is compiled for: powers of two from ``unit`` (a compile per size)."""
    return max(unit, 1 << int(math.ceil(math.log2(max(n, 1)))))


def simulate_trajectories_adaptive(n_walkers, diffusivity, geometry, T_max, block_h=None, *, seed=0, r0=None,
                                   steps_per_round=16, safety_sigma=6.0, n_classes=4, sub_steps=None,
                                   walker_batch_size=100_000, require_gpu=None, candidate_cache=True, candidate_k_start=64,
                                   field_basis=None, field_reuse_intervals=4, field_read_interval=None, K_c3=None,
                                   field_list_k=256, spec=None, spool=None, backend=JAX, f_band=None, K=None, block_order=None,
                                   window_T=None, raw_subset=0, envelope=None):
    """A :class:`~dmipy_sim.coefficient_walk.CoefficientWalk` of ``geometry`` with adaptive stepping (module
    docstring). ``geometry`` must offer ``wall_scales``, ``reflect_with_log_weight`` and ``classify_positions_exact``
    and be impermeable (so the occupancy is one static label per walker).

    The walk accumulates what the fused producer accumulates (:mod:`dmipy_sim.engine.accumulate`), per ROUND: a
    block of ``h`` (the band's own, :func:`~dmipy_sim.engine.accumulate.block_length`, unless ``block_h`` gives one)
    is a whole number of rounds, every lane's clock advances one round at a time, and a lane's block ends when its
    clock crosses the block boundary -- which every lane's does together, since a round is one ``dt_round`` for every
    walker whatever its class. Each round adds the moments of the round's displacement and contact about the block's
    centre (``M_j += tau^j dx``, ``B_j += tau^j dlog``, ``tau`` the round's midpoint), and at the block's end the
    moments are projected onto the ``K + 1`` cosine modes against the block's row of
    :func:`~dmipy_sim.replay.compression.block_table` into a float32 level-2 accumulator. A round's interior path is
    unresolved (a free lane takes one Gaussian step for it), so the quadrature a certificate charges is the block
    table's ``rel(m, h)`` AND the round's ``rel(0, dt_round)`` (``block["round_s"]``). ``steps_per_round`` is the
    finest class's steps per round; ``n_classes`` the radius classes; ``sub_steps`` overrides the finest sub-step
    count per block (rounded up to a multiple of ``steps_per_round``). With ``candidate_cache`` (the default) each
    round gathers, per walker, the segments whose surface lies within the round's deterministic excursion once
    (``candidate_k_start`` entries, widened when a walker has more in reach) and steps against those instead of
    the 27-cell gather at every step.

    With ``field_basis`` (a :class:`~dmipy_sim.fields.strand_field.StrandFieldBasis`) the walk READS the field's
    channels itself at the end of every ``n_read``-th round at the walker's position, ``n_read`` the largest divisor
    of the rounds per block whose interval is within the read rule's ``dt_f`` (:func:`field_read_of`, unless
    ``field_read_interval`` gives one: a pilot of :data:`~dmipy_sim.engine.accumulate.PILOT_WALKERS` walkers reads
    every round and measures the read series' quadratic variation and the trapezoid's constant; the round itself
    when the rule asks for less -- the read is never sub-stepped along a round), the TRAPEZOID of consecutive reads
    accumulated onto ``K_c3`` cosine modes per window and the read series' quadratic variation beside them (``K_c3``
    given, or derived on the same pilot's block means by :func:`~dmipy_sim.replay.bank.derive_field_band`). Once per ``field_reuse_intervals``
    blocks every walker gathers the nearest segment of each strand within the basis' cutoff plus the margin the
    walkers can travel in between (six sigma of that excursion), and the reads evaluate the channels against that
    list, masked by the true distance when evaluated, so reuse costs no strand within the cutoff; the domain mean
    is subtracted. ``raw_subset=n`` keeps the first ``n`` walkers' block ends (positions, the block's contact) and
    their field reads at the walk's read instants. ``spec`` is the situation the walk records (the bundle spec ``walk_spec`` drives it by);
    without one the geometry writes its own, which for 12,196 cited centerlines is 100 MB of Python lists.
    ``spool`` names the walk's batches in the run's record (``"<spool>-batch-NNNN"``): every finished batch's
    accumulators are written there at once (:meth:`dmipy_sim.run.Run.spool`), and a run resumed in that record reads
    the batches it finds instead of walking them again -- a killed walk costs the batch in progress; ``None``
    spools nothing. Another ``backend`` is refused by name: the adaptive walk's backend route returned block ends,
    which an accumulating walk does not take (dmrai-lab/dmipy-sim#747 is the backend's coefficient form).
    """
    with Run("simulate_trajectories_adaptive", params=dict(n_walkers=n_walkers, diffusivity=diffusivity, geometry=type(geometry).__name__,
                                                            T_max=T_max, block_h=block_h, walker_batch_size=walker_batch_size)) as run:
        from .gpu import check_gpu
        from .accumulate import (block_length, default_K, window_count, BLOCK_ORDER, PRECISION, PILOT_WALKERS,
                                 coefficient_form, field_mode_form, field_rule_scale, read_checks, pilot_walkers)
        from ..replay.compression import block_table, cosine_band_scale
        from ..coefficient_walk import CoefficientWalk
        if not hasattr(geometry, "wall_scales"):
            raise TypeError(f"{type(geometry).__name__} offers no wall_scales; adaptive stepping needs the distance to the "
                            "nearest wall and its curvature scale")
        if geometry.permeability is not None:
            raise NotImplementedError("adaptive stepping is for impermeable substrates: a permeable wall needs the kernel at "
                                      "every step (the crossing decision is per contact)")
        _backend = resolve_backend(backend)
        _backend_name = JAX if _backend is None else str(_backend.name)
        check_gpu(n_walkers, require_gpu, what="simulate_trajectories_adaptive")
        D = float(diffusivity)
        # the windows, the band and the block (the fused producer's resolution, dmrai-lab/dmipy-sim#740)
        n_win, T_win = window_count(T_max, window_T)
        K = default_K(T_win) if K is None else int(K)
        m = BLOCK_ORDER if block_order is None else int(block_order)
        if not 0 <= m <= 3:
            raise ValueError(f"block_order is the order m of the block table, 0..3; got {m}")
        if block_h is None:
            h, block_rule = block_length(T_win, K), "derived"
        else:
            h, block_rule = float(T_win) / max(int(round(float(T_win) / float(block_h))), 1), "given"
        n_bw = int(round(float(T_win) / h)); n_blocks = n_win * n_bw; dt_actual = float(T_win) / n_bw
        n_t = n_blocks + 1
        plan = adaptive_plan(geometry, D, dt_actual, steps_per_round=steps_per_round, safety_sigma=safety_sigma,
                             n_classes=n_classes, sub_steps=sub_steps, candidate_k_start=candidate_k_start, f_band=f_band)
        Ks, n_min, n_rounds, dt_min, dt_round, n_classes = plan.steps_per_round, plan.n_min, plan.n_rounds, plan.dt_min, plan.dt_round, plan.n_classes
        R_min, sigma_round, far_at = plan.R_min, plan.sigma_round, plan.far_at
        steps_c = list(plan.steps_c)
        step_l_c = [jnp.float32(l) for l in plan.step_l_c]
        R_bounds = np.array(plan.R_bounds)
        log.info("adaptive walk: %d finest steps per block (dt_min %.3g us) in %d rounds of %d, %d radius classes from "
                 "R_min %.2f um (steps per round %s), free step beyond %.2f um of a wall",
                 n_min, dt_min * 1e6, n_rounds, Ks, n_classes, R_min * 1e6, steps_c, far_at * 1e6)

        sampling = field_basis is not None
        field_read = None
        if sampling:
            from ..fields.strand_field import StrandFieldBasis
            if not isinstance(field_basis, StrandFieldBasis):
                raise TypeError("field_basis must be a StrandFieldBasis: the walk reads the strand field along the path")
            # the caller's read, or the round until the pilot has measured the read series (RPK.md 9.5.5)
            field_read = field_read_of(_spec_holder(spec, geometry), envelope, T_win, plan, given=field_read_interval)
        elif K_c3 is not None or field_read_interval is not None:
            raise ValueError("K_c3 and field_read_interval are the field channel's band and read interval: they go with field_basis=")

        reflect_lw = geometry.reflect_with_log_weight
        cached = bool(candidate_cache) and hasattr(geometry, "reach_candidates") and hasattr(geometry, "_reflect_with")
        NUDGE = 1e-4 * R_min

        def _kernel(k_cand):
            """One jitted dispatch per round and class: gather the class's walkers by index, step them ``m`` times at
            ``step_l`` (both runtime scalars, so one program serves every class), scatter them back; the padded tail
            of ``sel`` repeats a walker whose result is discarded. With a candidate cache the segments a walker can
            meet in the round -- those whose surface lies within its deterministic excursion ``m step_l`` of its
            start -- are gathered once and the ``m`` steps test only them."""
            def one(r, key, M, B, Q, Qb, mm, step_l, reach, tau0, dtau):
                """``mm`` steps of one walker, each step's increments added to the block's moments at the step's own
                midpoint ``tau0 + (i + 1/2) dtau`` (the block's unit): the near-wall walkers are accumulated at the
                step, as the fused producer accumulates every walker."""
                if cached:
                    cand, valid, n_within = geometry.reach_candidates(r, reach, k_cand)

                def body(i, carry):
                    r, key, dlog, M, B, Q, Qb = carry
                    key, sub = jax.random.split(key)
                    step = pair_step(sub) * step_l
                    if cached:
                        r_new, d_perp, _, _ = geometry._step_with(r, step, cand, valid)
                        dlw = -2.0 * d_perp
                    else:
                        r_new, dlw = reflect_lw(r, step, jnp.float32(1.0))
                    dx = r_new - r
                    pw = (tau0 + (i.astype(jnp.float32) + 0.5) * dtau) ** js
                    return (r_new, key, dlog + dlw, M + pw[:, None] * dx[None, :], B + pw * dlw, Q + dx * dx, Qb + dlw * dlw)
                r_f, key_f, dlog_f, M, B, Q, Qb = jax.lax.fori_loop(0, mm, body, (r, key, jnp.float32(0.0), M, B, Q, Qb))
                return r_f, key_f, dlog_f, M, B, Q, Qb, (n_within if cached else jnp.int32(0))
            stepped = jax.vmap(one, in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None, None))

            def run_(r, keys, dlog, M, B, Q, Qb, order, start, n_real, mm, step_l, reach, tau0, dtau):
                """The class's walkers are ``order[start:start + n_real]`` (the walkers sorted by class, on the device);
                the window is ``n_pad`` wide, static, and its tail beyond ``n_real`` is stepped and dropped."""
                sel = jax.lax.dynamic_slice(order, (start,), (n_pad,))
                sel_g = jnp.minimum(sel, r.shape[0] - 1)
                r_c, key_c, dl_c, M_c, B_c, Q_c, Qb_c, n_w = stepped(r[sel_g], keys[sel_g], M[sel_g], B[sel_g], Q[sel_g], Qb[sel_g],
                                                                     mm, step_l, reach, tau0, dtau)
                real = jnp.arange(n_pad) < n_real
                sel_r = jnp.where(real, sel, r.shape[0])                      # out of bounds: dropped (-1 would wrap)
                r = r.at[sel_r].set(r_c, mode="drop"); keys = keys.at[sel_r].set(key_c, mode="drop")
                dlog = dlog.at[sel_r].add(dl_c, mode="drop")
                M = M.at[sel_r].set(M_c, mode="drop"); B = B.at[sel_r].set(B_c, mode="drop")
                Q = Q.at[sel_r].set(Q_c, mode="drop"); Qb = Qb.at[sel_r].set(Qb_c, mode="drop")
                return r, keys, dlog, M, B, Q, Qb, jnp.where(real, n_w, 0).max()
            return jit_with_tables(geometry, geometry.TABLES, run_)

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

        reach_c = list(plan.reach_c)
        k_cand = int(plan.candidate_k)
        scales_dev = geometry._wall_scales_device()

        @jax.jit
        def _bucket(d_wall, R_near):
            """Per walker: -1 for a free step, else its radius class."""
            far = d_wall > jnp.float32(far_at)
            cls = jnp.clip(jnp.floor(2.0 * jnp.log2(jnp.maximum(R_near, jnp.float32(R_min)) / jnp.float32(R_min))), 0, n_classes - 1)
            return jnp.where(far, -1, cls).astype(jnp.int8), far

        js = jnp.arange(m + 1, dtype=jnp.float32)
        taus = jnp.asarray((np.arange(n_rounds) + 0.5) / n_rounds - 0.5, jnp.float32)     # a round's midpoint in the block's unit
        tau_starts = jnp.asarray(np.arange(n_rounds) / n_rounds - 0.5, jnp.float32)      # a round's start

        @jax.jit
        def _free_step(r, keys, d_wall, far, M, Q, j):
            """One Gaussian step of the round for the far walkers, truncated at their wall distance, added to the
            block's moments at the round's midpoint (the round's interior path is unresolved: the block table's
            order 0 at the round, which the certificate charges as ``rel(0, dt_round)``); every walker's key
            advances (the far ones spend a draw), so the stream stays per walker."""
            def one(k):
                k, sub = jax.random.split(k)
                return k, jax.random.normal(sub, (3,), dtype=jnp.float32) * jnp.float32(sigma_round)
            keys_new, g = jax.vmap(one)(keys)
            norm = jnp.linalg.norm(g, axis=1, keepdims=True)
            cap = (jnp.minimum(d_wall, jnp.float32(far_at)) * jnp.float32(0.999))[:, None]
            g = jnp.where(norm > cap, g * cap / jnp.maximum(norm, 1e-30), g)
            g = jnp.where(far[:, None], g, 0.0)
            pw = taus[j] ** js
            return r + g, jnp.where(far[:, None], keys_new, keys), M + pw[None, :, None] * g[:, None, :], Q + g * g

        # ── the accumulation per block (engine/accumulate.py's rule): the kernel-stepped walkers at their steps, the
        # free ones at their round
        table = jnp.asarray(block_table(1.0, n_bw, K, m), jnp.float32)                       # (n_bw, K+1, m+1)
        scale = jnp.asarray(cosine_band_scale(K, T_win), jnp.float32)
        hi = jax.lax.Precision.HIGHEST
        n_ch = 13 if sampling else 0
        K_f = None

        @jax.jit
        def _read_moments(F, Qf, Qd, b_prev, b, j_mid, half_dt_f):
            """The trapezoid increment of a read interval added to the field's moments at the interval's midpoint, its
            square to ``Qf`` and the read series' increment squared to ``Qd``."""
            inc = (b_prev + b) * half_dt_f
            pw = (jnp.float32(j_mid) / n_rounds - 0.5) ** js
            db = b - b_prev
            return F + pw[None, :, None] * inc[:, None, :], Qf + inc * inc, Qd + db * db

        @jax.jit
        def _check_reads(E, bpd, b_prev, b, hit, half_dt, half_dt_d):
            """The trapezoid's checks of a pilot that reads every round: per check ``d`` the trapezoid of every ``d``-th
            read (``hit`` where this read is one) less the trapezoid of every read, and the quadratic variation of the
            series of every ``d``-th read (``E`` ``(nb, 2, n_checks, n_ch)``)."""
            inc = (b_prev + b) * half_dt
            h3 = hit[None, :, None]
            e_d = E[:, 0] - inc[:, None, :] + jnp.where(h3, (bpd + b[:, None, :]) * half_dt_d[None, :, None], 0.0)
            q_d = E[:, 1] + jnp.where(h3, (b[:, None, :] - bpd) ** 2, 0.0)
            return jnp.stack([e_d, q_d], axis=1), jnp.where(h3, b[:, None, :], bpd)

        @jax.jit
        def _project(acc, M, row, table_f, acc_f, F):
            """The block's moments projected onto the modes (the block's row of the table) and added to the level-2
            accumulators."""
            out = acc + jnp.einsum("kj,wjd->wkd", row, M, precision=hi)
            if acc_f is not None:
                out_f = acc_f + jnp.einsum("kj,wjc->wkc", table_f, F, precision=hi)
                return out, out_f
            return out, None

        @jax.jit
        def _project_b(acc_b, B, row):
            return acc_b + jnp.einsum("kj,wj->wk", row, B, precision=hi)

        _, r0_all, keys_all = seed_walkers(geometry, n_walkers, seed, r0)      # the fused producer's own draws
        r0_all = np.asarray(r0_all, np.float32)
        comp_all = np.minimum(np.asarray(geometry.classify_positions_exact(jnp.asarray(r0_all)), np.int32), 1).astype(np.int8)

        if cached:                                                       # size the list from the start positions (the widest reach)
            sample = r0_all[np.random.default_rng(int(seed) + 3).choice(n_walkers, size=min(n_walkers, 20_000), replace=False)]
            counts = jit_with_tables(geometry, geometry.TABLES, jax.vmap(lambda p: geometry.reach_candidates(p, jnp.float32(max(reach_c)), 1)[2]))(jnp.asarray(sample))
            k_cand = max(int(candidate_k_start), 1 << int(math.ceil(math.log2(1.5 * max(int(np.asarray(counts).max()), 1)))))
            log.info("adaptive: candidate list of %d (up to %d segments within %.2f um of a start position)", k_cand,
                     int(np.asarray(counts).max()), max(reach_c) * 1e6)
        if sampling:
            f_reuse = max(1, int(field_reuse_intervals))
            f_margin = 6.0 * math.sqrt(2.0 * D * dt_actual * f_reuse)     # six sigma of the excursion over the reused blocks
            f_reach = float(field_basis.gather_radius_m)                 # the closed form's reach: the far switch with a far grid
            f_radius = min(f_reach + f_margin, 2.0 * f_reach)
            _at = field_basis.channels_at_device()
            f_mean = jnp.asarray(field_basis.mean, jnp.float32)
            f_chunk = 8192                                                # the gather's per-strand minima are n_strands per walker
            sample = r0_all[np.random.default_rng(int(seed) + 5).choice(n_walkers, size=min(n_walkers, 20_000), replace=False)]
            n_probe = np.concatenate([np.asarray(field_basis.within_device(radius_m=f_radius, k=1)(jnp.asarray(sample[i:i + f_chunk]))[2])
                                      for i in range(0, sample.shape[0], f_chunk)])
            f_k = min(field_basis.segments_max + 1, 1 << int(math.ceil(math.log2(1.5 * max(int(n_probe.max()), 1)))))
            _list = dict(k=f_k, f=field_basis.within_device(radius_m=f_radius, k=f_k))   # widened when a walker outgrows it
            log.info("adaptive: field read in the walk every %d of %d rounds per block (dt_f %.3g us, rule %s): list of %d segments gathered "
                     "every %d blocks at %.1f um (cutoff %.0f um + margin), up to %d in reach at the start", field_read["n_read"], n_rounds,
                     field_read["dt_f"] * 1e6, field_read["rule"], f_k, f_reuse, f_radius * 1e6, f_reach * 1e6, int(n_probe.max()))

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
                                        for i in range(0, r.shape[0], f_chunk)]) - f_mean

        if _backend is not None:
            if n_win > 1:
                raise ValueError(f"backend {_backend_name!r}: a walk of several storage windows is walked by the jax kernels only "
                                 "(a backend accumulates one window per walk; dmipy-sim#748 follow-up 1)")
            plan = plan._replace(candidate_k=int(k_cand))
            # the request a backend walks (#747's names): the field read every dt_f seconds (its K_c3 modes, the pilot's
            # or the given), the raw subset's block ends and the field's block means (sample_every = 1) beside them
            req = WalkRequest(geometry=geometry, block_h=float(dt_actual), n_blocks=int(n_blocks), sub_steps=int(n_min), dt_sim=float(dt_min),
                              K=int(K), m=int(m), K_c3=(0 if K_c3 is None else int(K_c3)),
                              field_read_interval=(0.0 if field_read is None else float(field_read["dt_f"])), events=0,
                              raw_subset=int(raw_subset), diffusivity=D, record=True, kappa_over_D=0.0, count_walls=bool(geometry.count_walls),
                              bounce_budget=geometry.bounce_budget, seed=int(seed), stepping=plan,
                              field=(FieldSampling(field_basis, 1, f_reuse, float(f_radius), int(field_list_k)) if sampling else None))
            _reason = _backend.refuses(req)
            if _reason:
                raise ValueError(f"backend {_backend_name!r} refuses this walk of {type(geometry).__name__}: {_reason}")

        n_raw = int(min(max(int(raw_subset), 0), n_walkers))
        pidx = pilot_walkers(n_walkers)                                  # spread over the walk, not its first walkers
        n_pilot = int(pidx.shape[0])

        def _walk_batch(s, e, *, raw, K_f, rd=None):
            """The walk of walkers ``s:e``: the accumulators per window, the quadratic variation, the raw block ends
            when ``raw``; the field's modes at ``K_f`` modes (``None``: the pilot's block means alone), read per ``rd =
            (n_read, dt_f, checks, reads)`` -- every ``n_read`` rounds, the trapezoid's ``checks`` of a pilot that
            reads every round, and the raw subset's reads at their instants (``reads``) or each block's mean."""
            n_read_, dt_f_, checks_, keep_reads = rd if rd is not None else _rd[0]
            half_dt = jnp.float32(0.5 * dt_f_)
            half_dt_d = jnp.asarray(checks_, jnp.float32) * jnp.float32(0.5 * float(dt_round)) if checks_ else None
            rpb = n_rounds // int(n_read_)
            sel = slice(s, e) if e is not None else np.asarray(s)              # a range of walkers, or a pilot's indices
            nb = int(r0_all[sel].shape[0])
            r = jnp.asarray(r0_all[sel]); keys = keys_all[sel]
            C_out = np.empty((nb, n_win, K + 2, 3), np.float32); resid_out = np.empty((nb, n_win, 3), np.float32)
            blt_end = np.empty((nb, n_win), np.float32); blt_bands = np.empty((nb, n_win, K), np.float32)
            qv_pos = np.empty((nb, n_win, 3), np.float32); qv_blt = np.empty((nb, n_win), np.float32)
            fc_out = np.empty((nb, n_win, K_f, n_ch), np.float32) if (sampling and K_f) else None
            qv_f = np.empty((nb, n_win, n_ch), np.float32) if sampling else None
            qv_d = np.empty((nb, n_win, n_ch), np.float32) if sampling else None
            chk = np.empty((nb, n_win, 2, len(checks_), n_ch), np.float32) if (sampling and checks_) else None
            pos = np.empty((nb, n_t, 3), np.float32) if raw else None
            dlog_raw = np.zeros((nb, n_t), np.float32) if raw else None
            n_fs = (n_blocks * rpb + 1) if keep_reads else n_t
            fs_raw = np.empty((nb, n_fs, n_ch), np.float32) if (raw and sampling) else None
            if raw:
                pos[:, 0] = np.asarray(r)
            n_free_0 = n_kernel_0 = 0
            b_prev = seg = keep = None
            if sampling:
                seg, keep, _ = within_dev(r)
                b_prev = at_dev(r, seg, keep)
                if raw:
                    fs_raw[:, 0] = np.asarray(b_prev)
            table_f = jnp.asarray(block_table(1.0, n_bw, int(K_f) - 1, m), jnp.float32) if (sampling and K_f) else None
            for w in range(n_win):
                r_start = r
                acc = jnp.zeros((nb, K + 1, 3), jnp.float32); acc_b = jnp.zeros((nb, K + 1), jnp.float32)
                acc_f = jnp.zeros((nb, int(K_f), n_ch), jnp.float32) if (sampling and K_f) else None
                qp = jnp.zeros((nb, 3), jnp.float32); qb = jnp.zeros((nb,), jnp.float32)
                qf = jnp.zeros((nb, n_ch), jnp.float32) if sampling else None
                qd = jnp.zeros((nb, n_ch), jnp.float32) if sampling else None
                ec = jnp.zeros((nb, 2, len(checks_), n_ch), jnp.float32) if (sampling and checks_) else None
                blt_a = jnp.zeros((nb,), jnp.float32)
                for b in range(n_bw):
                    t = w * n_bw + b + 1                                             # the block's end on the walk, 1-based
                    M = jnp.zeros((nb, m + 1, 3), jnp.float32); B = jnp.zeros((nb, m + 1), jnp.float32)
                    F = jnp.zeros((nb, m + 1, n_ch), jnp.float32) if sampling else None
                    dlog_blk = jnp.zeros(nb, jnp.float32)
                    if sampling and (t - 1) % f_reuse == 0:                       # the list, reused over f_reuse blocks with its margin
                        seg, keep, _ = within_dev(r)
                    if ec is not None:
                        bpd = jnp.broadcast_to(b_prev[:, None, :], (nb, len(checks_), n_ch))   # every check reads at the block's start
                    for j in range(n_rounds):
                        d_wall, R_near = scales_dev(r)
                        bucket, far_j = _bucket(d_wall, R_near)
                        r, keys, M, qp = _free_step(r, keys, d_wall, far_j, M, qp, jnp.int32(j))
                        order, counts = _order(bucket)
                        counts = np.asarray(counts)                          # the one small host transfer per round
                        n_free_0 += int(counts[0]); start = int(counts[0])
                        dlog_int = jnp.zeros(nb, jnp.float32)
                        for c in range(n_classes):
                            n_c = int(counts[c + 1])
                            if n_c == 0:
                                continue
                            while True:                                          # the candidate list widens until it holds every segment in reach
                                kc = nonlocal_k[0]
                                r_try, keys_try, dlog_try, M_try, B_try, qp_try, qb_try, n_max = kernel_for(kc, _pad_to(n_c))(
                                    r, keys, dlog_int, M, B, qp, qb, order, start, n_c, steps_c[c], step_l_c[c], jnp.float32(reach_c[c]),
                                    tau_starts[j], jnp.float32(1.0 / (n_rounds * steps_c[c])))
                                if not cached or int(n_max) <= kc:
                                    r, keys, dlog_int, M, B, qp, qb = r_try, keys_try, dlog_try, M_try, B_try, qp_try, qb_try
                                    break
                                nonlocal_k[0] = 1 << int(math.ceil(math.log2(int(n_max))))
                                log.info("adaptive: candidate list widened to %d (a walker had %d segments in reach)", nonlocal_k[0], int(n_max))
                            n_kernel_0 += n_c * steps_c[c]; start += n_c
                        dlog_blk = dlog_blk + dlog_int
                        if sampling and (j + 1) % n_read_ == 0:
                            b_now = at_dev(r, seg, keep)
                            F, qf, qd = _read_moments(F, qf, qd, b_prev, b_now, jnp.float32(j + 1 - 0.5 * n_read_), half_dt)
                            if ec is not None:
                                hit = jnp.asarray([(j + 1) % d == 0 for d in checks_])
                                ec, bpd = _check_reads(ec, bpd, b_prev, b_now, hit, half_dt, half_dt_d)
                            if raw and keep_reads:
                                fs_raw[:, (t - 1) * rpb + (j + 1) // n_read_] = np.asarray(b_now)
                            b_prev = b_now
                    acc, acc_f = _project(acc, M, table[b], (None if table_f is None else table_f[b]), acc_f, F)
                    acc_b = _project_b(acc_b, B, table[b])
                    blt_a = blt_a + dlog_blk
                    if raw:
                        pos[:, t] = np.asarray(r); dlog_raw[:, t] = np.asarray(dlog_blk)
                        if sampling and not keep_reads:
                            fs_raw[:, t] = np.asarray(F[:, 0, :] / jnp.float32(dt_actual))
                    if e is not None:                                            # the pilot reports no walk progress
                        run.progress(s + nb * t / n_t, n_walkers)
                C, resid = jax.vmap(lambda a, b_, ap: coefficient_form(a, b_, ap, scale))(r_start, r, acc)
                C_out[:, w] = np.asarray(C); resid_out[:, w] = np.asarray(resid)
                blt_end[:, w] = np.asarray(acc_b[:, 0]); blt_bands[:, w] = np.asarray(scale[None, :] * acc_b[:, 1:])
                qv_pos[:, w] = np.asarray(qp); qv_blt[:, w] = np.asarray(qb)
                if sampling:
                    qv_f[:, w] = np.asarray(qf); qv_d[:, w] = np.asarray(qd)
                    if chk is not None:
                        chk[:, w] = np.asarray(ec)
                    if K_f:
                        fc_out[:, w] = np.asarray(jax.vmap(lambda a: field_mode_form(a, T_win))(acc_f))
            comp_end = np.minimum(np.asarray(geometry.classify_positions_exact(r), np.int32), 1)
            n_illegal = int((comp_end != comp_all[sel]).sum())
            out = dict(C=C_out, resid=resid_out, blt_end=blt_end, blt_bands=blt_bands, qv_pos=qv_pos, qv_blt=qv_blt)
            if sampling:
                out["qv_f"] = qv_f; out["qv_d"] = qv_d
                if chk is not None:
                    out["chk"] = chk
                if K_f:
                    out["fc"] = fc_out
            if raw:
                out.update(pos=pos, dlog=dlog_raw)
                if sampling:
                    out["fs"] = fs_raw
            return out, dict(n_free=n_free_0, n_kernel_steps=n_kernel_0, n_illegal=n_illegal)

        def _backend_batch(s, e, *, raw, K_f, rd=None):
            """The walk of walkers ``s:e`` through the backend (#747, #742): its
            :class:`~dmipy_sim.engine.backends.CoefficientBatch` under the names of the JAX batch -- the bands as the
            backend finished them, the contact, the field's modes at ``K_f`` and its quadratic variation, the start
            pool from the batch (no second classification), the raw subset's block ends and the field's block means
            (the pilot's input, the basis's mean subtracted); nothing converted, a walk of saves refused by name."""
            from .backends import CoefficientBatch
            sel = slice(s, e) if e is not None else np.asarray(s)
            nb = int(r0_all[sel].shape[0])
            r0_b = np.ascontiguousarray(r0_all[sel], np.float32)
            keys_b = np.ascontiguousarray(np.asarray(keys_all[sel]), np.uint32)
            n_read_, dt_f_, _checks, keep_reads = rd if rd is not None else _rd[0]
            if sampling and raw and keep_reads:
                raise ValueError(f"backend {_backend_name!r}: the raw subset of a walk that reads the field keeps the reads at the "
                                 f"walk's read instants (the measured certificate's oracle); the backend returns block means")
            rq = req._replace(raw_subset=(nb if raw else 0), K_c3=int(K_f or 0),
                              field_read_interval=(float(dt_f_) if sampling else 0.0))
            batch = _backend.walk_batch(rq, r0_b, keys_b)
            if not isinstance(batch, CoefficientBatch):
                raise ValueError(f"backend {_backend_name!r} returned a {type(batch).__name__} for the adaptive coefficient walk; "
                                 f"the producer takes a CoefficientBatch (the bands accumulated in the walk) and converts "
                                 f"nothing: a walk of saves is refused (dmipy-sim#742)")
            if (int(batch.K), int(batch.m), int(batch.sub_steps)) != (K, m, int(n_min)):
                raise ValueError(f"backend {_backend_name!r} accumulated (K, m, sub_steps) = {(batch.K, batch.m, batch.sub_steps)} "
                                 f"for a request of {(K, m, int(n_min))}")
            C = np.asarray(batch.position_coeffs, np.float32)
            if C.shape != (nb, K + 2, 3):
                raise ValueError(f"backend {_backend_name!r} returned position coefficients of shape {C.shape}; "
                                 f"the batch of {nb} walkers at K = {K} has {(nb, K + 2, 3)}")
            if batch.boundary_endpoint is None or batch.boundary_bands is None or batch.compartment is None or batch.qv_boundary is None:
                raise ValueError(f"backend {_backend_name!r} returned no contact or compartment channel for the adaptive walk")
            comp_all[sel] = np.minimum(np.asarray(batch.compartment, np.int32), 1).astype(np.int8)
            out = dict(C=C[:, None], resid=(np.zeros((nb, 1, 3), np.float32) if batch.k0_residual is None
                                           else np.asarray(batch.k0_residual, np.float32)[:, None]),
                       blt_end=np.asarray(batch.boundary_endpoint, np.float32)[:, None],
                       blt_bands=np.asarray(batch.boundary_bands, np.float32)[:, None],
                       qv_pos=np.asarray(batch.qv_positions, np.float32)[:, None], qv_blt=np.asarray(batch.qv_boundary, np.float32)[:, None])
            if sampling:
                if batch.field_qv is None:
                    raise ValueError(f"backend {_backend_name!r} returned no field quadratic variation for a walk that reads the field")
                if batch.field_qv_increments is None:
                    raise ValueError(f"backend {_backend_name!r}: the backend does not return the field increments (the read "
                                     f"series' quadratic variation, CoefficientBatch.field_qv_increments), which the field read's "
                                     f"rule and term are measured on (RPK.md 9.5.5); a walk that reads the field is refused")
                out["qv_f"] = np.asarray(batch.field_qv, np.float32)[:, None]
                out["qv_d"] = np.asarray(batch.field_qv_increments, np.float32)[:, None]
                if K_f:
                    fb = np.asarray(batch.field_bands, np.float32) if batch.field_bands is not None else None
                    if fb is None or fb.shape != (nb, n_ch, int(K_f)):
                        raise ValueError(f"backend {_backend_name!r} returned field modes of shape {None if fb is None else fb.shape}; "
                                         f"the batch of {nb} walkers at K_c3 = {K_f} has {(nb, n_ch, int(K_f))}")
                    out["fc"] = np.transpose(fb, (0, 2, 1))[:, None]
            if raw:
                sv = batch.saves
                if sv is None or sv.positions is None:
                    raise ValueError(f"backend {_backend_name!r} returned no block ends for the raw subset of {nb} walkers")
                pos = np.asarray(sv.positions, np.float32)
                if pos.shape != (nb, n_t, 3):
                    raise ValueError(f"backend {_backend_name!r} returned block ends of shape {pos.shape} for {nb} walkers over {n_blocks} blocks")
                out["pos"] = pos
                out["dlog"] = (np.zeros((nb, n_t), np.float32) if sv.boundary_local_time is None else np.asarray(sv.boundary_local_time, np.float32))
                if sampling:
                    fs = sv.field_samples
                    if fs is None or np.asarray(fs).shape != (nb, n_t, n_ch):
                        raise ValueError(f"backend {_backend_name!r} returned field block means of shape "
                                         f"{None if fs is None else np.asarray(fs).shape} for the raw subset; {(nb, n_t, n_ch)} are needed")
                    out["fs"] = np.asarray(fs, np.float32) - np.asarray(field_basis.mean, np.float32)[None, None, :]
            return out, dict(n_free=None, n_kernel_steps=None, n_illegal=int(np.asarray(batch.illegal, np.int64).sum()))

        _batch = _walk_batch if _backend is None else _backend_batch
        _rd = [(1, float(dt_round), (), True)]                       # the read the batches take: the round's, then the walk's
        nonlocal_k = [k_cand]
        # ── the field's pilot: PILOT_WALKERS walkers spread over the walk reading every round -- the read series'
        # quadratic variation (the read's rule), the trapezoid's constant at every coarser read a block allows (what
        # the walk's read is charged at, given or derived), and the block means (the field band's ladder)
        field_band = None
        if sampling:
            import time as _time
            checks = read_checks(n_rounds) if _backend is None else ()
            run.phase("field pilot", walkers=n_pilot)
            _t0 = _time.time()
            got, _ = _batch(pidx, None, raw=K_c3 is None, K_f=None, rd=(1, float(dt_round), checks, False))
            B0, chi = field_rule_scale(_spec_holder(spec, geometry).spec, envelope)
            qd = np.asarray(got["qv_d"], np.float64)
            name_of = {int(p.id): p.name for p in ((spec or geometry.spec).pools if (spec or geometry.spec) is not None else [])}
            labels = np.asarray(comp_all[pidx], np.int64)
            pools = {name_of.get(int(l), str(int(l))): (qd[labels == l].mean(axis=(0, 1)), int((labels == l).sum()))
                     for l in np.unique(labels)}
            chk = got.get("chk")
            chk = None if chk is None else np.asarray(chk, np.float64)            # (n_pilot, n_win, 2, n_checks, n_ch)
            field_read = field_read_of(_spec_holder(spec, geometry), envelope, T_win, plan, given=field_read_interval,
                                       pools=pools, checks=checks, err_sq=(None if chk is None else (chk[:, :, 0] ** 2).mean(axis=(0, 1))),
                                       qv_checks=(None if chk is None else chk[:, :, 1].mean(axis=(0, 1))),
                                       qv_all=qd.mean(axis=(0, 1)), channels=list(field_basis.channel_names))
            field_read.update(pilot_walkers=int(n_pilot), pilot_seconds=float(_time.time() - _t0))
            if K_c3 is None:
                from ..replay.bank import derive_field_band
                K_f, bits, field_band = derive_field_band(np.asarray(got["fs"], np.float32)[:, 1:], list(field_basis.channel_names),
                                                           float(dt_actual), envelope, K_max=K, B0=B0, chi=chi,
                                                           floor_scale=float(np.sqrt(n_pilot / float(n_walkers))))
                field_band = dict(field_band, pilot_walkers=n_pilot, production_walkers=int(n_walkers), bits=int(bits))
            if K_c3 is not None:
                K_f = int(K_c3)
                if K_f < 1 or K_f > n_bw:
                    raise ValueError(f"K_c3 is the field's modes per window, 1..{n_bw} (the blocks of a window); got {K_f}")
            _rd[0] = (int(field_read["n_read"]), float(field_read["dt_f"]), (), True)     # the walk's read; the raw subset keeps the reads
        run.phase("walk")
        rows = {}
        n_free = n_kernel_steps = n_illegal = 0
        for bi, (s, e) in enumerate(run.batches(n_walkers, walker_batch_size, what="adaptive")):
            pieces = ([(s, min(e, n_raw), True)] if s < n_raw else []) + ([(max(s, n_raw), e, False)] if e > n_raw else [])
            for a, b_, raw in pieces:
                name = f"{spool}-batch-{bi:04d}-{'raw' if raw else 'acc'}"
                got = None
                if spool is not None:
                    found = run.spooled(name)
                    if found is not None:
                        got, hdr = found
                        n_free += int(hdr["n_free"]); n_kernel_steps += int(hdr["n_kernel_steps"])
                        log.info("  adaptive: batch %d read from the spool (%d walkers)", bi, b_ - a)
                if got is None:
                    got, cnt = _batch(a, b_, raw=raw, K_f=K_f)
                    k_cand = nonlocal_k[0]
                    n_illegal += cnt["n_illegal"]
                    if _backend is None:
                        n_free += cnt["n_free"]; n_kernel_steps += cnt["n_kernel_steps"]
                    if spool is not None and not cnt["n_illegal"]:
                        run.spool(name, got, dict(batch=bi, start=a, end=b_, raw=bool(raw), n_free=cnt["n_free"], n_kernel_steps=cnt["n_kernel_steps"]))
                for k, v in got.items():
                    rows.setdefault(k, []).append(np.asarray(v))
        if n_illegal:
            raise RuntimeError(f"adaptive walk: {n_illegal} walker(s) changed pool -- an illegal crossing; the walk is refused "
                               "(the pool compared is the end's against the start's: a walker that crossed a wall and left "
                               "again before the end is not counted here)")
        cat = lambda k: (np.concatenate(rows[k], axis=0) if k in rows else None)
        total_steps = n_walkers * n_blocks * n_min
        counted = _backend is None                                       # a backend's batch carries no stepping counts
        if counted:
            log.info("adaptive walk: %.1f%% of walker-rounds were free steps; kernel steps %.3g of the fused producer's %.3g "
                     "(%.1fx fewer)", 100.0 * n_free / max(n_walkers * n_blocks * n_rounds, 1), n_kernel_steps, total_steps,
                     total_steps / max(n_kernel_steps, 1))
        stepping = dict(rule="adaptive", candidate_cache=cached, candidate_k=int(k_cand),
                        free_fraction=((n_free / max(n_walkers * n_blocks * n_rounds, 1)) if counted else None),
                        kernel_steps=(int(n_kernel_steps) if counted else None), fused_steps=int(total_steps),
                        kernel_steps_ratio=((total_steps / max(n_kernel_steps, 1)) if counted else None), steps_per_round_by_class=steps_c,
                        radius_class_bounds_m=R_bounds.tolist(), far_at_m=far_at, safety_sigma=float(safety_sigma),
                        eps_D=float(plan.eps_D), f_band_hz=float(plan.f_band_hz), dt_band_s=float(plan.dt_band_s),
                        round_s=float(dt_round), rounds_per_block=int(n_rounds), backend=_backend_name)
        resid = cat("resid"); disp = cat("C")[:, :, 1, :]; qv_p = cat("qv_pos")
        block = dict(h=float(dt_actual), m=m, precision=PRECISION, n_blocks=int(n_blocks), rule=block_rule, ring=1,
                     accumulation="the moments of each block summed per step for the kernel-stepped walkers and per ROUND for the "
                                  "free-stepping ones (a round's displacement at its midpoint), every block projected in one float32 "
                                  "product and chained into a float32 level-2 accumulator",
                     round_s=float(dt_round),
                     k0_residual_m=float(np.abs(resid).max()), k0_residual_relative=float(np.abs(resid).max() / max(np.sqrt(np.mean(disp ** 2)), 1e-300)),
                     quadratic_variation=dict(positions_mean_m2=[float(x) for x in qv_p.sum(axis=1).mean(axis=0)],
                                              positions_over_2DT=[float(x) / (2.0 * D * float(T_max)) for x in qv_p.sum(axis=1).mean(axis=0)],
                                              boundary_mean_m2=float(cat("qv_blt").sum(axis=1).mean())))
        if sampling:
            block["field_read"] = dict(field_read)
            block["field_band"] = dict(field_band) if field_band is not None else dict(rule="given", K=int(K_f))
        work = dict(walker_steps=(int(n_kernel_steps) if counted else None), n_hits=None, n_crossings=None, exhausted_steps=None, backend=_backend_name,
                    bounce_budget=(None if geometry.bounce_loop is None else int(geometry.bounce_loop.budget)),
                    step_rule=dict(eps_D=float(plan.eps_D), f_band_hz=float(plan.f_band_hz), dt_p=float(dt_min), sub_steps=int(n_min)),
                    block=block)
        walk = CoefficientWalk(
            position_coeffs=cat("C"), T=float(T_max), K=K, m=m, block=block, n_windows=n_win, dt_sim=float(dt_min), sub_steps=int(n_min),
            boundary_endpoint=cat("blt_end"), boundary_bands=cat("blt_bands"), compartment=comp_all,
            field_basis=(field_basis if sampling else None), field_coeffs=cat("fc"), K_c3=(int(K_f) if sampling else None),
            field_read_interval=(float(field_read["dt_f"]) if sampling else None), qv_field=cat("qv_f"), qv_field_increments=cat("qv_d"),
            raw_positions=cat("pos"), raw_boundary=cat("dlog"), raw_compartment=(None if not n_raw else np.repeat(comp_all[:n_raw, None], n_t, axis=1)),
            raw_field=cat("fs"), qv_positions=qv_p, qv_boundary=cat("qv_blt"), illegal_crossings=0, seed=int(seed), diffusivity=D,
            geometry=geometry, spec=spec, stepping=stepping, work=work)
        object.__setattr__(walk, "run", run)
        return walk


def field_read_of(spec_holder, envelope, T_win, plan, *, given=None, pools=None, checks=None, err_sq=None, qv_checks=None,
                  qv_all=None, channels=None):
    """The field read of an adaptive walk of ``plan`` (its rounds per block, ``plan.dt_round``) in a window of ``T_win``
    seconds (:func:`~dmipy_sim.engine.accumulate.field_read_of_pilot` on rounds): from a pilot that read every round
    (``pools`` the read series' quadratic variation per pool, ``checks`` / ``err_sq`` / ``qv_checks`` its measurement
    of the trapezoid's constant, ``qv_all`` over every pool), at the envelope's ``B0`` and the spec's ``chi``, or the ``given`` interval; read every
    ``n_read`` rounds, never finer than a round. Without a pilot and without ``given`` the read is the round (the
    pilot's own). The record ``CoefficientWalk.block["field_read"]`` carries; a substrate kit's request takes its
    ``n_read`` with the interval the recipe gives."""
    from .accumulate import field_read_of_pilot, field_rule_scale, FIELD_READ_FLOOR
    B0, chi = field_rule_scale(spec_holder.spec, envelope)
    pilot = bool(pools)
    out = field_read_of_pilot(pools or {}, B0=B0, chi=chi, step_s=float(plan.dt_round), steps=int(plan.n_rounds), T_e=float(T_win),
                              floor=FIELD_READ_FLOOR, given=(given if (given is not None or pilot) else float(plan.dt_round)),
                              checks=checks, err_sq=err_sq, qv_checks=qv_checks, qv_all=qv_all, channels=channels)
    if given is None and not pilot:
        out["rule"] = "pilot"
    return out


def _spec_holder(spec, geometry):
    """What the field rule reads the nominal chi from: the spec when the walk is driven by one, else the geometry."""
    class _H:
        pass
    h = _H(); h.spec = spec if spec is not None else getattr(geometry, "spec", None)
    return h
