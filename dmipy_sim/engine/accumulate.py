"""The walk's band accumulation: what a walk (``simulate_trajectories``, a
:class:`~dmipy_sim.coefficient_walk.CoefficientWalk`) carries per walker.

A walk runs in blocks of ``h`` seconds, the band's own unless given (:func:`block_length`: ``omega_K h =``
:data:`BLOCK_OMEGA_H` at the top band, the table's order :data:`BLOCK_ORDER`). Inside a block the step scan adds, per
walker, the moments of each channel's increments about the block's centre in the block's own unit ``tau = (t - t_b) /
h`` (each step at its midpoint): ``M_j += tau^j dx`` for the positions, ``B_j += tau^j dlog`` for the local time and
``F_j += tau^j (b_i + b_{i+1}) dt_f / 2`` for the field (:func:`moment_scan`), the field read at the kernel's position
every ``n_read`` steps (:func:`field_read_steps`) and each interval's increment the TRAPEZOID of its two reads. At the
end of every :data:`RING` blocks the ring's moments are projected onto the ``K + 1`` cosine modes (``K_c3 + 1`` for the
field) as one product against the ring's rows of :func:`~dmipy_sim.replay.compression.block_table` and added to the
walker's level-2 accumulator (:func:`ring_scan`): the sum over a ring runs inside one float32 product of ``RING (m +
1)`` terms and the chain over the walk has ``n_blocks / RING`` links. A walk whose walkers can change pool keeps, per
walker, the EVENTS ``(step, label)`` of every change in a buffer of declared capacity (:func:`events_init`, exact in
the walk's own time; overflow is counted and refused by the producer), the start label held beside it. Nothing is
emitted per block but the raw block ends of a declared subset, so a walker's working set is the accumulators, the
running position, the last field read and the events buffer, whatever ``h`` and the walk's length.

The field read (RPK.md 9.5.5): between two reads the walker moves ``sqrt(2 D dt_f)`` and the trapezoid of the two
reads misses the path's deviation from the chord; over a walk that adds to the field phase an independent term of
variance ``v_read = (gamma B0 chi)^2 dt_f^2 (sum_c sqrt(kappa_c QV_c))^2``, ``QV_c = sum_i (b_{i+1} - b_i)^2`` the quadratic
variation of channel ``c``'s read series (:func:`moment_scan` accumulates it beside the trapezoid, one multiply-add
per read) and ``kappa_c`` the trapezoid's constant, :data:`READ_CONSTANT` ``= 1/12`` for a field linear over one read
or the larger constant a pilot measured on that channel (:func:`read_constants`). The certificate's term is
``field_read = v_read / 2`` (:func:`field_read_term`). A walk's read is derived from a PILOT's ``QV`` per pool
(:func:`field_read_of_pilot`): the coarsest interval holding the TERM at or below :data:`FIELD_READ_FLOOR` at the
envelope's ``B0`` and ``chi`` (:func:`field_read_rule`; the criterion of every other term), the finest over the walk's
pools, read every ``n`` steps (:func:`field_read_steps`; never finer than the step, never coarser than the block) and
at every block end, a block's last interval the shorter one, so that the block's moments are exact at its ends; a
read the floor would want finer than the step is the step, and the envelope states the ``B0 chi`` product under which
the term stays below the floor (:func:`field_read_limit_B0_chi`, RPK.md 9.4 rule 5): the read's own limit, which is not the
envelope's certified ``B0 chi``.
"""
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..constants import GAMMA

#: blocks per ring: the projection's inner sum and the stride of the level-2 chain
RING = 16

#: the block table's order a walk takes by default (dmipy-sim-cuda#74)
BLOCK_ORDER = 3

#: ``omega_K h`` at the top band of the block a walk takes by default: with ``m = 3`` the quadrature adds
#: ``quadrature_rel(3, h, omega_K) = 0.8^8 / 1327104 = 1.3e-7`` of the top band's phase variance, for every class
#: (measured 1.41e-7 on a T4, dmipy-sim-cuda#74)
BLOCK_OMEGA_H = 0.8


#: the storage band a walk accumulates by default (Hz): K = 256 bands per 100 ms (#740)
STORAGE_BAND_HZ = 1280.0

#: the floor the field read's rule holds its term to (the signal floor a storage-grade pack is ordered at,
#: ``build_to_floor``'s ``sigma_star``): ``v_read / 2 <= FIELD_READ_FLOOR`` at the envelope's ``B0`` and ``chi``
FIELD_READ_FLOOR = 1e-3

#: the library's default field strength (T), the largest of :func:`~dmipy_sim.replay.compression.default_envelope`'s
#: ``B0_list``, and susceptibility scale (myelin's isotropic chi) for a substrate whose spec states no magnetic pool:
#: what :func:`~dmipy_sim.replay.compression.certified_envelope` writes into an envelope that states none
FIELD_RULE_B0 = 7.0
FIELD_RULE_CHI = 1.06e-6

#: walkers of the pilots a walk runs before itself: the field band's ladder (:func:`~dmipy_sim.replay.bank.derive_field_band`)
#: and the events buffer's capacity (the largest count a pilot walker reached, with :data:`EVENTS_MARGIN`)
PILOT_WALKERS = 1024
EVENTS_MARGIN = (1.5, 8)      # capacity = ceil(1.5 x the pilot's largest count) + 8


def spec_chi_scale(spec):
    """The susceptibility scale of a substrate: the largest ``|chi_iso| + |chi_aniso|`` over the spec's magnetic
    pools, :data:`FIELD_RULE_CHI` when it states none (the default ``chi_max`` an envelope is certified at)."""
    vals = []
    for p in (getattr(spec, "field_source_pools", None) or []) if spec is not None else []:
        s = p.susceptibility
        vals.append(abs(float(s.chi_iso or 0.0)) + abs(float(s.chi_aniso or 0.0)))
    return float(max(vals)) if vals and max(vals) > 0.0 else float(FIELD_RULE_CHI)


def field_rule_scale(spec, env=None):
    """``(B0, chi)`` the field read's rule, the field band's pilot and the field's certificate terms are derived at:
    the certified envelope's largest field strength and its ``chi_max``
    (:func:`~dmipy_sim.replay.compression.certified_envelope` of ``env`` on ``spec``), whose product is the one
    certified field scale."""
    from ..replay.compression import certified_envelope, envelope_B0_max
    e = certified_envelope(env, spec)
    return envelope_B0_max(e), float(e["chi_max"])


def pilot_walkers(n_walkers, n_pilot=None):
    """The walkers a pilot walks: :data:`PILOT_WALKERS` (or ``n_pilot``) indices spread evenly over the walk's
    ``n_walkers`` -- a producer seeds its walkers pool by pool, so the first ones are one pool, and a pilot of them
    would measure that pool alone."""
    n_walkers = int(n_walkers)
    n_pilot = min(int(PILOT_WALKERS if n_pilot is None else n_pilot), n_walkers)
    return np.unique(np.round(np.linspace(0, n_walkers - 1, n_pilot)).astype(np.int64))


def default_K(T):
    """The bands of a window of ``T`` seconds at :data:`STORAGE_BAND_HZ`: ``ceil(2 f T)``."""
    return max(1, int(np.ceil(2.0 * STORAGE_BAND_HZ * float(T) - 1e-9)))


def window_count(T_max, window_T=None):
    """``(n_windows, T_window)``: a walk within one storage window (``window_T``, the storage rule's
    :data:`~dmipy_sim.replay.bank.WINDOW_T` by default) is one window; a longer one is a whole number of windows,
    each accumulated on its own (RPK.md 4.3), and is refused otherwise."""
    if window_T is None:
        from ..replay.bank import WINDOW_T as window_T
    T_max, window_T = float(T_max), float(window_T)
    if not window_T > 0.0:
        raise ValueError(f"window_T is the duration of a storage window, a positive number; got {window_T!r}")
    if T_max <= window_T * (1.0 + 1e-9):
        return 1, T_max
    n = int(round(T_max / window_T))
    if abs(n * window_T - T_max) > 1e-9 * T_max:
        raise ValueError(f"a walk of {T_max:.6g} s longer than one storage window is a whole number of {window_T:.6g} s "
                         "windows; walk a whole number of them, or pass the window_T that divides it")
    return n, window_T


def block_length(T, K):
    """The block of a walk of ``K`` bands over a window of ``T`` seconds when none is given: ``h = BLOCK_OMEGA_H T /
    (pi K)``, so that ``omega_K h = 0.8`` at the top band ``omega_K = pi K / T``, rounded to a whole number of blocks
    over the walk (``T / round(T / h)``) -- 100 us at K = 256 over 100 ms. The band sets it, never the walker count or
    the step: the quadrature it costs is the same fraction of the top band for every scanner class."""
    h = float(BLOCK_OMEGA_H) * float(T) / (np.pi * int(K))
    return float(T) / max(int(round(float(T) / h)), 1)


#: the floating-point type the block contributions are accumulated in across blocks (``walk_params.block.precision``,
#: RPK.md 9.5.5), and how (``walk_params.block.accumulation``)
PRECISION = "float32"
ACCUMULATION = (f"the moments of each block summed per step, a ring of {RING} blocks projected in one float32 product, "
                "the rings chained into a float32 level-2 accumulator")


# ------------------------------------------------------------------------------------------- the field read's rule
#: the trapezoid's constant: one read interval's deviation from its chord along a diffusing path, for a field linear
#: over the interval, has variance ``dt_f^2 (b_{i+1} - b_i)^2 / 12`` in expectation (the integral of a Brownian
#: bridge); the chord's curvature across a sheath raises it, which a pilot measures per channel (:func:`read_constants`)
READ_CONSTANT = 1.0 / 12.0


def read_constants(measured, n_ch):
    """The trapezoid's constant per channel, ``(n_ch,)``: :data:`READ_CONSTANT`, or the larger constant a pilot
    measured on the channel (``measured`` ``(n_ch,)`` or None)."""
    out = np.full(int(n_ch), READ_CONSTANT)
    if measured is not None:
        out = np.maximum(out, np.nan_to_num(np.asarray(measured, np.float64), nan=0.0))
    return out


def _read_weight(qv_increments, constants):
    """``(sum_c sqrt(kappa_c QV_c))^2`` over the channels: the read error's variance per unit ``(gamma B0 chi dt_f)^2``.
    The channels are linear functionals of one path, so their read errors are correlated, and a contraction at any
    orientation weighs them by at most one: the bound is the square of the summed standard deviations."""
    qv = np.asarray(qv_increments, np.float64).reshape(-1)
    kap = read_constants(constants, qv.shape[0])
    return float(np.sum(np.sqrt(kap * np.clip(qv, 0.0, None)))) ** 2


def field_read_term(dt_f, qv_increments, B0, chi, constants=None):
    """``field_read = v_read / 2`` (RPK.md 9.5.5): ``v_read = (gamma B0 chi)^2 dt_f^2 (sum_c sqrt(kappa_c QV_c))^2``
    with ``qv_increments`` the read series' quadratic variation per channel (the walkers' mean) and ``kappa_c`` the
    trapezoid's constant per channel (:func:`read_constants` of ``constants``; ``1/12`` gives ``dt_f^2 / 12 (sum_c
    sqrt(QV_c))^2``): the channels folded as the square of their summed roots, the bound over every orientation of
    one path's correlated components, and the gate bounded by ``|s| <= 1``."""
    return 0.5 * (GAMMA * float(B0) * float(chi) * float(dt_f)) ** 2 * _read_weight(qv_increments, constants)


def field_read_rule(qv_increments, B0, chi, floor=FIELD_READ_FLOOR, constants=None):
    """The read interval ``dt_f`` (s) that holds the term ``v_read / 2`` at ``floor`` for a walk whose read series
    has the quadratic variation ``qv_increments`` per channel (RPK.md 9.5.5): ``sqrt(2 floor) / (gamma B0 chi sum_c
    sqrt(kappa_c QV_c))``; ``inf`` for a series that does not vary (a stuck pool, a uniform field)."""
    denom = GAMMA * float(B0) * float(chi) * math.sqrt(_read_weight(qv_increments, constants))
    if not denom > 0.0:
        return math.inf
    return math.sqrt(2.0 * float(floor)) / denom


def field_read_limit_B0_chi(dt_f, qv_increments, floor=FIELD_READ_FLOOR, constants=None):
    """The product ``B0 chi`` (T) under which the term at ``dt_f`` stays below ``floor``: what a pack whose read is
    the step states in its envelope (RPK.md 9.4 rule 5)."""
    base = (float(dt_f)) ** 2 * _read_weight(qv_increments, constants)
    if not base > 0.0:
        return math.inf
    return math.sqrt(2.0 * float(floor) / base) / GAMMA


def field_read_steps(dt_f_rule, dt_sim, sub_steps):
    """``(n_read, dt_f)``: the read every ``n_read`` steps (and at every block end), the most steps, at most the
    block's ``sub_steps``, whose interval ``n_read dt_sim`` is within the rule's ``dt_f_rule`` -- the step itself
    when the rule asks for less than a step (a read along the chord between two steps carries the same bridge
    deviation: the read is never sub-stepped), the block when the rule allows more than one. ``dt_f`` is the longest
    interval, the one the term is charged at; a block's last interval, ``sub_steps mod n_read`` steps, is shorter."""
    sub_steps = int(sub_steps)
    ratio = float(dt_f_rule) / float(dt_sim) * (1.0 + 1e-9)
    n = sub_steps if not math.isfinite(ratio) else int(min(max(math.floor(ratio), 1), sub_steps))
    return n, n * float(dt_sim)


def read_intervals(steps, n_read):
    """The read intervals of a block of ``steps`` steps read every ``n_read`` (and at its end), in steps: ``n_read``
    each, the last ``steps mod n_read`` when it does not divide."""
    steps, n_read = int(steps), int(n_read)
    q, r = divmod(steps, n_read)
    return (n_read,) * q + ((r,) if r else ())


#: the most coarser reads a pilot measures the trapezoid's constant at: a geometric ladder over the block's steps
READ_CHECK_RATIO = 1.5


def read_checks(steps):
    """The read intervals, in steps, at which a pilot that reads every step measures the trapezoid's constant: a
    ladder ``2, 3, 4, 7, 10, 15, 23, ...`` (ratio :data:`READ_CHECK_RATIO`, rounded) up to the block's ``steps``, the block itself the
    last rung; a walk's read is charged at the rung at or above it (:func:`field_read_of_pilot`)."""
    steps = int(steps)
    out, d = [], 2.0
    while int(round(d)) < steps:
        if not out or int(round(d)) > out[-1]:
            out.append(int(round(d)))
        d *= READ_CHECK_RATIO
    if steps >= 2:
        out.append(steps)
    return tuple(out)



def field_read_statistics(reads, *, step_s, rounds_per_block, blocks_per_window, checks):
    """Read-series calibration, shared by native and JAX pilots.

    Reads are mean-subtracted field channels at every round, including the initial value. Each block's
    last interval is kept at its actual length. Error integrals sum over a window before squaring, whereas
    increment variations sum squared increments. Only one block is promoted to float64 at a time.
    """
    reads = np.asarray(reads)
    rounds, blocks = int(rounds_per_block), int(blocks_per_window)
    checks = tuple(int(d) for d in checks)
    if reads.ndim != 3 or rounds < 1 or blocks < 1 or not np.isfinite(step_s) or step_s <= 0:
        raise ValueError("field reads need (walkers, rounds + 1, channels), positive step and block dimensions")
    n_steps = reads.shape[1] - 1
    if n_steps < 1 or n_steps % (rounds * blocks) or any(not 2 <= d <= rounds for d in checks):
        raise ValueError("the read series must cover whole windows and checks within each block")
    n, _, channels = reads.shape
    windows = n_steps // (rounds * blocks)
    means = np.empty((n, windows * blocks, channels), np.float32)
    qv = np.zeros((n, windows, channels), np.float64)
    qd = np.zeros_like(qv)
    measured = np.zeros((n, windows, 2, len(checks), channels), np.float64)
    for block in range(windows * blocks):
        values = np.asarray(reads[:, block * rounds:(block + 1) * rounds + 1], np.float64)
        if not np.isfinite(values).all():
            raise ValueError("field calibration reads must be finite")
        window = block // blocks
        increments = 0.5 * (values[:, :-1] + values[:, 1:]) * float(step_s)
        fine = increments.sum(axis=1)
        means[:, block] = fine / (rounds * float(step_s))
        qv[:, window] += np.sum(increments ** 2, axis=1)
        qd[:, window] += np.sum(np.diff(values, axis=1) ** 2, axis=1)
        for i, d in enumerate(checks):
            spans = np.asarray(read_intervals(rounds, d), np.int64)
            ends = np.cumsum(spans); starts = ends - spans
            coarse = (0.5 * (values[:, starts] + values[:, ends]) *
                      (spans * float(step_s))[None, :, None]).sum(axis=1)
            difference = np.where(np.all(values == values[:, :1], axis=1), 0.0, coarse - fine)
            measured[:, window, 0, i] += difference
            measured[:, window, 1, i] += np.sum(
                ((values[:, ends] - values[:, starts]) * (spans / float(d))[None, :, None]) ** 2, axis=1)
    return dict(means=means, qv_field=qv, qv_increments=qd, checks=measured)


def calibrate_field_reads(reads, *, labels, pool_names, step_s, rounds_per_block, blocks_per_window,
                          B0, chi, T_e, floor, confidence=.99, given=None):
    """Choose a measured read interval, with simultaneous jackknife-ratio uncertainty.

    The empirical comparison is against the every-round reference, not the continuous path. Its residual error
    must still be covered by the scoped final-answer budget. No constant is borrowed from a different read rung.
    """
    from statistics import NormalDist
    rounds = int(rounds_per_block)
    confidence = float(confidence)
    if not 0 < confidence < 1:
        raise ValueError("field-read confidence is in (0, 1)")
    candidates = set(read_checks(rounds))
    if given is not None:
        candidates.add(field_read_steps(float(given), float(step_s), rounds)[0])
    checks = tuple(sorted(d for d in candidates if d > 1))
    statistics = field_read_statistics(reads, step_s=step_s, rounds_per_block=rounds,
                                       blocks_per_window=blocks_per_window, checks=checks)
    labels = np.asarray(labels)
    if labels.shape != (len(reads),):
        raise ValueError("one pool label is required per calibration walker")
    pools = {int(label):np.flatnonzero(labels == label) for label in np.unique(labels)}
    channels = int(np.asarray(reads).shape[-1])
    comparisons = max(1, (rounds - 1) * channels * len(pools))
    z = NormalDist().inv_cdf(1 - (1 - confidence) / comparisons)
    bounds, qv_at, details = {}, {}, {}
    resolved = all(len(indices) >= 2 for indices in pools.values())
    for i, d in enumerate(checks):
        upper = np.full(channels, READ_CONSTANT)
        qv_pools = {}
        for label, indices in pools.items():
            # Windows of one walker are not independent Monte Carlo replicates.
            error = np.mean(statistics["checks"][indices, :, 0, i] ** 2, axis=1)
            qv = np.mean(statistics["checks"][indices, :, 1, i], axis=1)
            n = len(indices); sx = error.sum(axis=0); sq = qv.sum(axis=0)
            scale = (d * float(step_s)) ** 2
            point = np.divide(sx, scale * sq, out=np.zeros_like(sx), where=sq > 0)
            alias = (sq == 0) & (sx > 0)
            if n > 1:
                denominator = scale * (sq - qv)
                leave_one = np.divide(sx - error, denominator, out=np.zeros_like(error), where=denominator > 0)
                invalid = np.any((denominator <= 0) & (sx - error > 0), axis=0)
                se = np.sqrt((n - 1) / n * np.sum((leave_one - leave_one.mean(axis=0)) ** 2, axis=0))
                value = point + z * se
                value[alias | invalid] = np.inf
            else:
                value = np.full(channels, np.inf)
            upper = np.maximum(upper, value)
            qv_pools[label] = qv.mean(axis=0)
        bounds[d] = upper
        qv_at[d] = qv_pools
        details[str(d)] = [float(v) if np.isfinite(v) else None for v in upper]
    finite = [value for value in bounds.values() if np.isfinite(value).all()]
    bounds[1] = np.maximum.reduce(finite) if finite else np.full(channels, READ_CONSTANT)
    qv_at[1] = {label:statistics["qv_increments"][indices].mean(axis=(0,1)) for label,indices in pools.items()}
    terms = {d:max(field_read_term(d*step_s, qv_at[d][label], B0, chi, constants) for label in pools)
             for d,constants in bounds.items() if np.isfinite(constants).all()}
    if given is not None:
        selected = field_read_steps(float(given), float(step_s), rounds)[0]
        if selected not in terms:
            raise ValueError(f"the given field read has unresolved or aliased reference variation at rung {selected} "
                             f"with pool counts {[len(v) for v in pools.values()]}; choose a finer interval or resolve its pilot")
    else:
        feasible = [d for d,term in terms.items() if term <= floor]
        selected = max(feasible) if feasible else 1
    fine_pools = {pool_names.get(label,str(label)):(qv_at[1][label], len(indices)) for label,indices in pools.items()}
    output = field_read_of_pilot(fine_pools, B0=B0, chi=chi, step_s=step_s, steps=rounds, T_e=T_e,
                                 floor=floor, given=given if given is not None else selected*step_s)
    output.update(rule="given" if given is not None else "pilot", constants=bounds[selected].tolist(),
                  constants_measured=bool(checks), constants_measured_at_read=bool(selected > 1),
                  measured_constants=details, term_pilot=float(terms[selected]),
                  budget_met=bool(terms[selected] <= floor), calibration_resolved=bool(resolved),
                  constant_evidence=dict(method="simultaneous_one_sided_normal_jackknife_ratio",
                                         reference="every-round trapezoids", confidence=confidence, z=z,
                                         comparisons=comparisons, selected_rung=int(selected),
                                         finest_read=(selected == 1)))
    limit = min(field_read_limit_B0_chi(selected*step_s, value, floor, bounds[selected])
                for value in qv_at[selected].values())
    output["read_limit_B0_chi_T_pilot"] = float(limit) if np.isfinite(limit) else None
    output["candidate_terms"] = {str(d):float(terms[d]) if d in terms else None for d in sorted(bounds)}
    for label, indices in pools.items():
        own = output["pools"][pool_names.get(label, str(label))]
        own.update(constants=bounds[selected].tolist(),
                   selected_qv_increments=qv_at[selected][label].tolist(),
                   term_pilot=float(field_read_term(selected*step_s, qv_at[selected][label], B0, chi, bounds[selected])),
                   dt_f_rule=float(field_read_rule(qv_at[1][label], B0, chi, floor, bounds[selected])))
    if given is None and not output["budget_met"]:
        output["dt_f_rule"] = min(field_read_rule(value, B0, chi, floor, bounds[1]) for value in qv_at[1].values())
        output["at_step"] = bool(output["dt_f_rule"] < step_s)
    return output, statistics


def measured_read_constants(err_sq, qv_checks, step_s, checks):
    """The trapezoid's constant per check and channel measured on a pilot that read every step: ``err_sq`` ``(n_checks,
    n_ch)`` the walkers' mean of ``E_d^2``, ``E_d`` the integral of the read series by the trapezoid of every ``d``-th
    read less its integral by the trapezoid of every read, over the window; ``qv_checks`` ``(n_checks, n_ch)`` the
    walkers' mean quadratic variation of the series of every ``d``-th read -- the series a walk reading every ``d``
    steps measures and is charged on. ``kappa_c(d) = <E_d^2> / ((d dt)^2 <QV_c(d)>)``, :data:`READ_CONSTANT` for a
    field linear over ``d`` steps; ``{d: (n_ch,)}``, NaN on a channel that does not vary."""
    err_sq = np.asarray(err_sq, np.float64); qv = np.asarray(qv_checks, np.float64)
    out = {}
    for i, d in enumerate(checks):
        with np.errstate(divide="ignore", invalid="ignore"):
            k = err_sq[i] / ((float(d) * float(step_s)) ** 2 * qv[i])
        out[int(d)] = np.where(qv[i] > 0.0, k, np.nan)
    return out


def field_read_of_pilot(pools, *, B0, chi, step_s, steps, T_e, floor=FIELD_READ_FLOOR, given=None, checks=None,
                        err_sq=None, qv_checks=None, qv_all=None, channels=None):
    """The field read of a walk of ``steps`` physics steps of ``step_s`` seconds per block, in a window of ``T_e``
    seconds, from its pilot (RPK.md 9.5.5). ``pools`` is ``{name: (qv_increments (n_ch,), walkers)}``, the pilot's
    walkers' mean quadratic variation of the read series per pool, read at every step; per pool the rule's interval
    (:func:`field_read_rule` at ``B0`` and ``chi``), the walk's read the finest over its pools (one interval serves
    every walker of a walk), every ``n`` steps and at every block end (:func:`field_read_steps`).
    ``given`` (s) replaces the rule's interval. ``checks`` / ``err_sq`` / ``qv_checks`` are the pilot's measurement of
    the trapezoid's constant (:func:`read_checks`, the walkers' mean ``E_d^2`` and quadratic variation of every
    ``d``-th read per check and channel, :func:`measured_read_constants`), ``qv_all`` the walkers' mean quadratic
    variation of the per-step reads over every pool: the constant charged on a channel is the larger of
    :data:`READ_CONSTANT` and the one measured at the walk's read (at the finest check when the read is the step),
    and a measured constant above it moves the read again. Returns the record a walk carries in
    ``CoefficientWalk.block["field_read"]``."""
    step_s, steps = float(step_s), int(steps)
    measured = None
    if checks and err_sq is not None and qv_checks is not None:
        measured = measured_read_constants(err_sq, qv_checks, step_s, checks)
    n_ch = None
    for qv, _n in pools.values():
        n_ch = int(np.asarray(qv).reshape(-1).shape[0])

    def at_read(n):                                    # the constants measured at the rung at or above a read of n steps
        if measured is None:
            return None
        up = [d for d in measured if d >= n]
        return read_constants(measured[min(up) if up else max(measured)], n_ch)

    constants, seen = None, set()
    while True:
        rules = {name: field_read_rule(qv, B0, chi, floor, constants) for name, (qv, _n) in pools.items()}
        dt_rule = float(given) if given is not None else (min(rules.values()) if rules else math.inf)
        n_read, dt_f = field_read_steps(dt_rule, step_s, steps)
        new = at_read(n_read)
        if new is None or given is not None or n_read in seen or (constants is not None and np.array_equal(new, constants)):
            constants = new if new is not None else constants
            break
        seen.add(n_read)
        constants = new
    if constants is None and n_ch is not None:
        constants = read_constants(None, n_ch)
    weight = None
    if qv_all is not None:
        weight = np.asarray(qv_all, np.float64)
    finite = lambda x: float(x) if math.isfinite(float(x)) else None          # a series that does not vary sets no interval
    out = dict(rule=("given" if given is not None else "pilot"), dt_f_rule=finite(dt_rule), dt_f=float(dt_f), n_read=int(n_read),
               reads_per_block=len(read_intervals(steps, n_read)), at_step=bool(n_read == 1 and dt_rule < step_s), B0=float(B0), chi=float(chi),
               floor=float(floor), T_e=float(T_e), step_s=step_s, steps_per_block=int(steps), constant=READ_CONSTANT,
               constants=(None if constants is None else [float(x) for x in constants]),
               constants_measured=bool(measured is not None),
               pools={str(name): dict(walkers=int(n), qv_increments=[float(x) for x in np.asarray(qv, np.float64).reshape(-1)],
                                      dt_f_rule=finite(rules[name]))
                      for name, (qv, n) in pools.items()})
    if channels is not None:
        out["channels"] = [str(c) for c in channels]
    if measured is not None:
        out["measured_constants"] = {str(d): [None if not np.isfinite(x) else float(x) for x in k] for d, k in measured.items()}
    if weight is not None:
        out["term_pilot"] = float(field_read_term(dt_f, weight, B0, chi, constants))
        out["read_limit_B0_chi_T_pilot"] = finite(field_read_limit_B0_chi(dt_f, weight, floor, constants))
    return out


# ------------------------------------------------------------------------------------------------------ the kinds
class Kind(NamedTuple):
    """What the accumulation reads off one physics step: the step itself and the channels in its carry."""
    inner: object             #: ``(carry, None) -> (carry, None)``, one physics step
    pos_of: object            #: the recorded (continuous) position ``(3,)`` in the carry
    read_pos_of: object       #: the position the field is read at (the in-cell one of a periodic kernel)
    blt_of: object = None     #: the cumulative local time in the carry, or None
    label_of: object = None   #: the pool label (int8) in the carry, or None
    bound_of: object = None   #: the bound state (int8, 0 free / 1 bound) in the carry, or None


class Events(NamedTuple):
    """One column's events buffer in the carry: the changes of a label, exact at the step they happen."""
    n: object                 #: int32: changes so far (counts past the capacity too)
    steps: object             #: (cap,) int32: the global step index the new label holds from
    labels: object            #: (cap,) int8: the label entered
    prev: object              #: int8: the label at the last step


def events_init(label0, cap):
    """The empty buffer of a column that starts at ``label0`` with ``cap`` entries (0: count only)."""
    return Events(jnp.int32(0), jnp.zeros((int(cap),), jnp.int32), jnp.zeros((int(cap),), jnp.int8), jnp.asarray(label0).astype(jnp.int8))


def _events_step(ev, lab, gstep):
    """The buffer after a step that ended on ``lab`` at global step ``gstep``: one entry when the label changed."""
    lab = jnp.asarray(lab).astype(jnp.int8)
    changed = lab != ev.prev
    n = ev.n + changed.astype(jnp.int32)
    if ev.steps.shape[0] == 0:
        return Events(n, ev.steps, ev.labels, lab)
    idx = jnp.minimum(ev.n, ev.steps.shape[0] - 1)
    steps = ev.steps.at[idx].set(jnp.where(changed, gstep.astype(jnp.int32), ev.steps[idx]))
    labels = ev.labels.at[idx].set(jnp.where(changed, lab, ev.labels[idx]))
    return Events(n, steps, labels, lab)


# ---------------------------------------------------------------------------------------------------- the block
def moment_scan(kind, carry, length, m, *, read=None, n_ch=0, n_read=1, dt_f=0.0, step0=None, checks=(), keep_reads=False):
    """One block of ``length`` steps of ``kind`` from ``carry = (phys, b_prev, ev_comp, ev_bound)``: the physics carry
    stepped ``length`` times, with the moments of the increments ``M_j = sum_i tau_i^j (pos(c_i+1) - pos(c_i))`` (m+1,
    3) and the same ``B_j`` of the cumulative local time, ``tau_i = (i + 1/2) / length - 1/2``; each channel's
    quadratic variation ``Q = sum_i (dx_i)^2`` per axis and ``Qb = sum_i (dlog_i)^2``; with ``read`` the field read
    every ``n_read`` steps and at the block's end (:func:`read_intervals`: the last interval the shorter one when
    ``n_read`` does not divide ``length``; ``dt_f = n_read dt`` the full interval) and the TRAPEZOID increment of each
    read interval, ``(b_prev + b) tau_i / 2`` over its own length ``tau_i``, at the interval's midpoint: ``F_j`` (m+1,
    n_ch), ``Qf = sum (inc)^2`` (n_ch,) and the read series' quadratic variation ``Qd = sum ((b - b_prev) tau_i /
    dt_f)^2`` (n_ch,; a shorter interval length-weighted, so that ``dt_f^2 / 12 Qd`` is the sum of every interval's own
    bridge variance, RPK.md 9.5.5), the last read carried in ``b_prev``; with ``checks`` (a walk that reads every step) ``E`` (2, n_checks,
    n_ch): per check ``d`` the trapezoid of every ``d``-th read and the block's end less the trapezoid of every read,
    over the block, and the quadratic variation of the series of those reads (the pilot's measurement of the
    trapezoid's constant, :func:`measured_read_constants`); with ``keep_reads`` the reads themselves (n_reads, n_ch)
    at the ends of the block's read intervals; with an events buffer the label after every step recorded on change at
    the global step ``step0 + i + 1``. Returns ``(carry, (M, B, Q, Qb, F, Qf, Qd, E, reads))`` (None where a channel
    is not recorded)."""
    phys, b_prev, ev_c, ev_b = carry
    length, n_read, m = int(length), int(n_read), int(m)
    if not 1 <= n_read <= length:
        raise ValueError(f"the field is read every {n_read} steps of a block of {length}: between 1 and the block")
    checks = tuple(int(d) for d in checks)
    if checks and (n_read != 1 or any(not 2 <= d <= length for d in checks)):
        raise ValueError(f"the trapezoid's checks {checks} need a read at every step and reads of 2 .. {length} steps")
    n_full, rem = divmod(length, n_read)
    has_b = kind.blt_of is not None
    js = jnp.arange(m + 1, dtype=jnp.float32)
    inv_len = jnp.float32(1.0 / length)
    dt_step = float(dt_f) / n_read
    s0 = jnp.int32(0) if step0 is None else jnp.asarray(step0).astype(jnp.int32)
    ds = jnp.asarray(checks, jnp.int32) if checks else None
    half_dt = jnp.float32(0.5 * dt_step)

    def step(c, i):
        cc, mp, mb, qp, qb, evc, evb = c
        r0 = kind.pos_of(cc)
        b0 = kind.blt_of(cc) if has_b else None
        cc, _ = kind.inner(cc, None)
        tau = (i.astype(jnp.float32) + 0.5) * inv_len - 0.5
        pw = tau ** js
        dx = kind.pos_of(cc) - r0
        mp = mp + pw[:, None] * dx[None, :]
        qp = qp + dx * dx
        if has_b:
            db = kind.blt_of(cc) - b0
            mb = mb + pw * db
            qb = qb + db * db
        if evc is not None:
            evc = _events_step(evc, kind.label_of(cc), s0 + i + 1)
        if evb is not None:
            evb = _events_step(evb, kind.bound_of(cc), s0 + i + 1)
        return (cc, mp, mb, qp, qb, evc, evb), None

    def group_of(L):
        """A read interval of ``L`` steps starting at step ``start`` of the block (``L`` static)."""
        half = jnp.float32(0.5 * L * dt_step)
        wq = jnp.float32(L / n_read)                       # a shorter interval enters the QV as (db dt_i / dt_f)^2

        def group(c, start):
            cc, mp, mb, qp, qb, evc, evb, bp, mf, qf, qd, bpd, ec = c
            idx = start + jnp.arange(L, dtype=jnp.int32)
            (cc, mp, mb, qp, qb, evc, evb), _ = jax.lax.scan(step, (cc, mp, mb, qp, qb, evc, evb), idx)
            b = None
            if read is not None:
                b = read(kind.read_pos_of(cc))
                inc = (bp + b) * half                                            # the trapezoid over the read interval
                tau = (start.astype(jnp.float32) + 0.5 * L) * inv_len - 0.5        # the interval's midpoint
                mf = mf + (tau ** js)[:, None] * inc[None, :]
                qf = qf + inc * inc
                db = (b - bp) * wq
                qd = qd + db * db                                                # the read series' quadratic variation
                if checks:                                                       # the d-th reads and the block's end
                    k = start + 1
                    hit = (k % ds == 0) | (k == length)
                    span = jnp.where(k % ds == 0, ds, k % ds).astype(jnp.float32)[:, None]
                    e_d = ec[0] - inc[None, :] + jnp.where(hit[:, None], (bpd + b[None, :]) * span * half_dt, 0.0)
                    q_d = ec[1] + jnp.where(hit[:, None], ((b[None, :] - bpd) * span / ds.astype(jnp.float32)[:, None]) ** 2, 0.0)
                    ec = jnp.stack([e_d, q_d])
                    bpd = jnp.where(hit[:, None], b[None, :], bpd)
                bp = b
            return (cc, mp, mb, qp, qb, evc, evb, bp, mf, qf, qd, bpd, ec), (b if keep_reads else None)
        return group

    m0 = jnp.zeros((m + 1, 3), jnp.float32)
    b0 = jnp.zeros((m + 1,), jnp.float32) if has_b else None
    q0 = jnp.zeros((3,), jnp.float32)
    qb0 = jnp.float32(0.0) if has_b else None
    f0 = jnp.zeros((m + 1, int(n_ch)), jnp.float32) if read is not None else None
    qf0 = jnp.zeros((int(n_ch),), jnp.float32) if read is not None else None
    qd0 = jnp.zeros((int(n_ch),), jnp.float32) if read is not None else None
    bpd0 = jnp.broadcast_to(b_prev[None, :], (len(checks), int(n_ch))) if (read is not None and checks) else None
    ec0 = jnp.zeros((2, len(checks), int(n_ch)), jnp.float32) if (read is not None and checks) else None
    c = (phys, m0, b0, q0, qb0, ev_c, ev_b, b_prev, f0, qf0, qd0, bpd0, ec0)
    c, reads = jax.lax.scan(group_of(n_read), c, jnp.arange(n_full, dtype=jnp.int32) * n_read)
    if rem:                                                                      # the block's end: the shorter interval
        c, last = group_of(rem)(c, jnp.int32(n_full * n_read))
        reads = None if reads is None else jnp.concatenate([reads, last[None]], axis=0)
    phys, mp, mb, qp, qb, ev_c, ev_b, b_prev, mf, qf, qd, _bpd, ec = c
    return (phys, b_prev, ev_c, ev_b), (mp, mb, qp, qb, mf, qf, qd, ec, reads)


def ring_scan(interval, carry, n_blocks, K, m, *, has_blt, K_c3=None, n_ch=0, emit=None, count=None, offset=0, n_checks=0):
    """The ``n_blocks`` blocks of one walker (one window): ``interval(carry, b) -> (carry, ys, (M, B, Q, Qb, F, Qf, Qd,
    E, reads))`` walks block ``b`` (its moments from :func:`moment_scan`; ``b + offset`` is the walk-wide block index).
    Returns ``(carry, acc_pos (K+1, 3), acc_blt (K+1,) or None, acc_field (K_c3, n_ch) or None, emitted, counted,
    (qv_pos (3,), qv_blt () or None, qv_field (n_ch,) or None, qv_increments (n_ch,) or None, checks (2, n_checks,
    n_ch) or None))``: the accumulators of the cosine projections
    (:func:`~dmipy_sim.replay.compression.block_table` evaluated per ring, the field's ``K_c3`` modes ``k = 0 .. K_c3 - 1`` at its own band),
    ``emit(ys)`` stacked over the blocks (None without ``emit``), ``count(ys)`` summed over them (int32; None without
    ``count``), each channel's quadratic variation over the window (the field's of its trapezoid increments and of its
    read series) and the trapezoid's checks, summed per ring like the projections."""
    from ..replay.compression import block_table
    hi = jax.lax.Precision.HIGHEST
    n_blocks, K, m = int(n_blocks), int(K), int(m)
    reads_f = int(n_ch) > 0                                                            # the field is read: its QVs
    has_f = K_c3 is not None and reads_f                                               # and its modes accumulated
    K_c3 = int(K_c3) if has_f else 1                                                   # modes k = 0 .. K_c3 - 1

    def block(c, b):
        c, ys, mom = interval(c, b + offset)
        return c, (mom, None if emit is None else emit(ys), None if count is None else count(ys).astype(jnp.int32))

    def ring(state, idx):
        c, ap, ab, af, n, qp, qb, qf, qd, ec = state
        c, (mom, e, k) = jax.lax.scan(block, c, idx)
        tab = block_table(1.0, n_blocks, K, m, blocks=idx, xp=jnp)                    # (R, K+1, m+1), no walker in it
        ap = ap + jnp.einsum("bkj,bjd->kd", tab, mom[0], precision=hi)
        qp = qp + mom[2].sum(axis=0)
        if has_blt:
            ab = ab + jnp.einsum("bkj,bj->k", tab, mom[1], precision=hi)
            qb = qb + mom[3].sum()
        if has_f:
            tab_f = tab if K_c3 == K + 1 else block_table(1.0, n_blocks, K_c3 - 1, m, blocks=idx, xp=jnp)
            af = af + jnp.einsum("bkj,bjc->kc", tab_f, mom[4], precision=hi)
        if reads_f:
            qf = qf + mom[5].sum(axis=0)
            qd = qd + mom[6].sum(axis=0)
            if ec is not None:
                ec = ec + mom[7].sum(axis=0)
        if count is not None:
            n = n + k.sum().astype(jnp.int32)
        return (c, ap, ab, af, n, qp, qb, qf, qd, ec), e

    state = (carry, jnp.zeros((K + 1, 3), jnp.float32), jnp.zeros((K + 1,), jnp.float32) if has_blt else None,
             jnp.zeros((K_c3, int(n_ch)), jnp.float32) if has_f else None,
             jnp.int32(0) if count is not None else None, jnp.zeros((3,), jnp.float32),
             jnp.float32(0.0) if has_blt else None, jnp.zeros((int(n_ch),), jnp.float32) if reads_f else None,
             jnp.zeros((int(n_ch),), jnp.float32) if reads_f else None,
             jnp.zeros((2, int(n_checks), int(n_ch)), jnp.float32) if (reads_f and n_checks) else None)
    n_full, rem = divmod(n_blocks, RING)
    parts = []
    if n_full:
        idx = jnp.arange(n_full * RING, dtype=jnp.int32).reshape(n_full, RING)
        state, e = jax.lax.scan(ring, state, idx)
        parts.append(jax.tree_util.tree_map(lambda a: a.reshape((n_full * RING,) + a.shape[2:]), e))
    if rem:
        state, e = ring(state, n_full * RING + jnp.arange(rem, dtype=jnp.int32))
        parts.append(e)
    stacked = parts[0] if len(parts) == 1 else jax.tree_util.tree_map(lambda *a: jnp.concatenate(a, axis=0), *parts)
    c, ap, ab, af, n, qp, qb, qf, qd, ec = state
    return c, ap, ab, af, stacked, n, (qp, qb, qf, qd, ec)


def coefficient_form(r0, r_end, acc_pos, scale):
    """A walker's position coefficients ``(K+2, 3)`` from its exact start and end and its cosine accumulators:
    ``[x(0), x(T) - x(0), scale_k c_k]`` (:func:`~dmipy_sim.replay.compression.cosine_band_scale`), and the
    float32 chain's residual on the ``k = 0`` mode, ``c_0 - (x(T) - x(0))`` (3,), which the walk records as the
    measured precision of its accumulation."""
    d = r_end - r0
    C = jnp.concatenate([r0[None, :], d[None, :], scale[:, None] * acc_pos[1:]], axis=0)
    return C, acc_pos[0] - d


def field_mode_form(acc_field, T):
    """A walker's field modes ``(K_c3, n_ch)`` on the orthonormal cosine basis of the window, ``c_k = int e_k b dt``
    with ``e_0 = T^-1/2`` and ``e_k = sqrt(2 / T) cos(pi k t / T)`` (RPK.md 9.5.3), from the accumulated projections
    ``sum cos(pi k t / T) b dt`` (:func:`~dmipy_sim.replay.compression.cosine_mode_scale`)."""
    from ..replay.compression import cosine_mode_scale
    return jnp.asarray(cosine_mode_scale(acc_field.shape[0] - 1, T), jnp.float32)[:, None] * acc_field


def accumulator_bytes(K, m, record, K_c3=0, n_ch=0, events_cap=0):
    """Device bytes one walker's accumulation holds through a batch: the level-2 accumulators (``(K+1)`` modes x 3
    axes, + the local time's ``K+1`` when recorded, + the field's ``(K_c3+1) n_ch``), a ring's moments and the block's
    running moments, the last field read, and an events buffer of ``events_cap`` entries (five bytes each), float32."""
    ch = 3 + (1 if record else 0)
    out = 4 * ch * ((int(K) + 1) + (RING + 1) * (int(m) + 1))
    if n_ch:
        out += 4 * int(n_ch) * (int(K_c3) + (RING + 1) * (int(m) + 1) + 1)
    out += 5 * int(events_cap)
    return out
