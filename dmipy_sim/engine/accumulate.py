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

The field read's rule (RPK.md 9.5.5, dmrai-lab/dmipy-sim#742): between reads the walker moves ``sqrt(2 D dt_f)``, and a
midpoint sum of the field along a path of that roughness adds to the field phase an independent term of variance
``v_read = (gamma B0 chi)^2 2 D T_e |grad b|^2 dt_f^2``; :func:`field_read_interval` is the ``dt_f`` that holds
``v_read`` under the square of :data:`FIELD_READ_FLOOR` at the envelope's ``B0`` and ``chi``, and the walk reads at the
largest divisor of its step count that fits it (never finer than the step, never coarser than the block). With the
trapezoid the error between two reads is the path's deviation from the chord, ``|grad b|^2 2 D dt_f^3 / 12`` per
interval against the rectangle rule's ``/ 3`` (``v_read`` above is stated for the midpoint sum, the conservative
figure); the certificate's term is ``field_read = v_read / 2`` (:func:`field_read_term`).
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
#: ``build_to_floor``'s ``sigma_star``): ``v_read <= FIELD_READ_FLOOR^2`` at the envelope's ``B0`` and ``chi``
FIELD_READ_FLOOR = 1e-3

#: the field strength (T) and susceptibility scale the read rule and the field band are derived at when the
#: envelope states none: the certificate's own defaults (``B0_list`` [3, 7]; myelin's isotropic chi)
FIELD_RULE_B0 = 7.0
FIELD_RULE_CHI = 1.06e-6

#: walkers of the pilots a walk runs before itself: the field band's ladder (:func:`~dmipy_sim.replay.bank.derive_field_band`)
#: and the events buffer's capacity (the largest count a pilot walker reached, with :data:`EVENTS_MARGIN`)
PILOT_WALKERS = 1024
EVENTS_MARGIN = (1.5, 8)      # capacity = ceil(1.5 x the pilot's largest count) + 8


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
def field_read_interval(D, T_e, grad_b, B0=FIELD_RULE_B0, chi=FIELD_RULE_CHI, floor=FIELD_READ_FLOOR):
    """The read interval ``dt_f`` (s) that holds ``v_read = (gamma B0 chi)^2 2 D T_e grad_b^2 dt_f^2`` under
    ``floor^2``: ``floor / (gamma B0 chi sqrt(2 D T_e) grad_b)``. ``grad_b`` (1/m) is the field basis's largest
    gradient over the sheath (``gradient_scale`` of the basis), ``T_e`` the duration the read error accrues over
    (the storage window), ``D`` the walk's diffusivity."""
    denom = GAMMA * float(B0) * float(chi) * math.sqrt(2.0 * float(D) * float(T_e)) * float(grad_b)
    if not denom > 0.0:
        return math.inf
    return float(floor) / denom


def field_read_term(dt_f, D, T_e, grad_b, B0, chi):
    """``field_read = v_read / 2``: the signal bound of the read at ``dt_f`` (RPK.md 9.5.5's term)."""
    return 0.5 * (GAMMA * float(B0) * float(chi)) ** 2 * 2.0 * float(D) * float(T_e) * float(grad_b) ** 2 * float(dt_f) ** 2


def field_read_B0_chi_max(dt_f, D, T_e, grad_b, floor=FIELD_READ_FLOOR):
    """The product ``B0 chi`` (T) under which the read term at ``dt_f`` stays below ``floor``: what a pack whose read is
    the step states in its envelope (RPK.md 9.4 rule 5)."""
    base = 2.0 * float(D) * float(T_e) * float(grad_b) ** 2 * float(dt_f) ** 2
    if not base > 0.0:
        return math.inf
    return math.sqrt(2.0 * float(floor) / base) / GAMMA


def field_read_steps(dt_f_rule, dt_sim, sub_steps):
    """``(n_read, dt_f)``: the read every ``n_read`` steps, the largest divisor of the block's ``sub_steps`` whose
    interval ``n_read dt_sim`` is within the rule's ``dt_f_rule`` -- the step itself when the rule asks for less
    than a step (a read along the chord between two steps carries the same bridge deviation: the read is never
    sub-stepped), the block when the rule allows more than one."""
    sub_steps = int(sub_steps)
    n_max = int(math.floor(float(dt_f_rule) / float(dt_sim) * (1.0 + 1e-9)))
    n = 1
    for d in range(1, sub_steps + 1):
        if sub_steps % d == 0 and d <= n_max:
            n = d
    return n, n * float(dt_sim)


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
def moment_scan(kind, carry, length, m, *, read=None, n_ch=0, n_read=1, dt_f=0.0, step0=None):
    """One block of ``length`` steps of ``kind`` from ``carry = (phys, b_prev, ev_comp, ev_bound)``: the physics carry
    stepped ``length`` times, with the moments of the increments ``M_j = sum_i tau_i^j (pos(c_i+1) - pos(c_i))`` (m+1,
    3) and the same ``B_j`` of the cumulative local time, ``tau_i = (i + 1/2) / length - 1/2``; each channel's
    quadratic variation ``Q = sum_i (dx_i)^2`` per axis and ``Qb = sum_i (dlog_i)^2``; with ``read`` the field read at
    every ``n_read``-th step (``length`` a multiple of it) and the TRAPEZOID increment of each read interval,
    ``(b_prev + b) dt_f / 2``, at the interval's midpoint: ``F_j`` (m+1, n_ch), ``Qf = sum (inc)^2`` (n_ch,), the last
    read carried in ``b_prev``; with an events buffer the label after every step recorded on change at the global step
    ``step0 + i + 1``. Returns ``(carry, (M, B, Q, Qb, F, Qf))`` (None where a channel is not recorded)."""
    phys, b_prev, ev_c, ev_b = carry
    length, n_read, m = int(length), int(n_read), int(m)
    if length % n_read:
        raise ValueError(f"the field is read every {n_read} steps, which does not divide the block's {length}")
    n_groups = length // n_read
    has_b = kind.blt_of is not None
    js = jnp.arange(m + 1, dtype=jnp.float32)
    inv_len = jnp.float32(1.0 / length)
    half_dt_f = jnp.float32(0.5 * float(dt_f))
    s0 = jnp.int32(0) if step0 is None else jnp.asarray(step0).astype(jnp.int32)

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

    def group(c, g):
        cc, mp, mb, qp, qb, evc, evb, bp, mf, qf = c
        idx = g * n_read + jnp.arange(n_read, dtype=jnp.int32)
        (cc, mp, mb, qp, qb, evc, evb), _ = jax.lax.scan(step, (cc, mp, mb, qp, qb, evc, evb), idx)
        if read is not None:
            b = read(kind.read_pos_of(cc))
            inc = (bp + b) * half_dt_f                                       # the trapezoid over the read interval
            tau = ((g.astype(jnp.float32) + 0.5) * n_read) * inv_len - 0.5     # the interval's midpoint
            mf = mf + (tau ** js)[:, None] * inc[None, :]
            qf = qf + inc * inc
            bp = b
        return (cc, mp, mb, qp, qb, evc, evb, bp, mf, qf), None

    m0 = jnp.zeros((m + 1, 3), jnp.float32)
    b0 = jnp.zeros((m + 1,), jnp.float32) if has_b else None
    q0 = jnp.zeros((3,), jnp.float32)
    qb0 = jnp.float32(0.0) if has_b else None
    f0 = jnp.zeros((m + 1, int(n_ch)), jnp.float32) if read is not None else None
    qf0 = jnp.zeros((int(n_ch),), jnp.float32) if read is not None else None
    (phys, mp, mb, qp, qb, ev_c, ev_b, b_prev, mf, qf), _ = jax.lax.scan(
        group, (phys, m0, b0, q0, qb0, ev_c, ev_b, b_prev, f0, qf0), jnp.arange(n_groups, dtype=jnp.int32))
    return (phys, b_prev, ev_c, ev_b), (mp, mb, qp, qb, mf, qf)


def ring_scan(interval, carry, n_blocks, K, m, *, has_blt, K_c3=None, n_ch=0, emit=None, count=None, offset=0):
    """The ``n_blocks`` blocks of one walker (one window): ``interval(carry, b) -> (carry, ys, (M, B, Q, Qb, F, Qf))``
    walks block ``b`` (its moments from :func:`moment_scan`; ``b + offset`` is the walk-wide block index). Returns
    ``(carry, acc_pos (K+1, 3), acc_blt (K+1,) or None, acc_field (K_c3, n_ch) or None, emitted, counted, (qv_pos
    (3,), qv_blt () or None, qv_field (n_ch,) or None))``: the accumulators of the cosine projections
    (:func:`~dmipy_sim.replay.compression.block_table` evaluated per ring, the field's ``K_c3`` modes ``k = 0 .. K_c3 - 1`` at its own band),
    ``emit(ys)`` stacked over the blocks (None without ``emit``), ``count(ys)`` summed over them (int32; None without
    ``count``), each channel's quadratic variation over the window, summed per ring like the projections."""
    from ..replay.compression import block_table
    hi = jax.lax.Precision.HIGHEST
    n_blocks, K, m = int(n_blocks), int(K), int(m)
    has_f = K_c3 is not None and int(n_ch) > 0
    K_c3 = int(K_c3) if has_f else 1                                                   # modes k = 0 .. K_c3 - 1

    def block(c, b):
        c, ys, mom = interval(c, b + offset)
        return c, (mom, None if emit is None else emit(ys), None if count is None else count(ys).astype(jnp.int32))

    def ring(state, idx):
        c, ap, ab, af, n, qp, qb, qf = state
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
            qf = qf + mom[5].sum(axis=0)
        if count is not None:
            n = n + k.sum().astype(jnp.int32)
        return (c, ap, ab, af, n, qp, qb, qf), e

    state = (carry, jnp.zeros((K + 1, 3), jnp.float32), jnp.zeros((K + 1,), jnp.float32) if has_blt else None,
             jnp.zeros((K_c3, int(n_ch)), jnp.float32) if has_f else None,
             jnp.int32(0) if count is not None else None, jnp.zeros((3,), jnp.float32),
             jnp.float32(0.0) if has_blt else None, jnp.zeros((int(n_ch),), jnp.float32) if has_f else None)
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
    c, ap, ab, af, n, qp, qb, qf = state
    return c, ap, ab, af, stacked, n, (qp, qb, qf)


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
