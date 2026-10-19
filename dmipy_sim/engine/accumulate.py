"""The walk's band accumulation: what a coefficient walk (``simulate_trajectories(accumulate=K)``) carries per walker
in place of its saves.

A walk runs in blocks of ``h`` seconds, the band's own unless given (:func:`block_length`: ``omega_K h =``
:data:`BLOCK_OMEGA_H` at the top band, the table's order :data:`BLOCK_ORDER`). Inside a block the step scan adds, per walker, the moments of each channel's
increments about the block's centre in the block's own unit ``tau = (t - t_b) / h`` (each step at its midpoint):
``M_j += tau^j dx`` for the positions and ``B_j += tau^j dlog`` for the local time, ``j = 0..m`` (:func:`moment_scan`).
At the end of every :data:`RING` blocks the ring's moments are projected onto the ``K + 1`` cosine modes as one
product against the ring's rows of :func:`~dmipy_sim.replay.compression.block_table` and added to the walker's
level-2 accumulator (:func:`ring_scan`): the sum over a ring runs inside one float32 product of ``RING (m + 1)``
terms and the chain over the walk has ``n_blocks / RING`` links, so the float32 rounding of a 5000-block walk is a
chain of 313 additions rather than 5000. Nothing is emitted per block but what a channel cannot hold otherwise (the
occupancy of a walk whose walkers can change pool, the raw positions of a declared subset), so a walker's working set
is the accumulators, the running position and the exact start, whatever ``h`` and the walk's length.
"""
import jax
import jax.numpy as jnp
import numpy as np

#: blocks per ring: the projection's inner sum and the stride of the level-2 chain
RING = 16

#: the block table's order a coefficient walk takes by default (dmipy-sim-cuda#74)
BLOCK_ORDER = 3

#: ``omega_K h`` at the top band of the block a coefficient walk takes by default: with ``m = 3`` the quadrature adds
#: ``quadrature_rel(3, h, omega_K) = 0.8^8 / 1327104 = 1.3e-7`` of the top band's phase variance, for every class
#: (measured 1.41e-7 on a T4, dmipy-sim-cuda#74)
BLOCK_OMEGA_H = 0.8


def block_length(T, K):
    """The block of a coefficient walk of ``K`` bands over ``T`` seconds when none is given: ``h = BLOCK_OMEGA_H T /
    (pi K)``, so that ``omega_K h = 0.8`` at the top band ``omega_K = pi K / T``, rounded to a whole number of blocks
    over the walk (``T / round(T / h)``) -- 100 us at K = 256 over 100 ms. The band sets it, never the walker count or
    the step: the quadrature it costs is the same fraction of the top band for every scanner class."""
    h = float(BLOCK_OMEGA_H) * float(T) / (np.pi * int(K))
    return float(T) / max(int(round(float(T) / h)), 1)


#: what the walk records as its accumulation precision (``CoefficientWalk.block["precision"]``)
PRECISION = f"float32: the moments of each block summed per step, a ring of {RING} blocks projected in one product, the rings chained into a level-2 accumulator"


def moment_scan(inner, init, length, pos_of, blt_of, m):
    """``inner`` stepped ``length`` times from ``init`` with the moments of the increments: ``(carry, (M (m+1, 3),
    B (m+1,) or None, Q (3,), Qb () or None))``, ``M_j = sum_i tau_i^j (pos_of(c_i+1) - pos_of(c_i))`` and the same of
    ``blt_of`` (a cumulative local time in the carry), ``tau_i = (i + 1/2) / length - 1/2``, and each channel's
    quadratic variation over the block, ``Q = sum_i (dx_i)^2`` per axis and ``Qb = sum_i (dlog_i)^2``: what the
    block table's error is proportional to (:func:`~dmipy_sim.replay.compression.quadrature_rel` times it)."""
    tau = (jnp.arange(length, dtype=jnp.float32) + 0.5) / jnp.float32(length) - 0.5
    pw = tau[:, None] ** jnp.arange(int(m) + 1, dtype=jnp.float32)[None, :]          # (length, m+1), shared by every walker
    has_b = blt_of is not None

    def body(c, p):
        cc, mp, mb, qp, qb = c
        r0 = pos_of(cc)
        b0 = blt_of(cc) if has_b else None
        cc, _ = inner(cc, None)
        dx = pos_of(cc) - r0
        mp = mp + p[:, None] * dx[None, :]
        qp = qp + dx * dx
        if has_b:
            db = blt_of(cc) - b0
            mb = mb + p * db
            qb = qb + db * db
        return (cc, mp, mb, qp, qb), None

    m0 = jnp.zeros((int(m) + 1, 3), jnp.float32)
    b0 = jnp.zeros((int(m) + 1,), jnp.float32) if has_b else None
    q0 = jnp.zeros((3,), jnp.float32)
    qb0 = jnp.float32(0.0) if has_b else None
    (out, mp, mb, qp, qb), _ = jax.lax.scan(body, (init, m0, b0, q0, qb0), pw)
    return out, (mp, mb, qp, qb)


def ring_scan(interval, carry, n_blocks, K, m, *, has_blt, emit=None, count=None):
    """The ``n_blocks`` blocks of one walker: ``interval(carry, None) -> (carry, ys, (M, B))`` walks one block
    (its moments from :func:`moment_scan`). Returns ``(carry, acc_pos (K+1, 3), acc_blt (K+1,) or None, emitted,
    counted, (qv_pos (3,), qv_blt () or None))``: the accumulators of the cosine projections
    (:func:`~dmipy_sim.replay.compression.block_table` evaluated per ring), ``emit(ys)`` stacked over the blocks (None
    without ``emit``), ``count(ys)`` summed over them (int32; None without ``count``) and each channel's quadratic
    variation over the walk, summed per ring like the projections."""
    from ..replay.compression import block_table
    hi = jax.lax.Precision.HIGHEST
    n_blocks, K, m = int(n_blocks), int(K), int(m)

    def block(c, _):
        c, ys, mom = interval(c, None)
        return c, (mom, None if emit is None else emit(ys), None if count is None else count(ys).astype(jnp.int32))

    def ring(state, idx):
        c, ap, ab, n, qp, qb = state
        c, (mom, e, k) = jax.lax.scan(block, c, None, length=int(idx.shape[0]))
        tab = block_table(1.0, n_blocks, K, m, blocks=idx, xp=jnp)                    # (R, K+1, m+1), no walker in it
        ap = ap + jnp.einsum("bkj,bjd->kd", tab, mom[0], precision=hi)
        qp = qp + mom[2].sum(axis=0)
        if has_blt:
            ab = ab + jnp.einsum("bkj,bj->k", tab, mom[1], precision=hi)
            qb = qb + mom[3].sum()
        if count is not None:
            n = n + k.sum()
        return (c, ap, ab, n, qp, qb), e

    state = (carry, jnp.zeros((K + 1, 3), jnp.float32), jnp.zeros((K + 1,), jnp.float32) if has_blt else None,
             jnp.int32(0) if count is not None else None, jnp.zeros((3,), jnp.float32),
             jnp.float32(0.0) if has_blt else None)
    n_full, rem = divmod(n_blocks, RING)
    parts = []
    if n_full:
        idx = jnp.arange(n_full * RING, dtype=jnp.int32).reshape(n_full, RING)
        state, e = jax.lax.scan(ring, state, idx)
        if emit is not None:
            parts.append(jax.tree_util.tree_map(lambda a: a.reshape((n_full * RING,) + a.shape[2:]), e))
    if rem:
        state, e = ring(state, n_full * RING + jnp.arange(rem, dtype=jnp.int32))
        if emit is not None:
            parts.append(e)
    emitted = None
    if emit is not None:
        emitted = parts[0] if len(parts) == 1 else jax.tree_util.tree_map(lambda *a: jnp.concatenate(a, axis=0), *parts)
    c, ap, ab, n, qp, qb = state
    return c, ap, ab, emitted, n, (qp, qb)


def coefficient_form(r0, r_end, acc_pos, scale):
    """A walker's position coefficients ``(K+2, 3)`` from its exact start and end and its cosine accumulators:
    ``[x(0), x(T) - x(0), scale_k c_k]`` (:func:`~dmipy_sim.replay.compression.cosine_band_scale`), and the
    float32 chain's residual on the ``k = 0`` mode, ``c_0 - (x(T) - x(0))`` (3,), which the walk records as the
    measured precision of its accumulation."""
    d = r_end - r0
    C = jnp.concatenate([r0[None, :], d[None, :], scale[:, None] * acc_pos[1:]], axis=0)
    return C, acc_pos[0] - d


def accumulator_bytes(K, m, record):
    """Device bytes one walker's accumulation holds through a batch: the level-2 accumulators (``(K+1)`` modes x 3
    axes, + the local time's ``K+1`` when recorded), a ring's moments and the block's running moments, float32."""
    ch = 3 + (1 if record else 0)
    return 4 * ch * ((int(K) + 1) + (RING + 1) * (int(m) + 1))
