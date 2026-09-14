"""A wall nudge a float32 coordinate can carry.

The curved family sets a walker ``1e-4 R`` clear of the wall it bounced off. On the 1 mm DiSCo strands that is 72 pm
against a float32 ulp of 116 pm at 1 mm, so the placed point rounds onto the wall (or past it) and the next
classification reads it as outside its tube: the adaptive walk refused the intra pool for one such walker in
123k. The nudge is now the larger of ``1e-4 R`` and eight ulps at the domain's largest coordinate.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import dmipy_sim as d
from dmipy_sim.geometry._boundary import representable_nudge

R = 0.7182e-6


def _pack(offset):
    cl = [np.array([[-30e-6, 0, 0], [30e-6, 0, 0]]) + offset, np.array([[-30e-6, 3 * R, 0], [30e-6, 3 * R, 0]]) + offset]
    return d.PackedCurvedCylinders(cl, [R, R], interior=True)


def test_the_nudge_is_at_least_eight_ulps_of_the_largest_coordinate():
    assert representable_nudge(7e-11, 1e-3) == 8 * float(np.spacing(np.float32(1e-3))) > 7e-11
    assert representable_nudge(7e-11, 1e-6) == 7e-11
    near, far = _pack(np.zeros(3)), _pack(np.array([1e-3, 1e-3, 1e-3]))
    assert near.nudge_m == 1e-4 * R and far.nudge_m > 1e-4 * R


def test_every_walker_bounced_off_a_wall_a_millimetre_away_is_still_inside_its_tube():
    """20k exits through the wall of a strand at 1 mm: after the bounce every walker classifies inside. With the
    72 pm nudge a share of them sat on the wall to float32 and read as outside."""
    off = np.array([1e-3, 1e-3, 1e-3])
    g = _pack(off)
    rng = np.random.default_rng(0); n = 20_000
    ph = rng.uniform(0, 2 * np.pi, n)
    start = np.stack([rng.uniform(-20e-6, 20e-6, n), 0.9 * R * np.cos(ph), 0.9 * R * np.sin(ph)], 1) + off   # inside tube 0
    step = np.stack([rng.normal(0, 0.05 * R, n), 0.3 * R * np.cos(ph), 0.3 * R * np.sin(ph)], 1)             # out through the wall
    r = jax.jit(jax.vmap(g.reflect))(jnp.asarray(start, jnp.float32), jnp.asarray(step, jnp.float32))
    tube = np.asarray(jax.vmap(g.classify_position)(r))
    assert (tube == 1).all(), f"{(tube != 1).sum()} of {n} walkers read as outside their tube after the bounce"


def test_the_geometry_seeds_inside_by_its_nudge_at_a_millimetre():
    """The packed and the single-strand seeders draw the disc to the wall less the nudge: at 1 mm every seed reads
    as inside to the geometry's own float32 classification (drawn to the wall, 11 of a million seeds on DiSCo sat
    within rounding of it, read as outside, and the adaptive walk refused the pool)."""
    off = np.array([1e-3, 1e-3, 1e-3])
    g = _pack(off)
    P = np.asarray(g.init_positions(400_000, jax.random.PRNGKey(5)))
    assert (np.asarray(g.classify_positions_exact(P)) > 0).all()
    one = d.CurvedCylinder(np.array([[-30e-6, 0, 0], [30e-6, 0, 0]]) + off, R)
    Q = np.asarray(one.init_positions(200_000, jax.random.PRNGKey(6)))
    assert (np.asarray(one.classify_positions_exact(Q)) == 1).all()
