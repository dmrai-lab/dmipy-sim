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


def test_every_point_at_the_seeders_radius_reads_inside_at_a_millimetre():
    """The packed and the single-strand seeders draw the disc to the wall less the nudge, so the farthest point a
    seed can sit is at ``R - nudge`` from the axis. Every such point, around the whole circle, along the whole
    tube and at 1 mm, reads as inside to the geometry's own float32 classification; the wall itself, one nudge
    further, is where that stops being decidable."""
    off = np.array([1e-3, 1e-3, 1e-3])
    g = _pack(off)
    ph = np.linspace(0.0, 2 * np.pi, 3600, endpoint=False)
    xs = np.array([-30e-6, -29.9e-6, -15e-6, 0.0, 15e-6, 29.9e-6, 30e-6])
    rho = R - g.nudge_m
    for y0, tube in ((0.0, 1), (3 * R, 2)):                         # both strands of the pack
        P = np.stack([np.repeat(xs, ph.size), np.tile(y0 + rho * np.cos(ph), xs.size), np.tile(rho * np.sin(ph), xs.size)], 1) + off
        lab = np.asarray(g.classify_positions_exact(P))
        assert (lab == tube).all(), f"{(lab != tube).sum()} of {P.shape[0]} points at R - nudge read outside strand {tube}"
    one = d.CurvedCylinder(np.array([[-30e-6, 0, 0], [30e-6, 0, 0]]) + off, R)
    rho1 = R - one.nudge_m
    Q = np.stack([np.repeat(xs, ph.size), np.tile(rho1 * np.cos(ph), xs.size), np.tile(rho1 * np.sin(ph), xs.size)], 1) + off
    assert (np.asarray(one.classify_positions_exact(Q)) == 1).all()
    # and the seeders do stay within that radius: a draw is never past R - nudge
    S = np.asarray(g.init_positions(4_000, jax.random.PRNGKey(5)), np.float64) - off
    r_axis = np.sqrt(np.minimum(S[:, 1] ** 2, (S[:, 1] - 3 * R) ** 2) + S[:, 2] ** 2)
    assert (r_axis <= rho * (1 + 1e-6)).all()
