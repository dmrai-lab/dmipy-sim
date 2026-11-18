"""The step a cell of myelinated cylinders only is sized for, and what happens to a draw longer than it.

The wall kernel of a :class:`~dmipy_sim.geometry.PackedMyelinatedCylinders` cell sizes its bounce budget and its
candidate search from ``MyelinatedCell.step_max``. A cell of sheathed cylinders only is sized for ``sqrt(6 D dt)``; the
step law's longest draw is :data:`~dmipy_sim.math.step_law.LONGEST_STEP` times that, and a cell holding a plain
cylinder is sized for it. Exact, not statistical:

* the sizing of both classes, to the stored float32 diffusivity, and the budgets it gives;
* the loop's rule: a budget that runs out mid-step stops the walker at its last collision and reports it;
* a witness lane of a lumen whose budget runs out at the longest draw: the kernel's end is the stop point of an
  independent float64 specular ray, and the point a flown remainder would reach is a different one;
* constructed near-tangent lanes of the lumen, of the sheath (diffusivity above zero) and of the extra-axonal pool,
  at the longest draw and at small steps: where the budget runs out the walker ends in its own pool at the reference's
  stop point, and the point the unflown remainder would reach lies in the walker's own pool as well;
* the candidate count as a function of the step over the smallest lumen radius.
"""
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from dmipy_sim.compartments import Compartments, Pool
from dmipy_sim.engine import physics as ph
from dmipy_sim.geometry import PackedMyelinatedCylinders
from dmipy_sim.geometry._boundary import BounceLoop
from dmipy_sim.geometry.myelin import concentric_step_max
from dmipy_sim.math.step_law import LONGEST_STEP

D = 2e-9
R = 2e-6
OUTER = np.array([R, 0.8 * R, R, 0.9 * R])
L = 4.4 * R
CENTRES = np.array([[0.0, 0.0], [2.2, 0.0], [0.0, 2.2], [2.2, 2.2]]) * R
G_RATIO = 0.7
INNER = OUTER * G_RATIO
NUDGE = 1e-4 * INNER.min()                      #: the kernel's push off a wall: 1e-4 of the smallest lumen radius
DRY_SHEATH = Compartments(intra=Pool(D=D), myelin=Pool(D=0.0), extra=Pool(D=D))
WET_SHEATH = Compartments(intra=Pool(D=D), myelin=Pool(D=D), extra=Pool(D=D))
N_DEPTH, N_OFFSET = 300, 301                    #: the near-tangent grid: start depths and direction offsets
MAX_STOP_DEVIATION = 1e-9                       #: m, the kernel (float32) against the reference (float64)


def _cell(g, compartments=DRY_SHEATH):
    g = np.asarray(g, float)
    return PackedMyelinatedCylinders(OUTER * g, g, CENTRES, L, N_max=4, compartments=compartments)


@functools.lru_cache(maxsize=None)
def _all_myelinated(wet=False):
    geometry = _cell([G_RATIO] * 4, WET_SHEATH if wet else DRY_SHEATH)
    geometry.count_walls = True
    return geometry


@functools.lru_cache(maxsize=None)
def _substep(dt, wet=False):
    return ph.make_myelin_substep(_all_myelinated(wet), dt)


# ── an independent reference: specular rays among circles, in float64 ───────────────────────────────────────────────

def _images(circles):
    """``circles`` ``(cx, cy, radius, side)`` with its 3 x 3 periodic images; ``side`` +1 where the walker is inside."""
    return np.array([(cx + i * L, cy + j * L, r, s) for i in (-1, 0, 1) for j in (-1, 0, 1) for cx, cy, r, s in circles])


def _reference(p, angle, step, walls, budget):
    """The path of a ray that reflects specularly off ``walls`` (``(m, 4)``: ``cx, cy, radius, side``), ``budget``
    collisions at most, each reflection leaving the walker ``NUDGE`` inside its side of the wall and costing that much
    path. Returns the point where the ray stops, the point its unflown remainder would reach, whether the budget ran
    out with path left, and that remainder."""
    p = np.array(p, np.float64)
    d = np.stack([np.cos(angle), np.sin(angle)], 1)
    rem = np.full(len(p), float(step))
    inside, radius, rows = walls[:, 3] > 0, walls[:, 2], np.arange(len(p))
    hit = np.zeros(len(p), bool)
    for _ in range(budget):
        rel = p[:, None, :] - walls[None, :, :2]
        b = np.sum(rel * d[:, None, :], -1)
        disc = b * b - (np.sum(rel * rel, -1) - radius ** 2)
        root = np.sqrt(np.maximum(disc, 0.0))
        t = np.where(inside, -b + root, -b - root)                      # the exit root from inside, the entry from outside
        t = np.where((disc > 0) & (t > 0) & (t < rem[:, None]), t, np.inf)
        j = np.argmin(t, 1)
        t_hit = t[rows, j]
        hit = np.isfinite(t_hit) & (rem > 0)
        t_hit = np.where(hit, t_hit, 0.0)
        n = rel[rows, j] + t_hit[:, None] * d
        n = n / np.linalg.norm(n, axis=1, keepdims=True)
        d_reflected = d - 2.0 * np.sum(d * n, 1, keepdims=True) * n
        p_reflected = walls[j, :2] + (radius[j] - np.where(inside[j], NUDGE, -NUDGE))[:, None] * n
        p = np.where(hit[:, None], p_reflected, p + rem[:, None] * d)
        d = np.where(hit[:, None], d_reflected, d)
        rem = np.where(hit, np.maximum(rem - t_hit - NUDGE, 0.0), 0.0)
    exhausted = hit & (rem > 0)
    return p, p + rem[:, None] * d, exhausted, rem


def _minimum_image(delta):
    return delta - L * np.round(delta / L)


def _in_pool(points, walls):
    """Whether each of ``points`` lies in the pool ``walls`` bound: inside one image of a ``side`` +1 circle (when
    there is one) and outside every ``side`` -1 circle; the walker's position is the kernel's, folded into the cell."""
    r = np.linalg.norm(_minimum_image(points[:, None, :] - walls[None, :, :2]), axis=2)
    inside = walls[None, :, 3] > 0
    ok = ~np.any(~inside & (r <= walls[None, :, 2]), 1)
    if np.any(inside):
        ok &= np.any(inside & (r < walls[None, :, 2]), 1)
    return ok


def _near_tangent_lanes(radius, centre, side, theta=0.3):
    """Starts 0.05 to 6 nudges off the circle of ``radius`` about ``centre`` (``side`` +1: outside it, -1: inside it),
    directions within +-0.05 rad of the tangent there: ``N_DEPTH x N_OFFSET`` lanes, depth major."""
    depth, offset = (a.ravel() for a in np.meshgrid(NUDGE * np.geomspace(0.05, 6.0, N_DEPTH),
                                                    np.linspace(-0.05, 0.05, N_OFFSET), indexing="ij"))
    rad = radius + side * depth
    xy = centre + np.stack([rad * math.cos(theta), rad * math.sin(theta)], 1)
    return xy.astype(np.float32).astype(np.float64), theta + np.pi / 2 + offset


def _longest_draw(dt):
    return float(np.float32(LONGEST_STEP * math.sqrt(6.0 * D * dt)))


def _move(dt, xy, angle, label, wet=False):
    """The kernel's sub-step of ``dt`` seconds of every lane ``xy`` (in-plane, float32-representable), ``angle``, one
    longest draw long, in pool ``label``: ``(end, label after, pool at the end, exhausted, reflections)``."""
    n = len(xy)
    step = np.float32(_longest_draw(dt))
    s = np.stack([step * np.cos(angle), step * np.sin(angle), np.zeros(n)], 1).astype(np.float32)
    r0 = np.concatenate([xy, np.zeros((n, 1))], 1).astype(np.float32)
    substep, geometry = _substep(dt, wet), _all_myelinated(wet)
    f = jax.jit(jax.vmap(lambda r_, s_: substep.move(r_, s_, jnp.int32(label), jnp.float32(0.99), None)))
    r_new, comp_new, _, _, (hits, _, exhausted) = f(jnp.asarray(r0), jnp.asarray(s))
    pool = np.asarray(jax.vmap(geometry.classify_position)(r_new))
    return (np.asarray(r_new)[:, :2].astype(np.float64), np.asarray(comp_new), pool, np.asarray(exhausted),
            np.asarray(hits))


def _compare(dt, xy, angle, label, walls, wet=False):
    """One search: the kernel and the reference on the same lanes. Returns the counts the claims are made of."""
    end, comp, pool, exhausted, hits = _move(dt, xy, angle, label, wet)
    stop, fly, ref_exhausted, remainder = _reference(xy, angle, _longest_draw(dt), walls, _substep(dt, wet).bounce_loop.budget)
    return dict(
        budget=_substep(dt, wet).bounce_loop.budget, lanes=len(xy), exhausted=int(exhausted.sum()),
        reference_exhausted=int(ref_exhausted.sum()), max_reflections=int(hits.max()),
        label_changed=int((comp != label).sum()), end_not_in_label=int((pool != comp).sum()),
        end_outside_pool=int((~_in_pool(end, walls)).sum()),
        fly_outside_pool=int((ref_exhausted & ~_in_pool(fly, walls)).sum()),
        max_stop_deviation=float(np.linalg.norm(_minimum_image(end - stop), axis=1).max()),
        max_remainder=float(remainder[ref_exhausted].max()) if ref_exhausted.any() else 0.0)


# ── the loop's rule ─────────────────────────────────────────────────────────────────────────────────────────────────

def _toy(budget):
    """A walker on a line that bounces at every unit of path: ``rule`` moves it by ``min(1, remaining)`` and reverses it
    while path remains. Five units of path take six collisions' worth of budget to spend."""
    def rule(p, d, rem, state, i):
        t = jnp.minimum(1.0, rem)
        return p + t * d, jnp.where(rem > 1, -d, d), rem - t, state, 0, False, rem > 1
    return BounceLoop(budget)(rule, jnp.zeros(1), jnp.ones(1), jnp.float32(5.0))


def test_a_bounce_loop_that_runs_out_stops_at_its_last_collision_and_reports_it():
    out = _toy(2)
    assert out.exhausted
    assert np.asarray(out.r) == [0.0]               # flying the remainder would end at 3
    done = _toy(6)
    assert not done.exhausted
    assert np.asarray(done.r) == [1.0]


# ── the lumen at the longest draw ───────────────────────────────────────────────────────────────────────────────────

def _lumen(k):
    return np.array([(*CENTRES[k], INNER[k], 1)])


def test_the_longest_draw_exhausts_the_all_myelinated_budget_at_the_witness_lane_and_stops_at_the_reference_point():
    """The lane of the near-tangent grid of the smallest lumen at ``dt`` 1e-6 whose unflown remainder is the
    reference's largest. It is chosen by the float64 reference alone, so the choice does not depend on the kernel's
    arithmetic; the kernel's budget (5) runs out there, the walker stays in its lumen and ends at the point the
    reference stops at, not at the point the remainder would carry it to."""
    dt, k = 1e-6, 1
    walls = _lumen(k)
    xy, angle = _near_tangent_lanes(INNER[k], CENTRES[k], -1)
    budget = _substep(dt).bounce_loop.budget
    stop, fly, exhausted, remainder = _reference(xy, angle, _longest_draw(dt), walls, budget)
    lane = int(np.argmax(np.where(exhausted, remainder, -1.0)))
    assert divmod(lane, N_OFFSET) == (0, 147)
    assert 8.5e-9 < remainder[lane] < 9.2e-9                    # about 5 % of the longest draw
    end, comp, pool, kernel_exhausted, hits = _move(dt, xy[lane:lane + 1], angle[lane:lane + 1], k + 1)
    assert kernel_exhausted[0] and hits[0] == budget
    assert comp[0] == k + 1 and pool[0] == k + 1
    assert np.linalg.norm(_minimum_image(end[0] - stop[lane])) < MAX_STOP_DEVIATION
    assert np.linalg.norm(_minimum_image(end[0] - fly[lane])) > 5e-9


def test_a_budget_that_runs_out_in_a_lumen_leaves_the_walker_on_the_reference_path_in_its_pool():
    """Near-tangent lanes of every lumen at the longest draw: where the budget runs out (the smallest lumen at the small
    steps) the label does not change, the end position is in the pool its label names and at the reference's stop
    point, and the point the remainder would reach is in the lumen as well; at ``dt`` 4e-6 the budget is not reached."""
    for dt, k, runs_out in ((2e-7, 1, True), (1e-6, 1, True), (2e-7, 3, True), (4e-6, 1, False)):
        out = _compare(dt, *_near_tangent_lanes(INNER[k], CENTRES[k], -1), k + 1, _lumen(k))
        assert (out["exhausted"] > 0) == runs_out, (dt, k, out)
        assert out["label_changed"] == out["end_not_in_label"] == out["end_outside_pool"] == out["fly_outside_pool"] == 0, \
            (dt, k, out)
        assert out["max_stop_deviation"] < MAX_STOP_DEVIATION, (dt, k, out)


# ── the sheath and the extra-axonal pool ────────────────────────────────────────────────────────────────────────────

def test_a_budget_that_runs_out_in_a_sheath_leaves_the_walker_on_the_reference_path_in_its_pool():
    """A sheath of positive diffusivity: lanes starting inside it, grazing its outer wall from inside (the whispering
    path along the concave wall) and its inner wall from outside, at the longest draw. The budget runs out on the outer
    wall of the smaller cylinders at ``dt`` 2e-7 and nowhere at 5e-7; the walker never changes label and the remainder
    would not have left the sheath."""
    k = 1
    walls = np.array([(*CENTRES[k], OUTER[k], 1), (*CENTRES[k], INNER[k], -1)])
    label = 4 + k + 1                                            # N_max + k + 1: the sheath of cylinder k
    for dt, runs_out in ((2e-7, True), (5e-7, False)):
        for radius, side in ((OUTER[k], -1), (INNER[k], 1)):
            out = _compare(dt, *_near_tangent_lanes(radius, CENTRES[k], side), label, walls, wet=True)
            assert (out["exhausted"] > 0) == (runs_out and radius == OUTER[k]), (dt, radius, out)
            assert out["label_changed"] == out["end_not_in_label"] == out["end_outside_pool"] == out["fly_outside_pool"] == 0, \
                (dt, radius, out)
            assert out["max_stop_deviation"] < MAX_STOP_DEVIATION, (dt, radius, out)


def test_the_extra_axonal_pool_makes_one_reflection_where_the_walls_are_convex_and_never_runs_out():
    """From outside, an outer wall is convex: lanes grazing it, and lanes anywhere in the narrowest gap between two
    cylinders at every direction, make at most one reflection at the longest draw, so the smallest budget (3, at
    ``dt`` 2e-7) is never reached."""
    walls = _images([(*CENTRES[i], OUTER[i], -1) for i in range(4)])
    gap = np.array([0.0, 1.1 * R])                               # the middle of the gap between cylinders 0 and 2
    rng = np.random.default_rng(0)
    xy = (gap + rng.uniform(-1.5e-6, 1.5e-6, (30000, 2))).astype(np.float32).astype(np.float64)
    xy = xy[_in_pool(xy, walls) & (np.min(np.linalg.norm(xy[:, None] - walls[None, :, :2], axis=2) - walls[None, :, 2], 1)
                                   > 3 * NUDGE)]
    lanes = [(xy, rng.uniform(0.0, 2.0 * np.pi, len(xy)))] + [_near_tangent_lanes(OUTER[k], CENTRES[k], 1) for k in (0, 1)]
    for xy_, angle in lanes:
        out = _compare(2e-7, xy_, angle, 0, walls, wet=True)
        assert out["max_reflections"] <= 1 and out["exhausted"] == 0, out
        assert out["label_changed"] == out["end_not_in_label"] == out["end_outside_pool"] == 0, out
        assert out["max_stop_deviation"] < MAX_STOP_DEVIATION, out


# ── the candidate count ─────────────────────────────────────────────────────────────────────────────────────────────

def _candidates(x, n_cylinders):
    """``min(N, max(8, ceil(pi (1 + x)) + 2))``: the walls within reach of a step of ``x`` smallest lumen radii."""
    return min(n_cylinders, max(8, math.ceil(math.pi * (1.0 + x)) + 2))


def _sixteen_cylinders():
    centres = np.array([[(i + 0.5) * 5e-6, (j + 0.5) * 5e-6] for i in range(4) for j in range(4)])
    return PackedMyelinatedCylinders(np.full(16, G_RATIO * R), np.full(16, G_RATIO), centres, 2e-5, N_max=16,
                                     compartments=DRY_SHEATH)


def test_the_candidate_count_is_8_at_step_max_and_at_the_longest_draw_up_to_x_0_55():
    """``x`` is the step over the smallest lumen radius. The kernel is sized at ``step_max``; the count a longest draw
    would ask for is 8 up to ``x`` 0.55 and 9 beyond (``x`` 0.56 and 0.6, coarser than the step law's own ceiling)."""
    cell = _sixteen_cylinders()
    R_in = G_RATIO * R
    for x, at_longest in ((0.1, 8), (0.3, 8), (0.55, 8), (0.56, 9), (0.6, 9)):
        dt = (x * R_in) ** 2 / (6.0 * D)
        assert ph.make_myelin_substep(cell, dt).n_cand == _candidates(x, 16) == 8
        assert _candidates(LONGEST_STEP * x, 16) == at_longest


# ── the sizing of the two classes ───────────────────────────────────────────────────────────────────────────────────

def test_the_sheathed_class_is_sized_for_sqrt_6_D_dt_and_the_plain_class_for_the_longest_draw():
    D32 = float(np.float32(D))                                   # the stored diffusivity: the kernel reads it
    sheathed, mixed = _cell([G_RATIO] * 4), _cell([G_RATIO, 1.0, G_RATIO, 1.0])
    for dt, budgets in ((1e-6, (5, 6)), (4e-6, (8, 11))):
        step = math.sqrt(6.0 * D32 * dt)
        for cell, factor, budget in ((sheathed, 1.0, budgets[0]), (mixed, LONGEST_STEP, budgets[1])):
            size = concentric_step_max(cell._D_intra_jax, cell._D_myelin_jax, cell._D_extra_jax, dt, bool(cell.has_plain))
            assert abs(size / (factor * step) - 1.0) < 1e-12
            assert ph.make_myelin_substep(cell, dt).bounce_loop.budget == budget
