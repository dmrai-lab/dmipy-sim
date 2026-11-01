"""The acceleration grid must be a speed knob, not a physics knob.

`feature_radius` sets the cell size of the triangle-lookup grid. Nothing about the physics depends on it:
the same closed surface confines the same walkers whatever resolution we happen to index it at. So a step must
end where it ends however the surface is indexed, and no step may leave an impermeable mesh at any setting.

This is the property disimpy's `test_mesh_diffusion` samples through the signal (it sweeps `n_sv` over `[1,1,1]`,
`[1,5,20]`, `[10,10,10]` -- including brute force with no acceleration at all -- and checks the signal against MISST
in every case). It is held here on the wall interaction itself, `Mesh.interact`, which is the whole of what the grid
touches: a walk is a sequence of these steps, so a grid that changes the signal changes some step first, and the
step shows it exactly where the signal can only show it as scatter on a few hundred walkers.

**These run with `reject_escape` disabled, deliberately.** That guard rejects a step whose start and end
classify differently, and it is a net, not a collision test: it cannot see an escape when both ends of the
step read the same, and rejecting a step freezes the walker for that interval, which is its own bias. What
has to be correct underneath is the collision response, so that is what is measured.

The surface is identical at every subdivision (linear subdivision adds triangles, not curvature), so one set of steps is
valid for all three arms and the grid resolution is the only variable. The steps start within two step lengths of a
wall -- the side of the cylinder and its caps -- since a step that meets no wall is the same free flight on any index.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from dmipy_sim.engine.physics import resolve_sub_steps
from dmipy_sim.geometry.mesh import Mesh

from tests._containment import inside as contains

UM = 1e-6
D = 1.0e-9
DT = 1.018e-4                     # s: the waveform step of the walk these arms were first measured on
SUBDIVISIONS = (0, 1, 2)          # same surface, 384 -> 1536 -> 6144 triangles
RADIUS, HALF_HEIGHT = 3.0, 12.0   # um
# The step every arm can take: the engine's own sub-step is 0.276, 0.145 and 0.136 um at the three settings
# (8, 29 and 33 sub-steps of the 0.781 um step), so the shortest is within every arm's collision budget.
STEP_UM = 0.136
# A walker parked against the wall is a nudge (1e-4 of a step) off it and has not escaped; one that has is a step
# or more clear of it. Judge by depth, with a tolerance well above the nudge and well below one step.
ESCAPE_TOL_UM = 1e-3
N_STEPS = 3_000       # per arm: zero escapes bound the escape probability of a near-wall step below 0.15 % at 99 %
# The edge rule backscatters a hit within 1e-4 (barycentric) of a seam (MC/DC's rule, `edge_margin`), and a finer index has
# more seams: 12 % of the steps bounce and 6e-4 of the bounces land within the margin, so a pair of arms differs on 0.4 of
# the 3,000 steps in the mean. At most 3 may: the Poisson quantile of that mean at 1e-3.
MAX_DIVERGENT = 3
CHUNK = 1_000         # steps per device call: the 27-cell gather of every step is held at once
SEED = 20260816


def _surface(subdivisions):
    """One closed cylinder, at unit scale.

    Built at unit scale and converted by the caller: trimesh unitizes against an absolute tolerance, so a
    cylinder constructed directly in metres comes back degenerate.
    """
    m = trimesh.creation.cylinder(radius=RADIUS, height=2 * HALF_HEIGHT, sections=96)
    for _ in range(subdivisions):
        m = m.subdivide()
    return m


def _arm(subdivisions):
    tri = _surface(subdivisions)
    V = np.asarray(tri.vertices, float) * UM
    F = np.asarray(tri.faces, np.int64)
    e = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
    mesh = Mesh(V, F, periodic=False, voxel_min=V.min(0) - 2 * UM, voxel_max=V.max(0) + 2 * UM,
                feature_radius=0.5 * float(np.median(e)),
                reject_escape=False)  # measure the collision response, not the safety net
    return tri, mesh


@pytest.fixture(scope="module")
def arms():
    return {s: _arm(s) for s in SUBDIVISIONS}


def _near_wall_starts(base, step_um, n, rng):
    """``n`` points of the cylinder within two steps of its side or of a cap, half of each."""
    band = 2 * step_um
    r = np.sqrt(rng.uniform(size=3 * n) * (RADIUS ** 2 - (RADIUS - band) ** 2) + (RADIUS - band) ** 2)
    th = rng.uniform(0, 2 * np.pi, 3 * n)
    z = rng.uniform(-HALF_HEIGHT + band, HALF_HEIGHT - band, 3 * n)
    side = np.stack([r * np.cos(th), r * np.sin(th), z], axis=1)
    rc = RADIUS * np.sqrt(rng.uniform(size=3 * n))
    zc = np.sign(rng.uniform(-1, 1, 3 * n)) * (HALF_HEIGHT - rng.uniform(0, band, 3 * n))
    cap = np.stack([rc * np.cos(th), rc * np.sin(th), zc], axis=1)
    pts = np.concatenate([side[contains(base, side)][: n // 2], cap[contains(base, cap)][: n - n // 2]])
    assert len(pts) == n, "the sampler under-filled"
    return pts


def _steps(n, step_um, rng):
    d = rng.normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1)[:, None] * step_um


def _fire(mesh, starts_um, steps_um):
    """The endpoints (um) of the steps ``steps_um`` from ``starts_um``: the wall interaction, chunk by chunk."""
    f = jax.jit(jax.vmap(lambda p, q: mesh.interact(p, q).r))
    starts = jnp.asarray(starts_um * UM, jnp.float32)
    steps = jnp.asarray(steps_um * UM, jnp.float32)
    return np.concatenate([np.asarray(f(starts[i:i + CHUNK], steps[i:i + CHUNK])) for i in range(0, len(starts), CHUNK)]) / UM


def test_a_step_ends_at_the_same_point_at_every_grid_resolution(arms):
    """The same steps from the same points end at the same points when only the indexing resolution differs, so the
    answer of a walk must not move either. Refining a mesh is the natural response to a suspicious result; it must
    not change the physics.

    Claim: the endpoints of a step are those of the coarsest index, to float32 rounding (1e-4 um = 1e-3 of the
    step), at the two finer ones, but for the rare step the edge rule sends elsewhere (``MAX_DIVERGENT`` of the
    ``N_STEPS``, never by more than the step). 12 % of the steps bounce off a wall (measured), so the claim is held on
    collisions, not on free flights. The steps are exact, not sampled: there is no noise."""
    rng = np.random.default_rng(SEED)
    starts = _near_wall_starts(arms[0][0], STEP_UM, N_STEPS, rng)
    steps = _steps(N_STEPS, STEP_UM, rng)
    ends = {s: _fire(arms[s][1], starts, steps) for s in SUBDIVISIONS}
    bounced = np.linalg.norm(ends[0] - starts, axis=1) < STEP_UM * (1 - 1e-3)
    assert bounced.mean() > 0.05, "the table does not reach a wall: nothing is held"
    for s in SUBDIVISIONS[1:]:
        gap = np.linalg.norm(ends[s] - ends[0], axis=1)
        elsewhere = int((gap >= 1e-4).sum())
        assert elsewhere <= MAX_DIVERGENT and gap.max() < STEP_UM, (
            f"{elsewhere} of {N_STEPS} steps end elsewhere at subdivision {s} than at 0 "
            f"(largest {gap.max():.4f} um, {(gap[bounced] >= 1e-4).sum()} of them bounces)")


def test_an_impermeable_mesh_confines_its_steps_at_every_grid_resolution(arms):
    """A closed surface must confine, whatever resolution the triangle index happens to use.

    Every step starts verified-inside, so anything outside at the end crossed a wall it should have
    bounced off. Each arm takes the engine's own sub-step for it (0.28, 0.15 and 0.14 um); before the collision
    fix this arm lost 55% of its walkers at the coarsest setting and 90% at the finest.

    Claim: a near-wall sub-step leaves an impermeable mesh with probability below 0.15 %: of ``N_STEPS`` steps none
    outside miss a leak of that rate with probability ``(1 - 0.0015)^3000`` = 1 %. The collision fix's residual
    (dmrai-lab/dmipy-sim#51) is 0.02 % of walkers over a whole walk of thousands of such steps, orders of magnitude
    below this bound; the walk's accumulation of it is not asserted."""
    rng = np.random.default_rng(SEED + 1)
    base = arms[0][0]
    for s in SUBDIVISIONS:
        tri, mesh = arms[s]
        step_um = float(np.sqrt(6 * D * DT / resolve_sub_steps(mesh, D, DT))) / UM
        starts = _near_wall_starts(base, step_um, N_STEPS, rng)
        ends = _fire(mesh, starts, _steps(N_STEPS, step_um, rng))
        depth = -trimesh.proximity.signed_distance(tri, ends)          # positive = outside
        escaped = int((depth > ESCAPE_TOL_UM).sum())
        assert escaped == 0, (f"subdivision {s}: {escaped} of {N_STEPS} near-wall steps of {step_um:.3f} um ended outside "
                              f"the closed surface (deepest {depth.max():.3f} um)")
