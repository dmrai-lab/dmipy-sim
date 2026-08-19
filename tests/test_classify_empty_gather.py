"""A point deep inside a thick, finely-meshed object has no wall in its 27-cell gather.

`_classify_arr` decides interior/exterior from the nearest triangle in a local gather, treating an EMPTY
gather as exterior. That guard is right for the case it was written for -- a thin structure in a big box,
where "no wall nearby" really does mean "outside". Its validity condition is on the OBJECT, though: every
genuine interior point must have a wall inside its gather.

The cell size scales with the TRIANGLE size, not the object size, so refining a mesh shrinks the gather
while the object stays as wide as it was. Measured on a real 366-strand axon bundle (median edge 0.371 um
-> cell 0.124 um -> reach ~0.19 um, lumen radii 1-2 um): 19.6% of genuinely interior points reported
exterior, 99.4% of them with an empty gather.

Seeding on that answer biases an intra pool towards the wall and lets deep intra points into an extra pool
-- silently, with a plausible-looking volume fraction either way.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from dmipy_sim.mesh import Mesh, _classify_arr, _gather_is_populated


def _thick_finely_meshed_tube(radius=3.0, height=20.0, sections=96, subdivisions=1):
    """A tube far wider than its own triangles -- the regime where the guard's assumption fails.

    `sections` sets the triangle size and therefore the gather reach; `radius` sets how deep the interior
    goes. Their RATIO is what breaks the classifier, which is why refining a mesh makes this worse rather
    than better.
    """
    m = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    for _ in range(subdivisions):
        m = m.subdivide()
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64), m


def _mesh_for(V, F):
    e = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
    lo, hi = V.min(0) - 2.0, V.max(0) + 2.0
    return Mesh(V, F, periodic=False, voxel_min=lo, voxel_max=hi,
                feature_radius=0.5 * float(np.median(e))), lo, hi


def test_deep_interior_points_have_an_empty_gather():
    """The mechanism, isolated: interior points beyond the gather's reach see no triangle at all."""
    V, F, tri = _thick_finely_meshed_tube()
    mesh, lo, hi = _mesh_for(V, F)

    P = np.random.default_rng(0).uniform(lo, hi, (3000, 3)).astype(np.float32)
    interior = tri.contains(P.astype(float))
    populated = np.asarray(jax.jit(jax.vmap(_gather_is_populated, in_axes=(None, 0)))(mesh._A, P))

    assert interior.sum() > 200, "test geometry must have a substantial interior"
    blind = interior & ~populated
    assert blind.sum() > 0, (
        "no interior point is beyond the gather here; the geometry no longer exercises the failure "
        "(make the tube wider or the mesh finer)")


def test_the_classifier_calls_those_points_exterior():
    """What the empty gather costs: genuine interior reported as outside, with nothing raised."""
    V, F, tri = _thick_finely_meshed_tube()
    mesh, lo, hi = _mesh_for(V, F)

    P = np.random.default_rng(1).uniform(lo, hi, (3000, 3)).astype(np.float32)
    interior = tri.contains(P.astype(float))
    lab = np.asarray(jax.jit(jax.vmap(_classify_arr, in_axes=(None, 0)))(mesh._A, P))

    missed = interior & (lab == 1)
    frac = missed.sum() / max(interior.sum(), 1)
    assert frac > 0.30, (
        f"only {100*frac:.1f}% of the interior misreported; the geometry has stopped exercising the "
        f"defect rather than the defect being fixed")

    # every miss is an empty gather, not a sidedness error -- the mechanism, pinned
    populated = np.asarray(jax.jit(jax.vmap(_gather_is_populated, in_axes=(None, 0)))(mesh._A, P))
    assert (missed & populated).sum() == 0, "a miss with a populated gather would be a different defect"


def test_refining_the_mesh_makes_the_classifier_worse():
    """The counter-intuitive signature, and the reason this is not a meshing-quality problem.

    Gather reach scales with the TRIANGLE size; the object does not shrink with it. So a finer mesh sees
    LESS of its own interior -- the opposite of every other accuracy knob here, and the reason "just mesh
    it better" is not the fix.
    """
    fracs = []
    for subdiv in (0, 1):
        V, F, tri = _thick_finely_meshed_tube(subdivisions=subdiv)
        mesh, lo, hi = _mesh_for(V, F)
        P = np.random.default_rng(2).uniform(lo, hi, (2000, 3)).astype(np.float32)
        interior = tri.contains(P.astype(float))
        lab = np.asarray(jax.jit(jax.vmap(_classify_arr, in_axes=(None, 0)))(mesh._A, P))
        fracs.append((interior & (lab == 1)).sum() / max(interior.sum(), 1))
    assert fracs[1] > fracs[0], (
        f"refining should widen the blind interior, got {100*fracs[0]:.1f}% -> {100*fracs[1]:.1f}%")


def test_seeding_fills_the_lumen_rather_than_hugging_the_wall():
    """The consequence that matters, and the fix.

    Rejection-seeding on the raw classifier accepts only points close enough to a wall to see one, so the
    pool collects in a shell instead of filling the object. Seeds must reach the centre.
    """
    V, F, tri = _thick_finely_meshed_tube()
    mesh, _, _ = _mesh_for(V, F)

    pts = np.asarray(mesh.init_positions(1500, jax.random.PRNGKey(3), intra=True), float)
    # exact, not "mostly": seeding decides containment by ray parity, so a seed outside is a bug
    assert tri.contains(pts).all(), (
        f"{100*(~tri.contains(pts)).mean():.2f}% of the intra pool is outside the tube")

    # radial distance from the tube axis, in units of the radius
    frac = np.linalg.norm(pts[:, :2], axis=1) / 3.0
    # a uniformly filled disc has median r/R = 1/sqrt(2) ~ 0.707; a wall-hugging shell sits far above it
    assert np.median(frac) < 0.80, (
        f"median radial position {np.median(frac):.3f} R -- seeds are collecting near the wall instead of "
        f"filling the lumen (uniform fill gives ~0.707)")
    assert (frac < 0.5).mean() > 0.15, "almost nothing seeded in the inner half of the tube"


def test_exterior_seeding_does_not_swallow_the_interior():
    """The mirror failure: deep interior points read as exterior, so an 'outside' pool absorbs them."""
    V, F, tri = _thick_finely_meshed_tube()
    mesh, _, _ = _mesh_for(V, F)

    pts = np.asarray(mesh.init_positions(1500, jax.random.PRNGKey(4), intra=False), float)
    assert not tri.contains(pts).any(), (
        f"{100*tri.contains(pts).mean():.1f}% of the exterior pool is actually inside the tube")


def test_seeding_is_exact_at_every_grid_resolution():
    """The complementary failure: the points the classifier is TRUSTED on are the ones it judges worst.

    The empty-gather case above is a point too deep to see a wall. This is the opposite end -- a point
    near enough that its gather IS populated, where the classifier's verdict was taken as final. It
    decides sidedness from the nearest triangle CENTROID, which is a poor stand-in for the surface when
    triangles are large, so the error grows as the mesh gets coarser: seeding "intra" on a closed
    cylinder put 6.27% of the pool outside at 384 triangles, 1.13% at 1536 and 0.73% at 6144.

    Grid resolution is a speed knob. It must not decide whether a seed is inside its own surface.
    """
    for subdivisions in (0, 1, 2):
        V, F, tri = _thick_finely_meshed_tube(height=24.0, subdivisions=subdivisions)
        mesh, _, _ = _mesh_for(V, F)
        pts = np.asarray(mesh.init_positions(600, jax.random.PRNGKey(5), intra=True), float)
        outside = ~tri.contains(pts)
        assert not outside.any(), (
            f"subdivision {subdivisions} ({len(F)} triangles): {100*outside.mean():.2f}% of the intra "
            f"pool was seeded OUTSIDE the surface")


# ---------------------------------------------------------------------------
# The walk's consumer of the same answer: reject_escape
# ---------------------------------------------------------------------------

def test_reject_escape_does_not_discard_steps_inside_the_lumen():
    """A step between two genuinely interior points must survive, however deep it goes.

    `reject_escape` is the impermeable-leak net, and it fired on a label *change*. Deep interior reads
    exterior here, so a step crossing the gather frontier -- with no wall within reach of either end --
    looked like a wall crossing and was discarded in both directions. That seals the near-wall shell off
    from the bulk, and since the boundary local time is generated entirely within one step of the surface,
    it starves the channel that surface relaxation and MT binding both read: a mesh sphere reached 67% of
    the analytic local time and an MT bound fraction of 0.2296 against an analytic 0.3333.
    """
    V, F, tri = _thick_finely_meshed_tube()
    mesh, lo, hi = _mesh_for(V, F)

    rng = np.random.default_rng(1)
    P = rng.uniform(lo, hi, (4000, 3))
    interior = tri.contains(P)
    populated = np.asarray(jax.jit(jax.vmap(_gather_is_populated, in_axes=(None, 0)))(
        mesh._A, P.astype(np.float32)))

    # near-wall interior -> deep interior: the frontier crossing, both ends genuinely inside
    near = P[interior & populated]
    deep = P[interior & ~populated]
    assert len(near) > 20 and len(deep) > 20, "geometry must supply both near-wall and deep interior"
    n = min(len(near), len(deep), 200)
    esc = np.asarray(jax.jit(jax.vmap(mesh._escaped))(
        near[:n].astype(np.float32), deep[:n].astype(np.float32)))
    assert not esc.any(), (
        f"{esc.sum()}/{n} steps between two interior points rejected as escapes")


def test_reject_escape_still_catches_a_wall_crossing():
    """The net must survive the fix: an interior->exterior step across the wall is still an escape.

    Gating on decidability necessarily narrows the guard, so this pins the half that has to keep working.
    A walker that has just tunnelled is within one step of the wall it passed through, so its gather is
    populated and the label comparison is still meaningful.
    """
    V, F, tri = _thick_finely_meshed_tube()
    mesh, lo, hi = _mesh_for(V, F)

    # radial pairs straddling the wall, well inside the tube's height so the caps are not involved
    rng = np.random.default_rng(2)
    th = rng.uniform(0, 2 * np.pi, 200)
    z = rng.uniform(-5.0, 5.0, 200)
    radial = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], axis=1)
    inside = radial * 2.7 + np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    outside = radial * 3.3 + np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    assert tri.contains(inside).all() and not tri.contains(outside).any(), "pairs must straddle the wall"

    esc = np.asarray(jax.jit(jax.vmap(mesh._escaped))(
        inside.astype(np.float32), outside.astype(np.float32)))
    assert esc.all(), f"only {esc.sum()}/{len(esc)} genuine wall crossings flagged as escapes"
