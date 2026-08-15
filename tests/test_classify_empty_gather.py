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
    assert tri.contains(pts).mean() > 0.97, "seeds must actually be inside the tube"

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
    assert tri.contains(pts).mean() < 0.02, (
        f"{100*tri.contains(pts).mean():.1f}% of the exterior pool is actually inside the tube")
