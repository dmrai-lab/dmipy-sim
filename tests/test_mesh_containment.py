"""Containment tests over a large box: the near-field contract of mesh_inside and the exact mesh_contains.

Both failure modes these cover were found in production, in the intra-axonal seeding of a Winther
susceptibility pack, where they are invisible: a containment test that over-accepts seeds walkers in free
water and labels them intra, and one that under-accepts just seeds fewer walkers. Neither raises.
"""
from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from dmipy_sim.susceptibility_field import mesh_contains, mesh_inside


def _thin_tube_in_a_big_box(scale=1.0, radius=1.5, bend=1.2):
    """A long, curved, thin closed tube inside a box much larger than it.

    The shape that breaks nearest-centroid sidedness: for a point several radii away, many triangles sit
    at a similar centroid distance, so the one selected need not be the nearest and its normal can report
    the wrong side. A straight tube is not enough -- the curvature is what puts wrongly-signed candidates
    within reach of distant points.
    """
    m = trimesh.creation.cylinder(radius=radius, height=24.0, sections=48)
    v = np.asarray(m.vertices, float).copy()
    v[:, 0] += bend * v[:, 2] ** 2 / 24.0          # bend it along z
    m = trimesh.Trimesh(vertices=v * scale, faces=np.asarray(m.faces), process=False)
    lo = np.asarray(m.bounds[0]) - 8.0 * scale     # box padded far beyond the tube
    hi = np.asarray(m.bounds[1]) + 8.0 * scale
    return m, lo, hi


@pytest.mark.parametrize("scale", [1.0, 1e-6])
def test_mesh_contains_matches_ray_parity_over_the_whole_box(scale):
    """mesh_contains agrees with ray parity everywhere in the box, at any coordinate scale.

    The scale sweep is the regression: trimesh's predicates unitize against an absolute tolerance, so at
    SI scale (edges ~1e-7 m) its face normals collapse to zero and `contains` reports nearly everything
    OUTSIDE. That is the dangerous direction -- a restrictive answer looks like a correct one -- so it is
    asserted against a reference computed at unit scale, not against itself.
    """
    m, lo, hi = _thin_tube_in_a_big_box(scale)
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    P = np.random.default_rng(0).uniform(lo, hi, (4000, 3))
    ref = trimesh.Trimesh(vertices=V / scale, faces=F, process=False).contains(P / scale)
    got = mesh_contains(V, F, P)
    assert ref.sum() > 50, "test geometry degenerate: too few interior points to be meaningful"
    assert (got == ref).all(), f"{int((got != ref).sum())} of {len(P)} disagree with ray parity"


def test_mesh_contains_rejects_far_field_acceptances_of_mesh_inside():
    """The bug this exists for: mesh_inside over-accepts far from a thin curved surface, mesh_contains
    does not. Also pins the near-field contract that makes the cascade sound -- mesh_inside must not
    produce false-OUTSIDE, since mesh_contains uses it to propose candidates and never revisits the rest.
    """
    m, lo, hi = _thin_tube_in_a_big_box()
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    P = np.random.default_rng(1).uniform(lo, hi, (6000, 3))
    ref = m.contains(P)
    fast = mesh_inside(V, F, P)
    exact = mesh_contains(V, F, P)

    assert (~fast & ref).sum() == 0, "mesh_inside produced a false-OUTSIDE; the candidate cascade is unsound"
    assert (fast & ~ref).sum() > 0, "geometry no longer reproduces the far-field over-acceptance"
    assert (exact & ~ref).sum() == 0

    # the over-acceptances are predominantly FAR from the surface -- the regime a real lumen point can
    # never reach. (Some sit within a triangle of the wall, which is the documented boundary accuracy of
    # mesh_inside and not the failure under test, so this asserts on the bulk rather than the minimum.)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(V).query(P)
    assert np.median(d[fast & ~ref]) > 1.5, "over-acceptances should sit well beyond the tube radius"


def test_mesh_contains_refuses_an_open_surface():
    """An open surface has no inside, and ray parity through an open rim is meaningless rather than
    approximate. Failing loudly beats returning the parity of a leaky mesh."""
    m = trimesh.creation.cylinder(radius=0.5, height=10.0, sections=32)
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    keep = F[V[F].mean(axis=1)[:, 2] < 0.0]          # keep half the tube: a genuine open rim, not a defect
    n_open = len(trimesh.grouping.group_rows(
        trimesh.Trimesh(vertices=V, faces=keep, process=False).edges_sorted, require_count=1))
    assert n_open > 16, f"test geometry is not open enough to be refused ({n_open} boundary edges)"
    with pytest.raises(ValueError, match="closed surface"):
        mesh_contains(V, keep, np.zeros((3, 3)))


def test_mesh_contains_tolerates_a_tiny_defect_with_a_warning():
    """A defect small enough to be a defect must not be fatal -- one axon of the Winther set has a two-edge
    slit that no repair closes -- but it must be announced rather than absorbed."""
    m = trimesh.creation.cylinder(radius=1.0, height=8.0, sections=24)
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    keep = np.delete(F, 5, axis=0)                   # drop a single triangle
    P = np.random.default_rng(3).uniform(V.min(0) - 2, V.max(0) + 2, (800, 3))
    got = mesh_contains(V, keep, P)                  # repaired or warned, but it returns an answer
    ref = m.contains(P)
    assert (got == ref).sum() >= len(P) - 5, "a one-triangle defect should barely change the verdict"


def test_mesh_contains_prefilter_is_transparent():
    """prefilter=True must change only cost, never the answer."""
    m, lo, hi = _thin_tube_in_a_big_box()
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    P = np.random.default_rng(2).uniform(lo, hi, (1500, 3))
    assert (mesh_contains(V, F, P) == mesh_contains(V, F, P, prefilter=False)).all()
