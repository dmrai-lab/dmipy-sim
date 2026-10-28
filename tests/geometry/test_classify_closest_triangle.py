"""The mesh classifier reads the side of the CLOSEST triangle, not of the triangle with the nearest centroid.

Two closed tubes a few nanometres apart, meshed with triangles much wider than the gap: a point inside one tube
near the face it turns to the other is often nearer to a centroid of the OTHER tube, whose normal then calls it
outside (the 344-strand CACTUS bundle: 0.2 % of the points with a wall in reach, 0.3 % of its extra-axonal space
called interior, dmipy-sim#711). The closest point of a closed surface sees the surface's own side, so
`_classify_arr` errs on none of them; the nearest-centroid rule, computed here beside it, is the self-guard that
the geometry provokes the failure at all.
"""
from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from dmipy_sim.geometry.mesh import Mesh, _classify_batch, _populated_batch, _closest_on_triangles

from tests._containment import inside as contains


def _two_close_tubes(radius=1.0, gap=0.02, height=6.0, sections=10):
    """Two coarse tubes along z, ``gap`` apart, as one surface soup (and their union for the exact test)."""
    parts = []
    for cx in (-(radius + gap / 2), radius + gap / 2):
        m = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
        m.apply_translation([cx, 0.0, 0.0])
        parts.append(m)
    soup = trimesh.util.concatenate(parts)
    return soup, parts


def _labels(scale=1e-6, n=40_000, seed=0):
    soup, parts = _two_close_tubes()
    V = np.asarray(soup.vertices, float) * scale
    F = np.asarray(soup.faces, np.int64)
    lo, hi = np.array([-2.5, -1.5, -2.5]), np.array([2.5, 1.5, 2.5])
    mesh = Mesh(V, F, periodic=False, voxel_min=lo * scale, voxel_max=hi * scale, feature_radius=0.3 * scale,
                box_reflect=True)
    P = np.random.default_rng(seed).uniform(lo, hi, (n, 3))
    truth = contains(parts[0], P) | contains(parts[1], P)
    pj = np.asarray(P * scale, np.float32)
    lab = np.asarray(_classify_batch(mesh._A, pj)).astype(bool)
    pop = np.asarray(_populated_batch(mesh._A, pj))
    # the nearest-centroid rule over the same gather, for the self-guard
    tri = np.asarray(soup.triangles, float)
    cent, nrm = tri.mean(1), np.asarray(soup.face_normals, float)
    reach = 1.5 * float(mesh.cell_size) / scale
    near = []
    for p in P[pop]:
        k = np.flatnonzero(np.all(np.abs(cent - p) < 3 * reach, axis=1))
        j = k[np.argmin(((cent[k] - p) ** 2).sum(1))]
        near.append(np.dot(p - cent[j], nrm[j]) < 0)
    return truth, lab, pop, np.asarray(near, bool)


def test_the_closest_triangle_classifies_every_point_with_a_wall_in_reach():
    truth, lab, pop, _ = _labels()
    assert pop.sum() > 10_000
    wrong = (lab != truth) & pop
    assert wrong.sum() == 0, f"{wrong.sum()} of {pop.sum()} points with a wall in reach classified on the wrong side"


def test_the_nearest_centroid_would_not():
    truth, _, pop, near = _labels()
    assert (near != truth[pop]).sum() > 0, "the geometry no longer provokes the nearest-centroid failure"


def test_closest_point_is_the_exact_point_triangle_distance():
    """Against trimesh's closest point (in um: its tolerances are absolute) on random points and triangles."""
    rng = np.random.default_rng(1)
    T = rng.normal(size=(500, 3, 3))
    p = rng.normal(size=3) * 2.0
    d2, c = _closest_on_triangles(np.asarray(T, np.float32), np.asarray(p, np.float32))
    ref = trimesh.triangles.closest_point(T, np.repeat(p[None], len(T), 0))
    np.testing.assert_allclose(np.asarray(c), ref, atol=2e-5)
    np.testing.assert_allclose(np.asarray(d2), ((ref - p) ** 2).sum(1), rtol=1e-4, atol=1e-6)
