"""Two float32 boundary cases of the mesh wall, each a correction or an undecided label and never a refusal (#630).

1. A path that ENDS on a facet plane -- within the tie band float32 cannot resolve the side in -- is a hit: the walker
   is reflected and nudged off on its own side. Left on the plane it is frozen: an inward ray meets the plane at
   t <= 0 (no hit) and lands inside, an outward one meets it at a tiny t > 0 from behind and is nudged inward, and the
   escape guard refuses both, for the rest of the walk (0.2 % of the CACTUS bundle's extra-axonal walkers).
2. A label is decided only where the gather certifies the closest triangle: within the distance from the point to the
   edge of its 3x3x3 block of cells, every nearer triangle is in the gather. Beyond it the closest GATHERED triangle
   need not be the mesh's, and walkers 20-550 nm from every surface of the CACTUS bundle were refused step after step
   on its side.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from dmipy_sim.geometry.mesh import (Mesh, _closest_on_triangles, _gather_arr, _label_batch, _populated_batch,
                                     _wrap_arr)

from tests._containment import inside as contains

UM = 1e-6


def _sphere_pair():
    parts = []
    for cx in (-1.05, 1.05):
        m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        m.apply_translation([cx, 0.0, 0.0])
        parts.append(m)
    soup = trimesh.util.concatenate(parts)
    lo, hi = np.array([-2.5, -1.5, -1.5]), np.array([2.5, 1.5, 1.5])
    mesh = Mesh(np.asarray(soup.vertices) * UM, np.asarray(soup.faces), periodic=False, voxel_min=lo * UM,
                voxel_max=hi * UM, feature_radius=0.1 * UM, pool="extra", box_reflect=True)
    return mesh, soup, parts, lo, hi


def _end_on_a_facet(mesh, face):
    """A start 30 nm outside facet ``face`` (within a lookup cell) and the step that ends on its centroid."""
    tri = np.asarray(mesh._TRIS)[face].astype(np.float64)
    n = np.asarray(mesh._NRM)[face].astype(np.float64)
    c = tri.mean(0)
    r0 = c + 0.03 * UM * n
    return jnp.asarray(r0, jnp.float32), jnp.asarray(c - r0, jnp.float32), c, n


def _side(r, c, n):
    return float(np.dot(np.asarray(r, np.float64) - c, n))


def test_a_path_ending_on_a_facet_is_nudged_off_on_its_own_side():
    mesh, *_ = _sphere_pair()
    for face in range(0, 1280, 64):
        r0, step, c, n = _end_on_a_facet(mesh, face)
        hit = mesh.interact(r0, step, key=jax.random.PRNGKey(face))
        assert not bool(hit.illegal)
        assert _side(hit.r, c, n) > 0.5 * float(mesh._NUDGE), "the walker was left on the plane"


def test_without_the_tie_band_the_walker_is_left_on_the_plane(monkeypatch):
    """The self-guard: with the band closed, a step ending on a facet leaves the walker on the plane whenever float32
    puts the end on the near side (otherwise the facet is hit as an ordinary crossing). On the CUDA kernel a walker
    left there was refused every later step (the CACTUS trace, dmipy-sim#630); here the observable is the position."""
    mesh, *_ = _sphere_pair()
    monkeypatch.setattr(mesh, "_TIE", jnp.float32(0.0))
    left = 0
    for face in range(0, 1280, 64):
        r0, step, c, n = _end_on_a_facet(mesh, face)
        left += abs(_side(mesh.interact(r0, step, key=jax.random.PRNGKey(face)).r, c, n)) < 4.0 * float(mesh._EPS)
    assert left > 0, "no step ended on the plane: the failure this test guards against no longer occurs"


def test_a_decided_label_reads_the_closest_triangle_of_the_whole_mesh():
    mesh, soup, parts, lo, hi = _sphere_pair()
    P = np.random.default_rng(0).uniform(lo, hi, (3000, 3))
    Q = jnp.asarray(P * UM, jnp.float32)
    lab, decides = (np.asarray(a) for a in _label_batch(mesh._A, Q))
    populated = np.asarray(_populated_batch(mesh._A, Q))
    T = jnp.asarray(mesh._TRIS)
    d_all = np.array([float(jnp.min(_closest_on_triangles(T, q)[0])) for q in Q])
    d_gather = []
    for q in Q:
        ci, valid = _gather_arr(mesh._A, _wrap_arr(mesh._A, q))
        d2, _ = _closest_on_triangles(mesh._TRIS[ci], q)
        d_gather.append(float(jnp.min(jnp.where(valid, d2, jnp.inf))))
    d_gather = np.array(d_gather)
    np.testing.assert_allclose(d_gather[decides], d_all[decides], rtol=1e-5)
    truth = np.zeros(len(P), bool)
    for m in parts:
        truth |= contains(m, P)
    assert np.all(lab[decides].astype(bool) == truth[decides])
    # the self-guard: a populated gather whose closest triangle is not the mesh's exists and is left undecided
    assert np.any(populated & ~decides & (d_gather > d_all * (1 + 1e-5)))
