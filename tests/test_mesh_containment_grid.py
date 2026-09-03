"""Grid-accelerated point-in-mesh containment: exactness first, speed second.

The containment test decides which compartment a walker is seeded into, so a flipped bit here is
not a small error -- it puts a walker in the wrong tissue for the whole walk, and after the fact
that is indistinguishable from a wall leak. These tests hold the fast path to the same answers as
the ray-engine oracle, and to ANALYTIC truth where the oracle itself is unreliable.
"""
import numpy as np
import pytest
import trimesh

from dmipy_sim.susceptibility_field import mesh_contains, mesh_contains_fast, mesh_inside


def _prims():
    return [
        ("icosphere", trimesh.creation.icosphere(subdivisions=4, radius=1.0)),
        ("box", trimesh.creation.box(extents=(1.0, 2.0, 3.0))),
        ("torus", trimesh.creation.torus(major_radius=1.0, minor_radius=0.35)),
        ("two_bodies", trimesh.util.concatenate([
            trimesh.creation.icosphere(subdivisions=3).apply_translation([-1.2, 0, 0]),
            trimesh.creation.icosphere(subdivisions=3).apply_translation([1.2, 0, 0])])),
    ]


@pytest.mark.parametrize("name,mesh", _prims(), ids=[n for n, _ in _prims()])
def test_it_agrees_with_the_ray_engine_on_closed_primitives(name, mesh):
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, np.int64)
    lo, hi = V.min(0) - 0.2, V.max(0) + 0.2
    pts = np.random.default_rng(0).uniform(lo, hi, (3000, 3))
    assert np.array_equal(mesh_contains_fast(V, F, pts), mesh.contains(pts))


def test_it_stays_exact_where_the_ray_engine_itself_degrades():
    """Substrate coordinates are in METRES, and the NumPy ray engine loses accuracy there.

    Measured on a sphere of radius 1e-5 m: trimesh agrees with analytic truth only 97.6% of the
    time, while the grid path is exact. So on real substrate scales the oracle is the weaker of
    the two, and validating against it unscaled would have condemned the correct implementation.
    """
    for R in (1.0, 1e-3, 1e-5):
        m = trimesh.creation.icosphere(subdivisions=4, radius=R)
        V = np.asarray(m.vertices, float); F = np.asarray(m.faces, np.int64)
        p = np.random.default_rng(1).uniform(-1.3 * R, 1.3 * R, (3000, 3))
        r = np.linalg.norm(p, axis=1)
        clear = np.abs(r - R) > 0.02 * R            # away from the facetted surface
        truth = r < R
        assert np.array_equal(mesh_contains_fast(V, F, p)[clear], truth[clear]), f"R={R}"


def test_it_has_no_false_outside_on_a_dense_multi_body_mesh():
    """The property `mesh_inside` lacks, and the reason #62 turned the prefilter off.

    A false-OUTSIDE is the dangerous direction: the cascade used `mesh_inside` as an upstream
    proposal gate, so any interior point it missed was never ray cast and stayed 'outside'. That
    seeded 3.8% of a nominally extra-axonal pool inside fibres. The grid path must never do it.
    """
    rng = np.random.default_rng(2)
    parts = [trimesh.creation.icosphere(subdivisions=3, radius=0.30).apply_translation(c)
             for c in rng.uniform(-1, 1, (40, 3))]
    m = trimesh.util.concatenate(parts)
    V = np.asarray(m.vertices, float); F = np.asarray(m.faces, np.int64)
    pts = rng.uniform(V.min(0), V.max(0), (4000, 3))
    truth = m.contains(pts)
    fast = mesh_contains_fast(V, F, pts)
    false_outside = int((truth & ~fast).sum())
    assert false_outside == 0, f"{false_outside} genuinely-inside points reported outside"
    # and the fast heuristic really does have the flaw, so the test is not vacuous
    assert int((truth & ~np.asarray(mesh_inside(V, F, pts))).sum()) > 0


def test_the_default_path_is_the_grid_and_trimesh_stays_reachable():
    m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    V = np.asarray(m.vertices, float); F = np.asarray(m.faces, np.int64)
    pts = np.random.default_rng(3).uniform(-1.2, 1.2, (600, 3))
    assert np.array_equal(mesh_contains(V, F, pts), mesh_contains_fast(V, F, pts))
    assert np.array_equal(mesh_contains(V, F, pts, method="trimesh"), m.contains(pts))
    with pytest.raises(ValueError, match="'grid' or 'trimesh'"):
        mesh_contains(V, F, pts, method="embree")


def test_empty_input_is_handled():
    m = trimesh.creation.icosphere(subdivisions=2)
    V = np.asarray(m.vertices, float); F = np.asarray(m.faces, np.int64)
    out = mesh_contains_fast(V, F, np.zeros((0, 3)))
    assert out.shape == (0,) and out.dtype == bool


def test_cost_does_not_scale_with_the_triangle_count_per_point():
    """The whole point: each point tests the triangles over its own xy bin, not all of them.

    A brute-force engine costs O(points x triangles); refining the mesh 4x should then cost ~4x
    per point. The grid path should be far flatter than that.
    """
    import time
    def timed(subdiv, n=2000):
        m = trimesh.creation.icosphere(subdivisions=subdiv, radius=1.0)
        V = np.asarray(m.vertices, float); F = np.asarray(m.faces, np.int64)
        p = np.random.default_rng(4).uniform(-1.2, 1.2, (n, 3))
        mesh_contains_fast(V, F, p[:50])                       # warm
        t = time.time(); mesh_contains_fast(V, F, p); return time.time() - t, len(F)
    t_lo, n_lo = timed(3)
    t_hi, n_hi = timed(5)
    growth = t_hi / max(t_lo, 1e-6)
    assert growth < 0.5 * (n_hi / n_lo), (
        f"cost grew {growth:.1f}x for a {n_hi/n_lo:.0f}x triangle increase — that is "
        f"brute-force scaling, so the bin index is not doing its job")
