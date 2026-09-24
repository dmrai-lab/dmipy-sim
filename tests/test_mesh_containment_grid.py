"""Grid-accelerated point-in-mesh containment: exactness first, speed second.

The containment test decides which compartment a walker is seeded into, so a flipped bit here is
not a small error -- it puts a walker in the wrong tissue for the whole walk, and after the fact
that is indistinguishable from a wall leak. These tests hold the fast path to the same answers as
the ray-engine oracle, and to ANALYTIC truth where the oracle itself is unreliable.
"""
import numpy as np
import pytest

# Guarded like every other mesh module in the suite: trimesh ships in the [dev] extra, but a
# bare top-level import turns its absence into a COLLECTION ERROR, which pytest treats as fatal
# and which therefore takes down the whole run -- not just this file. importorskip degrades to
# a skip instead (#91).
trimesh = pytest.importorskip("trimesh")

from dmipy_sim.fields.susceptibility_field import mesh_contains, mesh_contains_fast, mesh_inside


# Built on FIRST USE, never at import/collection time. As a `parametrize` argument this ran at
# collection -- and because the decorator called it twice (once for the params, once for the
# ids) it built all four primitives TWICE, subdivision-4 icosphere included, before a single
# test ran. Every pytest invocation paid that, including runs that deselect this module (#91).
_PRIMS = {}


def _prim(name):
    if not _PRIMS:
        _PRIMS.update({
            "icosphere": trimesh.creation.icosphere(subdivisions=4, radius=1.0),
            "box": trimesh.creation.box(extents=(1.0, 2.0, 3.0)),
            "torus": trimesh.creation.torus(major_radius=1.0, minor_radius=0.35),
            "two_bodies": trimesh.util.concatenate([
                trimesh.creation.icosphere(subdivisions=3).apply_translation([-1.2, 0, 0]),
                trimesh.creation.icosphere(subdivisions=3).apply_translation([1.2, 0, 0])]),
        })
    return _PRIMS[name]


@pytest.mark.parametrize("name", ["icosphere", "box", "torus", "two_bodies"])
def test_it_agrees_with_the_ray_engine_on_closed_primitives(name):
    mesh = _prim(name)
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
    # subdivisions=2, not 3. Measured: the reference (`trimesh.contains`) is 80.2 s of this
    # test's 85 s at subdivisions=3 and 30.8 s at 2, while the code under test is 1.35 s and
    # 0.22 s -- the third-party ray caster dominated a test that is not about it. Coarsening
    # the spheres changes neither the arrangement nor the point count, and the flaw this
    # test must keep detecting is still found 210 times (222 at subdivisions=3).
    #
    # Two cheaper ideas were tried and REJECTED, both because they removed the coverage:
    #   * analytic ground truth (|p - centre| < inradius) instead of `trimesh.contains` --
    #     the bodies OVERLAP, and `concatenate` does not boolean them, so ray parity counts
    #     crossings through both surfaces and disagrees with an analytic union on 154 of
    #     4000 points. Parity is the mesh's own semantics; the analytic union is not.
    #   * non-overlapping lattice placement, to make those two agree -- then `mesh_inside`
    #     stops missing interior points entirely and the test goes vacuous. The flaw needs a
    #     point deep inside one body whose nearest facet belongs to a neighbour pointing the
    #     other way, which only overlap produces.
    rng = np.random.default_rng(2)
    parts = [trimesh.creation.icosphere(subdivisions=2, radius=0.30).apply_translation(c)
             for c in rng.uniform(-1, 1, (40, 3))]
    m = trimesh.util.concatenate(parts)
    V = np.asarray(m.vertices, float); F = np.asarray(m.faces, np.int64)
    pts = rng.uniform(V.min(0), V.max(0), (4000, 3))
    truth = m.contains(pts)
    fast = mesh_contains_fast(V, F, pts)
    false_outside = int((truth & ~fast).sum())
    assert false_outside == 0, f"{false_outside} genuinely-inside points reported outside"
    # and the fast heuristic really does have the flaw, so the test is not vacuous
    missed = int((truth & ~np.asarray(mesh_inside(V, F, pts))).sum())
    assert missed > 50, (
        f"mesh_inside missed only {missed} interior points; this test detects the flaw by "
        f"a wide margin (210 at these settings) or not meaningfully at all")


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


def _grazing_points(V, F, n_bins=256):
    """Points whose +z ray grazes the mesh in projection: the xy of every vertex and edge midpoint and of points
    along projected edges, and the xy-bin boundaries of the grid, each nudged 1e-9 to either side; with the z
    that puts them inside (z = 0) and outside (above the top). The analytic truth comes with them."""
    from dmipy_sim.fields.susceptibility_field import _xy_bins
    xy = [V[:, :2]]
    E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    for t in (0.5, 0.25, 0.75):
        xy.append((1 - t) * V[E[:, 0], :2] + t * V[E[:, 1], :2])
    xy = np.concatenate(xy)
    scale = 1.0 / float(np.median(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)))
    _o, _t, lo, inv = _xy_bins(V[F] * scale, n_bins)
    k = np.arange(1, n_bins, 37)
    bx = np.stack([lo[0] + k / inv[0], np.full(k.size, lo[1] + 0.5 * n_bins / inv[1])], 1) / scale
    by = np.stack([np.full(k.size, lo[0] + 0.5 * n_bins / inv[0]), lo[1] + k / inv[1]], 1) / scale
    xy = np.concatenate([xy, bx, by])
    out = []
    for d in (-1e-9, 0.0, 1e-9):
        out.append(xy + d)
    return np.concatenate(out)


def test_rays_that_graze_an_edge_in_projection_are_retried_and_decided_right():
    """The grid path counts crossings of a +z ray; a ray through a projected edge or vertex meets the shared edge of two
    triangles twice or not at all, so it is re-cast from a jittered origin. Uniform random points never reach that
    path. On a box every such point has an analytic answer: with z = 0 it is inside wherever its xy is inside the
    footprint, and above the box it is outside; both must come back right, and the points must actually have been
    ambiguous on the first cast."""
    from dmipy_sim.fields.susceptibility_field import _parity_vertical, _xy_bins
    box = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    V = np.asarray(box.vertices, float); F = np.asarray(box.faces, np.int64)
    xy = _grazing_points(V, F)
    # The box's outline is where its vertical faces project to edges: a ray there is re-cast from an origin
    # jittered by a thousandth of an edge, so a point within that of the outline is decided by where the jitter
    # lands -- at the wall, which is the one place a side is not a property of the point. The rows here are the
    # grazes INSIDE the footprint: the two face diagonals and the grid's bin boundaries, where the answer is exact.
    interior = (np.abs(xy[:, 0]) < 0.5 - 0.02) & (np.abs(xy[:, 1]) < 1.0 - 0.02)
    xy = xy[interior]
    P = np.concatenate([np.column_stack([xy, np.zeros(len(xy))]), np.column_stack([xy, np.full(len(xy), 2.0)])])
    truth = np.concatenate([np.ones(len(xy), bool), np.zeros(len(xy), bool)])
    scale = 1.0 / float(np.median(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)))
    tri = V[F] * scale
    offsets, tri_ids, lo, inv = _xy_bins(tri, 256)
    _ins, ambiguous = _parity_vertical(tri, P * scale, offsets, tri_ids, lo, inv, 256, 1e-9)
    E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    A, B = V[E[:, 0], :2], V[E[:, 1], :2]                                    # every projected edge, as a segment
    keep = np.linalg.norm(B - A, axis=1) > 0                                 # a vertical edge projects to a point
    A, B = A[keep], B[keep]
    d = A[None] - P[:, None, :2]; e = B[None] - A[None]
    t = np.clip(-(d * e).sum(2) / (e * e).sum(2), 0.0, 1.0)
    on_edge = (np.linalg.norm(d + t[..., None] * e, axis=2).min(1) * scale < 1e-12) & (P[:, 2] == 0.0)   # exactly on an edge, below the top
    # a ray from the bottom face's diagonal crosses the top face's interior and is not a graze; the top's is
    assert ambiguous[on_edge].sum() >= 3, "no ray through the top face's projected diagonal was flagged ambiguous: the retry path is not reached"
    got = mesh_contains_fast(V, F, P)
    wrong = got != truth
    assert not wrong.any(), f"{wrong.sum()} of {len(P)} grazing points decided wrong (e.g. {P[wrong][:3]})"
    with pytest.warns(RuntimeWarning, match="graze"):                       # and an unresolved graze is said, not trusted
        mesh_contains_fast(V, F, P[ambiguous][:8], max_retries=0)


def test_grazing_rays_on_an_icosphere_agree_with_the_solid():
    """The same construction on a facetted sphere, where the truth is the inscribed solid: at z = 0 a point within
    0.9 R of the axis is inside every facet, at z = 1.5 R it is outside."""
    m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    V = np.asarray(m.vertices, float); F = np.asarray(m.faces, np.int64)
    xy = _grazing_points(V, F)
    xy = xy[np.linalg.norm(xy, axis=1) < 0.9]
    P = np.concatenate([np.column_stack([xy, np.zeros(len(xy))]), np.column_stack([xy, np.full(len(xy), 1.5)])])
    truth = np.concatenate([np.ones(len(xy), bool), np.zeros(len(xy), bool)])
    got = mesh_contains_fast(V, F, P)
    assert np.array_equal(got, truth), f"{(got != truth).sum()} of {len(P)} grazing points on the icosphere decided wrong"
