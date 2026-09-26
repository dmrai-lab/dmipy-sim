"""The order of ``merge_duplicate_vertices``'s two passes, on a synthetic ring that needs no data.

The duplicate pass merges within a thousandth of a median edge; the crack pass, which closes a seam whose two
lips are a few nanometres apart, merges within HALF an edge. The loose pass may therefore only ever see the
boundary the tight one leaves. Against the raw boundary it reads a merely duplicated seam as two crack lips,
and union-find then chains a whole polygonal ring into one vertex, because on a fine ring every neighbour is
within half an edge of the next.

A ring is the smallest surface that shows it: the fixture is a closed tube of ``N_SIDE`` sides whose seam
vertices are written twice, exactly as a mesh writer that does not weld its seam leaves them.
"""
import numpy as np
import pytest

from dmipy_sim.geometry.mesh import merge_duplicate_vertices, surface_topology

N_SIDE = 48          #: sides per ring -- fine enough that neighbours are within half an edge of each other
N_RING = 4           #: rings along the tube
R = 5.0e-6           #: ring radius, metres
DZ = 5.0e-6          #: ring spacing, metres -- the median edge, so the ring pitch is the smaller length


def split_ring_tube():
    """A closed tube whose faces above and below one interior ring hold their OWN coincident copies of it.

    Returns ``(V, F)``. Every vertex of that ring appears twice at exactly the same point, so the RAW surface
    has the whole ring -- ``2 * N_SIDE`` vertices, ``2 * N_SIDE`` boundary edges -- on its boundary. That is the
    shape of the defect: Disimpy's ``cylinder_mesh_closed.pkl`` writes 352 vertices for 296 distinct ones and
    carries 112 raw boundary edges for the same reason.

    It is what the earlier version of this fixture failed to be. Duplicating only one vertex per ring left 10
    vertices on the raw boundary, far fewer than a ring, so the crack pass had nothing to chain and the test
    passed against the broken function as well as the fixed one -- it asserted the right property about a
    surface that could not exhibit the failure.
    """
    th = np.arange(N_SIDE) * 2 * np.pi / N_SIDE
    V, rings = [], []
    for k in range(N_RING):
        idx = []
        for j in range(N_SIDE):
            V.append([R * np.cos(th[j]), R * np.sin(th[j]), k * DZ]); idx.append(len(V) - 1)
        rings.append(idx)
    split = N_RING // 2                                   # the ring the two halves each get a copy of
    upper = []
    for j in range(N_SIDE):
        V.append(list(V[rings[split][j]])); upper.append(len(V) - 1)     # exactly coincident
    F = []
    for k in range(N_RING - 1):
        lo = rings[k] if k != split else upper            # faces above the split use the copy
        hi = rings[k + 1]
        if k == split - 1:
            lo, hi = rings[k], rings[k + 1]               # faces below it keep the original
        for j in range(N_SIDE):
            jn = (j + 1) % N_SIDE
            # Wound so that each triangle's FIRST edge is axial. `merge_duplicate_vertices` takes its length
            # scale from the median of edge (0, 1) alone, so the scale -- and therefore whether a ring's
            # neighbours fall within half of it -- depends on the winding. Disimpy's pickle is wound this way;
            # a fixture wound the other way makes the median equal the ring gap and cannot exhibit the failure.
            F += [[lo[j], hi[j], hi[jn]], [lo[j], hi[jn], lo[jn]]]
    for cap, z, flip in ((rings[0], -DZ, False), (rings[-1], N_RING * DZ, True)):
        V.append([0.0, 0.0, z]); c = len(V) - 1
        for j in range(N_SIDE):
            jn = (j + 1) % N_SIDE
            F.append([c, cap[jn], cap[j]] if flip else [c, cap[j], cap[jn]])
    return np.asarray(V, float), np.asarray(F, np.int64)


def mains_merge(V, F, rel_tol=1e-3, crack_rel_tol=0.5):
    """``merge_duplicate_vertices`` as it stood before the passes were ordered: the crack pass is given the RAW
    boundary. Kept here, not imported, so the test states the thing it is guarding against instead of depending
    on a revision that will move.
    """
    from scipy.spatial import cKDTree
    from dmipy_sim.geometry.mesh import _boundary_vertices
    V = np.asarray(V, np.float64); F = np.asarray(F, np.int64)
    edge = float(np.median(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)))
    pairs = cKDTree(V).query_pairs(rel_tol * edge, output_type="ndarray")
    bnd = _boundary_vertices(V, F)
    if len(bnd) > 1:
        near = cKDTree(V[bnd]).query_pairs(crack_rel_tol * edge, output_type="ndarray")
        if len(near):
            pairs = np.concatenate([pairs.reshape(-1, 2), bnd[near]], axis=0)
    parent = np.arange(len(V))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    for x, y in pairs:
        rx, ry = root(x), root(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)
    rep = np.array([root(i) for i in range(len(V))])
    keep = np.flatnonzero(rep == np.arange(len(V)))
    rank = np.full(len(V), -1, np.int64); rank[keep] = np.arange(len(keep))
    Fm = rank[rep[F]]
    ok = (Fm[:, 0] != Fm[:, 1]) & (Fm[:, 1] != Fm[:, 2]) & (Fm[:, 0] != Fm[:, 2])
    return V[keep], Fm[ok], int(len(V) - len(keep))


def test_the_fixture_puts_a_whole_ring_on_the_raw_boundary():
    """Without this the rest proves nothing: the crack pass can only chain what is on the boundary it sees."""
    V, F = split_ring_tube()
    from dmipy_sim.geometry.mesh import _boundary_vertices
    assert len(_boundary_vertices(V, F)) == 2 * N_SIDE
    assert surface_topology(V, F)["boundary_edges"] == 2 * N_SIDE
    ring_gap = 2 * np.pi * R / N_SIDE
    # the length scale the function itself uses: the median of edge (0, 1), not of all three
    edge = float(np.median(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)))
    assert ring_gap < 0.5 * edge, (
        f"neighbouring ring vertices are {ring_gap:.3g} m apart and half a median edge is {0.5 * edge:.3g} m; "
        f"unless the first is smaller the crack pass would not chain them and the fixture is inert")


def test_the_unordered_crack_pass_collapses_the_ring_and_the_ordered_one_does_not():
    """The whole point, both ways round on the same surface.

    Given the raw boundary, the crack pass reads a ring of coincident copies as a rank of crack lips: every
    neighbour is within half a median edge of the next, so union-find chains the ring into ONE vertex. Given the
    boundary the duplicate pass leaves, there is no boundary at all and nothing to chain.
    """
    V, F = split_ring_tube()
    n_distinct = len(V) - N_SIDE
    Vb, Fb, nb = mains_merge(V, F)
    Vg, Fg, merged = merge_duplicate_vertices(V, F)
    # Measured against `origin/main`'s own function, extracted from git rather than transcribed: 147 vertices
    # of 194 and 95 merged where 48 are duplicates, with the rings destroyed. The transcription above agrees
    # with it exactly, which is what licenses keeping a copy here instead of a git dependency.
    assert (len(Vb), nb) == (147, 95), f"the unordered pass gave {(len(Vb), nb)}, not the measured (147, 95)"
    assert len(Vb) < n_distinct - N_SIDE // 2, (
        f"the unordered pass left {len(Vb)} vertices; it is supposed to collapse the ring, so this fixture no "
        f"longer exhibits the failure it guards")
    assert len(Vg) == n_distinct and merged == N_SIDE
    assert surface_topology(Vg, Fg)["boundary_edges"] == 0
    for k in range(N_RING):
        ring = Vg[np.isclose(Vg[:, 2], k * DZ)]
        assert len(ring) == N_SIDE
        np.testing.assert_allclose(np.linalg.norm(ring[:, :2], axis=1), R, rtol=1e-12)


def duplicated_seam_tube():
    """A closed tube with caps, whose seam column of vertices is written TWICE (once per adjoining quad).

    Returns ``(V, F)``. This one duplicates a single vertex per ring, which is a real writer defect and is what
    the duplicate pass is for, but it leaves only 10 vertices on the raw boundary and so cannot exhibit the
    ordering failure -- see :func:`split_ring_tube`.
    """
    th = np.arange(N_SIDE) * 2 * np.pi / N_SIDE
    rings, V = [], []
    for k in range(N_RING):
        idx = []
        for j in range(N_SIDE):
            V.append([R * np.cos(th[j]), R * np.sin(th[j]), k * DZ]); idx.append(len(V) - 1)
        V.append([R * np.cos(th[0]), R * np.sin(th[0]), k * DZ])
        idx.append(len(V) - 1)
        rings.append(idx)
    F = []
    for k in range(N_RING - 1):
        a, b = rings[k], rings[k + 1]
        for j in range(N_SIDE):
            F += [[a[j], a[j + 1], b[j + 1]], [a[j], b[j + 1], b[j]]]
    for cap, z in ((rings[0], -DZ), (rings[-1], N_RING * DZ)):
        V.append([0.0, 0.0, z]); c = len(V) - 1
        for j in range(N_SIDE):
            F.append([cap[j], cap[j + 1], c])
    return np.asarray(V, float), np.asarray(F, np.int64)


def test_the_duplicate_pass_welds_the_seam_and_leaves_a_closed_surface():
    """The seam's coincident copies become one vertex and the tube closes."""
    V, F = duplicated_seam_tube()
    n_dup = N_RING                                              # one extra copy per ring
    assert surface_topology(V, F)["boundary_edges"] > 0          # open as written
    V2, F2, merged = merge_duplicate_vertices(V, F)
    assert merged == n_dup
    assert len(V2) == len(V) - n_dup
    assert surface_topology(V2, F2)["boundary_edges"] == 0


def test_the_crack_pass_does_not_collapse_a_ring_it_should_not_touch():
    """The whole point of the ordering.

    Every neighbour on this ring is ``2 pi R / 48`` = 0.65 um apart while half a median edge is 2.5 um, so a
    crack pass that sees the RAW boundary chains the ring into one vertex. Ordered, the ring survives: the
    surface keeps every distinct vertex and every face.
    """
    V, F = duplicated_seam_tube()
    V2, F2, _ = merge_duplicate_vertices(V, F)
    assert len(V2) == N_RING * N_SIDE + 2                        # N rings of N_SIDE, plus two cap apices
    assert len(F2) == len(F)                                     # no face lost to a collapse
    # the rings are still rings: N_SIDE distinct radii-and-angles at each z
    for k in range(N_RING):
        ring = V2[np.isclose(V2[:, 2], k * DZ)]
        assert len(ring) == N_SIDE
        np.testing.assert_allclose(np.linalg.norm(ring[:, :2], axis=1), R, rtol=1e-12)


def test_a_real_crack_is_still_welded():
    """The crack pass must still do its job: a seam whose two lips are a nanometre apart is welded.

    Built by splitting the welded tube along one interior ring -- the faces above it get their own copy of that
    ring, displaced a nanometre -- which leaves two rims that nearly coincide, i.e. exactly the writer defect
    the crack pass exists for.
    """
    V, F = duplicated_seam_tube()
    V, F, _ = merge_duplicate_vertices(V, F)
    k = 2
    ring = np.flatnonzero(np.isclose(V[:, 2], k * DZ))
    copy_of = {int(i): len(V) + n for n, i in enumerate(ring)}
    V2 = np.vstack([V, V[ring] + [0.0, 0.0, 1e-9]])
    above = np.array([max(V[f, 2]) > k * DZ + 0.5 * DZ for f in F])      # faces on the far side of the ring
    F2 = F.copy()
    F2[above] = np.vectorize(lambda i: copy_of.get(int(i), int(i)))(F[above])
    topo = surface_topology(V2, F2)
    assert topo["boundary_edges"] > 0, "the split must leave two rims"
    V3, F3, merged = merge_duplicate_vertices(V2, F2)
    assert merged >= len(ring), f"the crack's {len(ring)} lip pairs must be welded, got {merged}"
    assert surface_topology(V3, F3)["boundary_edges"] == 0, "welding the crack must close the surface"


@pytest.mark.parametrize("scale", [1.0, 1e-6, 1e6])
def test_the_merge_is_scale_free(scale):
    """Both tolerances are relative to the median edge, so the result cannot depend on the unit the mesh is in
    -- the failure mode an absolute tolerance has on a micrometre object written in metres."""
    V, F = duplicated_seam_tube()
    a = merge_duplicate_vertices(V * scale, F)
    b = merge_duplicate_vertices(V, F)
    assert a[2] == b[2] and len(a[0]) == len(b[0]) and len(a[1]) == len(b[1])
