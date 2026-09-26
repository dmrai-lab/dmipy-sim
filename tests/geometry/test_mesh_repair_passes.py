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


def duplicated_seam_tube():
    """A closed tube with caps, whose seam column of vertices is written TWICE (once per adjoining quad).

    Returns ``(V, F)`` with 2 coincident copies of every seam vertex, so the surface reads as open until they
    are welded -- which is what the duplicate pass is for.
    """
    th = np.arange(N_SIDE) * 2 * np.pi / N_SIDE
    rings, V = [], []
    for k in range(N_RING):
        idx = []
        for j in range(N_SIDE):
            V.append([R * np.cos(th[j]), R * np.sin(th[j]), k * DZ]); idx.append(len(V) - 1)
        # the seam: one extra, coincident copy of vertex 0 of this ring
        V.append([R * np.cos(th[0]), R * np.sin(th[0]), k * DZ])
        idx.append(len(V) - 1)                       # used as the "wrap" vertex by the last quad
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
