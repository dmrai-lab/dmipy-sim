"""A surface whose faces disagree about which way is out, and what the walk does with it.

The laboratory case is the shape of Disimpy's ``tests/cylinder_mesh_closed.pkl``: a capped regular 49-gon
prism of circumradius 5 um and length 25 um, with one triangle of every wall quad and the whole top cap
written the other way round: 294 of 588 faces, 686 of 882 manifold edges here, 784 in the file itself. No
random walk and no statistics:
uniform points in the lumen, the classifier's answer on them, and one step each through
:meth:`~dmipy_sim.geometry.mesh.Mesh.reflect`.

Measured on that surface with the file's own normals (dmrai-lab/dmipy-sim#479): the classifier calls **50.5 %**
of the lumen exterior, so ``reject_escape`` discards **15.6 %** of 775 nm steps -- the walker does not move at
all -- and the mean squared displacement per step falls to 0.81 of a free step against the analytic cylinder's
0.97. A 70 ms PGSE walk of it (8,000 walkers, b to 3e9 s/m^2 along x) then sits 1.97e-2 below MISST where the
analytic cylinder of the same radius sits 2.6e-3 from it. Reoriented, the same surface refuses no step and
lands 2.3e-3 from MISST.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_sim.geometry.mesh import Mesh, enclosed_volume, orient_faces, winding_inconsistency

UM = 1e-6
RADIUS = 5 * UM
SIDES = 49
Z_RINGS = np.arange(6) * 5 * UM


def _prism(sides=SIDES, radius=RADIUS):
    """A capped regular ``sides``-gon prism, every face wound outward."""
    th = 2 * np.pi * np.arange(sides) / sides
    ring = np.stack([radius * np.cos(th), radius * np.sin(th)], 1)
    V = np.concatenate([
        np.concatenate([np.tile(ring, (len(Z_RINGS), 1)), np.repeat(Z_RINGS, sides)[:, None]], 1),
        np.array([[0.0, 0.0, Z_RINGS[0]], [0.0, 0.0, Z_RINGS[-1]]])])
    c0, c1 = len(V) - 2, len(V) - 1
    F = []
    for r in range(len(Z_RINGS) - 1):
        a, b = r * sides, (r + 1) * sides
        for k in range(sides):
            k2 = (k + 1) % sides
            F += [[a + k, a + k2, b + k2], [a + k, b + k2, b + k]]
    F += [[c0, (k + 1) % sides, k] for k in range(sides)]
    top = (len(Z_RINGS) - 1) * sides
    F += [[c1, top + k, top + (k + 1) % sides] for k in range(sides)]
    return V, np.asarray(F, np.int64)


def _as_written(F):
    """The fixture's winding: the second triangle of every wall quad, and the whole top cap, reversed."""
    F = F.copy()
    n_wall = 2 * SIDES * (len(Z_RINGS) - 1)
    rev = np.zeros(len(F), bool)
    rev[1:n_wall:2] = True
    rev[n_wall + SIDES:] = True
    F[rev] = F[rev][:, [0, 2, 1]]
    return F, int(rev.sum())


def _lumen_points(n, rng, radius=RADIUS, sides=SIDES):
    """Uniform points strictly inside the prism (the exact inscribed-polygon test, no mesh query)."""
    apothem = radius * np.cos(np.pi / sides)
    out = []
    while sum(map(len, out)) < n:
        p = rng.uniform(-radius, radius, (4 * n, 2))
        r = np.hypot(p[:, 0], p[:, 1])
        step = 2 * np.pi / sides
        off = np.abs(((np.arctan2(p[:, 1], p[:, 0]) + step / 2) % step) - step / 2)
        out.append(p[r * np.cos(off) < apothem - 1e-9])
    xy = np.concatenate(out)[:n]
    z = rng.uniform(Z_RINGS[0] + UM, Z_RINGS[-1] - UM, n)
    return np.concatenate([xy, z[:, None]], 1).astype(np.float32)


def _mesh(V, F, pad=0.0):
    return Mesh(V, F, periodic=False, voxel_min=np.array([-RADIUS - pad, -RADIUS - pad, Z_RINGS[0] - pad]),
                voxel_max=np.array([RADIUS + pad, RADIUS + pad, Z_RINGS[-1] + pad]), feature_radius=RADIUS)


def _with_normals_of(mesh, F):
    """The same mesh reading ``F``'s face normals: what the walk saw before the faces were oriented."""
    t = np.asarray(mesh.vertices)[F]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    mesh._NRM = jnp.asarray(n, jnp.float32)
    mesh._A = mesh._A._replace(NRM=mesh._NRM)
    return mesh


def test_the_fixtures_winding_is_found_and_the_faces_are_reoriented():
    V, F = _prism()
    written, n_reversed = _as_written(F)
    assert (n_reversed, winding_inconsistency(V, F)) == (294, 0)
    assert winding_inconsistency(V, written) == 686

    fixed, report = orient_faces(V, written)
    assert winding_inconsistency(V, fixed) == 0
    assert (report["reoriented"], report["inconsistent_edges"], report["components"],
            report["open_components"]) == (n_reversed, 686, 1, 0)
    assert sorted(map(sorted, fixed.tolist())) == sorted(map(sorted, written.tolist()))   # the same triangles
    t = V[fixed]
    volume = float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)
    ideal = 0.5 * SIDES * RADIUS ** 2 * np.sin(2 * np.pi / SIDES) * (Z_RINGS[-1] - Z_RINGS[0])
    assert volume == pytest.approx(ideal, rel=1e-12)        # outward, and exactly the prism it encloses

    with pytest.warns(UserWarning, match="mesh winding"):
        mesh = _mesh(V, written)
    assert (mesh.winding_inconsistent_edges, mesh.n_faces_reoriented) == (686, 294)
    assert winding_inconsistency(V, mesh.faces) == 0
    assert _mesh(V, F).n_faces_reoriented == 0              # a consistent surface is left alone

    with pytest.warns(UserWarning, match="mesh winding"):    # consistent, and every normal pointing in
        turned = _mesh(V, F[:, [0, 2, 1]])
    assert turned.n_faces_reoriented == len(F) and enclosed_volume(V, turned.faces) > 0


def test_the_lumen_reads_as_interior_and_no_step_is_refused_once_the_faces_agree():
    """The defect and its absence, measured on the same 4,000 points and the same step."""
    V, F = _prism()
    written, _ = _as_written(F)
    with pytest.warns(UserWarning, match="mesh winding"):
        mesh = _mesh(V, written)
    rng = np.random.default_rng(0)
    p = _lumen_points(4000, rng)
    d = rng.normal(size=(len(p), 3))
    step = jnp.asarray(7.746e-7 * d / np.linalg.norm(d, axis=1, keepdims=True), jnp.float32)

    classify = jax.jit(jax.vmap(mesh.classify_position))
    reflect = jax.jit(jax.vmap(mesh.reflect))
    interior = np.asarray(classify(jnp.asarray(p)))
    refused = np.asarray(np.abs(np.asarray(reflect(jnp.asarray(p), step)) - p).max(1) == 0)
    assert interior.mean() > 0.999, "the classifier does not read the lumen of an oriented surface as interior"
    assert refused.mean() == 0.0, "reject_escape discards a step inside an oriented surface"

    raw = _with_normals_of(mesh, written)
    interior_raw = np.asarray(jax.jit(jax.vmap(raw.classify_position))(jnp.asarray(p)))
    refused_raw = np.asarray(np.abs(np.asarray(jax.jit(jax.vmap(raw.reflect))(jnp.asarray(p), step)) - p).max(1) == 0)
    assert interior_raw.mean() < 0.6, "the file's own normals no longer misclassify the lumen"
    assert refused_raw.mean() > 0.1, "the file's own normals no longer cost the walk its steps"

    reported = np.asarray(jax.jit(jax.vmap(lambda q, s: raw.interact(q, s).illegal))(jnp.asarray(p), step))
    assert (reported == refused_raw).all(), "a refused step is not the one PersistentWalk.illegal_crossings counts"


def test_one_inward_cell_of_a_bundle_is_turned_though_the_bundle_looks_consistent():
    """The case a whole-surface test cannot see, and the reason the repair is not gated on one.

    A bundle spec concatenates every wall into ONE `Mesh`, so a substrate of two cells whose second is written
    inward has no inconsistent edge anywhere (each cell agrees with itself) and a positive volume overall (the
    first cell is larger). Per component the second encloses a negative volume and is turned.
    """
    V1, F1 = _prism()
    V2, F2 = _prism(radius=0.5 * RADIUS)
    V2 = V2 + np.array([12 * UM, 0.0, 0.0])
    V = np.concatenate([V1, V2])
    F = np.concatenate([F1, F2[:, [0, 2, 1]] + len(V1)])            # the second cell written inward
    assert winding_inconsistency(V, F) == 0 and enclosed_volume(V, F) > 0

    fixed, report = orient_faces(V, F)
    assert (report["components"], report["open_components"], report["reoriented"]) == (2, 0, len(F2))
    assert enclosed_volume(V, fixed) == pytest.approx(
        enclosed_volume(V1, F1) - enclosed_volume(V2, F2[:, [0, 2, 1]]), rel=1e-12)
    with pytest.warns(UserWarning, match="mesh winding"):
        mesh = _mesh(V, F, pad=8 * UM)
    assert mesh.n_faces_reoriented == len(F2)
