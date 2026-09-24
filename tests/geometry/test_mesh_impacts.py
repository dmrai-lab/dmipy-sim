"""Laboratory impact table for meshes: one walker fired at one wall of a closed mesh from a known start, direction
and length, its end tested by exact containment. No random walk, no statistics.

The rows are the cases a walk reaches rarely or hides in an aggregate: starts on facet centroids, edge midpoints
and vertices, inset by the nudge, the float32 epsilon and a thousandth of a step; aimed head-on, at 45 and 89
degrees, along the tangent, at the nearest edge midpoint and at the nearest vertex; for a tenth of a walk step,
half a collision cell and the collision rule's own step (0.9 of a cell). Steps longer than the cell are not
rows: the collision sub-step rule (physics.collision_sub_steps) is what keeps a walk from taking them, and that
rule has its own test.

A walker fired from an edge midpoint or a vertex of a closed cylinder mesh, head-on or along the edge, passes
through the shared edge (dmrai-lab/dmipy-sim#430); those rows are expected failures against that issue and
the walk-level tests keep their tolerance until it is fixed.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from dmipy_sim.geometry.mesh import Mesh
from dmipy_sim.fields.susceptibility_field import mesh_contains

UM = 1e-6
STEP = float(np.sqrt(6 * 2e-9 * 1e-4))                     # a walk step of 1.1 um
LEAK = 1e-9                                                # more than a nanometre outside is a leak, not surface noise
_MESHES = {}


def _mesh(name):
    """Built on first use, never at collection (a Mesh holds device arrays)."""
    if name not in _MESHES:
        kind, n = name.split("-")
        m = (trimesh.creation.icosphere(subdivisions=int(n), radius=3.0) if kind == "ico"
             else trimesh.creation.cylinder(radius=3.0, height=12.0, sections=int(n)))
        V, F = np.asarray(m.vertices, float) * UM, np.asarray(m.faces, np.int64)
        e = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
        mesh = Mesh(V, F, periodic=False, voxel_min=V.min(0) - 2 * UM, voxel_max=V.max(0) + 2 * UM,
                    feature_radius=0.5 * float(np.median(e)))
        fire = jax.jit(jax.vmap(lambda p, s: mesh.interact(p, s).r))       # compiled once per mesh, shared by the rows
        _MESHES[name] = (mesh, V, F, m, fire, trimesh.proximity.ProximityQuery(m))
    return _MESHES[name]


STARTS = ("centroid", "edge_mid", "vertex")
DIRECTIONS = ("head_on", "45", "89", "tangent", "at_edge", "at_vertex")
_CRACK = {("cyl", "edge_mid", "head_on"), ("cyl", "edge_mid", "at_edge"), ("cyl", "vertex", "head_on"),
          ("cyl", "vertex", "at_vertex")}


def _table(name, start_kind, direction, n_per=20, seed=0):
    """(label, start, step) for one start kind and direction: every inset and length."""
    mesh, V, F, m, _fire_fn, _prox = _mesh(name)
    rng = np.random.default_rng(seed)
    E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    vn = np.asarray(m.vertex_normals)
    if start_kind == "centroid":
        P, N = V[F].mean(1), np.asarray(m.face_normals)
    elif start_kind == "edge_mid":
        P, N = 0.5 * (V[E[:, 0]] + V[E[:, 1]]), 0.5 * (vn[E[:, 0]] + vn[E[:, 1]])
    else:
        P, N = V, vn
    N = N / np.linalg.norm(N, axis=1, keepdims=True)
    sel = rng.choice(len(P), min(n_per, len(P)), replace=False); P, N = P[sel], N[sel]
    t = np.cross(N, np.array([0.3, 0.5, 0.8])); t /= np.linalg.norm(t, axis=1, keepdims=True)
    out = []
    for iname, ins in (("nudge", float(mesh._NUDGE)), ("eps", float(mesh._EPS)), ("1e-3step", 1e-3 * STEP)):
        start = P - ins * N                                                   # inside, along the inward normal
        if direction in ("at_edge", "at_vertex"):
            targets = 0.5 * (V[E[:, 0]] + V[E[:, 1]]) if direction == "at_edge" else V
            dd = targets[None] - start[:, None]; dist = np.linalg.norm(dd, axis=2); dist[dist < 1e-12] = np.inf
            d = dd[np.arange(len(start)), dist.argmin(1)]
        else:
            th = {"head_on": 0.0, "45": 45.0, "89": 89.0, "tangent": 90.0}[direction]
            d = np.cos(np.deg2rad(th)) * N + np.sin(np.deg2rad(th)) * t
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        for lname, L in (("short", 0.1 * STEP), ("half_cell", 0.5 * mesh.cell_size), ("rule", 0.9 * mesh.cell_size)):
            out.append((f"{iname}/{lname}", start, d * L))
    return out


def _fire(name, start_kind, direction):
    mesh, V, F, m, f, prox = _mesh(name)
    rows = []
    for label, start, step in _table(name, start_kind, direction):
        inside = mesh_contains(V, F, start)                                    # a start the mean normal put outside is no row
        start, step = start[inside], step[inside]
        if len(start) == 0:
            continue
        r = np.asarray(f(jnp.asarray(start, jnp.float32), jnp.asarray(step, jnp.float32)))
        depth = -prox.signed_distance(r.astype(np.float64) / UM) * UM          # metres outside; negative is inside
        moved = np.linalg.norm(r - start, axis=1); asked = np.linalg.norm(step, axis=1)
        rows.append((label, depth, moved > asked * (1 + 1e-4) + 1e-12))
    return rows


@pytest.mark.parametrize("name", ["ico-2", "cyl-24"])
@pytest.mark.parametrize("start_kind", STARTS)
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_an_impact_on_a_closed_mesh_never_ends_outside(request, name, start_kind, direction):
    """Every impact of the table ends inside the mesh (to a nanometre) and moves no further than it stepped."""
    if (name.split("-")[0], start_kind, direction) in _CRACK:
        request.node.add_marker(pytest.mark.xfail(reason="a walker through a shared edge or vertex of a closed "
                                                         "cylinder mesh (dmrai-lab/dmipy-sim#430)", strict=False))
    rows = _fire(name, start_kind, direction)
    assert rows, "no start of this kind lies inside the mesh"
    leaked = [(label, float(depth.max()), int((depth > LEAK).sum()), len(depth)) for label, depth, _ in rows if (depth > LEAK).any()]
    if leaked:
        lines = "\n".join(f"      {lab:16} {n}/{k} outside, worst {d * 1e6:.3f} um" for lab, d, n, k in leaked)
        pytest.fail(f"{name} {start_kind} {direction}: impacts ended outside the mesh:\n{lines}")
    assert not any(far.any() for _, _, far in rows), f"{name} {start_kind} {direction}: a reflection added distance"


def _periodic_tube(R=4e-6, L=12e-6, nt=32, nz=24):
    """An open tube along z whose ends are the periodic faces of the box: the triangles at each end are
    replicated as ghosts across the seam, so a walker crossing z = L meets the tube's own wall on the far side."""
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False); zs = np.linspace(0, L, nz)
    V = np.array([[R * np.cos(t), R * np.sin(t), z] for z in zs for t in th])
    F = []
    for iz in range(nz - 1):
        for j in range(nt):
            a = iz * nt + j; b = iz * nt + (j + 1) % nt; c = (iz + 1) * nt + j; d = (iz + 1) * nt + (j + 1) % nt
            F.append([a, b, d]); F.append([a, d, c])
    mesh = Mesh(V, np.array(F), periodic=(False, False, True), voxel_min=[-6e-6, -6e-6, 0.0], voxel_max=[6e-6, 6e-6, L],
                feature_radius=R)
    return mesh, R, L


def test_a_walker_crossing_the_periodic_seam_keeps_to_its_tube():
    """The rows the closed bodies cannot give: a walker just inside the +z face of the box, at the tube's axis,
    halfway out, and a nudge and an epsilon inside its wall, fired across the seam along z, across it and at the
    wall at 45 and 89 degrees, along the wall, and radially, for a tenth of a step, half a cell and the rule's
    step. Wrapped back into the box, every impact is still inside the tube and moved no further than it stepped;
    the ghost triangles across the seam are what makes that hold."""
    mesh, R, L = _periodic_tube()
    fire = jax.jit(jax.vmap(lambda p, s: mesh.interact(p, s).r))
    nudge, eps = float(mesh._NUDGE), float(mesh._EPS)
    cases = []
    for zname, z in (("eps", L - eps), ("nudge", L - nudge), ("1e-3step", L - 1e-3 * STEP), ("half_step", L - 0.5 * STEP)):
        for rname, rho in (("axis", 0.0), ("half", 0.5 * R), ("wall_nudge", R - nudge), ("wall_eps", R - eps)):
            start = np.array([rho, 0.0, z])
            out_n = np.array([1.0, 0.0, 0.0])                                # the wall's outward normal here
            for dname, d in (("+z", [0, 0, 1.0]), ("-z", [0, 0, -1.0]), ("45_to_wall", [1.0, 0, 1.0]), ("89_to_wall", [1.0, 0, np.tan(np.deg2rad(1))]),
                             ("along_wall", [0, 1.0, 1.0]), ("radial", [1.0, 0, 0])):
                d = np.asarray(d, float); d /= np.linalg.norm(d)
                for lname, dist in (("short", 0.1 * STEP), ("half_cell", 0.5 * mesh.cell_size), ("rule", 0.9 * mesh.cell_size)):
                    cases.append((f"{zname}/{rname}/{dname}/{lname}", start, d * dist))
    starts = np.stack([c[1] for c in cases]); steps = np.stack([c[2] for c in cases])
    r = np.asarray(fire(jnp.asarray(starts, jnp.float32), jnp.asarray(steps, jnp.float32)))
    # the interaction leaves a crossing walker beyond the face (the walk wraps it afterwards); the tube is the
    # same at every z, so its radius is the test either side of the seam
    radial = np.linalg.norm(r[:, :2], axis=1)
    out = radial > R * (1 + 1e-6)
    if out.any():
        rows = "\n".join(f"      {cases[i][0]:36} -> radial {radial[i] / R:.4f} R, z {r[i, 2] * 1e6:.3f} um" for i in np.flatnonzero(out)[:12])
        pytest.fail(f"{out.sum()}/{len(cases)} impacts across the seam ended outside the tube:\n{rows}")
    moved = np.linalg.norm(r - starts, axis=1); asked = np.linalg.norm(steps, axis=1)
    assert not (moved > asked * (1 + 1e-4) + 1e-12).any(), "a reflection at the seam added distance"
    crossed = (r[:, 2] > L) | (r[:, 2] < 0.0)
    assert crossed.sum() >= len(cases) // 6, f"only {crossed.sum()} of {len(cases)} impacts crossed the seam: the rows barely reach it"
