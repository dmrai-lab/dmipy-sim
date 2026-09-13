"""A walker's compartment must not depend on how finely its own fibre is meshed.

`_classify_arr` reads interior/exterior from a 27-cell gather and defaults to exterior when that gather is
empty. Deep inside a thick fibre the gather IS empty (the cell size scales with the triangle size, not the
object), so a walker that has gone nowhere near a wall reads as extra-axonal -- and reads that way for as
long as it stays in the middle of its own axon.

Compartment is a state that changes at crossings, not a property to be re-measured from local geometry
every step. A walker with no wall within reach cannot have crossed one, so its label is carried; only the
initial labels, which have nothing to carry, are resolved exactly.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from dmipy_sim import simulate, set_b
from dmipy_sim.geometry.mesh import Mesh
from dmipy_sim.sequences import pgse

from ._containment import inside as contains

D = 1.0e-9


UM = 1e-6


def _thick_tube(radius=3.0, height=24.0, sections=96, subdivisions=1):
    """Built at UNIT scale then converted to metres.

    trimesh unitizes against an absolute tolerance, so a cylinder constructed directly at 1e-6 comes back
    degenerate -- the same trap that makes its face normals collapse at SI scale. `tri` therefore stays in
    unit coordinates and query points are scaled up to meet it.
    """
    m = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    for _ in range(subdivisions):
        m = m.subdivide()
    V, F = np.asarray(m.vertices, float) * UM, np.asarray(m.faces, np.int64)
    e = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
    lo, hi = V.min(0) - 2 * UM, V.max(0) + 2 * UM
    mesh = Mesh(V, F, periodic=False, voxel_min=lo, voxel_max=hi,
                feature_radius=0.5 * float(np.median(e)))
    return mesh, m


def test_initial_labels_are_exact_not_defaulted():
    """Seeds inside the tube must be labelled interior, including the deep ones the gather cannot see."""
    mesh, tri = _thick_tube()
    pts = np.asarray(mesh.init_positions(400, jax.random.PRNGKey(0), pool="intra"), float)
    lab = np.asarray(mesh.classify_positions_exact(pts))

    truth = contains(tri, pts / UM)      # tri is in unit coordinates
    assert truth.mean() > 0.97, "precondition: seeds should be inside"
    assert (lab[truth] == 1).mean() > 0.98, (
        f"only {100*(lab[truth]==1).mean():.1f}% of genuinely interior seeds labelled interior")


@pytest.mark.parametrize("subdivisions", [0, 2])
def test_a_walker_the_gather_cannot_see_keeps_its_label(subdivisions):
    """The carry rule itself, on constructed positions, at two refinements of the same tube: where the 27-cell
    gather is empty the label is whatever it was (1 stays 1, 0 stays 0), where it is populated the label is the
    classifier's. On the refined tube most deep interior points have an empty gather and the raw classifier
    calls them exterior -- the defect's signature -- and the carried label does not move."""
    from dmipy_sim.geometry.mesh import _gather_is_populated, _classify_arr
    mesh, tri = _thick_tube(subdivisions=subdivisions)
    rng = np.random.default_rng(subdivisions)
    n = 2000
    rr = 3.0 * UM * np.sqrt(rng.uniform(0, 1, n)); ph = rng.uniform(0, 2 * np.pi, n)
    pts = np.stack([rr * np.cos(ph), rr * np.sin(ph), rng.uniform(-8e-6, 8e-6, n)], 1)      # inside the tube
    pts = pts[contains(tri, pts / UM)]
    P = jax.numpy.asarray(pts, jax.numpy.float32)
    seen = np.asarray(jax.vmap(lambda r: _gather_is_populated(mesh._A, r))(P))
    raw = np.asarray(jax.vmap(lambda r: _classify_arr(mesh._A, r))(P))
    keep1 = np.asarray(jax.vmap(lambda r: mesh.classify_position_carry(r, jax.numpy.int32(1)))(P))
    keep0 = np.asarray(jax.vmap(lambda r: mesh.classify_position_carry(r, jax.numpy.int32(0)))(P))
    assert (keep1[~seen] == 1).all() and (keep0[~seen] == 0).all()             # nothing in reach: carried
    assert (keep1[seen] == raw[seen]).all() and (keep0[seen] == raw[seen]).all()  # a wall in reach: classified
    assert (raw[seen] == 1).mean() > 0.98                                       # and classified right
    if subdivisions == 2:
        assert (~seen).mean() > 0.3 and (raw[~seen] == 0).all()                 # the raw label reads exterior deep inside
    assert (keep1 == 1).mean() > 0.98                                           # an interior walker stays interior


def test_labels_agree_with_exact_containment_at_the_end_of_a_short_walk():
    """The carry in the engine's scan: after a short walk the label reports where the walker IS, not what its
    mesh can see. Asserted against exact parity on the final positions rather than against confinement, because
    this mesh does not confine reliably once refined (dmrai-lab/dmipy-sim#40) and bundling the two would leave
    this test failing for a reason it does not test."""
    mesh, tri = _thick_tube()
    wf = set_b(pgse([[1, 0, 0]], 0.5e-3, 1.5e-3, gradient_strengths=0.05, n_t=20), 5e7)

    out = simulate(300, D, wf, mesh, seed=7, return_compartments='final',
                   return_positions=True, require_gpu=False)
    arrs = [np.asarray(a) for a in out]
    pos = [a for a in arrs if a.ndim == 2 and a.shape[-1] == 3][-1]
    comp = [a for a in arrs if a.ndim == 1 and a.dtype.kind in "iu"][-1]

    inside = contains(tri, pos / UM)
    assert inside.sum() > 100, "precondition: some walkers must end inside"
    assert (comp[inside] == 1).mean() > 0.98, (
        f"only {100*(comp[inside]==1).mean():.1f}% of walkers that ARE inside are labelled interior")
    if (~inside).sum() > 50:
        assert (comp[~inside] == 0).mean() > 0.90, (
            f"only {100*(comp[~inside]==0).mean():.1f}% of walkers outside are labelled exterior")
