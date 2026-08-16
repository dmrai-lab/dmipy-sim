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
from dmipy_sim.mesh import Mesh
from dmipy_sim.waveforms import pgse

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
    pts = np.asarray(mesh.init_positions(1200, jax.random.PRNGKey(0), intra=True), float)
    lab = np.asarray(mesh.classify_positions_exact(pts))

    truth = tri.contains(pts / UM)      # tri is in unit coordinates
    assert truth.mean() > 0.97, "precondition: seeds should be inside"
    assert (lab[truth] == 0).mean() > 0.98, (
        f"only {100*(lab[truth]==0).mean():.1f}% of genuinely interior seeds labelled interior")


def test_labels_agree_with_exact_containment_at_the_end_of_the_walk():
    """What the carry guarantees: the label reports where the walker IS, not what its mesh can see.

    Asserted against exact parity on the final positions rather than against confinement, because this
    mesh does not confine reliably once refined (a separate defect, dmrai-lab/dmipy-sim#40) and bundling
    the two would leave this test failing for a reason it does not test.
    """
    mesh, tri = _thick_tube()
    wf = set_b(pgse(delta=5e-3, DELTA=15e-3, G_magnitude=0.05, bvecs=[[1, 0, 0]], n_t=200), 5e8)

    out = simulate(1200, D, wf, mesh, seed=7, return_compartments='final',
                   return_positions=True, require_gpu=False)
    arrs = [np.asarray(a) for a in out]
    pos = [a for a in arrs if a.ndim == 2 and a.shape[-1] == 3][-1]
    comp = [a for a in arrs if a.ndim == 1 and a.dtype.kind in "iu"][-1]

    inside = tri.contains(pos / UM)
    assert inside.sum() > 100, "precondition: some walkers must end inside"
    assert (comp[inside] == 0).mean() > 0.98, (
        f"only {100*(comp[inside]==0).mean():.1f}% of walkers that ARE inside are labelled interior")
    if (~inside).sum() > 50:
        assert (comp[~inside] == 1).mean() > 0.90, (
            f"only {100*(comp[~inside]==1).mean():.1f}% of walkers outside are labelled exterior")


def test_label_accuracy_does_not_depend_on_mesh_refinement():
    """The signature of the defect: a re-derived label degrades as the mesh is refined, because the gather
    shrinks while the object does not. A carried label must not."""
    accs = []
    for subdiv in (0, 2):
        mesh, tri = _thick_tube(subdivisions=subdiv)
        wf = set_b(pgse(delta=5e-3, DELTA=15e-3, G_magnitude=0.05, bvecs=[[1, 0, 0]], n_t=200), 5e8)
        out = simulate(1000, D, wf, mesh, seed=11, return_compartments='final',
                       return_positions=True, require_gpu=False)
        arrs = [np.asarray(a) for a in out]
        pos = [a for a in arrs if a.ndim == 2 and a.shape[-1] == 3][-1]
        comp = [a for a in arrs if a.ndim == 1 and a.dtype.kind in "iu"][-1]
        inside = tri.contains(pos / UM)
        accs.append(float((comp[inside] == 0).mean()) if inside.any() else 1.0)

    assert min(accs) > 0.95, f"label accuracy for interior walkers: {accs}"
    assert abs(accs[0] - accs[1]) < 0.05, (
        f"accuracy moved {accs[0]:.3f} -> {accs[1]:.3f} on refinement alone; the label is tracking the "
        f"mesh rather than the walker")
