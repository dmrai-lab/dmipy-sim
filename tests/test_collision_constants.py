"""The two length constants guarding the collision test must stay in a fixed relationship.

`_EPS` is the smallest length float32 can resolve on this domain; a candidate collision closer than it is
numerical noise -- in practice the walker re-detecting the triangle it just left -- and gets discarded.
`_NUDGE` is how far off the wall a reflected walker is parked, i.e. the clearance that guard has to see
past. If `_EPS` grows to a comparable size, the guard stops distinguishing "the wall I just left" from
"a wall just ahead", and the walker passes through the second one.

They were scaled to different things -- `_EPS` to the box, `_NUDGE` to the step -- so their ratio was a
free variable of the geometry:

    _EPS/_NUDGE = 6e-3 * box / feature_radius

which passes 1 for a fine mesh in a large box. That is not exotic: CACTUS generates at `min_rad 0.27 um`
in a 30 um box, so resolving to half the thinnest feature gives 1.33. Measured on a closed cylinder with
`reject_escape` off, escape counted beyond 1 nm of depth:

    ratio 0.16 (shipped)   0 / 4,000
    ratio 1.33 (CACTUS)   38 / 4,000   = 0.95%
    ratio 16             4811 / 20,000 = 24%

A 1% walker loss sits at the Monte-Carlo floor the packs certify against (9.3e-3), so it biases rather
than scatters. dmrai-lab/dmipy-sim#53.
"""
from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from dmipy_sim.mesh import Mesh

UM = 1e-6
# The guard must stay well inside the clearance. 1/8 is a generous ceiling on the 1/16 the floor gives,
# so this fails on a regression of the relationship rather than on a tweak of the constant.
MAX_EPS_OVER_NUDGE = 1.0 / 8.0


def _mesh(box_um, feature_radius_um, sections=96):
    """A closed tube of a given size, indexed at a given feature radius.

    `box_um` is what sets `_EPS` (it scales with the domain) and `feature_radius_um` is what sets the
    step and therefore `_NUDGE`, so the pair is exactly the knob this is about.
    """
    m = trimesh.creation.cylinder(radius=box_um / 10.0, height=box_um * 0.8, sections=sections)
    V = np.asarray(m.vertices, float) * UM
    F = np.asarray(m.faces, np.int64)
    pad = 0.5 * box_um * UM
    return Mesh(V, F, periodic=False, voxel_min=V.min(0) - pad, voxel_max=V.max(0) + pad,
                feature_radius=feature_radius_um * UM)


# 100 um box at 0.05 um features is deliberately absent: the lookup grid is O((box/cell)^3) and asks for
# 24.5 TiB before any of this matters. That ceiling is a separate limitation, not this one.
@pytest.mark.parametrize("box_um, fr_um, label", [
    (10.0, 0.600, "Winther axon06 intra"),
    (6.0, 0.375, "test cylinder, subdiv 2"),
    (30.0, 0.270, "CACTUS, fr = min_rad"),
    (30.0, 0.135, "CACTUS, fr = min_rad/2"),
])
def test_the_collision_guard_stays_inside_the_clearance_it_guards(box_um, fr_um, label):
    """Whatever the mesh, `_EPS` must stay far below `_NUDGE`.

    Without a floor this ratio is `6e-3 * box/feature_radius` and grows without limit as meshes get
    finer or boxes get larger -- the direction the substrate bank is moving in.
    """
    mesh = _mesh(box_um, fr_um)
    ratio = float(mesh._EPS) / float(mesh._NUDGE)
    assert ratio <= MAX_EPS_OVER_NUDGE, (
        f"{label}: _EPS/_NUDGE = {ratio:.3f} exceeds {MAX_EPS_OVER_NUDGE:.4f}. The collision guard is "
        f"now comparable to the clearance a reflected walker is given, so a wall just ahead reads as the "
        f"wall just left and the walker passes through it.")


def test_the_nudge_stays_physically_negligible():
    """The floor must not buy safety by shoving walkers off the wall.

    A nudge is an unphysical displacement along the surface normal at every collision; it is acceptable
    only while it is far below the step length the walk resolves.
    """
    for box_um, fr_um in ((10.0, 0.600), (30.0, 0.135)):
        mesh = _mesh(box_um, fr_um)
        step_l = mesh.radius / 6.0          # how Mesh derives it
        assert float(mesh._NUDGE) < 1e-2 * step_l, (
            f"nudge {float(mesh._NUDGE):.2e} m is {float(mesh._NUDGE)/step_l:.1e} of a step "
            f"({step_l:.2e} m) -- large enough to perturb the walk it is protecting")
