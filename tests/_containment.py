"""One containment oracle for the whole suite: the native grid test, never trimesh's ray engine.

`trimesh.contains` picks its backend at import -- native `embreex` when present, else a pure-NumPy
engine that tests every ray against every triangle, with no warning either way. `embreex` publishes
no aarch64 wheels, so on ARM that fallback is permanent, and a test that uses `.contains` merely to
establish ground truth silently becomes minutes of O(points x triangles) work.

`mesh_contains_fast` has the SAME ray-parity semantics and the same answers (that equivalence is
itself asserted in tests/test_mesh_containment_grid.py against trimesh as an independent oracle);
what differs is only which triangles get tested. Use this everywhere ground truth is wanted.

Deliberate exception: the containment modules that VALIDATE our implementation against trimesh as
an external reference keep calling `.contains` directly -- that comparison is their whole point.
"""
import numpy as np

from dmipy_sim.susceptibility_field import mesh_contains_fast


def inside(tri, pts):
    """Boolean "inside this closed trimesh", via the native grid test. `pts` in the mesh's own units."""
    return mesh_contains_fast(np.asarray(tri.vertices, float),
                              np.asarray(tri.faces, np.int64),
                              np.asarray(pts, float))
