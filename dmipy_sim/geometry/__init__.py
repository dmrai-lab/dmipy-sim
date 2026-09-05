"""Geometry: where a walker may be, and what happens when it reaches a wall.

Split out of the former 3,548-line ``geometries.py`` (see #88). The division follows the
only real coupling in that file: every class inherits :class:`base.Geometry` and shares the
frame helper, and *nothing else* crosses these boundaries.

    _boundary   the boundary RULES, one implementation each, for every geometry here
    base        the ABC + the geometries with no curved wall
    analytic    one closed surface with a closed-form ray intersection
    packed      periodic packings of many objects (minimum-image)
    myelin      concentric multi-compartment (carried state, fused kernels)
    packing     position generators for the packed geometries
    curved_tube sphere-swept polylines
    mesh        arbitrary triangular meshes, grid-accelerated
    mesh_shapes procedural mesh + susceptibility-source builders

Import from here rather than the submodules: ``from dmipy_sim.geometry import Cylinder``.
"""
from .base import Geometry, LengthScales, FreeDiffusion, Box1D, initial_positions
from .analytic import Sphere, Cylinder, Ellipsoid, PermeableSlab1D, PermeableShell
from .packed import PackedCylinders, PackedSpheres
from .myelin import MyelinatedCylinder, PackedMyelinatedCylinders
from .packing import pack_cylinders, pack_spheres, pack_myelinated_cylinders
from .curved_tube import CurvedTube, MultiShellCurvedTube, PackedCurvedTubes
from .mesh import Mesh

__all__ = [
    "Geometry", "LengthScales", "FreeDiffusion", "Box1D", "initial_positions",
    "Sphere", "Cylinder", "Ellipsoid", "PermeableSlab1D", "PermeableShell",
    "PackedCylinders", "PackedSpheres",
    "MyelinatedCylinder", "PackedMyelinatedCylinders",
    "pack_cylinders", "pack_spheres", "pack_myelinated_cylinders",
    "CurvedTube", "MultiShellCurvedTube", "PackedCurvedTubes", "Mesh",
]
