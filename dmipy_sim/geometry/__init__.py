"""Geometry: where a walker may be, and what happens when it reaches a wall.

The division follows the only real coupling: every class inherits :class:`base.Geometry` and shares the
frame helper, and *nothing else* crosses these boundaries.

    _boundary   the boundary RULES, one implementation each, for every geometry here
    base        the ABC + the geometries with no curved wall
    analytic    one closed surface with a closed-form ray intersection
    packed      periodic packings of many objects (minimum-image)
    myelin      concentric multi-compartment (carried state, fused kernels)
    packing     position generators for the packed geometries (random sequential addition)
    densify     one compression path of a periodic packing, snapshotted at packing fractions
    curved_cylinder sphere-swept polylines
    label_volume a segmented image: the wall is the faces between voxels of different pools
    mesh        arbitrary triangular meshes, grid-accelerated
    mesh_shapes procedural mesh + susceptibility-source builders

Import from here rather than the submodules: ``from dmipy_sim.geometry import Cylinder``.
"""
from .base import Geometry, LengthScales, FreeDiffusion, Box1D, initial_positions, PermeabilitySchedule
from .analytic import Sphere, Cylinder, Ellipsoid, PermeableSlab1D, PermeableShell
from .packed import PackedCylinders, PackedSpheres
from .myelin import MyelinatedCylinder, PackedMyelinatedCylinders
from .packing import pack_cylinders, pack_spheres, pack_myelinated_cylinders
from .densify import densify_packing
from .curved_cylinder import CurvedCylinder, CurvedMyelinatedCylinder, PackedCurvedCylinders
from .mesh import Mesh
from .sphere_union import SphereUnion
from .label_volume import LabelVolume

__all__ = [
    "Geometry", "LengthScales", "FreeDiffusion", "Box1D", "initial_positions", "PermeabilitySchedule",
    "Sphere", "Cylinder", "Ellipsoid", "PermeableSlab1D", "PermeableShell",
    "PackedCylinders", "PackedSpheres",
    "MyelinatedCylinder", "PackedMyelinatedCylinders",
    "pack_cylinders", "pack_spheres", "pack_myelinated_cylinders", "densify_packing",
    "CurvedCylinder", "CurvedMyelinatedCylinder", "PackedCurvedCylinders", "SphereUnion", "LabelVolume", "Mesh",
]
