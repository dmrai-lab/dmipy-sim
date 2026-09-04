"""Compatibility shim: geometry classes now live in :mod:`dmipy_sim.geometry`.

Kept because downstream imports this path directly (`dmipy-fit`, `dmipy-design`) and
because moving 3,548 lines should not be a breaking change. New code should import from
``dmipy_sim.geometry``; this module is a re-export and will not gain anything new.
"""
from .geometry.base import (Geometry, FreeDiffusion, Box1D, initial_positions,  # noqa: F401
                            _rotation_to_z, _is_inside_batch)
from .geometry.analytic import (Sphere, Cylinder, Ellipsoid,  # noqa: F401
                                PermeableSlab1D, PermeableShell)
from .geometry.packed import PackedCylinders, PackedSpheres  # noqa: F401
from .geometry.myelin import MyelinatedCylinder, PackedMyelinatedCylinders  # noqa: F401
from .geometry.packing import (pack_cylinders, pack_spheres,  # noqa: F401
                               pack_myelinated_cylinders)
