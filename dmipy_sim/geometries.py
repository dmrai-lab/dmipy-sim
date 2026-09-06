"""Compatibility shim: geometry classes now live in :mod:`dmipy_sim.geometry`.

Kept because downstream imports this path directly (`dmipy-fit`, `dmipy-design`) and
because moving 3,548 lines should not be a breaking change. New code should import from
``dmipy_sim.geometry``; this module is a re-export and will not gain anything new.
"""
import warnings as _w
_w.warn("dmipy_sim.geometries moved to dmipy_sim.geometry; this shim goes away in the next release.",
        DeprecationWarning, stacklevel=2)
from .geometry.base import (Geometry, FreeDiffusion, Box1D, initial_positions,  # noqa: F401,E402
                            _rotation_to_z)
from .geometry.analytic import (Sphere, Cylinder, Ellipsoid,  # noqa: F401
                                PermeableSlab1D, PermeableShell)
from .geometry.packed import PackedCylinders, PackedSpheres  # noqa: F401
from .geometry.myelin import MyelinatedCylinder, PackedMyelinatedCylinders  # noqa: F401
from .geometry.packing import (pack_cylinders, pack_spheres,  # noqa: F401
                               pack_myelinated_cylinders)
