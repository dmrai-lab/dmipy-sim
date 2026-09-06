"""Compatibility shim: ``dmipy_sim.mesh_bundle`` moved to ``dmipy_sim.replay.builders.mesh_bundle``."""
import warnings as _w

_w.warn("dmipy_sim.mesh_bundle moved to dmipy_sim.replay.builders.mesh_bundle; this shim goes away next release",
        DeprecationWarning, stacklevel=2)
from .replay.builders.mesh_bundle import *          # noqa: F401,F403,E402
from .replay.builders import mesh_bundle as _m      # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
