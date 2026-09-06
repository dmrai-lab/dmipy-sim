"""Compatibility shim: ``dmipy_sim.mesh_axon`` moved to ``dmipy_sim.replay.builders.mesh_axon``."""
import warnings as _w

_w.warn("dmipy_sim.mesh_axon moved to dmipy_sim.replay.builders.mesh_axon; this shim goes away next release",
        DeprecationWarning, stacklevel=2)
from .replay.builders.mesh_axon import *          # noqa: F401,F403,E402
from .replay.builders import mesh_axon as _m      # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
