"""Compatibility shim: ``dmipy_sim.compression`` moved to ``dmipy_sim.replay.compression``."""
import warnings as _w

_w.warn("dmipy_sim.compression moved to dmipy_sim.replay.compression; this shim goes away next release",
        DeprecationWarning, stacklevel=2)
from .replay.compression import *          # noqa: F401,F403,E402
from .replay import compression as _m      # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
