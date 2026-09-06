"""Compatibility shim: ``dmipy_sim.bloch`` moved to ``dmipy_sim.engine.bloch``."""
import warnings as _w

_w.warn("dmipy_sim.bloch moved to dmipy_sim.engine.bloch; this shim goes away next release",
        DeprecationWarning, stacklevel=2)
from .engine.bloch import *          # noqa: F401,F403,E402
from .engine import bloch as _m      # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
