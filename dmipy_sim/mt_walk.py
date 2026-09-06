"""Compatibility shim: ``dmipy_sim.mt_walk`` moved to ``dmipy_sim.engine.mt_walk``."""
import warnings as _w

_w.warn("dmipy_sim.mt_walk moved to dmipy_sim.engine.mt_walk; this shim goes away next release",
        DeprecationWarning, stacklevel=2)
from .engine.mt_walk import *          # noqa: F401,F403,E402
from .engine import mt_walk as _m      # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
