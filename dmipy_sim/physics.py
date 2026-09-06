"""Compatibility shim: ``dmipy_sim.physics`` moved to ``dmipy_sim.engine.physics``."""
import warnings as _w

_w.warn("dmipy_sim.physics moved to dmipy_sim.engine.physics; this shim goes away next release",
        DeprecationWarning, stacklevel=2)
from .engine.physics import *          # noqa: F401,F403,E402
from .engine import physics as _m      # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
