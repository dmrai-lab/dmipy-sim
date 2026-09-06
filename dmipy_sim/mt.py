"""Compatibility shim: ``dmipy_sim.mt`` moved to ``dmipy_sim.engine.mt``."""
import warnings as _w

_w.warn("dmipy_sim.mt moved to dmipy_sim.engine.mt; this shim goes away next release",
        DeprecationWarning, stacklevel=2)
from .engine.mt import *          # noqa: F401,F403,E402
from .engine import mt as _m      # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
