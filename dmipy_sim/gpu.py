"""Compatibility shim: ``dmipy_sim.gpu`` moved to ``dmipy_sim.engine.gpu``."""
import warnings as _w

_w.warn("dmipy_sim.gpu moved to dmipy_sim.engine.gpu; this shim goes away next release",
        DeprecationWarning, stacklevel=2)
from .engine.gpu import *          # noqa: F401,F403,E402
from .engine import gpu as _m      # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
