"""Compatibility shim: ``dmipy_sim.rf`` moved to ``dmipy_sim.acquisition.rf``."""
import warnings as _w

_w.warn("dmipy_sim.rf moved to dmipy_sim.acquisition.rf; this shim goes away next release",
        DeprecationWarning, stacklevel=2)
from .acquisition.rf import *          # noqa: F401,F403,E402
from .acquisition import rf as _m      # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
