"""Compatibility shim: ``dmipy_sim.noise`` moved to ``dmipy_sim.acquisition.noise``."""
import warnings as _w

_w.warn("dmipy_sim.noise moved to dmipy_sim.acquisition.noise; this shim goes away next release",
        DeprecationWarning, stacklevel=2)
from .acquisition.noise import *          # noqa: F401,F403,E402
from .acquisition import noise as _m      # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
