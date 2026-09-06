"""Compatibility shim: ``dmipy_sim._gpu_config`` moved to ``dmipy_sim.engine._gpu_config``."""
import warnings as _w

_w.warn("dmipy_sim._gpu_config moved to dmipy_sim.engine._gpu_config; this shim goes away next release",
        DeprecationWarning, stacklevel=2)
from .engine._gpu_config import *          # noqa: F401,F403,E402
from .engine import _gpu_config as _m      # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
