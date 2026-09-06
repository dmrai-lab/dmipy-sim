"""Compatibility shim: ``dmipy_sim.core`` moved to ``dmipy_sim.engine.core``."""
import warnings as _w

_w.warn("dmipy_sim.core moved to dmipy_sim.engine.core; this shim goes away next release",
        DeprecationWarning, stacklevel=2)
from .engine.core import *          # noqa: F401,F403,E402
from .engine import core as _m      # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
