"""Compatibility shim: ``dmipy_sim.gaunt`` moved to ``dmipy_sim.replay.gaunt``."""
import warnings as _w

_w.warn("dmipy_sim.gaunt moved to dmipy_sim.replay.gaunt; this shim goes away next release",
        DeprecationWarning, stacklevel=2)
from .replay.gaunt import *          # noqa: F401,F403,E402
from .replay import gaunt as _m      # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
