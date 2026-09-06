"""Compatibility shim: ``dmipy_sim.bank`` moved to ``dmipy_sim.replay.bank``."""
import warnings as _w

_w.warn("dmipy_sim.bank moved to dmipy_sim.replay.bank; this shim goes away next release",
        DeprecationWarning, stacklevel=2)
from .replay.bank import *          # noqa: F401,F403,E402
from .replay import bank as _m      # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
