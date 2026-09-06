"""Compatibility shim: ``dmipy_sim.pulse_sequence`` moved to ``dmipy_sim.engine.pulse_sequence``."""
import warnings as _w

_w.warn("dmipy_sim.pulse_sequence moved to dmipy_sim.engine.pulse_sequence; this shim goes away next release",
        DeprecationWarning, stacklevel=2)
from .engine.pulse_sequence import *          # noqa: F401,F403,E402
from .engine import pulse_sequence as _m      # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
