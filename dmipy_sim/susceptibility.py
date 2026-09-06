"""Compatibility shim: ``dmipy_sim.susceptibility`` moved to ``dmipy_sim.fields.susceptibility``."""
import warnings as _w

_w.warn("dmipy_sim.susceptibility moved to dmipy_sim.fields.susceptibility; this shim goes away next release",
        DeprecationWarning, stacklevel=2)
from .fields.susceptibility import *          # noqa: F401,F403,E402
from .fields import susceptibility as _m      # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
