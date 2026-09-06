"""Compatibility shim: ``dmipy_sim.sh_convolution`` moved to ``dmipy_sim.replay.sh_convolution``."""
import warnings as _w

_w.warn("dmipy_sim.sh_convolution moved to dmipy_sim.replay.sh_convolution; this shim goes away next release",
        DeprecationWarning, stacklevel=2)
from .replay.sh_convolution import *          # noqa: F401,F403,E402
from .replay import sh_convolution as _m      # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
