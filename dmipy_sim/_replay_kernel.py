"""Compatibility shim: ``dmipy_sim._replay_kernel`` moved to ``dmipy_sim.replay._replay_kernel``."""
import warnings as _w

_w.warn("dmipy_sim._replay_kernel moved to dmipy_sim.replay._replay_kernel; this shim goes away next release",
        DeprecationWarning, stacklevel=2)
from .replay._replay_kernel import *          # noqa: F401,F403,E402
from .replay import _replay_kernel as _m      # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
