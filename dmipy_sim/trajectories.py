"""Compatibility shim: ``dmipy_sim.trajectories`` moved to ``dmipy_sim.replay.trajectories``."""
import warnings as _w

_w.warn("dmipy_sim.trajectories moved to dmipy_sim.replay.trajectories; this shim goes away next release",
        DeprecationWarning, stacklevel=2)
from .replay.trajectories import *          # noqa: F401,F403,E402
from .replay import trajectories as _m      # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
