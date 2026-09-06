"""Compatibility shim: ``dmipy_sim.pedagogy`` moved to ``dmipy_sim.viz.pedagogy``."""
import warnings as _w

_w.warn("dmipy_sim.pedagogy moved to dmipy_sim.viz.pedagogy; this shim goes away next release",
        DeprecationWarning, stacklevel=2)
from .viz.pedagogy import *          # noqa: F401,F403,E402
from .viz import pedagogy as _m      # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
