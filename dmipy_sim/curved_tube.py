"""Compatibility shim: moved to :mod:`dmipy_sim.geometry.curved_tube` (see #88)."""
import warnings as _w
_w.warn("dmipy_sim.curved_tube moved to dmipy_sim.geometry.curved_tube; this shim goes away in the next release.",
        DeprecationWarning, stacklevel=2)
from .geometry.curved_tube import *  # noqa: F401,F403
from .geometry import curved_tube as _m
import sys as _sys; _sys.modules[__name__].__dict__.update(
    {k: v for k, v in vars(_m).items() if not k.startswith("__")})
