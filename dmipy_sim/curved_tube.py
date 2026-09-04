"""Compatibility shim: moved to :mod:`dmipy_sim.geometry.curved_tube` (see #88)."""
from .geometry.curved_tube import *  # noqa: F401,F403
from .geometry import curved_tube as _m
import sys as _sys; _sys.modules[__name__].__dict__.update(
    {k: v for k, v in vars(_m).items() if not k.startswith("__")})
