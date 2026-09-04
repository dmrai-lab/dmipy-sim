"""Compatibility shim: moved to :mod:`dmipy_sim.geometry.mesh` (see #88)."""
from .geometry.mesh import *  # noqa: F401,F403
from .geometry import mesh as _m
import sys as _sys; _sys.modules[__name__].__dict__.update(
    {k: v for k, v in vars(_m).items() if not k.startswith("__")})
