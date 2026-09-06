"""``dmipy_sim.viz``: the package, re-exporting :mod:`dmipy_sim.viz.viz` so the
flat name keeps its public surface."""
from .viz import *          # noqa: F401,F403
from . import viz as _m     # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
