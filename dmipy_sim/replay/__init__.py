"""``dmipy_sim.replay``: the package, re-exporting :mod:`dmipy_sim.replay.replay` so the
flat name keeps its public surface."""
from .replay import *          # noqa: F401,F403
from . import replay as _m     # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if k.startswith("_") and not k.startswith("__")})
