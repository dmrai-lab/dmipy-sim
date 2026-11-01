"""The mathematics of the walk that dmipy-sim owns (dmipy-fit imports it; dmipy-sim imports nothing from dmipy-fit).

* matrix_method        -- the eigenmode closed form of restricted diffusion in a plane, cylinder or sphere
* step_law             -- the walk's step law (the pair of lengths the engine draws) and the stepped matrix method

The q/b/g conversions and the analytical Watson and Bingham harmonics are dmipy-replay's
(``dmipy_replay.math.gradient_conversions``, ``dmipy_replay.math.sh_analytical``).
"""
from .step_law import (PAIR_SHORT, PAIR_LONG, PAIR_P_LONG, PAIR_LENGTHS, PAIR_WEIGHTS, LONGEST_STEP,
                       step_moments, bulk_multiplier, SteppedPore)

__all__ = [
    'PAIR_SHORT', 'PAIR_LONG', 'PAIR_P_LONG', 'PAIR_LENGTHS', 'PAIR_WEIGHTS', 'LONGEST_STEP',
    'step_moments', 'bulk_multiplier', 'SteppedPore',
]


def __getattr__(name):
    from .._moved import refuse
    raise refuse(__name__, name)
