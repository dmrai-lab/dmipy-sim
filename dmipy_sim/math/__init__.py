"""The mathematics of the walk that dmipy-sim owns.

* step_law             -- the walk's step law (the pair of lengths the engine draws) and the stepped matrix method

The q/b/g conversions are dmipy-replay's (``dmipy_replay.math.gradient_conversions``); the eigenmode (matrix) closed
form of restricted diffusion and the Watson and Bingham harmonics are dmipy-analytic's
(``dmipy_analytic.restricted.matrix_method``, ``dmipy_analytic.orientation``).
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
