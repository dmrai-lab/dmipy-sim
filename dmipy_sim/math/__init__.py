"""Self-contained math utilities for dmipy-sim (the forward truth).

These were previously imported from dmipy-fit; they live here now so dmipy-sim
imports nothing from dmipy-fit (the dependency runs fit -> sim, one direction).

* gradient_conversions -- q/b/g conversions (exact PGSE algebra)
* sh_analytical        -- exact Watson and Bingham spherical-harmonic coefficients
* matrix_method        -- the eigenmode closed form of restricted diffusion in a plane, cylinder or sphere
* step_law             -- the walk's step law (the pair of lengths the engine draws) and the stepped matrix method
"""
from dmipy_replay.math.gradient_conversions import q_from_b, b_from_q, q_from_g, g_from_q, b_from_g, g_from_b
from .step_law import (PAIR_SHORT, PAIR_LONG, PAIR_P_LONG, PAIR_LENGTHS, PAIR_WEIGHTS, LONGEST_STEP,
                       step_moments, bulk_multiplier, SteppedPore)
from .sh_analytical import (watson_zonal_ratios, watson_sh, bingham_normalization,
                            bingham_canonical_sh, bingham_sh)

__all__ = [
    'q_from_b', 'b_from_q', 'q_from_g', 'g_from_q', 'b_from_g', 'g_from_b',
    'watson_zonal_ratios', 'watson_sh',
    'PAIR_SHORT', 'PAIR_LONG', 'PAIR_P_LONG', 'PAIR_LENGTHS', 'PAIR_WEIGHTS', 'LONGEST_STEP',
    'step_moments', 'bulk_multiplier', 'SteppedPore',
]
