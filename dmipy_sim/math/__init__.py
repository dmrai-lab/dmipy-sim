"""Self-contained math utilities for dmipy-sim (the forward truth).

These were previously imported from dmipy-fit; they live here now so dmipy-sim
imports nothing from dmipy-fit (the dependency runs fit -> sim, one direction).

* gradient_conversions -- q/b/g conversions (exact PGSE algebra)
* sh_analytical        -- exact Watson-ODF spherical-harmonic coefficients
"""
from .gradient_conversions import (
    q_from_b, b_from_q, q_from_g, g_from_q, b_from_g, g_from_b,
)
from .sh_analytical import watson_zonal_ratios, watson_sh

__all__ = [
    'q_from_b', 'b_from_q', 'q_from_g', 'g_from_q', 'b_from_g', 'g_from_b',
    'watson_zonal_ratios', 'watson_sh',
]
