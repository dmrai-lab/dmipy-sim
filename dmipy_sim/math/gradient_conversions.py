"""q/b/g conversions for idealised PGSE encoding (exact algebra).

Vendored into dmipy-sim from dmipy_fit.core.gradient_conversions so the sim
package is self-contained. The gyromagnetic ratio is dmipy-sim's own
:data:`dmipy_sim.constants.GAMMA` (= 267.513e6 rad/s/T, identical to the value
fit uses), so results match the fit implementation bit-for-bit.
"""
import numpy as np

from ..constants import GAMMA

__all__ = [
    'q_from_b', 'b_from_q', 'q_from_g', 'g_from_q', 'b_from_g', 'g_from_b',
]


def q_from_b(b, delta, Delta):
    """Compute q-value from b-value. Standard units."""
    tau = Delta - delta / 3
    return np.sqrt(b / tau) / (2 * np.pi)


def b_from_q(q, delta, Delta):
    """Compute b-value from q-value. Standard units."""
    tau = Delta - delta / 3
    return (q * (2 * np.pi)) ** 2 * tau


def q_from_g(g, delta, gyromagnetic_ratio=GAMMA):
    """Compute q-value from gradient strength. Standard units."""
    return g * delta * gyromagnetic_ratio / (2 * np.pi)


def g_from_q(q, delta, gyromagnetic_ratio=GAMMA):
    """Compute gradient strength from q-value. Standard units."""
    return q * (2 * np.pi) / (delta * gyromagnetic_ratio)


def b_from_g(g, delta, Delta, gyromagnetic_ratio=GAMMA):
    """Compute b-value from gradient strength. Standard units."""
    tau = Delta - delta / 3
    return (g * gyromagnetic_ratio * delta) ** 2 * tau


def g_from_b(b, delta, Delta, gyromagnetic_ratio=GAMMA):
    """Compute gradient strength from b-value. Standard units."""
    tau = Delta - delta / 3
    return np.sqrt(b / tau) / (gyromagnetic_ratio * delta)
