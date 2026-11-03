"""The membrane's coefficient of the step law (dmipy-sim#890): with the transmission ``x / (1 + x)`` the slowest exchange
mode of a permeable slab decays at a rate within ``MEMBRANE_COEFFICIENT * lambda_1 dt / 5`` of the exact one at every
permeability, ``lambda_1 = pi^2 D / R^2`` the slab's slowest mode -- second order in the step -- where the linear
``2 (kappa / D) d_perp`` is first order. Computed with the lattice walk of
:func:`~dmipy_sim.math.step_law.membrane_rate_error` (no Monte Carlo)."""
import numpy as np
import pytest

from dmipy_sim.engine.physics import MEMBRANE_COEFFICIENT
from dmipy_sim.math.step_law import membrane_rate_error

KAPPA_STARS = (0.01, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 4.0, 16.0, 50.0)


def _bound(h):
    """``MEMBRANE_COEFFICIENT lambda_1 dt / 5`` at ``h = sqrt(6 D dt)`` (units ``R = D = 1``)."""
    return MEMBRANE_COEFFICIENT * np.pi ** 2 * (h * h / 6.0) / 5.0


@pytest.mark.parametrize("h", [0.18, 0.09, 0.045])
def test_the_coefficient_bounds_the_exchange_rate_at_every_permeability(h):
    """Over ``kappa R / D`` from 0.01 to 50 the walk's exchange is slower than the exact one by at most the bound, and
    at the finest step the bound is within a tenth of the largest error: the coefficient is the sup, not a margin."""
    err = np.array([membrane_rate_error(k, h) for k in KAPPA_STARS])
    assert (err < 0).all() and (-err <= _bound(h)).all(), dict(zip(KAPPA_STARS, err / _bound(h)))
    if h == 0.045:
        assert -err.min() >= 0.9 * _bound(h)


def _linear(kod, d):
    return np.minimum(1.0, 2.0 * kod * d)


def test_the_linear_transmission_is_first_order_and_x_over_one_plus_x_second():
    """At ``kappa R / D = 1`` halving the step halves the linear rule's rate error (+6 % at ``h = 0.09 R``, an
    effective permeability too large by ``1.2 kappa h / D``) and quarters that of ``x / (1 + x)``."""
    lin = [membrane_rate_error(1.0, h, transmission=_linear) for h in (0.09, 0.045)]
    pad = [membrane_rate_error(1.0, h) for h in (0.09, 0.045)]
    assert lin[0] > 0.05 and 1.8 < lin[0] / lin[1] < 2.2
    assert 3.5 < pad[0] / pad[1] < 4.5 and abs(pad[0]) < lin[0] / 50
