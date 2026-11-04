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


# ── several membranes, each of its own permeability, and a permeability that changes in time ──────────────────────
WIDTHS = ((1.0, 1.0, 1.0), (1.0, 2.0, 1.0), (2.0, 1.0, 1.5))
PAIRS = ((0.03, 0.03), (0.03, 1.0), (1.0, 30.0), (30.0, 0.03), (1.0, 1.0), (30.0, 30.0))


@pytest.mark.parametrize("h", [0.18, 0.09])
def test_the_coefficient_bounds_the_exchange_of_membranes_of_their_own_permeability(h):
    """Three layers between reflecting walls, the two membranes each at its own ``kappa w_min / D`` from 0.03 to 30
    (``w_min`` the narrowest layer, whose slowest Neumann rate ``pi^2 D / w_min^2`` is the geometry's ``pore_rate``): the
    walk's slowest decay rate is within ``MEMBRANE_COEFFICIENT lambda_1 dt / 5`` of the exact root at every pair. The
    walk's error is local to each membrane, so the membranes' first-order terms add, each weighted by the mode's share
    at it; their sum stays under the one-membrane bound, which is therefore the bound of the largest permeability over
    the objects too (measured: at most 0.14 of ``lambda_1 dt / 5`` here, against 0.31 for one membrane)."""
    from dmipy_sim.math.step_law import LayeredLattice, layered_exchange_root
    worst = 0.0
    for widths in WIDTHS:
        L = LayeredLattice(widths, h * min(widths))
        lam1 = np.pi ** 2 / float(np.min(L.widths)) ** 2
        for pair in PAIRS:
            kod = np.asarray(pair) / min(widths)
            rel = L.slowest_rate(kod) / layered_exchange_root(L.widths, kod) - 1.0
            assert rel < 0, (widths, pair, rel)                       # the walk exchanges slower, as one membrane's
            worst = max(worst, -rel / (lam1 * L.dt / 5.0))
    assert worst <= MEMBRANE_COEFFICIENT, worst


def test_membranes_of_their_own_permeability_are_second_order():
    """Halving the step quarters the slowest rate's error with two membranes of ``kappa R / D`` 2 and 0.25 (layers
    ``R``, ``1.5 R``, ``R``), as it does with one."""
    from dmipy_sim.math.step_law import LayeredLattice, layered_exchange_root
    W, K = (1.0, 1.5, 1.0), np.array([2.0, 0.25])
    err = []
    for h in (0.09, 0.045):
        L = LayeredLattice(W, h)
        err.append(L.slowest_rate(K) / layered_exchange_root(L.widths, K) - 1.0)
    assert 3.6 < err[0] / err[1] < 4.4, err


def test_a_step_change_of_each_membranes_permeability_is_second_order():
    """The lattice walk of the slab of ``tests/geometry/test_permeable_layers.py`` (layers ``R``, ``1.5 R``, ``R``, the
    two membranes swapping ``kappa R / D`` 2 and 0.25 at ``t_b`` on a step boundary, the step reading its midpoint's
    permeability) against the 1-D Bloch-Torrey equation restarted at ``t_b``: the largest error of the layers'
    occupancy at 0.3 and 1 ``R^2 / D`` falls by more than three when the step halves (3.5, 3.7 from 0.18 to 0.09 to
    0.045 R), and the trial at the start's permeability throughout misses by far more than the walk's error."""
    bt = pytest.importorskip("dmipy_analytic.bloch_torrey")
    from dmipy_sim.math.step_law import LayeredLattice
    W, K1, K2 = (1.0, 1.5, 1.0), np.array([2.0, 0.25]), np.array([0.25, 2.0])
    start, times = (1.0, 0.0, 1.0), (0.3, 1.0)
    err, stuck = [], None
    for h in (0.09, 0.045):
        L = LayeredLattice(W, h)
        nb = int(round(0.1 / L.dt))
        n = [int(round(t / L.dt)) for t in times]
        occ = L.occupancy(start, lambda m: K1 if m < nb else K2, n)
        edges = np.concatenate([[0.0], np.cumsum(L.widths)])
        model = bt.BlochTorrey1D(edges, 1.0, permeability=(K1, K2), permeability_breakpoints=(nb * L.dt,),
                                 density=start, n_cells=800)
        ref = np.array([model.signal(np.zeros(m), L.dt, by_layer=True).real for m in n])
        err.append(np.abs(occ - ref).max())
        if stuck is None:
            stuck = np.abs(L.occupancy(start, lambda m: K1, n) - ref).max()
    assert err[0] / err[1] > 3.0, err
    assert stuck > 50 * err[0], (stuck, err)
