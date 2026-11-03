"""The permeable walk's exchange against the exact answer of the two-compartment slab (dmipy-sim#890).

The slab: compartments of half-width ``R`` on either side of a membrane of permeability ``kappa``, reflecting outer
walls, every walker starting in A. The exact occupancy of A is the series of the antisymmetric modes,
``k tan(k R) = 2 kappa / D`` (the symmetric ones carry no difference between the compartments), which
``dmipy_analytic.bloch_torrey.BlochTorrey1D`` reproduces to 3e-7 on the same slab.

The transmission ``x / (1 + x)`` makes the walk's error second order in the step: at ``kappa R / D = 2`` and the coarse
step ``h = 0.18 R`` the occupancy is within 2e-3 of the exact one (computed with the lattice walk of
:func:`dmipy_sim.math.step_law.membrane_rate_error`'s construction), where the linear transmission
``2 (kappa / D) d_perp`` is 0.011, 0.022 and 0.032 low at ``t = 0.03, 0.1, 0.3 R^2 / D``: the walker count is the one
at which that error is twice the tolerance (``n_resolving``).
"""
import numpy as np
import pytest

from tests.conftest import n_resolving, se_tol

D = 2e-9
R = 5e-6
KAPPA_STAR = 2.0
H = 0.18                                   # the step over R
TIMES = (0.03, 0.1, 0.3)                   # in R^2 / D
LINEAR_ERROR = 0.011                       # the linear rule's smallest error over TIMES, from the lattice walk


def exact_occupancy(kappa_star, t, n_modes=400):
    """The fraction of the walkers in A at ``t`` (units ``R^2 / D``) when every walker starts uniform in A: ``1/2`` plus
    half the difference of the compartments, ``sum_n b_n sin(k_n) / k_n exp(-k_n^2 t)`` over the roots ``k_n`` of
    ``k tan k = 2 kappa*``, ``b_n`` the projection of the uniform start on ``cos(k_n x)`` over ``[0, 1]``."""
    from scipy.optimize import brentq
    k = np.array([brentq(lambda z: z * np.tan(z) - 2.0 * kappa_star, n * np.pi + 1e-12, n * np.pi + np.pi / 2 - 1e-12)
                  for n in range(n_modes)])
    b = (np.sin(k) / k) / (0.5 + np.sin(2.0 * k) / (4.0 * k))
    return 0.5 + 0.5 * float(np.sum(b * np.sin(k) / k * np.exp(-k * k * t)))


def test_the_series_is_the_two_compartment_law():
    """At small ``kappa`` the series is the closed two-compartment exchange ``1/2 + 1/2 exp(-2 kappa t / R)`` the slab
    states (its rate low by the diffusion-limited ``2 kappa R / 3 D`` of itself), and at ``t = 0`` every walker is in A."""
    assert exact_occupancy(KAPPA_STAR, 0.0, n_modes=4000) == pytest.approx(1.0, abs=2e-4)
    for t in (10.0, 100.0, 500.0):
        assert 2.0 * exact_occupancy(1e-3, t) - 1.0 == pytest.approx(np.exp(-2e-3 * t), rel=1e-3)


def test_the_walk_exchanges_as_the_exact_slab():
    """The walk's occupancy of A at ``TIMES`` against the series, at the coarse step: every one within ``se_tol`` (the
    three comparisons, the binomial spread of a fraction), a tolerance the linear transmission's error is twice of."""
    from dmipy_sim.geometry import PermeableSlab1D
    from dmipy_sim.engine.core import _walk_geometry
    sd = 0.5
    n = n_resolving(LINEAR_ERROR, sd, comparisons=len(TIMES))
    tol = se_tol(n, sd, comparisons=len(TIMES))
    assert 2.0 * tol <= LINEAR_ERROR * (1 + 1e-9)
    dt = (H * R) ** 2 / (6.0 * D)
    n_blocks = int(round(max(TIMES) * R * R / D / dt))
    g = PermeableSlab1D(length=2 * R, permeability=KAPPA_STAR * D / R)
    w = _walk_geometry(n, D, g, n_blocks * dt, dt, sub_steps=1, seed=7, raw_subset=n, require_gpu=False)
    lab = np.asarray(w.raw_compartment)                                  # (n, n_blocks + 1), 1 = A
    for t in TIMES:
        i = int(round(t * R * R / D / dt))
        got = float((lab[:, i] == 1).mean())
        want = exact_occupancy(KAPPA_STAR, i * dt * D / R ** 2)
        assert abs(got - want) < tol, f"t = {t} R^2/D: occupancy {got:.4f} against the exact {want:.4f} (tolerance {tol:.4f})"
