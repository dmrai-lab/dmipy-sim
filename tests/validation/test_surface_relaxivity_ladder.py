"""GPU regression: surface relaxivity reproduces exact Robin theory in a sphere.

Reuses the runnable example
``examples/validation/surface_relaxivity_1d_to_3d.py`` as the single source of truth
(lowest-Robin-eigenvalue solvers + MC survival-decay fit) and asserts each closed relaxing
cell's Monte-Carlo relaxation time against the exact finite-diffusion eigenvalue
``tau_1 = 1/(D lambda_1^2)`` (NOT the fast-limit ``rho*S/V``).

Marked ``gpu`` (tier 2); the CPU suite skips it via ``pytest -m "not gpu"``.  Requires scipy (Bessel functions).
"""
import os
import sys

import pytest

pytestmark = pytest.mark.gpu

from tests.conftest import n_resolving, se_tol

_EXAMPLES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                         "examples", "validation"))
sys.path.insert(0, _EXAMPLES)
ladder = pytest.importorskip("surface_relaxivity_1d_to_3d")

RHO = 20e-6         # m/s  -> rho*R/D ~ 0.1 (finite-diffusion regime)
TOL_TAU = 0.02      # 2% vs the exact Robin eigenvalue

# The fit of the five survival times (1 to 3 tau) has the relative standard deviation SD_TAU / sqrt(N): 0.22 measured on
# the sphere (five seeds of 600 walkers, 0.89 % each), taken as 0.25. The walkers' weights spread by about half their mean,
# and the windows share their walkers' seed, so they share most of the noise. A 4 % error in the relaxation time fails
# the 2 % tolerance with the margin the noise passes it at N = n_resolving(0.04, SD_TAU) = 1,700 walkers, where the walk
# takes steps of R/12 (the same time to 0.5 %, measured: mean of five seeds -0.4 %) instead of R/50.
SD_TAU = 0.25
STEP_FRACTION = 12


def test_relaxivity_3d_sphere(monkeypatch):
    from dmipy_sim import Sphere
    n = n_resolving(2 * TOL_TAU, SD_TAU)
    monkeypatch.setattr(ladder, "NW", n)
    R = 10e-6
    sph = Sphere(radius=R, surface_relaxivity_t2=RHO)
    tau_ex = ladder.tau_sph(R, RHO)
    tau_mc = ladder.mc_relax_tau(sph, tau_ex, R, step_fraction=STEP_FRACTION)
    assert abs(tau_mc - tau_ex) / tau_ex < se_tol(n, SD_TAU), (tau_mc, tau_ex, n)
