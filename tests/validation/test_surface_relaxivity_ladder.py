"""Heavy GPU regression: surface relaxivity reproduces exact Robin theory, 1D->2D->3D.

Companion to ``test_permeability_ladder.py``.  Reuses the runnable example
``examples/validation/surface_relaxivity_1d_to_3d.py`` as the single source of truth
(lowest-Robin-eigenvalue solvers + MC survival-decay fit) and asserts each closed relaxing
cell's Monte-Carlo relaxation time against the exact finite-diffusion eigenvalue
``tau_1 = 1/(D lambda_1^2)`` (NOT the fast-limit ``rho*S/V``).

Marked ``slow`` (GPU, minutes): run with ``pytest -m slow``; the fast suite skips it via
``pytest -m 'not slow'``.  Requires scipy (Bessel functions).
"""
import os
import sys

import pytest

pytestmark = pytest.mark.slow

_EXAMPLES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                         "examples", "validation"))
sys.path.insert(0, _EXAMPLES)
ladder = pytest.importorskip("surface_relaxivity_1d_to_3d")

RHO = 20e-6         # m/s  -> rho*R/D ~ 0.1 (finite-diffusion regime)
TOL_TAU = 0.02      # 2% vs the exact Robin eigenvalue


def test_relaxivity_1d_slab():
    from dmipy_sim import Box1D
    L = 20e-6
    slab = Box1D(length=L, surface_relaxivity_t2=RHO)
    tau_ex = ladder.tau_slab(L, RHO)
    tau_mc = ladder.mc_relax_tau(slab, tau_ex, L / 2)
    assert abs(tau_mc - tau_ex) / tau_ex < TOL_TAU, (tau_mc, tau_ex)


def test_relaxivity_2d_cylinder():
    from dmipy_sim import Cylinder
    R = 10e-6
    cyl = Cylinder(radius=R, orientation=(0.0, 0.0, 1.0), surface_relaxivity_t2=RHO)
    tau_ex = ladder.tau_cyl(R, RHO)
    tau_mc = ladder.mc_relax_tau(cyl, tau_ex, R)
    assert abs(tau_mc - tau_ex) / tau_ex < TOL_TAU, (tau_mc, tau_ex)


def test_relaxivity_3d_sphere():
    from dmipy_sim import Sphere
    R = 10e-6
    sph = Sphere(radius=R, surface_relaxivity_t2=RHO)
    tau_ex = ladder.tau_sph(R, RHO)
    tau_mc = ladder.mc_relax_tau(sph, tau_ex, R)
    assert abs(tau_mc - tau_ex) / tau_ex < TOL_TAU, (tau_mc, tau_ex)
