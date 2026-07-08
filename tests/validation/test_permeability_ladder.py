"""Heavy GPU regression: membrane permeability reproduces exact analytics, 1D->2D->3D.

This is the first-principles validation ladder (see examples/validation/
permeability_findings.md) as an ASSERTED test.  It reuses the runnable example
`examples/validation/permeability_1d_to_3d.py` as the single source of truth (eigenvalue
solvers + MC exchange), and checks each closed-cell geometry's Monte-Carlo exchange time
against the exact finite-diffusion eigenvalue plus the equilibrium partition (detailed
balance).

Marked ``slow`` (GPU, minutes): run with ``pytest -m slow``; the fast suite skips it via
``pytest -m 'not slow'``.  Requires scipy (Bessel functions).
"""
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.slow

_EXAMPLES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                         "examples", "validation"))
sys.path.insert(0, _EXAMPLES)
ladder = pytest.importorskip("permeability_1d_to_3d")   # skips if scipy/example missing

KAPPA = 20e-6
TOL_TAU = 0.02      # 2% vs exact eigenvalue (500k walkers -> ~1% noise)
TOL_EQ = 0.02       # detailed balance: equilibrium partition


def test_permeability_1d_slab():
    from dmipy_sim import PermeableSlab1D
    L = 20e-6
    slab = PermeableSlab1D(length=L, permeability=KAPPA)
    tau_ex = ladder.tau_slab(L, KAPPA)
    tau_mc, f_eq = ladder.mc_exchange_tau(slab, tau_ex, 0.5, L / 2)
    assert abs(tau_mc - tau_ex) / tau_ex < TOL_TAU, (tau_mc, tau_ex)
    assert abs(f_eq - 0.5) < TOL_EQ, f_eq


@pytest.mark.parametrize("kind,feq_th", [("cylinder", 0.25), ("sphere", 0.125)])
def test_permeability_shell(kind, feq_th):
    from dmipy_sim.geometries import PermeableShell
    R = 10e-6
    geom = PermeableShell(R, 2 * R, KAPPA, kind=kind)
    tau_ex = ladder._tau_shell(R, 2 * R, KAPPA, kind)
    tau_mc, f_eq = ladder.mc_exchange_tau(geom, tau_ex, feq_th, R)
    assert abs(tau_mc - tau_ex) / tau_ex < TOL_TAU, (kind, tau_mc, tau_ex)
    assert abs(f_eq - feq_th) < TOL_EQ, (kind, f_eq, feq_th)
