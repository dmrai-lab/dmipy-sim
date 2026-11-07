"""The boundary local time of a permeable wall counts every encounter that touches it, against the exact 1-D answer.

The substrate: a slab of two layers of ``R`` between reflecting walls and one membrane of ``kappa h / D = 0.5`` between
them (``h = 0.09 R`` the step), the membrane relaxing each face at ``rho R / D = 0.2``
(:class:`~dmipy_sim.geometry.PermeableSlab1D`), every walker uniform over the slab. The reference is the 1-D
Bloch-Torrey equation on the same layers with that membrane relaxivity
(:class:`dmipy_analytic.bloch_torrey.BlochTorrey1D`, ``membrane_relaxivity``), exact up to its cells (twice the
Richardson estimate of a refinement is added to the tolerance).

A walker's weight is ``exp(-(rho / D) L)`` with ``L`` the walk's boundary local time. The walk is at one step per block,
so a transmitted encounter is a step whose end lies across the membrane and its own increment is ``2 d_perp`` with
``d_perp`` the end's distance from the membrane: the local time that counts reflections only is the walk's less those.
That rule leaves the weight ~0.006 high at ``t = 0.1 R^2 / D`` and ~0.018 at ``0.3`` where the standard error is
2e-4 to 4e-4; the walk's rule holds the reference to the tolerance.
"""
import numpy as np
import pytest

from tests.conftest import se_tol

bt = pytest.importorskip("dmipy_analytic.bloch_torrey")

D = 2e-9
R = 5e-6
H = 0.09                                     # the step, in R
KAPPA_STAR = 0.5                             # kappa h / D
RHO_STAR = 0.2                               # rho R / D
TIMES = (0.1, 0.3)                           # in R^2 / D
N = 40_000
DT = (H * R) ** 2 / (6.0 * D)


def _reference(kappa, rho, step):
    """The exact weight after ``step`` steps of ``DT`` and its error (twice the Richardson estimate of the cells)."""
    m = [bt.BlochTorrey1D(np.array([0.0, 1.0, 2.0]) * R, D, permeability=(kappa,), density=(1.0, 1.0),
                          membrane_relaxivity=rho, n_cells=n) for n in (400, 800)]
    s = [float(x.signal(np.zeros(step), DT).real) for x in m]
    return s[1], 2.0 * abs(s[0] - s[1]) * 4.0 / 3.0


def _weights(walk, step, rho):
    """``(every touching encounter, reflections only)``: the weight ``exp(-(rho / D) L)`` of each walker at ``step``
    with the walk's local time and with the transmitted encounters' ``2 d_perp`` taken out."""
    x = np.asarray(walk.raw_positions, np.float64)[:, :step + 1, 0]
    L = -np.asarray(walk.raw_boundary, np.float64)[:, :step + 1].sum(1)
    across = (x[:, 1:] > R) != (x[:, :-1] > R)
    L_reflect = L - (2.0 * np.abs(x[:, 1:] - R) * across).sum(1)
    return np.exp(-rho / D * L), np.exp(-rho / D * L_reflect)


def test_local_time_counts_every_touching_encounter_and_is_the_continuum_at_kappa_star_one_half():
    from dmipy_sim.engine.core import _walk_geometry
    from dmipy_sim.geometry import PermeableSlab1D
    kappa, rho = KAPPA_STAR * D / (H * R), RHO_STAR * D / R
    steps = [int(round(t * R * R / D / DT)) for t in TIMES]
    g = PermeableSlab1D(length=2 * R, permeability=kappa, surface_relaxivity_t2=rho)
    w = _walk_geometry(N, D, g, max(steps) * DT, DT, sub_steps=1, seed=3, raw_subset=N, require_gpu=False)
    assert w.work["exhausted_steps"] == 0 and w.work["local_time_rule"] == "every_touching_encounter"
    for t, s in zip(TIMES, steps):
        ref, e_ref = _reference(kappa, rho, s)
        new, old = _weights(w, s, rho)
        tol = se_tol(N, new.std(), comparisons=len(TIMES)) + e_ref
        assert abs(new.mean() - ref) < tol, f"t = {t} R^2/D: {new.mean():.5f} against {ref:.5f} (tolerance {tol:.5f})"
        assert abs(old.mean() - ref) > 2.0 * tol, f"t = {t} R^2/D: reflections only {old.mean():.5f} against {ref:.5f}"
