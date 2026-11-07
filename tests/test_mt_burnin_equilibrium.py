"""The MT burn-in stops at the bound pool's equilibrium (#925).

The burn-in's rule is stated (:class:`~dmipy_sim.engine.mt.BurnInCriterion`): where the geometry gives the equilibrium
bound fraction ``P_eq = x / (1 + x)``, ``x = kappa_MT dwell S / V`` (:func:`~dmipy_sim.engine.mt.equilibrium_bound_fraction`),
the burn-in runs at least five relaxation times and stops with the ensemble's bound fraction within ``0.01 P_eq`` plus
two standard errors of it; where it gives none, a plateau test over a window of eight dwells. The walk records the
rule and where it stopped (``work["mt_equilibration"]``).

* The bound fraction at t = 0, pooled over seeds, is ``P_eq`` to its standard error: on the permeable slab, a sealed
  isolated cylinder, and permeable packed cylinders (``P_eq`` written here from the geometry's own lengths).
* The late signal of the slab is the exact chain's (``BlochTorrey1D`` with binding): the offset the early stop left
  (+0.0038, dmrai-lab/dmipy-sim#924) is within the standard error of the pooled seeds.
* A walk without MT is untouched and records no burn-in.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.engine import mt
from tests.conftest import se_tol
from tests import test_mt_permeable_walk as slab

SEEDS = (21, 22, 23, 24)
N = slab.N                                      # per seed


def _pooled(per_seed):
    """``(mean, sd of one walker)`` of the walkers of every seed."""
    x = np.concatenate(per_seed)
    return float(x.mean()), float(x.std())


def _t0_bound(geometry, n, steps, dt, D, seeds, **kw):
    runs = [_walk_geometry(n, D, geometry, steps * dt, dt, seed=s, raw_subset=n, require_gpu=False, **kw)
            for s in seeds]
    return [np.asarray(w.raw_bound, np.float64)[:, 0] for w in runs], runs[0]


def _assert_equilibrium(per_seed, walk, p_eq):
    mean, sd = _pooled(per_seed)
    tol = se_tol(sum(len(x) for x in per_seed), sd, comparisons=3)
    assert abs(mean - p_eq) < tol, f"bound fraction at t = 0 {mean:.5f} against its equilibrium {p_eq:.5f} (tolerance {tol:.5f})"
    rec = walk.work["mt_equilibration"]
    assert rec["kind"] == "equilibrium" and rec["converged"] and rec["p_eq"] == pytest.approx(p_eq, rel=1e-6)
    assert rec["chunks_run"] >= rec["min_chunks"]


@pytest.mark.parametrize("kappa_star", slab.KAPPA_STAR)
def test_the_slab_starts_at_its_equilibrium(kappa_star):
    """Both faces of the membrane bind in the slab's length: ``S / V = 2 / L = 1 / R``."""
    x = slab.K_ON * slab.TAU / slab.R
    per_seed, w0 = [], None
    for s in SEEDS:
        w = slab._walk(kappa_star * slab.D / (slab.H * slab.R), s)
        per_seed.append(np.asarray(w.raw_bound, np.float64)[:, 0])
        w0 = w0 or w
    _assert_equilibrium(per_seed, w0, x / (1.0 + x))


def test_the_sealed_cylinder_starts_at_its_equilibrium():
    R, D = slab.R, slab.D
    dt = slab.DT
    x = slab.K_ON * slab.TAU * 2.0 / R                        # S / V = 2 / R
    per_seed, w = _t0_bound(d.Cylinder(radius=R, orientation=(0, 0, 1)), 4000, 3, dt, D, SEEDS[:3],
                            kappa_MT=slab.K_ON, dwell_time=slab.TAU)
    _assert_equilibrium(per_seed, w, x / (1.0 + x))


def test_permeable_packed_cylinders_start_at_their_equilibrium():
    R, L, D, dt = 1.5e-6, 12e-6, 2e-9, 5e-5
    centres = np.array([[3e-6, 6e-6], [9e-6, 6e-6]])
    k_on, tau = 2e-4, 2.8e-3                                 # k_on tau S / V ~ 0.16: a bound fraction that resolves
    pack = d.PackedCylinders(np.array([R, R]), centres, L, permeability=4e-5)
    x = k_on * tau * (2.0 * 2.0 * np.pi * 2.0 * R) / L ** 2   # both faces of two cylinders in the cell of area L^2
    per_seed, w = _t0_bound(pack, 2000, 4, dt, D, SEEDS[:3], kappa_MT=k_on, dwell_time=tau)
    _assert_equilibrium(per_seed, w, x / (1.0 + x))


def test_sealed_packed_cylinders_weight_each_pool_by_its_seeding():
    R, L = 1.5e-6, 12e-6
    centres = np.array([[3e-6, 6e-6], [9e-6, 6e-6]])
    pack = d.PackedCylinders(np.array([R, R]), centres, L)
    k_on, tau = 4e-5, 1.4e-2
    a_in, a_out = 2.0 * np.pi * R ** 2, L ** 2 - 2.0 * np.pi * R ** 2
    x_in, x_out = k_on * tau * 2.0 / R, k_on * tau * 2.0 * 2.0 * np.pi * R / a_out
    expect = (a_in * x_in / (1 + x_in) + a_out * x_out / (1 + x_out)) / L ** 2
    assert mt.equilibrium_bound_fraction(pack, k_on, tau) == pytest.approx(expect, rel=1e-12)


def test_a_geometry_without_a_closed_form_is_burned_in_to_a_plateau():
    """A pool seeded alone in a permeable pack has no stationary closed form here: the plateau rule runs, is recorded,
    and the burn-in lasts at least its window."""
    R, L = 1.5e-6, 12e-6
    pack = d.PackedCylinders(np.array([R]), np.array([[6e-6, 6e-6]]), L, permeability=4e-5, pool="extra")
    assert mt.equilibrium_bound_fraction(pack, 2e-4, 2.8e-3) is None
    w = _walk_geometry(300, 2e-9, pack, 4 * 5e-5, 5e-5, seed=1, raw_subset=300, require_gpu=False,
                       kappa_MT=2e-4, dwell_time=2.8e-3)
    rec = w.work["mt_equilibration"]
    assert rec["kind"] == "plateau" and rec["p_eq"] is None and rec["chunks_run"] >= rec["window"]


def test_the_late_signal_of_the_slab_is_the_exact_chains():
    """The early stop left the walk +0.0038 above ``BlochTorrey1D`` with binding at the late time (#924); pooled over
    the seeds the offset is now within its standard error."""
    kappa = slab.KAPPA_STAR[0] * slab.D / (slab.H * slab.R)
    step = slab.STEPS[1]
    (ref, err), _ = slab._signals(kappa, step)
    sig = []
    for s in SEEDS:
        w = slab._walk(kappa, s)
        t_b = np.asarray(w.raw_bound, np.float64)[:, 1:step + 1].sum(1) * slab.DT
        sig.append(np.exp(-slab.R2B * t_b) * np.cos(slab.OMEGA_B * t_b))
    mean, sd = _pooled(sig)
    tol = se_tol(len(SEEDS) * N, sd) + err
    assert abs(mean - ref) < tol, f"late signal {mean:.5f} against the exact chain {ref:.5f} (tolerance {tol:.5f})"


def test_a_walk_without_mt_records_no_burn_in():
    cyl = d.Cylinder(radius=slab.R, orientation=(0, 0, 1))
    w = _walk_geometry(200, slab.D, cyl, 3 * slab.DT, slab.DT, seed=1, raw_subset=200, require_gpu=False)
    assert "mt_equilibration" not in w.work
