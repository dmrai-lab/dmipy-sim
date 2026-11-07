"""The MT walk on permeable packed cylinders, a permeability over time, and the forward vector-Bloch oracle (#909).

* A cylinder in a periodic cell, with the walker's side carried, walks the same MT physics as the isolated
  :class:`~dmipy_sim.geometry.Cylinder`: the bound fraction and the fraction inside at the end agree to the sampling.
* A permeability per object: the sealed cylinder keeps every walker inside and binds as the sealed isolated cylinder
  does; the permeable one is the permeable isolated cylinder.
* A permeability over time on the slab: the walk against ``BlochTorrey1D`` with the same breakpoints, with bound pools.
* The forward vector-Bloch oracle on the slab against ``BlochTorrey1D``: the complex signal of an excited ensemble.

Tolerances are ``se_tol`` over the family of comparisons of each test.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.geometry.base import PermeabilitySchedule
from tests.conftest import se_tol

D = 2e-9
R = 2e-6
L = 40e-6                                   # the periodic cell, 20 R: a walker does not meet its image in T
T = 6e-3
DT = 5e-5
KAPPA = 4e-5                                # kappa h / D ~ 0.5 at the step law's h
KON = 4e-5                                  # the binding reactivity, m/s
TAU = 2e-3                                  # the dwell, s
N = 4000
CENTRES = np.array([[10e-6, 20e-6], [30e-6, 20e-6]])


def _inside_start(rng, n, centre):
    rho = R * np.sqrt(rng.uniform(0.0, 1.0, n))
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    r0 = np.zeros((n, 3), np.float32)
    r0[:, 0] = centre[0] + rho * np.cos(phi)
    r0[:, 1] = centre[1] + rho * np.sin(phi)
    return r0


def _walk(geometry, r0, seed):
    return _walk_geometry(len(r0), D, geometry, T, DT, seed=seed, raw_subset=len(r0), require_gpu=False, r0=r0,
                          kappa_MT=KON, dwell_time=TAU, equilibrate_binding="off")


def _stats(w, centre, periodic):
    """``(inside at the end, bound at the end)``, per walker, relative to the cylinder axis at ``centre``."""
    x = np.asarray(w.raw_positions, np.float64)[:, -1, :2] - np.asarray(centre)
    if periodic:
        x = (x + L / 2.0) % L - L / 2.0
    inside = (np.hypot(x[:, 0], x[:, 1]) < R).astype(float)
    bound = np.asarray(w.raw_bound, np.float64)[:, -1]
    return inside, bound


def _agree(a, b, comparisons, what):
    """Two samples of one statistic agree to ``se_tol`` of their pooled spread."""
    sd = np.sqrt(0.5 * (a.var() + b.var()))
    tol = se_tol(len(a), sd, comparisons=comparisons, runs=2)
    assert abs(a.mean() - b.mean()) < tol, f"{what}: {a.mean():.4f} against {b.mean():.4f} (tolerance {tol:.4f})"


def test_a_packed_cylinder_with_its_side_carried_is_the_isolated_cylinder():
    rng = np.random.default_rng(1)
    pack = d.PackedCylinders(np.array([R]), CENTRES[:1], L, permeability=KAPPA)
    iso = d.Cylinder(radius=R, orientation=(0, 0, 1), permeability=KAPPA)
    w_pack = _walk(pack, _inside_start(rng, N, CENTRES[0]), 2)
    w_iso = _walk(iso, _inside_start(rng, N, (0.0, 0.0)), 3)
    ip, bp = _stats(w_pack, CENTRES[0], True)
    ii, bi = _stats(w_iso, (0.0, 0.0), False)
    assert 0.2 < ip.mean() < 0.95 and bp.mean() > 0.05              # it crosses, it binds: the comparison resolves both
    _agree(ip, ii, 2, "fraction inside")
    _agree(bp, bi, 2, "fraction bound")
    assert w_pack.work["exhausted_steps"] == 0


def test_a_permeability_per_object_is_each_cylinders_own():
    rng = np.random.default_rng(4)
    pack = d.PackedCylinders(np.array([R, R]), CENTRES, L, permeability=[KAPPA, 0.0])
    r0 = np.concatenate([_inside_start(rng, N, CENTRES[0]), _inside_start(rng, N, CENTRES[1])])
    w = _walk(pack, r0, 5)
    leaky = slice(0, N)
    sealed = slice(N, 2 * N)
    # the walks of the two populations, one walk of the pack: the sealed cylinder keeps every walker inside
    ip, bp = _stats(type("W", (), dict(raw_positions=w.raw_positions[leaky], raw_bound=w.raw_bound[leaky])), CENTRES[0], True)
    isl, bsl = _stats(type("W", (), dict(raw_positions=w.raw_positions[sealed], raw_bound=w.raw_bound[sealed])), CENTRES[1], True)
    assert isl.min() == 1.0
    iso_leaky = _walk(d.Cylinder(radius=R, orientation=(0, 0, 1), permeability=KAPPA), _inside_start(rng, N, (0.0, 0.0)), 6)
    iso_sealed = _walk(d.Cylinder(radius=R, orientation=(0, 0, 1)), _inside_start(rng, N, (0.0, 0.0)), 7)
    il, bl = _stats(iso_leaky, (0.0, 0.0), False)
    _, bs = _stats(iso_sealed, (0.0, 0.0), False)
    assert bs.mean() > 0.05
    _agree(ip, il, 4, "leaky: fraction inside")
    _agree(bp, bl, 4, "leaky: fraction bound")
    _agree(bsl, bs, 4, "sealed: fraction bound")
