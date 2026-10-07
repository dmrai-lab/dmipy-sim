"""Regression test for dmrai-lab/dmipy-sim#627.

``Sphere``/``Cylinder``/``Ellipsoid.init_positions`` and the packed family's
``_seed_periodic`` all accept a candidate point by a membership test run in float64, then cast
the accepted batch to float32. A draw within half a float32 ulp of the wall is strictly inside
by the float64 test (so it is kept) but its nearest float32 representable value sits ON or
PAST the wall -- and the first step's sentinel (``inside0 = dot(r, r) < R*R``, computed on the
very same stored float32 value) then reads it as the far compartment for good: #627 found one
such walker in 500,000 on a live 2 um cylinder.

The bug is 1-in-500,000 at that radius, so a plain random seed-and-classify test is a weak,
flaky detector of it. These tests instead construct the ingredient directly: a float64 value
``x64 < R`` whose perturbation away from ``R`` (``R * 1e-12``) is ~10^5 times smaller than a
float32 ulp at that magnitude (~1.2e-7 relative) but easily resolved in float64 (~2.2e-16
relative) -- so ``x64`` is unambiguously accepted by the float64 membership test, while
``np.float32(x64)`` rounds to exactly ``np.float32(R)``, the wall itself. The RNG each seeder
uses is monkeypatched to hand out that one point as its very first candidate; everything else
about ``init_positions`` runs untouched.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_sim.geometry import analytic, packed
from dmipy_sim.geometry.analytic import Sphere, Cylinder, Ellipsoid
from dmipy_sim.geometry.packed import PackedCylinders, PackedSpheres

_REAL_DEFAULT_RNG = np.random.default_rng   # captured before any test monkeypatches it


def _round_onto_wall(W):
    """A float64 value strictly less than ``W`` whose nearest float32 is ``np.float32(W)``.

    The perturbation (1e-12 relative) is ~5 orders of magnitude below a float32 ulp (~1.2e-7
    relative) but ~4 orders above a float64 ulp (~2.2e-16 relative): resolved as "less than W"
    in float64, invisible once rounded to float32.
    """
    x64 = np.float64(W) * (1.0 - 1e-12)
    assert x64 < W, "construction failed: not strictly inside in float64"
    assert np.float32(x64) == np.float32(W), "construction failed: did not round onto the wall"
    return x64


class _FixedFirstDraw:
    """Stand-in for ``np.random.Generator``: the first ``uniform`` call plants ``first_rows`` at
    the start of the batch (so they are guaranteed to be first after rejection filtering, which
    preserves order), padding the rest with real draws from ``fallback`` so the batch still has
    enough accepted points to finish in one pass. Any further call is pure ``fallback``."""

    def __init__(self, first_rows, fallback_seed=12345):
        self._first_rows = np.asarray(first_rows, dtype=np.float64)
        self._used = False
        # the real generator, captured once -- `default_rng` itself may be monkeypatched by
        # the time this is constructed (it is built inside the patched factory function)
        self._fallback = _REAL_DEFAULT_RNG(fallback_seed)

    def uniform(self, lo, hi, size):
        out = self._fallback.uniform(lo, hi, size)
        if not self._used:
            self._used = True
            out[: self._first_rows.shape[0]] = self._first_rows
        return out


@pytest.mark.parametrize("name,R,make_geom,first_row", [
    pytest.param("Sphere", 2e-6, lambda R: Sphere(radius=R),
                 lambda x: [x, 0.0, 0.0], id="Sphere"),
    pytest.param("Cylinder", 2e-6, lambda R: Cylinder(radius=R, orientation=(0, 0, 1)),
                 lambda x: [x, 0.0], id="Cylinder"),
])
def test_analytic_seed_on_float32_wall_stays_inside(monkeypatch, name, R, make_geom, first_row):
    x = _round_onto_wall(R)
    monkeypatch.setattr(analytic.np.random, "default_rng",
                        lambda seed: _FixedFirstDraw([first_row(x)]))
    geom = make_geom(R)
    pos = geom.init_positions(1, jax.random.PRNGKey(0))
    label = int(geom.classify_position(pos[0]))
    assert label == 1, (
        f"{name}: a walker accepted inside in float64 (x = R*(1 - 1e-12)) was classified "
        f"EXTRA by the geometry's own classifier after the float32 cast -- #627")


def test_ellipsoid_seed_on_float32_wall_stays_inside(monkeypatch):
    """Ellipsoid samples the UNIT ball then scales by the semiaxes, so the rounding happens on
    the unit-ball coordinate, not on the semiaxis itself: the same construction with ``W = 1``."""
    semiaxes = [2e-6, 3e-6, 1.5e-6]
    u = _round_onto_wall(1.0)   # unit-ball coordinate, accepted at float64 but rounds to 1.0 in float32
    monkeypatch.setattr(analytic.np.random, "default_rng",
                        lambda seed: _FixedFirstDraw([[u, 0.0, 0.0]]))
    geom = Ellipsoid(semiaxes=semiaxes)
    pos = geom.init_positions(1, jax.random.PRNGKey(0))
    label = int(geom.classify_position(pos[0]))
    assert label == 1, (
        "Ellipsoid: a walker accepted inside the unit ball at float64 was classified EXTRA "
        "by the geometry's own classifier after the float32 cast and semiaxis scaling -- #627")


@pytest.mark.parametrize("name,Geom,ndim,extra_kw", [
    pytest.param("PackedCylinders", PackedCylinders, 2, dict(orientation=(0, 0, 1.0)),
                 id="PackedCylinders"),
    pytest.param("PackedSpheres", PackedSpheres, 3, {}, id="PackedSpheres"),
])
def test_packed_intra_seed_on_float32_wall_stays_inside(monkeypatch, name, Geom, ndim, extra_kw):
    """Same ingredient, through ``_seed_periodic`` (packed.py): one cylinder/sphere of radius R
    at the cell origin, pool='intra', first candidate at (R*(1 - 1e-12), 0, ...)."""
    R, L = 2e-6, 50e-6
    x = _round_onto_wall(R)
    first_row = [x] + [0.0] * (ndim - 1)
    monkeypatch.setattr(packed.np.random, "default_rng",
                        lambda seed: _FixedFirstDraw([first_row]))
    geom = Geom(radii=[R], centers=np.zeros((1, ndim)), L=L, pool="intra", **extra_kw)
    pos = geom.init_positions(1, jax.random.PRNGKey(0))
    label = int(geom.classify_position(pos[0]))
    assert label > 0, (
        f"{name}: a walker accepted intra at float64 (x = R*(1 - 1e-12)) was classified "
        f"EXTRA (id 0) by the geometry's own classifier after the float32 cast -- #627")


# ── belt-and-suspenders: many walkers, every one must classify as seeded ─────────────────────
# Not the primary regression (the bug is 1-in-500,000 at this radius, so this alone would be a
# flaky detector of it -- the constructed tests above are the real net), but a cheap general
# check that nothing else about the fix broke bulk seeding.

@pytest.mark.parametrize("name,geom,want_label", [
    pytest.param("Sphere", Sphere(radius=2e-6), 1, id="Sphere"),
    pytest.param("Cylinder", Cylinder(radius=2e-6, orientation=(0, 0, 1)), 1, id="Cylinder"),
    pytest.param("Ellipsoid", Ellipsoid(semiaxes=[2e-6, 3e-6, 1.5e-6]), 1, id="Ellipsoid"),
])
def test_bulk_seed_all_classify_as_seeded(name, geom, want_label):
    n = 20_000
    pos = geom.init_positions(n, jax.random.PRNGKey(7))
    labels = np.asarray(jax.vmap(geom.classify_position)(pos))
    n_wrong = int((labels != want_label).sum())
    assert n_wrong == 0, f"{name}: {n_wrong}/{n} seeded walkers misclassified by their own geometry"
