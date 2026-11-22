"""The placement's seed is a typed int32 draw, so a walker's start does not depend on JAX's 64-bit mode.

`jax.random.randint` without a `dtype` returns the process's default integer: int32 with the 64-bit mode off,
int64 with it on, and a different value for the same key. `geometry.base.placement_seed` types the draw `int32`,
which equals the default draw with the mode off. A start is a function of the seed, its index and the geometry
alone in both modes.
"""
from pathlib import Path

import jax
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import Compartments, PackedMyelinatedCylinders, Pool
from dmipy_sim.geometry.base import placement, placement_seed
from dmipy_sim.geometry.curved_cylinder import (CurvedCylinder, CurvedMyelinatedCylinder,
                                                PackedCurvedCylinders)

N_WALKERS = 300


def _straight_centreline(half_length=20e-6, n=9):
    # as tests/geometry/test_curved_cylinder.py
    return np.stack([np.zeros(n), np.zeros(n), np.linspace(-half_length, half_length, n)], axis=1)


def _octahedron(r=5e-6):
    """``(V, F)`` of a closed triangle mesh built here, no fixture file: an octahedron of circumradius ``r``."""
    V = np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]], float)
    F = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                  [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    return V, F


_FAT = np.array([[0.0, 0.0, -20e-6], [0.0, 0.0, 20e-6]])
_THIN = np.array([[1.8e-6, 0.0, -20e-6], [1.8e-6, 0.0, 20e-6]])

# One small instance of every geometry class that owns a placement seed line. Each construction is the one the
# existing tests of that class build (the file named in the comment).
GEOMETRIES = {
    "Sphere": lambda: d.Sphere(5e-6),                                                            # tests/geometry/test_sphere.py
    "Cylinder": lambda: d.Cylinder(radius=5e-6, orientation=(0, 0, 1)),                          # tests/geometry/test_cylinder.py
    "Ellipsoid": lambda: d.Ellipsoid(np.ones(3) * 5e-6),                                          # tests/geometry/test_ellipsoid.py
    "CurvedCylinder": lambda: CurvedCylinder(_straight_centreline(), radius=5e-6),                # tests/geometry/test_curved_cylinder.py
    "CurvedMyelinatedCylinder": lambda: CurvedMyelinatedCylinder(_straight_centreline(), r_in=3e-6, r_out=5e-6),  # tests/geometry/test_curved_cylinder.py
    "PackedCurvedCylinders": lambda: PackedCurvedCylinders([_FAT, _THIN], [2.0e-6, 0.5e-6], interior=True),        # tests/geometry/test_curved_cylinder.py
    "Mesh": lambda: d.Mesh(*_octahedron()),                                                       # built here, no file
    "MyelinatedCylinder": lambda: d.MyelinatedCylinder(inner_radius=3e-6, outer_radius=5e-6, orientation=(0, 0, 1),
                                                      compartments=Compartments(intra=Pool(D=2e-9), myelin=Pool(D=0.0),
                                                                                extra=Pool(D=2e-9))),   # tests/geometry/test_myelin.py
    "PackedMyelinatedCylinders": lambda: PackedMyelinatedCylinders(inner_radii=np.full(3, 3e-6), g_ratios=np.full(3, 0.7),
                                                                  centers=np.array([[0.0, 0.0], [12e-6, 0.0], [-12e-6, 0.0]]),
                                                                  cell_size=40e-6, N_max=3,
                                                                  compartments=Compartments(intra=Pool(D=2e-9), myelin=Pool(D=0.1e-9),
                                                                                            extra=Pool(D=2e-9))),  # tests/geometry/test_packed_myelinated_cylinders.py
    "PackedCylinders": lambda: d.PackedCylinders(radii=np.array([2e-6]), centers=np.array([[0.0, 0.0]]),
                                                L=10e-6, orientation=[0, 0, 1]),                  # tests/geometry/test_packed_cylinders.py
    "PackedSpheres": lambda: d.PackedSpheres([1.5e-6, 2e-6], [[0.0, 0.0, 0.0], [4e-6, 4e-6, 4e-6]], 10e-6),  # tests/test_walker_indexed_seeding.py
}


def test_the_placement_seed_is_the_same_in_both_modes(request):
    """``placement_seed(key)`` is the same int with JAX's 64-bit mode off and on, and with the mode off it is the
    value the untyped draw gives there -- the value the walks made in that mode use."""
    keys = [jax.random.PRNGKey(i) for i in (3, 17, 2026)]
    seed_off = [placement_seed(k) for k in keys]
    untyped_off = [int(jax.random.randint(k, (), 0, 2**30)) for k in keys]     # the mode is off here
    assert seed_off == untyped_off
    request.getfixturevalue("x64")                                             # the 64-bit mode on for the rest
    assert [placement_seed(k) for k in keys] == seed_off


@pytest.mark.parametrize("build", list(GEOMETRIES.values()), ids=list(GEOMETRIES))
def test_the_starts_do_not_depend_on_the_mode(build, request):
    """``placement(g, 300, key, 0)`` with JAX's 64-bit mode off and with it on places the same starts (as float32)
    and the same seeding labels. Each mode places a freshly built geometry, so the placement cache of one mode
    cannot answer for the other."""
    key = jax.random.PRNGKey(11)
    r0_off, labels_off = placement(build(), N_WALKERS, key, 0)
    request.getfixturevalue("x64")                                             # the 64-bit mode on for the rest
    r0_on, labels_on = placement(build(), N_WALKERS, key, 0)
    np.testing.assert_array_equal(np.asarray(r0_off, dtype=np.float32), np.asarray(r0_on, dtype=np.float32))
    if labels_off is None:
        assert labels_on is None
    else:
        np.testing.assert_array_equal(np.asarray(labels_off), np.asarray(labels_on))


def test_no_untyped_integer_draw_remains():
    """No line of the package's source draws ``jax.random.randint`` without a ``dtype``: every placement seed is a
    typed draw."""
    package = Path(d.__file__).resolve().parent
    offenders = []
    for path in sorted(package.rglob("*.py")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if "random.randint(" in line and "dtype" not in line:
                offenders.append(f"{path}:{number}: {line.strip()}")
    assert offenders == []
