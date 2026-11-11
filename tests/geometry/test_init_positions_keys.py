"""Geometry placement consumes each JAX key at most once, including split parents."""

from functools import wraps
import inspect

import jax
import numpy as np
import pytest

from dmipy_sim import Compartments, Pool
from dmipy_sim import geometry as g
from dmipy_sim.geometry import mesh_shapes
from dmipy_sim.geometry.base import placement


def geometries():
    """Small instances of every concrete geometry, plus both shell draw paths."""
    cl = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10e-6]])
    pools = Compartments(intra=Pool(D=2e-9), myelin=Pool(D=0.0), extra=Pool(D=2e-9))
    vertices, faces = mesh_shapes.icosphere(3e-6, subdivisions=1)
    labels = np.ones((4, 4, 4), np.uint8)
    labels[1:3] = 0
    return [
        g.FreeDiffusion(),
        g.Box1D(4e-6),
        g.Sphere(3e-6),
        g.Cylinder(3e-6, (0, 0, 1)),
        g.Ellipsoid((2e-6, 3e-6, 4e-6)),
        g.PermeableSlab1D(6e-6, 1e-5),
        g.PermeableSlab1D(6e-6, 1e-5, membranes=[2e-6, 4e-6]),
        g.PermeableShell(2e-6, 4e-6, 1e-5, kind="sphere"),
        g.PermeableShell(2e-6, 4e-6, 1e-5, kind="cylinder"),
        g.PackedCylinders([1e-6], [[0.0, 0.0]], 6e-6),
        g.PackedSpheres([1e-6], [[0.0, 0.0, 0.0]], 6e-6),
        g.MyelinatedCylinder(2e-6, 3e-6, (0, 0, 1), compartments=pools),
        g.PackedMyelinatedCylinders([1e-6], 0.7, [[0.0, 0.0]], 6e-6,
                                   N_max=1, compartments=pools),
        g.CurvedCylinder(cl, 1e-6),
        g.CurvedMyelinatedCylinder(cl, 1e-6, 1.5e-6),
        g.PackedCurvedCylinders([cl], [1e-6]),
        g.Mesh(vertices, faces, feature_radius=1e-6),
        g.SphereUnion([[0.0, 0.0, 0.0]], [2e-6]),
        g.LabelVolume(labels, 1e-6),
    ]


@pytest.mark.parametrize("key_factory", [jax.random.PRNGKey, jax.random.key],
                         ids=["legacy", "typed"])
@pytest.mark.parametrize("through_placement", [False, True], ids=["direct", "placement"])
def test_init_positions_consumes_each_key_once(monkeypatch, key_factory, through_placement):
    instances = geometries()
    concrete = {cls for cls in vars(g).values()
                if inspect.isclass(cls) and issubclass(cls, g.Geometry)
                and not inspect.isabstract(cls)}
    assert {type(instance) for instance in instances} == concrete

    consumed = set()

    def record(function):
        @wraps(function)
        def checked(key, *args, **kwargs):
            bits = tuple(np.asarray(jax.random.key_data(key)).ravel())
            assert bits not in consumed, (
                f"{type(instance).__name__}: {function.__name__} reused a consumed key: {bits}")
            consumed.add(bits)
            return function(key, *args, **kwargs)
        return checked

    # Record the public primitive call once; their internal implementations are untouched.
    # Splitting consumes its parent just as a draw does, so drawing from a split parent
    # or drawing twice fails deterministically, independent of sample statistics.
    for name in ("split", "normal", "uniform", "randint", "permutation"):
        monkeypatch.setattr(jax.random, name, record(getattr(jax.random, name)))

    for instance in instances:
        consumed.clear()
        if through_placement:
            positions, _ = placement(instance, 32, key_factory(42))
        else:
            positions = instance.init_positions(32, key_factory(42))
        assert positions.shape == (32, 3), type(instance).__name__
        assert np.isfinite(np.asarray(positions)).all(), type(instance).__name__
