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


# --- the placement cache under concurrency ------------------------------------------------------------------------
# A producer places the next walk's walkers in a thread while the current walk runs; these hold the cache to its
# contract: a cached placement never waits for another key's computation, one key is computed once whoever asks for
# it, a failed computation leaves nothing behind, and the arrays do not depend on which thread computed them.


class _StandIn:
    """A geometry as `placement` sees it: an `init_positions` that counts its calls, can be held on an event and can
    fail once."""

    def __init__(self):
        import threading
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.fail_next = False

    def init_positions(self, n, key):
        self.calls += 1
        self.started.set()
        assert self.release.wait(10.0), "the stand-in was never released"
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("the stand-in's seeding failed")
        return np.asarray(jax.random.uniform(key, (n, 3)), np.float32)


def _in_thread(function, *args):
    """Run ``function(*args)`` in a thread; return the thread and a dict that receives its result or exception."""
    import threading
    out = {}

    def run():
        try:
            out["result"] = function(*args)
        except BaseException as error:                                   # handed to the test
            out["error"] = error
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, out


def test_a_cached_placement_does_not_wait_for_another_being_computed():
    import time
    geo = _StandIn()
    cached = placement(geo, 16, jax.random.PRNGKey(1))
    geo.release.clear()
    geo.started.clear()
    thread, out = _in_thread(placement, geo, 16, jax.random.PRNGKey(2))
    assert geo.started.wait(10.0)                                        # key 2 is being computed
    try:
        t0 = time.perf_counter()
        hit = placement(geo, 16, jax.random.PRNGKey(1))
        waited = time.perf_counter() - t0
    finally:
        geo.release.set()
        thread.join(10.0)
    assert hit is cached
    assert waited < 0.05, f"a cached placement waited {waited:.3f} s for another key's computation"
    assert "result" in out and geo.calls == 2


def test_one_key_asked_from_eight_threads_is_computed_once():
    geo = _StandIn()
    geo.release.clear()
    runs = [_in_thread(placement, geo, 16, jax.random.PRNGKey(3)) for _ in range(8)]
    assert geo.started.wait(10.0)
    geo.release.set()
    for thread, _ in runs:
        thread.join(10.0)
    results = [out["result"] for _, out in runs]
    assert geo.calls == 1
    assert all(r is results[0] for r in results)


def test_a_failed_placement_is_not_cached():
    import time
    geo = _StandIn()
    geo.fail_next = True
    geo.release.clear()
    first, first_out = _in_thread(placement, geo, 16, jax.random.PRNGKey(4))
    assert geo.started.wait(10.0)
    waiter, waiter_out = _in_thread(placement, geo, 16, jax.random.PRNGKey(4))
    time.sleep(0.05)                                                     # usually waiting on the failing computation
    geo.release.set()
    first.join(10.0)
    waiter.join(10.0)
    assert isinstance(first_out.get("error"), RuntimeError)
    # a request that waited on the failing computation gets its exception; one that came after computes afresh
    assert waiter_out.get("error") is first_out["error"] or waiter_out["result"][0].shape == (16, 3)
    r0, _ = placement(geo, 16, jax.random.PRNGKey(4))                 # computed again, not a stuck entry
    assert geo.calls == 2 and r0.shape == (16, 3)
    assert not any(type(v).__name__ == "_PendingPlacement" for v in geo._placement_cache.values())


def test_concurrent_placements_equal_sequential_ones_starts_and_labels():
    """Two keys placed at once on a labelled cell give the starts and labels of placing them one after the other."""
    def cell():
        pools = Compartments(intra=Pool(D=2e-9), myelin=Pool(D=0.0), extra=Pool(D=2e-9))
        return g.PackedMyelinatedCylinders([1e-6], 0.7, [[0.0, 0.0]], 6e-6, N_max=1, compartments=pools)
    requests = [(600, jax.random.PRNGKey(5), 100), (600, jax.random.PRNGKey(6), 4000)]
    alone = cell()
    expected = [placement(alone, n, k, f) for n, k, f in requests]
    together = cell()
    runs = [_in_thread(placement, together, n, k, f) for n, k, f in requests]
    for thread, _ in runs:
        thread.join(60.0)
    for (r0, lab), (_, out) in zip(expected, runs):
        got_r0, got_lab = out["result"]
        assert lab is not None
        assert np.asarray(got_r0).tobytes() == np.asarray(r0).tobytes()
        assert np.asarray(got_lab).tobytes() == np.asarray(lab).tobytes()
