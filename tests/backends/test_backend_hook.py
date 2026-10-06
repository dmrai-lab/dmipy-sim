"""The backend hook (dmrai-lab/dmipy-sim#635): another execution of the same physics, resolved by name, refused
when absent, invisible otherwise. The JAX kernels are the default and the only backend this package ships; a stub
stands in for another here, and the parity harness runs on whatever is installed."""
import inspect
import re

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine import backends
from dmipy_sim.engine.core import simulate_trajectories

D = 2e-9
R = 2e-6


class _Still(backends.Backend):
    """A backend whose walkers never move: enough to see that the producer took the dispatch and the record."""
    name = "still"

    def __init__(self, refuse=None):
        self._refuse = refuse
        self.requests = []

    def refuses(self, request):
        return self._refuse

    def walk_batch(self, request, r0, keys):
        self.requests.append(request)
        n = r0.shape[0]
        pos = np.repeat(np.asarray(r0, np.float32)[:, None, :], request.n_t, axis=1)
        zero = np.zeros((n,), np.int32)
        blt = np.zeros((n, request.n_t), np.float32) if request.record else None
        comp = np.ones((n, request.n_t), np.float32) if request.record else None
        work = ((zero.copy(), zero.copy(), zero.copy()) if request.count_walls else (None, None, zero.copy()))
        return backends.WalkBatch(pos, blt, comp, zero.copy(), work)


def _walk(g, **kw):
    return simulate_trajectories(64, D, g, 2e-3, 0.5e-3, seed=11, tiers="all", walker_batch_size=64, **kw)


def test_jax_is_the_default_and_the_same_walk_by_name():
    a, b = _walk(d.Sphere(R)), _walk(d.Sphere(R), backend="jax")
    np.testing.assert_array_equal(a.positions, b.positions)
    assert a.work["backend"] == b.work["backend"] == "jax"


def test_an_unknown_backend_is_refused_naming_what_is_installed():
    with pytest.raises(ValueError, match=r"no backend 'nope' is installed; installed: \['jax'"):
        _walk(d.Sphere(R), backend="nope")
    with pytest.raises(TypeError, match="walk_batch"):
        _walk(d.Sphere(R), backend=object())


def test_a_backend_object_takes_the_dispatch_and_signs_the_record():
    g = d.Sphere(R); g.count_walls = True
    still = _Still()
    w = _walk(g, backend=still, bounce_budget=5)
    assert (w.positions == w.positions[:, :1]).all()                 # nobody moved: the stub walked
    assert w.work["backend"] == "still" and w.work["n_hits"] == 0 and w.work["exhausted_steps"] == 0
    assert w.work["bounce_budget"] == 5
    req = still.requests[0]
    assert isinstance(req, backends.WalkRequest)
    assert req.geometry is g and req.record and req.count_walls and req.bounce_budget == 5
    assert req.sub_steps == w.sub_steps and req.n_t == w.positions.shape[1] and req.kappa_over_D == 0.0
    assert w.has_surface and w.compartment is not None


def test_a_backend_may_refuse_and_the_producer_never_falls_back():
    with pytest.raises(ValueError, match="backend 'still' refuses this walk of Sphere: no spheres here"):
        _walk(d.Sphere(R), backend=_Still(refuse="no spheres here"))


def test_the_jax_only_paths_are_refused_for_another_backend():
    from dmipy_sim.geometry.myelin import MyelinatedCylinder
    g = MyelinatedCylinder(inner_radius=R, outer_radius=1.4 * R, orientation=[0, 0, 1], D_intra=D, D_extra=D,
                           D_myelin=0.1e-9, water_fractions=(1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="the myelinated substrates, which only the jax kernels implement"):
        _walk(g, backend=_Still())
    with pytest.raises(ValueError, match="the compartment guard"):
        _walk(d.Sphere(R), backend=_Still(), enforce_compartment=True)


def test_walk_spec_passes_the_backend_through_and_refuses_it_with_adaptive_steps():
    from dmipy_sim.spec import walk_spec
    g = d.Sphere(R)
    w = walk_spec(g.spec, 64, 2e-3, 0.5e-3, seed=11, diffusivity=D, walker_batch_size=64, backend=_Still())
    assert w.work["backend"] == "still"
    with pytest.raises(ValueError, match="adaptive producer"):
        walk_spec(g.spec, 64, 2e-3, 0.5e-3, seed=11, diffusivity=D, walker_batch_size=64, backend=_Still(),
                  adaptive_steps=True)


def test_the_producer_reads_nothing_from_a_backend_but_the_interface():
    """`core` may read `name`, call `refuses` and `walk_batch`, and nothing else of a backend."""
    src = inspect.getsource(simulate_trajectories)
    used = set(re.findall(r"_backend\.([A-Za-z_]+)", src))
    assert used <= {"name", "refuses", "walk_batch"}, used


def test_the_parity_harness_admits_what_is_installed():
    names = sorted(backends.installed())
    if not names:
        pytest.skip("no backend beside jax is installed: nothing to compare")
    for name in names:
        report = backends.parity_report(name, n_walkers=2000)
        bad = [r for r in report if r["ok"] is False]
        assert not bad, f"{name}: {bad}"
