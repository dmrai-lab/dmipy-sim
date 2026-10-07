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

    def walk_batch(self, request, r0, keys, *, out=None):
        self.requests.append(request); self.outs = getattr(self, "outs", []) + [out]
        n = r0.shape[0]
        pos = np.repeat(np.asarray(r0, np.float32)[:, None, :], request.n_t, axis=1)
        if out is not None:                                           # a backend may write into the producer's arrays
            out["positions"][...] = pos; pos = out["positions"]
        zero = np.zeros((n,), np.int32)
        blt = np.zeros((n, request.n_t), np.float32) if request.record else None
        comp = np.ones((n, request.n_t), np.float32) if request.record else None
        work = ((zero.copy(), zero.copy(), zero.copy()) if request.count_walls else (None, None, zero.copy()))
        field = None
        if request.field is not None:                                 # the field's channels at the start, at every sample
            f = request.field
            n_tf = len(range(0, request.n_t, f.sample_every))
            field = np.repeat(np.asarray(f.basis.channels(np.asarray(r0, np.float32)), np.float32)[:, None, :], n_tf, axis=1)
        return backends.WalkBatch(pos, blt, comp, zero.copy(), work, field_samples=field)


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


def test_walk_spec_passes_the_backend_through_with_and_without_adaptive_steps():
    from dmipy_sim.spec import walk_spec, SpecError
    g = d.Sphere(R)
    w = walk_spec(g.spec, 64, 2e-3, 0.5e-3, seed=11, diffusivity=D, walker_batch_size=64, backend=_Still())
    assert w.work["backend"] == "still"
    # the adaptive producer takes the backend too (#651): the sphere is refused for not being a bundle, by the spec,
    # not for the backend
    with pytest.raises(SpecError, match="adaptive stepping is for a bundle"):
        walk_spec(g.spec, 64, 2e-3, 0.5e-3, seed=11, diffusivity=D, walker_batch_size=64, backend=_Still(),
                  adaptive_steps=True)


def test_the_adaptive_producer_hands_a_backend_the_plan_and_signs_the_record():
    from dmipy_sim.engine.adaptive import simulate_trajectories_adaptive, AdaptivePlan, adaptive_plan
    g = backends.parity_cases()["strands_intra"]()
    still = _Still()
    w = simulate_trajectories_adaptive(64, D, g, 1e-3, 2.5e-4, seed=3, require_gpu=False, walker_batch_size=64, backend=still)
    assert (w.positions == w.positions[:, :1]).all() and w.work["backend"] == "still" and w.stepping["backend"] == "still"
    assert w.stepping["kernel_steps"] is None and w.work["walker_steps"] is None      # the stub counted nothing
    req = still.requests[0]
    assert isinstance(req, backends.WalkRequest) and isinstance(req.stepping, AdaptivePlan)
    assert req.sub_steps == req.stepping.n_min == w.sub_steps and req.kappa_over_D == 0.0 and req.record
    assert req.stepping == adaptive_plan(g, D, 2.5e-4)
    assert req.stepping.n_rounds * req.stepping.steps_per_round == req.stepping.n_min
    assert len(req.stepping.steps_c) == req.stepping.n_classes and req.stepping.steps_c[0] == req.stepping.steps_per_round
    with pytest.raises(ValueError, match="refuses this walk of PackedCurvedCylinders: no strands"):
        simulate_trajectories_adaptive(64, D, g, 1e-3, 2.5e-4, seed=3, require_gpu=False, walker_batch_size=64,
                                       backend=_Still(refuse="no strands"))


def test_the_adaptive_producer_hands_a_backend_the_field_sampling_and_subtracts_the_mean():
    from dmipy_sim.engine.adaptive import simulate_trajectories_adaptive
    from dmipy_sim.fields.strand_field import StrandFieldBasis
    g = backends.parity_cases()["strands_intra"]()
    basis = StrandFieldBasis(g.centerlines, 0.7 * g.radii, g.radii, cutoff_m=8e-6, domain=(g.box[0], g.box[1]))
    still = _Still()
    w = simulate_trajectories_adaptive(64, D, g, 1e-3, 2.5e-4, seed=3, require_gpu=False, walker_batch_size=64, backend=still,
                                       field_basis=basis, field_sample_every=2, field_reuse_intervals=3)
    req = still.requests[0]
    f = req.field
    assert isinstance(f, backends.FieldSampling) and f.basis is basis and f.sample_every == 2 and f.reuse_intervals == 3
    assert f.radius_m >= basis.gather_radius_m and f.list_k == 256            # no JAX probe with a backend: the given width
    assert w.field_samples.shape == (64, 3, 13) and w.field_sample_every == 2 and w.field_basis is basis
    assert still.outs[0] is not None and set(still.outs[0]) == {"positions", "boundary_local_time", "field_samples"}
    assert still.outs[0]["positions"].shape == (64, w.positions.shape[1], 3) and still.outs[0]["field_samples"].shape == (64, 3, 13)
    # the stub returned the start's bare channels at every sample: the record holds them with the mean subtracted
    expect = np.asarray(basis.channels(np.asarray(w.positions[:, 0])), np.float32) - np.asarray(basis.mean, np.float32)
    np.testing.assert_allclose(w.field_samples[:, 0], expect, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(w.field_samples[:, 2], expect, rtol=1e-5, atol=1e-7)


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
