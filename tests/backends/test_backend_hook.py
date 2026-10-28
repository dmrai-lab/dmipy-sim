"""The backend hook (dmrai-lab/dmipy-sim#635, #747): another execution of the same physics, resolved by name, refused
when absent, invisible otherwise. The JAX kernels are the default and the only backend this package ships; a stub
returning the coefficient form stands in for another here, and the parity harness runs on whatever is installed."""
import inspect
import re

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine import backends
from dmipy_sim.engine.core import simulate_trajectories
from dmipy_sim.fields.susceptibility_field import FieldGrid

D = 2e-9
R = 2e-6


def coefficient_batch(request, r0, *, precision="fake64", bands=None, blt_bands=None, comp=None, events=None,
                      qv=None, k0_residual=None, field_increments=True, field_modes=True):
    """A :class:`~dmipy_sim.engine.backends.CoefficientBatch` of walkers that never move (their start exact, every band
    0 unless ``bands`` / ``blt_bands`` give them), with the block ends of the request's ``raw_subset`` as saves."""
    r0 = np.asarray(r0, np.float32)
    n, K = r0.shape[0], int(request.K)
    C = np.zeros((n, K + 2, 3))
    C[:, 0] = r0
    if bands is not None:
        C[:, 2:] = bands
    zero = np.zeros((n,), np.int32)
    work = (zero.copy(), zero.copy(), zero.copy()) if request.count_walls else (None, None, zero.copy())
    rec = bool(request.record)
    comp = np.ones((n,), np.int8) if comp is None else np.asarray(comp, np.int8)
    saves = None
    if request.raw_subset:
        n_t = int(request.n_blocks) + 1
        pos = np.repeat(r0[:, None, :], n_t, axis=1)
        saves = backends.WalkBatch(pos, np.zeros((n, n_t), np.float32) if rec else None,
                                   np.repeat(comp[:, None].astype(np.float32), n_t, axis=1) if rec else None, zero.copy(), work)
    if rec and events is None and request.kappa_over_D > 0.0:
        events = (np.zeros((n + 1,), np.int64), np.zeros((0,), np.float64), np.zeros((0,), np.int8))
    if events is False:                                               # a backend that keeps no event list
        events = None
    field_bands = field_qv = field_qv_increments = None
    if request.field is not None:                                     # the field: every mode 0, the start's bare channels at every block end
        grid = request.field if isinstance(request.field, FieldGrid) else request.field.basis
        n_ch = len(grid.channel_names)
        field_qv = np.zeros((n, n_ch))
        field_qv_increments = np.zeros((n, n_ch)) if field_increments else None
        if request.K_c3 and field_modes:
            field_bands = np.zeros((n, n_ch, int(request.K_c3)))
        if saves is not None:
            fs = np.repeat(np.asarray(grid.channels(r0), np.float32)[:, None, :], int(request.n_blocks) + 1, axis=1)
            saves = saves._replace(field_samples=fs)
    return backends.CoefficientBatch(
        C, np.zeros(n) if rec else None, np.zeros(n) if rec else None,
        (np.zeros((n, K)) if blt_bands is None else np.asarray(blt_bands, np.float64)) if rec else None,
        comp if rec else None, zero.copy(), work, K, int(request.m), int(request.sub_steps), precision, saves=saves,
        qv_positions=(np.zeros((n, 3)) if qv is None else qv), qv_boundary=(np.zeros(n) if rec else None),
        events=(events if rec else None), k0_residual=k0_residual, field_bands=field_bands, field_qv=field_qv,
        field_qv_increments=field_qv_increments,
        dt_f=(float(request.field_read_interval) if request.field is not None else None))


class _Still(backends.Backend):
    """A backend whose walkers never move, in the coefficient form: enough to see that the producer took the dispatch,
    the bands, the events, the raw subset and the record."""
    name = "still"

    def __init__(self, refuse=None, **kw):
        self._refuse = refuse
        self.kw = kw
        self.requests = []

    def refuses(self, request):
        return self._refuse

    def walk_batch(self, request, r0, keys, *, out=None):
        self.requests.append(request)
        kw = {k: (v(request, r0) if callable(v) else v) for k, v in self.kw.items()}
        return coefficient_batch(request, r0, **kw)


class _Saves(backends.Backend):
    """A backend that still returns a walk of saves."""
    name = "saves"

    def refuses(self, request):
        return None

    def walk_batch(self, request, r0, keys, *, out=None):
        n = r0.shape[0]; n_t = request.n_blocks + 1
        zero = np.zeros((n,), np.int32)
        return backends.WalkBatch(np.repeat(np.asarray(r0, np.float32)[:, None, :], n_t, axis=1), None, None, zero, (None, None, zero))


def _walk(g, **kw):
    kw.setdefault("walker_batch_size", 64)
    return simulate_trajectories(64, D, g, 2e-3, 0.5e-3, seed=11, tiers="all", require_gpu=False, **kw)


def test_jax_is_the_default_and_the_same_walk_by_name():
    a, b = _walk(d.Sphere(R)), _walk(d.Sphere(R), backend="jax")
    np.testing.assert_array_equal(a.position_coeffs, b.position_coeffs)
    assert a.work["backend"] == b.work["backend"] == "jax"


def test_an_unknown_backend_is_refused_naming_what_is_installed():
    with pytest.raises(ValueError, match=r"no backend 'nope' is installed; installed: \['jax'"):
        _walk(d.Sphere(R), backend="nope")
    with pytest.raises(TypeError, match="walk_batch"):
        _walk(d.Sphere(R), backend=object())


def test_the_request_names_the_walk_and_nothing_of_a_save_grid():
    """The request's fields are the agreed names (#747, dmipy-sim-cuda#73 section 3): the block, the blocks, the steps
    per block, the band, the order, the field's modes and read interval, the event capacity, the raw subset; no
    ``n_t``, no ``dt_save``, no ``field_sample_every``."""
    assert backends.WalkRequest._fields == ("geometry", "block_h", "n_blocks", "sub_steps", "dt_sim", "diffusivity", "record",
                                            "kappa_over_D", "count_walls", "bounce_budget", "seed", "K", "m", "K_c3",
                                            "field_read_interval", "events", "raw_subset", "stepping", "field")
    still = _Still()
    w = _walk(d.Sphere(R), backend=still, K=8, block_order=2, raw_subset=10, walker_batch_size=32)
    reqs = still.requests
    assert [r.raw_subset for r in reqs] == [10, 0, 0]                   # the raw piece, then the rest, batch by batch
    r = reqs[0]
    assert r.K == 8 and r.m == 2 and r.n_blocks == 4 and r.block_h == pytest.approx(0.5e-3) and r.T == pytest.approx(2e-3)
    assert r.sub_steps == w.sub_steps and r.dt_sim == pytest.approx(w.dt_sim) and r.record and r.kappa_over_D == 0.0
    assert r.events == 0 and r.K_c3 == 0 and r.field_read_interval == 0.0 and r.stepping is None and r.field is None
    perm = _Still()
    _walk(d.Sphere(R, permeability=1e-5), backend=perm)
    r = perm.requests[0]
    assert r.kappa_over_D == pytest.approx(1e-5 / D)
    assert r.events == backends.event_capacity(r.n_blocks * r.sub_steps) > 0
    # the capacity is the crossing rule's bound: Binomial(n_steps, CROSSING_P_MAX)'s mean plus six sigma, at least 16
    from dmipy_sim.engine.physics import CROSSING_P_MAX
    mu = CROSSING_P_MAX * 10000
    assert backends.event_capacity(10000) == int(np.ceil(mu + 6 * np.sqrt(mu * (1 - CROSSING_P_MAX)))) == 63
    assert backends.event_capacity(1) == 16


def test_the_walk_is_the_backends_coefficient_form_as_it_came():
    """The producer takes the batch's bands, contact, quadratic variation and record as they are, to float32, converting
    nothing; the block record states the backend's own precision and no residual it did not report."""
    rng = np.random.default_rng(0)
    made = []                                                          # (bands, contact bands) per batch, in order

    def make(request, r0):
        n, K = r0.shape[0], request.K
        made.append((rng.standard_normal((n, K, 3)) * 1e-6, rng.standard_normal((n, K)) * 1e-7))
        return made[-1][0]

    still = _Still(bands=make, blt_bands=lambda req, r0: made[-1][1],
                   qv=lambda req, r0: np.full((r0.shape[0], 3), 2 * D * req.T))
    w = _walk(d.Sphere(R), backend=still, K=6, walker_batch_size=32)
    assert w.work["backend"] == "still" and w.K == 6 and w.m == 3 and w.n_windows == 1 and w.n_blocks == 4
    assert len(made) == 2
    assert w.position_coeffs.shape == (64, 1, 8, 3) and w.position_coeffs.dtype == np.float32
    np.testing.assert_array_equal(w.position_coeffs[:, 0, 2:], np.concatenate([b for b, _ in made]).astype(np.float32))
    np.testing.assert_array_equal(w.boundary_bands[:, 0], np.concatenate([c for _, c in made]).astype(np.float32))
    np.testing.assert_array_equal(w.position_coeffs[:, 0, 1], 0.0)
    np.testing.assert_allclose(w.qv_positions[:, 0], 2 * D * 2e-3, rtol=1e-6)
    np.testing.assert_array_equal(w.boundary_endpoint, 0.0)
    assert w.compartment.shape == (64,) and w.compartment.dtype == np.int8 and w.comp_events is None
    blk = w.block
    assert blk["precision"] == "fake64" and blk["k0_residual_m"] is None and blk["k0_residual_relative"] is None
    assert blk["ring"] is None and "still" in blk["accumulation"] and blk["n_blocks"] == 4 and blk["m"] == 3
    assert w.work["n_hits"] is None and w.work["exhausted_steps"] == 0 and w.illegal_crossings == 0
    # a reported residual is recorded
    w2 = _walk(d.Sphere(R), backend=_Still(k0_residual=lambda req, r0: np.full((r0.shape[0], 3), 1e-9)), K=6)
    assert w2.block["k0_residual_m"] == pytest.approx(1e-9)


def test_the_raw_subset_is_the_backends_block_ends_and_the_events_are_joined():
    """``raw_subset=n``: the first n walkers' block ends come back as the backend's saves and are the walk's raw
    positions; a walk whose walkers can change pool keeps C1 as the batches' event lists joined in walker order."""
    def events(request, r0):
        n = r0.shape[0]
        off = np.zeros(n + 1, np.int64); off[1:] = 1                 # walker 0 of every batch changes pool once
        return off, np.array([request.block_h * 0.5]), np.array([0], np.int8)

    still = _Still(events=events)
    w = _walk(d.Sphere(R, permeability=1e-5), backend=still, raw_subset=10, walker_batch_size=32)
    assert w.raw_positions.shape == (10, 5, 3) and w.raw_boundary.shape == (10, 5) and w.raw_compartment.shape == (10, 5)
    np.testing.assert_array_equal(w.raw_positions[:, 0], w.position_coeffs[:10, 0, 0])
    assert (w.raw_positions == w.raw_positions[:, :1]).all()
    sv = w
    assert sv.raw_positions.shape == (10, 5, 3)
    off, t, lab = w.comp_events
    assert off.shape == (65,) and off[-1] == 3 and t.shape == (3,) and lab.shape == (3,)
    assert [int(x) for x in np.flatnonzero(np.diff(off))] == [0, 10, 32]
    assert w.compartment.shape == (64,)
    # take() carries the events of the walkers it keeps
    sub = w.take(np.array([32, 0, 5]))
    o2, t2, l2 = sub.comp_events
    assert o2.tolist() == [0, 1, 2, 2] and t2.tolist() == [t[1], t[0]]
    # a backend that returns no events for such a walk is refused by name
    with pytest.raises(ValueError, match="no C1 event list"):
        _walk(d.Sphere(R, permeability=1e-5), backend=_Still(events=False))


def test_a_backend_returning_a_walk_of_saves_is_refused_and_nothing_is_converted():
    with pytest.raises(ValueError, match="returned a WalkBatch for the coefficient walk.*converts nothing.*refused"):
        _walk(d.Sphere(R), backend=_Saves())


def test_a_batch_of_another_resolution_or_shape_is_refused():
    class _Wrong(_Still):
        def walk_batch(self, request, r0, keys, *, out=None):
            b = coefficient_batch(request, r0)
            return b._replace(K=request.K + 1)
    with pytest.raises(ValueError, match=r"accumulated \(K, m, sub_steps\)"):
        _walk(d.Sphere(R), backend=_Wrong())

    class _Short(_Still):
        def walk_batch(self, request, r0, keys, *, out=None):
            b = coefficient_batch(request, r0)
            return b._replace(position_coeffs=b.position_coeffs[:, :-1])
    with pytest.raises(ValueError, match="position coefficients of shape"):
        _walk(d.Sphere(R), backend=_Short())


def test_the_jax_only_paths_are_refused_for_another_backend():
    from dmipy_sim.geometry.myelin import MyelinatedCylinder
    g = MyelinatedCylinder(inner_radius=R, outer_radius=1.4 * R, orientation=[0, 0, 1], D_intra=D, D_extra=D,
                           D_myelin=0.1e-9, water_fractions=(1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="the myelinated substrates.*which only the jax kernels implement"):
        _walk(g, backend=_Still())
    with pytest.raises(ValueError, match="the compartment guard"):
        _walk(d.Sphere(R), backend=_Still(), enforce_compartment=True)
    with pytest.raises(ValueError, match="several storage windows"):
        simulate_trajectories(16, D, d.Sphere(R), 0.2, 2e-3, seed=1, require_gpu=False, K=4, sub_steps=2, backend=_Still())
    still = _Still(refuse="no spheres here")
    with pytest.raises(ValueError, match="backend 'still' refuses this walk of Sphere: no spheres here"):
        _walk(d.Sphere(R), backend=still)
    assert not still.requests


def _layer_grid():
    """A 1-D layer (iso_local 1 for x > 0) on a 40-node raster across the sphere: the slab's kind of field."""
    nx = 40; vs = np.array([2 * R / nx, 1.0, 1.0]); x = -R + (np.arange(nx) + 0.5) * vs[0]
    return FieldGrid(dict(iso_local=(x > 0).astype(np.float32).reshape(nx, 1, 1), iso_P=np.zeros((6, nx, 1, 1), np.float32),
                          shape=(nx, 1, 1), voxel_size=vs), np.array([-R, -0.5, -0.5]))


def test_a_raster_field_is_the_backends_request_and_its_modes_are_checked():
    """A FieldGrid read along the fixed-step walk is handed to the backend as the request's own field, every batch at
    the walk's read interval and band (the field's pilot walked by the JAX kernels: the read rule's checks and the band's
    block means are theirs); a backend that returns no modes for it, or modes at another read, is refused by name."""
    fg = _layer_grid()
    still = _Still()
    kw = dict(field_basis=fg, K_c3=3, field_read_interval=4e-5)
    w = _walk(d.Sphere(R), backend=still, **kw)
    assert w.field_coeffs.shape[-2:] == (3, 6)
    reqs = [r for r in still.requests if r.K_c3]
    assert reqs and all(r.field is fg and r.K_c3 == 3 for r in reqs)
    assert all(r.field_read_interval == pytest.approx(w.field_read_interval, rel=1e-12) for r in reqs)
    with pytest.raises(ValueError, match="returned field modes of shape None"):
        _walk(d.Sphere(R), backend=_Still(field_modes=False), **kw)


def test_walk_spec_passes_the_backend_through_with_and_without_adaptive_steps():
    from dmipy_sim.spec import walk_spec, SpecError
    g = d.Sphere(R)
    still = _Still()
    w = walk_spec(g.spec, 64, 2e-3, 0.5e-3, seed=11, diffusivity=D, walker_batch_size=64, backend=still)
    assert w.work["backend"] == "still" and still.requests
    # the adaptive producer takes the backend too (#651): the sphere is refused for not being a bundle, by the spec,
    # not for the backend
    with pytest.raises(SpecError, match="adaptive stepping is for a bundle"):
        walk_spec(g.spec, 64, 2e-3, 0.5e-3, seed=11, diffusivity=D, walker_batch_size=64, backend=_Still(),
                  adaptive_steps=True)


def test_the_adaptive_producer_hands_a_backend_the_plan_and_signs_the_record():
    """The adaptive producer walks a backend in the coefficient form too (#742): the request carries the plan, the
    block, the band and the raw subset; the walk takes the batch's bands and signs the record with the backend's
    name and no stepping counts (a backend's batch carries none); a backend handing back saves is refused by name."""
    from dmipy_sim.engine.adaptive import simulate_trajectories_adaptive, AdaptivePlan, adaptive_plan
    g = backends.parity_cases()["strands_intra"]()
    still = _Still()
    w = simulate_trajectories_adaptive(64, D, g, 1e-3, 2.5e-4, seed=3, require_gpu=False, walker_batch_size=64, backend=still, K=6)
    assert w.position_coeffs.shape == (64, 1, 8, 3) and (w.position_coeffs[:, 0, 1:] == 0).all()
    assert w.work["backend"] == "still" and w.stepping["backend"] == "still"
    assert w.stepping["kernel_steps"] is None and w.work["walker_steps"] is None      # the batch counted nothing
    assert w.raw_positions is None and w.comp_events is None and w.compartment.shape == (64,)
    req = still.requests[0]
    assert isinstance(req, backends.WalkRequest) and isinstance(req.stepping, AdaptivePlan)
    assert req.K == 6 and req.m == 3 and req.n_blocks == 4 and req.block_h == pytest.approx(2.5e-4) and req.raw_subset == 0
    assert req.K_c3 == 0 and req.field_read_interval == 0.0 and req.events == 0 and req.field is None
    assert req.sub_steps == req.stepping.n_min == w.sub_steps and req.kappa_over_D == 0.0 and req.record
    assert req.stepping == adaptive_plan(g, D, 2.5e-4)._replace(candidate_k=req.stepping.candidate_k)
    assert req.stepping.n_rounds * req.stepping.steps_per_round == req.stepping.n_min
    assert len(req.stepping.steps_c) == req.stepping.n_classes and req.stepping.steps_c[0] == req.stepping.steps_per_round
    with pytest.raises(ValueError, match="refuses this walk of PackedCurvedCylinders: no strands"):
        simulate_trajectories_adaptive(64, D, g, 1e-3, 2.5e-4, seed=3, require_gpu=False, walker_batch_size=64,
                                       backend=_Still(refuse="no strands"))
    with pytest.raises(ValueError, match="a walk of saves is refused"):
        simulate_trajectories_adaptive(64, D, g, 1e-3, 2.5e-4, seed=3, require_gpu=False, walker_batch_size=64, backend=_Saves())
    with pytest.raises(ValueError, match="several storage windows"):
        simulate_trajectories_adaptive(64, D, g, 2e-3, 2.5e-4, seed=3, require_gpu=False, walker_batch_size=64, backend=_Still(),
                                       window_T=1e-3)


def test_the_adaptive_producer_takes_the_start_pool_from_the_batch():
    """The walk's start pool is the batch's ``compartment`` (no second classification of the starts through JAX)."""
    from dmipy_sim.engine.adaptive import simulate_trajectories_adaptive
    g = backends.parity_cases()["strands_intra"]()                     # every walker is seeded inside a tube:
    n = 32                                                              # classify_positions_exact would give pool 1 for all
    comp0 = np.zeros(n, np.int8); comp0[::2] = 1                        # a pattern no real classification of these starts gives
    w = simulate_trajectories_adaptive(n, D, g, 1e-3, 2.5e-4, seed=3, require_gpu=False, walker_batch_size=n, backend=_Still(comp=comp0))
    np.testing.assert_array_equal(np.asarray(w.compartment), comp0)
    assert w.illegal_crossings == 0


def test_the_adaptive_producer_hands_a_backend_the_field_sampling_and_the_pilots_read():
    """With a strand field the request carries the sampling (the basis, the list's reuse and width) and the read
    (``K_c3``, ``field_read_interval``): the pilot walks on the JAX kernels whatever the backend (the read series'
    quadratic variation, the trapezoid's constants and the band are the producer's own measurement), the backend's
    batches at the read the pilot derives; the walk takes the batch's field modes, its quadratic variation and its
    read series' quadratic variation. The raw subset of a field walk is walked by the JAX kernels too (its reads at
    the read instants). A backend that returns no read series' quadratic variation is refused by name."""
    from dmipy_sim.engine.adaptive import simulate_trajectories_adaptive
    from dmipy_sim.fields.strand_field import StrandFieldBasis
    g = backends.parity_cases()["strands_intra"]()
    basis = StrandFieldBasis(g.centerlines, 0.7 * g.radii, g.radii, cutoff_m=8e-6, domain=(g.box[0], g.box[1]))
    kw = dict(seed=3, require_gpu=False, walker_batch_size=64, field_basis=basis, field_reuse_intervals=3, K_c3=2, K=6)
    still = _Still()
    w = simulate_trajectories_adaptive(64, D, g, 1e-3, 2.5e-4, backend=still, **kw)
    walk = still.requests[-1]
    f = walk.field
    assert isinstance(f, backends.FieldSampling) and f.basis is basis and f.sample_every == 1 and f.reuse_intervals == 3
    assert f.radius_m >= basis.gather_radius_m and f.list_k == 256            # no JAX probe with a backend: the given width
    fr = w.block["field_read"]
    assert fr["rule"] == "pilot" and fr["constants_measured"] and w.stepping["field_pilot_backend"] == "jax"
    assert all(r.field_read_interval == pytest.approx(w.field_read_interval) for r in still.requests)   # the backend walks the walk only
    assert walk.K_c3 == 2 and w.field_coeffs.shape == (64, 1, 2, 12) and w.K_c3 == 2 and w.field_basis is basis
    assert w.qv_field.shape == (64, 1, 12) and w.qv_field_increments.shape == (64, 1, 12)
    with pytest.raises(ValueError, match="the backend does not return the field increments"):
        simulate_trajectories_adaptive(64, D, g, 1e-3, 2.5e-4, backend=_Still(field_increments=False), **kw)
    raw = _Still()
    wr = simulate_trajectories_adaptive(64, D, g, 1e-3, 2.5e-4, backend=raw, raw_subset=16, **kw)
    assert wr.raw_field.shape == (16, wr.n_blocks * wr.reads_per_block + 1, 12) and wr.stepping["raw_subset_backend"] == "jax"
    assert all(int(r.raw_subset) == 0 for r in raw.requests)                     # the backend walks no raw subset


def test_the_producer_reads_nothing_from_a_backend_but_the_interface():
    """`core` may read `name`, call `refuses` and `walk_batch`, and nothing else of a backend."""
    src = inspect.getsource(simulate_trajectories)
    used = set(re.findall(r"_backend\.([A-Za-z_]+)", src))
    assert used <= {"name", "refuses", "walk_batch"}, used


def test_the_parity_harness_compares_the_coefficient_form():
    """``parity_report`` on the JAX kernels against themselves (one walk, the same starts): every check passes with
    z = 0, on the bands, the quadratic variation, the work, confinement and the replayed signal."""
    report = backends.parity_report(None, cases={"sphere": backends.parity_cases()["sphere"]}, n_walkers=200)
    assert len(report) == 1 and report[0]["ok"] is True
    ch = report[0]["checks"]
    assert set(ch) == {"n_hits", "n_crossings", "exhausted_steps", "illegal", "qv_positions", "qv_boundary", "bands", "confinement", "pgse"}
    assert ch["n_hits"]["jax"] == ch["n_hits"]["backend"] > 0 and ch["bands"]["z_max"] == 0.0
    assert ch["pgse"]["jax"] == ch["pgse"]["backend"] and max(ch["qv_positions"]["z"]) == 0.0
    # a backend whose bands are not the walk's is caught at the band level
    bad = _Still(qv=lambda req, r0: np.full((r0.shape[0], 3), 2 * D * req.T))
    report = backends.parity_report(bad, cases={"sphere": backends.parity_cases()["sphere"]}, n_walkers=200)
    assert report[0]["ok"] is False and report[0]["checks"]["bands"]["ok"] is False and report[0]["checks"]["pgse"]["ok"] is False
    assert report[0]["checks"]["confinement"]["ok"] is True


def test_the_parity_harness_admits_what_is_installed():
    names = sorted(backends.installed())
    if not names:
        pytest.skip("no backend beside jax is installed: nothing to compare")
    for name in names:
        report = backends.parity_report(name)
        bad = [r for r in report if r["ok"] is False]
        assert not bad, f"{name}: {bad}"


def test_the_request_carries_the_step_law():
    """A backend's request states the step law beside the length scale: the pair's two atoms and the long one's
    probability, the values ``dmipy_sim.math.step_law`` defines, so no kernel holds a length of its own."""
    from dmipy_sim.math import step_law as sl
    fields = backends.request_struct_fields(n_blocks=2, sub_steps=2, step_l=1e-6, kappa_over_D=0.0, record=False, count=False, budget=4,
                                            block_h=1e-4, dt_sim=5e-5, K=8, m=3, events=5, raw_subset=0)
    assert fields["n_blocks"] == 2 and fields["ring"] == 64 and fields["events"] == 0      # no events on a walk that records none
    assert fields["atom_short"] == pytest.approx(sl.PAIR_SHORT, rel=1e-6)
    assert fields["atom_long"] == pytest.approx(sl.PAIR_LONG, rel=1e-6) and fields["atom_long"] == pytest.approx(sl.LONGEST_STEP, rel=1e-6)
    assert fields["p_long"] == pytest.approx(sl.PAIR_P_LONG, rel=1e-6)
    assert list(fields) == list(backends.REQUEST_TEMPLATE_KEYS) == [
        "n_blocks", "sub_steps", "step_l", "atom_short", "atom_long", "p_long", "kappa_over_D", "record", "count", "budget",
        "seed", "block_h", "dt_sim", "K", "m", "ring", "events", "raw_subset"]                 # DscRequest, dmipy-sim-cuda#78
