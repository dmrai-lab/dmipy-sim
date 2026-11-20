"""The capabilities a backend declares beyond the common path: the packed myelinated class
(:data:`~dmipy_sim.engine.backends.PACKED_MYELINATED`), a closed-form field read in the walk
(:data:`~dmipy_sim.engine.backends.FIELD_IN_WALK`) and the record of the membrane trials
(:data:`~dmipy_sim.engine.backends.TRIAL_RECORD`).

A stand-in backend whose walkers never move (``tests/backends/test_backend_hook.py``'s ``_Still``) declares each; the walk
that needs the capability is handed to it with the request's fields and is the stand-in's batch as it came; a stand-in
that does not declare it is refused by the text of the JAX-only list (the trial record by its own), asserted in full; each
refusal has a neighbour that walks with the one thing changed, and a declaring stand-in that returns the capability's
arrays missing or of another shape is refused by name."""
import numpy as np
import pytest

import dmipy_sim as d
import dmipy_sim.engine.core as core
from dmipy_sim.compartments import Compartments, Pool
from dmipy_sim.engine import backends
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.fields.hollow_cylinder import HollowCylinderField
from dmipy_sim.geometry import PackedCylinders, PackedMyelinatedCylinders
from dmipy_sim.geometry.myelin import MyelinatedCell, MyelinatedCylinder
from dmipy_sim.trial_record import TrialRecordSpec, edge_steps, u_max_of

from tests.backends.test_backend_hook import _Still

D = 2e-9
R = 2e-6
G = np.array([0.7, 1.0, 0.7, 1.0])                                       # cylinders 1 and 3 plain
OUTER = np.array([R, 0.8 * R, R, 0.9 * R])
CENTRES = np.array([[0.0, 0.0], [2.2, 0.0], [0.0, 2.2], [2.2, 2.2]]) * R
L_CELL = 4.4 * R
COMPS = Compartments(intra=Pool(D=D), myelin=Pool(D=0.0), extra=Pool(D=D))
FIELD = dict(K_c3=3, field_read_interval=4e-5)
SPEC = TrialRecordSpec(modes=3, band_edges=(0.0, 1e-3, 2e-3))
REFUSED = "which only the jax kernels implement; it does not fall back"


def _cell(**kw):
    return PackedMyelinatedCylinders(OUTER * G, G, CENTRES, L_CELL, N_max=8, compartments=COMPS, **kw)


def _packing():
    return PackedCylinders(list(OUTER), CENTRES, L_CELL)


def _walk(g, n=64, **kw):
    kw.setdefault("walker_batch_size", 32)
    return _walk_geometry(n, D, g, 2e-3, 0.5e-3, seed=11, tiers="all", require_gpu=False, **kw)


class _Reads(_Still):
    """A still backend that reads a field basis at given positions (its field pilot's read)."""

    def sample_raster(self, basis, points):
        return basis.channels(points)


class _Cell(_Reads):
    name = "cell"
    capabilities = frozenset({backends.PACKED_MYELINATED})


class _Field(_Reads):
    name = "field"
    capabilities = frozenset({backends.FIELD_IN_WALK})


class _CellField(_Reads):
    name = "cellfield"
    capabilities = frozenset({backends.PACKED_MYELINATED, backends.FIELD_IN_WALK})


class _Record(_Still):
    name = "record"
    capabilities = frozenset({backends.TRIAL_RECORD})


def _random_bands(seed):
    rng = np.random.default_rng(seed)
    made = []

    def bands(request, r0):
        made.append(rng.standard_normal((r0.shape[0], int(request.n_windows), int(request.K), 3)) * 1e-6)
        return made[-1]
    return bands, made


# ── the names ──────────────────────────────────────────────────────────────────────────────────────────────────────

def test_the_three_capabilities_are_names_the_jax_kernels_declare():
    assert (backends.PACKED_MYELINATED, backends.FIELD_IN_WALK, backends.TRIAL_RECORD) == (
        "packed_myelinated", "field_in_walk", "trial_record")
    assert {backends.PACKED_MYELINATED, backends.FIELD_IN_WALK, backends.TRIAL_RECORD} <= backends.JAX_CAPABILITIES
    assert backends.CoefficientBatch._fields[-1] == "trial_record"
    assert backends.CoefficientBatch._field_defaults["trial_record"] is None
    assert all(backends.WalkRequest._field_defaults[f] is None
               for f in ("trial_record", "trial_steps", "trial_radii", "trial_u_max"))


# ── the public descriptions a backend reads ───────────────────────────────────────────────────────────────────────

def test_the_cell_arrays_are_every_input_of_the_cell_kernel(monkeypatch):
    """Every argument the coefficient walk's step factory passes to the wall kernel is a field of the cell's
    description or a stated property of the walk, compared with the argument itself; the list is the kernel's
    signature, complete."""
    import inspect
    import dmipy_sim.geometry.myelin as myelin
    from dmipy_sim.engine.physics import make_packed_myelin_traj_step_fn
    real = myelin.concentric_wall_kernel
    seen = {}

    def kernel(*a, **kw):
        seen.update(inspect.signature(real).bind(*a, **kw).arguments)
        return real(*a, **kw)
    monkeypatch.setattr(myelin, "concentric_wall_kernel", kernel)
    g = _cell(kappa_outer=[0.0, 4e-5, 0.0, 4e-5], kappa_inner=1e-5, rho_inner=1e-6, rho_outer=2e-6)
    g.bounce_budget = 9
    dt = 1e-6
    make_packed_myelin_traj_step_fn(g, dt)
    c = g.cell_arrays()
    assert isinstance(c, MyelinatedCell)
    fields = dict(centers="centers", inner="inner_radii", outer="outer_radii", D_intra="D_intra", D_myelin="D_myelin",
                  D_extra="D_extra", kappa_inner="kappa_inner", kappa_outer="kappa_outer")
    scalars = dict(eps="eps", nudge="nudge", min_gap="min_gap")
    of_the_walk = ("L", "step_max", "rho_weights", "count", "budget", "trials", "search")
    assert set(inspect.signature(real).parameters) == set(fields) | set(scalars) | set(of_the_walk)
    for arg, name in fields.items():
        a, k = getattr(c, name), np.asarray(seen[arg])
        assert a.dtype == k.dtype and a.shape == (8,) + k.shape[1:], name
        np.testing.assert_array_equal(a, k, err_msg=name)
        assert not a.flags.writeable, name
    for arg, name in scalars.items():
        assert np.float32(getattr(c, name)) == np.float32(seen[arg]), name
    assert np.float32(c.cell_size) == np.asarray(seen["L"])
    assert c.step_max(dt) == seen["step_max"]
    assert c.bounce_budget == seen["budget"] == 9
    np.testing.assert_array_equal(np.asarray(seen["rho_weights"]), np.zeros((8, 4), np.float32))
    assert not seen["count"] and not seen["trials"]
    np.testing.assert_array_equal(c.plain_mask, g.plain_mask)
    assert (c.N_actual, c.N_max) == (4, 8)
    with pytest.raises(AttributeError):
        c.cell_size = 1.0
    with pytest.raises(ValueError):
        c.centers[0, 0] = 1.0
    plainless = PackedMyelinatedCylinders(OUTER * 0.7, 0.7, CENTRES, L_CELL, N_max=8, compartments=COMPS)
    seen.clear()
    make_packed_myelin_traj_step_fn(plainless, dt)
    assert plainless.cell_arrays().step_max(dt) == seen["step_max"]           # no plain cylinder: the Gaussian step


def test_the_device_arrays_are_the_arrays_device_read_passes():
    f = HollowCylinderField.of(_cell())
    arrays, _, _ = f.device_read()
    got = f.device_arrays()
    assert [a.shape for a in got] == [(2, 2), (2,), (2,), (2,), (2,), ()]
    for a, ref in zip(got, arrays):
        assert a.dtype == np.float32 and not a.flags.writeable
        np.testing.assert_array_equal(a, np.asarray(ref))


# ── PACKED_MYELINATED ─────────────────────────────────────────────────────────────────────────────────────────────

def test_a_backend_that_declares_the_packed_myelinated_class_walks_the_cell_as_its_batch_came():
    """The cell's own branch of the walk reaches the backend: the request's geometry is the cell, whose public
    description is its kernel's, its budget the request's; the walk's bands are the stand-in's, batch for batch."""
    bands, made = _random_bands(5)
    be = _Cell(bands=bands)
    g = _cell()
    w = _walk(g, backend=be)
    assert w.work["backend"] == "cell" and len(be.requests) == 2
    assert all(r.geometry is g for r in be.requests)
    assert be.requests[0].bounce_budget is None
    np.testing.assert_array_equal(w.position_coeffs[:, :, 2:], np.concatenate(made).astype(np.float32))
    g7 = _cell()
    g7.bounce_budget = 7
    be7 = _Cell()
    _walk(g7, backend=be7)
    assert be7.requests[0].bounce_budget == 7


def test_a_backend_that_does_not_declare_the_packed_myelinated_class_is_refused_by_the_jax_only_list():
    with pytest.raises(ValueError) as e:
        _walk(_cell(), backend=_Still())
    assert str(e.value) == f"backend 'still': this walk uses the myelinated substrates, {REFUSED}"
    w = _walk(d.Sphere(R), backend=_Still())                              # the neighbour: no myelinated cell
    assert w.work["backend"] == "still"


def test_declaring_the_packed_class_does_not_lift_the_isolated_myelinated_cylinder():
    g = MyelinatedCylinder(inner_radius=R, outer_radius=1.4 * R, orientation=[0, 0, 1],
                           compartments=Compartments(intra=Pool(D=D, water_fraction=1.0), myelin=Pool(D=0.1e-9, water_fraction=0.0),
                                                     extra=Pool(D=D, water_fraction=0.0)))
    be = _Cell()
    with pytest.raises(ValueError) as e:
        _walk(g, backend=be)
    assert str(e.value) == f"backend 'cell': this walk uses the myelinated substrates, {REFUSED}"
    assert not be.requests


# ── FIELD_IN_WALK ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_the_cell_with_a_closed_form_field_is_refused_for_what_the_backend_does_not_declare():
    """The strings of the walk that needs both capabilities: the items left, in the list's order."""
    g = _cell()
    kw = dict(FIELD, field_basis=HollowCylinderField.of(g))
    expected = {
        _Reads: "the myelinated substrates, in-walk field sampling of a HollowCylinderField",
        _Cell: "in-walk field sampling of a HollowCylinderField",
        _Field: "the myelinated substrates",
    }
    for cls, items in expected.items():
        be = cls()
        with pytest.raises(ValueError) as e:
            _walk(g, backend=be, **kw)
        assert str(e.value) == f"backend {be.name!r}: this walk uses {items}, {REFUSED}"
        assert not be.requests
    _walk(g, backend=_CellField(), **kw)                                  # the neighbour: both declared


def test_a_backend_that_declares_the_field_is_handed_the_basis_and_its_modes_are_the_walks():
    g = _cell()
    f = HollowCylinderField.of(g)
    rng = np.random.default_rng(9)
    made = []

    def modes(request, r0):
        made.append(rng.standard_normal((r0.shape[0], int(request.n_windows), 12, int(request.K_c3))))
        return made[-1]

    class _Modes(_CellField):
        def walk_batch(self, request, r0, keys, *, out=None):
            b = super().walk_batch(request, r0, keys, out=out)
            return b._replace(field_bands=modes(request, r0)) if request.K_c3 else b

    be = _Modes()
    w = _walk(g, backend=be, field_basis=f, **FIELD)
    reqs = [r for r in be.requests if r.K_c3]
    assert len(reqs) == 2 and all(r.field is f and r.K_c3 == 3 for r in reqs)
    assert all(r.field_read_interval == pytest.approx(w.field_read_interval, rel=1e-12) for r in reqs)
    assert w.field_basis is f
    np.testing.assert_array_equal(w.field_coeffs, np.transpose(np.concatenate(made), (0, 1, 3, 2)).astype(np.float32))


def test_a_backend_that_declares_the_field_and_returns_modes_of_another_shape_is_refused():
    g = _cell()
    kw = dict(FIELD, field_basis=HollowCylinderField.of(g))
    with pytest.raises(ValueError, match="returned field modes of shape None for a walk that reads the 12 channels of its "
                                         "HollowCylinderField at K_c3 = 3"):
        _walk(g, backend=_CellField(field_modes=False), **kw)

    class _Short(_CellField):
        def walk_batch(self, request, r0, keys, *, out=None):
            b = super().walk_batch(request, r0, keys, out=out)
            return b._replace(field_bands=b.field_bands[..., :-1]) if request.K_c3 else b
    with pytest.raises(ValueError, match=r"returned field modes of shape \(32, 1, 12, 2\)"):
        _walk(g, backend=_Short(), **kw)


# ── TRIAL_RECORD ──────────────────────────────────────────────────────────────────────────────────────────────────

def _record(seed, n_obj=4, n_bands=2, modes=3):
    rng = np.random.default_rng(seed)
    made = []

    def record(request, r0):
        n = r0.shape[0]
        made.append((rng.integers(0, 50, (n, n_obj, n_bands)).astype(np.int32),
                     rng.integers(0, 5, (n, n_obj, n_bands)).astype(np.int32),
                     rng.standard_normal((n, n_obj, n_bands, modes)).astype(np.float32),
                     np.zeros((n,), np.int32)))
        return made[-1]
    return record, made


class _Recording(_Record):
    def __init__(self, record, **kw):
        super().__init__(**kw)
        self.record = record

    def walk_batch(self, request, r0, keys, *, out=None):
        return super().walk_batch(request, r0, keys, out=out)._replace(trial_record=self.record(request, r0))


def test_a_backend_that_does_not_declare_the_trial_record_is_refused_by_name_before_the_run(monkeypatch):
    started = []
    real = core.Run

    def run(*a, **kw):
        started.append(a)
        return real(*a, **kw)
    monkeypatch.setattr(core, "Run", run)
    be = _Still()
    with pytest.raises(ValueError) as e:
        _walk(_packing(), backend=be, record_trials=SPEC)
    assert str(e.value) == ("trial_record: the membrane trials are recorded by the JAX kernels; the backend 'still' "
                            "walks without the record")
    assert not started and not be.requests
    w = _walk(_packing(), backend=be)                                     # the neighbour: no record
    assert started and w.trial_record is None


def test_a_backend_that_declares_the_trial_record_is_handed_its_resolution_and_the_record_is_its_batches():
    record, made = _record(4)
    be = _Recording(record)
    g = _packing()
    w = _walk(g, backend=be, record_trials=SPEC)
    assert len(be.requests) == 2
    r = be.requests[0]
    radii, D_obj = g.trial_objects(D)
    assert r.trial_record == SPEC and r.count_walls
    np.testing.assert_array_equal(r.trial_radii, radii)
    np.testing.assert_array_equal(r.trial_u_max, u_max_of(radii, D_obj, r.dt_sim))
    n_steps = r.n_blocks * r.sub_steps
    np.testing.assert_array_equal(r.trial_steps, edge_steps(SPEC, 2e-3, r.dt_sim, n_steps, n_steps))
    tr = w.trial_record
    for i, name in enumerate(("n", "N", "S")):
        np.testing.assert_array_equal(getattr(tr, name), np.concatenate([m[i] for m in made]), err_msg=name)
    assert tr.band_edges == SPEC.band_edges and tr.modes == SPEC.modes
    np.testing.assert_array_equal(tr.object_radii, radii)


def test_a_backend_that_declares_the_trial_record_and_returns_none_or_another_shape_is_refused():
    with pytest.raises(ValueError, match=r"backend 'record' returned a trial record of shapes None for a walk that records "
                                         r"its membrane trials: \(n, N, S, exceed\) of shapes \(\(32, 4, 2\), \(32, 4, 2\), "
                                         r"\(32, 4, 2, 3\), \(32,\)\)"):
        _walk(_packing(), backend=_Record(), record_trials=SPEC)
    record, _ = _record(4, modes=2)
    with pytest.raises(ValueError, match=r"returned a trial record of shapes \(\(32, 4, 2\), \(32, 4, 2\), \(32, 4, 2, 2\), \(32,\)\)"):
        _walk(_packing(), backend=_Recording(record), record_trials=SPEC)

    def exceeding(request, r0):
        n = r0.shape[0]
        z = np.zeros((n, 4, 2), np.int32)
        return z, z, np.zeros((n, 4, 2, 3), np.float32), np.ones((n,), np.int32)
    with pytest.raises(ValueError, match=r"trial_record: 64 trials exceed u_max = "):
        _walk(_packing(), backend=_Recording(exceeding), record_trials=SPEC)


def test_a_trial_record_of_non_integer_or_out_of_range_counts_is_refused_and_integer_counts_join():
    def record(n_dtype, n_value, S_dtype=np.float32):
        def make(request, r0):
            n = r0.shape[0]
            c = np.full((n, 4, 2), n_value, dtype=n_dtype)
            return c, np.zeros((n, 4, 2), np.int64), np.ones((n, 4, 2, 3), S_dtype), np.zeros((n,), np.int64)
        return make
    with pytest.raises(ValueError, match=r"backend 'record' returned the trial record's n as float64 in \[2.7, 2.7\]: "
                                         r"a count is an integer in \[0, 2\^31\)"):
        _walk(_packing(), backend=_Recording(record(np.float64, 2.7)), record_trials=SPEC)
    with pytest.raises(ValueError, match=r"returned the trial record's n as int64 in \[-1, -1\]"):
        _walk(_packing(), backend=_Recording(record(np.int64, -1)), record_trials=SPEC)
    with pytest.raises(ValueError, match=r"returned the trial record's n as int64 in \[4294967296, 4294967296\]"):
        _walk(_packing(), backend=_Recording(record(np.int64, 2 ** 32)), record_trials=SPEC)
    with pytest.raises(ValueError, match=r"returned the trial record's S as int32: the sums are floating point"):
        _walk(_packing(), backend=_Recording(record(np.int64, 3, S_dtype=np.int32)), record_trials=SPEC)
    w = _walk(_packing(), backend=_Recording(record(np.int64, 3, S_dtype=np.float64)), record_trials=SPEC)   # the neighbour
    assert w.trial_record.n.dtype == np.int32 and np.all(w.trial_record.n == 3)
    assert w.trial_record.S.dtype == np.float64                                # the backend's precision, as returned
