"""A backend that walks a permeability per walker (:data:`~dmipy_sim.engine.backends.PERMEABILITY_FACTOR`): the walk's
``permeability_factor=`` reaches it as each batch's own rows (``WalkRequest.permeability_factor``, ``r0``'s row order),
in the walk's batches, the events and trial-record pilots and the field pilot's batches; a backend that does not
declare it is refused by the capability's name before any batch is walked.

The stand-ins (``tests/backends/test_backend_hook.py``'s ``_Still`` and its kin) keep every walker at its start, so a
batch's rows are found by its starts among the walk's (``position_coeffs[:, 0, 0]``, the exact start of each walker in
walker order): every call's factors must equal ``FAC`` at those walkers, exactly."""
import numpy as np
import pytest

from dmipy_sim.engine import backends
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.fields.hollow_cylinder import HollowCylinderField
from dmipy_sim.geometry import PackedCylinders
from dmipy_sim.trial_record import TrialRecordSpec

from tests.backends.test_backend_hook import _Still
from tests.backends.test_backend_capabilities import (CENTRES, L_CELL, OUTER, FIELD, _CellField, _Recording, _cell,
                                                      _record)

D = 2e-9
N = 32
FAC = np.arange(N, dtype=np.float32) / np.float32(N)
KAPPA = 1e-4                                         # m/s, every cylinder of the packing


def _packing():
    return PackedCylinders(list(OUTER), CENTRES, L_CELL, permeability=KAPPA)


def _walk(g, **kw):
    kw.setdefault("walker_batch_size", 8)
    return _walk_geometry(N, D, g, 2e-3, 0.5e-3, seed=11, tiers="all", require_gpu=False, **kw)


class _Seen:
    """Keeps what every ``walk_batch`` call was handed: its request and a copy of its starts and factors."""

    def walk_batch(self, request, r0, keys, *, out=None):
        fac = request.permeability_factor
        self.calls.append((request, np.array(r0, np.float32), None if fac is None else np.array(fac)))
        return super().walk_batch(request, r0, keys, out=out)


class _Factors(_Seen, _Still):
    name = "factors"
    capabilities = frozenset({backends.PERMEABILITY_FACTOR})

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = []


class _CellFieldFactors(_Seen, _CellField):
    name = "cellfieldfactors"
    capabilities = frozenset({backends.PACKED_MYELINATED, backends.FIELD_IN_WALK, backends.PERMEABILITY_FACTOR})

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = []


class _RecordingFactors(_Seen, _Recording):
    name = "recordfactors"
    capabilities = frozenset({backends.TRIAL_RECORD, backends.PERMEABILITY_FACTOR})

    def __init__(self, record, **kw):
        super().__init__(record, **kw)
        self.calls = []


def _rows(w, r0):
    """The walker indices of the starts ``r0`` in the walk ``w``."""
    starts = np.asarray(w.position_coeffs[:, 0, 0], np.float32)
    where = {s.tobytes(): i for i, s in enumerate(starts)}
    assert len(where) == N, "the starts identify the walkers"
    return np.array([where[s.tobytes()] for s in np.asarray(r0, np.float32)])


def _check_rows(w, calls):
    """Every call was handed exactly its walkers' factors, float32, in its rows' order."""
    for request, r0, fac in calls:
        assert fac is not None and fac.dtype == np.float32 and fac.shape == (r0.shape[0],)
        np.testing.assert_array_equal(fac, FAC[_rows(w, r0)])


def test_a_declaring_backend_is_handed_each_batchs_own_rows():
    """32 walkers in batches of 8: every walk batch (and the events pilot before them) receives its own rows."""
    be = _Factors()
    w = _walk(_packing(), backend=be, permeability_factor=FAC)
    batches = [r0 for _, r0, _ in be.calls if r0.shape[0] == 8]
    assert len(batches) == 4 and len(be.calls) == 5                       # the events pilot of all 32, then 4 x 8
    _check_rows(w, be.calls)
    assert sorted(np.concatenate([_rows(w, r0) for r0 in batches]).tolist()) == list(range(N))
    np.testing.assert_array_equal(w.permeability_factor, FAC)


def test_a_backend_that_does_not_declare_the_capability_is_refused_by_its_name():
    be = _Still()
    with pytest.raises(ValueError, match=r"does not declare that it walks a permeability per walker "
                                         r"\(capability 'permeability_factor'\)"):
        _walk(_packing(), backend=be, permeability_factor=FAC)
    assert not be.requests                                                 # walk_batch never called
    w = _walk(_packing(), backend=be)                                      # the neighbour: no factor
    assert w.work["backend"] == "still" and w.permeability_factor is None
    req = be.requests[0]
    assert req.permeability_factor is None
    with_fac = req._replace(permeability_factor=FAC[:8])
    assert backends.capability_refusal(be, "still", with_fac) == (
        "backend 'still': this walk walks each walker at its own permeability (permeability_factor), which the backend "
        "does not declare it walks (capability 'permeability_factor')")
    assert backends.capability_refusal(_Factors(), "factors", with_fac) is None
    assert backends.capability_refusal(be, "still", req) is None


def _no_events(request, r0):
    """An empty C1 event list: the stand-in's walkers never move, so none changes pool."""
    return np.zeros((r0.shape[0] + 1,), np.int64), np.zeros((0,), np.float64), np.zeros((0,), np.int8)


def _field_calls(be):
    """The field pilot's calls: one physics step a block, every walker in the raw subset."""
    return [c for c in be.calls if c[0].sub_steps == 1 and c[0].raw_subset == c[1].shape[0]]


def test_the_field_pilots_batches_receive_their_walkers_rows(monkeypatch):
    """The field pilot walked on the backend: one batch at the default read budget, then batches of 5 walkers at a
    budget of five walkers' positions and reads; each batch its own rows."""
    import dmipy_sim.engine.accumulate as accumulate
    g = _cell(kappa_outer=[0.0, 4e-5, 0.0, 4e-5])
    be = _CellFieldFactors(events=_no_events)
    w = _walk(g, backend=be, field_basis=HollowCylinderField.of(g), permeability_factor=FAC, **FIELD)
    field = _field_calls(be)
    assert len(field) == 1 and field[0][1].shape[0] == N
    _check_rows(w, be.calls)

    steps = int(field[0][0].n_blocks)                                      # the pilot's steps
    record = bool(field[0][0].record)
    monkeypatch.setattr(accumulate, "PILOT_READ_BYTES", 5 * (steps + 1) * (3 + 2 * int(record) + 12) * 4)
    be = _CellFieldFactors(events=_no_events)
    w = _walk(g, backend=be, field_basis=HollowCylinderField.of(g), permeability_factor=FAC, **FIELD)
    field = _field_calls(be)
    assert [c[1].shape[0] for c in field] == [5] * 6 + [2]
    _check_rows(w, be.calls)
    assert sorted(np.concatenate([_rows(w, c[1]) for c in field]).tolist()) == list(range(N))


def test_the_trial_pilot_receives_its_walkers_rows():
    """The trial record's pilot (no slots stated) is walked on the backend: it and the walk's batches get their rows."""
    record, _ = _record(5)
    be = _RecordingFactors(record)
    spec = TrialRecordSpec(modes=3, band_edges=(0.0, 1e-3, 2e-3))
    w = _walk(_packing(), backend=be, record_trials=spec, permeability_factor=FAC)
    assert w.work["trial_slots"]["rule"] == "pilot"
    assert [c[1].shape[0] for c in be.calls] == [N, 8, 8, 8, 8]           # the pilot, then the walk
    _check_rows(w, be.calls)


def test_the_returned_array_and_record_are_the_jax_walks():
    be = _Factors()
    a = _walk(_packing(), backend=be, permeability_factor=FAC)
    b = _walk(_packing(), permeability_factor=FAC)
    assert a.work["backend"] == "factors" and b.work["backend"] == "jax"
    assert a.permeability_factor.dtype == b.permeability_factor.dtype == np.float32
    np.testing.assert_array_equal(a.permeability_factor, b.permeability_factor)
    assert a.work["permeability_factor"] == b.work["permeability_factor"]
