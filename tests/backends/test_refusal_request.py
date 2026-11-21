"""The request a backend's refusal check receives is the request its batches receive: the walk's resolution of a recorded
walk -- the spec, the edge steps, the radii, the overshoot bounds and the slots -- is in it before the first batch.
A stand-in backend refuses a recorded walk without whole-number slots, as a backend that sizes a record by its request
does, and keeps the request its refusal check saw."""
import numpy as np

from dmipy_sim import trial_record
from dmipy_sim.trial_record import TrialRecordSpec

from tests.backends.test_backend_capabilities import SPEC, _Record, _packing, _record, _walk

_TRIAL_FIELDS = ("trial_record", "trial_steps", "trial_radii", "trial_u_max", "trial_slots")


class _Sizing(_Record):
    """A backend that declares the record and refuses a recorded walk whose request carries no slots (a whole number of
    at least 1); every request its refusal check receives is kept. It returns ``record``'s batch for a recorded walk."""
    name = "sizing"

    def __init__(self, record=None, **kw):
        super().__init__(**kw)
        self.record = record
        self.refused = []

    def walk_batch(self, request, r0, keys, *, out=None):
        batch = super().walk_batch(request, r0, keys, out=out)
        return batch if request.trial_record is None else batch._replace(trial_record=self.record(request, r0))

    def refuses(self, request):
        self.refused.append(request)
        slots = request.trial_slots
        if request.trial_record is not None and (isinstance(slots, bool) or not isinstance(slots, (int, np.integer))
                                                 or int(slots) < 1):
            return f"a recorded walk carries trial_slots, a whole number of at least 1, not {slots!r}"
        return None


def test_the_refusal_check_sees_the_slots_the_spec_gives():
    record, _ = _record(3)
    be = _Sizing(record)
    w = _walk(_packing(), backend=be, record_trials=SPEC)
    assert len(be.refused) == 1 and be.refused[0].trial_slots == 8 == SPEC.slots
    assert w.work["trial_slots"]["rule"] == "given"
    for name in _TRIAL_FIELDS:                      # the refusal's request and the first batch's agree on the record
        a, b = getattr(be.refused[0], name), getattr(be.requests[0], name)
        np.testing.assert_array_equal(np.asarray(a, dtype=object) if name == "trial_record" else a,
                                      np.asarray(b, dtype=object) if name == "trial_record" else b, err_msg=name)


def test_the_refusal_check_sees_the_slots_the_pilot_starts_at(monkeypatch):
    """With no slots given the pilot walks at ``min(PILOT_SLOTS, pairs)``: that is what the refusal check sees, the
    first request the backend walks, and, with the pilot's cap lowered to 5, no longer the number of pairs."""
    spec = TrialRecordSpec(modes=3, band_edges=SPEC.band_edges)
    for cap, expected in ((trial_record.PILOT_SLOTS, 8), (5, 5)):
        monkeypatch.setattr(trial_record, "PILOT_SLOTS", cap)
        record, _ = _record(4)
        be = _Sizing(record)
        w = _walk(_packing(), backend=be, record_trials=spec)
        assert [r.trial_slots for r in be.refused] == [expected], cap
        assert be.requests[0].trial_slots == expected, cap
        assert w.work["trial_slots"]["rule"] == "pilot", cap


def test_a_walk_without_a_record_hands_the_refusal_check_no_slots():
    be = _Sizing()
    w = _walk(_packing(), backend=be)
    assert [r.trial_slots for r in be.refused] == [None] and w.trial_record is None
