"""MT binding through the backend hook (dmrai-lab/dmrai#72): a walk that binds (the ``"mt"`` kind) is a request a
backend that declares :data:`~dmipy_sim.engine.backends.MT_BINDING` walks -- the binding's ``kappa_MT_over_D`` and
``dwell_time`` in the request, each walker's dwell at the start handed to ``walk_batch(dwell0=)`` -- and it returns
the bound column as ``bound_events``; a backend that does not declare it is refused by the capability's name, the JAX
burn-in is refused for a backend, and a backend that returns no bound events is refused."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine import backends
from dmipy_sim.engine.core import _walk_geometry
from tests.backends.test_backend_hook import coefficient_batch

D = 2e-9
R = 2e-6
KON = 4e-5
TAU = 1e-3


class _StillMT(backends.Backend):
    """Walkers that never move and bind once, at the first block's end, for one block: a backend that declares
    MT_BINDING, recording what it was handed."""
    name = "still_mt"
    capabilities = frozenset({backends.MT_BINDING})

    def __init__(self, bound_events=True):
        self.calls = []
        self.bound_events = bound_events

    def walk_batch(self, request, r0, keys, *, out=None, dwell0=None):
        self.calls.append((request, None if dwell0 is None else np.array(dwell0)))
        b = coefficient_batch(request, r0)
        n = r0.shape[0]
        off = np.arange(0, 2 * n + 1, 2, dtype=np.int64)
        h = float(request.block_h)
        ev = (off, np.tile([h, 2 * h], n).astype(np.float64), np.tile([1, 0], n).astype(np.int8))
        return b._replace(bound_events=(ev if self.bound_events else None))


class _NoMT(_StillMT):
    name = "no_mt"
    capabilities = frozenset()


def _walk(backend, **kw):
    kw.setdefault("equilibrate_binding", "off")
    return _walk_geometry(64, D, d.Sphere(R), 2e-3, 0.5e-3, seed=11, tiers="all", require_gpu=False, K=6,
                          walker_batch_size=32, kappa_MT=KON, dwell_time=TAU, backend=backend, **kw)


def test_a_walk_that_binds_is_handed_its_binding_and_the_walkers_dwell():
    be = _StillMT()
    w = _walk(be, raw_subset=10)
    req, dwell0 = be.calls[-1]
    assert req.kappa_MT_over_D == pytest.approx(KON / D) and req.dwell_time == pytest.approx(TAU) and req.record
    assert req.events > 0                                          # the bound column's capacity, as C1's
    assert dwell0 is not None and dwell0.shape == (32,) and np.all(dwell0 == 0.0)   # 'off': every walker free
    assert np.array_equal(w.bound_start, np.zeros(64, np.int8))
    off, t, lab = w.bound_events
    assert off[-1] == 128 and np.allclose(t[:2], [0.5e-3, 1.0e-3]) and lab[:2].tolist() == [1, 0]
    # the raw subset's bound state at the block ends, read off the events
    assert w.raw_bound.shape == (10, 5) and w.raw_bound[0].tolist() == [0, 1, 0, 0, 0]
    assert w.raw_compartment.shape == (10, 5)


def test_a_backend_that_does_not_declare_mt_binding_is_refused_by_its_name():
    with pytest.raises(ValueError, match="mt_binding"):
        _walk(_NoMT())


def test_the_burn_in_is_refused_for_a_backend():
    with pytest.raises(ValueError, match="burn-in on the JAX kernels"):
        _walk(_StillMT(), equilibrate_binding="burnin")


def test_a_backend_that_returns_no_bound_events_is_refused():
    with pytest.raises(ValueError, match="no bound events"):
        _walk(_StillMT(bound_events=False))
