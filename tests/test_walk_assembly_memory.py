"""The engine assembles a walk into one array per channel as its batches arrive, never a list of batches beside
their concatenation (dmrai-lab/dmipy-sim#438)."""
import tracemalloc

import numpy as np

import dmipy_sim as d

UM = 1e-6


def _walk(**kw):
    return d.simulate_trajectories(3000, 2e-9, d.Sphere(3 * UM), T_max=4e-3, dt_save=2e-5, seed=0, tiers="all",
                                   require_gpu=False, **kw)


def test_the_walk_is_held_once_while_it_is_assembled():
    w = _walk(walker_batch_size=500)                                   # warm the kernels outside the measurement
    held = w.positions.nbytes + w.boundary_local_time.nbytes + w.compartment.nbytes
    del w
    tracemalloc.start()
    w = _walk(walker_batch_size=500)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert held > 8e6, held                                            # a walk big enough to see: 10 MB of channels
    assert peak < 1.5 * held, f"peak {peak / 1e6:.0f} MB for {held / 1e6:.0f} MB of channels"


def test_the_assembled_walk_is_the_concatenation_of_its_batches(monkeypatch):
    """Bit for bit what a list of batches concatenated at the end gave."""
    from dmipy_sim.engine import core

    class _ListRows(list):
        def array(self):
            return np.concatenate(self, axis=0)

    a = _walk(walker_batch_size=500)
    monkeypatch.setattr(core, "_Rows", lambda n: _ListRows())
    b = _walk(walker_batch_size=500)
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.boundary_local_time, b.boundary_local_time)
    assert np.array_equal(a.compartment, b.compartment)
