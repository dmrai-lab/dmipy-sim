"""The curved-tube family reports its wall work (dmrai-lab/dmipy-sim#648): the bounces taken as hits when the
geometry counts, the guard's correction as ``illegal``, nothing exhausted (two bounces by argument, no budget
loop); counting changes no position. And the strands are parity cases a backend is admitted on."""
import numpy as np

import dmipy_sim as d
from dmipy_sim.engine import backends

D = 2e-9


def _pack(interior):
    return backends.parity_cases()[f"strands_{'intra' if interior else 'extra'}"]()


def _walk(g, **kw):
    return d.simulate_trajectories(400, D, g, 1e-3, 2.5e-4, seed=5, require_gpu=False, tiers="all",
                                   walker_batch_size=400, **kw)


def test_counting_reports_the_bounces_and_changes_no_position():
    for interior in (True, False):
        plain = _walk(_pack(interior))
        g = _pack(interior); g.count_walls = True
        counted = _walk(g)
        np.testing.assert_array_equal(plain.positions, counted.positions)
        np.testing.assert_array_equal(plain.boundary_local_time, counted.boundary_local_time)
        np.testing.assert_array_equal(plain.compartment, counted.compartment)
        assert plain.work["n_hits"] is None
        assert counted.work["n_hits"] > 0, interior
        assert counted.work["n_crossings"] == 0 and counted.work["exhausted_steps"] == 0
        # the hits are the contact: a walker that bounced has a non-zero local time, one that did not has none
        blt = np.asarray(counted.boundary_local_time)
        assert (blt != 0).any()


def test_the_strands_are_parity_cases():
    cases = backends.parity_cases()
    for name, interior in (("strands_intra", True), ("strands_extra", False)):
        g = cases[name]()
        assert isinstance(g, d.PackedCurvedCylinders) and g.interior is interior and g.permeability is None
