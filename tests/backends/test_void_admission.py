"""The void admission case (dmrai-lab/dmipy-sim#675): a backend walking one tube in a 200 um box from starts in
the empty space beyond its own gather must still bound every walker by the tube it CAN reach, not by an unbounded
gather read as infinite. The invariant is per-save containment, never a rate: :func:`backends.void_invariant`."""
import numpy as np

import dmipy_sim as d
from dmipy_sim.engine import backends
from dmipy_sim.engine.core import simulate_trajectories

D = 2e-9


class _LeakyVoid(backends.Backend):
    """A backend that returns a real walk's own positions, with one walker's last save moved inside the tube: a
    gather that reads an empty 27-cell neighbourhood as an unbounded wall would leak exactly like this."""
    name = "leaky_void"

    def __init__(self, true_positions, leak_point):
        self._positions = np.array(true_positions, copy=True)
        self._positions[0, -1] = leak_point

    def refuses(self, request):
        return None

    def walk_batch(self, request, r0, keys, *, out=None):
        n = r0.shape[0]
        zero = np.zeros((n,), np.int32)
        blt = np.zeros((n, request.n_t), np.float32) if request.record else None
        comp = np.ones((n, request.n_t), np.float32) if request.record else None
        work = (zero.copy(), zero.copy(), zero.copy()) if request.count_walls else (None, None, zero.copy())
        return backends.WalkBatch(self._positions, blt, comp, zero.copy(), work)


def _void_case():
    return backends.parity_cases()["strands_void"]()


def test_the_void_case_is_a_parity_case_with_its_own_starts_and_scale():
    g = _void_case()
    assert isinstance(g, d.PackedCurvedCylinders) and g.interior is False and g.permeability is None
    assert callable(g.parity_starts)
    r0 = g.parity_starts(64, 11)
    assert r0.shape == (64, 3) and r0.dtype == np.float32
    # every start is beyond the tube's own gather reach, in the void
    xt = yt = 100e-6
    assert np.allclose(r0[:, 0], xt - 2e-6 - 2.5 * g.cell_size)
    assert (np.abs(r0[:, 1] - yt) <= 5e-6).all()
    overrides = g.parity_overrides
    assert overrides["n_walkers"] == 4096 and overrides["T_max"] == 40e-3 and overrides["dt_save"] == 0.5e-3


def test_the_jax_rounds_pass_the_void_invariant():
    g = _void_case()
    ov = g.parity_overrides
    r0 = g.parity_starts(ov["n_walkers"], 11)
    w = simulate_trajectories(ov["n_walkers"], D, g, ov["T_max"], ov["dt_save"], seed=11, tiers="all",
                              walker_batch_size=ov["n_walkers"], r0=r0, require_gpu=False)
    checks = backends.void_invariant(g, w)
    assert checks["contained"]["ok"] and checks["contained"]["n_inside"] == 0
    assert checks["illegal"]["ok"] and checks["illegal"]["n_illegal"] == 0
    assert checks["reaches_wall"]["ok"], checks["reaches_wall"]      # the premise: many walkers reach the wall
    assert checks["reaches_wall"]["n_near"] > 100


def test_the_parity_report_admits_the_jax_backend_on_the_void_case():
    report = backends.parity_report(None, cases={"strands_void": backends.parity_cases()["strands_void"]})
    assert len(report) == 1 and report[0]["case"] == "strands_void" and report[0]["ok"] is True


def test_a_backend_that_leaks_one_walker_into_the_tube_is_refused_by_the_invariant():
    g = _void_case()
    ov = g.parity_overrides
    r0 = g.parity_starts(ov["n_walkers"], 11)
    true_w = simulate_trajectories(ov["n_walkers"], D, g, ov["T_max"], ov["dt_save"], seed=11, tiers="all",
                                   walker_batch_size=ov["n_walkers"], r0=r0, require_gpu=False)
    leak_point = np.array([100e-6, 100e-6, 100e-6], np.float32)   # the tube's axis: deep inside it
    bad = _LeakyVoid(true_w.positions, leak_point)
    report = backends.parity_report(bad, cases={"strands_void": _void_case})
    rec = report[0]
    assert rec["ok"] is False
    assert rec["checks"]["contained"]["ok"] is False and rec["checks"]["contained"]["n_inside"] >= 1
    # the leaked walker ends in its own pool (it started and ends outside), so the coarse end-vs-start guard
    # alone does not see it -- the per-save containment invariant is what catches it
    assert rec["checks"]["illegal"]["n_illegal"] == 0
