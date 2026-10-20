"""The void admission case (dmrai-lab/dmipy-sim#675): a backend walking one tube in a 200 um box from starts in
the empty space beyond its own gather must still bound every walker by the tube it CAN reach, not by an unbounded
gather read as infinite. The invariant is per-save containment, never a rate: :func:`backends.void_invariant`."""
import numpy as np

import dmipy_sim as d
from dmipy_sim.engine import backends
from dmipy_sim.engine.core import simulate_trajectories

D = 2e-9


class _LeakyVoid(backends.Backend):
    """A backend that returns a real walk's own coefficients and block ends, with one walker's last block end moved
    inside the tube: a gather that reads an empty 27-cell neighbourhood as an unbounded wall would leak exactly like
    this, and the raw subset's block ends are where the invariant reads it."""
    name = "leaky_void"

    def __init__(self, walk, leak_point):
        self._walk = walk
        self._positions = np.array(walk.raw_positions, copy=True)
        self._positions[0, -1] = leak_point

    def refuses(self, request):
        return None

    def walk_batch(self, request, r0, keys, *, out=None):
        from .test_backend_hook import coefficient_batch
        w = self._walk
        n = r0.shape[0]
        assert request.raw_subset == n == w.n_walkers
        b = coefficient_batch(request, r0, precision="float32", bands=w.position_coeffs[:, 0, 2:],
                              blt_bands=w.boundary_bands[:, 0], comp=w.compartment,
                              qv=w.qv_positions[:, 0])
        zero = np.zeros((n,), np.int32)
        work = (zero.copy(), zero.copy(), zero.copy()) if request.count_walls else (None, None, zero.copy())
        saves = backends.WalkBatch(self._positions, w.raw_boundary, np.asarray(w.raw_compartment, np.float32), zero.copy(), work)
        return b._replace(position_coeffs=np.concatenate([w.position_coeffs[:, 0, :2], b.position_coeffs[:, 2:]], axis=1).astype(np.float64),
                          boundary_endpoint=np.asarray(w.boundary_endpoint[:, 0], np.float64), saves=saves)


def _void_case():
    return backends.parity_cases()["strands_void"]


def test_the_void_case_declares_its_own_starts_and_scale():
    case = _void_case()
    assert isinstance(case, backends.ParityCase) and callable(case.starts)
    g = case()
    assert isinstance(g, d.PackedCurvedCylinders) and g.interior is False and g.permeability is None
    r0 = case.starts(64, case.seed)
    assert r0.shape == (64, 3) and r0.dtype == np.float32
    # every start is beyond the tube's own gather reach, in the void
    xt = yt = 100e-6
    assert np.allclose(r0[:, 0], xt - 2e-6 - 2.5 * g.cell_size)
    assert (np.abs(r0[:, 1] - yt) <= 5e-6).all()
    assert case.n_walkers == 4096 and case.T_max == 40e-3 and case.block_h == 0.5e-3 and case.seed == 5
    # calling the case still builds the geometry alone, exactly as every other case's bare builder does
    assert isinstance(backends.parity_cases()["sphere"](), d.Sphere)


def test_the_jax_rounds_pass_the_void_invariant():
    case = _void_case()
    g = case()
    r0 = case.starts(case.n_walkers, case.seed)
    w = simulate_trajectories(case.n_walkers, D, g, case.T_max, case.block_h, seed=case.seed, tiers="all",
                              walker_batch_size=case.n_walkers, r0=r0, require_gpu=False, raw_subset=case.n_walkers).saves()
    checks = backends.void_invariant(g, w)
    assert checks["contained"]["ok"] and checks["contained"]["n_inside"] == 0
    assert checks["illegal"]["ok"] and checks["illegal"]["n_illegal"] == 0
    assert checks["reaches_wall"]["ok"], checks["reaches_wall"]      # the premise: many walkers reach the wall
    assert checks["reaches_wall"]["n_near"] > 100


def test_the_parity_report_admits_the_jax_backend_on_the_void_case():
    report = backends.parity_report(None, cases={"strands_void": _void_case()})
    assert len(report) == 1 and report[0]["case"] == "strands_void" and report[0]["ok"] is True


def test_a_walk_that_leaks_one_walker_into_the_tube_is_refused_by_the_invariant():
    """A walk with one walker's last block end moved onto the tube's axis fails the per-block containment invariant;
    a backend handing such a walk back (its coefficients and the raw subset's block ends, dmipy-sim#747) is admitted
    on the void case by that invariant alone, and so refused."""
    import dataclasses
    case = _void_case()
    g = case()
    r0 = case.starts(case.n_walkers, case.seed)
    cw = simulate_trajectories(case.n_walkers, D, g, case.T_max, case.block_h, seed=case.seed, tiers="all",
                               walker_batch_size=case.n_walkers, r0=r0, require_gpu=False, raw_subset=case.n_walkers)
    true_w = cw.saves()
    pos = np.array(true_w.positions, copy=True)
    pos[0, -1] = np.array([100e-6, 100e-6, 100e-6], np.float32)       # the tube's axis: deep inside it
    checks = backends.void_invariant(g, dataclasses.replace(true_w, positions=pos))
    assert checks["contained"]["ok"] is False and checks["contained"]["n_inside"] >= 1
    # the leaked walker ends in its own pool (it started and ends outside), so the coarse end-vs-start guard
    # alone does not see it -- the per-block containment invariant is what catches it
    assert checks["illegal"]["n_illegal"] == 0
    report = backends.parity_report(_LeakyVoid(cw, pos[0, -1]), cases={"strands_void": _void_case()})
    assert report[0]["ok"] is False and report[0]["checks"]["contained"]["n_inside"] == 1
    assert report[0]["checks"]["illegal"]["ok"] and report[0]["checks"]["reaches_wall"]["ok"]
