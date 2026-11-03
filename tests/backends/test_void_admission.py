"""The void admission case (dmrai-lab/dmipy-sim#675): a backend walking one tube in a 200 um box from starts in
the empty space beyond its own gather must still bound every walker by the tube it CAN reach, not by an unbounded
gather read as infinite. The invariant is per-save containment, never a rate: :func:`backends.void_invariant`."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine import backends
from dmipy_sim.engine.core import _walk_geometry

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


#: The case declares 4096 walkers for 40 ms; the walking tests use a quarter of them (1024) and walk the case ONCE (a
#: module fixture): the invariant is read from the raw block ends of that one walk, and the report path and the leaking
#: backend are exercised on it. The invariant is per walker, so it needs no count; its premise (walkers do reach the
#: wall) was 430 of 4096 near it, 108 of 1024, so the premise bound scales to a quarter of 100.
N_VOID = 1024


@pytest.fixture(scope="module")
def void_walk():
    case = _void_case()._replace(n_walkers=N_VOID)
    g = case()
    r0 = case.starts(case.n_walkers, case.seed)
    w = _walk_geometry(case.n_walkers, D, g, case.T_max, case.block_h, seed=case.seed, tiers="all",
                       walker_batch_size=case.n_walkers, r0=r0, require_gpu=False, raw_subset=case.n_walkers)
    return g, w


def test_the_jax_rounds_pass_the_void_invariant(void_walk):
    """The jax kernels' own walk of the case passes the invariant: no walker inside the tube, no refused crossing,
    and the premise holds -- many walkers reach the wall. (``parity_report(None, ...)`` is this same walk with the
    wall counters on, read by this same invariant; the last test drives that report path.)"""
    g, w = void_walk
    checks = backends.void_invariant(g, w)
    assert checks["contained"]["ok"] and checks["contained"]["n_inside"] == 0
    # A refused step (the walker is held at its position when a step ends in another pool with no crossing granted) is
    # the float32 kernels' rare grazing event, not a leak: CI saw 0 in 4096 walkers on one platform and 1 in 1024 on
    # another. The hard claim for the jax kernels is containment; refusals are bounded by a rate of 1e-3 per walker
    # (a Poisson tail far above the observed 1 in 1000+; a gather read as unbounded would refuse or leak by the hundred).
    assert checks["illegal"]["n_illegal"] <= max(1, int(1e-3 * w.n_walkers) + 4), checks["illegal"]
    assert checks["reaches_wall"]["ok"], checks["reaches_wall"]      # the premise: many walkers reach the wall
    assert checks["reaches_wall"]["n_near"] > 25


def test_a_walk_that_leaks_one_walker_into_the_tube_is_refused_by_the_invariant(void_walk):
    """A walk with one walker's last block end moved onto the tube's axis fails the per-block containment invariant;
    a backend handing such a walk back (its coefficients and the raw subset's block ends, dmipy-sim#747) is admitted
    on the void case by that invariant alone, and so refused. The unmoved walk is admitted by the same report."""
    import dataclasses
    g, cw = void_walk
    pos = np.array(cw.raw_positions, copy=True)
    pos[0, -1] = np.array([100e-6, 100e-6, 100e-6], np.float32)       # the tube's axis: deep inside it
    checks = backends.void_invariant(g, dataclasses.replace(cw, raw_positions=pos))
    assert checks["contained"]["ok"] is False and checks["contained"]["n_inside"] >= 1
    # the leaked walker ends in its own pool (it started and ends outside), so the coarse end-vs-start guard
    # alone does not see it -- the per-block containment invariant is what catches it
    assert checks["illegal"]["n_illegal"] == int(np.asarray(cw.illegal_crossings).sum())    # moving a block end adds none
    cases = {"strands_void": _void_case()._replace(n_walkers=N_VOID)}
    report = backends.parity_report(_LeakyVoid(cw, pos[0, -1]), cases=cases, sweep=2)
    assert report[0]["ok"] is False and report[0]["checks"]["contained"]["n_inside"] == 1
    assert report[0]["checks"]["illegal"]["ok"] and report[0]["checks"]["reaches_wall"]["ok"]
    honest = backends.parity_report(_LeakyVoid(cw, cw.raw_positions[0, -1]), cases=cases, sweep=2)    # the true block ends handed back
    assert honest[0]["ok"] is True
