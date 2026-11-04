"""Membranes of their own permeability, and a permeability that changes in time, against the exact 1-D answer.

The substrate: a slab of three layers, ``R``, ``1.5 R`` and ``R`` wide between reflecting walls, the outer two pool 1
and the middle one pool 0, two membranes of ``kappa R / D`` 2 and 0.25 (:class:`~dmipy_sim.geometry.PermeableSlab1D`
with ``membranes``), every walker starting in pool 1. The reference is the 1-D Bloch-Torrey equation on the same
layers (:class:`dmipy_analytic.bloch_torrey.BlochTorrey1D`), exact up to its cells (its error, twice the Richardson
estimate of a refinement, is added to every tolerance), with the permeability restarted at a breakpoint for the
schedule.

Each walk is at the coarse step ``h = 0.18 R`` (one step per block, so the raw subset holds every step). There the
walk's own error is about 1.5e-3 in occupancy (``LayeredLattice``: second order, ``tests/test_membrane_coefficient.py``),
far inside the tolerances. The tolerances come from the walk's standard errors (``n_resolving``): the walker count is
the one at which the deliberately wrong variant -- the trial reading one permeability for every membrane, or the
permeability of the walk's start for the whole walk -- misses the reference by twice the tolerance at EVERY time it
is compared at, its error computed with the same reference, not typed.

PGSE across the layers (``delta = 0.05``, ``Delta = 0.25 R^2 / D``, ``qR`` 2 and 4) is the walk's own phase, the
trapezoid of each walker's positions under the gradient held over each step, against the reference's signal; its
tolerance is ``z`` times the standard error of the walkers' phases projected on the mean's phase.
"""
import numpy as np
import pytest

from tests.conftest import n_resolving, se_tol, z_family

bt = pytest.importorskip("dmipy_analytic.bloch_torrey")

D = 2e-9
R = 5e-6
H = 0.18                                     # the step over the narrowest layer
EDGES = (0.0, 1.0, 2.5, 3.5)                 # in R: layers of R, 1.5 R and R; pools 1, 0, 1
KAPPA = (2.0, 0.25)                          # kappa R / D of each membrane
START = (1.0, 0.0, 1.0)                      # every walker in pool 1, uniform over its layers
TIMES = (0.03, 0.1, 0.3, 1.0)                # in R^2 / D
PGSE_DELTA, PGSE_BIG_DELTA, QR = 0.05, 0.25, (2.0, 4.0)
DT = (H * R) ** 2 / (6.0 * D)
MAX_WALKERS = 200_000


def _kappa(k):
    return tuple(x * D / R for x in k)


def _model(kappa, breaks=(), n_cells=400):
    kw = dict(permeability_breakpoints=breaks) if breaks else {}
    return bt.BlochTorrey1D(np.asarray(EDGES) * R, D, permeability=kappa, density=START, n_cells=n_cells, **kw)


def _occupancy(kappa, steps, breaks=()):
    """The reference occupancy of each layer after each number of steps of ``DT``, and its error (twice the Richardson
    estimate: four thirds of the change from 400 to 800 cells)."""
    occ = [np.array([_model(kappa, breaks, n).signal(np.zeros(s), DT, by_layer=True).real for s in steps])
           for n in (400, 800)]
    return occ[0], 2.0 * np.abs(occ[0] - occ[1]) * 4.0 / 3.0


def _pgse(qR):
    """The bipolar PGSE on the walk's steps: ``+G`` for ``delta`` from the second step, ``-G`` for ``delta`` from
    ``Delta`` later; ``G`` from ``q R = gamma G delta R``."""
    from dmipy_analytic.constants import GAMMA
    nd, nD = int(round(PGSE_DELTA * R * R / D / DT)), int(round(PGSE_BIG_DELTA * R * R / D / DT))
    G = qR / R / (GAMMA * nd * DT)
    g = np.zeros(nD + nd + 2)
    g[1:1 + nd] = G
    g[1 + nD:1 + nD + nd] = -G
    return g


def _signal_reference(kappa, g, breaks=()):
    """The reference signal of the waveform ``g`` (one value per step of ``DT``) and its error: twice the Richardson
    estimate of the cells (400 to 800) and of the Strang sub-steps (4 to 8)."""
    run = lambda n, sub: _model(kappa, breaks, n).signal(g, DT, sub_steps=sub)
    S, S_cells, S_steps = run(400, 4), run(800, 4), run(400, 8)
    return S, 2.0 * (abs(abs(S) - abs(S_cells)) + abs(abs(S) - abs(S_steps))) * 4.0 / 3.0


def _walk(geometry, n, n_steps, seed):
    from dmipy_sim.engine.core import _walk_geometry
    w = _walk_geometry(n, D, geometry, n_steps * DT, DT, sub_steps=1, seed=seed, raw_subset=n, require_gpu=False)
    assert w.work["exhausted_steps"] == 0
    return w, np.asarray(w.raw_positions, np.float64)[:, :, 0]


def _layers(x):
    """The fraction of the walkers in each layer: ``x`` (n_walkers,) in metres."""
    e = np.asarray(EDGES) * R
    return np.array([((x >= a) & (x < b)).mean() for a, b in zip(e[:-1], e[1:])])


def _walk_signal(x, g):
    """``(|S|, se)``: the walkers' PGSE signal, each phase the trapezoid of its positions under the gradient held over
    each step, and the standard error of the magnitude (the spread projected on the mean's phase)."""
    from dmipy_analytic.constants import GAMMA
    n_t = len(g)
    phi = GAMMA * DT * ((x[:, :n_t] + x[:, 1:n_t + 1]) * 0.5) @ g
    z = np.exp(1j * phi)
    S = z.mean()
    u = S / abs(S)
    return abs(S), float(np.sqrt(np.real(z * np.conj(u)).var() / len(z)))


def _steps(times):
    return [int(round(t * R * R / D / DT)) for t in times]


def test_two_membranes_of_their_own_permeability_exchange_as_the_layered_equation():
    """Occupancy of each layer at ``TIMES`` and PGSE at ``QR``, two membranes of ``kappa R / D`` 2 and 0.25. The trial
    reading one permeability for both (either membrane's) misses the reference by twice the occupancy tolerance at
    every time."""
    from dmipy_sim.geometry import PermeableSlab1D
    steps = _steps(TIMES)
    want, e_ref = _occupancy(_kappa(KAPPA), steps)
    wrong = min(np.abs(_occupancy(_kappa((k, k)), steps)[0] - want).max(axis=1).min() for k in KAPPA)
    comparisons = want.size + len(QR)
    n = n_resolving(wrong, 0.5, comparisons=comparisons)
    assert n <= MAX_WALKERS, n
    tol = se_tol(n, 0.5, comparisons=comparisons)
    g = PermeableSlab1D(length=EDGES[-1] * R, permeability=_kappa(KAPPA), membranes=np.asarray(EDGES[1:-1]) * R)
    w, x = _walk(g, n, max(steps), seed=11)
    for t, s, ref, e in zip(TIMES, steps, want, e_ref):
        got = _layers(x[:, s])
        assert np.all(np.abs(got - ref) < tol + e), f"t = {t} R^2/D: layers {got} against {ref} (tolerance {tol:.4f})"
    z = z_family(comparisons)
    for qR in QR:
        wf = _pgse(qR)
        S_ref, e_S = _signal_reference(_kappa(KAPPA), wf)
        S, se = _walk_signal(x, wf)
        assert abs(S - abs(S_ref)) < z * se + e_S, f"qR = {qR}: |S| {S:.4f} against {abs(S_ref):.4f} (se {se:.4f})"


def test_a_step_change_of_each_membranes_permeability_is_the_restarted_equation():
    """The two membranes swap their permeabilities at ``t_b = 0.1 R^2 / D`` (on a step boundary): occupancy at
    ``TIMES`` and PGSE (whose ``Delta`` spans the breakpoint) against the reference restarted at ``t_b``. The walk that
    keeps the start's permeabilities misses it by twice the tolerance at every time after the breakpoint."""
    from dmipy_sim.geometry import PermeableSlab1D, PermeabilitySchedule
    steps = _steps(TIMES)
    t_b = _steps((0.1,))[0] * DT
    sched = (_kappa(KAPPA), _kappa(KAPPA[::-1]))
    want, e_ref = _occupancy(sched, steps, breaks=(t_b,))
    stuck = _occupancy(_kappa(KAPPA), steps)[0]
    after = [i for i, s in enumerate(steps) if s * DT > t_b]
    wrong = np.abs(stuck - want).max(axis=1)[after].min()
    comparisons = want.size + len(QR)
    n = n_resolving(wrong, 0.5, comparisons=comparisons)
    assert n <= MAX_WALKERS, n
    tol = se_tol(n, 0.5, comparisons=comparisons)
    g = PermeableSlab1D(length=EDGES[-1] * R, permeability=PermeabilitySchedule([t_b], list(sched)),
                        membranes=np.asarray(EDGES[1:-1]) * R)
    w, x = _walk(g, n, max(steps), seed=12)
    rec = w.work["permeability"]
    assert rec["rule"] == "midpoint" and rec["taken_at_s"] == pytest.approx([t_b], rel=1e-12)
    assert rec["shift_s"][0] == pytest.approx(0.0, abs=1e-9 * DT)
    for t, s, ref, e in zip(TIMES, steps, want, e_ref):
        got = _layers(x[:, s])
        assert np.all(np.abs(got - ref) < tol + e), f"t = {t} R^2/D: layers {got} against {ref} (tolerance {tol:.4f})"
    z = z_family(comparisons)
    for qR in QR:
        wf = _pgse(qR)
        S_ref, e_S = _signal_reference(sched, wf, breaks=(t_b,))
        S, se = _walk_signal(x, wf)
        assert abs(S - abs(S_ref)) < z * se + e_S, f"qR = {qR}: |S| {S:.4f} against {abs(S_ref):.4f} (se {se:.4f})"


def test_the_scalar_walk_of_simulate_reads_the_schedule():
    """``simulate`` on the slab (not served from a coefficient walk) runs the fused walk, which reads the schedule at
    each step's midpoint too: sealed membranes opening to their own permeabilities at ``t_b``, PGSE at ``qR = 3``
    across the breakpoint, against the restarted reference with the conservative per-walker spread ``sd = 1``; the
    start's (sealed) membranes throughout miss it by twice that tolerance."""
    from dmipy_sim import simulate
    from dmipy_sim.geometry import PermeableSlab1D, PermeabilitySchedule
    from dmipy_replay.sequences.builders import from_waveform
    t_b = _steps((0.1,))[0] * DT
    sched = ((0.0, 0.0), _kappa(KAPPA))
    wf = _pgse(3.0)
    S_ref, e_S = _signal_reference(sched, wf, breaks=(t_b,))
    S_stuck, _ = _signal_reference(sched[0], wf)
    wrong = abs(abs(S_stuck) - abs(S_ref))
    n = n_resolving(wrong, 1.0)
    assert n <= MAX_WALKERS, n
    tol = se_tol(n, 1.0)
    G = np.zeros((1, len(wf), 3))
    G[0, :, 0] = wf                                  # one step of DT per sample, the sample's gradient held over it
    seq = from_waveform(G, DT, [[1.0, 0.0, 0.0]])
    g = PermeableSlab1D(length=EDGES[-1] * R, permeability=PermeabilitySchedule([t_b], list(sched)),
                        membranes=np.asarray(EDGES[1:-1]) * R)
    S = complex(np.asarray(simulate(n, D, seq, g, seed=13, sub_steps=1, require_gpu=False)).ravel()[0])
    assert abs(abs(S) - abs(S_ref)) < tol + e_S, (abs(S), abs(S_ref), tol)
