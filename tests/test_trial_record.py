"""The record of a walk's membrane trials at its plain walls (:mod:`dmipy_sim.trial_record`).

One test per requirement, each exact: integer counts compared as integers, sums at relative 1e-12 in float64 (the
tests that compare sums accumulate in float64, under the ``x64`` fixture). The requirements:

1. a trial is accounted in the cell of its object and the band of its step, with its overshoot and its outcome;
2. only the trials of steps the walk keeps are recorded (a walker held bound records none);
3. every encounter of a plain wall is a trial, later encounters of a step and sealed objects included, and the counts
   are the walk's work counts on an all-plain cell;
4. the record changes nothing: one step of each wall kernel is the same bits with and without it, over every branch;
   the record is exactly consistent with its own walk's counts and events; and the whole walk with and without it is
   the same bits on the recorded build and CPU families (the ``test_recorded_bits_*`` tests, ``tests/recorded_bits.py``);
5. a trial's overshoot is within its object's a priori bound, and a walk that exceeds it is refused;
6. a scheduled permeability is read at the step's midpoint and the band is the step's end;
8. the trial count is stored beside the granted count;
9. a recorded walk's start and field read are stated, not chosen by its own walkers;
10. a walker's row is its index in the seed, through joins, windows and the walk file;
11. nothing is rounded: the sums are a recomputation of the kernel's own per-trial log;
12. the record is held in bounded slots per walker, one per (object, band) pair met: the slotted record is the record
    of one cell per pair, exactly, on the same walk; a walker that meets more pairs than the slots refuses the walk by
    name with a counted, never dropped, overflow; the slots are stated or set by a pilot whose rule is recorded; the
    walk itself does not change; and the record's bytes do not grow with the cell's object count.

The record of one cell per (object, band) pair that the slotted record is compared to is :func:`_dense_record` below,
a test-only reference carried through the walk beside the slotted record by ``_walk_geometry``'s ``_trial_device``.

Requirement 7 (the coin) is ``tests/geometry/test_trial_coin.py``.
"""
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_sim.compartments import Compartments, Pool
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.geometry import PackedCylinders, PackedMyelinatedCylinders, pack_cylinders
from dmipy_sim.geometry._boundary import TrialLog
from dmipy_sim.geometry.myelin import concentric_wall_kernel
from dmipy_sim.geometry.packed import packed_wall_kernel
from dmipy_sim.geometry.packing import periodic_min_gap
from dmipy_sim.math.step_law import LONGEST_STEP
from dmipy_sim.trial_record import TrialRecord, TrialRecordSpec, chebyshev_terms, device_record, u_max_of

D = 2e-9
T = 4e-3
BLOCK = 2.5e-4
RADII = np.array([1.0, 1.3, 0.8, 1.1, 0.9, 1.2]) * 1e-6
KAPPA = np.array([4e-3, 0.0, 4e-3, 4e-3, 4e-3, 4e-3])          # object 1 sealed
SPEC = TrialRecordSpec(modes=3, band_edges=(0.0, 1e-3, 2.5e-3, T))

R = 2e-6
OUTER = np.array([R, 0.8 * R, R, 0.9 * R])
L_CELL = 4.4 * R
CENTRES = np.array([[0.0, 0.0], [2.2, 0.0], [0.0, 2.2], [2.2, 2.2]]) * R
COMPS = Compartments(intra=Pool(D=D), myelin=Pool(D=0.0), extra=Pool(D=D))


def _dense(**kw):
    c, Lc, _ = pack_cylinders(RADII, target_vf=0.45, seed=1)
    return PackedCylinders(RADII, c, Lc, permeability=kw.pop("permeability", KAPPA), **kw)


def _mixed(**kw):
    g = np.array([0.7, 1.0, 0.7, 1.0])                                   # cylinders 1 and 3 plain
    return PackedMyelinatedCylinders(OUTER * g, g, CENTRES, L_CELL, N_max=4, compartments=COMPS, kappa_inner=1e-5,
                                     kappa_outer=[0.0, 4e-3, 0.0, 4e-3], **kw)


def _walk(geo, n=32, T_max=T, **kw):
    kw.setdefault("K", 8)
    return _walk_geometry(n, D, geo, T_max, BLOCK, seed=3, tiers="all", require_gpu=False, **kw)


def _one_walker_dense(rec, u_max, band_edges, modes):
    """One walker's slotted record (``device_record``'s ``(key, n, N, S, count, exceed)``) as ``(n, N, S)`` of one cell
    per (object, band) pair (:meth:`TrialRecord.to_dense`)."""
    key, n, N, S, count, _ = (np.asarray(x)[None] for x in rec)
    r = TrialRecord.from_slots(key, n, N, S, count, u_max=u_max, band_edges=band_edges, modes=modes,
                               object_radii=np.ones(len(u_max)))
    d = r.to_dense()
    return d.n[0], d.N[0], d.S[0]


def _dense_record(steps, u_max, modes, dtype):
    """The test's reference: the record of one cell per (object, band) pair, carried per walker as ``(init, add)`` in the
    form of :func:`~dmipy_sim.trial_record.device_record` (``init`` takes the slots and ignores them). ``init`` is ``(n
    (n_objects, n_bands) int32, N (n_objects, n_bands) int32, S (n_objects, n_bands, modes), exceed int32)``; ``add``
    scatters each step's trial log into the cell of its object and band, as one ``.at[].add`` per array."""
    n_bands = int(len(steps) - 1)
    interior = jnp.asarray(np.asarray(steps[1:-1], np.int64), jnp.int32)
    umax = jnp.asarray(np.asarray(u_max, np.float64), dtype)
    n_objects = len(u_max)

    def init(slots):
        z = jnp.zeros((n_objects, n_bands), jnp.int32)
        return (z, z, jnp.zeros((n_objects, n_bands, modes), dtype), jnp.int32(0))

    def add(rec, log, step, keep=None):
        n, N, S, exceed = rec
        valid = log.valid if keep is None else (log.valid & keep)
        band = jnp.sum(interior < (jnp.asarray(step, jnp.int32) + 1)).astype(jnp.int32)
        obj = log.obj.astype(jnp.int32)
        n = n.at[obj, band].add(valid.astype(jnp.int32))
        N = N.at[obj, band].add((valid & log.granted).astype(jnp.int32))
        u = log.u.astype(dtype)
        um = umax[obj]
        terms = u[:, None] * chebyshev_terms(2 * u / um - 1, modes, jnp)
        S = S.at[obj, band].add(jnp.where(valid[:, None], terms, jnp.zeros((), dtype)))
        exceed = exceed + jnp.sum(valid & (u > um)).astype(jnp.int32)
        return (n, N, S, exceed)

    return init, add


def _dense_and_slotted(steps, u_max, modes, dtype):
    """The reference and the slotted record carried side by side in one walk: ``((n, N, S, exceed), (key, n, N, S,
    count, exceed))`` per walker, each step's one trial log added to both."""
    di, da = _dense_record(steps, u_max, modes, dtype)
    si, sa = device_record(steps, u_max, modes, dtype)
    return (lambda slots: (di(slots), si(slots)),
            lambda rec, log, step, keep=None: (da(rec[0], log, step, keep), sa(rec[1], log, step, keep)))


ARRAYS = ("position_coeffs", "boundary_endpoint", "boundary_bands", "compartment", "bound_start", "qv_positions",
          "qv_boundary", "field_coeffs", "qv_field", "qv_field_increments")


def _assert_same_walk(a, b):
    for name in ARRAYS:
        x, y = getattr(a, name), getattr(b, name)
        assert (x is None) == (y is None), name
        if x is not None:
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
    for name in ("comp_events", "bound_events"):
        x, y = getattr(a, name), getattr(b, name)
        assert (x is None) == (y is None), name
        if x is not None:
            for f in ("offsets", "t", "label"):
                np.testing.assert_array_equal(getattr(x, f), getattr(y, f), err_msg=f"{name}.{f}")
    wa = {k: v for k, v in a.work.items() if k not in ("block", "trial_slots")}     # the record states its slots
    wb = {k: v for k, v in b.work.items() if k not in ("block", "trial_slots")}
    assert wa == wb
    assert a.illegal_crossings == b.illegal_crossings


@pytest.fixture(scope="module")
def plain_walks():
    """The dense all-plain cell walked with the record and without it (counting on in both)."""
    ref = _dense()
    ref.count_walls = True
    return _walk(ref), _walk(_dense(), record_trials=SPEC)


@pytest.fixture(scope="module")
def mixed_walks():
    ref = _mixed()
    ref.count_walls = True
    return _walk(ref), _walk(_mixed(), record_trials=SPEC)


# ── 1. a trial's content ────────────────────────────────────────────────────────────────────────────────────────────

def test_a_trial_is_accounted_in_its_cell_with_its_overshoot_and_outcome(x64):
    """Six trials (three granted) over two steps in two cells: ``n``, ``N`` and ``S_j`` are the sums by hand, ``S_j``
    against ``numpy.polynomial.chebyshev`` at relative 1e-12."""
    steps = np.array([0, 4, 10])                                       # two bands: steps ending in [0, 4) and [4, 10]
    u_max = np.array([0.5, 0.25, 0.4])
    init, add = device_record(steps, u_max, 4, jnp.float64)
    rec = init(6)
    # step 1 (ends at 2: band 0): three entries of the loop, two trials at object 0, one iteration without a trial
    log1 = TrialLog(jnp.array([True, True, False, False]), jnp.array([0, 0, 2, 0], jnp.int32),
                    jnp.array([0.1, 0.3, 0.2, 0.0], jnp.float32), jnp.array([True, False, True, False]))
    # step 6 (ends at 7: band 1): four trials at object 2
    log2 = TrialLog(jnp.array([True, True, True, True]), jnp.array([2, 2, 2, 2], jnp.int32),
                    jnp.array([0.05, 0.35, 0.15, 0.25], jnp.float32), jnp.array([False, True, True, False]))
    rec = add(rec, log1, jnp.int32(1))
    rec = add(rec, log2, jnp.int32(6))
    exceed = np.asarray(rec[5])
    n, N, S = _one_walker_dense(rec, u_max, (0.0, 4.0, 10.0), 4)
    want_n = np.zeros((3, 2), int); want_n[0, 0] = 2; want_n[2, 1] = 4
    want_N = np.zeros((3, 2), int); want_N[0, 0] = 1; want_N[2, 1] = 2
    np.testing.assert_array_equal(n, want_n)
    np.testing.assert_array_equal(N, want_N)
    assert int(exceed) == 0
    want_S = np.zeros((3, 2, 4))
    for (obj, band, us) in ((0, 0, [0.1, 0.3]), (2, 1, [0.05, 0.35, 0.15, 0.25])):
        u = np.asarray(np.asarray(us, np.float32), np.float64)        # the overshoot as the kernel holds it
        x = 2.0 * u / u_max[obj] - 1.0
        for j in range(4):
            want_S[obj, band, j] = np.sum(u * np.polynomial.chebyshev.chebval(x, np.eye(4)[j]))
    np.testing.assert_allclose(S, want_S, rtol=1e-12, atol=0.0)


# ── 2. only the deciding trials ─────────────────────────────────────────────────────────────────────────────────────

def test_a_walker_held_bound_records_no_trial(monkeypatch):
    """The MT walk on the all-plain cell, started with about half the walkers bound for a dwell far longer than the
    walk ('fast', at a stated surface-to-volume): a walker bound at the start and never released records ``n = 0`` in
    every cell, and the record of the free-and-bound walk sums to its own work count."""
    from dmipy_sim.engine import mt
    monkeypatch.setattr(mt, "surface_to_volume", lambda geometry: 1.0e4)          # kappa_MT dwell S/V = 1: P_eq = 1/2
    w = _walk(_dense(), n=64, kappa_MT=1e-5, dwell_time=10.0, equilibrate_binding="fast", record_trials=SPEC)
    r = w.trial_record
    off = np.asarray(w.bound_events.offsets)
    held = (np.asarray(w.bound_start) == 1) & (np.diff(off) == 0)                  # bound at 0, no event after
    assert 8 <= held.sum() <= 56, held.sum()
    assert int(r.n[held].sum()) == 0 and int(r.N[held].sum()) == 0
    assert int(r.n[~held].sum()) > 0
    assert int(r.n.sum()) == w.work["n_hits"] and int(r.N.sum()) == w.work["n_crossings"]


# ── 3. every trial at which the permeability enters ─────────────────────────────────────────────────────────────────

def _kernel_steps(radii, centres, Lc, n, seed, step_max):
    """``n`` starts (half uniform, half near a wall on either side) and in-plane steps up to ``step_max``."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-Lc / 2, Lc / 2, (n // 2, 2))
    k = rng.integers(0, len(radii), n - n // 2)
    off = rng.choice([1e-3, 0.1, 0.5, 1.0], len(k)) * step_max * rng.choice([1.0, -1.0], len(k))
    ang = rng.uniform(0, 2 * np.pi, len(k))
    near = centres[k] + np.maximum(radii[k] + off, 0.1 * radii[k])[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)
    pts = (np.concatenate([pts, near]) + Lc / 2) % Lc - Lc / 2
    th = rng.uniform(0, 2 * np.pi, n)
    q = pts[:, None, :] - centres[None]
    q -= Lc * np.round(q / Lc)
    inside = (np.sum(q * q, -1) < radii[None] ** 2)
    return (pts.astype(np.float32), np.stack([np.cos(th), np.sin(th)], 1).astype(np.float32),
            (step_max * rng.uniform(0.0, 1.0, n) ** 0.5).astype(np.float32), inside.any(1), np.argmax(inside, 1))


def test_every_encounter_of_a_step_is_a_trial():
    """Two cylinders 0.05 R apart and steps up to the longest the step law takes: per step, the trials logged equal
    the encounters the loop counts (reflections plus crossings), steps meeting three walls and more included, each
    at one of the two walls; a sealed wall's encounters are trials that are never granted."""
    r = 1e-6
    radii = np.array([r, r]); centres = np.array([[0.0, 0.0], [2.05 * r, 0.0]]); Lc = 20 * r
    f32 = np.float32
    step_max = LONGEST_STEP * r / 6
    w = packed_wall_kernel(jnp.asarray(centres, f32), jnp.asarray(radii, f32), f32(Lc), f32(1e-7 * r), f32(1e-4 * r),
                           step_max, periodic_min_gap(centres, radii, Lc), count=True, trials=True)
    pts, dirs, ls, inside, _ = _kernel_steps(radii, centres, Lc, 4000, 5, 8 * step_max)   # long: many encounters
    keys = jax.random.split(jax.random.PRNGKey(5), len(pts))
    kod = jnp.asarray([2e6, 0.0], jnp.float32)                         # kappa / D (1/m); object 1 sealed
    h = jax.jit(jax.vmap(lambda p, d, l, s, k: w(p, d, l, s, kod, f32(0.0), k)))(
        jnp.asarray(pts), jnp.asarray(dirs), jnp.asarray(ls), jnp.asarray(inside), keys)
    valid = np.asarray(h.trials.valid); obj = np.asarray(h.trials.obj); granted = np.asarray(h.trials.granted)
    n_hits = np.asarray(h.n_hits)
    np.testing.assert_array_equal(valid.sum(1), n_hits)
    assert (n_hits >= 3).sum() > 0
    many = n_hits >= 3
    assert set(np.unique(obj[many][valid[many]])) == {0, 1}
    assert (valid & (obj == 1)).sum() > 0 and not (granted & valid & (obj == 1)).any()
    assert (granted & valid & (obj == 0)).sum() > 0


def test_the_records_counts_are_the_walks_work_on_an_all_plain_cell(plain_walks):
    """On the all-plain cell ``sum n`` is ``work["n_hits"]`` and ``sum N`` is ``work["n_crossings"]``, exactly; the
    sealed object's trials are counted and never granted."""
    _, w = plain_walks
    r = w.trial_record
    assert int(r.n.sum()) == w.work["n_hits"] and int(r.N.sum()) == w.work["n_crossings"]
    d = r.to_dense()
    assert int(d.n[:, 1].sum()) > 0 and int(d.N[:, 1].sum()) == 0


def test_a_mixed_cells_record_is_its_plain_walls_encounters():
    """2000 steps through the mixed cell's kernel (plain and myelinated cylinders, the walkers outside the sheaths or
    in a plain lumen) and through the plain packing's kernel on the same disks (a myelinated cylinder's outer wall
    reflecting): the cell's trials per plain object are the packing's encounters of that object, step for step; a step
    meeting only myelinated walls logs nothing."""
    c, Lc, _ = pack_cylinders(RADII, target_vf=0.45, seed=1)
    radii = RADII / 0.999
    plain = np.array([True, False, True, False, True, False])
    N = len(radii); f32 = np.float32
    nudge = f32(1e-4 * min(radii.min(), (radii * 0.7).min()))
    eps = f32(1e-7 * radii.min())
    step_max = LONGEST_STEP * radii.min() / 6
    min_gap = periodic_min_gap(c, radii, Lc)
    kap = np.where(plain, 4e-2, 0.0).astype(f32)
    inner = np.where(plain, radii, 0.7 * radii)
    cj, rj = jnp.asarray(c, f32), jnp.asarray(radii, f32)
    Dj = jnp.full((N,), f32(D))
    kod = jnp.asarray(kap) / jnp.maximum(Dj, 1e-30)
    w_p = packed_wall_kernel(cj, rj, f32(Lc), eps, nudge, step_max, min_gap, count=True, trials=True)
    w_u = concentric_wall_kernel(cj, jnp.asarray(inner, f32), rj, f32(Lc), Dj, Dj, Dj, jnp.zeros(N, f32), jnp.asarray(kap),
                                 jnp.zeros((N, 4), f32), eps, nudge, step_max, min_gap, count=True, trials=True)
    pts, dirs, ls, inside, own = _kernel_steps(radii, c, Lc, 2400, 9, step_max)
    keep = ~inside | plain[own]
    pts, dirs, ls, inside, own = pts[keep][:2000], dirs[keep][:2000], ls[keep][:2000], inside[keep][:2000], own[keep][:2000]
    keys = jax.random.split(jax.random.PRNGKey(9), len(pts))
    words = jax.vmap(lambda k: jax.random.bits(k, (w_p.bounce_loop.budget,), dtype=jnp.uint32))(keys)
    hp = jax.jit(jax.vmap(lambda p, d, l, s, k: w_p(p, d, l, s, kod, f32(0.0), k)))(
        jnp.asarray(pts), jnp.asarray(dirs), jnp.asarray(ls), jnp.asarray(inside), keys)
    hu = jax.jit(jax.vmap(lambda p, d, l, pl, k, u: w_u(p, d, l, pl, k, f32(0.5), u)))(
        jnp.asarray(pts), jnp.asarray(dirs), jnp.asarray(ls), jnp.asarray(inside.astype(np.int32)),
        jnp.asarray(own.astype(np.int32)), words[:, :w_u.bounce_loop.budget])

    def per_object(t, objects):
        v, o = np.asarray(t.valid), np.asarray(t.obj)
        return np.stack([(v & (o == k)).sum(1) for k in objects], 1)

    plain_ids = np.flatnonzero(plain)
    n_p = per_object(hp.trials, plain_ids)                                # the packing's encounters of the plain walls
    n_u = per_object(hu.trials, range(N))
    np.testing.assert_array_equal(n_u[:, plain_ids], n_p)
    assert n_u[:, ~plain].sum() == 0
    only_sheathed = (np.asarray(hp.n_hits) > 0) & (n_p.sum(1) == 0)
    assert only_sheathed.sum() > 0 and n_u[only_sheathed].sum() == 0
    np.testing.assert_array_equal(np.asarray(hu.trials.granted)[np.asarray(hu.trials.valid)].sum(),
                                  np.asarray(hp.trials.granted)[np.asarray(hp.trials.valid)].sum())


# ── 4. the record changes nothing ───────────────────────────────────────────────────────────────────────────────────
#
# The record reads the walk and enters none of its arithmetic: one step of each wall kernel is the same bits with and
# without it, over states that reach every branch (below, on every host and build). The whole walk with and without it
# is two compiled programs; on the recorded build and CPU families they are the same bits (the test_recorded_bits_*
# tests, tests/recorded_bits.py), and another build may arrange the programs so that a position's last bits differ, which the
# reflections can grow into another encounter: two valid walks of one law. What a reweighted read needs is the record
# consistent with its own walk, which is asserted exactly on every host (``test_the_record_is_exactly_its_own_walks_count``).

def _branch_states(radii, centres, Lc, plain, step_max, n, seed):
    """Starts and steps reaching every branch of a step: no wall met, a trial granted, a trial refused, a sealed
    face, several encounters, and steps up to eight times the longest so that a small budget runs out."""
    pts, dirs, ls, inside, own = _kernel_steps(radii, centres, Lc, n, seed, step_max)
    keep = ~inside | plain[own]
    long = np.random.default_rng(seed + 1).uniform(0.0, 8.0, keep.sum()).astype(np.float32)
    ls = np.where(np.arange(keep.sum()) % 4 == 0, ls[keep] * long, ls[keep])
    return pts[keep], dirs[keep], ls, inside[keep], own[keep]


def _assert_tree_equal(a, b, what):
    for i, (x, y) in enumerate(zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=f"{what}: output {i}")


@pytest.mark.parametrize("budget", [None, 1], ids=["derived_budget", "budget_1"])
def test_one_step_of_the_plain_kernel_is_the_same_with_and_without_the_record(budget):
    """The packed wall kernel with and without its trial log, the same starts, steps and words: every output bit for
    bit (end position, contact, crossing, sentinel, encounters, exhaustion), over steps that meet no wall, grant and
    refuse trials, meet a sealed object, meet several walls and exhaust the budget."""
    c, Lc, _ = pack_cylinders(RADII, target_vf=0.45, seed=1)
    f32 = np.float32
    step_max = LONGEST_STEP * RADII.min() / 6
    args = (jnp.asarray(c, f32), jnp.asarray(RADII, f32), f32(Lc), f32(1e-7 * RADII.min()), f32(1e-4 * RADII.min()),
            step_max, periodic_min_gap(c, RADII, Lc))
    w0 = packed_wall_kernel(*args, count=True, budget=budget)
    w1 = packed_wall_kernel(*args, count=True, budget=budget, trials=True)
    pts, dirs, ls, inside, _ = _branch_states(RADII, c, Lc, np.ones(6, bool), step_max, 6000, 21)
    keys = jax.random.split(jax.random.PRNGKey(21), len(pts))
    kod = jnp.asarray(np.where(np.arange(6) == 1, 0.0, 4e-1 / D), jnp.float32)          # object 1 sealed
    run = lambda w: jax.jit(jax.vmap(lambda p, d, l, s, k: w(p, d, l, s, kod, f32(1e5), k)))(
        jnp.asarray(pts), jnp.asarray(dirs), jnp.asarray(ls), jnp.asarray(inside), keys)
    h0, h1 = run(w0), run(w1)
    assert h0.trials is None
    _assert_tree_equal(h0._replace(trials=None), h1._replace(trials=None), "packed kernel")
    v, g, o = (np.asarray(x) for x in (h1.trials.valid, h1.trials.granted, h1.trials.obj))
    n_hits = np.asarray(h1.n_hits)
    assert (n_hits == 0).sum() > 0 and (v & g).sum() > 0 and (v & ~g).sum() > 0 and (v & (o == 1)).sum() > 0
    if budget == 1:
        assert np.asarray(h1.exhausted).sum() > 0
    else:
        assert (n_hits >= 3).sum() > 0


@pytest.mark.parametrize("budget", [None, 1], ids=["derived_budget", "budget_1"])
def test_one_step_of_the_mixed_kernel_is_the_same_with_and_without_the_record(budget):
    """The concentric wall kernel of a cell of plain and myelinated cylinders with and without its trial log: every
    output bit for bit, over steps that meet no wall, a plain wall (granted, refused, sealed) and a sheath, and that
    exhaust the budget."""
    c, Lc, _ = pack_cylinders(RADII, target_vf=0.45, seed=1)
    radii = RADII / 0.999
    plain = np.array([True, False, True, False, True, False])
    N, f32 = len(radii), np.float32
    inner = np.where(plain, radii, 0.7 * radii)
    step_max = LONGEST_STEP * inner.min() / 6
    Dj = jnp.full((N,), f32(D))
    kap = np.where(plain, np.where(np.arange(N) == 2, 0.0, 4e-1), 4e-3).astype(f32)    # plain cylinder 2 sealed
    args = (jnp.asarray(c, f32), jnp.asarray(inner, f32), jnp.asarray(radii, f32), f32(Lc), Dj, Dj, Dj,
            jnp.full(N, f32(4e-3)), jnp.asarray(kap), jnp.full((N, 4), f32(1e5)), f32(1e-7 * inner.min()),
            f32(1e-4 * inner.min()), step_max, periodic_min_gap(c, radii, Lc))
    w0 = concentric_wall_kernel(*args, count=True, budget=budget)
    w1 = concentric_wall_kernel(*args, count=True, budget=budget, trials=True)
    pts, dirs, ls, inside, own = _branch_states(radii, c, Lc, plain, step_max, 6000, 23)
    keys = jax.random.split(jax.random.PRNGKey(23), len(pts))
    words = jax.vmap(lambda k: jax.random.bits(k, (w0.bounce_loop.budget,), dtype=jnp.uint32))(keys)
    us = jax.vmap(lambda k: jax.random.uniform(k, dtype=jnp.float32))(keys)
    run = lambda w: jax.jit(jax.vmap(lambda p, d, l, pl, k, u, ws: w(p, d, l, pl, k, u, ws)))(
        jnp.asarray(pts), jnp.asarray(dirs), jnp.asarray(ls), jnp.asarray(inside.astype(np.int32)),
        jnp.asarray(own.astype(np.int32)), us, words)
    h0, h1 = run(w0), run(w1)
    assert h0.trials is None
    _assert_tree_equal(h0._replace(trials=None), h1._replace(trials=None), "concentric kernel")
    v, g, o = (np.asarray(x) for x in (h1.trials.valid, h1.trials.granted, h1.trials.obj))
    n_hits = np.asarray(h1.n_hits)
    assert (n_hits == 0).sum() > 0 and (v & g).sum() > 0 and (v & ~g).sum() > 0 and (v & (o == 2)).sum() > 0
    assert (n_hits > v.sum(1)).sum() > 0                                                  # a sheath met too
    if budget == 1:
        assert np.asarray(h1.exhausted).sum() > 0


def _event_counts(w, pools):
    """Per walker and band of the walk's record: the C1 events that change the label between the two pools ``pools``
    (a plain wall's two sides), read off the event stream alone. An event at step end ``j dt`` is in the band whose
    ``(edge_k, edge_k+1]`` holds it, the edges here on step ends (block multiples)."""
    r = w.trial_record
    q = np.asarray(r.band_edges) / w.dt_sim
    steps = np.rint(q).astype(np.int64)
    assert np.allclose(q, steps, rtol=0.0, atol=1e-6)
    ev = w.comp_events
    out = np.zeros((w.n_walkers, len(r.band_edges) - 1), np.int64)
    off = np.asarray(ev.offsets)
    for i in range(w.n_walkers):
        prev = int(w.compartment[i])
        for k in range(off[i], off[i + 1]):
            lab = int(ev.label[k])
            if {prev, lab} == set(pools):
                j = int(np.rint(float(ev.t[k]) / w.dt_sim))
                out[i, int(np.sum(steps[1:-1] < j))] += 1
            prev = lab
    return out


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("cell", ["plain", "mixed"])
def test_the_record_is_exactly_its_own_walks_count(cell, dtype, request):
    """The record against what its own walk reports, on every host: the encounters are the walk's (an all-plain cell:
    ``sum n == work["n_hits"]``), a myelinated cylinder's cells are empty, ``N <= n`` per cell, and per walker and band
    the granted trials less the walk's own crossings of a plain wall (its C1 events between lumen and extra) are even
    and not negative -- a step changes the label exactly when it grants an odd number of crossings -- and on the
    all-plain cell the walk's crossing steps (``work["n_crossings"]``) are its events, one each. In two storage windows, each window's record holds
    the same for its own events, and in either accumulation dtype."""
    if dtype == "float64":
        request.getfixturevalue("x64")
    spec = TrialRecordSpec(modes=2, band_edges=(0.0, 1e-3, 2e-3, 3e-3, T))
    geo = _dense() if cell == "plain" else _mixed()
    w = _walk(geo, record_trials=spec, window_T=2e-3, K=4, accumulation_dtype=dtype)
    r = w.trial_record
    assert r.S.dtype == np.dtype(dtype) and w.n_windows == 2
    assert np.all(r.N <= r.n)
    d = r.to_dense()
    ev = _event_counts(w, (0, 1))
    granted = d.N.sum(axis=1)                                                        # (walkers, bands)
    assert np.all(granted >= ev) and np.all((granted - ev) % 2 == 0)
    if cell == "plain":
        assert int(np.asarray(w.comp_events.t).size) == w.work["n_crossings"]   # a crossing step changes the label
        assert int(r.n.sum()) == w.work["n_hits"]
    else:                          # a mixed cell's step counts as crossing when it grants any, an even number included
        assert int(np.asarray(w.comp_events.t).size) <= w.work["n_crossings"]
        assert int(d.n[:, [0, 2]].sum()) == 0 and int(d.n[:, [1, 3]].sum()) > 0
        assert int(r.n.sum()) <= w.work["n_hits"]
    assert int(ev.sum()) > 0
    for i in range(2):
        wi = w.window(i)
        evi = _event_counts(wi, (0, 1))
        np.testing.assert_array_equal(evi, ev[:, 2 * i:2 * i + 2])
        np.testing.assert_array_equal(wi.trial_record.to_dense().N, d.N[:, :, 2 * i:2 * i + 2])


def test_recorded_bits_the_record_changes_no_bit_of_a_plain_cells_walk(recorded_bits, plain_walks):
    """On the recording build and host the two programs walk the same bits: every array, the work, the events; and the
    walk carrying the record of one cell per (object, band) pair (:func:`_dense_record`) is the same bits as both."""
    _assert_same_walk(*plain_walks)
    dense = _walk(_dense(), record_trials=SPEC, _trial_device=_dense_record)
    _assert_same_walk(dense, plain_walks[1])
    _assert_same_walk(dense, plain_walks[0])


def test_recorded_bits_the_record_changes_no_bit_of_a_mixed_cells_walk(recorded_bits, mixed_walks):
    _assert_same_walk(*mixed_walks)


# ── 5. the overshoot's bound ────────────────────────────────────────────────────────────────────────────────────────

def test_every_overshoot_is_within_its_objects_bound():
    """A dense cell walked to at least 1e5 trials: every trial's overshoot is within its object's ``u_max`` (the walk
    counts the trials beyond it and refuses a walk with any), the bound stated per object as ``2 LONGEST_STEP h / R``."""
    w = _walk(_dense(), n=320, T_max=5e-3, record_trials=TrialRecordSpec(modes=2, band_edges=(0.0, 5e-3)), K=4)
    r = w.trial_record
    assert int(r.n.sum()) >= 100_000
    np.testing.assert_allclose(r.u_max, u_max_of(RADII, np.full(6, D), w.dt_sim), rtol=0, atol=0)
    d = r.to_dense()
    mean_u = d.S[..., 0].sum(axis=(0, 2)) / np.maximum(d.n.sum(axis=(0, 2)), 1)
    assert np.all(mean_u < r.u_max)


def test_a_step_longer_than_the_step_laws_is_refused(monkeypatch):
    from dmipy_sim.engine import core
    longer = core.pair_step
    monkeypatch.setattr(core, "pair_step", lambda key: jnp.float32(3.0) * longer(key))
    with pytest.raises(ValueError, match=r"^trial_record: \d+ trials exceed u_max = \["):
        _walk(_dense(), n=8, T_max=1e-3, record_trials=TrialRecordSpec(modes=1, band_edges=(0.0, 1e-3)))


# ── 6. a schedule at the step's midpoint, the band at its end ───────────────────────────────────────────────────────

def test_a_schedules_breakpoint_at_a_band_edge_splits_the_trials():
    """kappa 4e-3 then 0 at 1 ms, a band edge there: the first band's counts are those of the walk of the first 1 ms
    (the same walkers, steps and keys: its own work record), the second band's trials are never granted."""
    from dmipy_sim.geometry.base import PermeabilitySchedule
    sched = PermeabilitySchedule([1e-3], [4e-3, 0.0])
    spec = TrialRecordSpec(modes=1, band_edges=(0.0, 1e-3, 2e-3))
    full = _walk(_dense(permeability=sched), T_max=2e-3, sub_steps=None, record_trials=spec)
    head = _dense(permeability=sched)
    head.count_walls = True
    first = _walk(head, T_max=1e-3, sub_steps=full.sub_steps, K=4)
    r = full.trial_record.to_dense()
    assert full.dt_sim == first.dt_sim
    assert int(r.n[:, :, 0].sum()) == first.work["n_hits"] and int(r.N[:, :, 0].sum()) == first.work["n_crossings"]
    assert int(r.N[:, :, 0].sum()) > 0
    assert int(r.n[:, :, 1].sum()) > 0 and int(r.N[:, :, 1].sum()) == 0


def test_a_trial_goes_to_the_band_of_its_steps_end():
    """Edges at step ends 0, 4, 6.5 (inside step 6) and 10, bands ``(a, b]``: the step ending at the edge 4 is in the
    earlier band, the step straddling 6.5 (ending at 7) in the later band, and the walk's last step, ending at
    ``T_max``, in the final band."""
    from dmipy_sim.trial_record import edge_steps
    spec = TrialRecordSpec(modes=1, band_edges=(0.0, 4.0, 6.5, 10.0))
    steps = edge_steps(spec, 10.0, 1.0, 10, 10)
    np.testing.assert_array_equal(steps, [0, 4, 6, 10])
    init, add = device_record(steps, [1.0], 1, jnp.float32)
    one = TrialLog(jnp.array([True]), jnp.array([0], jnp.int32), jnp.array([0.5], jnp.float32), jnp.array([False]))
    bands = {}
    for s in range(10):
        n, _, _ = _one_walker_dense(add(init(1), one, jnp.int32(s)), [1.0], (0.0, 4.0, 6.5, 10.0), 1)
        bands[s] = int(np.flatnonzero(n[0])[0])
    assert bands == {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2}


# ── 8. the stored count ─────────────────────────────────────────────────────────────────────────────────────────────

def test_the_record_stores_the_trial_count_beside_the_granted_count(plain_walks):
    r = plain_walks[1].trial_record
    assert r.n.dtype == np.int32 and r.N.dtype == np.int32 and np.all(r.N <= r.n)
    width = int(r.rows.max())
    assert r.rows.shape == (32,) and 0 < width <= 6 * 3
    assert r.n.shape == r.N.shape == r.obj.shape == r.band.shape == (32, width) and r.S.shape == (32, width, 3)
    d = r.to_dense()
    assert d.n.shape == d.N.shape == (32, 6, 3) and d.S.shape == (32, 6, 3, 3)
    assert r.modes == 3 and r.band_edges == SPEC.band_edges
    np.testing.assert_array_equal(r.object_radii, RADII)


# ── 9. a common start, a stated field read ──────────────────────────────────────────────────────────────────────────

def test_a_recorded_walk_refuses_the_burnin():
    with pytest.raises(ValueError, match=re.escape("trial_record: the start of a recorded walk is common to every "
                                                   "reweighted read: use equilibrate_binding='fast' or 'off'")):
        _walk(_dense(), kappa_MT=1e-5, dwell_time=1e-3, record_trials=SPEC)


@pytest.mark.parametrize("missing", ["K_c3", "field_read_interval"])
def test_a_recorded_walk_refuses_a_field_read_its_pilot_would_choose(missing):
    from dmipy_sim.fields.hollow_cylinder import HollowCylinderField
    geo = _mixed()
    kw = dict(K_c3=4, field_read_interval=1.25e-5)
    kw.pop(missing)
    with pytest.raises(ValueError, match=re.escape(f"trial_record: {missing} is stated, not chosen by a pilot of "
                                                   f"this walk's own walkers")):
        _walk(geo, field_basis=HollowCylinderField.of(geo), field_pilot="jax", record_trials=SPEC, **kw)


def test_a_recorded_walk_walks_the_stated_forms():
    """'off' with binding, and a field channel at a stated band and read interval."""
    from dmipy_sim.fields.hollow_cylinder import HollowCylinderField
    w = _walk(_dense(), n=8, T_max=1e-3, kappa_MT=1e-5, dwell_time=1e-3, equilibrate_binding="off",
              record_trials=TrialRecordSpec(modes=1, band_edges=(0.0, 1e-3)))
    assert int(w.trial_record.n.sum()) == w.work["n_hits"]
    geo = _mixed()
    w = _walk(geo, n=8, T_max=1e-3, field_basis=HollowCylinderField.of(geo), K_c3=4, field_read_interval=1.25e-5,
              record_trials=TrialRecordSpec(modes=1, band_edges=(0.0, 1e-3)))
    assert w.field_coeffs is not None and int(w.trial_record.n.sum()) > 0


# ── 10. a stable walker index ───────────────────────────────────────────────────────────────────────────────────────

def _assert_same_record(a, b):
    for f in ("obj", "band", "n", "N", "S", "rows", "u_max", "object_radii"):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f), err_msg=f)
    assert a.band_edges == b.band_edges and a.modes == b.modes


def test_a_walkers_row_is_its_index_in_the_seed(tmp_path):
    """The walk of walkers [0, 32) and the join of the walks of [0, 16) and [16, 32) (each its own batch size) record
    the same; a walk file holds the record."""
    from dmipy_sim.coefficient_walk import CoefficientWalk, join_walks
    spec = TrialRecordSpec(modes=SPEC.modes, band_edges=SPEC.band_edges, slots=18)     # stated: the pieces' records agree
    whole = _walk(_dense(), n=32, record_trials=spec, events_capacity=64)
    a = _walk_geometry(16, D, _dense(), T, BLOCK, seed=3, tiers="all", require_gpu=False, K=8, record_trials=spec,
                       events_capacity=64, walker_batch_size=8)
    b = _walk_geometry(16, D, _dense(), T, BLOCK, seed=3, tiers="all", require_gpu=False, K=8, record_trials=spec,
                       events_capacity=64, first_walker=16)
    _assert_same_record(join_walks([a, b]).trial_record, whole.trial_record)
    _assert_same_record(whole.take(np.arange(16)).trial_record, a.trial_record)
    whole.save(tmp_path / "w.safetensors")
    back = CoefficientWalk.load(tmp_path / "w.safetensors")
    _assert_same_record(back.trial_record, whole.trial_record)
    assert back.nbytes == whole.nbytes and whole.nbytes > whole.trial_record.nbytes > 0


def test_a_windows_record_is_its_own_bands():
    """A walk of two 2 ms windows: ``window(i)`` holds the bands of window ``i``, and window 0's record is the record of
    the walk of its 2 ms alone (the same walkers, steps and positions): a step ending on the window boundary is in the
    earlier band, window 0's. Both walks carry the same 24 slots, so that their programs carry records of one shape."""
    spec = TrialRecordSpec(modes=2, band_edges=(0.0, 1e-3, 2e-3, 3e-3, T), slots=24)
    w = _walk(_dense(), window_T=2e-3, record_trials=spec, K=4)
    assert w.n_windows == 2
    w0, w1 = w.window(0).trial_record, w.window(1).trial_record
    assert w0.band_edges == (0.0, 1e-3, 2e-3) and w1.band_edges == (2e-3, 3e-3, T)
    np.testing.assert_array_equal(np.concatenate([w0.to_dense().n, w1.to_dense().n], axis=2), w.trial_record.to_dense().n)
    alone = _walk(_dense(), T_max=2e-3, sub_steps=w.sub_steps, K=4,
                  record_trials=TrialRecordSpec(modes=2, band_edges=(0.0, 1e-3, 2e-3), slots=24))
    np.testing.assert_array_equal(np.asarray(alone.position_coeffs), np.asarray(w.window(0).position_coeffs))
    _assert_same_record(alone.trial_record, w0)


def test_an_edge_half_a_step_off_a_window_boundary_is_refused():
    """The window-boundary check reads the edges as given: an edge half a physics step before the 2 ms boundary does
    not contain it, although it would be placed on the boundary's step end."""
    from dmipy_sim.engine.core import walk_cost
    dt = walk_cost(4, D, _dense(), T, BLOCK, tiers="all", K=8, window_T=2e-3).dt_sim
    spec = TrialRecordSpec(modes=1, band_edges=(0.0, 2e-3 - 0.5 * dt, T))
    with pytest.raises(ValueError, match="^" + re.escape("trial_record: band_edges must contain the window boundary 0.002") + "$"):
        _walk(_dense(), n=4, window_T=2e-3, record_trials=spec)


def test_a_trial_is_attributed_through_the_candidate_index():
    """A cell of 40 cylinders, more than the kernel's candidates per step: every logged trial is at an object whose
    wall lies within the step's reach of its start (an attribution by candidate slot instead of object would name a
    distant object), and objects beyond the candidate count are among those met."""
    rng = np.random.default_rng(3)
    radii = rng.uniform(0.8, 1.3, 40) * 1e-6
    c, Lc, _ = pack_cylinders(radii, target_vf=0.5, seed=2)
    f32 = np.float32
    step_max = LONGEST_STEP * radii.min() / 6
    w = packed_wall_kernel(jnp.asarray(c, f32), jnp.asarray(radii, f32), f32(Lc), f32(1e-7 * radii.min()),
                           f32(1e-4 * radii.min()), step_max, periodic_min_gap(c, radii, Lc), count=True, trials=True)
    assert w.n_cand < len(radii)
    pts, dirs, ls, inside, _ = _kernel_steps(radii, c, Lc, 8000, 13, step_max)
    keys = jax.random.split(jax.random.PRNGKey(13), len(pts))
    kod = jnp.full((len(radii),), 4e-3 / D, jnp.float32)
    h = jax.jit(jax.vmap(lambda p, d, l, s, k: w(p, d, l, s, kod, f32(0.0), k)))(
        jnp.asarray(pts), jnp.asarray(dirs), jnp.asarray(ls), jnp.asarray(inside), keys)
    valid = np.asarray(h.trials.valid); obj = np.asarray(h.trials.obj)
    np.testing.assert_array_equal(valid.sum(1), np.asarray(h.n_hits))
    s_idx, i_idx = np.nonzero(valid)
    o = obj[s_idx, i_idx]
    q = pts[s_idx].astype(np.float64) - c[o]
    q -= Lc * np.round(q / Lc)
    wall_distance = np.abs(np.linalg.norm(q, axis=1) - radii[o])
    assert len(o) > 1000
    assert np.all(wall_distance <= ls[s_idx] * (1 + 1e-5) + 1e-4 * radii.min())
    assert (o >= w.n_cand).sum() > 0


def test_a_mixed_cells_trial_is_attributed_through_the_candidate_index():
    """A mixed cell of 40 cylinders, every other one plain, more than the kernel's candidates per step: every logged
    trial is at a plain cylinder whose wall lies within the step's reach of its start, and cylinders beyond the
    candidate count are among those met."""
    rng = np.random.default_rng(4)
    radii = rng.uniform(0.8, 1.3, 40) * 1e-6
    c, Lc, _ = pack_cylinders(radii, target_vf=0.45, seed=5)
    N = len(radii)
    plain = (np.arange(N) % 2) == 0
    f32 = np.float32
    step_max = LONGEST_STEP * (0.7 * radii).min() / 6
    inner = np.where(plain, radii, 0.7 * radii)
    Dj = jnp.full((N,), f32(D))
    kap = np.where(plain, 4e-3, 0.0).astype(f32)
    w = concentric_wall_kernel(jnp.asarray(c, f32), jnp.asarray(inner, f32), jnp.asarray(radii, f32), f32(Lc), Dj, Dj, Dj,
                               jnp.zeros(N, f32), jnp.asarray(kap), jnp.zeros((N, 4), f32), f32(1e-7 * inner.min()),
                               f32(1e-4 * inner.min()), step_max, periodic_min_gap(c, radii, Lc), count=True, trials=True)
    assert w.n_cand < N
    pts, dirs, ls, inside, own = _kernel_steps(radii, c, Lc, 20000, 17, step_max)
    keep = ~inside | plain[own]                                            # outside the sheaths, or in a plain lumen
    pts, dirs, ls, inside, own = pts[keep], dirs[keep], ls[keep], inside[keep], own[keep]
    keys = jax.random.split(jax.random.PRNGKey(17), len(pts))
    words = jax.vmap(lambda k: jax.random.bits(k, (w.bounce_loop.budget,), dtype=jnp.uint32))(keys)
    h = jax.jit(jax.vmap(lambda p, d, l, pl, k, u: w(p, d, l, pl, k, f32(0.5), u)))(
        jnp.asarray(pts), jnp.asarray(dirs), jnp.asarray(ls), jnp.asarray(inside.astype(np.int32)),
        jnp.asarray(own.astype(np.int32)), words)
    valid = np.asarray(h.trials.valid); obj = np.asarray(h.trials.obj)
    s_idx, i_idx = np.nonzero(valid)
    o = obj[s_idx, i_idx]
    assert len(o) > 1000 and plain[o].all()
    q = pts[s_idx].astype(np.float64) - c[o]
    q -= Lc * np.round(q / Lc)
    wall_distance = np.abs(np.linalg.norm(q, axis=1) - radii[o])
    assert np.all(wall_distance <= ls[s_idx] * (1 + 1e-5) + 1e-4 * radii.min())
    assert (o >= w.n_cand).sum() > 0


# ── 11. no rounding ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_the_sums_are_the_kernels_own_trials_recomputed(x64):
    """2000 steps of the dense cell through the plain kernel, logged: the record built in float64 from each step's log
    equals a numpy recomputation from the same per-trial log (``u`` as the kernel holds it, ``T_j`` by
    ``numpy.polynomial.chebyshev``) at relative 1e-12, and its integers are the encounters and the granted ones."""
    c, Lc, _ = pack_cylinders(RADII, target_vf=0.45, seed=1)
    f32 = np.float32
    step_max = LONGEST_STEP * RADII.min() / 6
    w = packed_wall_kernel(jnp.asarray(c, f32), jnp.asarray(RADII, f32), f32(Lc), f32(1e-7 * RADII.min()),
                           f32(1e-4 * RADII.min()), step_max, periodic_min_gap(c, RADII, Lc), count=True, trials=True)
    pts, dirs, ls, inside, _ = _kernel_steps(RADII, c, Lc, 2000, 11, step_max)
    keys = jax.random.split(jax.random.PRNGKey(11), len(pts))
    kod = jnp.asarray(KAPPA * 1e2 / D, jnp.float32)                  # a large permeability: many granted
    h = jax.jit(jax.vmap(lambda p, d, l, s, k: w(p, d, l, s, kod, f32(0.0), k)))(
        jnp.asarray(pts), jnp.asarray(dirs), jnp.asarray(ls), jnp.asarray(inside), keys)
    u_max = 2.0 * LONGEST_STEP * (RADII.min() / 6) / RADII
    steps = np.array([0, 1000, 2000])
    init, add = device_record(steps, u_max, 5, jnp.float64)
    rec = init(12)
    log = jax.tree_util.tree_map(np.asarray, h.trials)
    for s in range(len(pts)):
        rec = add(rec, jax.tree_util.tree_map(lambda a: jnp.asarray(a[s]), h.trials), jnp.int32(s))
    exceed = np.asarray(rec[5])
    assert int(rec[4]) <= 12                                          # every pair in a slot
    n, N, S = _one_walker_dense(rec, u_max, (0.0, 1000.0, 2000.0), 5)
    assert int(exceed) == 0
    want_n = np.zeros((6, 2), np.int64); want_N = np.zeros((6, 2), np.int64); want_S = np.zeros((6, 2, 5))
    for s in range(len(pts)):
        band = 0 if s + 1 < 1000 else 1
        for i in np.flatnonzero(log.valid[s]):
            o = int(log.obj[s, i]); u = float(log.u[s, i])
            want_n[o, band] += 1; want_N[o, band] += int(log.granted[s, i])
            x = 2.0 * u / u_max[o] - 1.0
            want_S[o, band] += u * np.array([np.polynomial.chebyshev.chebval(x, np.eye(5)[j]) for j in range(5)])
    np.testing.assert_array_equal(n, want_n)
    np.testing.assert_array_equal(N, want_N)
    assert n.sum() == np.asarray(h.n_hits).sum() and N.sum() > 0
    np.testing.assert_allclose(S, want_S, rtol=1e-12, atol=0.0)


# ── refusals, one mutation each ─────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("make, message", [
    (lambda: TrialRecordSpec(modes=0, band_edges=(0.0, T)), "trial_record: modes must be a whole number of at least 1, got 0"),
    (lambda: TrialRecordSpec(modes=1, band_edges=(0.0, 2e-3, 2e-3, T)),
     "trial_record: band_edges must be strictly increasing, got (0.0, 0.002, 0.002, 0.004)"),
    (lambda: TrialRecordSpec(modes=1, band_edges=tuple(np.linspace(0.0, T, 66))), "trial_record: 65 bands exceed 64"),
    (lambda: TrialRecordSpec(modes=1, band_edges=(0.0, T), slots=0),
     "trial_record: slots must be a whole number of at least 1, or None for a pilot, got 0"),
])
def test_a_malformed_record_spec_is_refused(make, message):
    with pytest.raises(ValueError, match="^" + re.escape(message) + "$"):
        make()


@pytest.mark.parametrize("geo, spec, kw, message", [
    ("dense", TrialRecordSpec(modes=1, band_edges=(0.0, 2e-3)), {},
     "trial_record: band_edges must run from 0 to T_max = 0.004, got [0.0, 0.002]"),
    ("dense", TrialRecordSpec(modes=1, band_edges=(1e-4, T)), {},
     "trial_record: band_edges must run from 0 to T_max = 0.004, got [0.0001, 0.004]"),
    ("dense", TrialRecordSpec(modes=1, band_edges=(0.0, 1e-3, T)), dict(window_T=2e-3),
     "trial_record: band_edges must contain the window boundary 0.002"),
    ("dense", TrialRecordSpec(modes=50, band_edges=tuple(np.linspace(0.0, T, 65)), slots=400), {},
     "trial_record: 400 slots of 50 modes are 84808 bytes per walker, above 65536"),
    ("myelinated", SPEC, {}, "trial_record: PackedMyelinatedCylinders has no plain wall whose trials a walk records"),
])
def test_a_record_the_walk_cannot_hold_is_refused(geo, spec, kw, message):
    g = _dense() if geo == "dense" else PackedMyelinatedCylinders(OUTER * 0.7, [0.7] * 4, CENTRES, L_CELL, N_max=4,
                                                                  compartments=COMPS)
    with pytest.raises(ValueError, match="^" + re.escape(message) + "$"):
        _walk(g, n=4, record_trials=spec, **kw)


# ── 12. bounded slots per walker ────────────────────────────────────────────────────────────────────────────────────

def _side_by_side(geo, spec, dtype="float32", **kw):
    """The walk carrying the reference (:func:`_dense_record`) and the slotted record side by side, each step's trial
    log added to both: ``(walk, (n, N, S, exceed) of the reference, (key, n, N, S, count, exceed) of the slots)``,
    walker first."""
    w = _walk(geo, record_trials=spec, accumulation_dtype=dtype, _trial_device=_dense_and_slotted, **kw)
    leaves = w.trial_record
    assert isinstance(leaves, tuple) and len(leaves) == 10
    return w, leaves[:4], leaves[4:]


def _slotted_record(geo, spec, w, slots):
    radii, D_obj = geo.trial_objects(D)
    return TrialRecord.from_slots(*slots[:5], u_max=u_max_of(radii, D_obj, w.dt_sim), band_edges=spec.band_edges,
                                  modes=spec.modes, object_radii=radii)


@pytest.mark.parametrize("cell, bands, dtype", [("plain", "several", "float32"), ("plain", "one", "float64"),
                                               ("mixed", "one", "float32"), ("mixed", "several", "float64")])
def test_the_slotted_record_is_the_dense_record_of_its_walk(cell, bands, dtype, request):
    """The reference and the slotted record of ONE walk (the same trial logs, added in the same order): ``n`` and ``N``
    equal, ``S`` bit for bit, ``exceed`` equal, and each walker's rows are exactly the pairs it met, ``count`` exact.
    All-plain and mixed cells, one band and three, float32 and float64 accumulation; in the mixed cell walkers in a
    sheathed lumen meet no plain wall and hold no row."""
    if dtype == "float64":
        request.getfixturevalue("x64")
    spec = SPEC if bands == "several" else TrialRecordSpec(modes=2, band_edges=(0.0, T))
    geo = _dense() if cell == "plain" else _mixed()
    w, dense, slots = _side_by_side(geo, spec, dtype)
    r = _slotted_record(geo, spec, w, slots)
    d = r.to_dense()
    np.testing.assert_array_equal(d.n, dense[0])
    np.testing.assert_array_equal(d.N, dense[1])
    assert d.S.dtype == dense[2].dtype == np.dtype(dtype)
    # the sums: the same trials added in the same order. The overshoot's own sum (mode 0) is the same bits; a higher
    # mode is within 4 last places of the overshoot's sum S_0 (|T_j| <= 1, so S_0 bounds every |S_j|): the two forms
    # evaluate the terms in separately compiled code of one program, and a term may round differently in its last place
    # (measured on the mixed cell: at most 0.5 of S_0's last place, 2 of S_j's)
    np.testing.assert_array_equal(d.S[..., 0], dense[2][..., 0])
    ulp = np.spacing(np.abs(dense[2][..., :1]).astype(dense[2].dtype)).astype(np.float64)
    assert np.all(np.abs(d.S.astype(np.float64) - dense[2].astype(np.float64)) <= 4 * ulp)
    np.testing.assert_array_equal(slots[5], dense[3])
    met = (dense[0] > 0).sum(axis=(1, 2))
    np.testing.assert_array_equal(r.rows, met)
    assert int(dense[0].sum()) > 0 and np.all(r.n[np.arange(r.obj.shape[1])[None, :] < r.rows[:, None]] > 0)
    if cell == "mixed":
        assert (r.rows == 0).sum() > 0 and int(dense[0][r.rows == 0].sum()) == 0


def test_a_walker_beyond_its_slots_is_counted_and_nothing_is_added_to_another_pair():
    """One slot per walker beside the reference in one walk: the count is exact while it is at most 1 and at least the
    pairs met beyond; the one slot holds the first pair met with exactly its own trials -- the trials of the pairs that
    found no slot are added nowhere."""
    spec = TrialRecordSpec(modes=SPEC.modes, band_edges=SPEC.band_edges, slots=1)
    w, dense, slots = _side_by_side(_dense(), spec)
    key, n, N, S, count, _ = slots
    assert key.shape == (32, 1)
    met = (dense[0] > 0).sum(axis=(1, 2))
    assert (met > 1).sum() > 0
    np.testing.assert_array_equal(count[met <= 1], met[met <= 1])
    assert np.all(count[met > 1] >= met[met > 1])
    first = count > 0
    o, k = key[first, 0] // 3, key[first, 0] % 3
    rows = np.flatnonzero(first)
    np.testing.assert_array_equal(n[first, 0], dense[0][rows, o, k])
    np.testing.assert_array_equal(N[first, 0], dense[1][rows, o, k])
    np.testing.assert_array_equal(S[first, 0], dense[2][rows, o, k])
    assert np.all(n[~first] == 0) and np.all(key[~first] == -1)


def test_a_walk_beyond_its_slots_is_refused_with_the_count_and_a_capacity():
    from dmipy_sim.trial_record import suggested_capacity
    spec = TrialRecordSpec(modes=SPEC.modes, band_edges=SPEC.band_edges, slots=1)
    with pytest.raises(ValueError, match=r"^trial_record: a walker met at least (\d+) \(object, band\) pairs where the "
                                         r"record holds 1 slots: the walk is refused rather than a trial dropped\. Walk "
                                         r"again with TrialRecordSpec\(slots=(\d+)\) \(the record's slots: "
                                         r"\{'rule': 'given', 'pairs': 18, 'capacity': 1, 'bytes_per_walker': 32\}\)$") as e:
        _walk(_dense(), record_trials=spec)
    top, suggested = (int(x) for x in re.search(r"at least (\d+) .*slots=(\d+)\)", str(e.value)).groups())
    assert top > 1 and suggested == suggested_capacity(top) >= top


def test_the_pilot_states_the_slots_and_they_hold_the_walk(monkeypatch):
    """A pilot of 8 of the 32 walkers that first carry 2 slots: it walks again until its largest count fits, and the
    capacity is the stated rule on that count (the events buffer's margin), at least the largest count of the whole walk,
    with the rule's inputs in ``work["trial_slots"]``; a stated capacity is recorded as given."""
    from dmipy_sim import trial_record
    from dmipy_sim.engine import accumulate
    monkeypatch.setattr(accumulate, "PILOT_WALKERS", 8)
    monkeypatch.setattr(trial_record, "PILOT_SLOTS", 2)
    w = _walk(_dense(), record_trials=SPEC)
    rec = w.work["trial_slots"]
    assert trial_record.SLOTS_MARGIN == accumulate.EVENTS_MARGIN
    assert rec["rule"] == "pilot" and rec["pilot_walkers"] == 8 and rec["pairs"] == 18
    assert rec["pilot_slots"] > 2 and rec["pilot_max_pairs"] <= rec["pilot_slots"]
    assert rec["margin"] == dict(factor=1.5, plus=8)
    assert rec["capacity"] == trial_record.pilot_capacity(rec["pilot_max_pairs"], 18)
    assert rec["capacity"] >= int(w.trial_record.rows.max())
    assert rec["bytes_per_walker"] == trial_record.bytes_per_walker(rec["capacity"], 3, 4)
    given = _walk(_dense(), n=8, record_trials=TrialRecordSpec(modes=1, band_edges=(0.0, T), slots=6))
    assert given.work["trial_slots"] == dict(rule="given", pairs=6, capacity=6, bytes_per_walker=6 * 16 + 8)


def test_the_rows_survive_take_bands_concatenate_and_the_walk_file(plain_walks, tmp_path):
    """``take``, ``bands`` and ``concatenate`` are the dense record's selections, and the walk file holds the rows."""
    r = plain_walks[1].trial_record
    d = r.to_dense()
    idx = np.array([5, 0, 31, 7])
    for name in ("n", "N", "S"):
        np.testing.assert_array_equal(getattr(r.take(idx).to_dense(), name), getattr(d, name)[idx])
        np.testing.assert_array_equal(getattr(r.bands(1e-3, T).to_dense(), name), getattr(d, name)[:, :, 1:])
        np.testing.assert_array_equal(getattr(r.bands(0.0, 1e-3).to_dense(), name), getattr(d, name)[:, :, :1])
    b = r.bands(1e-3, T)
    assert b.band_edges == SPEC.band_edges[1:] and np.all(b.band[b.obj >= 0] < 2)
    joined = TrialRecord.concatenate([r.take(slice(0, 11)), r.take(slice(11, 32))])
    for f in ("obj", "band", "n", "N", "S", "rows"):
        np.testing.assert_array_equal(getattr(joined, f), getattr(r, f), err_msg=f)
    back = TrialRecord.from_arrays(r.to_arrays())
    for f in ("obj", "band", "n", "N", "S", "rows", "u_max", "object_radii"):
        np.testing.assert_array_equal(getattr(back, f), getattr(r, f), err_msg=f)
    n_tot, N_tot = r.totals()
    np.testing.assert_array_equal(n_tot, d.n.sum(axis=(1, 2)))
    np.testing.assert_array_equal(N_tot, d.N.sum(axis=(1, 2)))
    one = r.walker(7)
    assert len(one.obj) == r.rows[7]
    np.testing.assert_array_equal(one.n, d.n[7][one.obj, one.band])


def test_a_walk_file_of_the_dense_tensors_is_refused_by_name(plain_walks, tmp_path):
    import json
    from safetensors import safe_open
    from safetensors.numpy import save_file
    from dmipy_sim.coefficient_walk import CoefficientWalk
    w = plain_walks[1]
    path = tmp_path / "w.safetensors"
    w.save(path)
    with safe_open(str(path), framework="numpy") as f:
        meta = f.metadata()
        arrays = {k: f.get_tensor(k) for k in f.keys()}
    d = w.trial_record.to_dense()
    arrays = {k: v for k, v in arrays.items() if not k.startswith("trial_row")}
    arrays.update(trial_n=d.n, trial_N=d.N, trial_S=d.S)
    save_file(arrays, str(tmp_path / "dense.safetensors"), metadata=meta)
    with pytest.raises(ValueError, match="^" + re.escape(
            "trial_record: the walk file holds the record as one cell per (object, band) pair (trial_n, trial_N, "
            "trial_S); this revision reads the record's rows (trial_row_obj, trial_row_band, trial_row_n, trial_row_N, "
            "trial_row_S, trial_rows, trial_u_max, trial_band_edges, trial_object_radii): re-walk it") + "$"):
        CoefficientWalk.load(tmp_path / "dense.safetensors")
    assert json.loads(meta["walk"])["work"]["trial_slots"]["rule"] == "pilot"


def test_the_dense_form_is_refused_above_its_size():
    """``to_dense`` of 10 000 objects in 64 bands (24 B a cell) is refused by name; the rows hold it."""
    from dmipy_sim.trial_record import MAX_DENSE_BYTES_PER_WALKER, dense_bytes_per_walker
    n_obj = 10_000
    r = TrialRecord(obj=np.array([[9_999]]), band=np.array([[63]]), n=np.array([[2]]), N=np.array([[1]]),
                    S=np.ones((1, 1, 4), np.float32), rows=np.array([1]), u_max=np.ones(n_obj),
                    band_edges=tuple(np.linspace(0.0, T, 65)), modes=4, object_radii=np.ones(n_obj))
    b = dense_bytes_per_walker(n_obj, 64, 4, 4)
    assert b == 15_360_000 > MAX_DENSE_BYTES_PER_WALKER
    with pytest.raises(ValueError, match="^" + re.escape(
            f"trial_record: the dense form of 10000 objects and 64 bands is {b} bytes per walker, above 65536") + "$"):
        r.to_dense()
    assert r.walker(0).obj.tolist() == [9_999] and r.totals()[0].tolist() == [2]


def test_a_cell_of_ten_thousand_objects_is_recorded_in_bounded_slots():
    """10 000 parallel cylinders, 3 000 of them plain (g-ratio 1, the class's plain cylinder), at a volume fraction of
    0.6 (:func:`~dmipy_sim.geometry.densify.densify_packing`, gaps at least 0.03 of the mean radius, Gamma radii of CV
    0.25, seed 0): 64 walkers over 312 steps record in the pilot's slots, a few hundred bytes per walker, where one
    cell per (object, band) pair would hold 960 000 bytes per walker, above the dense form's limit."""
    from dmipy_sim.geometry.densify import densify_packing
    from dmipy_sim.trial_record import MAX_BYTES_PER_WALKER, MAX_DENSE_BYTES_PER_WALKER, dense_bytes_per_walker
    n, cv = 10_000, 0.25
    rng = np.random.default_rng(0)
    radii = rng.gamma(1 / cv ** 2, 1e-6 * cv ** 2, n)
    (centres, L), = densify_packing(radii, (0.6,), 2, seed=0, min_gap=0.03 * float(radii.mean()))
    g = np.full(n, 0.7)
    g[rng.permutation(n)[:3000]] = 1.0
    geo = PackedMyelinatedCylinders(radii * g, g, centres, L, N_max=n, compartments=COMPS, kappa_inner=1e-5,
                                    kappa_outer=np.where(g == 1.0, 4e-3, 0.0))
    T_big = 6.4e-6                                         # a few hundred steps at this cell's step
    spec = TrialRecordSpec(modes=4, band_edges=tuple(np.linspace(0.0, T_big, 5)))
    w = _walk_geometry(64, D, geo, T_big, T_big / 4, seed=0, tiers="all", require_gpu=False, K=4, record_trials=spec)
    r, rec = w.trial_record, w.work["trial_slots"]
    assert 200 <= w.work["walker_steps"] // 64 <= 400
    assert r.n_objects == n and rec["pairs"] == 4 * n and rec["rule"] == "pilot"
    assert int(r.rows.max()) <= rec["capacity"] and rec["bytes_per_walker"] <= MAX_BYTES_PER_WALKER
    assert int(r.n.sum()) > 0 and np.all(r.obj[r.obj >= 0] < n) and np.all(g[r.obj[r.obj >= 0]] == 1.0)
    assert dense_bytes_per_walker(n, 4, 4, 4) == 960_000 > MAX_DENSE_BYTES_PER_WALKER
    with pytest.raises(ValueError, match=r"^trial_record: the dense form of 10000 objects and 4 bands is 960000 bytes"):
        r.to_dense()
