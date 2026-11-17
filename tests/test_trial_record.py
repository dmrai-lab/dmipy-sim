"""The record of a walk's membrane trials at its plain walls (:mod:`dmipy_sim.trial_record`).

One test per requirement, each exact: integer counts compared as integers, sums at relative 1e-12 in float64 (the
tests that compare sums accumulate in float64, under the ``x64`` fixture). The requirements:

1. a trial is accounted in the cell of its object and the band of its step, with its overshoot and its outcome;
2. only the trials of steps the walk keeps are recorded (a walker held bound records none);
3. every encounter of a plain wall is a trial, later encounters of a step and sealed objects included, and the counts
   are the walk's work counts on an all-plain cell;
4. the record changes nothing: the walk with it is the walk without it, bit for bit, at the same counting;
5. a trial's overshoot is within its object's a priori bound, and a walk that exceeds it is refused;
6. a scheduled permeability is read at the step's midpoint and the band is the step's end;
8. the trial count is stored beside the granted count;
9. a recorded walk's start and field read are stated, not chosen by its own walkers;
10. a walker's row is its index in the seed, through joins, windows and the walk file;
11. nothing is rounded: the sums are a recomputation of the kernel's own per-trial log.

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
from dmipy_sim.trial_record import TrialRecordSpec, device_record, u_max_of

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
    wa = {k: v for k, v in a.work.items() if k != "block"}
    wb = {k: v for k, v in b.work.items() if k != "block"}
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
    init, add = device_record(3, steps, u_max, 4, jnp.float64)
    rec = init()
    # step 1 (ends at 2: band 0): three entries of the loop, two trials at object 0, one iteration without a trial
    log1 = TrialLog(jnp.array([True, True, False, False]), jnp.array([0, 0, 2, 0], jnp.int32),
                    jnp.array([0.1, 0.3, 0.2, 0.0], jnp.float32), jnp.array([True, False, True, False]))
    # step 6 (ends at 7: band 1): four trials at object 2
    log2 = TrialLog(jnp.array([True, True, True, True]), jnp.array([2, 2, 2, 2], jnp.int32),
                    jnp.array([0.05, 0.35, 0.15, 0.25], jnp.float32), jnp.array([False, True, True, False]))
    rec = add(rec, log1, jnp.int32(1))
    rec = add(rec, log2, jnp.int32(6))
    n, N, S, exceed = (np.asarray(x) for x in rec)
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
    assert int(r.n[:, 1].sum()) > 0 and int(r.N[:, 1].sum()) == 0


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

def test_the_record_changes_no_array_of_a_plain_cells_walk(plain_walks):
    _assert_same_walk(*plain_walks)
    assert plain_walks[0].trial_record is None


def _diagnose(a, b):
    """TEMPORARY diagnostic: per array the largest absolute / relative difference, the walkers whose displacement
    differs by more than 1e-6 of the cell, the discrete outcomes, and the host."""
    import platform, subprocess
    out = []
    for name in ARRAYS:
        x, y = getattr(a, name), getattr(b, name)
        if x is None:
            continue
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        d = np.abs(x - y)
        rel = d / np.maximum(np.abs(y), 1e-300)
        out.append(f"{name}: max_abs {d.max():.3e} max_rel {rel.max():.3e} n_diff {int((d > 0).sum())}/{d.size}")
    disp = np.abs(np.asarray(a.position_coeffs, np.float64)[:, :, 1, :] - np.asarray(b.position_coeffs, np.float64)[:, :, 1, :])
    out.append(f"walkers |d disp| > 1e-6 L: {int((disp.max(axis=(1, 2)) > 1e-6 * L_CELL).sum())}, max |d disp| / L {disp.max() / L_CELL:.3e}")
    wa = {k: v for k, v in a.work.items() if k in ('n_hits', 'n_crossings', 'exhausted_steps', 'label_deviations')}
    wb = {k: v for k, v in b.work.items() if k in ('n_hits', 'n_crossings', 'exhausted_steps', 'label_deviations')}
    out.append(f"work off {wa} on {wb}")
    ea, eb = a.comp_events, b.comp_events
    out.append(f"events equal: {all(np.array_equal(getattr(ea, f), getattr(eb, f)) for f in ('offsets', 't', 'label'))}; "
               f"n events {len(ea.t)} vs {len(eb.t)}; illegal {a.illegal_crossings} vs {b.illegal_crossings}")
    try:
        cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        model = [l for l in cpu.splitlines() if "Model name" in l]
        out.append(f"cpu {model} avx512f {'avx512f' in cpu}")
    except Exception as e:                                             # noqa: BLE001 -- a diagnostic
        out.append(f"cpu: {e}")
    import os
    out.append(f"jax cache dir {jax.config.jax_compilation_cache_dir} env {os.environ.get('DMIPY_JAX_CACHE')}")
    return "\n".join(out)


def test_the_record_changes_no_array_of_a_mixed_cells_walk(mixed_walks):
    a, b = mixed_walks
    same = all(np.array_equal(np.asarray(getattr(a, n)), np.asarray(getattr(b, n)))
               for n in ARRAYS if getattr(a, n) is not None)
    if not same:                                                       # TEMPORARY: the diagnosis, then the fresh compile
        import subprocess, sys, os
        code = ("import sys; sys.path.insert(0, 'tests'); import numpy as np; import test_trial_record as t; "
                "r = t._mixed(); r.count_walls = True; a = t._walk(r); b = t._walk(t._mixed(), record_trials=t.SPEC); "
                "print(t._diagnose(a, b))")
        fresh = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               env=dict(os.environ, DMIPY_JAX_CACHE="0"))
        raise AssertionError("DIAGNOSIS (in-process, persistent cache as configured):\n" + _diagnose(a, b)
                             + "\nFRESH PROCESS, NO PERSISTENT CACHE:\n" + fresh.stdout[-3000:] + fresh.stderr[-1500:])
    _assert_same_walk(*mixed_walks)
    r = mixed_walks[1].trial_record
    assert r.n.shape == (32, 4, 3)
    assert int(r.n[:, [0, 2]].sum()) == 0 and int(r.n[:, [1, 3]].sum()) > 0       # the myelinated cylinders' cells empty


# ── 5. the overshoot's bound ────────────────────────────────────────────────────────────────────────────────────────

def test_every_overshoot_is_within_its_objects_bound():
    """A dense cell walked to at least 1e5 trials: every trial's overshoot is within its object's ``u_max`` (the walk
    counts the trials beyond it and refuses a walk with any), the bound stated per object as ``2 LONGEST_STEP h / R``."""
    w = _walk(_dense(), n=320, T_max=5e-3, record_trials=TrialRecordSpec(modes=2, band_edges=(0.0, 5e-3)), K=4)
    r = w.trial_record
    assert int(r.n.sum()) >= 100_000
    np.testing.assert_allclose(r.u_max, u_max_of(RADII, np.full(6, D), w.dt_sim), rtol=0, atol=0)
    mean_u = r.S[..., 0].sum(axis=(0, 2)) / np.maximum(r.n.sum(axis=(0, 2)), 1)
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
    r = full.trial_record
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
    init, add = device_record(1, steps, [1.0], 1, jnp.float32)
    one = TrialLog(jnp.array([True]), jnp.array([0], jnp.int32), jnp.array([0.5], jnp.float32), jnp.array([False]))
    bands = {}
    for s in range(10):
        n = np.asarray(add(init(), one, jnp.int32(s))[0])[0]
        bands[s] = int(np.flatnonzero(n)[0])
    assert bands == {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2}


# ── 8. the stored count ─────────────────────────────────────────────────────────────────────────────────────────────

def test_the_record_stores_the_trial_count_beside_the_granted_count(plain_walks):
    r = plain_walks[1].trial_record
    assert r.n.dtype == np.int32 and r.N.dtype == np.int32 and np.all(r.N <= r.n)
    assert r.n.shape == r.N.shape == (32, 6, 3) and r.S.shape == (32, 6, 3, 3)
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
    for f in ("n", "N", "S", "u_max", "object_radii"):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f), err_msg=f)
    assert a.band_edges == b.band_edges and a.modes == b.modes


def test_a_walkers_row_is_its_index_in_the_seed(tmp_path):
    """The walk of walkers [0, 32) and the join of the walks of [0, 16) and [16, 32) (each its own batch size) record
    the same; a walk file holds the record."""
    from dmipy_sim.coefficient_walk import CoefficientWalk, join_walks
    whole = _walk(_dense(), n=32, record_trials=SPEC, events_capacity=64)
    a = _walk_geometry(16, D, _dense(), T, BLOCK, seed=3, tiers="all", require_gpu=False, K=8, record_trials=SPEC,
                       events_capacity=64, walker_batch_size=8)
    b = _walk_geometry(16, D, _dense(), T, BLOCK, seed=3, tiers="all", require_gpu=False, K=8, record_trials=SPEC,
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
    earlier band, window 0's."""
    spec = TrialRecordSpec(modes=2, band_edges=(0.0, 1e-3, 2e-3, 3e-3, T))
    w = _walk(_dense(), window_T=2e-3, record_trials=spec, K=4)
    assert w.n_windows == 2
    w0, w1 = w.window(0).trial_record, w.window(1).trial_record
    assert w0.band_edges == (0.0, 1e-3, 2e-3) and w1.band_edges == (2e-3, 3e-3, T)
    np.testing.assert_array_equal(np.concatenate([w0.n, w1.n], axis=2), w.trial_record.n)
    alone = _walk(_dense(), T_max=2e-3, sub_steps=w.sub_steps, K=4,
                  record_trials=TrialRecordSpec(modes=2, band_edges=(0.0, 1e-3, 2e-3)))
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
    init, add = device_record(6, steps, u_max, 5, jnp.float64)
    rec = init()
    log = jax.tree_util.tree_map(np.asarray, h.trials)
    for s in range(len(pts)):
        rec = add(rec, jax.tree_util.tree_map(lambda a: jnp.asarray(a[s]), h.trials), jnp.int32(s))
    n, N, S, exceed = (np.asarray(x) for x in rec)
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
    ("dense", TrialRecordSpec(modes=50, band_edges=tuple(np.linspace(0.0, T, 65))), {},
     "trial_record: 79872 bytes per walker exceed 65536"),
    ("myelinated", SPEC, {}, "trial_record: PackedMyelinatedCylinders has no plain wall whose trials a walk records"),
])
def test_a_record_the_walk_cannot_hold_is_refused(geo, spec, kw, message):
    g = _dense() if geo == "dense" else PackedMyelinatedCylinders(OUTER * 0.7, [0.7] * 4, CENTRES, L_CELL, N_max=4,
                                                                  compartments=COMPS)
    with pytest.raises(ValueError, match="^" + re.escape(message) + "$"):
        _walk(g, n=4, record_trials=spec, **kw)
