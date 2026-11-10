"""The MT walk's pool label (dmrai-lab/dmrai#72): every recorded time carries the pool of the position the walker is
seen at (``_pool2``: 0 extra, 1 enclosed), bound, at the release and free.

The MT walk records the position a walker is SEEN at (``inner_step_mt_seen``): where it resumes from, except while it is
bound on the near face of an encounter it crossed, when it is seen at the start of the step it bound in. The label is
carried with that position (the near face's pool while bound there, the resume position's pool otherwise), so

* at every block end of the raw subset the label is the exact classification of the recorded position, with binding
  on (permeable packed cylinders, which carry their side, and packed spheres, which classify) and off;
* the C1 events, read at the block ends, are the raw subset's labels (the events are the label's changes, exact in
  time);
* a near-face bind is seen on its face's side until its release: some label changes fall on a release, and none on
  a step in which the walker stayed bound;
* an impermeable MT walk keeps every walker's start label.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine.core import _walk_geometry

D = 2e-9
R = 2e-6
L_CYL = 4.4 * R                            # the splice tests' four cylinders (a 0.4 um gap)
L_SPH = 4.0 * R
T = 4e-3
H = 5e-5                                   # the block
KAPPA = 4e-5                               # kappa h / D ~ 0.5 at the step law's h
KON = 4e-5                                 # the binding reactivity, m/s
TAU = 1e-3                                 # the dwell, s
N = 1500


def _cylinders(permeability):
    c = np.array([[0.0, 0.0], [2.2, 0.0], [0.0, 2.2], [2.2, 2.2]]) * R
    return d.PackedCylinders(np.array([R, 0.8 * R, R, 0.9 * R]), c, L_CYL, permeability=permeability)


def _spheres(permeability):
    c = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]) * L_SPH
    return d.PackedSpheres(np.array([R, 0.9 * R, R, 0.85 * R]), c, L_SPH, permeability=permeability)


def _walk(geometry, kappa_MT, seed=3):
    return _walk_geometry(N, D, geometry, T, H, seed=seed, raw_subset=N, require_gpu=False, K=8, kappa_MT=kappa_MT,
                          dwell_time=TAU if kappa_MT > 0.0 else 0.0,
                          **(dict(equilibrate_binding="off") if kappa_MT > 0.0 else {}))


def _pool_of_positions(geometry, pos):
    """The exact pool (0 extra, 1 enclosed) of every recorded position ``(n, n_t, 3)``."""
    flat = np.asarray(pos, np.float32).reshape(-1, 3)
    ids = np.asarray(geometry.classify_positions_exact(flat), np.int64)
    return np.minimum(ids, 1).reshape(pos.shape[:2])


def _labels_at_block_ends(walk):
    """The C1 column read at every block end: the start label until the first event, then the label of the last event
    at or before the instant (events hold from their time)."""
    n_t = walk.raw_positions.shape[1]
    t_b = np.arange(n_t) * float(walk.block["h"])
    lab = np.repeat(np.asarray(walk.compartment, np.int64)[:, None], n_t, axis=1)
    if walk.comp_events is None:
        return lab
    off, t, label = walk.comp_events
    half = 0.5 * float(walk.dt_sim)
    for i in range(lab.shape[0]):
        for k in range(off[i], off[i + 1]):
            lab[i, t_b >= t[k] - half] = int(label[k])
    return lab


@pytest.fixture(scope="module")
def mt_cylinders():
    geo = _cylinders(KAPPA)
    return geo, _walk(geo, KON)


@pytest.mark.parametrize("which", ["cylinders_mt", "spheres_mt", "cylinders_free", "spheres_free"])
def test_the_label_is_the_pool_of_the_recorded_position_at_every_block_end(which, mt_cylinders):
    if which == "cylinders_mt":
        geo, w = mt_cylinders
    else:
        geo = (_cylinders if which.startswith("cylinders") else _spheres)(KAPPA)
        w = _walk(geo, KON if which.endswith("_mt") else 0.0)
    lab = np.asarray(w.raw_compartment, np.int64)
    pool = _pool_of_positions(geo, w.raw_positions)
    assert lab.shape == pool.shape
    crossed = (lab != lab[:, :1]).any(axis=1).mean()
    assert crossed > 0.05                                      # the walk crosses: the comparison resolves changes
    bad = int((lab != pool).sum())
    assert bad == 0, f"{bad} of {lab.size} recorded times carry a label other than the pool of the recorded position"
    assert np.array_equal(lab[:, 0], np.asarray(w.compartment, np.int64))
    np.testing.assert_array_equal(_labels_at_block_ends(w), lab)   # the events are the label's changes, exact in time
    if which.endswith("_mt"):
        assert float(np.asarray(w.raw_bound, np.float64)[:, -1].mean()) > 0.02    # it binds
        bound = np.asarray(w.raw_bound) > 0
        assert int((lab != pool)[bound].sum()) == 0               # bound walkers too


def test_a_near_face_bind_is_seen_on_its_face_until_the_release(mt_cylinders):
    """A label change falls on a release (the walker bound on the near face of a crossing, seen there, resumes on the
    far side) and none on a step the walker spent bound throughout."""
    geo, w = mt_cylinders
    off_c, t_c, _ = w.comp_events
    off_b, t_b, lab_b = w.bound_events
    dt = float(w.dt_sim)
    at_release = 0
    while_bound = 0
    for i in range(len(off_c) - 1):
        tc = np.round(t_c[off_c[i]:off_c[i + 1]] / dt).astype(np.int64)
        tb = np.round(t_b[off_b[i]:off_b[i + 1]] / dt).astype(np.int64)
        lb = np.asarray(lab_b[off_b[i]:off_b[i + 1]])
        releases = set(tb[lb == 0].tolist())
        at_release += sum(1 for s in tc if s in releases)
        # the bound intervals [bind, release): a label change strictly inside one is a change while held
        start = int(np.asarray(w.bound_start)[i])
        state, since = start, 0
        for s, l in zip(tb, lb):
            if state == 1 and l == 0:
                while_bound += int(np.sum((tc > since) & (tc < s)))
            state, since = int(l), int(s)
    assert at_release > 0, "no label change at a release: the near face's binds are not seen on their side"
    assert while_bound == 0, f"{while_bound} label changes while the walker was held"


def test_an_impermeable_mt_walk_keeps_every_start_label():
    geo = _cylinders(None)
    w = _walk(geo, KON)
    lab = np.asarray(w.raw_compartment, np.int64)
    assert w.comp_events is None
    assert np.array_equal(lab, np.repeat(lab[:, :1], lab.shape[1], axis=1))
    assert np.array_equal(lab[:, 0], np.asarray(w.compartment, np.int64))
    assert np.array_equal(lab, _pool_of_positions(geo, w.raw_positions))
    assert w.work["label_deviations"] == 0


def test_the_mt_walk_reports_its_own_work_record(mt_cylinders):
    """The MT walk's work record is its carry's (hits, crossings, exhaustion), not the seen position's: with counting
    on, the crossings are the walkers' granted crossings, at least as many as their label changes."""
    geo = _cylinders(KAPPA)
    geo.count_walls = True
    w = _walk(geo, KON)
    n_label_changes = int(np.diff(w.comp_events.offsets).sum())
    assert w.work["n_crossings"] is not None and w.work["n_hits"] is not None
    assert w.work["n_hits"] >= w.work["n_crossings"] > 0
    assert w.work["n_crossings"] >= n_label_changes                 # every label change is a crossing's, at it or at its release
    assert w.work["exhausted_steps"] == 0
