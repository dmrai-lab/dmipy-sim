"""The record of a walk's membrane trials at its plain walls.

A plain wall (:class:`~dmipy_sim.geometry.PackedCylinders`, :class:`~dmipy_sim.geometry.PackedSpheres`, a plain
cylinder of :class:`~dmipy_sim.geometry.PackedMyelinatedCylinders`) decides a transmission trial at every encounter of
a step, with probability ``p = x / (1 + x)``, ``x = 2 (kappa / D) d_perp``
(:func:`~dmipy_sim.geometry._boundary.transmit_probability`). A walk asked for it (``record_trials=``
:class:`TrialRecordSpec` on :func:`~dmipy_sim.engine.core._walk_geometry` and :func:`~dmipy_sim.walk`) returns, per
walker and per (object, time band) pair the walker met, the count of its trials ``n``, the count of granted trials
``N`` and the ``M`` sums ``S_j = sum u T_j(2 u / u_max - 1)``, ``u = 2 d_perp / R`` the trial's overshoot (``R`` the
radius of the object met), ``T_j`` the Chebyshev polynomial of the first kind and ``u_max`` the object's a priori bound
on ``u`` (:func:`u_max_of`): the :class:`TrialRecord` of :attr:`~dmipy_sim.coefficient_walk.CoefficientWalk.trial_record`.

The record is held in slots. Each walker carries ``slots`` of them, one per (object, band) pair it met, in the order it
first met them, and a count of the distinct pairs it met; a pair the walker never met has no slot and its ``n``, ``N``
and ``S`` are 0. The record's memory is :func:`bytes_per_walker` of the slots, independent of the cell's object
count, refused above :data:`MAX_BYTES_PER_WALKER`. The capacity is the spec's ``slots``, or, when that is None, the
largest count of a pilot of the walk's own walkers with the margin :data:`SLOTS_MARGIN` (the events buffer's rule),
stated in the walk's ``work["trial_slots"]``. A walker that meets more pairs than the capacity refuses the walk by name
with the largest count reached and a suggested capacity: no trial is dropped or added to another pair's sums.

The rules, each a condition a consumer's arithmetic relies on:

* every encounter of a plain wall is a trial and is recorded, the second and later encounters of a step included and
  an object of permeability 0 included (its trials are always refused: ``N`` is 0, ``n`` counts them); a step the walk
  discards (a walker held bound through its dwell) records none;
* a trial belongs to the step it is decided in, and a step to the band whose interval contains the step's END time
  ``(step + 1) dt``: band ``k`` is the left-open, right-closed ``(edge_k, edge_k+1]``, so a step ending exactly on an
  edge belongs to the earlier band, the final band contains ``T_max``, and a storage window's bands hold exactly its
  own steps; a scheduled permeability is read at the step's midpoint (:func:`~dmipy_sim.engine.physics.membrane_kappa`);
* the counts are integers and the sums are held in the walk's accumulation dtype, each pair's sums added trial by
  trial in the walk's order; nothing is rounded or dithered, and the record draws nothing (the walk's random stream is
  the walk's without the record);
* the record reads the walk and enters none of its arithmetic: one step of a wall kernel is the same bits with and
  without it. The whole walk with it and without it are two compiled programs: the same bits on the recording build
  and host, while another JAX build may arrange them so that a position's last bits differ, which the reflections can
  grow into another encounter -- two valid walks of one law. The record is exactly consistent with its own walk (the
  record turns ``Geometry.count_walls`` on, of which it is the pair-resolved form: ``sum n`` over the rows of an
  all-plain cell is ``work["n_hits"]`` and ``sum N`` is ``work["n_crossings"]``).
"""
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

#: The most bands a record holds.
MAX_BANDS = 64
#: The most bytes the slotted record holds per walker (:func:`bytes_per_walker`).
MAX_BYTES_PER_WALKER = 65536
#: The most bytes per walker :meth:`TrialRecord.to_dense` forms (:func:`dense_bytes_per_walker`).
MAX_DENSE_BYTES_PER_WALKER = 65536
#: The pilot's capacity rule: ``ceil(SLOTS_MARGIN[0] x the pilot's largest count) + SLOTS_MARGIN[1]`` slots, the
#: events buffer's own margin (:data:`~dmipy_sim.engine.accumulate.EVENTS_MARGIN`).
SLOTS_MARGIN = (1.5, 8)
#: The slots a pilot walker first carries; a pilot whose walkers meet more pairs walks again at twice its largest
#: count, so that the count it states is exact.
PILOT_SLOTS = 64
#: An edge within this fraction of a physics step of a step's end is that step's end (the rounding of seconds).
EDGE_SNAP = 1e-9


@dataclass(frozen=True)
class TrialRecordSpec:
    """What a walk records of its membrane trials: ``modes`` Chebyshev sums per (object, band) pair (``M``, at least 1)
    over the time bands ``band_edges`` (seconds, strictly increasing, from 0 to the walk's ``T_max``, at most
    :data:`MAX_BANDS` bands, containing every boundary of the walk's storage windows so that a band never straddles two
    windows), in ``slots`` slots per walker (a whole number of at least 1, or None: the capacity a pilot of the walk's
    own walkers states, :func:`pilot_capacity`)."""
    modes: int
    band_edges: tuple
    slots: object = None

    def __post_init__(self):
        m = self.modes
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or int(m) < 1:
            raise ValueError(f"trial_record: modes must be a whole number of at least 1, got {m!r}")
        edges = np.asarray(self.band_edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
            raise ValueError(f"trial_record: band_edges must be at least two finite times in seconds, got {self.band_edges!r}")
        if not np.all(np.diff(edges) > 0):
            raise ValueError(f"trial_record: band_edges must be strictly increasing, got {self.band_edges!r}")
        if edges.size - 1 > MAX_BANDS:
            raise ValueError(f"trial_record: {edges.size - 1} bands exceed {MAX_BANDS}")
        s = self.slots
        if s is not None and (isinstance(s, bool) or not isinstance(s, (int, np.integer)) or int(s) < 1):
            raise ValueError(f"trial_record: slots must be a whole number of at least 1, or None for a pilot, got {s!r}")
        object.__setattr__(self, "modes", int(m))
        object.__setattr__(self, "band_edges", tuple(float(e) for e in edges))
        object.__setattr__(self, "slots", None if s is None else int(s))

    @property
    def n_bands(self):
        return len(self.band_edges) - 1


def edge_steps(spec, T_max, dt, n_steps, window_steps, window_s=None):
    """The step-end index of every band edge of ``spec`` for a walk of ``n_steps`` physics steps of ``dt`` seconds over
    ``T_max`` seconds in storage windows of ``window_steps`` steps (``window_s`` seconds each, the walk's
    ``window_T``): edge ``e`` is at the last step end ``j dt <= e``, ``j = floor(e / dt)`` (an edge within
    :data:`EDGE_SNAP` of a step of a step end is that step end, the rounding of seconds), so the steps ending in
    ``(edge_k, edge_k+1]`` are those of index ``j`` with ``steps[k] < j + 1 <= steps[k + 1]``. Refused by name: edges that
    do not run from 0 to ``T_max``, and edges that miss a window boundary, checked on the times as given (an edge
    farther than :data:`EDGE_SNAP` of a step from the boundary misses it, before any edge is placed on a step end)."""
    edges = np.asarray(spec.band_edges, np.float64)
    if edges[0] != 0.0 or edges[-1] != float(T_max):
        raise ValueError(f"trial_record: band_edges must run from 0 to T_max = {float(T_max)!r}, got "
                         f"[{float(edges[0])!r}, {float(edges[-1])!r}]")
    window_steps = int(window_steps)
    tol = EDGE_SNAP * float(dt)
    for w in range(1, int(n_steps) // window_steps):
        t = w * (float(window_s) if window_s is not None else window_steps * float(dt))
        if not np.any(np.abs(edges - t) <= tol):
            raise ValueError(f"trial_record: band_edges must contain the window boundary {t!r}")
    q = edges / float(dt)
    near = np.rint(q)
    steps = np.where(np.abs(q - near) <= EDGE_SNAP, near, np.floor(q)).astype(np.int64)
    steps[0], steps[-1] = 0, int(n_steps)                     # the walk's first and last instants
    return steps


def u_max_of(radii, D, dt):
    """The a priori bound on a trial's overshoot ``u = 2 d_perp / R`` at each object: ``2 LONGEST_STEP h / R``,
    ``h = sqrt(6 D dt)`` with ``D`` the larger diffusivity of the two pools the object's wall separates, since
    ``d_perp`` is at most the step and the step law's longest step is ``LONGEST_STEP h``
    (:data:`~dmipy_sim.math.step_law.LONGEST_STEP`). ``(n_objects,)`` float64."""
    from .math.step_law import LONGEST_STEP
    radii, D = np.asarray(radii, np.float64), np.asarray(D, np.float64)
    return 2.0 * float(LONGEST_STEP) * np.sqrt(6.0 * D * float(dt)) / radii


def bytes_per_walker(slots, modes, itemsize):
    """The slotted record's bytes per walker: ``slots * (3 * 4 + modes * itemsize) + 2 * 4`` (per slot an int32 key and
    two int32 counts and ``modes`` sums of ``itemsize`` bytes; per walker the int32 count of pairs met and of trials
    beyond ``u_max``)."""
    return int(slots) * (12 + int(modes) * int(itemsize)) + 8


def check_bytes(slots, modes, itemsize):
    """:func:`bytes_per_walker`, refused by name above :data:`MAX_BYTES_PER_WALKER`."""
    b = bytes_per_walker(slots, modes, itemsize)
    if b > MAX_BYTES_PER_WALKER:
        raise ValueError(f"trial_record: {int(slots)} slots of {int(modes)} modes are {b} bytes per walker, above "
                         f"{MAX_BYTES_PER_WALKER}")
    return b


def dense_bytes_per_walker(n_objects, n_bands, modes, itemsize):
    """The bytes per walker of the record as one cell per (object, band) pair (:meth:`TrialRecord.to_dense`):
    ``n_objects * n_bands * (2 * 4 + modes * itemsize)``."""
    return int(n_objects) * int(n_bands) * (8 + int(modes) * int(itemsize))


def pilot_capacity(top, n_pairs):
    """The capacity from a pilot whose walkers met at most ``top`` distinct (object, band) pairs: ``ceil(SLOTS_MARGIN[0]
    top) + SLOTS_MARGIN[1]`` (:data:`SLOTS_MARGIN`), and never more than the ``n_pairs`` pairs there are."""
    return int(min(int(np.ceil(SLOTS_MARGIN[0] * int(top))) + int(SLOTS_MARGIN[1]), int(n_pairs)))


def suggested_capacity(top):
    """The capacity suggested to a walk refused for a walker that met ``top`` pairs: the pilot's rule on ``top``."""
    return int(np.ceil(SLOTS_MARGIN[0] * int(top))) + int(SLOTS_MARGIN[1])


def chebyshev_terms(x, modes, xp):
    """``(T_0(x), ..., T_{modes-1}(x))`` stacked on a new last axis, by the three-term recurrence in ``x``'s dtype."""
    out = [xp.ones_like(x)]
    if modes > 1:
        out.append(x)
    for _ in range(2, modes):
        out.append(2 * x * out[-1] - out[-2])
    return xp.stack(out, axis=-1)


class StepTrials(NamedTuple):
    """One step's trials as the record adds them (:attr:`DeviceRecord.trials`), one entry per entry of the step's
    :class:`~dmipy_sim.geometry._boundary.TrialLog`, in its order."""
    valid: object    #: (L,) bool -- a trial of a step the walk keeps
    obj: object      #: (L,) int32 -- the object met
    band: object     #: int32 -- the band whose ``(edge_k, edge_k+1]`` holds the step's end
    granted: object  #: (L,) bool -- the crossing was granted
    terms: object    #: (L, modes) in the record's dtype -- ``u T_j(2 u / u_max - 1)``, ``j < modes``
    exceeds: object  #: (L,) bool -- a trial whose ``u`` exceeds its object's ``u_max``


class DeviceRecord(NamedTuple):
    """The record as a walk carries it per walker (:func:`device_record`)."""
    init: object        #: ``init(slots)``: the empty record
    add: object         #: ``add(rec, log, step, keep=None)``: ``add_trials(rec, trials(log, step, keep))``
    trials: object      #: ``trials(log, step, keep=None)``: the step's :class:`StepTrials`
    add_trials: object  #: ``add_trials(rec, trials)``: the record with a step's trials added


def device_record(steps, u_max, modes, dtype):
    """The record as a walk carries it per walker: a :class:`DeviceRecord`. ``init(slots)`` is the empty record ``(key
    (slots,) int32, n (slots,) int32, N (slots,) int32, S (slots, modes) dtype, count int32, exceed int32)``, every key
    -1 (no pair). ``trials(log, step, keep=None)`` is the :class:`StepTrials` of a
    :class:`~dmipy_sim.geometry._boundary.TrialLog` of the walk-wide step ``step`` (int32): its band, the band ``k``
    whose ``(edge_k, edge_k+1]`` holds the step's end, and each trial's terms, ``keep`` False discarding the step (a
    walker held bound). ``add_trials(rec, trials)`` adds them in the log's order: an entry of a pair already held (key
    ``object * n_bands + k``) adds its terms to that slot's sums; an entry of a new pair takes the next free slot and
    adds 1 to ``count``. ``count`` keeps counting past the slots: exact while it is at most ``slots``, an upper count
    beyond (the keys of the pairs that found no slot are not held), and an entry that finds no slot adds to no slot.
    ``exceed`` counts the trials whose ``u`` exceeds the object's ``u_max``. ``add(rec, log, step, keep=None)`` is
    ``add_trials(rec, trials(log, step, keep))``."""
    import jax.numpy as jnp
    n_bands = int(len(steps) - 1)
    interior = jnp.asarray(np.asarray(steps[1:-1], np.int64), jnp.int32)
    umax = jnp.asarray(np.asarray(u_max, np.float64), dtype)
    modes = int(modes)

    def init(slots):
        slots = int(slots)
        z = jnp.zeros((slots,), jnp.int32)
        return (jnp.full((slots,), -1, jnp.int32), z, z, jnp.zeros((slots, modes), dtype), jnp.int32(0), jnp.int32(0))

    def trials(log, step, keep=None):
        valid = log.valid if keep is None else (log.valid & keep)
        band = jnp.sum(interior < (jnp.asarray(step, jnp.int32) + 1)).astype(jnp.int32)    # (edge_k, edge_k+1]
        obj = log.obj.astype(jnp.int32)
        u = log.u.astype(dtype)
        um = umax[obj]
        terms = u[:, None] * chebyshev_terms(2 * u / um - 1, modes, jnp)
        return StepTrials(valid, obj, band, log.granted, terms, valid & (u > um))

    def add_trials(rec, t):
        key, n, N, S, count, exceed = rec
        slot_index = jnp.arange(key.shape[0], dtype=jnp.int32)
        entry_key = t.obj * jnp.int32(n_bands) + t.band
        for i in range(int(t.valid.shape[0])):            # the log's entries in order (the bounce loop's iterations)
            v = t.valid[i]
            held = (key == entry_key[i]) & v
            opens = v & ~jnp.any(held)
            new = (slot_index == count) & opens          # the next free slot; none once the slots are full
            key = jnp.where(new, entry_key[i], key)
            at = held | new
            n = n + at.astype(jnp.int32)
            N = N + (at & t.granted[i]).astype(jnp.int32)
            # the sums: the entry's terms, as formed, added to its slot by a scatter-add (a select of ``S + terms``
            # over the slots compiles to a fused multiply-add of the product into the sum, one rounding fewer)
            S = S.at[jnp.argmax(at)].add(jnp.where(jnp.any(at), t.terms[i], jnp.zeros((), S.dtype)))
            count = count + opens.astype(jnp.int32)
        exceed = exceed + jnp.sum(t.exceeds).astype(jnp.int32)
        return (key, n, N, S, count, exceed)

    def add(rec, log, step, keep=None):
        return add_trials(rec, trials(log, step, keep))

    return DeviceRecord(init, add, trials, add_trials)


class DenseTrials(NamedTuple):
    """A record as one cell per (object, band) pair (:meth:`TrialRecord.to_dense`)."""
    n: np.ndarray   #: (n_walkers, n_objects, n_bands) int32
    N: np.ndarray   #: (n_walkers, n_objects, n_bands) int32
    S: np.ndarray   #: (n_walkers, n_objects, n_bands, modes) in the record's dtype


class WalkerRows(NamedTuple):
    """One walker's rows (:meth:`TrialRecord.walker`), in the order it first met each pair."""
    obj: np.ndarray    #: (rows,) int32 the object
    band: np.ndarray   #: (rows,) int32 the band
    n: np.ndarray      #: (rows,) int32 the trials
    N: np.ndarray      #: (rows,) int32 the granted trials
    S: np.ndarray      #: (rows, modes) the sums


#: The tensors of a walk file that hold a record (:meth:`TrialRecord.to_arrays`).
FILE_TENSORS = ("trial_row_obj", "trial_row_band", "trial_row_n", "trial_row_N", "trial_row_S", "trial_rows",
                "trial_u_max", "trial_band_edges", "trial_object_radii")
#: The tensors of a walk file that held the record as one cell per (object, band) pair: refused on reading.
DENSE_FILE_TENSORS = ("trial_n", "trial_N", "trial_S")


@dataclass(frozen=True)
class TrialRecord:
    """A walk's membrane trials at its plain walls, per walker (rows in walker order: row ``w`` is walker
    ``first_walker + w`` of the seed) and per (object, time band) pair the walker met.

    Walker ``w``'s pairs are its first ``rows[w]`` columns, in the order it first met them; the columns after are
    padding (object and band -1, counts and sums 0), and the width is the largest ``rows``, so that sums over a column
    axis are the walker's totals. A pair absent from a walker's rows was never met by it.

    Attributes
    ----------
    obj : (n_walkers, width) int32
        The object of each row, in the geometry's object order (``TrialLog.obj``); -1 in the padding.
    band : (n_walkers, width) int32
        The band of each row; band ``k`` holds the steps ending in ``(edge_k, edge_k+1]``; -1 in the padding.
    n : (n_walkers, width) int32
        The trials: every encounter of the object's wall by a step ending in the band.
    N : (n_walkers, width) int32
        The granted trials (crossings).
    S : (n_walkers, width, modes) in the walk's accumulation dtype
        ``S_j = sum u T_j(2 u / u_max[o] - 1)`` over the row's trials, ``u = 2 d_perp / R_o`` the overshoot.
    rows : (n_walkers,) int32
        The pairs each walker met.
    u_max : (n_objects,) float64
        Each object's bound on ``u`` (:func:`u_max_of`).
    band_edges : tuple of float
        The bands' edges in seconds on the walk's clock.
    modes : int
        ``M``, the sums per row.
    object_radii : (n_objects,) float64
        The radius ``R_o`` of each object (m), in the geometry's object order.
    """
    obj: np.ndarray
    band: np.ndarray
    n: np.ndarray
    N: np.ndarray
    S: np.ndarray
    rows: np.ndarray
    u_max: np.ndarray
    band_edges: tuple
    modes: int
    object_radii: np.ndarray = field(repr=False)

    def __post_init__(self):
        rows = np.asarray(self.rows, np.int32)
        width = int(rows.max()) if rows.size else 0
        obj = np.asarray(self.obj, np.int32)
        if obj.ndim != 2 or obj.shape[0] != rows.shape[0] or obj.shape[1] < width:
            raise ValueError(f"trial_record: rows of shape {obj.shape} for {rows.shape[0]} walkers of up to {width} rows")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "obj", obj[:, :width])
        object.__setattr__(self, "band", np.asarray(self.band, np.int32)[:, :width])
        object.__setattr__(self, "n", np.asarray(self.n, np.int32)[:, :width])
        object.__setattr__(self, "N", np.asarray(self.N, np.int32)[:, :width])
        object.__setattr__(self, "S", np.asarray(self.S)[:, :width])

    @classmethod
    def from_slots(cls, key, n, N, S, count, *, u_max, band_edges, modes, object_radii):
        """The record from the slots a walk carried per walker (:func:`device_record`: ``key`` ``object * n_bands +
        band`` or -1, ``n``, ``N``, ``S`` and the ``count`` of pairs met, at most the slots)."""
        key = np.asarray(key, np.int64)
        count = np.asarray(count, np.int64)
        if count.size and int(count.max()) > key.shape[1]:
            raise ValueError(f"trial_record: a walker met {int(count.max())} pairs where the record holds {key.shape[1]}")
        n_bands = len(band_edges) - 1
        held = key >= 0
        return cls(obj=np.where(held, key // n_bands, -1).astype(np.int32),
                   band=np.where(held, key % n_bands, -1).astype(np.int32),
                   n=np.asarray(n, np.int32), N=np.asarray(N, np.int32), S=np.asarray(S),
                   rows=count.astype(np.int32), u_max=np.asarray(u_max, np.float64),
                   band_edges=tuple(float(e) for e in band_edges), modes=int(modes),
                   object_radii=np.asarray(object_radii, np.float64))

    @property
    def n_walkers(self):
        return int(self.rows.shape[0])

    @property
    def n_bands(self):
        return len(self.band_edges) - 1

    @property
    def n_objects(self):
        return int(self.u_max.shape[0])

    @property
    def nbytes(self):
        return int(sum(np.asarray(a).nbytes for a in (self.obj, self.band, self.n, self.N, self.S, self.rows,
                                                      self.u_max, self.object_radii)))

    def totals(self):
        """Per walker the trials and the granted trials over all its rows: ``(n (n_walkers,), N (n_walkers,))`` int64."""
        return self.n.sum(axis=1, dtype=np.int64), self.N.sum(axis=1, dtype=np.int64)

    def walker(self, w):
        """Walker ``w``'s rows (:class:`WalkerRows`), in the order it first met each pair."""
        k = int(self.rows[w])
        return WalkerRows(self.obj[w, :k], self.band[w, :k], self.n[w, :k], self.N[w, :k], self.S[w, :k])

    def _replace_rows(self, sel=None, **kw):
        import dataclasses
        if sel is not None:
            kw = dict(obj=self.obj[sel], band=self.band[sel], n=self.n[sel], N=self.N[sel], S=self.S[sel],
                      rows=self.rows[sel], **kw)
        return dataclasses.replace(self, **kw)

    def take(self, idx):
        """The record of the walkers ``idx`` (an index array or a slice), in that order."""
        return self._replace_rows(idx)

    def bands(self, t0, t1):
        """The record of the bands within ``[t0, t1]`` (seconds), which must be band edges: each walker's rows of those
        bands, in their order, the bands renumbered from the first."""
        edges = np.asarray(self.band_edges)
        i0 = int(np.flatnonzero(np.isclose(edges, t0, rtol=0.0, atol=EDGE_SNAP * max(abs(t1 - t0), 1e-300)))[0])
        i1 = int(np.flatnonzero(np.isclose(edges, t1, rtol=0.0, atol=EDGE_SNAP * max(abs(t1 - t0), 1e-300)))[0])
        keep = (self.band >= i0) & (self.band < i1)
        order = np.argsort(~keep, axis=1, kind="stable")                  # the kept rows first, in their order
        rows = keep.sum(axis=1).astype(np.int32)
        pad = np.arange(self.obj.shape[1])[None, :] >= rows[:, None]
        g = lambda a: np.take_along_axis(a, order, axis=1)
        obj, band = np.where(pad, -1, g(self.obj)), np.where(pad, -1, g(self.band) - i0)
        n, N = np.where(pad, 0, g(self.n)), np.where(pad, 0, g(self.N))
        S = np.where(pad[:, :, None], np.zeros((), self.S.dtype), np.take_along_axis(self.S, order[:, :, None], axis=1))
        return self._replace_rows(obj=obj, band=band, n=n, N=N, S=S, rows=rows,
                                  band_edges=tuple(self.band_edges[i0:i1 + 1]))

    def same_layout(self, other):
        """Whether ``other`` records the same objects, bands and modes."""
        return (self.band_edges == other.band_edges and self.modes == other.modes
                and np.array_equal(self.u_max, other.u_max) and np.array_equal(self.object_radii, other.object_radii))

    @staticmethod
    def concatenate(records):
        """The records of contiguous walker ranges joined in order (the same layout, refused by name otherwise), each
        padded to the widest."""
        r0 = records[0]
        for j, r in enumerate(records[1:], 1):
            if not r0.same_layout(r):
                raise ValueError(f"trial_record: walk {j} records other objects, bands or modes than walk 0")
        width = max(r.obj.shape[1] for r in records)

        def cat(name, fill):
            parts = []
            for r in records:
                a = getattr(r, name)
                pad = [(0, 0), (0, width - a.shape[1])] + [(0, 0)] * (a.ndim - 2)
                parts.append(np.pad(a, pad, constant_values=fill))
            return np.concatenate(parts)
        return r0._replace_rows(obj=cat("obj", -1), band=cat("band", -1), n=cat("n", 0), N=cat("N", 0), S=cat("S", 0),
                                rows=np.concatenate([r.rows for r in records]))

    def to_dense(self):
        """The record as one cell per (object, band) pair (:class:`DenseTrials`), a pair a walker never met 0: for a
        small cell, refused by name above :data:`MAX_DENSE_BYTES_PER_WALKER` bytes per walker
        (:func:`dense_bytes_per_walker`)."""
        b = dense_bytes_per_walker(self.n_objects, self.n_bands, self.modes, self.S.dtype.itemsize)
        if b > MAX_DENSE_BYTES_PER_WALKER:
            raise ValueError(f"trial_record: the dense form of {self.n_objects} objects and {self.n_bands} bands is {b} "
                             f"bytes per walker, above {MAX_DENSE_BYTES_PER_WALKER}")
        W = self.n_walkers
        n = np.zeros((W, self.n_objects, self.n_bands), np.int32)
        N = np.zeros_like(n)
        S = np.zeros((W, self.n_objects, self.n_bands, self.modes), self.S.dtype)
        w, c = np.nonzero(self.obj >= 0)
        o, k = self.obj[w, c], self.band[w, c]
        n[w, o, k], N[w, o, k], S[w, o, k] = self.n[w, c], self.N[w, c], self.S[w, c]
        return DenseTrials(n, N, S)

    def to_arrays(self):
        """The record as the tensors of a walk file (:data:`FILE_TENSORS`)."""
        return {"trial_row_obj": self.obj, "trial_row_band": self.band, "trial_row_n": self.n, "trial_row_N": self.N,
                "trial_row_S": self.S, "trial_rows": self.rows, "trial_u_max": self.u_max,
                "trial_band_edges": np.asarray(self.band_edges, np.float64), "trial_object_radii": self.object_radii}

    @classmethod
    def from_arrays(cls, arrays):
        """The record from :meth:`to_arrays`, or None when the file holds none. A file that holds the record as one
        cell per (object, band) pair (:data:`DENSE_FILE_TENSORS`) is refused by name."""
        dense = [k for k in DENSE_FILE_TENSORS if k in arrays]
        if dense:
            raise ValueError(f"trial_record: the walk file holds the record as one cell per (object, band) pair "
                             f"({', '.join(dense)}); this revision reads the record's rows ({', '.join(FILE_TENSORS)}): "
                             f"re-walk it")
        if "trial_rows" not in arrays:
            return None
        S = arrays["trial_row_S"]
        return cls(obj=arrays["trial_row_obj"], band=arrays["trial_row_band"], n=arrays["trial_row_n"],
                   N=arrays["trial_row_N"], S=S, rows=arrays["trial_rows"], u_max=arrays["trial_u_max"],
                   band_edges=tuple(float(e) for e in arrays["trial_band_edges"]), modes=int(S.shape[-1]),
                   object_radii=arrays["trial_object_radii"])
