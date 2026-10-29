"""The coefficient walk: what every producer returns -- a walk kept as the bands a pack stores.

Its walkers' paths are, per storage window, the exact start and end and the sine bands of the bridge between them on
the continuous basis, accumulated while the walk ran (:mod:`dmipy_sim.engine.accumulate`); the contact channel (C2) is
the cumulative local time in the same form; the field (C3) the cosine modes of the field read along the path at the
walk's own read interval; the occupancy (C1) one label per walker and, on a substrate whose walkers can change pool,
the events ``(t, label)`` of every change, exact in time; the MT bound column its bind / unbind events likewise.
:func:`~dmipy_sim.replay.bank.build_replay_pack` takes it as it is. A walker's raw block ends exist only for the
``raw_subset`` it was asked to keep, as plain arrays on the walk (``raw_positions``, ``raw_boundary``,
``raw_compartment``, ``raw_bound``, ``raw_field``): the dense oracle a measured certificate reads.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class EventList:
    """One column's events over a walk: per walker the changes of its label, ``(t, label)`` with ``t`` in seconds from
    the walk's start, strictly increasing within a walker; ``offsets`` (n_walkers + 1) int64 delimit walker ``i``'s
    entries ``offsets[i]:offsets[i+1]``. The label held before the first change is the walker's start label, kept
    beside the list (``CoefficientWalk.compartment``, the bound column's start)."""
    __slots__ = ("offsets", "t", "label")

    def __init__(self, offsets, t, label):
        self.offsets = np.asarray(offsets, np.int64)
        self.t = np.asarray(t, np.float64)
        self.label = np.asarray(label)
        if self.offsets.ndim != 1 or self.t.shape != self.label.shape or self.offsets[-1] != self.t.shape[0]:
            raise ValueError("an event list is (offsets (n_w + 1), t (n_ev), label (n_ev)) with offsets[-1] == n_ev")

    @property
    def n_walkers(self):
        return int(self.offsets.shape[0] - 1)

    @property
    def n_events(self):
        return int(self.t.shape[0])

    @property
    def counts(self):
        return np.diff(self.offsets)

    def __iter__(self):
        """The list as its three arrays, ``offsets, t, label = events`` (RPK.md 9.5.7's tensors)."""
        return iter((self.offsets, self.t, self.label))

    def take(self, idx):
        """The lists of the walkers ``idx``, in that order."""
        idx = np.asarray(idx, np.int64)
        counts = self.counts[idx]
        off = np.concatenate([[0], np.cumsum(counts)])
        src = np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in idx]) if idx.size else np.zeros(0, np.int64)
        return EventList(off, self.t[src], self.label[src])

    def shifted(self, dt):
        """The same events ``dt`` seconds later on the clock."""
        return EventList(self.offsets, self.t + float(dt), self.label)

    @staticmethod
    def concatenate(lists):
        """The lists of consecutive walker ranges as one."""
        lists = list(lists)
        offs = [lists[0].offsets]
        for ev in lists[1:]:
            offs.append(ev.offsets[1:] + offs[-1][-1])
        return EventList(np.concatenate(offs), np.concatenate([ev.t for ev in lists]), np.concatenate([ev.label for ev in lists]))

    def __iter__(self):
        """``offsets, t, label = events``: the three arrays, in that order."""
        return iter((self.offsets, self.t, self.label))

    def __eq__(self, other):
        return (isinstance(other, EventList) and np.array_equal(self.offsets, other.offsets)
                and np.array_equal(self.t, other.t) and np.array_equal(self.label, other.label))

    def __repr__(self):
        return f"EventList({self.n_walkers} walkers, {self.n_events} events)"


def events_label_at(events, start, t):
    """The label every walker holds at time ``t`` (s): its ``start`` label, or the label of its last change at or
    before ``t``."""
    start = np.asarray(start)
    out = np.array(start, copy=True)
    if events is None or events.n_events == 0:
        return out
    w = np.repeat(np.arange(events.n_walkers), events.counts)
    sel = events.t <= float(t)
    if not sel.any():
        return out
    # the last event at or before t per walker: the entries are in time order within a walker, so the last selected wins
    last = {}
    for i, lab in zip(w[sel], events.label[sel]):
        last[i] = lab
    for i, lab in last.items():
        out[i] = lab
    return out


def events_within(events, start, t0, t1):
    """The changes with ``t0 < t <= t1`` per walker and the label held at ``t0``: ``(EventList, label_at_t0)`` --
    one window's list of the walk's (RPK.md 9.5.7)."""
    lab0 = events_label_at(events, start, t0)
    if events is None:
        return None, lab0
    w = np.repeat(np.arange(events.n_walkers), events.counts)
    sel = (events.t > float(t0)) & (events.t <= float(t1))
    counts = np.bincount(w[sel], minlength=events.n_walkers)
    return EventList(np.concatenate([[0], np.cumsum(counts)]), events.t[sel], events.label[sel]), lab0


def events_at_instants(events, start, times):
    """The label of every walker at each of ``times`` (s), ``(n_walkers, len(times))``: a walk's occupancy sampled on
    a reader's grid (never a stored form)."""
    times = np.asarray(times, np.float64).reshape(-1)
    start = np.asarray(start)
    out = np.repeat(start[:, None], times.size, axis=1)
    if events is None or events.n_events == 0:
        return out
    w = np.repeat(np.arange(events.n_walkers), events.counts)
    # for each (walker, time): the last event with t <= time; events are time-ordered within a walker
    for i in range(events.n_walkers):
        a, b = events.offsets[i], events.offsets[i + 1]
        if b == a:
            continue
        pos = np.searchsorted(events.t[a:b], times, side="right") - 1
        has = pos >= 0
        out[i, has] = events.label[a:b][pos[has]]
    return out


@dataclass(frozen=True)
class CoefficientWalk:
    """One walk kept as its bridge bands.

    Attributes
    ----------
    position_coeffs : (n_walkers, n_windows, K + 2, 3) float32
        Per window and axis ``[x(t0), x(t0 + T_w) - x(t0), beta_1..beta_K]``: the window's exact start and
        displacement and its bands on ``s_k = sqrt(2 / T_w) sin(pi k (t - t0) / T_w)``.
    T : float
        The walk's duration (s).
    K : int
        Bands per axis and channel, per window.
    m : int
        The order of the block table (:func:`~dmipy_sim.replay.compression.block_table`).
    block : dict
        ``{h, m, n_blocks, rule, precision, k0_residual_m, quadratic_variation, field_read?, field_band?, events?,
        ...}``: the block length (s), the order, the block count over the walk, whether the block was derived from
        the band or given, how the accumulation was summed, the float32 chain's measured residual on the ``k = 0``
        mode against the exact displacement (the largest over the walkers and windows, metres), the channels' mean
        quadratic variation, the field read's rule and the field band's pilot ladder, the events buffers' pilot.
    n_windows : int
        The storage windows the walk was accumulated in.
    dt_sim : float
        The physics step (s).
    sub_steps : int
        Steps per block.
    boundary_endpoint : (n_walkers, n_windows) float32 or None
        Each window's total boundary local time at ``rho2 / D = 1`` (its start is 0), exact.
    boundary_bands : (n_walkers, n_windows, K) float32 or None
        Its bridge bands.
    compartment : (n_walkers,) int8 or None
        The pool label every walker starts in (the ``.rpk`` convention: 0 extra, positive enclosed pools); None when
        the walk recorded no channels.
    comp_events : EventList or None
        The changes of that label over the walk, exact in time, on a substrate whose walkers can change pool (a
        permeable wall); None on one no walker leaves its pool in (the label is static).
    bound_start : (n_walkers,) int8 or None
        The MT bound state (0 free, 1 bound) every walker starts in, of a binding walk.
    bound_events : EventList or None
        Its bind / unbind events.
    field_basis, field_coeffs, K_c3, field_read_interval, qv_field, qv_field_increments
        The field basis the walk read and its modes ``(n_walkers, n_windows, K_c3 + 1, n_ch)`` on the window's
        orthonormal cosine basis, ``c_k = int e_k b dt``, accumulated from reads every ``field_read_interval``
        seconds at the kernel's position (the trapezoid between consecutive reads); ``qv_field`` ``(n_walkers,
        n_windows, n_ch)`` the sum of the squared increments ``(b dt_f)^2`` per component, ``qv_field_increments``
        the same shape, the read series' quadratic variation ``sum (b_{i+1} - b_i)^2`` per component (the field
        read's term, RPK.md 9.5.5).
    raw_positions, raw_boundary, raw_compartment, raw_bound, raw_field : arrays of the first ``n_raw`` walkers or None
        Their positions ``(n_raw, n_blocks + 1, 3)``, contact per block, pool label and bound state at every block
        end (int8), and the field's reads at the walk's read instants ``(n_raw, n_blocks * reads_per_block + 1,
        n_ch)``, read ``i`` at ``i * field_read_interval`` (read 0 the start's), kept with ``raw_subset=``.
    qv_positions : (n_walkers, n_windows, 3) float32
        Each walker's quadratic variation per window and axis, ``sum (dx)^2`` over every step (``2 D T_w`` per axis
        for a free walk): what the block table's error is proportional to.
    qv_boundary : (n_walkers, n_windows) float32 or None
        The cumulative local time's quadratic variation per window, ``sum (dlog)^2`` (each hit's contact squared).
    """
    position_coeffs: np.ndarray
    T: float
    K: int
    m: int
    block: dict
    n_windows: int
    dt_sim: float
    sub_steps: int
    boundary_endpoint: Optional[np.ndarray] = None
    boundary_bands: Optional[np.ndarray] = None
    compartment: Optional[np.ndarray] = None
    comp_events: Optional[EventList] = None
    bound_start: Optional[np.ndarray] = None
    bound_events: Optional[EventList] = None
    field_basis: object = field(default=None, compare=False, repr=False)
    field_coeffs: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    K_c3: Optional[int] = None
    field_read_interval: Optional[float] = None
    qv_field: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    qv_field_increments: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    raw_positions: Optional[np.ndarray] = None
    raw_boundary: Optional[np.ndarray] = None
    raw_compartment: Optional[np.ndarray] = None
    raw_bound: Optional[np.ndarray] = None
    raw_field: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    qv_positions: Optional[np.ndarray] = None
    qv_boundary: Optional[np.ndarray] = None
    illegal_crossings: int = 0
    pool_changed: Optional[np.ndarray] = None   #: (n,) bool the walkers whose label changed in an impermeable walk walked
                                                #: with ``drop_pool_changes``; None when the walk was not asked for them
    seed: Optional[int] = None
    diffusivity: Optional[float] = None
    geometry: object = field(default=None, compare=False, repr=False)
    spec: object = field(default=None, compare=False, repr=False)
    weights: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    run: object = field(default=None, compare=False, repr=False)
    work: Optional[dict] = field(default=None, compare=False)
    scanner: Optional[str] = None
    stepping: Optional[dict] = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if self.spec is None and self.geometry is not None:
            try:
                object.__setattr__(self, "spec", self.geometry.spec)
            except Exception:                       # a duck-typed geometry with no spec spelling
                pass

    @property
    def n_walkers(self):
        return int(self.position_coeffs.shape[0])

    @property
    def h(self):
        """The block length (s)."""
        return float(self.block["h"])

    @property
    def n_blocks(self):
        """The blocks over the walk."""
        return int(self.block["n_blocks"])

    @property
    def T_window(self):
        """One window's duration (s)."""
        return float(self.T) / int(self.n_windows)

    @property
    def has_surface(self):
        return self.boundary_bands is not None

    @property
    def has_compartments(self):
        return self.compartment is not None

    @property
    def has_binding(self):
        return self.bound_start is not None

    @property
    def has_field(self):
        return self.field_coeffs is not None

    @property
    def crossing(self):
        """Whether the walk's walkers can change pool (the occupancy is an event list, not a static label)."""
        return self.comp_events is not None

    @property
    def field_channel_names(self):
        return None if self.field_basis is None else list(self.field_basis.channel_names)

    @property
    def reads_per_block(self):
        """The field reads per block (every ``n_read`` steps and at the block's end), 0 without a field."""
        if self.field_read_interval is None:
            return 0
        fr = self.block.get("field_read") or {}
        if fr.get("reads_per_block") is not None:
            return int(fr["reads_per_block"])
        return max(int(np.ceil(float(self.h) / float(self.field_read_interval) - 1e-9)), 1)

    @property
    def field_read_instants(self):
        """The instants (s, from the walk's start) of the field reads a raw subset keeps, ``(n_blocks *
        reads_per_block + 1,)``: read 0 at the start, then every ``n_read`` steps and at every block end."""
        from .engine.accumulate import read_intervals
        fr = self.block.get("field_read") or {}
        steps, n_read, step_s = int(fr["steps_per_block"]), int(fr["n_read"]), float(fr["step_s"])
        within = np.cumsum(read_intervals(steps, n_read)) * step_s
        return np.concatenate([[0.0], (np.arange(int(self.n_blocks))[:, None] * float(self.h) + within[None, :]).ravel()])

    @property
    def nbytes(self):
        """Host bytes of the walk's arrays."""
        n = sum(np.asarray(a).nbytes for a in (self.position_coeffs, self.boundary_endpoint, self.boundary_bands,
                                                self.compartment, self.bound_start, self.field_coeffs, self.qv_field,
                                                self.qv_field_increments,
                                                self.raw_positions, self.raw_boundary, self.raw_compartment, self.raw_bound,
                                                self.raw_field, self.qv_positions, self.qv_boundary) if a is not None)
        for ev in (self.comp_events, self.bound_events):
            if ev is not None:
                n += ev.offsets.nbytes + ev.t.nbytes + ev.label.nbytes
        return int(n)

    #: the per-walker arrays, walker first: what :meth:`take` reorders
    _WALKER_ARRAYS = ("position_coeffs", "boundary_endpoint", "boundary_bands", "compartment", "bound_start", "field_coeffs",
                      "qv_field", "qv_field_increments", "qv_positions", "qv_boundary", "weights", "pool_changed")

    def take(self, idx):
        """The walk of the walkers ``idx`` (an index array), in that order; the raw subset is kept only when ``idx``
        starts with it unchanged."""
        import dataclasses
        idx = np.asarray(idx)
        out = {k: (None if getattr(self, k) is None else np.asarray(getattr(self, k))[idx]) for k in self._WALKER_ARRAYS}
        for k in ("comp_events", "bound_events"):
            ev = getattr(self, k)
            out[k] = None if ev is None else ev.take(idx)
        n_raw = 0 if self.raw_positions is None else int(self.raw_positions.shape[0])
        keep_raw = n_raw and idx.shape[0] >= n_raw and np.array_equal(idx[:n_raw], np.arange(n_raw))
        if not keep_raw:
            out.update(raw_positions=None, raw_boundary=None, raw_compartment=None, raw_bound=None, raw_field=None)
        return dataclasses.replace(self, **out)

    def window(self, i):
        """Window ``i`` as a one-window coefficient walk: its coefficients, contact, field modes and quadratic
        variation, the labels held at its start and the events inside it (times from the walk's start), and the raw
        subset on its blocks (sharing the boundary block end)."""
        import dataclasses
        n, i = int(self.n_windows), int(i)
        if not 0 <= i < n:
            raise IndexError(f"window {i} of a walk of {n}")
        if n == 1:
            return self
        n_bw = self.n_blocks // n
        T_w = self.T_window
        ts = float(self.block.get("t_start", 0.0))
        k0, k1 = i * n_bw, (i + 1) * n_bw + 1
        cut = lambda a: None if a is None else np.asarray(a)[:, k0:k1]
        rpb = self.reads_per_block
        cut_reads = lambda a: None if a is None else np.asarray(a)[:, k0 * rpb:(k1 - 1) * rpb + 1]
        sl = lambda a: None if a is None else np.asarray(a)[:, i:i + 1]
        comp, comp_ev = self.compartment, self.comp_events
        if comp_ev is not None:
            comp_ev, comp = events_within(comp_ev, comp, ts + i * T_w, ts + (i + 1) * T_w)
        bnd, bnd_ev = self.bound_start, self.bound_events
        if bnd_ev is not None:
            bnd_ev, bnd = events_within(bnd_ev, bnd, ts + i * T_w, ts + (i + 1) * T_w)
        return dataclasses.replace(
            self, position_coeffs=sl(self.position_coeffs), T=T_w, n_windows=1,
            block=dict(self.block, n_blocks=n_bw, window=dict(index=i, of=n, t0=ts + i * T_w)),
            boundary_endpoint=sl(self.boundary_endpoint), boundary_bands=sl(self.boundary_bands),
            compartment=comp, comp_events=comp_ev, bound_start=bnd, bound_events=bnd_ev,
            field_coeffs=sl(self.field_coeffs), qv_field=sl(self.qv_field), qv_field_increments=sl(self.qv_field_increments),
            raw_positions=cut(self.raw_positions), raw_boundary=cut(self.raw_boundary),
            raw_compartment=cut(self.raw_compartment), raw_bound=cut(self.raw_bound), raw_field=cut_reads(self.raw_field),
            qv_positions=sl(self.qv_positions), qv_boundary=sl(self.qv_boundary))

    def shift_time(self, t_start):
        """This walk placed at ``t_start`` seconds on a longer clock (a continuation appended after a pack's windows):
        its events shifted, its windows' opening instants following (``block["t_start"]``)."""
        import dataclasses
        t_start = float(t_start)
        return dataclasses.replace(
            self, block=dict(self.block, t_start=t_start + float(self.block.get("t_start", 0.0))),
            comp_events=(None if self.comp_events is None else self.comp_events.shifted(t_start)),
            bound_events=(None if self.bound_events is None else self.bound_events.shifted(t_start)))

    #: the arrays a walk file holds (those present); the events as three tensors each; the rest in the header
    _FILE_ARRAYS = ("position_coeffs", "boundary_endpoint", "boundary_bands", "compartment", "bound_start", "field_coeffs",
                    "qv_field", "qv_field_increments", "raw_positions", "raw_boundary", "raw_compartment", "raw_bound", "raw_field", "qv_positions",
                    "qv_boundary", "weights")
    _FILE_SCALARS = ("T", "K", "m", "block", "n_windows", "dt_sim", "sub_steps", "K_c3", "field_read_interval",
                     "illegal_crossings", "seed", "diffusivity", "work", "scanner", "stepping")

    def save(self, path):
        """The walk as one safetensors file: its arrays as tensors (an event list as ``<column>_events_offsets`` /
        ``_t`` / ``_label``), the scalars, the block record, the spec, the field basis's record and the run's summary
        as JSON under the ``"walk"`` header key. The geometry is not stored: the spec is the situation, and
        :meth:`load` gives the walk that; the field basis comes back as its record
        (:class:`~dmipy_sim.fields.strand_field.StrandFieldRecord` /
        :class:`~dmipy_sim.fields.susceptibility_field.FieldGridRecord`; a closed-form field is rebuilt from its record),
        from which a pack is built the same."""
        import json
        from safetensors.numpy import save_file
        arrays = {k: np.ascontiguousarray(getattr(self, k)) for k in self._FILE_ARRAYS if getattr(self, k) is not None}
        for col, ev in (("comp", self.comp_events), ("bound", self.bound_events)):
            if ev is not None:
                arrays[f"{col}_events_offsets"] = ev.offsets; arrays[f"{col}_events_t"] = ev.t; arrays[f"{col}_events_label"] = ev.label
        run = self.run
        fb = self.field_basis
        header = {k: getattr(self, k) for k in self._FILE_SCALARS}
        header.update(spec=(None if self.spec is None else self.spec.to_dict()),
                      field=(None if fb is None else dict(kind=("grid" if hasattr(fb, "basis") else "hollow_cylinder"
                                                                if fb.meta.get("kind") == "hollow_cylinder" else "strands"), meta=fb.meta)),
                      run=(None if run is None else (run.summary if hasattr(run, "summary") else run)),
                      geometry=(None if self.geometry is None else type(self.geometry).__name__))
        save_file(arrays, str(path), metadata={"walk": json.dumps(header, default=float)})

    @classmethod
    def load(cls, path):
        """The walk from :meth:`save`."""
        import json
        from safetensors import safe_open
        arrays = {}
        with safe_open(str(path), framework="numpy") as f:
            h = json.loads((f.metadata() or {}).get("walk") or "{}")
            for k in f.keys():
                arrays[k] = f.get_tensor(k)
        if "position_coeffs" not in arrays or "block" not in h:
            raise ValueError(f"{path} is not a coefficient walk file (no position_coeffs / block); a walk of saves is "
                             f"not read by this revision -- re-walk it")
        spec = None
        if h.get("spec") is not None:
            from .spec.substrate import SubstrateSpec
            spec = SubstrateSpec.from_dict(h["spec"])
        basis = None
        if h.get("field") is not None:
            if h["field"]["kind"] == "grid":
                from .fields.susceptibility_field import FieldGridRecord
                basis = FieldGridRecord(h["field"]["meta"])
            elif h["field"]["kind"] == "hollow_cylinder":                 # the closed form: its record IS the field
                from .fields.hollow_cylinder import HollowCylinderField
                basis = HollowCylinderField.from_meta(h["field"]["meta"])
            else:
                from .fields.strand_field import StrandFieldRecord
                basis = StrandFieldRecord(h["field"]["meta"])
        events = {}
        for col, name in (("comp", "comp_events"), ("bound", "bound_events")):
            if f"{col}_events_offsets" in arrays:
                events[name] = EventList(arrays[f"{col}_events_offsets"], arrays[f"{col}_events_t"], arrays[f"{col}_events_label"])
        kw = {k: arrays.get(k) for k in cls._FILE_ARRAYS}
        kw.update({k: h.get(k) for k in cls._FILE_SCALARS})
        kw.update(K=int(h["K"]), m=int(h["m"]), n_windows=int(h["n_windows"]), sub_steps=int(h["sub_steps"]),
                  T=float(h["T"]), dt_sim=float(h["dt_sim"]), illegal_crossings=int(h.get("illegal_crossings") or 0),
                  K_c3=(None if h.get("K_c3") is None else int(h["K_c3"])), spec=spec, field_basis=basis, run=h.get("run"), **events)
        return cls(**kw)


def merge_work(works):
    """The work record of a walk assembled from several (a bundle's pools, a fill's shards): the counts summed,
    an entry None when any part has it None, ``bounce_budget`` the common value or, when the parts differ, the
    largest, ``step_rule`` the one rule when the parts share it and the list of the parts' rules otherwise.
    ``None`` when no part carries a record."""
    works = [w for w in works if w]
    if not works:
        return None
    out = {}
    for key in ("walker_steps", "n_hits", "n_crossings", "exhausted_steps"):
        vals = [w.get(key) for w in works]
        out[key] = None if any(v is None for v in vals) else int(sum(vals))
    budgets = [w.get("bounce_budget") for w in works if w.get("bounce_budget") is not None]
    out["bounce_budget"] = None if not budgets else int(max(budgets))
    rules = [w.get("step_rule") for w in works if w.get("step_rule") is not None]
    out["step_rule"] = None if not rules else (rules[0] if all(r == rules[0] for r in rules) else rules)
    return out
