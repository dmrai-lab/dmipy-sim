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
from dmipy_replay.replay.events import EventList, events_within










def _t_start(block):
    """The instant (s) a walk opens on the clock it is placed on: 0 -- a walk's clock starts at its first step -- unless
    :meth:`CoefficientWalk.shift_time` placed it later (``block["t_start"]``)."""
    return float(block["t_start"]) if "t_start" in block else 0.0


def _window_t0(block):
    """The instant (s) a walk's first block, or a window's (``block["window"]["t0"]``, :meth:`CoefficientWalk.window`),
    opens on the walk's clock."""
    w = block.get("window")
    return float(w["t0"]) if w is not None else _t_start(block)


@dataclass(frozen=True)
class CoefficientWalk:
    """One walk kept as its bridge bands.

    Attributes
    ----------
    position_coeffs : (n_walkers, n_windows, K + 2, 3) in the accumulation's dtype (``block["precision"]``)
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
        the band or given, the accumulators' dtype (float32, or float64 for the reference accumulation of the same
        walk), how the accumulation was summed, the accumulation chain's measured residual on the ``k = 0``
        mode against the exact displacement (the largest over the walkers and windows, metres), the channels' mean
        quadratic variation, the field read's rule and the field band's pilot ladder, the events buffers' pilot.
    n_windows : int
        The storage windows the walk was accumulated in.
    dt_sim : float
        The physics step (s).
    sub_steps : int
        Steps per block.
    boundary_endpoint : (n_walkers, n_windows) in the accumulation's dtype, or None
        Each window's total boundary local time at ``rho2 / D = 1`` (its start is 0), exact.
    boundary_bands : (n_walkers, n_windows, K) in the accumulation's dtype, or None
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
    qv_positions : (n_walkers, n_windows, 3) in the accumulation's dtype
        Each walker's quadratic variation per window and axis, ``sum (dx)^2`` over every step (``2 D T_w`` per axis
        for a free walk): what the block table's error is proportional to.
    qv_boundary : (n_walkers, n_windows) in the accumulation's dtype, or None
        The cumulative local time's quadratic variation per window, ``sum (dlog)^2`` (each hit's contact squared).
    trial_record : TrialRecord or None
        The membrane trials at the walk's plain walls per walker, one row per (object, time band) pair it met
        (:class:`~dmipy_sim.trial_record.TrialRecord`), of a walk asked for it (``record_trials=``); None otherwise.
    segment_rows : (n_walkers, n_seg, 3) in the accumulation's dtype, or None
        Per walker, declared time segment ``[a_s, b_s]`` and axis, ``int_{a_s}^{b_s} (x(t) - x(0)) dt`` (m s, ``x(0)``
        the walker's start) with ``x`` linear between the walk's steps: the trapezoid of its positions at its own step,
        accumulated in the walk (``segments=`` of :func:`~dmipy_sim.engine.core._walk_geometry`); no band enters it.
        None for a walk that declared no segments.
    segments : dict or None
        The declaration the rows were accumulated for: ``edges`` (s), their ``blocks`` (walk-wide block indices), the
        block ``h`` and the ``step`` (s), the ``weight`` rule, what a ``row`` is and the accumulation ``precision``.
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
    first_walker: int = 0      #: the index of the walk's first walker in its seed: the walk is the seed's walkers
                               #: ``first_walker .. first_walker + n_walkers - 1``
    diffusivity: Optional[float] = None
    geometry: object = field(default=None, compare=False, repr=False)
    spec: object = field(default=None, compare=False, repr=False)
    weights: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    run: object = field(default=None, compare=False, repr=False)
    work: Optional[dict] = field(default=None, compare=False)
    envelope: object = None    #: the :class:`~dmipy_replay.replay.envelope.Envelope` the walk was walked for (``walk``)
    stepping: Optional[dict] = field(default=None, compare=False, repr=False)
    trial_record: object = field(default=None, compare=False, repr=False)
    segment_rows: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    segments: Optional[dict] = None
    permeability_factor: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    #: (n,) float32 each walker's permeability factor (``_walk_geometry(permeability_factor=)``): walker ``w`` met the
    #: walls the geometry scales at ``kappa * permeability_factor[w]``; None when every walker walked at the stated value
    object_nodes: object = field(default=None, compare=False, repr=False)
    #: the walk's :class:`~dmipy_sim.engine.physics.ObjectNodes` (``_walk_geometry(object_nodes=)``): the factors
    #: (float32), fractions (float64), seed (int) and mask (bool, walker order) it walked; None without object nodes
    object_node_index: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    #: (n,) int64 each walker's stable index in the seed (``first_walker + row`` of the walk that walked it), the walker
    #: index the node hash read: with the seed and the realised masses of ``work["object_nodes"]`` it recomputes every
    #: walker's node at every object, through :meth:`take` and :func:`join_walks`; None without object nodes

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
                                                self.raw_field, self.qv_positions, self.qv_boundary, self.segment_rows,
                                                self.permeability_factor, self.object_node_index) if a is not None)
        if self.object_nodes is not None:
            n += sum(np.asarray(a).nbytes for a in (self.object_nodes.factors, self.object_nodes.fractions,
                                                    self.object_nodes.walkers))
        for ev in (self.comp_events, self.bound_events):
            if ev is not None:
                n += ev.offsets.nbytes + ev.t.nbytes + ev.label.nbytes
        if self.trial_record is not None:
            n += self.trial_record.nbytes
        return int(n)

    #: the per-walker arrays, walker first: what :meth:`take` reorders
    _WALKER_ARRAYS = ("position_coeffs", "boundary_endpoint", "boundary_bands", "compartment", "bound_start", "field_coeffs",
                      "qv_field", "qv_field_increments", "qv_positions", "qv_boundary", "weights", "pool_changed", "segment_rows",
                      "permeability_factor", "object_node_index")

    def start_error(self, t, *, per_face_local_time=False):
        """The start's term of a read at time ``t`` (s after the walk's start): ``{"relative_error", "flag"}``
        (:func:`~dmipy_sim.engine.physics.start_error`). It is 0 with no flag for a walk whose walkers started at
        spatial equilibrium (``work["start"]``) or that has no permeable wall. A walk out of it states
        :data:`~dmipy_sim.engine.physics.TRANSIENT_COEFFICIENT`'s ``h^2 / (D t)`` and a flag.
        With ``per_face_local_time=True``, a direct read of one face's local time instead states
        ``0.5 h (1 / sqrt(D t) + kappa_max / D)`` (maximum over objects and schedule intervals) with the distinct per-face local-time flag."""
        from .engine.physics import start_error
        return start_error((self.work or {}).get("start"), self.diffusivity, self.dt_sim, t,
                           per_face_local_time=per_face_local_time)

    def take(self, idx):
        """The walk of the walkers ``idx`` (an index array), in that order; the raw subset is kept only when ``idx``
        starts with it unchanged."""
        import dataclasses
        idx = np.asarray(idx)
        out = {k: (None if getattr(self, k) is None else np.asarray(getattr(self, k))[idx]) for k in self._WALKER_ARRAYS}
        for k in ("comp_events", "bound_events"):
            ev = getattr(self, k)
            out[k] = None if ev is None else ev.take(idx)
        out["trial_record"] = None if self.trial_record is None else self.trial_record.take(idx)
        n_raw = 0 if self.raw_positions is None else int(self.raw_positions.shape[0])
        keep_raw = n_raw and idx.shape[0] >= n_raw and np.array_equal(idx[:n_raw], np.arange(n_raw))
        if not keep_raw:
            out.update(raw_positions=None, raw_boundary=None, raw_compartment=None, raw_bound=None, raw_field=None)
        out["object_nodes"] = (None if self.object_nodes is None else
                               self.object_nodes._replace(walkers=np.asarray(self.object_nodes.walkers)[idx]))
        work = _with_factor_count(self.work, out["permeability_factor"])
        work = _with_node_counts(work, out["object_nodes"])
        return dataclasses.replace(self, **out, **({} if work is self.work else dict(work=work)))

    def window(self, i):
        """Window ``i`` as a one-window coefficient walk: its coefficients, contact, field modes and quadratic
        variation, the labels held at its start and the events inside it (times from the walk's start), the raw
        subset on its blocks (sharing the boundary block end), and the bands of the trial record within it (their
        edges on the walk's clock). A one-window walk is its own window 0, its segment rows with it; a window of a walk
        of several carries no segment rows, since a row is ``int (x(t) - x(0)) dt`` from the walk's start and the
        window's coefficients are referred to the window's own start."""
        import dataclasses
        n, i = int(self.n_windows), int(i)
        if not 0 <= i < n:
            raise IndexError(f"window {i} of a walk of {n}")
        if n == 1:
            return self
        n_bw = self.n_blocks // n
        T_w = self.T_window
        ts = _t_start(self.block)
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
            qv_positions=sl(self.qv_positions), qv_boundary=sl(self.qv_boundary),
            trial_record=(None if self.trial_record is None else self.trial_record.bands(i * T_w, (i + 1) * T_w)),
            segment_rows=None, segments=None)

    def shift_time(self, t_start):
        """This walk placed at ``t_start`` seconds on a longer clock (a continuation appended after a pack's windows):
        its events shifted, its windows' opening instants following (``block["t_start"]``)."""
        import dataclasses
        t_start = float(t_start)
        return dataclasses.replace(
            self, block=dict(self.block, t_start=t_start + _t_start(self.block)),
            comp_events=(None if self.comp_events is None else self.comp_events.shifted(t_start)),
            bound_events=(None if self.bound_events is None else self.bound_events.shifted(t_start)))

    #: the arrays a walk file holds (those present); the events as three tensors each; the rest in the header
    _FILE_ARRAYS = ("position_coeffs", "boundary_endpoint", "boundary_bands", "compartment", "bound_start", "field_coeffs",
                    "qv_field", "qv_field_increments", "raw_positions", "raw_boundary", "raw_compartment", "raw_bound", "raw_field", "qv_positions",
                    "qv_boundary", "weights", "segment_rows", "permeability_factor", "object_node_index")
    _FILE_SCALARS = ("T", "K", "m", "block", "n_windows", "dt_sim", "sub_steps", "K_c3", "field_read_interval",
                     "illegal_crossings", "seed", "first_walker", "diffusivity", "work", "stepping")

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
        if self.trial_record is not None:
            arrays.update({k: np.ascontiguousarray(v) for k, v in self.trial_record.to_arrays().items()})
        if self.object_nodes is not None:
            on = self.object_nodes
            arrays.update(object_nodes_factors=np.ascontiguousarray(on.factors),
                          object_nodes_fractions=np.ascontiguousarray(on.fractions),
                          object_nodes_walkers=np.ascontiguousarray(on.walkers))
        run = self.run
        fb = self.field_basis
        header = {k: getattr(self, k) for k in self._FILE_SCALARS}
        header.update(envelope=(None if self.envelope is None else self.envelope.to_dict()), spec=(None if self.spec is None else self.spec.to_dict()),
                      field=(None if fb is None else dict(kind=("grid" if hasattr(fb, "basis") else "hollow_cylinder"
                                                                if fb.meta.get("kind") == "hollow_cylinder" else "strands"), meta=fb.meta)),
                      run=(None if run is None else (run.summary if hasattr(run, "summary") else run)),
                      geometry=(None if self.geometry is None else type(self.geometry).__name__))
        if self.segments is not None:
            header["segments"] = self.segments
        if self.object_nodes is not None:
            header["object_nodes_seed"] = int(self.object_nodes.seed)
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
            from dmipy_replay.spec.substrate import SubstrateSpec
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
        if h.get("envelope") is not None:
            from dmipy_replay.replay.envelope import Envelope
            kw["envelope"] = Envelope.from_dict(h["envelope"])
        kw.update(K=int(h["K"]), m=int(h["m"]), n_windows=int(h["n_windows"]), sub_steps=int(h["sub_steps"]),
                  T=float(h["T"]), dt_sim=float(h["dt_sim"]), illegal_crossings=int(h["illegal_crossings"]),
                  first_walker=int(h["first_walker"]),
                  K_c3=(None if h.get("K_c3") is None else int(h["K_c3"])), spec=spec, field_basis=basis, run=h.get("run"),
                  segments=h.get("segments"), **events)
        from .trial_record import TrialRecord
        kw["trial_record"] = TrialRecord.from_arrays(arrays)
        if "object_nodes_factors" in arrays:
            from .engine.physics import ObjectNodes
            if h.get("object_nodes_seed") is None:
                raise ValueError("the walk file holds object-node arrays and no object_nodes_seed")
            kw["object_nodes"] = ObjectNodes(arrays["object_nodes_factors"], arrays["object_nodes_fractions"],
                                             int(h["object_nodes_seed"]), arrays["object_nodes_walkers"])
        walk = cls(**kw)
        check_permeability_factor(walk)
        check_object_nodes(walk)
        return walk


#: the entries of a walk's block record that are statistics of its walkers (:func:`ensemble_block_stats`)
ENSEMBLE_BLOCK_KEYS = ("k0_residual_m", "k0_residual_relative", "quadratic_variation")

#: the counts of a walk's work record, summed over the walks :func:`join_walks` joins
WORK_COUNTS = ("walker_steps", "n_hits", "n_crossings", "exhausted_steps", "label_deviations")


def ensemble_block_stats(position_coeffs, qv_positions, qv_boundary, k0_residual_m, diffusivity, T):
    """The entries of a block record that are statistics of the walk's walkers, from the arrays the walk stores: the
    accumulation chain's largest residual on the ``k = 0`` mode (``k0_residual_m``, given: the walk keeps no residual per
    walker), that residual over the displacements' rms, and each channel's quadratic variation summed over the
    windows and averaged over the walkers. A function of the stored arrays in walker order, so a walk and the
    pieces of it joined in order (:func:`join_walks`) state the same record."""
    disp = np.asarray(position_coeffs)[:, :, 1, :]
    qv_p = np.asarray(qv_positions).sum(axis=1).mean(axis=0)
    rms = float(np.sqrt(np.mean(disp ** 2))) if disp.size else 0.0
    return dict(k0_residual_m=(None if k0_residual_m is None else float(k0_residual_m)),
                k0_residual_relative=(None if k0_residual_m is None else
                                      (float(k0_residual_m) / max(rms, 1e-300) if disp.size else 0.0)),
                quadratic_variation=dict(
                    positions_mean_m2=[float(x) for x in qv_p],
                    positions_over_2DT=([float(x) / (2.0 * float(diffusivity) * float(T)) for x in qv_p] if diffusivity else None),
                    boundary_mean_m2=(None if qv_boundary is None else float(np.asarray(qv_boundary).sum(axis=1).mean()))))


def _without_count(rec):
    """A permeability-factor record without its walker count (what the pieces of one walk share)."""
    return {k: v for k, v in rec.items() if k != "n"} if isinstance(rec, dict) else rec


def _with_factor_count(work, factor):
    """``work`` with its permeability-factor record's ``n`` the length of ``factor``; ``work`` itself when it states
    none."""
    if not work or "permeability_factor" not in work or factor is None:
        return work
    return dict(work, permeability_factor=dict(work["permeability_factor"], n=int(np.asarray(factor).shape[0])))


def check_permeability_factor(walk):
    """Refuse a walk whose work record states a permeability per walker that its array does not hold: the array
    missing, of another length than the record's ``n``, or of another dtype; or an array the record does not state."""
    rec = (walk.work or {}).get("permeability_factor")
    arr = walk.permeability_factor
    if rec is None and arr is None:
        return
    if rec is None:
        raise ValueError("the walk holds a permeability_factor array that its work record does not state")
    if arr is None:
        raise ValueError(f"the walk's work record states a permeability factor per walker (n = {rec.get('n')}) and the "
                         f"walk holds no permeability_factor array")
    arr = np.asarray(arr)
    if arr.shape != (int(rec["n"]),) or arr.dtype != np.dtype(rec["dtype"]):
        raise ValueError(f"the walk's permeability_factor array is {arr.dtype} of shape {arr.shape}; its work record "
                         f"states {rec['dtype']} of length {rec['n']}")


def _with_node_counts(work, nodes):
    """``work`` with its object-node record's ``n`` and ``n_per_object`` those of ``nodes``' mask; ``work`` itself
    when it states none."""
    if not work or "object_nodes" not in work or nodes is None:
        return work
    mask = np.asarray(nodes.walkers)
    return dict(work, object_nodes=dict(work["object_nodes"], n=int(mask.shape[0]), n_per_object=int(mask.sum())))


def _without_node_counts(rec):
    """An object-node record without its walker counts (what the pieces of one walk share)."""
    return {k: v for k, v in rec.items() if k not in ("n", "n_per_object")} if isinstance(rec, dict) else rec


def check_object_nodes(walk):
    """Refuse a walk whose work record states object nodes that its :attr:`CoefficientWalk.object_nodes` does not
    hold: missing, another seed or ``K``, a mask of another length or count, or arrays of another dtype; or object
    nodes the record does not state."""
    rec = (walk.work or {}).get("object_nodes")
    on = walk.object_nodes
    if rec is None and on is None:
        return
    if rec is None:
        raise ValueError("the walk holds object nodes that its work record does not state")
    if on is None:
        raise ValueError("the walk's work record states object nodes and the walk holds none")
    fac, frac, mask = np.asarray(on.factors), np.asarray(on.fractions), np.asarray(on.walkers)
    if (fac.dtype != np.float32 or frac.dtype != np.float64 or mask.dtype != np.bool_
            or fac.shape != (int(rec["K"]),) or frac.shape != (int(rec["K"]),)):
        raise ValueError(f"the walk's object nodes are factors {fac.dtype} {fac.shape}, fractions {frac.dtype} "
                         f"{frac.shape}; its work record states float32 and float64 of length K = {rec['K']}")
    if mask.shape != (int(rec["n"]),) or int(mask.sum()) != int(rec["n_per_object"]):
        raise ValueError(f"the walk's object-node mask is {mask.shape} with {int(mask.sum())} walkers; its work record "
                         f"states n = {rec['n']}, n_per_object = {rec['n_per_object']}")
    if int(on.seed) != int(rec["seed"]):
        raise ValueError(f"the walk's object-node seed {int(on.seed)} is not its work record's {rec['seed']}")
    idx = walk.object_node_index
    if idx is None or np.asarray(idx).dtype != np.int64 or np.asarray(idx).shape != (int(rec["n"]),):
        raise ValueError(f"the walk's object_node_index is {None if idx is None else (np.asarray(idx).dtype, np.asarray(idx).shape)}; "
                         f"its work record states n = {rec['n']} walkers, each with its int64 stable index")


def join_walks(walks):
    """One walk from walks of contiguous walker ranges of one seed, in order: walk ``j + 1``'s ``first_walker`` is walk
    ``j``'s plus its walkers. The per-walker arrays are concatenated, the events joined, the counts summed and the
    block's walker statistics recomputed on the union (:func:`ensemble_block_stats`), so with walker-indexed seeding
    the result is the walk of the union walked at once: what a producer that commits a walk batch by batch
    assembles. Everything else -- the walk's parameters, the block's rules and pilots, the work record's rules, the
    envelope, the spec -- must agree and is refused by name where it does not; a raw subset is kept from the first
    walk and refused on any other. The pieces of a walk with object nodes are each walked with them, a piece whose
    walkers are all outside the mask with an all-false mask; a piece walked without object nodes is another walk and
    does not join them."""
    import dataclasses
    walks = list(walks)
    if not walks:
        raise ValueError("join_walks takes at least one walk")
    w0 = walks[0]
    if not all(isinstance(w, CoefficientWalk) for w in walks):
        raise TypeError("join_walks takes CoefficientWalk objects")
    nxt = int(w0.first_walker)
    for j, w in enumerate(walks):
        if int(w.first_walker) != nxt:
            raise ValueError(f"walk {j} starts at walker {w.first_walker}, not {nxt}: join_walks joins contiguous ranges in order")
        nxt += w.n_walkers
        if j and w.raw_positions is not None:
            raise ValueError(f"walk {j} keeps a raw subset: a joined walk's raw subset leads it, so only the first walk keeps one")
    for key in ("T", "K", "m", "n_windows", "dt_sim", "sub_steps", "K_c3", "field_read_interval", "seed", "diffusivity", "segments"):
        vals = [getattr(w, key) for w in walks]
        if any(v != vals[0] for v in vals[1:]):
            raise ValueError(f"the walks differ in {key}: {vals}")
    for key in ("envelope", "spec"):
        vals = [getattr(w, key) for w in walks]
        dicts = [None if v is None else v.to_dict() for v in vals]
        if any(d != dicts[0] for d in dicts[1:]):
            raise ValueError(f"the walks differ in their {key}")
    rules = [{k: v for k, v in w.block.items() if k not in ENSEMBLE_BLOCK_KEYS} for w in walks]
    for j, r in enumerate(rules[1:], 1):
        if r != rules[0]:
            diff = sorted(k for k in set(r) | set(rules[0]) if r.get(k) != rules[0].get(k))
            raise ValueError(f"walk {j}'s block record differs from walk 0's in {diff}: a walk joined from pieces needs "
                             "matching events_capacity, K_c3, field_read_interval and field_calibration where used")
    for key, how in (("comp_events", None), ("bound_events", None)):
        present = [getattr(w, key) is not None for w in walks]
        if any(present) and not all(present):
            raise ValueError(f"some walks carry {key} and some do not")
    has_nodes = [w.object_nodes is not None for w in walks]
    if any(has_nodes) and not all(has_nodes):
        raise ValueError(f"object_nodes: walks {[j for j, h in enumerate(has_nodes) if h]} were walked with object nodes "
                         f"and walks {[j for j, h in enumerate(has_nodes) if not h]} without; a piece of a walk with "
                         f"object nodes is walked with them, its mask selecting none of its walkers if need be")
    for key in CoefficientWalk._WALKER_ARRAYS + ("trial_record",):
        present = [getattr(w, key) is not None for w in walks]
        if any(present) and not all(present):
            raise ValueError(f"some walks carry {key} and some do not")
    works = [w.work for w in walks]
    if any(x is None for x in works) and not all(x is None for x in works):
        raise ValueError("some walks carry a work record and some do not")
    work = None
    if works[0] is not None:
        same = [{k: (_without_count(v) if k == "permeability_factor" else _without_node_counts(v)
                     if k == "object_nodes" else v) for k, v in x.items()
                 if k not in WORK_COUNTS and k != "block"} for x in works]
        for j, r in enumerate(same[1:], 1):
            if r != same[0]:
                diff = sorted(k for k in set(r) | set(same[0]) if r.get(k) != same[0].get(k))
                raise ValueError(f"walk {j}'s work record differs from walk 0's in {diff}")
        work = dict(works[0])
        for k in WORK_COUNTS:
            if k in work:
                vals = [x.get(k) for x in works]
                work[k] = None if any(v is None for v in vals) else int(sum(vals))
    if len(walks) == 1:
        return w0
    out = {k: (None if getattr(w0, k) is None else np.concatenate([np.asarray(getattr(w, k)) for w in walks]))
           for k in CoefficientWalk._WALKER_ARRAYS}
    for key in ("comp_events", "bound_events"):
        out[key] = None if getattr(w0, key) is None else EventList.concatenate([getattr(w, key) for w in walks])
    if w0.trial_record is not None:
        from .trial_record import TrialRecord
        out["trial_record"] = TrialRecord.concatenate([w.trial_record for w in walks])
    k0 = [w.block.get("k0_residual_m") for w in walks]
    block = dict(rules[0], **ensemble_block_stats(out["position_coeffs"], out["qv_positions"], out["qv_boundary"],
                                                  None if any(v is None for v in k0) else max(k0), w0.diffusivity, w0.T))
    block = {k: block[k] for k in w0.block}                                   # the walk's own key order
    if work is not None and "block" in work:
        work["block"] = block
    nodes = [w.object_nodes for w in walks]
    if any(x is not None for x in nodes):                    # all of them: refused above otherwise
        for j, x in enumerate(nodes[1:], 1):
            if (int(x.seed) != int(nodes[0].seed) or not np.array_equal(x.factors, nodes[0].factors)
                    or not np.array_equal(x.fractions, nodes[0].fractions)):
                raise ValueError(f"walk {j}'s object nodes differ from walk 0's in their factors, fractions or seed")
        out["object_nodes"] = nodes[0]._replace(walkers=np.concatenate([np.asarray(x.walkers) for x in nodes]))
    work = _with_factor_count(work, out["permeability_factor"])
    work = _with_node_counts(work, out.get("object_nodes"))
    return dataclasses.replace(w0, **out, block=block, work=work,
                               illegal_crossings=int(sum(int(w.illegal_crossings) for w in walks)), run=None)


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
    starts = [w.get("start") for w in works]
    if all(s is not None for s in starts):
        out["start"] = merge_starts(starts)
    return out


def merge_starts(starts):
    """The start record (:func:`~dmipy_sim.engine.physics.start_record`) of a walk assembled from parts: the parts' one
    record when they share it. Otherwise it is out of spatial equilibrium when a part's is, not established when a part's
    is and none is out of it, else at equilibrium, with the largest transient coefficient and permeability over D a part states."""
    if all(s == starts[0] for s in starts):
        return dict(starts[0])
    eqs = [s.get("spatial_equilibrium") for s in starts]
    eq = False if any(e is False for e in eqs) else (None if any(e is None for e in eqs) else True)
    coeffs = [s["transient_coefficient"] for s in starts if s.get("transient_coefficient") is not None]
    return dict(spatial_equilibrium=eq, reason="the parts' starts differ: " + "; ".join(sorted({str(s.get("reason")) for s in starts})),
                transient_coefficient=(max(coeffs) if coeffs else None),
                kappa_over_D_max=(max(s["kappa_over_D_max"] for s in starts)
                                  if all(s.get("kappa_over_D_max") is not None for s in starts) else None))
