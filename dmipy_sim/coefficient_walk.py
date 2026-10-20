"""The coefficient walk: what :func:`~dmipy_sim.engine.core.simulate_trajectories` returns -- a walk kept as the bands a
pack stores.

Its walkers' paths are, per storage window, the exact start and end and the sine bands of the bridge between them on
the continuous basis, accumulated while the walk ran (:mod:`dmipy_sim.engine.accumulate`); the contact channel (C2) is
the cumulative local time in the same form; the occupancy (C1) is one label per walker on a substrate whose walkers
cannot change pool, else the per-block track. :func:`~dmipy_sim.replay.bank.build_replay_pack` takes it as it is. A
walker's raw positions exist only for the ``raw_subset`` it was asked to keep, and :meth:`CoefficientWalk.saves` is
that subset as a :class:`~dmipy_sim.persistent_walk.PersistentWalk` of its block ends.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


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
        ``{h, m, n_blocks, rule, precision, k0_residual_m, quadratic_variation, ...}``: the block length (s), the order,
        the block count over the walk, whether the block was derived from the band or given, how the accumulation
        was summed, the float32 chain's measured residual on the ``k = 0`` mode against the exact displacement (the
        largest over the walkers and windows, metres) and the channels' mean quadratic variation.
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
    compartment : (n_walkers,) int8, (n_walkers, n_blocks + 1) or None
        The pool label per walker (a substrate no walker leaves its pool in, or the start's pool of a walk kept as
        events), else the per-block track (sample 0 the start; a permeable wall's fractional occupancy as float32,
        a label as int8).
    comp_events : (offsets, t, label) or None
        C1 as an event list (dmipy-sim#742, #747; a backend's walk): walker ``w``'s pool changes are the entries
        ``offsets[w] .. offsets[w + 1]`` of ``t`` (s, the start of the step that ended in the pool entered) and
        ``label`` (int8, the pool entered); ``compartment`` is then each walker's start pool.
    bound_frac : (n_walkers, n_blocks + 1) or None
        The bound fraction per block of a binding (MT) walk.
    field_basis, field_samples, field_sample_every
        The field basis the walk sampled and its block means ``(n_walkers, n_f, n_ch)`` on the field's own grid of
        every ``field_sample_every``-th block (sample 0 the start position's), what the pack's path channel encodes.
    raw_positions, raw_boundary, raw_compartment : arrays of the first ``n_raw`` walkers or None
        Their positions ``(n_raw, n_blocks + 1, 3)``, contact per block and pool label (or occupancy) per block, kept
        with ``raw_subset=``.
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
    comp_events: Optional[tuple] = None
    bound_frac: Optional[np.ndarray] = None
    field_basis: object = field(default=None, compare=False, repr=False)
    field_samples: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    field_sample_every: int = 1
    raw_positions: Optional[np.ndarray] = None
    raw_boundary: Optional[np.ndarray] = None
    raw_compartment: Optional[np.ndarray] = None
    qv_positions: Optional[np.ndarray] = None
    qv_boundary: Optional[np.ndarray] = None
    illegal_crossings: int = 0
    seed: Optional[int] = None
    diffusivity: Optional[float] = None
    geometry: object = field(default=None, compare=False, repr=False)
    spec: object = field(default=None, compare=False, repr=False)
    weights: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    run: object = field(default=None, compare=False, repr=False)
    work: Optional[dict] = field(default=None, compare=False)
    scanner: Optional[str] = None

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
        return self.bound_frac is not None

    @property
    def nbytes(self):
        """Host bytes of the walk's arrays."""
        arrays = [self.position_coeffs, self.boundary_endpoint, self.boundary_bands, self.compartment, self.bound_frac,
                  self.field_samples, self.raw_positions, self.raw_boundary, self.raw_compartment, self.qv_positions,
                  self.qv_boundary] + list(self.comp_events or ())
        return int(sum(np.asarray(a).nbytes for a in arrays if a is not None))

    #: the per-walker arrays, walker first: what :meth:`take` reorders
    _WALKER_ARRAYS = ("position_coeffs", "boundary_endpoint", "boundary_bands", "compartment", "bound_frac", "field_samples",
                      "qv_positions", "qv_boundary", "weights")

    def take(self, idx):
        """The walk of the walkers ``idx`` (an index array), in that order; the raw subset is kept only when ``idx``
        starts with it unchanged."""
        import dataclasses
        idx = np.asarray(idx)
        out = {k: (None if getattr(self, k) is None else np.asarray(getattr(self, k))[idx]) for k in self._WALKER_ARRAYS}
        n_raw = 0 if self.raw_positions is None else int(self.raw_positions.shape[0])
        keep_raw = n_raw and idx.shape[0] >= n_raw and np.array_equal(idx[:n_raw], np.arange(n_raw))
        if not keep_raw:
            out.update(raw_positions=None, raw_boundary=None, raw_compartment=None)
        if self.comp_events is not None:
            off, t, lab = (np.asarray(a) for a in self.comp_events)
            counts = np.diff(off)[idx]
            new_off = np.zeros(idx.shape[0] + 1, np.int64); np.cumsum(counts, out=new_off[1:])
            sel = np.concatenate([np.arange(off[w], off[w + 1]) for w in idx]) if idx.size else np.zeros(0, np.int64)
            out["comp_events"] = (new_off, t[sel], lab[sel])
        return dataclasses.replace(self, **out)

    def window(self, i):
        """Window ``i`` as a one-window coefficient walk: its coefficients, contact and quadratic variation, the
        occupancy, bound fraction, field samples and raw subset on its blocks (sharing the boundary block end)."""
        import dataclasses
        n, i = int(self.n_windows), int(i)
        if not 0 <= i < n:
            raise IndexError(f"window {i} of a walk of {n}")
        if n == 1:
            return self
        if self.comp_events is not None:
            raise ValueError("a walk kept as events is accumulated in one window (dmipy-sim#747); window(i) does not cut an event list")
        n_bw = self.n_blocks // n
        k0, k1 = i * n_bw, (i + 1) * n_bw + 1
        cut = lambda a: None if a is None else (a if np.ndim(a) == 1 else np.asarray(a)[:, k0:k1])
        fs = None
        if self.field_samples is not None:
            e = int(self.field_sample_every)
            if k0 % e or n_bw % e:
                raise ValueError(f"the field was read every {e} blocks and window {i} does not start on that grid")
            fs = np.asarray(self.field_samples)[:, k0 // e:(k1 - 1) // e + 1]
        sl = lambda a: None if a is None else np.asarray(a)[:, i:i + 1]
        return dataclasses.replace(
            self, position_coeffs=sl(self.position_coeffs), T=self.T_window, n_windows=1,
            block=dict(self.block, n_blocks=n_bw, window=dict(index=i, of=n)),
            boundary_endpoint=sl(self.boundary_endpoint), boundary_bands=sl(self.boundary_bands),
            compartment=cut(self.compartment), bound_frac=cut(self.bound_frac), field_samples=fs,
            raw_positions=cut(self.raw_positions), raw_boundary=cut(self.raw_boundary),
            raw_compartment=cut(self.raw_compartment), qv_positions=sl(self.qv_positions), qv_boundary=sl(self.qv_boundary))

    #: the arrays a walk file holds (those present); the C1 event list as three tensors; the rest in the header
    _FILE_ARRAYS = ("position_coeffs", "boundary_endpoint", "boundary_bands", "compartment", "bound_frac", "field_samples",
                    "raw_positions", "raw_boundary", "raw_compartment", "qv_positions", "qv_boundary", "weights")
    _FILE_SCALARS = ("T", "K", "m", "block", "n_windows", "dt_sim", "sub_steps", "field_sample_every",
                     "illegal_crossings", "seed", "diffusivity", "work", "scanner")

    def save(self, path):
        """The walk as one safetensors file: its arrays as tensors (the C1 event list as ``comp_events_offsets`` /
        ``_t`` / ``_label``), the scalars, the block record, the spec and the run's summary as JSON under the ``"walk"``
        header key. The geometry is not stored: the spec is the situation, and :meth:`load` gives the walk that. A walk
        that sampled a field keeps its basis beside it, which this file does not hold, and is refused by name."""
        import json
        from safetensors.numpy import save_file
        if self.field_basis is not None:
            raise ValueError("this walk sampled a field basis, which a walk file does not hold; build its pack from the "
                             "walk in memory")
        arrays = {k: np.ascontiguousarray(getattr(self, k)) for k in self._FILE_ARRAYS if getattr(self, k) is not None}
        if self.comp_events is not None:
            off, t, lab = (np.ascontiguousarray(a) for a in self.comp_events)
            arrays.update(comp_events_offsets=off, comp_events_t=t, comp_events_label=lab)
        run = self.run
        header = {k: getattr(self, k) for k in self._FILE_SCALARS}
        header.update(spec=(None if self.spec is None else self.spec.to_dict()),
                      run=(None if run is None else (run.summary if hasattr(run, "summary") else run)),
                      geometry=(None if self.geometry is None else type(self.geometry).__name__))
        save_file(arrays, str(path), metadata={"walk": json.dumps(header, default=float)})

    @classmethod
    def load(cls, path):
        """The walk from :meth:`save`; a file that is not a coefficient walk is refused by name."""
        import json
        from safetensors import safe_open
        arrays = {}
        with safe_open(str(path), framework="numpy") as f:
            h = json.loads((f.metadata() or {}).get("walk") or "{}")
            for k in f.keys():
                arrays[k] = f.get_tensor(k)
        if "position_coeffs" not in arrays or "block" not in h:
            raise ValueError(f"{path} is not a coefficient walk file (no position_coeffs / block); a walk kept at its saves "
                             f"is not read by this revision -- re-walk it")
        spec = None
        if h.get("spec") is not None:
            from .spec.substrate import SubstrateSpec
            spec = SubstrateSpec.from_dict(h["spec"])
        kw = {k: arrays.get(k) for k in cls._FILE_ARRAYS}
        kw.update({k: h.get(k) for k in cls._FILE_SCALARS})
        if "comp_events_offsets" in arrays:
            kw["comp_events"] = (arrays["comp_events_offsets"], arrays["comp_events_t"], arrays["comp_events_label"])
        kw.update(K=int(h["K"]), m=int(h["m"]), n_windows=int(h["n_windows"]), sub_steps=int(h["sub_steps"]),
                  T=float(h["T"]), dt_sim=float(h["dt_sim"]), illegal_crossings=int(h.get("illegal_crossings") or 0),
                  field_sample_every=int(h.get("field_sample_every") or 1), spec=spec, run=h.get("run"))
        return cls(**kw)

    def saves(self):
        """The raw subset as a :class:`~dmipy_sim.persistent_walk.PersistentWalk` of its block ends: positions, the
        contact of the block ending at each (0 at the start), the pool label or occupancy per block, the bound
        fraction and the field samples of those walkers. Refused when the walk kept no raw subset."""
        from .persistent_walk import PersistentWalk
        if self.raw_positions is None:
            raise ValueError("this walk kept no raw positions: walk it with raw_subset=n")
        n = int(self.raw_positions.shape[0])
        head = lambda a: None if a is None else np.asarray(a)[:n]
        return PersistentWalk(np.asarray(self.raw_positions), self.h, int(self.sub_steps), float(self.dt_sim),
                              boundary_local_time=self.raw_boundary, compartment=self.raw_compartment,
                              bound_frac=head(self.bound_frac), illegal_crossings=int(self.illegal_crossings),
                              seed=self.seed, diffusivity=self.diffusivity, geometry=self.geometry, spec=self.spec,
                              weights=head(self.weights), field_basis=self.field_basis,
                              field_samples=head(self.field_samples), field_sample_every=int(self.field_sample_every),
                              run=self.run, work=self.work, scanner=self.scanner)
