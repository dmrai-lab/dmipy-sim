"""A walk kept at its block ends: positions and the boundary, compartment, binding and field channels at every block
end -- the raw subset of a coefficient walk (:meth:`~dmipy_sim.coefficient_walk.CoefficientWalk.saves`,
``raw_subset=``), the dense oracle and the picture of the paths. Its channels are attributes, present or ``None``
by what the walk recorded. No pack is built from it: a pack is the bands the walk accumulated
(:func:`~dmipy_sim.replay.bank.build_replay_pack` takes the :class:`~dmipy_sim.coefficient_walk.CoefficientWalk`).
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PersistentWalk:
    """One field-independent walk kept at its block ends (``dt`` apart): positions depend only on geometry,
    diffusivity and seed, never on the acquisition.

    Attributes
    ----------
    positions : (n_walkers, n_t, 3) float32
        Walker positions in metres at each block end, the start the first.
    dt : float
        The block length, ``T_max / (n_t - 1)``.
    sub_steps : int
        Physics sub-steps per block (:func:`physics.resolve_sub_steps`).
    dt_sim : float
        The physics step, ``dt / sub_steps``.
    boundary_local_time : (n_walkers, n_t) or None
        Per-block boundary log-weight at ``rho2/D = 1`` (``-2 * sum d_perp`` over the block's wall
        hits, non-positive); the surface-relaxivity channel. ``None`` when not recorded.
    compartment : (n_walkers, n_t) int8 or None
        Compartment id at every block end (0 extra-cellular, positive enclosed pools; the ``.rpk``
        convention).
    bound : (n_walkers, n_t) int8 or None
        The bound state (0 free, 1 bound) at every block end of a binding (MT) walk.
    illegal_crossings : int
        The fused producer's count of walker-steps that ended on the far side of a membrane without a granted
        crossing and were rejected back into their compartment, tested at every sub-step. The adaptive producer
        (:mod:`dmipy_sim.engine.adaptive`) counts coarser -- a walker's pool at the walk's end against its start,
        once -- and does not see a walker that crossed a wall and returned before the end: a pass-through within
        the walk is invisible to this count there. Non-zero means some walker ended in the wrong pool.
    seed : int or None
        The producer's master seed.
    diffusivity : float or None
        The free diffusivity the walk was run with (m^2/s); what a surface-relaxivity replay divides
        rho2 by.
    geometry : Geometry or None
        The substrate the walk was run on; the pack builder reads its pools and its field basis from
        it. Not part of equality and not stored in a pack.
    spec : SubstrateSpec or None
        The substrate spec the walk was driven by (a spec-driven multi-pool walk has no single
        geometry); the pack embeds it and reads the pools from it.
    weights : (n_walkers,) or None
        Per-walker spin weights when the seeding rule leaves them non-uniform.
    field_samples : (n_walkers, n_t, 7 | 13) or None
        The field basis channels along every walker's path, read BY THE WALK every ``field_read_interval`` seconds at
        the kernel's position and averaged over each block (the trapezoid of the reads; sample 0 the start
        position's read).
    field_read_interval : float or None
        The interval (s) between the walk's reads of the field.
    stepping : dict or None
        How the producer stepped when it was not one fixed step for every walker (the adaptive producer's
        free-step fraction, kernel steps against the fused count, its classes); None for the fused walk.
    field_basis : FieldGrid, StrandFieldBasis or None
        The substrate's susceptibility field basis when the producer computed it: a grid (a mesh, a
        rasterised substrate) or the per-segment closed form of a strand substrate; both answer
        ``channels(points)``.
    run : dmipy_sim.run.Run or None
        The record of the run that produced the walk (``run.summary`` once it has ended; a pack records it in
        its provenance).
    work : dict or None
        What the walk cost in wall interactions: ``walker_steps`` (walkers x saves x sub-steps), ``bounce_budget``
        (the geometry's :class:`~dmipy_sim.geometry._boundary.BounceLoop` budget, None without a loop) and, when
        the geometry counted (``Geometry.count_walls``), ``n_hits`` (reflections and granted crossings),
        ``n_crossings`` (steps that changed side) and ``exhausted_steps`` (steps whose bounce budget ran out with
        path untested; zero certifies the budget). The counts are None when the walk did not count. Not part of
        equality; a pack writes it beside itself in the run sidecar.
    """
    positions: np.ndarray
    dt: float
    sub_steps: int
    dt_sim: float
    boundary_local_time: Optional[np.ndarray] = None
    compartment: Optional[np.ndarray] = None
    bound: Optional[np.ndarray] = None
    illegal_crossings: int = 0
    seed: Optional[int] = None
    diffusivity: Optional[float] = None
    geometry: object = field(default=None, compare=False, repr=False)
    spec: object = field(default=None, compare=False, repr=False)
    weights: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    field_basis: object = field(default=None, compare=False, repr=False)
    stepping: Optional[dict] = field(default=None, compare=False, repr=False)
    field_samples: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    field_read_interval: Optional[float] = None
    run: object = field(default=None, compare=False, repr=False)
    work: Optional[dict] = field(default=None, compare=False)
    scanner: Optional[str] = None               #: the scanner class the save grid was derived for (walk_spec), else None

    def __post_init__(self):
        # a walk always knows the situation it was walked in: the spec it was driven by, else its geometry's
        if self.spec is None and self.geometry is not None:
            try:
                object.__setattr__(self, "spec", self.geometry.spec)
            except Exception:                       # a duck-typed geometry with no spec spelling
                pass

    @property
    def n_walkers(self):
        return int(self.positions.shape[0])

    @property
    def n_t(self):
        return int(self.positions.shape[1])

    @property
    def T_max(self):
        return float(self.dt) * (self.n_t - 1)

    @property
    def storage_dtype(self):
        return self.positions.dtype

    @property
    def has_surface(self):
        """The surface-relaxivity tier is replayable."""
        return self.boundary_local_time is not None

    @property
    def has_compartments(self):
        """Per-compartment relaxation is replayable."""
        return self.compartment is not None

    @property
    def has_binding(self):
        """The MT bound-pool blend is replayable."""
        return self.bound is not None

    #: the arrays a walk file holds (those present), and the header's scalars
    _FILE_ARRAYS = ("positions", "boundary_local_time", "compartment", "bound", "weights", "field_samples")

    def save(self, path):
        """The walk as one safetensors file: its arrays as tensors, the rest (``dt``, ``sub_steps``, ``dt_sim``,
        ``illegal_crossings``, ``seed``, ``diffusivity``, ``field_read_interval``, ``stepping``, ``work``, the spec,
        the field basis's record, the run's summary) as JSON under the ``"walk"`` header key. The geometry is not stored: the
        spec is the situation, and :meth:`load` gives the walk that; the field basis comes back as its record
        (:class:`~dmipy_sim.fields.strand_field.StrandFieldRecord`). An inspection file: no pack is built from it."""
        import json
        from safetensors.numpy import save_file
        arrays = {k: np.ascontiguousarray(getattr(self, k)) for k in self._FILE_ARRAYS if getattr(self, k) is not None}
        run = self.run
        header = dict(dt=float(self.dt), sub_steps=int(self.sub_steps), dt_sim=float(self.dt_sim), illegal_crossings=int(self.illegal_crossings),
                      seed=self.seed, diffusivity=self.diffusivity, field_read_interval=self.field_read_interval, stepping=self.stepping,
                      work=self.work, scanner=self.scanner,
                      spec=(None if self.spec is None else self.spec.to_dict()),
                      field=(None if self.field_basis is None else getattr(self.field_basis, "meta", None)),
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
        spec = None
        if h.get("spec") is not None:
            from .spec.substrate import SubstrateSpec
            spec = SubstrateSpec.from_dict(h["spec"])
        basis = None
        if h.get("field") is not None:
            from .fields.strand_field import StrandFieldRecord
            basis = StrandFieldRecord(h["field"])
        return cls(arrays["positions"], float(h["dt"]), int(h["sub_steps"]), float(h["dt_sim"]),
                   boundary_local_time=arrays.get("boundary_local_time"), compartment=arrays.get("compartment"),
                   bound=arrays.get("bound"), illegal_crossings=int(h.get("illegal_crossings", 0)), seed=h.get("seed"),
                   diffusivity=h.get("diffusivity"), spec=spec, weights=arrays.get("weights"), field_basis=basis,
                   stepping=h.get("stepping"), field_samples=arrays.get("field_samples"),
                   field_read_interval=h.get("field_read_interval"), run=h.get("run"), work=h.get("work"), scanner=h.get("scanner"))


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
