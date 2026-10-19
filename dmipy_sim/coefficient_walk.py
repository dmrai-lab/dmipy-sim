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
        The pool label per walker (a substrate no walker leaves its pool in), else the per-block track (sample 0 the
        start; a permeable wall's fractional occupancy as float32, a label as int8).
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
        return int(sum(np.asarray(a).nbytes for a in (self.position_coeffs, self.boundary_endpoint, self.boundary_bands,
                                                       self.compartment, self.bound_frac, self.field_samples,
                                                       self.raw_positions, self.raw_boundary, self.raw_compartment,
                                                       self.qv_positions, self.qv_boundary) if a is not None))

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
