"""The coefficient walk: a walk that kept the bands a pack stores instead of its saves.

``simulate_trajectories(accumulate=K)`` returns one. Its walkers' paths are the exact start and end and the sine bands
of the bridge between them, accumulated while the walk ran (:mod:`dmipy_sim.engine.accumulate`); the contact channel
(C2) is the cumulative local time in the same form; the occupancy (C1) is one label per walker on a substrate whose
walkers cannot change pool, else the per-block track. :func:`~dmipy_sim.replay.bank.build_replay_pack` takes it as it
is. It is not a :class:`~dmipy_sim.persistent_walk.PersistentWalk`: it holds no positions but the ``raw_subset`` it
was asked to keep beside the coefficients.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CoefficientWalk:
    """One walk kept as its bridge bands.

    Attributes
    ----------
    position_coeffs : (n_walkers, K + 2, 3) float32
        Per axis ``[x(0), x(T) - x(0), beta_1..beta_K]``: the exact start, the exact displacement and the bands on
        :attr:`basis`.
    T : float
        The walk's duration (s).
    K : int
        Bands per axis and channel.
    basis : str
        ``"continuous"`` (``m >= 1``: the projections on ``s_k = sqrt(2 / T) sin(pi k t / T)``) or ``"dst1"`` (``m =
        0``: the block moments are the DST-I bands of the walk at its block ends, exactly).
    m : int
        The order of the block table (:func:`~dmipy_sim.replay.compression.block_table`).
    block : dict
        ``{h, m, n_blocks, rule, precision, k0_residual_m, quadratic_variation}``: the block length (s), the order, the
        block count, whether the block was derived from the band or given, how the accumulation was summed, the
        float32 chain's measured residual on the ``k = 0`` mode against the exact displacement (the largest over the
        walkers, metres) and the channels' mean quadratic variation.
    dt_sim : float
        The physics step (s).
    sub_steps : int
        Steps per block.
    boundary_start, boundary_endpoint : (n_walkers,) float32 or None
        The cumulative boundary local time at ``rho2 / D = 1`` at the start (0) and the end, exact.
    boundary_bands : (n_walkers, K) float32 or None
        Its bridge bands on :attr:`basis`.
    compartment : (n_walkers,) int8, (n_walkers, n_blocks + 1) or None
        The pool label per walker (a substrate no walker leaves its pool in), else the per-block track (sample 0 the
        start; a permeable wall's fractional occupancy as float32, a label as int8).
    bound_frac : (n_walkers, n_blocks + 1) or None
        The bound fraction per block of a binding (MT) walk.
    qv_positions : (n_walkers, 3) float32
        Each walker's quadratic variation per axis, ``sum (dx)^2`` over every step (``2 D T`` per axis for a free
        walk): what the block table's error is proportional to, :func:`~dmipy_sim.replay.compression.quadrature_rel`
        times it.
    qv_boundary : (n_walkers,) float32 or None
        The cumulative local time's quadratic variation, ``sum (dlog)^2`` over the steps (each hit's contact squared).
    raw_positions : (n_raw, n_blocks + 1, 3) float32 or None
        The first ``n_raw`` walkers' positions at the block ends, kept with ``raw_subset=`` for a measured
        certificate (:func:`~dmipy_sim.replay.bank.build_replay_pack` ``fidelity="measured"``).
    """
    position_coeffs: np.ndarray
    T: float
    K: int
    basis: str
    m: int
    block: dict
    dt_sim: float
    sub_steps: int
    boundary_start: Optional[np.ndarray] = None
    boundary_endpoint: Optional[np.ndarray] = None
    boundary_bands: Optional[np.ndarray] = None
    compartment: Optional[np.ndarray] = None
    bound_frac: Optional[np.ndarray] = None
    raw_positions: Optional[np.ndarray] = None
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
    def n_blocks(self):
        return int(self.block["n_blocks"])

    @property
    def n_t(self):
        """The block ends, ``n_blocks + 1``: the grid the occupancy and a raw subset are on."""
        return self.n_blocks + 1

    @property
    def dt(self):
        """The block length ``h`` (s)."""
        return float(self.block["h"])

    @property
    def T_max(self):
        return float(self.T)

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
        return int(sum(np.asarray(a).nbytes for a in (self.position_coeffs, self.boundary_start, self.boundary_endpoint,
                                                       self.boundary_bands, self.compartment, self.bound_frac,
                                                       self.raw_positions, self.qv_positions, self.qv_boundary) if a is not None))
