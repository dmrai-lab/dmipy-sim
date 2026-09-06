"""The persistent walk: the microphysics a trajectory producer keeps, for any acquisition to replay.

A fused engine walks and returns a signal; nothing of the walk survives the call. The replay
engine walks once and keeps the walk -- positions and the boundary, compartment and binding
channels -- so every acquisition, relaxation and exchange hypothesis is applied afterwards without
walking again. `simulate_trajectories` and `simulate_mt_trajectories` return a `PersistentWalk`;
its channels are attributes, present or ``None`` by what the walk recorded; the replay functions
read the attributes they need and the bank builds a pack from `PersistentWalk.bank_dict`.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PersistentWalk:
    """One field-independent walk kept at ``dt`` granularity, the persistent half of the replay
    invariant: positions depend only on geometry, diffusivity and seed, never on the acquisition.

    Attributes
    ----------
    positions : (n_walkers, n_t, 3) in the storage dtype
        Walker positions in metres at each saved step; ``positions.dtype`` is the storage dtype.
    dt : float
        Saved time step, ``T_max / (n_t - 1)``.
    sub_steps : int
        Physics sub-steps per saved step (:func:`physics.resolve_sub_steps`).
    dt_sim : float
        The physics step, ``dt / sub_steps``.
    boundary_local_time : (n_walkers, n_t) or None
        Per-step boundary log-weight at ``rho/D = 1`` (``-2 * sum d_perp`` over the step's wall
        hits, non-positive); the surface-relaxivity channel. ``None`` when not recorded.
    compartment : (n_walkers, n_t) or None
        Compartment id per saved step (0 extra-cellular, positive enclosed pools; the ``.rpk``
        convention). A float array is a fractional occupancy over the step.
    bound_frac : (n_walkers, n_t) or None
        Bound-pool occupancy per saved step of a binding (MT) walk.
    illegal_crossings : int
        Walker-steps that ended on the far side of a membrane without a granted crossing and were
        rejected back into their compartment. Non-zero means the engine relabelled walkers.
    seed : int or None
        The producer's master seed.
    """
    positions: np.ndarray
    dt: float
    sub_steps: int
    dt_sim: float
    boundary_local_time: Optional[np.ndarray] = None
    compartment: Optional[np.ndarray] = None
    bound_frac: Optional[np.ndarray] = None
    illegal_crossings: int = 0
    seed: Optional[int] = None

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
        return self.bound_frac is not None

    def bank_dict(self, **extra):
        """The dict :func:`bank.build_replay_pack` reads: ``traj``, ``dt_traj``, ``T_max``,
        ``comp``, ``dlog_b``, ``bfrac``, ``n_walkers``, ``seed``, plus any substrate metadata
        passed as ``extra`` (``T2_per_comp``, ``w``, ``substrate_frame``, ...)."""
        out = dict(traj=self.positions, dt_traj=float(self.dt), T_max=self.T_max,
                   comp=self.compartment, dlog_b=self.boundary_local_time, bfrac=self.bound_frac,
                   n_walkers=self.n_walkers, seed=0 if self.seed is None else int(self.seed))
        out.update(extra)
        return out
