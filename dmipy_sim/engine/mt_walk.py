"""Magnetization-transfer random walk: the binding (sticking) physics, as a producer of the one coefficient walk.

The walker physics MT needs is the binding at a wall: the engine's per-hit boundary-local-time channel
(``interact(r, step, rho_over_D=1)`` returns ``dlog = -2 * Sum d_perp``) gives the stick decision at each step,

    p_stick = min(1, (kappa_MT / D) * (-dlog)) = min(1, 2 (kappa_MT/D) Sum d_perp)

the impact-angle rule of :func:`dmipy_sim.engine.mt.stick_probability`, consuming the SAME local-time quantity the
surface relaxivity uses (so it is timestep-independent by construction, no Cottaar ``sqrt(dt)`` factor). Because
relaxivity gives ``-d ln S/dt = rho * (S/V)`` from that channel, the emergent free->bound rate is ``k_f = kappa_MT *
(S/V)`` -- validated in ``tests/test_mt_binding.py``. A stuck walker FREEZES (stops diffusing) for an
exponentially-distributed dwell of mean ``dwell_time`` (drawn as ``-log(u) * dwell_time / dt`` sub-steps, matching
the continuum residence time), then is released. The bound state is carried in the walker like a compartment id, and
the walk records every bind and unbind as an EVENT exact in time (``CoefficientWalk.bound_events``, RPK.md 9.5.7)
beside the free pool's boundary local time (for surface relaxivity of the free pool at replay).

That step is the ``"mt"`` kind of :func:`dmipy_sim.engine.core._walk_geometry` (``kappa_MT > 0`` on a geometry
with walls); this module is the named entry point. The bound spins' relaxation / RF (the actual MT saturation
transfer) is applied at REPLAY by ``ReplayPack.replay_bloch``, consuming the bound occupancy
read off the events -- positions while bound are irrelevant to the MT observables (a bound spin has ~zero transverse
via its huge R2b). This is the replay counterpart of the forward vector-Bloch oracle ``bloch._simulate_bloch(kappa_MT=...)``.
"""
from __future__ import annotations
import logging

log = logging.getLogger(__name__)


def simulate_mt_trajectories(
    n_walkers: int,
    diffusivity: float,
    geometry,
    T_max: float,
    block_h: float = None,
    kappa_MT: float = 0.0,
    dwell_time: float = 0.0,
    seed: int = 42,
    r0=None,
    walker_batch_size: int = 50_000,
    sub_steps: int = None,
    equilibrate_binding="burnin",
    require_gpu=None,
    **walk_kwargs,
):
    """The coefficient walk with MT surface binding: :func:`~dmipy_sim.engine.core._walk_geometry` with
    ``kappa_MT`` and ``dwell_time``, on a geometry with walls (Sphere / Cylinder / Box1D / Ellipsoid / Mesh; a mesh
    with ``mt_side`` binds through ``reflect_with_binding``). At ``kappa_MT = 0`` this is the plain walk of
    :func:`~dmipy_sim.engine.core._walk_geometry` -- same random stream, same bands to the bit -- so an MT pack and a
    plain pack of one substrate differ only by what binding did.

    Parameters
    ----------
    n_walkers, diffusivity, geometry, T_max, block_h, seed, r0, walker_batch_size, sub_steps, require_gpu :
        As in :func:`dmipy_sim.engine.core._walk_geometry`. ``r0`` seeds the pool wanted when it is not the
        geometry's inside (a bundle's extra-axonal pool, whose geometry is the OUTER surface).
    kappa_MT : float
        MT surface reactivity (m/s).  0 disables binding (pure diffusion walk).
    dwell_time : float
        Mean bound-pool residence time (s).  Must be > 0 when kappa_MT > 0.
    equilibrate_binding : {'burnin', 'fast', 'off'}
        How the bound pool reaches its thermal-equilibrium occupancy BEFORE t=0;
        see :func:`dmipy_sim.engine.mt.resolve_equilibrate_mode`.
    walk_kwargs :
        The producer's other keywords (``K``, ``block_order``, ``window_T``, ``raw_subset``, ``events_capacity``, ...).

    Returns
    -------
    CoefficientWalk
        With ``bound_start`` / ``bound_events`` (the bound column) and the free pool's contact channel when
        ``kappa_MT > 0``.
    """
    from .core import _walk_geometry
    from ..spec.build import as_geometry
    geometry = as_geometry(geometry)
    if kappa_MT > 0.0 and dwell_time <= 0.0:
        raise ValueError("dwell_time must be > 0 when kappa_MT > 0.")
    if not geometry.has_walls:
        raise TypeError(
            f"{type(geometry).__name__} has no walls; MT binding "
            "needs the boundary-local-time channel (Sphere/Cylinder/Box1D/Ellipsoid/Mesh).")
    return _walk_geometry(n_walkers, diffusivity, geometry, T_max, block_h, seed=seed, r0=r0,
                                 walker_batch_size=walker_batch_size, sub_steps=sub_steps, require_gpu=require_gpu,
                                 kappa_MT=float(kappa_MT), dwell_time=float(dwell_time), equilibrate_binding=equilibrate_binding,
                                 **walk_kwargs)
