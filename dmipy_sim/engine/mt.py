"""Magnetization transfer (MT): the walk's microscopic bound-pool physics.

This module holds the *host-side* MT physics that the Monte-Carlo walk and the
vector-Bloch forward simulation build on. The two-region exchange relations and the
analytic two-pool Bloch--McConnell oracle they are validated against are
dmipy-analytic's (:mod:`dmipy_analytic.relaxation`).

Design: MT is modelled the way Cottaar's MCMRSimulator models it -- **emergently,
from the random walk** -- but the binding event uses *this* codebase's impact-angle
boundary-local-time rule, NOT an angle-blind per-collision count.  A free (liquid)
spin that hits a reactive surface **sticks** with a probability set by the
penetration depth ``d_perp`` of that hit:

    p_stick = min(1, 2 * (kappa_MT / D) * d_perp)                         (1)

the penetration-depth form the engine uses for permeability
(``p_transmit = x / (1 + x)``, ``x = 2 (kappa/D) d_perp``) and surface relaxivity
(``dlog_w = -2 (rho/D) d_perp``).  ``kappa_MT`` (m/s) is therefore a *surface
reactivity* -- a third member of the (rho, kappa, kappa_MT) reactive-velocity
family -- and it is consumed through the SAME per-hit ``d_perp`` channel, so it
inherits that channel's timestep-independence (``Sum 2 d_perp`` converges to the
boundary local time) with no separate ``sqrt(dt)`` factor.

A stuck spin (a) stops diffusing and (b) takes the bound pool's relaxation
(``T2_bound`` -- very short -- ``T1_bound``, ``off_resonance_bound``).  It is
released after an exponentially-distributed dwell time of mean ``dwell_time``.
RF pulses rotate bound spins too, so MT saturation and its transfer to the free
pool are emergent (no super-Lorentzian lineshape is hard-coded).

Two-region exchange bookkeeping (Brownstein--Tarr-style, S/V = surface-to-volume
ratio of the reactive surface):

    k_f = kappa_MT * (S/V)          forward (free -> bound) pseudo-first-order rate
    k_r = 1 / dwell_time            backward (bound -> free) rate
    f_bound = k_f / (k_f + k_r)     equilibrium bound-pool fraction

``k_f = kappa_MT * (S/V)`` is the exact analogue of the surface-relaxation rate
``1/T2_surf = rho * (S/V)``.  These relations (:func:`dmipy_analytic.relaxation.forward_rate`,
:func:`~dmipy_analytic.relaxation.bound_fraction`, ...) are the conversion between the
microscopic ``(kappa_MT, dwell_time)`` knobs and the macroscopic ``(f_bound, k)``
qMT parameterization.

References
----------
Henkelman RM et al. Magn Reson Med 1993;29:759 (two-pool MT).
Sled JG, Pike GB. J Magn Reson 2000/2001 (pulsed MT, super-Lorentzian).
Graham SJ, Henkelman RM. JMRI 1997;7:903 (pulsed MT).
Portnoy S, Stanisz GJ. Magn Reson Med 2007;58:144 (MT-in-simulation).
Stanisz GJ et al. Magn Reson Med 2005;54:507 (qMT tissue parameters).
Cottaar M et al. Imaging Neuroscience 2026, DOI 10.1162/IMAG.a.1177 (MCMRSimulator).
"""
from __future__ import annotations

import numpy as np

__all__ = ["stick_probability", "refuse_permeable_mt", "refuse_unsupported_permeable_mt", "surface_to_volume", "resolve_equilibrate_mode", "equilibrate_burnin_plateau"]


# ── per-hit binding probability (the impact-angle rule) ─────────────────────────
def stick_probability(d_perp, kappa_MT, D):
    """Probability a free spin sticks at a wall hit of penetration ``d_perp``.

    ``p = min(1, 2 * (kappa_MT / D) * d_perp)`` -- eq. (1).  ``d_perp`` (m) is the
    perpendicular penetration of the residual step past the surface (``remaining *
    cos_alpha`` in the engine), so a grazing hit sticks less than a normal one.
    Works on scalars or numpy arrays; the geometry step functions inline the same
    expression in pure JAX.

    Parameters
    ----------
    d_perp : float or array   penetration depth of the hit (m), >= 0.
    kappa_MT : float          MT surface reactivity (m/s), >= 0.
    D : float                 local diffusivity (m^2/s), > 0.
    """
    # a host-side (numpy) entry point for the analytic oracles, which run off-device.
    return np.minimum(1.0, 2.0 * (kappa_MT / D) * np.asarray(d_perp, dtype=float))


def refuse_permeable_mt(geometry, where):
    """Refuse a magnetization-transfer walk on a geometry whose walls ever let a walker cross.

    The walk that binds water at a wall (the impact-angle rule of :func:`stick_probability`) treats every wall as
    impermeable: it has no accounting of how binding and membrane transmission share one encounter. A geometry with
    a nonzero permeability -- one number, one per object, a :class:`~dmipy_sim.geometry.base.PermeabilitySchedule`, a
    Mesh's mapping of crossing directions -- is therefore refused by name (dmrai-lab/dmipy-sim#909, #910), never
    walked as an impermeable one. ``geometry.permeability`` is the largest permeability over objects, directions and
    time (``None`` for walls that never let a walker cross), so one test covers every form. The packed myelinated
    cylinders carry binding and membrane transmission in one step of their own and are not tested here.

    Raises
    ------
    NotImplementedError
        When ``geometry.permeability`` is not ``None``; ``where`` names the entry point refusing.
    """
    if geometry._is_packed_myelinated:
        return
    k = getattr(geometry, "permeability", None)
    if k is not None and float(k) > 0.0:
        raise NotImplementedError(
            f"{where}: magnetization transfer (kappa_MT > 0) on a permeable {type(geometry).__name__} "
            f"(permeability up to {float(k):.3g} m/s) is not supported: the MT walk binds water at a wall as if the "
            f"wall were impermeable and has no accounting of how binding and membrane transmission share an "
            f"encounter. Walk it impermeable (permeability=None) or with kappa_MT=0.0. "
            f"See dmrai-lab/dmipy-sim#909 and #910.")


def refuse_unsupported_permeable_mt(geometry, where):
    """Refuse the permeable geometries the MT walk does not yet walk, by name; walk the rest.

    The MT walk binds at a permeable wall by the cross-first rule: at each encounter that touches a wall the crossing
    is decided first (the transmission rule ``p = x / (1 + x)``), then the walker, on the face it ends on, binds with
    ``p = min(1, kappa_MT / D * l)`` for the encounter's local time ``l = 2 d_perp``, whether it crossed or was
    reflected, and freezes at the step's end (dmrai-lab/dmipy-sim#909). It walks a permeability constant in time, one
    number or one per object, on every geometry that does not carry a side. Refused by name: a
    :class:`~dmipy_sim.geometry.base.PermeabilitySchedule` (the burn-in that equilibrates the bound pool has no time to
    read it at), a geometry that carries the walker's side (the packed cylinders and spheres) and a Mesh's crossing
    directions.

    Raises
    ------
    NotImplementedError
        When the geometry is permeable in a way the walk does not carry; ``where`` names the entry point refusing.
    """
    if geometry._is_packed_myelinated:
        return
    k = getattr(geometry, "permeability", None)
    if k is None or not float(k) > 0.0:
        return
    table = geometry.permeability_table
    why = None
    if table is not None and np.asarray(table.breakpoints).size:
        why = "its permeability varies in time (a PermeabilitySchedule)"
    elif geometry.carries_side:
        why = "it carries the walker's side, which the MT walk does not yet carry"
    elif type(geometry).__name__ == "Mesh":
        why = "a Mesh's crossing directions are not yet walked with binding"
    if why is not None:
        raise NotImplementedError(
            f"{where}: magnetization transfer (kappa_MT > 0) on a permeable {type(geometry).__name__} (permeability "
            f"up to {float(k):.3g} m/s) is not supported: {why}. Walk it impermeable (permeability=None) or with "
            f"kappa_MT=0.0. See dmrai-lab/dmipy-sim#909.")


# ── bound-pool equilibration for the MT walk generator ──────────────────────────
# Shared by ``mt_walk.simulate_mt_trajectories`` (and the packed-myelin MT walk in
# ``core._walk_geometry``).  The saved walk carries no RF/gradient (those are
# replay knobs), so equilibration is purely of the bound-pool OCCUPANCY (and the
# equilibrated spatial state): an all-free start under-fills the macromolecular pool
# and biases the transfer whenever ``1/k_f`` is not ``<<`` the walk duration. The three
# helpers below are the ONE implementation; the forward vector-Bloch engine (``bloch``) and
# the MT trajectory producer (``mt_walk``) both read them.
def surface_to_volume(geometry):
    """S/V (1/m) for the analytic closed shapes used by the 'fast' equilibrium seed;
    None otherwise, so the caller falls back to the geometry-agnostic burn-in."""
    from .physics import _geometry_radius
    name = type(geometry).__name__
    R = _geometry_radius(geometry)
    if R is not None and R > 0.0:
        if name == "Sphere":
            return 3.0 / R
        if name == "Cylinder":
            return 2.0 / R                       # infinite cylinder, lateral wall
    return None


EQUILIBRATE_MODES = ("burnin", "fast", "off")


def resolve_equilibrate_mode(equilibrate_binding, geometry, *, G=None, T2=None, T2_bound=None):
    """Check ``equilibrate_binding`` -> 'burnin' | 'fast' | 'off' for every MT driver.

    Exactly ``'burnin'`` (the default when MT is on: safe, geometry-agnostic), ``'fast'`` or ``'off'``; every other
    value (``'auto'``, ``True``, ``False``, ``None``) is a ValueError naming the three. 'fast' seeds the equilibrium
    occupancy mid-air, so it needs a known S/V and is a ValueError without one. A driver with a readout passes
    ``G``, ``T2`` and ``T2_bound``: mid-air bound positions are position-invariant only when no gradient is on
    (``G == 0``) or the bound pool is MR-dark (``T2_bound << T2``), and 'fast' is a ValueError otherwise. A driver
    that only records positions (the MT trajectory producer) has no readout and passes none.
    """
    eb = equilibrate_binding
    if not isinstance(eb, str) or eb not in EQUILIBRATE_MODES:
        raise ValueError(f"equilibrate_binding must be exactly 'burnin' | 'fast' | 'off', got {eb!r}; "
                         f"'burnin' is the default, 'off' is spelled 'off' (not False or None), and 'auto' is 'burnin'")
    if eb == "fast":
        if surface_to_volume(geometry) is None:
            raise ValueError("equilibrate_binding='fast' needs a known surface-to-volume (an analytic Sphere or "
                             "Cylinder); this geometry has none: use 'burnin'")
        if G is not None:
            no_gradient = not bool(np.any(np.asarray(G)))
            dark = (T2 is None) or (T2_bound is not None and float(T2_bound) <= 0.02 * float(T2))
            if not (no_gradient or dark):
                raise ValueError("equilibrate_binding='fast' is unsafe here: a gradient is present and the bound pool "
                                 "is not MR-dark (T2_bound not << T2), so the mid-air bound positions would bias the "
                                 "signal: use 'burnin'")
    return eb


def equilibrate_burnin_plateau(chunk_fn, r0, keys, brem0, tol=0.01, max_chunks=40):
    """Adaptive occupancy-plateau burn-in (geometry-agnostic).

    ``chunk_fn(r, keys, brem)`` advances the whole ensemble one chunk (~one dwell) and
    returns ``(r, keys, brem, mean_occupancy)``; iterate until the ensemble bound
    occupancy stops changing (relative change < ``tol``), then DISCARD the preamble and
    return the equilibrated ``(r, keys, brem, occupancy, converged)``.
    """
    r, k, brem = r0, keys, brem0
    occ_prev, converged = -1.0, False
    for _ in range(int(max_chunks)):
        r, k, brem, occ = chunk_fn(r, k, brem)
        occ = float(occ)
        if occ_prev >= 0.0 and abs(occ - occ_prev) <= tol * max(occ, 1e-6):
            occ_prev, converged = occ, True
            break
        occ_prev = occ
    return r, k, brem, occ_prev, converged


def __getattr__(name):
    from .._moved import refuse
    raise refuse(__name__, name)
