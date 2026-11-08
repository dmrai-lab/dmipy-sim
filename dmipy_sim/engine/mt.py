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

from typing import NamedTuple

import numpy as np

__all__ = ["stick_probability", "refuse_unsupported_permeable_mt", "surface_to_volume", "resolve_equilibrate_mode", "equilibrate_burnin_plateau",
           "equilibrium_bound_fraction", "burnin_criterion", "BurnInCriterion"]


# ── per-hit binding probability (the binding clock on the local time) ─────────────
def stick_probability(d_perp, kappa_MT, D):
    """Probability a free spin sticks at a wall hit of penetration ``d_perp``.

    ``p = 1 - exp(-2 (kappa_MT / D) d_perp)``: the binding clock, of rate ``kappa_MT / D`` per unit boundary local
    time, ringing within the hit's local time ``2 d_perp`` (:func:`~dmipy_sim.geometry._boundary.bind_probability`,
    the walk's own, whose bind carries its geminate rebinds in the dwell, ``dwell_scale``). ``d_perp`` (m) is the
    perpendicular penetration of the residual step past the surface (``remaining * cos_alpha`` in the engine), so a
    grazing hit sticks less than a normal one. Works on scalars or numpy arrays.

    Parameters
    ----------
    d_perp : float or array   penetration depth of the hit (m), >= 0.
    kappa_MT : float          MT surface reactivity (m/s), >= 0.
    D : float                 local diffusivity (m^2/s), > 0.
    """
    # a host-side (numpy) entry point for the analytic oracles, which run off-device.
    return -np.expm1(-2.0 * (kappa_MT / D) * np.asarray(d_perp, dtype=float))


def refuse_unsupported_permeable_mt(geometry, where, *, schedule=True):
    """Refuse the permeable geometries the MT walk does not walk, by name; walk the rest.

    The MT walk binds at a permeable wall by the cross-first rule: at each encounter that touches a wall the crossing
    is decided first (the transmission rule ``p = x / (1 + x)``), then the walker, on the face it ends on, binds with
    ``p = 1 - exp(-kappa_MT / D * l)`` for the encounter's local time ``l = 2 d_perp``, whether it crossed or was
    reflected, and freezes at the step's end for a dwell that carries the encounter's geminate rebinds
    (dmrai-lab/dmipy-sim#909, #921). It walks every geometry with walls and a
    permeability of one number, one per object, or a :class:`~dmipy_sim.geometry.base.PermeabilitySchedule` (a
    geometry that carries the walker's side carries it through the walk, and the burn-in that equilibrates the bound
    pool before t = 0 reads the first interval of a schedule). The forward vector-Bloch walk takes a permeability
    constant in time, as its walk without binding does (``schedule=False``).

    Raises
    ------
    NotImplementedError
        For a Mesh with a permeable wall (its crossing directions are not walked with binding), and, with
        ``schedule=False``, for a permeability that varies in time; ``where`` names the entry point refusing.
    """
    if geometry._is_packed_myelinated:
        return
    k = getattr(geometry, "permeability", None)
    if k is None or not float(k) > 0.0:
        return
    table = geometry.permeability_table
    why = None
    if not schedule and table is not None and np.asarray(table.breakpoints).size:
        why = "its permeability varies in time (a PermeabilitySchedule), which the vector-Bloch walk does not take"
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


BURNIN_TOLERANCE = 0.01
"""The burn-in's relative tolerance: the bound fraction is at equilibrium when it is within ``0.01 P_eq`` plus two
standard errors of the ensemble's mean (never less than the binomial error of ``P_eq`` in the ensemble's walkers); with no known equilibrium the occupancy has plateaued when the two halves of the
window differ by no more than ``0.01`` of the occupancy plus two standard errors of their difference."""

BURNIN_RELAXATIONS = 5.0
"""Relaxation times a burn-in with a known equilibrium runs at least: ``exp(-5) = 0.7 %`` of the all-free start's
deficit is left, below the standard error of any ensemble that resolves the bound fraction."""

BURNIN_WINDOW = 8
"""Chunks (one dwell each) of the plateau window when no equilibrium is known: at least eight exchange times, the
exchange time ``1 / (k_f + k_r)`` being at most the dwell."""

BURNIN_MIN_CHUNKS = 40
"""The burn-in's cap in chunks, raised to the minimum plus twenty where the relaxation time asks for more."""


def _first_permeabilities(geometry, n_objects):
    """The permeability (m/s) of each of the geometry's ``n_objects`` walls at the first interval of time."""
    table = geometry.permeability_table
    if table is not None:
        return np.broadcast_to(np.asarray(table.values[0], float), (n_objects,))
    k = geometry.permeability
    return np.full(n_objects, 0.0 if k is None else float(k))


def equilibrium_bound_fraction(geometry, kappa_MT, dwell_time):
    """The bound fraction of the walkers a geometry seeds once the walk has equilibrated, or ``None`` where the
    geometry has no closed form here.

    A free spin binds at a wall at the rate ``k_f = kappa_MT S / V`` per unit free time and a bound one leaves at
    ``k_r = 1 / dwell_time``, so the stationary bound fraction of a closed domain of surface ``S`` (every face a
    walker reaches, counted once per face) and volume ``V`` is ``P_eq = x / (1 + x)``, ``x = kappa_MT dwell_time S / V``
    (``dmipy_analytic.relaxation.bound_fraction``; with the bound pool on each face of area density
    ``kappa_MT dwell_time`` times the free water's, the continuum slab of ``BlochTorrey1D``). ``S`` and ``V`` are the
    geometry's own:

    * a sealed :class:`~dmipy_sim.geometry.Sphere` or :class:`~dmipy_sim.geometry.Cylinder`: ``S / V = 3 / R``, ``2 / R``;
    * a :class:`~dmipy_sim.geometry.PermeableSlab1D` whose membranes all let water through: both faces of each of its
      ``n`` membranes in the length ``L``, ``S / V = 2 n / L``;
    * :class:`~dmipy_sim.geometry.PackedCylinders` / :class:`~dmipy_sim.geometry.PackedSpheres` seeded over the whole
      cell, all walls permeable: both faces of every object in the cell, ``S / V = 2 surface_area() / L^d``
      (``d = 2`` per unit length, ``3``); all walls sealed: each pool at its own ``S / V`` (an object's interior, the
      interstitial space with its one face of every object) weighted by the walkers it is seeded with, its volume
      fraction.

    Mixed sealed and permeable walls, a pool seeded alone in a permeable pack, a Mesh and the rest: ``None``.
    """
    from ..geometry import PackedCylinders, PackedSpheres, PermeableSlab1D
    from ..geometry.analytic import Cylinder, Sphere
    kappa, tau = float(kappa_MT), float(dwell_time)
    p = lambda sv: (kappa * tau * sv) / (1.0 + kappa * tau * sv)
    if isinstance(geometry, (Sphere, Cylinder)):
        if _first_permeabilities(geometry, 1)[0] > 0.0:
            return None
        return p((3.0 if isinstance(geometry, Sphere) else 2.0) / float(geometry.radius))
    if isinstance(geometry, PermeableSlab1D):
        if np.all(_first_permeabilities(geometry, len(geometry.membranes)) > 0.0):
            return p(2.0 * len(geometry.membranes) / geometry.length)
        return None
    if isinstance(geometry, (PackedCylinders, PackedSpheres)) and geometry.pool is None:
        spheres = isinstance(geometry, PackedSpheres)
        radii = np.asarray(geometry._radii_np, float)
        cell = geometry._L_float ** (3 if spheres else 2)
        volume = geometry.volume()                                # per unit length for cylinders: the cell's area is L^2
        k = _first_permeabilities(geometry, radii.size)
        if np.all(k > 0.0):
            return p(2.0 * geometry.surface_area() / cell)
        if np.all(k == 0.0):
            inside = (3.0 if spheres else 2.0) / radii
            v_k = ((4.0 / 3.0) * np.pi * radii ** 3) if spheres else (np.pi * radii ** 2)
            outside = geometry.surface_area() / (cell - volume)
            return float(np.sum(v_k / cell * p(inside)) + (cell - volume) / cell * p(outside))
        return None
    return None


class BurnInCriterion(NamedTuple):
    """The stopping rule of the MT burn-in, as the walk records it (``CoefficientWalk.work["mt_equilibration"]``).

    ``kind`` is ``"equilibrium"`` (``p_eq`` known: stop when the ensemble's bound fraction is within ``tolerance
    * p_eq`` plus two standard errors of it, after at least ``min_chunks`` chunks of the relaxation time) or
    ``"plateau"`` (stop when the two halves of the last ``window`` chunks differ by no more than ``tolerance`` of the
    occupancy plus two standard errors of the difference). ``chunk_s`` is a chunk's duration, ``max_chunks`` the cap."""
    kind: str
    p_eq: float | None
    tolerance: float
    window: int
    min_chunks: int
    max_chunks: int
    chunk_s: float


def burnin_criterion(geometry, kappa_MT, dwell_time, diffusivity, chunk_s):
    """The :class:`BurnInCriterion` of a walk: the equilibrium of :func:`equilibrium_bound_fraction` where it exists,
    else the plateau. The relaxation time is the larger of the dwell (the exchange time is at most it) and the pore's
    slowest diffusive time ``1 / (D pore_rate)`` (``LengthScales.pore_rate``, a sphere of the smallest feature when the
    geometry declares none)."""
    from .physics import length_scales_of
    p_eq = equilibrium_bound_fraction(geometry, kappa_MT, dwell_time)
    if p_eq is None:
        return BurnInCriterion("plateau", None, BURNIN_TOLERANCE, BURNIN_WINDOW, BURNIN_WINDOW,
                               BURNIN_MIN_CHUNKS, float(chunk_s))
    ls = length_scales_of(geometry)
    rate = ls.pore_rate if ls.pore_rate is not None else (
        (2.0816 / ls.min_feature) ** 2 if ls.min_feature else 0.0)
    t_relax = max(float(dwell_time), 1.0 / (float(diffusivity) * rate) if rate > 0.0 else 0.0)
    n_min = max(1, int(np.ceil(BURNIN_RELAXATIONS * t_relax / float(chunk_s))))
    return BurnInCriterion("equilibrium", float(p_eq), BURNIN_TOLERANCE, 1, n_min,
                           max(BURNIN_MIN_CHUNKS, n_min + 20), float(chunk_s))


def equilibrate_burnin_plateau(chunk_fn, r0, keys, brem0, criterion):
    """Adaptive burn-in of the bound pool (geometry-agnostic), stopped by a stated :class:`BurnInCriterion`.

    ``chunk_fn(r, keys, brem)`` advances the whole ensemble one chunk (~one dwell) and returns ``(r, keys, brem,
    occupancy)``, ``occupancy`` the walkers' chunk-mean bound fractions. The preamble is discarded and the equilibrated
    ``(r, keys, brem, record)`` returned; ``record`` is the criterion, the chunks run, the occupancy and whether it
    held (``converged``)."""
    r, k, brem = r0, keys, brem0
    history, converged, occ, n_run = [], False, float("nan"), 0
    for n_run in range(1, int(criterion.max_chunks) + 1):
        r, k, brem, bf = chunk_fn(r, k, brem)
        bf = np.asarray(bf, np.float64)
        history.append(bf)
        occ = float(bf.mean())
        se = float(bf.std() / np.sqrt(bf.size))
        if criterion.kind == "equilibrium":
            # the ensemble of n walkers resolves P_eq to the binomial error at best, however few of them are bound
            se = max(se, float(np.sqrt(criterion.p_eq * (1.0 - criterion.p_eq) / bf.size)))
            if n_run >= criterion.min_chunks and abs(occ - criterion.p_eq) <= criterion.tolerance * criterion.p_eq + 2.0 * se:
                converged = True
                break
        elif n_run >= 2 * (criterion.window // 2):
            a = np.mean(history[-criterion.window:-criterion.window // 2], axis=0)
            b = np.mean(history[-criterion.window // 2:], axis=0)
            diff = float(b.mean() - a.mean())
            se_diff = float(np.hypot(a.std(), b.std()) / np.sqrt(bf.size))
            if abs(diff) <= criterion.tolerance * occ + 2.0 * se_diff:
                converged = True
                break
    record = dict(kind=criterion.kind, p_eq=criterion.p_eq, tolerance=criterion.tolerance, window=criterion.window,
                  min_chunks=criterion.min_chunks, max_chunks=criterion.max_chunks, chunk_s=criterion.chunk_s,
                  chunks_run=n_run, occupancy=occ, converged=converged)
    return r, k, brem, record


def __getattr__(name):
    from .._moved import refuse
    raise refuse(__name__, name)
