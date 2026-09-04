"""Magnetization transfer (MT): microscopic bound-pool physics + analytic oracle.

This module holds the *host-side* MT physics that the Monte-Carlo walk and the
vector-Bloch forward simulation build on, plus the analytic two-pool
Bloch--McConnell oracle used to validate them.

Design: MT is modelled the way Cottaar's MCMRSimulator models it -- **emergently,
from the random walk** -- but the binding event uses *this* codebase's impact-angle
boundary-local-time rule, NOT an angle-blind per-collision count.  A free (liquid)
spin that hits a reactive surface **sticks** with a probability set by the
penetration depth ``d_perp`` of that hit:

    p_stick = min(1, 2 * (kappa_MT / D) * d_perp)                         (1)

exactly the structural form the engine already uses for permeability
(``p_transmit = min(1, 2 (kappa/D) d_perp)``) and surface relaxivity
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
``1/T2_surf = rho * (S/V)``.  These relations are the conversion between the
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

__all__ = [
    "stick_probability",
    "forward_rate", "backward_rate", "bound_fraction",
    "kappa_MT_from_forward_rate", "dwell_time_from_fraction",
    "surface_to_volume", "resolve_equilibrate_mode", "equilibrate_burnin_plateau",
    "two_pool_generator", "evolve_two_pool",
    "bloch_mcconnell_transverse", "bloch_mcconnell_longitudinal",
    "mt_z_spectrum",
]


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
    # identical to `_boundary.transmit_probability`; kept as a host-side (numpy) entry
    # point for the analytic oracles, which run off-device.
    return np.minimum(1.0, 2.0 * (kappa_MT / D) * np.asarray(d_perp, dtype=float))


# ── (kappa_MT, dwell_time) <-> (f_bound, k) conversions ─────────────────────────
def forward_rate(kappa_MT, S_over_V):
    """Forward (free->bound) exchange rate k_f = kappa_MT * (S/V), s^-1."""
    return float(kappa_MT) * float(S_over_V)


def backward_rate(dwell_time):
    """Backward (bound->free) exchange rate k_r = 1/dwell_time, s^-1."""
    if dwell_time <= 0.0:
        raise ValueError(f"dwell_time must be > 0, got {dwell_time}")
    return 1.0 / float(dwell_time)


def bound_fraction(kappa_MT, dwell_time, S_over_V):
    """Equilibrium bound-pool fraction f_b = k_f / (k_f + k_r)."""
    kf = forward_rate(kappa_MT, S_over_V)
    kr = backward_rate(dwell_time)
    return kf / (kf + kr) if (kf + kr) > 0 else 0.0


def kappa_MT_from_forward_rate(k_f, S_over_V):
    """Invert k_f = kappa_MT * (S/V) -> kappa_MT (m/s)."""
    if S_over_V <= 0.0:
        raise ValueError(f"S_over_V must be > 0, got {S_over_V}")
    return float(k_f) / float(S_over_V)


def dwell_time_from_fraction(f_bound, k_f):
    """Dwell time for a target equilibrium bound fraction at forward rate k_f.

    From detailed balance k_f*(1-f_b) relation: k_r = k_f*(1-f_b)/f_b, so
    dwell_time = 1/k_r = f_b / (k_f * (1 - f_b)).
    """
    if not (0.0 < f_bound < 1.0):
        raise ValueError(f"f_bound must be in (0,1), got {f_bound}")
    if k_f <= 0.0:
        raise ValueError(f"k_f must be > 0, got {k_f}")
    k_r = k_f * (1.0 - f_bound) / f_bound
    return 1.0 / k_r


# ── bound-pool equilibration for the MT walk generator ──────────────────────────
# Shared by ``mt_walk.simulate_mt_trajectories`` (and the packed-myelin MT walk in
# ``core.simulate_trajectories``).  The saved walk carries no RF/gradient (those are
# replay knobs), so equilibration is purely of the bound-pool OCCUPANCY (and the
# equilibrated spatial state): an all-free start under-fills the macromolecular pool
# and biases the transfer whenever ``1/k_f`` is not ``<<`` the walk duration.  These
# mirror the forward engine's ``bloch._surface_to_volume`` / ``_resolve_equilibrate_mode``
# / ``_equilibrate_burnin`` but need no G/T2_bound 'fast'-safety demotion (no readout yet).
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


def resolve_equilibrate_mode(equilibrate_binding, geometry):
    """Map ``equilibrate_binding`` -> 'off' | 'burnin' | 'fast' for MT walk generation.

    'auto' (the default when MT is on) -> 'burnin' (safe, geometry-agnostic).  'fast'
    needs a known S/V, else it warns and falls back to 'burnin'.
    """
    import warnings
    eb = equilibrate_binding
    if eb in (None, False, "off"):
        return "off"
    if eb in ("auto", True, "burnin"):
        return "burnin"
    if eb == "fast":
        if surface_to_volume(geometry) is None:
            warnings.warn("equilibrate_binding='fast' needs a known surface-to-volume "
                          "(analytic Sphere/Cylinder); falling back to 'burnin'.", stacklevel=2)
            return "burnin"
        return "fast"
    raise ValueError(f"equilibrate_binding must be 'auto'|'burnin'|'fast'|'off', got {eb!r}")


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


# ── analytic two-pool Bloch--McConnell oracle ───────────────────────────────────
# State vector x = [Mxa, Mya, Mza, Mxb, Myb, Mzb, 1] (augmented; last row keeps M0
# recovery affine).  Free pool a, bound pool b.  dx/dt = A x.  Everything constant
# over an interval -> exact via matrix exponential (scipy.linalg.expm).
def two_pool_generator(*, R1a, R2a, R1b, R2b, k_f, k_r,
                       M0a=1.0, M0b=None, dw_a=0.0, dw_b=0.0,
                       w1=0.0, rf_phase=0.0):
    """Build the 7x7 generator A of the two-pool Bloch--McConnell system.

    Rates in s^-1, angular frequencies (dw_a, dw_b, w1) in rad/s.  ``w1 = gamma*B1``
    is a continuous RF nutation rate; ``rf_phase`` (rad) sets the B1 axis in the
    transverse plane (0 = +x).  If ``M0b`` is None it is set from detailed balance
    ``M0b = M0a * k_f / k_r`` (so the given rates and pool sizes are consistent).

    Conventions (standard Bloch): precession dM/dt = M x (0,0,dw) -> dMx=-dw My,
    dMy=+dw Mx; RF about axis phi at rate w1 -> rotation about (cos phi, sin phi, 0).
    """
    if M0b is None:
        M0b = M0a * (k_f / k_r) if k_r > 0 else 0.0
    A = np.zeros((7, 7), dtype=float)
    ux, uy = np.cos(rf_phase), np.sin(rf_phase)

    def _block(base, R1, R2, dw, M0):
        ix, iy, iz = base, base + 1, base + 2
        # relaxation
        A[ix, ix] += -R2
        A[iy, iy] += -R2
        A[iz, iz] += -R1
        A[iz, 6] += R1 * M0                 # affine +R1*M0 recovery
        # off-resonance precession about z
        A[ix, iy] += -dw
        A[iy, ix] += +dw
        # RF nutation about axis (ux, uy, 0) at rate w1:  dM/dt = w1 * (u x M)
        # u x M = (uy*Mz - 0, 0 - ux*Mz, ux*My - uy*Mx)
        A[ix, iz] += w1 * uy
        A[iy, iz] += -w1 * ux
        A[iz, ix] += -w1 * uy
        A[iz, iy] += w1 * ux

    _block(0, R1a, R2a, dw_a, M0a)
    _block(3, R1b, R2b, dw_b, M0b)

    # exchange a <-> b (applies to all three components of each pool)
    for c in range(3):
        ia, ib = c, c + 3
        A[ia, ia] += -k_f
        A[ia, ib] += k_r
        A[ib, ib] += -k_r
        A[ib, ia] += k_f
    return A


def evolve_two_pool(state0, t, A):
    """Evolve the augmented state ``state0`` (7,) by time ``t`` under generator A."""
    from scipy.linalg import expm
    state0 = np.asarray(state0, dtype=float)
    return expm(A * float(t)) @ state0


def _equilibrium_state(M0a, M0b):
    return np.array([0, 0, M0a, 0, 0, M0b, 1.0], dtype=float)


def bloch_mcconnell_transverse(t, *, T2a, T2b, k_f, k_r, T1a=1.0, T1b=1.0,
                               M0a=1.0, M0b=None, dw_a=0.0, dw_b=0.0):
    """Free-pool transverse magnitude |Mxy,a|(t) after a 90 on both pools (no RF).

    Both pools start fully transverse (Mx=M0).  Returns |Mxa+iMya| at each ``t``.
    In the limit T2b -> 0 with dw=0 this reduces to exp(-(1/T2a + k_f) t) --
    the classic result that the forward exchange rate adds to the free-pool R2.
    """
    if M0b is None:
        M0b = M0a * (k_f / k_r) if k_r > 0 else 0.0
    A = two_pool_generator(R1a=1.0 / T1a, R2a=1.0 / T2a, R1b=1.0 / T1b,
                           R2b=1.0 / T2b, k_f=k_f, k_r=k_r, M0a=M0a, M0b=M0b,
                           dw_a=dw_a, dw_b=dw_b)
    s0 = np.array([M0a, 0, 0, M0b, 0, 0, 1.0], dtype=float)   # 90 -> transverse
    ts = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty(ts.shape, dtype=float)
    for i, tt in enumerate(ts):
        s = evolve_two_pool(s0, tt, A)
        out[i] = np.hypot(s[0], s[1])
    return out if out.size > 1 else float(out[0])


def bloch_mcconnell_longitudinal(t, *, T1a, T1b, k_f, k_r, M0a=1.0, M0b=None,
                                 Mza0=None, Mzb0=None):
    """Free-pool longitudinal Mza(t) under exchange (no RF, no transverse).

    Default initial condition inverts BOTH pools (Mz = -M0), i.e. an inversion
    recovery with exchange.  Pass ``Mza0``/``Mzb0`` for other preparations (e.g.
    saturate the bound pool: Mzb0=0, Mza0=M0a).
    """
    if M0b is None:
        M0b = M0a * (k_f / k_r) if k_r > 0 else 0.0
    if Mza0 is None:
        Mza0 = -M0a
    if Mzb0 is None:
        Mzb0 = -M0b
    A = two_pool_generator(R1a=1.0 / T1a, R2a=1e-9, R1b=1.0 / T1b, R2b=1e-9,
                           k_f=k_f, k_r=k_r, M0a=M0a, M0b=M0b)
    s0 = np.array([0, 0, Mza0, 0, 0, Mzb0, 1.0], dtype=float)
    ts = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty(ts.shape, dtype=float)
    for i, tt in enumerate(ts):
        out[i] = evolve_two_pool(s0, tt, A)[2]
    return out if out.size > 1 else float(out[0])


def mt_z_spectrum(offsets_hz, *, w1_hz, t_sat, T1a, T2a, T1b, T2b, k_f, k_r,
                  M0a=1.0, M0b=None, read_pool="a"):
    """Steady-ish Z-spectrum: free-pool Mz after a CW saturation pulse vs offset.

    A continuous-wave RF of nutation ``w1_hz`` (Hz, = gamma*B1/2pi) and duration
    ``t_sat`` is applied at each frequency ``offset`` (Hz) from the free-pool
    resonance; returns Mz of the read pool (default free ``a``) normalised to M0.
    The bound pool's short T2b makes it saturate over a broad offset range while
    the free pool saturates only near resonance -- the MT dip.  (No analytic
    lineshape is assumed; the bound pool is a real short-T2b spin, matching the MC.)
    """
    if M0b is None:
        M0b = M0a * (k_f / k_r) if k_r > 0 else 0.0
    offsets = np.atleast_1d(np.asarray(offsets_hz, dtype=float))
    w1 = 2.0 * np.pi * float(w1_hz)
    out = np.empty(offsets.shape, dtype=float)
    idx = 2 if read_pool == "a" else 5
    norm = M0a if read_pool == "a" else M0b
    for i, off in enumerate(offsets):
        dw = 2.0 * np.pi * off          # RF offset seen by both pools (same nucleus)
        A = two_pool_generator(R1a=1.0 / T1a, R2a=1.0 / T2a, R1b=1.0 / T1b,
                               R2b=1.0 / T2b, k_f=k_f, k_r=k_r, M0a=M0a, M0b=M0b,
                               dw_a=dw, dw_b=dw, w1=w1)
        s = evolve_two_pool(_equilibrium_state(M0a, M0b), t_sat, A)
        out[i] = s[idx] / norm if norm > 0 else 0.0
    return out if out.size > 1 else float(out[0])
