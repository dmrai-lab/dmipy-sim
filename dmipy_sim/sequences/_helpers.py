"""Pure waveform / timing helpers for the physical sequence constructors.

Vendored from dmipy_fit.core.acquisition_scheme so the constructors live in
dmipy-sim (the forward truth).  All gyromagnetic references use dmipy-sim's own
GAMMA (= 267.513e6 rad/s/T, identical to the fit value), so b-values match the
fit implementation bit-for-bit.
"""
import numpy as np
from warnings import warn

from ..constants import GAMMA

_TE_FLOOR_ATOL = 1e-9
_REFOCUS_ATOL = 1e-3


def _trap_profile(t, start, delta, eps):
    """Unit trapezoid amplitude (0..1) sampled at times ``t`` (s).

    Ramps 0->1 over ``eps``, holds, ramps 1->0 over ``eps``, ramp MIDPOINTS at
    ``start`` and ``start + delta`` (the half-amplitude width is ``delta``; the
    lobe physically spans ``delta + eps``).  ``eps <= 0`` gives a rectangle.
    """
    if eps <= 0:
        return ((t >= start) & (t < start + delta)).astype(np.float64)
    a = np.zeros_like(t, dtype=np.float64)
    up = (t >= start) & (t < start + eps)
    a[up] = (t[up] - start) / eps
    flat = (t >= start + eps) & (t < start + delta)
    a[flat] = 1.0
    dn = (t >= start + delta) & (t < start + delta + eps)
    a[dn] = 1.0 - (t[dn] - (start + delta)) / eps
    return a


def _trap_cosine_profile(t, sigma, f, slew, g_mag):
    """Trapezoidal (flat-top, slew-limited) cosine-OGSE amplitude, T/m.

    Triangle carrier aligned with the cosine, scaled so its slope equals the slew
    rate, clipped at +/- g_mag; leading/trailing ramps taper to zero (Drobnjak
    2016; for N=1 this is one +/- pair, i.e. PGSE).
    """
    P = 1.0 / f
    phi = (f * t) % 1.0
    tri = 1.0 - 4.0 * np.minimum(phi, 1.0 - phi)        # +1 at peak, -1 at trough
    trap = np.clip((slew * P / 4.0) * tri, -g_mag, g_mag)
    ramp = g_mag / slew
    env = np.clip(t / ramp, 0.0, 1.0) * np.clip((sigma - t) / ramp, 0.0, 1.0)
    env = np.where((t >= 0) & (t < sigma), env, 0.0)
    return trap * env


def _refocusing_residual(G, dt):
    """Relative net gradient moment ``max|q(TE)| / max|q|`` for one measurement."""
    G = np.asarray(G, dtype=np.float64)
    q = np.cumsum(G * dt, axis=0)
    qmax = float(np.max(np.abs(q)))
    if qmax <= 0.0:
        return 0.0
    return float(np.max(np.abs(q[-1]))) / qmax


def _calc_b_from_waveform(G, dt):
    """b = gamma^2 ∫ |q(t)|^2 dt with q(t) = gamma ∫ G dt'.  Shapes (n_m,n_t,3)->(n_m,)."""
    G_f64 = np.asarray(G, dtype=np.float64)
    q = np.cumsum(G_f64 * dt, axis=1) * GAMMA   # (n_m, n_t, 3) rad/m
    q_sq = np.sum(q ** 2, axis=2)               # (n_m, n_t)
    b = np.trapezoid(q_sq, dx=dt, axis=1)       # (n_m,) s/m^2
    return b.astype(np.float64)


def _btensor_from_waveform(G, dt):
    """B_ij = gamma^2 ∫ q_i q_j dt for each measurement.  (n_m,n_t,3) -> (n_m,3,3)."""
    G = np.asarray(G, dtype=np.float64)
    q = np.cumsum(G * dt, axis=1) * GAMMA       # (n_m, n_t, 3) rad/m
    return np.einsum('mti,mtj->mij', q, q) * dt


def _resolve_te(TE, t_total_min, n_m):
    """Resolve echo time against the minimum echo time; (TE_arr, was_auto)."""
    if TE is None:
        return np.full(n_m, float(t_total_min)), True
    TE_arr = np.broadcast_to(np.asarray(TE, dtype=np.float64), (n_m,)).copy()
    if np.any(TE_arr < t_total_min - _TE_FLOOR_ATOL):
        raise ValueError(
            "Echo time TE = {:.3f} ms is below the minimum echo time "
            "{:.3f} ms set by the gradient schedule; the echo cannot form "
            "before the encoding completes.".format(
                float(np.min(TE_arr)) * 1e3, float(t_total_min) * 1e3))
    return TE_arr, False


def unify_length_reference_delta_Delta(reference_array, delta, Delta, TE):
    """Broadcast scalar delta/Delta/TE to arrays the length of reference_array."""
    if delta is None:
        delta_ = delta
    elif isinstance(delta, (float, int)):
        delta_ = np.tile(delta, len(reference_array))
    else:
        delta_ = delta.copy()
    if Delta is None:
        Delta_ = Delta
    elif isinstance(Delta, (float, int)):
        Delta_ = np.tile(Delta, len(reference_array))
    else:
        Delta_ = Delta.copy()
    if TE is None:
        TE_ = TE
    elif isinstance(TE, (float, int)):
        TE_ = np.tile(TE, len(reference_array))
    else:
        TE_ = TE.copy()
    return delta_, Delta_, TE_


def check_acquisition_scheme(bqg_values, gradient_directions, delta, Delta, TE):
    "Check the validity of the input parameters."
    if bqg_values.ndim > 1:
        raise ValueError(
            "b/q/G input must be a one-dimensional array. Currently its "
            "dimensions is {}.".format(bqg_values.ndim))
    if len(bqg_values) != len(gradient_directions):
        raise ValueError(
            "b/q/G input and gradient_directions must have the same length. "
            "Currently their lengths are {} and {}.".format(
                len(bqg_values), len(gradient_directions)))
    if delta is not None:
        if len(bqg_values) != len(delta):
            raise ValueError(
                "b/q/G input and delta must have the same length. Currently "
                "their lengths are {} and {}.".format(len(bqg_values), len(delta)))
        if delta.ndim > 1:
            raise ValueError(
                "delta must be one-dimensional array. Currently its dimension "
                "is {}".format(delta.ndim))
        if np.min(delta) < 0:
            raise ValueError(
                "delta must be zero or positive. Currently its minimum value "
                "is {}.".format(np.min(delta)))
    if Delta is not None:
        if len(bqg_values) != len(Delta):
            raise ValueError(
                "b/q/G input and Delta must have the same length. Currently "
                "their lengths are {} and {}.".format(len(bqg_values), len(Delta)))
        if Delta.ndim > 1:
            raise ValueError(
                "Delta must be one-dimensional array. Currently its dimension "
                "is {}.".format(Delta.ndim))
        if np.min(Delta) < 0:
            raise ValueError(
                "Delta must be zero or positive. Currently its minimum value "
                "is {}.".format(np.min(Delta)))
    if gradient_directions.ndim != 2 or gradient_directions.shape[1] != 3:
        raise ValueError(
            "gradient_directions n must be two dimensional array of shape "
            "[N, 3]. Currently its shape is {}.".format(gradient_directions.shape))
    if np.min(bqg_values) < 0.:
        raise ValueError(
            "b/q/G input must be zero or positive. Minimum value found is "
            "{}.".format(bqg_values.min()))
    gradient_norms = np.linalg.norm(gradient_directions, axis=1)
    zero_norms = gradient_norms == 0.
    if not np.all(abs(gradient_norms[~zero_norms] - 1.) < 0.001):
        raise ValueError("gradient orientations n are not unit vectors. ")
    if TE is not None and len(TE) != len(bqg_values):
        pass  # (matches the fit reference: message built but not raised)
    if TE is not None:
        te_min = np.min(TE)
        te_max = np.max(TE)
        if te_min < 0.005:
            warn("TE minimum value {:.4f} s is below 5 ms. TE must be given in "
                 "seconds. Did you accidentally provide TE in milliseconds?"
                 .format(te_min), UserWarning)
        if te_max > 0.500:
            warn("TE maximum value {:.4f} s exceeds 500 ms. TE must be given in "
                 "seconds. Did you accidentally provide TE in milliseconds?"
                 .format(te_max), UserWarning)
