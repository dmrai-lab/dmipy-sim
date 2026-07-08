"""SH convolution for orientation distributions in dmipy-sim.

Theory
------
For any axially-symmetric substrate (cylinder, myelinated cylinder, packed
cylinders along z) the single-fiber signal E(theta) depends only on the polar
angle theta between the gradient waveform axis and the fiber axis.  This means
E(theta) has only m=0 spherical harmonic (zonal/Legendre) terms:

    E(theta) = sum_{l=0,2,...,lmax}  f_l * P_l(cos theta)

where P_l are Legendre polynomials and f_l are the rotational harmonic (RH)
coefficients of the fiber response.

The ensemble-averaged signal for an orientation distribution ODF(n) is:

    E_total = integral  E(n) * ODF(n)  dn
            = sum_l  (4*pi / (2*l+1))  *  f_l  *  c_l^0

where c_l^0 is the m=0 (zonal) SH coefficient of the ODF in the real
symmetric SH convention (Tournier/dipy descoteaux, same as dmipy-core).

SH convention (same as dmipy-core / Tournier / MRtrix):
    Y_l^0(theta, phi) = sqrt((2l+1)/(4*pi)) * P_l(cos theta)
    Y_0^0 = 1/(2*sqrt(pi))
    c_0^0 of any normalised PDF = 1/(2*sqrt(pi))

Compact even-order SH array layout (m=0 indices):
    l=0: start=0, m=0 at index 0
    l=2: start=1, m=0 at index 3   (5 coefficients, m=-2,-1,0,1,2)
    l=4: start=6, m=0 at index 10  (9 coefficients)
    l=6: start=15, m=0 at index 21 (13 coefficients)
    l=8: start=28, m=0 at index 36 (17 coefficients)
    General: M=l//2; start = M*(2M-1) for M>0, else 0; m0_idx = start + l

References
----------
Kaden, Knosche & Anwander (2007), NeuroImage. (SH convolution theory)
dmipy-core dmipy/utils/spherical_convolution.py (validated convention)
"""

from __future__ import annotations

import numpy as np
from .waveforms import rotate_waveform
from scipy.special import eval_legendre, roots_legendre


# ---------------------------------------------------------------------------
# Helper: m=0 index in compact even-order SH array
# ---------------------------------------------------------------------------

def _m0_idx(l):
    """Return index of (l, m=0) in compact even-order real SH array.

    Array layout: l=0 has 1 coeff; l=2 has 5; l=4 has 9; ...
    start_of_l = M*(2M-1) where M = l//2 (with M=0 → start=0)
    m=0 offset within order l block = l
    """
    M = l // 2
    start = M * (2 * M - 1) if M > 0 else 0
    return start + l


def _n_sh_coeffs(lmax):
    """Total number of coefficients in compact even-order SH array up to lmax."""
    return sum(2 * l + 1 for l in range(0, lmax + 1, 2))


# ---------------------------------------------------------------------------
# Waveform rotation by polar angle (about y-axis)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Fiber response function via MC simulation + Legendre fit
# ---------------------------------------------------------------------------

def compute_fiber_response(geometry, acquisition_scheme, n_walkers,
                            lmax=8, seed=0, diffusivity=None, T2=None):
    """Simulate single-fiber signal at n_angles polar angles and fit Legendre.

    Uses Gauss-Legendre quadrature nodes mapped to [0, 1] (cos theta in [0,1],
    upper hemisphere) for the numerical integration.

    For each angle theta_k the acquisition waveform is rotated so that the
    gradient direction makes angle theta_k with the fiber axis (z).  Simulation
    is run with the fiber fixed along z and the waveform rotated.

    Legendre coefficients f_l are fit via least squares to the n_angles signal
    values:  E(theta_k) = sum_{l=0,2,...,lmax} f_l * P_l(cos theta_k)

    Parameters
    ----------
    geometry : dmipy-sim geometry
        Any geometry oriented along z (Cylinder, MyelinatedCylinder, etc.).
    acquisition_scheme : Waveform or object with .waveform attribute
        Waveform to simulate.
    n_walkers : int
        Walkers per polar angle.
    lmax : int
        Maximum even SH order (default 8).  Must be even.
    seed : int
        Base random seed; angle k gets seed+k.

    Returns
    -------
    fiber_response : np.ndarray, shape (lmax//2+1, n_measurements)
        Legendre RH coefficients f_l for l=0,2,...,lmax.
    thetas : np.ndarray, shape (n_angles,)
        Polar angles used (radians).
    E_theta : np.ndarray, shape (n_angles, n_measurements)
        Raw simulated signals at each angle (for diagnostics / residuals).
    """
    from .core import simulate

    if lmax % 2 != 0:
        raise ValueError(f"lmax must be even, got {lmax}")

    n_orders = lmax // 2 + 1   # number of even orders: 0, 2, ..., lmax
    # Use 2*n_orders GL points on [-1,1] for full orthogonality
    # (GL with n points integrates polynomials of degree 2n-1 exactly)
    n_angles = 2 * n_orders

    # Unwrap AcquisitionScheme to Waveform if needed
    if hasattr(acquisition_scheme, 'waveform'):
        waveform = acquisition_scheme.waveform
    else:
        waveform = acquisition_scheme

    n_measurements = waveform.G.shape[0]

    # Gauss-Legendre nodes on [-1, 1]
    # x = cos(theta); for cylinder, E(x) = E(-x) (antipodal symmetry).
    # We only need to simulate at unique |x| values (upper hemisphere).
    gl_nodes, gl_weights = roots_legendre(n_angles)   # nodes in [-1, 1]

    # Due to E(x) = E(-x), nodes come in ±x pairs. Exploit symmetry:
    # simulate only the positive-x half (theta in [0, pi/2])
    # and set E at negative-x nodes to the same value.
    cos_thetas_unique = gl_nodes[n_angles // 2:]   # x > 0 (upper half)
    thetas_unique = np.arccos(cos_thetas_unique)    # theta in (0, pi/2)

    # Get D from geometry if available (for standard geometries)
    # Caller-supplied diffusivity takes precedence
    D_val = diffusivity if diffusivity is not None else getattr(geometry, '_D', None)
    T2_val = T2

    # Simulate at each unique angle
    E_unique = np.zeros((n_angles // 2, n_measurements), dtype=np.float64)
    for k, theta in enumerate(thetas_unique):
        waveform_k = rotate_waveform(waveform, theta=theta)
        sig = simulate(
            n_walkers=n_walkers,
            waveform=waveform_k,
            geometry=geometry,
            seed=seed + k,
            diffusivity=D_val,
            T2=T2_val,
        )
        E_unique[k] = np.asarray(sig, dtype=np.float64)

    # Construct E at all GL nodes: E[-x] = E[x]
    # gl_nodes is sorted ascending; negative nodes come first
    E_all = np.concatenate([E_unique[::-1], E_unique], axis=0)  # shape (n_angles, n_meas)
    thetas = np.arccos(gl_nodes)  # for return value

    # Compute Legendre coefficients via GL quadrature (exact for polynomials ≤ 2n-1):
    # f_l = (2l+1)/2 * sum_k w_k E(x_k) P_l(x_k)
    fiber_response = np.zeros((n_orders, n_measurements), dtype=np.float64)
    for j in range(n_orders):
        l = 2 * j
        P_l_vals = eval_legendre(l, gl_nodes)  # shape (n_angles,)
        # (2l+1)/2 * integral_{-1}^1 E(x) P_l(x) dx ≈ (2l+1)/2 * sum_k w_k E_k P_l(x_k)
        fiber_response[j] = ((2 * l + 1) / 2.0) * np.dot(
            gl_weights, E_all * P_l_vals[:, None])  # dot broadcasts correctly

    # E_theta for diagnostics: return the unique angles (upper hemisphere)
    return fiber_response, thetas_unique, E_unique


# ---------------------------------------------------------------------------
# Apply ODF via SH convolution
# ---------------------------------------------------------------------------

def apply_odf(fiber_response, odf_sh, lmax=8):
    """Convolve fiber response with ODF via SH convolution formula.

    Computes:
        E = sum_l  (4*pi / (2*l+1))  *  f_l  *  c_l^0

    where f_l = fiber_response[l//2] is the l-th Legendre (RH) coefficient
    and c_l^0 is the m=0 SH coefficient of the ODF at order l.

    This formula follows from:
        E = integral E(theta(n)) * ODF(n) dn
    with E(theta) = sum_l f_l P_l(cos theta) and ODF in real SH basis.

    Parameters
    ----------
    fiber_response : np.ndarray, shape (lmax//2+1, n_measurements)
        From compute_fiber_response().
    odf_sh : np.ndarray or None
        Real SH coefficients in compact even-order representation.
        None → isotropic (returns fiber_response[0], the l=0 Legendre coeff
        which is the solid-angle average).
    lmax : int

    Returns
    -------
    signal : np.ndarray, shape (n_measurements,)
    """
    if odf_sh is None:
        # Isotropic: return the l=0 Legendre coefficient
        # (which equals the solid-angle-weighted average of E(theta))
        return np.asarray(fiber_response[0], dtype=np.float64)

    odf_sh = np.asarray(odf_sh, dtype=np.float64)
    n_orders = lmax // 2 + 1

    if fiber_response.ndim == 1:
        signal = 0.0
    else:
        signal = np.zeros(fiber_response.shape[1], dtype=np.float64)

    for j in range(n_orders):
        l = 2 * j
        idx = _m0_idx(l)
        if idx >= len(odf_sh):
            break

        c_l0 = odf_sh[idx]
        factor = np.sqrt((4.0 * np.pi) / (2.0 * l + 1.0))
        signal = signal + factor * fiber_response[j] * c_l0

    return signal


# ---------------------------------------------------------------------------
# Watson ODF SH coefficients
# ---------------------------------------------------------------------------

def watson_odf_sh(kappa, mu=None, lmax=8):
    """Compute real SH coefficients for a Watson distribution.

    Delegates to dmipy.utils.sh_analytical.watson_sh, which uses an exact
    erfi-based recurrence (no quadrature, no hyp1f1 overflow issues) and
    supports arbitrary mean orientation mu.

    The output layout (compact even-order real SH, Tournier convention) is
    identical to the one used by apply_odf / compute_fiber_response.

    Parameters
    ----------
    kappa : float
        Concentration parameter.  kappa=0 → isotropic; kappa→inf → single fiber.
    mu : np.ndarray, shape (3,), optional
        Mean orientation as a unit Cartesian vector.  Default: z=[0,0,1].
    lmax : int
        Maximum even SH order.

    Returns
    -------
    odf_sh : np.ndarray, shape (n_sh_coeffs,)
        Compact even-order SH coefficients.
    """
    from .math.sh_analytical import watson_sh

    if mu is None:
        mu = np.array([0., 0., 1.], dtype=np.float64)
    return watson_sh(np.asarray(mu, dtype=np.float64), float(kappa), l_max=lmax)


def isotropic_odf_sh(lmax=8):
    """SH coefficients for uniform sphere (only l=0 term non-zero).

    For ODF(n) = 1/(4*pi):
        c_0^0 = integral (1/(4*pi)) * Y_0^0 dn = Y_0^0 = 1/(2*sqrt(pi))
        All higher-order coefficients = 0.

    Parameters
    ----------
    lmax : int

    Returns
    -------
    odf_sh : np.ndarray
    """
    odf_sh = np.zeros(_n_sh_coeffs(lmax), dtype=np.float64)
    odf_sh[0] = 1.0 / (2.0 * np.sqrt(np.pi))
    return odf_sh
