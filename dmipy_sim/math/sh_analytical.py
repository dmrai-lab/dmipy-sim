# -*- coding: utf-8 -*-
"""Exact analytical Watson-ODF spherical harmonics.

Vendored into dmipy-sim from dmipy_fit.utils.sh_analytical (the watson_* subset
used by sh_convolution) so the sim package is self-contained.  The numerics are
identical to the fit implementation.

For W(n; mu, kappa) ~ exp(kappa (n.mu)^2), the Tournier real SH coefficients are

    c_l^m = Y_l^m(mu) * J_l(kappa) / J_0(kappa)

with the zonal ratios r_l = J_l(kappa)/J_0(kappa) computed by an exact
erfi-based recurrence (no quadrature, no hyp1f1 overflow).

References
----------
Kaden E, Knosche TR, Anwander A (2007). Parametric spherical deconvolution.
NeuroImage 37(2):474-488.
"""
import numpy as np
from scipy.special import erfi

__all__ = ['watson_zonal_ratios', 'watson_sh']


def cart2sphere(cartesian_coordinates):
    """Spherical coordinates [r, theta, phi] from cartesian [x, y, z].

    range of theta [0, pi], range of phi [-pi, pi]; dipy notation.  Vendored
    from dmipy_fit.utils.utils.cart2sphere.
    """
    cartesian_coordinates = np.asarray(cartesian_coordinates)
    if np.ndim(cartesian_coordinates) == 1:
        x, y, z = cartesian_coordinates
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r) if r > 0 else 0.0
        phi = np.arctan2(y, x)
        return np.r_[r, theta, phi]
    elif np.ndim(cartesian_coordinates) == 2:
        x, y, z = cartesian_coordinates.T
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.where(r > 0, np.arccos(z / r), 0.)
        phi = np.arctan2(y, x)
        return np.c_[r, theta, phi]
    raise ValueError("coordinates must be array of size 3 or N x 3.")


def watson_zonal_ratios(kappa, l_max=8):
    r"""Exact zonal harmonic ratios r_l = J_l(kappa) / J_0(kappa) for even l.

    For a Watson ODF aligned on z the SH coefficients are
    c_l^0 = sqrt((2l+1)/(4 pi)) * r_l.  Only even orders are non-zero.

    For kappa > 700 the direct recurrence overflows float64; the saddle-point
    asymptotic r_l ~ 1 - l(l+1)/(4 kappa) is used (rel. error O(1/kappa^2)).
    """
    n_levels = l_max // 2 + 1
    r = np.zeros(n_levels)
    r[0] = 1.0

    if kappa < 1e-12:
        return r

    if kappa > 700.0:
        for m in range(1, n_levels):
            l = 2 * m
            r[m] = 1.0 - l * (l + 1) / (4.0 * kappa)
        return r

    sqrt_k = np.sqrt(kappa)
    J0 = np.sqrt(np.pi) * erfi(sqrt_k) / sqrt_k

    J = np.empty(n_levels)
    J[0] = J0
    K_curr = np.exp(kappa) / kappa - J0 / (2.0 * kappa)  # K_1

    for m in range(n_levels - 1):
        J[m + 1] = ((4 * m + 3) * K_curr - (2 * m + 1) * J[m]) / (2 * m + 2)
        K_curr = K_curr - (4 * m + 5) * J[m + 1] / (2.0 * kappa)

    return J / J0


def watson_sh(mu_cart, kappa, l_max=8):
    r"""Exact Tournier real SH coefficients of a Watson ODF.

    c_l^m = Y_l^m(mu) * r_l  with r_l = J_l(kappa)/J_0(kappa).  The l=0
    coefficient is exactly 1/(2 sqrt(pi)) for any kappa.

    Returns SH coefficients in Tournier (MRtrix) real ordering, legacy=False,
    shape ((l_max+1)(l_max+2)//2,).
    """
    from dipy.reconst.shm import real_sh_tournier

    mu_cart = np.asarray(mu_cart, dtype=np.float64)
    sph = cart2sphere(mu_cart)          # [r, theta, phi]
    theta_mu, phi_mu = float(sph[1]), float(sph[2])

    Y_mu = real_sh_tournier(l_max, theta_mu, phi_mu, legacy=False)[0][0]
    r = watson_zonal_ratios(kappa, l_max)

    n_coef = (l_max + 1) * (l_max + 2) // 2
    r_per_coef = np.empty(n_coef)
    counter = 0
    for order in range(0, l_max + 1, 2):
        n_in_order = 2 * order + 1
        r_per_coef[counter:counter + n_in_order] = r[order // 2]
        counter += n_in_order

    return (Y_mu * r_per_coef).astype(np.float64)
