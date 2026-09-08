# -*- coding: utf-8 -*-
"""Exact spherical-harmonic coefficients of the orientation distributions.

**This is the ecosystem's home for these forms.** The dependency runs one way -- fit and design
import sim, and sim imports neither -- so the base mathematics of the Watson and Bingham
distributions lives here and is written once, in the layer everything else can reach. A consumer's
own parameterisation (fit's ODI and psi, for instance) is that consumer's business; only the
mathematics is shared. Nothing here imports dipy: the normative basis is this package's
:func:`dmipy_sim.replay.so3.real_sh`, pinned to dipy's non-legacy ``tournier`` by an oracle test
(``test_required_basis_is_dipy_tournier_non_legacy``), so the claim is verified without a runtime
dependency on it.

For W(n; mu, kappa) ~ exp(kappa (n.mu)^2), the Tournier real SH coefficients are

    c_l^m = Y_l^m(mu) * J_l(kappa) / J_0(kappa)

with the zonal ratios r_l = J_l(kappa)/J_0(kappa) computed by an exact
erfi-based recurrence (no quadrature, no hyp1f1 overflow).

The Bingham is the two-concentration generalisation: its azimuthal integral is analytic in
modified Bessel functions and only the polar integral is quadrature, so its coefficients are exact
to machine precision in the same sense.

References
----------
Kaden E, Knosche TR, Anwander A (2007). Parametric spherical deconvolution.
NeuroImage 37(2):474-488.
"""
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import erfi, iv as bessel_iv, lpmv

__all__ = ['watson_zonal_ratios', 'watson_sh', 'bingham_normalization', 'bingham_canonical_sh',
           'bingham_sh']


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
    # Evaluated with this package's own basis rather than dipy's. RPH.md fixes the normative basis
    # as dipy's non-legacy `tournier`, and `so3.real_sh` IS that basis -- pinned to it at atol=1e-12
    # by test_required_basis_is_dipy_tournier_non_legacy. Calling dipy here would make it a runtime
    # dependency for a function we already implement; keeping dipy as the test ORACLE and our own
    # implementation in the package is the right split, and it keeps dipy a dev extra.
    from ..replay.so3 import real_sh

    mu_cart = np.asarray(mu_cart, dtype=np.float64)
    Y_mu = real_sh(l_max, (mu_cart / np.linalg.norm(mu_cart))[None, :])[0]
    r = watson_zonal_ratios(kappa, l_max)

    n_coef = (l_max + 1) * (l_max + 2) // 2
    r_per_coef = np.empty(n_coef)
    counter = 0
    for order in range(0, l_max + 1, 2):
        n_in_order = 2 * order + 1
        r_per_coef[counter:counter + n_in_order] = r[order // 2]
        counter += n_in_order

    return (Y_mu * r_per_coef).astype(np.float64)


# ------------------------------------------------------------------ Bingham
# Parameterisation: a frame and one concentration about each of its first two axes, so the pose axis
# is the frame's third column and larger concentrations are tighter. This matches
# `so3.Distribution.bingham` and `phantom.BinghamField`, which is where it is consumed.
# fit's `(mu, psi, kappa, beta)` maps in as kappa = k1, beta = k1 - k2, mu = the pose axis and
# mu_beta = the frame's second column.
def bingham_normalization(k1, k2):
    r"""Partition function of ``exp(-k1 (n.e1)^2 - k2 (n.e2)^2)`` over the sphere.

    Writing the exponent in the polar angle about the pose axis, the azimuthal integral is a
    modified Bessel function and what is left is a smooth one-dimensional integral, taken here on
    32 Gauss-Legendre nodes -- machine precision over any physical concentration.
    """
    k1, k2 = float(k1), float(k2)
    nodes, weights = leggauss(32)
    s2 = 1.0 - nodes ** 2                                   # sin^2 of the polar angle
    A = -(k1 + k2) / 2.0 * s2
    B = (k2 - k1) / 2.0 * s2
    return 2.0 * np.pi * float(np.dot(weights, np.exp(A) * bessel_iv(0, B)))


def bingham_canonical_sh(k1, k2, l_max=8):
    r"""Coefficients of the Bingham whose pose axis is ``z`` and whose concentrations are about
    ``x`` and ``y``: the canonical frame, before any rotation.

    The exponent is ``A(t) + B(t) cos(2 phi)`` with ``t`` the cosine of the polar angle, so the
    azimuthal integral closes analytically --- ``\int_0^{2\pi} e^{A + B\cos 2\phi}\cos(2q\phi)
    d\phi = 2\pi e^{A} I_q(B)``, and the odd and sine terms vanish --- leaving one Gauss-Legendre
    integral per coefficient. Only even ``l`` and even ``m >= 0`` survive, and ``c_0^0`` is exactly
    ``1/(2\sqrt\pi)``.
    """
    k1, k2 = float(k1), float(k2)
    nodes, weights = leggauss(32)
    s2 = 1.0 - nodes ** 2
    A = -(k1 + k2) / 2.0 * s2
    B = (k2 - k1) / 2.0 * s2
    exp_A = np.exp(A)
    Z = 2.0 * np.pi * float(np.dot(weights, exp_A * bessel_iv(0, B)))

    n_coef = (l_max + 1) * (l_max + 2) // 2
    sh = np.zeros(n_coef)
    counter = 0
    for l in range(0, l_max + 1, 2):
        n_in_order = 2 * l + 1
        for m_block in range(n_in_order):
            m = m_block - l
            if m < 0 or m % 2 != 0:                         # sine terms and odd m are identically 0
                continue
            Iq = bessel_iv(m // 2, B)
            if m == 0:
                N_lm = math.sqrt((2 * l + 1) / (4.0 * math.pi))
            else:
                N_lm = math.sqrt(2.0 * (2 * l + 1) / (4.0 * math.pi)
                                 * math.factorial(l - m) / float(math.factorial(l + m)))
            sh[counter + m_block] = (N_lm * 2.0 * np.pi
                                     * float(np.dot(weights, exp_A * Iq * lpmv(m, l, nodes))) / Z)
        counter += n_in_order
    return sh


def bingham_sh(frame, kappa, l_max=8):
    r"""Coefficients of a Bingham with a stated frame: the canonical form rotated.

    ``frame`` is a rotation whose third column is the pose axis and whose first two columns carry
    the two concentrations ``kappa = (k1, k2)``. Equal concentrations give the Watson of that
    concentration, so this contains :func:`watson_sh` as a special case; the rotation uses the
    harmonic rotation blocks of :func:`dmipy_sim.replay.so3.wigner_blocks`, which is the ecosystem's
    one implementation of them.
    """
    from ..replay.so3 import wigner_blocks

    R = np.asarray(frame, np.float64).reshape(3, 3)
    k1, k2 = (float(kappa[0]), float(kappa[1])) if np.ndim(kappa) else (float(kappa), float(kappa))
    c = bingham_canonical_sh(k1, k2, l_max)
    blocks = wigner_blocks(l_max, R[None])
    out, counter = np.zeros_like(c), 0
    for l in range(0, l_max + 1, 2):
        n_in_order = 2 * l + 1
        out[counter:counter + n_in_order] = blocks[l][0] @ c[counter:counter + n_in_order]
        counter += n_in_order
    return out
