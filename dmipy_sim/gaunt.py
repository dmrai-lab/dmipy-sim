"""Gaunt coefficients for composing replay packs over an orientation distribution.

Theory (paper Sec. 2.6)
-----------------------
A pack is one substrate at one pose; a voxel is a distribution of poses.  Because a
substrate rotation is ``G(t) -> U^T G(t)`` *together with* ``B0 -> U^T B0``, the gradient
and field axes move as one, and the composition integral is not a spherical convolution
once susceptibility makes the response read the fibre direction along a second axis.

For an axially symmetric substrate the response depends on the fibre direction ``n`` only
through the two invariants ``n.g`` and ``n.B0``, so it expands as

    E(n; g, B0) = sum_{l1,l2} Lambda_{l1 l2} P_l1(n.g) P_l2(n.B0),

with ``Lambda`` the *coupled angular spectrum*.  Separability is exactly the rank-one case
``Lambda = a (x) b``.  The addition theorem then separates the tissue direction from the two
acquisition directions and leaves the signal linear in the orientation distribution,

    S = sum_LM H_LM f_LM,
    H_LM = sum_{l1 l2} (4pi)^2 Lambda_{l1l2} / ((2l1+1)(2l2+1))
           * sum_{m1 m2} G^{LM}_{l1m1,l2m2} Y_{l1m1}(g) Y_{l2m2}(B0),

where ``G`` are the Gaunt coefficients ``\\int Y_LM Y_l1m1 Y_l2m2 dn``.

Why the basis is built here rather than imported
------------------------------------------------
The addition theorem holds only for an **orthonormal** basis, and several widely used real
spherical-harmonic conventions are not: ``dipy.reconst.shm.real_sh_tournier`` defaults to
``legacy=True``, whose ``m != 0`` functions have norm ``1/sqrt(2)``.  The error that
introduces **vanishes exactly when g is parallel to B0** -- the one configuration a cursory
test would use -- so it is the kind of mistake that survives a test suite.  This module
therefore builds its own basis and *asserts* orthonormality where the table is built
(:func:`assert_orthonormal`), rather than trusting a flag.  The Condon-Shortley phase is
irrelevant here: every consumer of the table uses the same basis it was built from, and the
addition theorem is a sum over ``m`` that is invariant to a per-``(l, m)`` sign.

Layout matches :mod:`dmipy_sim.sh_convolution`: even orders only, compact, ``m = -l..+l``
within each block.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, lpmv
from numpy.polynomial.legendre import leggauss

__all__ = ["real_sh", "sphere_quadrature", "assert_orthonormal", "gaunt_table",
           "n_sh_coeffs", "sh_block"]


def n_sh_coeffs(lmax):
    """Number of coefficients in the compact even-order layout up to ``lmax``."""
    return sum(2 * l + 1 for l in range(0, lmax + 1, 2))


def sh_block(l):
    """Slice of the compact even-order array holding order ``l`` (``m = -l..+l``)."""
    M = l // 2
    start = M * (2 * M - 1) if M > 0 else 0
    return slice(start, start + 2 * l + 1)


def real_sh(lmax, dirs):
    """Orthonormal real spherical harmonics, even orders, compact layout.

    ``dirs`` is ``(n, 3)`` unit Cartesian; returns ``(n, n_sh_coeffs(lmax))``.
    """
    d = np.asarray(dirs, np.float64).reshape(-1, 3)
    x = np.clip(d[:, 2], -1.0, 1.0)
    phi = np.arctan2(d[:, 1], d[:, 0])
    out = np.empty((d.shape[0], n_sh_coeffs(lmax)), np.float64)
    for l in range(0, lmax + 1, 2):
        blk = sh_block(l)
        col = out[:, blk]
        col[:, l] = np.sqrt((2 * l + 1) / (4 * np.pi)) * lpmv(0, l, x)
        for m in range(1, l + 1):
            # K_lm = sqrt((2l+1)/4pi * (l-m)!/(l+m)!), via gammaln for large l
            K = np.sqrt((2 * l + 1) / (4 * np.pi)
                        * np.exp(gammaln(l - m + 1) - gammaln(l + m + 1)))
            P = lpmv(m, l, x)
            col[:, l + m] = np.sqrt(2.0) * K * P * np.cos(m * phi)
            col[:, l - m] = np.sqrt(2.0) * K * P * np.sin(m * phi)
    return out


def sphere_quadrature(n_theta, n_phi):
    """Gauss-Legendre (cos theta) x uniform (phi) product rule; weights sum to 4 pi.

    Exact for spherical polynomials of degree < 2*n_theta in cos(theta) and < n_phi in phi,
    which is what makes the triple-product table below exact rather than approximate.
    """
    x, wx = leggauss(n_theta)
    phi = np.arange(n_phi) * (2.0 * np.pi / n_phi)
    st = np.sqrt(np.clip(1.0 - x ** 2, 0.0, 1.0))
    dirs = np.stack([np.outer(st, np.cos(phi)), np.outer(st, np.sin(phi)),
                     np.outer(x, np.ones(n_phi))], axis=-1).reshape(-1, 3)
    w = (wx[:, None] * (2.0 * np.pi / n_phi) * np.ones(n_phi)[None, :]).reshape(-1)
    return dirs, w


def assert_orthonormal(Y, w, atol=1e-10):
    """Raise if ``Y`` is not orthonormal under the quadrature ``w``.

    This is the guardrail the derivation requires, not a sanity check: the Gaunt route
    goes through the addition theorem, which is false for a non-orthonormal basis.
    """
    gram = np.einsum("qa,qb,q->ab", Y, Y, w)
    err = np.abs(gram - np.eye(gram.shape[0])).max()
    if err > atol:
        bad = np.abs(np.diag(gram) - 1.0)
        raise ValueError(
            f"spherical-harmonic basis is not orthonormal under this quadrature: "
            f"max|Gram - I| = {err:.3e} (tol {atol:g}); worst diagonal deviation "
            f"{bad.max():.3e}. The Gaunt contraction is only valid for an orthonormal "
            f"basis -- see the module docstring.")
    return err


_CACHE = {}


def gaunt_table(l_fod, l_g, l_b, n_theta=None, n_phi=None, atol=1e-10):
    """``G[a, b, c] = \\int Y_a Y_b Y_c dn`` for the three truncation orders.

    Indices ``a``, ``b``, ``c`` run over the compact even-order layouts of ``l_fod``,
    ``l_g`` and ``l_b`` respectively.  Built by exact quadrature in the basis of
    :func:`real_sh`, whose orthonormality is asserted first.  The table depends only on the
    harmonics -- not on substrate, acquisition or field -- so it is cached per order triple.

    The angular-momentum selection rules (``|l1-l2| <= L <= l1+l2``, ``l1+l2+L`` even,
    ``m`` additive) leave it sparse; they are not imposed here, they simply come out.
    """
    key = (int(l_fod), int(l_g), int(l_b), n_theta, n_phi)
    if key in _CACHE:
        return _CACHE[key]
    L = max(l_fod, l_g, l_b)
    # degree of the integrand is l_fod + l_g + l_b; the rule must be exact for it
    deg = l_fod + l_g + l_b
    nt = int(n_theta or (deg // 2 + 2))
    npx = int(n_phi or (2 * deg + 4))
    dirs, w = sphere_quadrature(nt, npx)
    Y = real_sh(L, dirs)
    assert_orthonormal(Y, w, atol=atol)
    G = np.einsum("qa,qb,qc,q->abc",
                  Y[:, :n_sh_coeffs(l_fod)], Y[:, :n_sh_coeffs(l_g)],
                  Y[:, :n_sh_coeffs(l_b)], w, optimize=True)
    G[np.abs(G) < 1e-12] = 0.0
    G.flags.writeable = False
    _CACHE[key] = G
    return G
