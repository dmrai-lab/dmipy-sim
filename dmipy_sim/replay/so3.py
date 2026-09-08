"""The pose of a substrate is a rotation, and a voxel is a distribution of rotations.

A pack answers for one microstructure at one pose. Composing it into a voxel means integrating its response
over the poses that voxel holds, and the pose is a **rotation**, not an axis: fixing the direction a substrate
points along leaves its spin about that axis unstated, and a substrate's response depends on it unless the
substrate happens to be axially symmetric. A finite bundle of tortuous strands is not, and neither is a fanned
population with two dispersion parameters, so nothing here assumes it.

The representation is therefore the Fourier basis of SO(3). For one measurement the response is a function on
the group, expanded in the real Wigner functions, and the composition is the Peter-Weyl inner product:

    E(R) = sum_l sum_{m,n} c^l_{mn} phi^l_{mn}(R),      phi^l_{mn}(R) = sqrt(2l+1) D^l_{mn}(R)
    S    = int f(R) E(R) dR = sum_{l,m,n} c^l_{mn} f^l_{mn}          (normalised Haar measure)

One expansion per measurement, one dot product per voxel. Three consequences worth stating, because they are
what the axis-only representation could not give:

* **Any dispersion.** A single pose, a Watson, a Bingham with its own principal frame, a crossing -- each is a
  coefficient set. The response and the distribution are both free to be non-axisymmetric.
* **Roll is a declaration, not an averaging loop.** A population with no preferred azimuth has coefficients
  only at ``n = 0``, so the roll integral is exact and costs nothing. An orientation distribution over
  directions alone is exactly that case.
* **No second acquisition axis appears.** The response is expanded in ``R``, so the field direction enters only
  through the response values themselves -- there is no ``g x B0`` axis to become degenerate when the gradient
  and the field are parallel.

Truncation is rectangular, ``l <= lmax`` and ``|n| <= nmax``, and is meant to be **measured**: :func:`fit`
reports the energy per ``l`` and per ``n``, so ``nmax`` states how non-axisymmetric a substrate is rather than
hiding it.
"""
from __future__ import annotations

import functools

import numpy as np

from .gaunt import n_sh_coeffs, real_sh, sh_block, sphere_quadrature

__all__ = ["n_so3_coeffs", "wigner_blocks", "so3_design", "haar_rotations", "so3_quadrature",
           "density_coeffs", "delta_coeffs", "axis_density_coeffs", "watson_coeffs", "bingham_coeffs",
           "fit", "evaluate", "rotation_of"]


# ------------------------------------------------------------------ layout
def _n_cols(l, nmax):
    """Columns kept in block ``l``: the ``|n| <= nmax`` band, all of it when ``nmax`` is None."""
    return 2 * l + 1 if nmax is None else 2 * min(int(nmax), l) + 1


def n_so3_coeffs(lmax, nmax=None):
    """Coefficients up to ``lmax`` with the azimuthal band ``nmax`` (all of it when None)."""
    return sum((2 * l + 1) * _n_cols(l, nmax) for l in range(int(lmax) + 1))


def so3_index(lmax, nmax=None):
    """``(l, m, n)`` of every coefficient, in the order :func:`so3_design` lays them out."""
    out = []
    for l in range(int(lmax) + 1):
        k = _n_cols(l, nmax) // 2
        for m in range(-l, l + 1):
            for n in range(-k, k + 1):
                out.append((l, m, n))
    return out


# ------------------------------------------------------------------ the basis
def _sh_l(l, dirs):
    """Order-``l`` block of the orthonormal real spherical harmonics, ``(n, 2l+1)``."""
    return real_sh(l, dirs, full=True)[:, sh_block(l, True)]


@functools.lru_cache(maxsize=64)
def _anchors(l):
    """Directions and the pseudo-inverse of their harmonics, for reading a rotation matrix off its action.

    ``M^l(R)`` is fixed by ``Y_l(R a) = M^l(R) Y_l(a)`` at ``2l+1`` independent directions. The anchor set is
    over-determined (a Fibonacci spiral of ``4l+4`` directions) and inverted in least squares, which keeps the
    read-off well conditioned without a hand-rolled Wigner recursion to get wrong.
    """
    k = 4 * l + 4
    i = np.arange(k) + 0.5
    z = 1.0 - 2.0 * i / k
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    phi = i * np.pi * (1.0 + np.sqrt(5.0))
    a = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    A = _sh_l(l, a)
    return a, np.linalg.pinv(A)


def wigner_blocks(lmax, R):
    """Per order ``l``, the ``(N, 2l+1, 2l+1)`` real rotation matrices ``M^l(R)`` of the harmonic basis.

    ``M^l`` is defined by ``Y_lm(R d) = sum_m' M^l(R)[m, m'] Y_lm'(d)``, so it is a group homomorphism and
    orthogonal, and its ``m' = 0`` column is the harmonics of the rotated pole.
    """
    R = np.asarray(R, np.float64).reshape(-1, 3, 3)
    out = []
    for l in range(int(lmax) + 1):
        a, Ainv = _anchors(l)
        Ra = np.einsum("nij,kj->nki", R, a)                       # R a_k for every rotation
        Y = _sh_l(l, Ra.reshape(-1, 3)).reshape(R.shape[0], a.shape[0], 2 * l + 1)
        out.append(np.einsum("nkm,jk->nmj", Y, Ainv))
    return out


def so3_design(lmax, R, nmax=None):
    """``(N, n_so3_coeffs(lmax, nmax))`` orthonormal basis values ``sqrt(2l+1) D^l_{mn}(R)``."""
    blocks = wigner_blocks(lmax, R)
    cols = []
    for l, M in enumerate(blocks):
        k = _n_cols(l, nmax) // 2
        cols.append(np.sqrt(2 * l + 1) * M[:, :, l - k:l + k + 1].reshape(M.shape[0], -1))
    return np.concatenate(cols, axis=1)


def rotation_of(axis, frame_axis=(0.0, 0.0, 1.0), roll=0.0):
    """A rotation taking ``frame_axis`` onto ``axis``, with ``roll`` (rad) about ``frame_axis`` applied first.

    Only a convention, and only needed to *name* a pose: nothing in the composition depends on which rotation
    of the family a given axis is paired with, because the distribution carries the azimuth explicitly.
    """
    f = np.asarray(frame_axis, np.float64); f = f / np.linalg.norm(f)
    n = np.asarray(axis, np.float64); n = n / np.linalg.norm(n)
    v = np.cross(f, n)
    s, c = np.linalg.norm(v), float(f @ n)
    if s < 1e-12:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        R = np.eye(3) + K + K @ K * (1.0 / (1.0 + c))
    if roll:
        K = np.array([[0.0, -f[2], f[1]], [f[2], 0.0, -f[0]], [-f[1], f[0], 0.0]])
        R = R @ (np.eye(3) + np.sin(roll) * K + (1.0 - np.cos(roll)) * (K @ K))
    return R


# ------------------------------------------------------------------ sampling
def haar_rotations(n, seed=0):
    """``(n, 3, 3)`` rotations drawn uniformly on SO(3) (Shoemake's quaternion sampling)."""
    u = np.random.default_rng(seed).random((int(n), 3))
    s1, s2 = np.sqrt(1.0 - u[:, 0]), np.sqrt(u[:, 0])
    t1, t2 = 2.0 * np.pi * u[:, 1], 2.0 * np.pi * u[:, 2]
    q = np.stack([s1 * np.sin(t1), s1 * np.cos(t1), s2 * np.sin(t2), s2 * np.cos(t2)], axis=1)
    return _rotations_from_quaternions(q)


def _rotations_from_quaternions(q):
    """``(n, 3, 3)`` from ``(n, 4)`` quaternions ``(x, y, z, w)``; normalised on the way in."""
    q = np.asarray(q, np.float64).reshape(-1, 4)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1)], axis=1)


def so3_quadrature(lmax, nmax=None, frame_axis=(0.0, 0.0, 1.0)):
    """Rotations and normalised Haar weights exact for the band: a sphere rule times a roll rule.

    Every rotation is ``R(axis) Rz(roll)``, so the grid factorises into the pole direction -- where the density
    of an orientation distribution lives -- and the azimuth about it. Weights sum to one.
    """
    lmax = int(lmax)
    n_t, n_p = lmax + 2, 2 * lmax + 4
    n_roll = 2 * (lmax if nmax is None else int(nmax)) + 4
    dirs, w = sphere_quadrature(n_t, n_p)
    rolls = np.arange(n_roll) * (2.0 * np.pi / n_roll)
    R = np.stack([rotation_of(d, frame_axis, r) for d in dirs for r in rolls])
    wq = np.repeat(w / (4.0 * np.pi), n_roll) / n_roll
    return R, wq, dirs, rolls


# ------------------------------------------------------------------ distributions
def density_coeffs(density, lmax, nmax=None, frame_axis=(0.0, 0.0, 1.0)):
    """``f^l_{mn}`` of any orientation density, by the quadrature of :func:`so3_quadrature`.

    ``density`` is called with the ``(N, 3, 3)`` rotations and returns the density at each, in the normalised
    Haar measure (so a uniform distribution is ``1``). Every distribution below goes through this one path,
    which is why their structural properties -- a roll-uniform density having no ``n != 0`` coefficients, a
    Bingham with equal dispersions being a Watson -- are results to test rather than assumptions to trust.
    """
    R, w, _dirs, _rolls = so3_quadrature(lmax, nmax, frame_axis)
    p = np.asarray(density(R), np.float64).reshape(-1)
    p = p / float(p @ w)                                              # a density integrates to one
    return (so3_design(lmax, R, nmax) * (p * w)[:, None]).sum(axis=0)


def delta_coeffs(R0, lmax, nmax=None):
    """One pose: ``f = delta(R0)``, whose coefficients are the basis evaluated there."""
    return so3_design(lmax, np.asarray(R0, np.float64).reshape(1, 3, 3), nmax)[0]


def axis_density_coeffs(sh_coeffs, lmax, nmax=None, frame_axis=(0.0, 0.0, 1.0)):
    """A distribution over **directions** with no preferred azimuth, from harmonics in the required basis.

    This is what an ODF states: where the substrate axis points, and nothing about its spin. The coefficients
    come out non-zero only at ``n = 0``, so composing against it integrates the roll away exactly.
    """
    c = np.asarray(sh_coeffs, np.float64).reshape(-1)
    lmax_sh = _lmax_of(c.size)
    f = np.asarray(frame_axis, np.float64)

    def rho(R):
        d = np.einsum("nij,j->ni", R, f / np.linalg.norm(f))
        return 4.0 * np.pi * np.clip(real_sh(lmax_sh, d) @ c, 0.0, None)

    return density_coeffs(rho, lmax, nmax, frame_axis)


def watson_coeffs(kappa, mu=(0.0, 0.0, 1.0), lmax=8, nmax=None, frame_axis=(0.0, 0.0, 1.0)):
    """Watson dispersion about ``mu``: one concentration, axially symmetric, no preferred azimuth."""
    m = np.asarray(mu, np.float64); m = m / np.linalg.norm(m)
    f = np.asarray(frame_axis, np.float64); f = f / np.linalg.norm(f)

    def rho(R):
        d = np.einsum("nij,j->ni", R, f)
        return np.exp(float(kappa) * (d @ m) ** 2)

    return density_coeffs(rho, lmax, nmax, frame_axis)


def bingham_coeffs(frame, kappa, lmax=8, nmax=None, roll_kappa=0.0, frame_axis=(0.0, 0.0, 1.0)):
    """Bingham dispersion: the axis fans **anisotropically**, with a concentration about each of two axes.

    ``frame`` is a rotation whose third column is the mean direction and whose first two columns are the axes
    the two concentrations belong to. ``kappa = (k1, k2)`` are concentrations, larger being tighter, so a fan
    spread in the first axis and narrow in the second is a small ``k1`` and a large ``k2``, and two equal values
    are a Watson of that concentration. ``roll_kappa`` ties the substrate's own azimuth to the frame (a von
    Mises about it); zero leaves the azimuth free, which is the right statement for a population whose members
    are rolled arbitrarily.
    """
    F = np.asarray(frame, np.float64).reshape(3, 3)
    k1, k2 = (float(kappa[0]), float(kappa[1])) if np.ndim(kappa) else (float(kappa), float(kappa))
    f = np.asarray(frame_axis, np.float64); f = f / np.linalg.norm(f)

    def rho(R):
        d = np.einsum("nij,j->ni", R, f)                              # where the substrate axis points
        e = d @ F                                                     # in the frame's own coordinates
        p = np.exp(-k1 * e[:, 0] ** 2 - k2 * e[:, 1] ** 2)
        if roll_kappa:
            # the azimuth of the substrate about its own axis, measured against the frame's first axis
            u = np.einsum("nij,j->ni", R, np.array([1.0, 0.0, 0.0]))
            ref = F[:, 0] - np.outer(d @ F[:, 0], np.ones(3)) * d
            nrm = np.linalg.norm(ref, axis=1, keepdims=True)
            ref = np.where(nrm > 1e-9, ref / np.maximum(nrm, 1e-30), np.array([1.0, 0.0, 0.0]))
            p = p * np.exp(float(roll_kappa) * (np.einsum("ni,ni->n", u, ref)) ** 2)
        return p

    return density_coeffs(rho, lmax, nmax, frame_axis)


def _lmax_of(n_c):
    for l in range(0, 33, 2):
        if n_sh_coeffs(l) == n_c:
            return l
    raise ValueError(f"{n_c} coefficients is not an even-order real spherical-harmonic block")


# ------------------------------------------------------------------ fitting
def fit(design, values):
    """Least-squares coefficients of a sampled response, with what the fit could not represent.

    Returns ``(coeffs, residual, energy)``: ``residual`` the relative L2 misfit of the truncation, ``energy``
    the coefficient energy summed per ``l`` and per ``n`` -- the second of which is the measurement of how far
    from axially symmetric the response is, and the basis for choosing ``nmax`` instead of asserting it.
    """
    A = np.asarray(design, np.float64)
    y = np.asarray(values)
    if np.iscomplexobj(y):
        both, *_ = np.linalg.lstsq(A, np.column_stack([y.real, y.imag]), rcond=None)
        c = both[:, 0] + 1j * both[:, 1]
    else:
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = float(np.linalg.norm(A @ c - y) / max(np.linalg.norm(y), 1e-300))
    return c, resid


def energy(coeffs, lmax, nmax=None):
    """``(per_l, per_n)`` coefficient energy: how much of the response sits at each order and each azimuth."""
    idx = so3_index(lmax, nmax)
    c2 = np.abs(np.asarray(coeffs).reshape(-1)) ** 2
    per_l = np.zeros(int(lmax) + 1)
    n_hi = max(abs(n) for _l, _m, n in idx) if idx else 0
    per_n = np.zeros(n_hi + 1)
    for (l, _m, n), e in zip(idx, c2):
        per_l[l] += e
        per_n[abs(n)] += e
    return per_l, per_n


def evaluate(coeffs, lmax, R, nmax=None):
    """The fitted response at given rotations."""
    return so3_design(lmax, R, nmax) @ np.asarray(coeffs).reshape(-1)
