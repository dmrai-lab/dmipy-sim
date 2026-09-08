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
from dataclasses import dataclass

import numpy as np

from scipy.special import gammaln, lpmv
from numpy.polynomial.legendre import leggauss

__all__ = ["real_sh", "sphere_quadrature", "n_sh_coeffs", "sh_block",
           "n_so3_coeffs", "so3_index", "wigner_blocks", "so3_design", "haar_design", "haar_rotations",
           "so3_quadrature",
           "density_coeffs", "delta_coeffs", "axis_density_coeffs", "axis_coeffs", "watson_coeffs",
           "bingham_coeffs",
           "project", "quadrature_design", "oversampled_design", "truncate_coeffs", "energy", "evaluate",
           "rotation_of", "rotations_from_quaternions", "rotate_coeffs",
           "Distribution"]


# ------------------------------------------------------------------ real spherical harmonics
# The basis is normative: orthonormal real spherical harmonics, DIPY's `real_sh_tournier(legacy=False)` (RPH.md
# 4.1). Several conventions in circulation are not orthonormal -- DIPY's default `legacy=True` scales every
# m != 0 function by 1/sqrt(2) -- and every identity here, from the rotation matrices to the Peter-Weyl inner
# product, holds only in an orthonormal one. Coefficients from elsewhere are converted on the way in
# (:class:`dmipy_sim.replay.fod.FOD`), never assumed.
def n_sh_coeffs(lmax, full=False):
    """Coefficients in the compact layout up to ``lmax`` (even orders only unless ``full``).

    ``full=True`` keeps odd orders too.  It is needed wherever the two acquisition directions
    are expanded: antipodal symmetry of the response constrains ``l1 + l2`` to be even, which
    admits odd--odd pairs such as ``P_1 P_1`` -- weaker than requiring each order even.
    """
    step = 1 if full else 2
    return sum(2 * l + 1 for l in range(0, lmax + 1, step))


def sh_block(l, full=False):
    """Slice of the compact array holding order ``l`` (``m = -l..+l``)."""
    if full:
        return slice(l * l, l * l + 2 * l + 1)
    M = l // 2
    start = M * (2 * M - 1) if M > 0 else 0
    return slice(start, start + 2 * l + 1)


def real_sh(lmax, dirs, full=False):
    """Orthonormal real spherical harmonics in the compact layout.

    ``dirs`` is ``(n, 3)`` unit Cartesian; returns ``(n, n_sh_coeffs(lmax, full))``.
    """
    d = np.asarray(dirs, np.float64).reshape(-1, 3)
    x = np.clip(d[:, 2], -1.0, 1.0)
    phi = np.arctan2(d[:, 1], d[:, 0])
    out = np.empty((d.shape[0], n_sh_coeffs(lmax, full)), np.float64)
    for l in range(0, lmax + 1, 1 if full else 2):
        blk = sh_block(l, full)
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
    """Order-``l`` block of the orthonormal real spherical harmonics, ``(n, 2l+1)``.

    Only that order: going through :func:`real_sh` would recompute every lower order on every call, which the
    rotation matrices do once per order and would pay ``lmax`` times over.
    """
    d = np.asarray(dirs, np.float64).reshape(-1, 3)
    x = np.clip(d[:, 2], -1.0, 1.0)
    phi = np.arctan2(d[:, 1], d[:, 0])
    out = np.empty((d.shape[0], 2 * l + 1), np.float64)
    out[:, l] = np.sqrt((2 * l + 1) / (4 * np.pi)) * lpmv(0, l, x)
    for m in range(1, l + 1):
        K = np.sqrt((2 * l + 1) / (4 * np.pi) * np.exp(gammaln(l - m + 1) - gammaln(l + m + 1)))
        P = np.sqrt(2.0) * K * lpmv(m, l, x)
        out[:, l + m] = P * np.cos(m * phi)
        out[:, l - m] = P * np.sin(m * phi)
    return out


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


@functools.lru_cache(maxsize=8)
def quadrature_design(lmax, nmax, frame_axis=(0.0, 0.0, 1.0)):
    """The quadrature rotations, their Haar weights and their design matrix, built once per band.

    This is where a response is sampled and projected: the rule is exact for the band, so the projection needs
    no solve, and every pack and acquisition at the same band shares the matrix.
    """
    R, w, _dirs, _rolls = so3_quadrature(int(lmax), None if nmax is None else int(nmax), tuple(frame_axis))
    return R, w, so3_design(int(lmax), R, None if nmax is None else int(nmax))


@functools.lru_cache(maxsize=8)
def haar_design(lmax, nmax, n_samples, seed=0):
    """Uniform rotations and their design matrix, built once and reused: the **off-grid** set, where a
    projection is checked against the response it claims to represent."""
    R = haar_rotations(int(n_samples), int(seed))
    return R, so3_design(int(lmax), R, None if nmax is None else int(nmax))


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
    """``(n, 3, 3)`` rotations drawn uniformly on SO(3)."""
    from scipy.spatial.transform import Rotation
    return Rotation.random(int(n), random_state=int(seed)).as_matrix().reshape(int(n), 3, 3)


def rotations_from_quaternions(q):
    """``(n, 3, 3)`` from ``(n, 4)`` quaternions in the ``(x, y, z, w)`` convention RPH.md 4 states."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(np.asarray(q, np.float64).reshape(-1, 4)).as_matrix().reshape(-1, 3, 3)


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
@functools.lru_cache(maxsize=16)
def oversampled_design(lmax, nmax, over, frame_axis=(0.0, 0.0, 1.0)):
    """A quadrature beyond the retained band, with the **retained** basis evaluated on it: ``(R, w, A)``.

    Two bands, not one. The rule has to integrate the product of what is being projected with the basis
    functions retained, so a grid sized to the retained band alone folds everything above it into the
    coefficients that are kept -- and it does so differently at different frames, since the grid is not
    rotation invariant. Measured on a Bingham: the same fan came out 0.12 apart (in coefficients of order 1)
    at two poses when integrated on its own band's grid, and 7e-5 apart on an oversampled one.
    """
    nm = None if nmax is None else int(nmax) + int(over)
    R, w, _A = quadrature_design(int(lmax) + int(over), nm, frame_axis)
    return R, w, so3_design(int(lmax), R, None if nmax is None else int(nmax))


def density_coeffs(density, lmax, nmax=None, frame_axis=(0.0, 0.0, 1.0), over=8):
    """``f^l_{mn}`` of any orientation density, by the quadrature of :func:`so3_quadrature`.

    ``density`` is called with the ``(N, 3, 3)`` rotations and returns the density at each, in the normalised
    Haar measure (so a uniform distribution is ``1``). Every distribution below goes through this one path,
    which is why their structural properties -- a roll-uniform density having no ``n != 0`` coefficients, a
    Bingham with equal dispersions being a Watson -- are results to test rather than assumptions to trust.
    """
    R, w, A = oversampled_design(int(lmax), None if nmax is None else int(nmax), int(over), tuple(frame_axis))
    p = np.asarray(density(R), np.float64).reshape(-1)
    p = p / float(p @ w)                                              # a density integrates to one
    return project(A, p * w, np.ones(R.shape[0]))


def delta_coeffs(R0, lmax, nmax=None):
    """One pose: ``f = delta(R0)``, whose coefficients are the basis evaluated there."""
    return so3_design(lmax, np.asarray(R0, np.float64).reshape(1, 3, 3), nmax)[0]


@functools.lru_cache(maxsize=32)
def _axis_map(lmax, nmax, frame_axis=(0.0, 0.0, 1.0)):
    """The linear map from a density over **directions** to SO(3) coefficients, ``(n_feat, n_sh)``.

    A distribution over directions says where the substrate axis points and nothing about the substrate's own
    azimuth, so it is uniform in that azimuth, and the map is a projection with a closed form: one quadrature
    pass, cached, and a matrix-vector product per voxel after that. The image lies entirely in the ``n = 0``
    coefficients, which is the exact statement that the azimuth has been integrated away.
    """
    R, w, A = quadrature_design(lmax, nmax, frame_axis)
    f = np.asarray(frame_axis, np.float64); f = f / np.linalg.norm(f)
    Y = real_sh(lmax, np.einsum("nij,j->ni", R, f), full=True)
    return 4.0 * np.pi * (A * w[:, None]).T @ Y


def _embed_sh(coeffs, lmax):
    """Even-order (or full) harmonic coefficients placed in the full layout up to ``lmax``."""
    c = np.asarray(coeffs, np.float64).reshape(-1)
    out = np.zeros(n_sh_coeffs(lmax, full=True))
    if c.size == n_sh_coeffs(lmax, full=True):
        return c
    for l in range(0, lmax + 1, 2):                                   # even orders, compact -> full
        src, dst = sh_block(l, False), sh_block(l, True)
        if src.stop <= c.size:
            out[dst] = c[src]
    return out


def axis_density_coeffs(sh_coeffs, lmax, nmax=None, frame_axis=(0.0, 0.0, 1.0)):
    """A distribution over **directions**, from harmonic coefficients in the required basis.

    This is what an ODF states. The coefficients come out non-zero only at ``n = 0``, so composing against it
    integrates the substrate's azimuth away exactly, with no sampling of it anywhere.
    """
    return _axis_map(int(lmax), None if nmax is None else int(nmax), tuple(frame_axis)) @ _embed_sh(sh_coeffs, int(lmax))


def axis_coeffs(direction, lmax, nmax=None, frame_axis=(0.0, 0.0, 1.0)):
    """One direction with its azimuth unstated: the zero-dispersion limit of an axis density (RPH.md 4).

    A peak is this, and not a rotation: naming a direction leaves the substrate's spin about it unsaid, so the
    composition is over every spin equally.
    """
    n = np.asarray(direction, np.float64).reshape(3)
    n = n / np.linalg.norm(n)
    return axis_density_coeffs(real_sh(int(lmax), n[None, :], full=True)[0], lmax, nmax, frame_axis)


def watson_coeffs(kappa, mu=(0.0, 0.0, 1.0), lmax=8, nmax=None, frame_axis=(0.0, 0.0, 1.0)):
    """Watson dispersion about ``mu``: one concentration, no preferred azimuth.

    The harmonic coefficients are exact (:func:`dmipy_sim.math.sh_analytical.watson_sh`), so this is
    the analytic form mapped onto the group rather than a density sampled on a grid.
    """
    from ..math.sh_analytical import watson_sh
    mu = np.asarray(mu, np.float64); mu = mu / np.linalg.norm(mu)
    # an axis density has only even orders, and the analytic forms are written in that layout, so the harmonic
    # order rounds down to even while the SO(3) band it is embedded in stays as asked
    return axis_density_coeffs(watson_sh(mu, float(kappa), l_max=2 * (int(lmax) // 2)), lmax, nmax, frame_axis)


def bingham_coeffs(frame, kappa, lmax=8, nmax=None, roll_kappa=0.0, frame_axis=(0.0, 0.0, 1.0)):
    """Bingham dispersion: the axis fans **anisotropically**, with a concentration about each of two
    axes of a declared frame.

    ``frame`` is a rotation whose third column is the mean direction and whose first two columns are
    the axes the two concentrations belong to. ``kappa = (k1, k2)`` are concentrations, larger being
    tighter, so a fan spread in the first axis and narrow in the second is a small ``k1`` and a large
    ``k2``, and two equal values are the Watson of that concentration.

    With the azimuth free -- the usual case, and what an orientation distribution states -- this is
    the analytic form of :func:`dmipy_sim.math.sh_analytical.bingham_sh`. ``roll_kappa`` ties the
    substrate's own azimuth to the frame (a von Mises about it), which is a density on the group
    rather than on the sphere and has no closed form here, so that case alone is projected by
    quadrature.
    """
    from ..math.sh_analytical import bingham_sh
    F = np.asarray(frame, np.float64).reshape(3, 3)
    k1, k2 = (float(kappa[0]), float(kappa[1])) if np.ndim(kappa) else (float(kappa), float(kappa))
    if not roll_kappa:
        return axis_density_coeffs(bingham_sh(F, (k1, k2), l_max=2 * (int(lmax) // 2)), lmax, nmax, frame_axis)

    f = np.asarray(frame_axis, np.float64); f = f / np.linalg.norm(f)

    def rho(R):
        d = np.einsum("nij,j->ni", R, f)                              # where the substrate axis points
        e = d @ F                                                     # in the frame's own coordinates
        p = np.exp(-k1 * e[:, 0] ** 2 - k2 * e[:, 1] ** 2)
        # the azimuth of the substrate about its own axis, measured against the frame's first axis
        u = np.einsum("nij,j->ni", R, np.array([1.0, 0.0, 0.0]))
        ref = F[:, 0] - np.outer(d @ F[:, 0], np.ones(3)) * d
        nrm = np.linalg.norm(ref, axis=1, keepdims=True)
        ref = np.where(nrm > 1e-9, ref / np.maximum(nrm, 1e-30), np.array([1.0, 0.0, 0.0]))
        return p * np.exp(float(roll_kappa) * (np.einsum("ni,ni->n", u, ref)) ** 2)

    return density_coeffs(rho, lmax, nmax, frame_axis)


@dataclass(frozen=True)
class Distribution:
    """A voxel's orientation distribution as SO(3) coefficients, with where they came from.

    Composing a pack's response against a voxel is the inner product of this with
    :class:`~dmipy_sim.replay.replay.PoseResponse`, so every kind of orientation statement -- one pose, an ODF
    over directions, a Watson cone, a Bingham fan -- reaches the composition in the same form and differs only
    in its coefficients. ``source`` records which statement it was, since the coefficients alone cannot say.
    """

    coeffs: np.ndarray
    lmax: int
    nmax: int
    source: str

    @classmethod
    def pose(cls, R, lmax=8, nmax=4):
        """One rotation: the whole distribution is a point, and composing it evaluates the response there."""
        return cls(delta_coeffs(R, lmax, nmax), int(lmax), int(nmax), "a single pose")

    @classmethod
    def axis(cls, direction, lmax=8, nmax=4):
        """One direction, azimuth unstated -- what a peak is, and the zero-dispersion limit of an axis density."""
        return cls(axis_coeffs(direction, lmax, nmax), int(lmax), int(nmax), "one axis, azimuth unstated")

    @classmethod
    def axis_density(cls, fod, lmax=8, nmax=4):
        """A distribution over **directions** -- an ODF -- with no statement about the substrate's own azimuth.

        Takes a :class:`~dmipy_sim.replay.fod.FOD`, whose basis provenance is checked there, because a bare
        coefficient array cannot say which spherical-harmonic convention it is in and the wrong one is silently
        wrong. The coefficients come out non-zero only at ``n = 0``, so the azimuth is integrated away exactly.
        """
        c = getattr(fod, "coeffs", None)
        if c is None:
            raise TypeError("axis_density takes a dmipy_sim.replay.fod.FOD, not a bare coefficient array: the "
                            "spherical-harmonic convention has to be named or the composition is silently wrong "
                            "(FOD.from_sh(coeffs, basis=...), FOD.native, FOD.watson)")
        return cls(axis_density_coeffs(c, lmax, nmax), int(lmax), int(nmax),
                   f"an axis density, roll-uniform ({getattr(fod, 'source', 'unknown source')})")

    @classmethod
    def watson(cls, kappa, mu=(0.0, 0.0, 1.0), lmax=8, nmax=4):
        """A cone of directions about ``mu``: one concentration, no preferred azimuth."""
        return cls(watson_coeffs(kappa, mu, lmax, nmax), int(lmax), int(nmax),
                   f"a Watson, kappa={float(kappa):.3g}")

    @classmethod
    def bingham(cls, frame, kappa, roll_kappa=0.0, lmax=8, nmax=4):
        """A fan: two concentrations about the axes of a declared frame, and optionally a tied azimuth."""
        k = (float(kappa[0]), float(kappa[1])) if np.ndim(kappa) else (float(kappa), float(kappa))
        return cls(bingham_coeffs(frame, k, lmax, nmax, roll_kappa), int(lmax), int(nmax),
                   f"a Bingham, kappa={k}, roll_kappa={float(roll_kappa):.3g}")

    @classmethod
    def uniform(cls, lmax=8, nmax=4):
        """Every pose equally: the powder average, which is the zeroth coefficient alone."""
        c = np.zeros(n_so3_coeffs(lmax, nmax))
        c[0] = 1.0
        return cls(c, int(lmax), int(nmax), "uniform (powder)")

    def rotated(self, R):
        """The same distribution with every pose rotated by ``R``: how a voxel's own frame enters the scanner."""
        return Distribution(rotate_coeffs(self.coeffs, R, self.lmax, self.nmax)[0], self.lmax, self.nmax,
                            f"{self.source}, rotated")


def rotate_coeffs(coeffs, R, lmax, nmax=None):
    """Rotate coefficient vectors: ``(N, n_feat)`` for ``(N, 3, 3)`` rotations, batched, either side broadcast.

    A density whose poses are all rotated by ``R`` has coefficients ``M^l(R) f^l``, block by block, because the
    basis is a representation: ``phi(R S) = M^l(R) phi(S)``. So a distribution's frame is a rotation of a
    canonical shape, and a whole grid of frames is one batched product rather than a quadrature per voxel.
    """
    c = np.atleast_2d(np.asarray(coeffs, np.float64))
    M = wigner_blocks(int(lmax), np.asarray(R, np.float64).reshape(-1, 3, 3))
    n_R = M[0].shape[0]
    if c.shape[0] not in (1, n_R) and n_R != 1:
        raise ValueError(f"{c.shape[0]} coefficient vectors and {n_R} rotations do not broadcast")
    n = max(c.shape[0], n_R)
    if c.shape[0] == 1:
        c = np.broadcast_to(c, (n, c.shape[1]))
    if n_R == 1:
        M = [np.broadcast_to(Ml, (n,) + Ml.shape[1:]) for Ml in M]
    out, i = np.empty((n, c.shape[1])), 0
    for l, Ml in enumerate(M):
        k = _n_cols(l, nmax)
        width = (2 * l + 1) * k
        blk = c[:, i:i + width].reshape(n, 2 * l + 1, k)
        out[:, i:i + width] = np.einsum("nmk,nkj->nmj", Ml, blk).reshape(n, width)
        i += width
    return out


@functools.lru_cache(maxsize=64)
def _truncation_index(lmax, nmax, keep_lmax, keep_nmax):
    """Where each coefficient of the ``(keep_lmax, keep_nmax)`` layout sits in the ``(lmax, nmax)`` one."""
    if keep_lmax > lmax or (nmax is not None and (keep_nmax is None or keep_nmax > nmax)):
        raise ValueError(f"cannot keep ({keep_lmax}, {keep_nmax}) out of a ({lmax}, {nmax}) expansion: the "
                         f"retained band has to be inside the one that was projected")
    src = {k: i for i, k in enumerate(so3_index(lmax, nmax))}
    return np.array([src[k] for k in so3_index(keep_lmax, keep_nmax)], np.int64)


def truncate_coeffs(coeffs, lmax, nmax, keep_lmax, keep_nmax):
    """Restrict coefficients from one band to a smaller one, exactly.

    Dropping a coefficient is lossless for any distribution that has none there, which is what makes the
    economy of the composition free rather than approximate: an orientation distribution over directions has
    nothing at ``n != 0``, and one of order ``L`` has nothing above it.
    """
    c = np.asarray(coeffs)
    idx = _truncation_index(int(lmax), nmax, int(keep_lmax), keep_nmax)
    return c[..., idx]


def _lmax_of(n_c):
    for l in range(0, 33, 2):
        if n_sh_coeffs(l) == n_c:
            return l
    raise ValueError(f"{n_c} coefficients is not an even-order real spherical-harmonic block")


# ------------------------------------------------------------------ fitting
def project(design, weights, values):
    """Project a sampled response onto the basis: ``c = sum_q w_q phi(R_q) E(R_q)``.

    The basis is orthonormal under the Haar measure and :func:`so3_quadrature` integrates its band exactly, so
    the projection is one weighted matrix product -- no normal equations, no factorisation, nothing to condition.
    (It is also the only affordable route on a machine whose LAPACK is slow: a 669-coefficient least-squares
    solve measured 47 s here against 0.1 s for the same-sized matrix product.)

    ``values`` is ``(n_samples,)`` or ``(n_samples, n_columns)``, real or complex -- a whole acquisition
    projects in one product. Content **outside** the band aliases into the coefficients rather than showing up
    here, which is why the caller checks the result at rotations off the grid.
    """
    A = np.asarray(design, np.float64)
    w = np.asarray(weights, np.float64)
    y = np.asarray(values)
    flat = y.ndim == 1
    Y = y.reshape(y.shape[0], -1)
    C = (A * w[:, None]).T @ Y
    return C[:, 0] if flat else C


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
