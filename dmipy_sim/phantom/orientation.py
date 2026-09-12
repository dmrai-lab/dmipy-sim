"""A pose per voxel for the substrates that have one (RPH.md 4).

A pose is a rotation, not an axis. The fields below differ in how much of it they pin down: :class:`Peaks` and
:class:`ODF` state a direction (or a density over directions) and leave the substrate's own azimuth unstated,
which the replay integrates away; :class:`Frames` states the whole rotation; :class:`Fan` states a frame and a
concentration about each of its first two axes, a population dispersed anisotropically. :class:`Watson` is a
field of Watson parameters -- a direction and a concentration per voxel -- whose coefficients come from
:func:`dmipy_sim.math.sh_analytical.watson_sh`, the one exact Watson in the ecosystem; nothing here evaluates a
distribution of its own.

Every field is a **volume**: grid + payload for one population per voxel, or grid + ``(K,)`` + payload with
``weights`` for ``K`` populations (a crossing).
"""
from __future__ import annotations

import numpy as np

from ..replay.so3 import n_sh_coeffs

__all__ = ["Peaks", "ODF", "Watson", "Frames", "Fan"]

#: How a source spells its spherical harmonics, mapped onto the two exact conversions of RPH.md 4.1
#: (``basis``, ``legacy`` of :meth:`dmipy_sim.replay.fod.FOD.from_sh`). MRtrix3 and dmipy-fit's CSD are the
#: required basis itself.
SH_BASES = {
    "mrtrix3": ("tournier07", False),
    "mrtrix-legacy": ("tournier07", True),
    "tournier07": ("tournier07", False),
    "tournier07-legacy": ("tournier07", True),
    "descoteaux07": ("descoteaux07", False),
    "dmipy-fit": ("tournier07", False),
}


class _Field:
    mode = None
    lmax = 0

    def _volume(self, a, trailing, name):
        a = np.asarray(a, np.float64)
        if a.ndim == len(trailing) + 3 and a.shape[3:] == tuple(trailing):
            return a[:, :, :, None, ...]                                    # one population
        if a.ndim == len(trailing) + 4 and a.shape[4:] == tuple(trailing):
            return a
        raise ValueError(f"{name} must be a volume of shape grid + {tuple(trailing)} (one population per voxel) or "
                         f"grid + (K,) + {tuple(trailing)} (K populations); got {a.shape}")

    def _weights(self, weights, shape):
        if weights is None:
            return None
        w = np.asarray(weights, np.float64)
        if w.shape != tuple(shape):
            raise ValueError(f"weights must have shape grid + (K,) = {tuple(shape)}; got {w.shape}")
        if (w < 0).any():
            raise ValueError("population weights cannot be negative")
        return w

    def _w(self, ijk, K):
        return np.ones(K) / K if self.weights is None else self.weights[ijk]

    @property
    def grid_shape(self):
        return self._shape[:3]

    @property
    def n_populations(self):
        return int(self._shape[3])


class Peaks(_Field):
    """Discrete directions: ``directions`` of shape grid + ``(3,)``, or grid + ``(K, 3)`` with ``weights`` for a
    crossing. The azimuth about each direction is unstated and integrated away (RPH.md 4). A zero vector is
    "no orientation here"."""

    mode = "peaks"

    def __init__(self, directions, *, weights=None):
        self.directions = self._volume(directions, (3,), "peak directions")
        self._shape = self.directions.shape[:4]
        self.weights = self._weights(weights, self._shape)

    def at(self, ijk):
        d = self.directions[ijk]
        w = self._w(ijk, d.shape[0])
        out = []
        for k in range(d.shape[0]):
            n = np.linalg.norm(d[k])
            if w[k] <= 0.0 or n == 0.0:
                continue
            out.append((float(w[k]), d[k] / n))
        return out

    def at_many(self, idx):
        """Every population of the voxels ``idx (N, 3)`` at once: ``(weights (N, K), payload (N, K, 3))``, a zero
        weight where a population is absent."""
        d = self.directions[tuple(idx.T)]                                       # (N, K, 3)
        n = np.linalg.norm(d, axis=-1)
        w = np.broadcast_to(np.ones(d.shape[1]) / d.shape[1], d.shape[:2]) if self.weights is None else self.weights[tuple(idx.T)]
        w = np.where((w > 0) & (n > 0), w, 0.0)
        return w, d / np.where(n > 0, n, 1.0)[..., None]


class ODF(_Field):
    """Orientation distributions over the grid, ``coeffs`` of shape grid + ``(n_c,)`` (or grid + ``(K, n_c)`` with
    ``weights``), in a **named** basis: one of :data:`SH_BASES`, converted per coefficient into the orthonormal
    basis RPH.md 4.1 requires. No default: the MRtrix legacy basis differs by a scale on the ``m != 0``
    coefficients that breaks the composition by an amount vanishing exactly where the gradient is parallel to
    B0, so a producer that cannot name its basis cannot declare one.

    Every voxel is normalised to unit integral. What the input integrated to -- a CSD's apparent fibre density
    -- is kept on :attr:`integral`, a volume, for the caller to put into the fractions if that is what it means;
    it is never silently eaten. An all-zero block is "no orientation here".
    """

    mode = "odf_sh"

    def __init__(self, coeffs, *, basis, weights=None):
        from ..replay.fod import FOD, _C00
        if basis not in SH_BASES:
            raise ValueError(f"unknown basis {basis!r}; name the source: {sorted(SH_BASES)}")
        self.basis = basis
        self._fod_basis, self._legacy = SH_BASES[basis]
        c = np.asarray(coeffs, np.float64)
        self.coeffs = self._volume(c, (c.shape[-1],), "odf coefficients")
        self._shape = self.coeffs.shape[:4]
        self.weights = self._weights(weights, self._shape)
        self.lmax = _lmax_of_n_coeffs(self.coeffs.shape[-1])
        self.integral = self.coeffs[..., 0] / _C00                      # grid + (K,): what each block integrates to
        self._FOD = FOD

    def at(self, ijk):
        c = self.coeffs[ijk]
        w = self._w(ijk, c.shape[0])
        out = []
        for k in range(c.shape[0]):
            if w[k] <= 0.0 or not np.any(c[k]):
                continue
            f = self._FOD.from_sh(c[k], basis=self._fod_basis, legacy=self._legacy, normalize=True)
            out.append((float(w[k]), f.coeffs))
        return out

    def at_many(self, idx):
        """Every population of the voxels ``idx (N, 3)``: ``(weights (N, K), coefficients (N, K, n_c))`` in the required
        basis and normalised, the conversion applied once to the whole block (RPH.md 4.1); an all-zero block
        has weight zero."""
        from ..replay.fod import _C00
        c = self.coeffs[tuple(idx.T)]                                            # (N, K, n_c)
        live = np.any(c != 0, axis=-1)
        w = np.broadcast_to(np.ones(c.shape[1]) / c.shape[1], c.shape[:2]) if self.weights is None else self.weights[tuple(idx.T)]
        w = np.where((w > 0) & live, w, 0.0)
        c = _convert_block(c, self._fod_basis, self._legacy, self.lmax)
        c0 = c[..., 0]
        if np.any((c0 <= 0) & live):
            raise ValueError("an ODF block has a non-positive l = 0 coefficient: not a density")
        return w, c * np.where(live, _C00 / np.where(c0 == 0, 1.0, c0), 0.0)[..., None]


class Watson(ODF):
    """A Watson distribution per voxel: ``mu`` a direction volume (grid + ``(3,)``, or grid + ``(K, 3)``), ``kappa``
    a concentration (a scalar, or a volume matching ``mu`` without its last axis). The coefficients are the exact
    ones of :func:`dmipy_sim.math.sh_analytical.watson_sh` at order ``lmax``, in the required basis."""

    def __init__(self, *, mu, kappa, lmax=8, weights=None):
        from ..replay.fod import FOD
        mu = self._volume(mu, (3,), "watson mu")
        kap = np.broadcast_to(np.asarray(kappa, np.float64), mu.shape[:4])
        c = np.zeros(mu.shape[:4] + (n_sh_coeffs(lmax),))
        for ijk in np.ndindex(mu.shape[:4]):
            n = np.linalg.norm(mu[ijk])
            if n == 0.0:
                continue
            c[ijk] = FOD.watson(float(kap[ijk]), mu=mu[ijk] / n, lmax=lmax).coeffs
        super().__init__(c, basis="tournier07", weights=weights)
        self.mu, self.kappa = mu, kap


class Frames(_Field):
    """A whole rotation per voxel: ``rotations`` of shape grid + ``(3, 3)``, or grid + ``(K, 3, 3)`` with
    ``weights``. What a substrate whose response is not axially symmetric needs to be placed unambiguously, and
    what an operation on the magnetisation vector needs: a pulse is applied to a pose, not to an axis."""

    mode = "frames"

    def __init__(self, rotations, *, weights=None):
        self.rotations = self._volume(rotations, (3, 3), "frame rotations")
        self._shape = self.rotations.shape[:4]
        self.weights = self._weights(weights, self._shape)

    def at(self, ijk):
        R = self.rotations[ijk]
        w = self._w(ijk, R.shape[0])
        out = []
        for k in range(R.shape[0]):
            if w[k] <= 0.0:
                continue
            if not np.allclose(R[k] @ R[k].T, np.eye(3), atol=1e-5) or np.linalg.det(R[k]) < 0:
                raise ValueError(f"the frame at {ijk} population {k} is not a proper rotation")
            out.append((float(w[k]), R[k]))
        return out

    def at_many(self, idx):
        """Every population of the voxels ``idx (N, 3)``: ``(weights (N, K), rotations (N, K, 3, 3))``, checked to be
        proper rotations wherever the weight is positive."""
        R = self.rotations[tuple(idx.T)]                                         # (N, K, 3, 3)
        w = np.broadcast_to(np.ones(R.shape[1]) / R.shape[1], R.shape[:2]) if self.weights is None else self.weights[tuple(idx.T)]
        w = np.where(w > 0, w, 0.0)
        live = w > 0
        RRt = np.einsum("...ij,...kj->...ik", R, R)
        bad = live & (~np.all(np.abs(RRt - np.eye(3)) < 1e-5, axis=(-2, -1)) | (np.linalg.det(R) < 0))
        if bad.any():
            v, k = np.argwhere(bad)[0]
            raise ValueError(f"the frame at voxel {tuple(idx[v])} population {k} is not a proper rotation")
        return w, R


class Fan(Frames):
    """A fan per voxel (RPH.md 4, bingham mode): a frame and a concentration about each of its first two axes.
    The frame's third column is the fibre axis; ``kappa = (k1, k2)`` concentrates the axis about the first and
    second columns (larger is tighter; equal values are the Watson cone of that width). ``roll_kappa`` ties the
    substrate's own azimuth to the frame; zero leaves it free.

    Reach for :meth:`from_axis` unless you already hold rotation matrices: it takes the two directions a user
    thinks in, the fibre axis and the direction the fan opens along, and assembles the frame.
    """

    mode = "bingham"

    def __init__(self, rotations, *, kappa, roll_kappa=0.0, weights=None):
        super().__init__(rotations, weights=weights)
        self.kappa = _with_population_axis(kappa, self._shape, (2,), "fan kappa")
        self.roll_kappa = _with_population_axis(roll_kappa, self._shape, (), "roll_kappa")
        if (self.kappa < 0).any() or (self.roll_kappa < 0).any():
            raise ValueError("concentrations are non-negative")

    @classmethod
    def from_axis(cls, *, axis, fan_towards, kappa_fan, kappa_perp, roll_kappa=0.0, weights=None):
        """The fan a user means: fibres along ``axis`` (a direction volume), spreading towards ``fan_towards`` (a
        direction volume, orthogonalised against the axis) with concentration ``kappa_fan`` in that plane and
        ``kappa_perp`` perpendicular to it. A small ``kappa_fan`` is a wide fan; equal concentrations are a cone.
        Scalars broadcast over the grid."""
        tmp = _Field()
        t = tmp._volume(axis, (3,), "axis")
        e = tmp._volume(fan_towards, (3,), "fan_towards")
        if e.shape != t.shape:
            raise ValueError(f"fan_towards {e.shape[:4]} must match axis {t.shape[:4]}")
        shape = t.shape[:4]
        nt = np.linalg.norm(t, axis=-1, keepdims=True)
        live = nt[..., 0] > 0
        t = np.where(live[..., None], t / np.where(nt == 0, 1.0, nt), 0.0)
        e = e - np.sum(e * t, axis=-1, keepdims=True) * t
        ne = np.linalg.norm(e, axis=-1, keepdims=True)
        if (live & (ne[..., 0] <= 1e-12)).any():
            raise ValueError("fan_towards is parallel to axis in some voxel: the fan plane is undefined there")
        e = np.where(live[..., None], e / np.where(ne == 0, 1.0, ne), 0.0)
        e2 = np.cross(t, e)
        R = np.stack([e, e2, t], axis=-1)                                     # columns: fan, perpendicular, axis
        R[~live] = np.eye(3)                                                  # no axis: identity, weight 0 below
        kf = np.broadcast_to(np.asarray(kappa_fan, np.float64), shape)
        kp = np.broadcast_to(np.asarray(kappa_perp, np.float64), shape)
        kappa = np.stack([kf, kp], axis=-1)
        w = None if weights is None else np.asarray(weights, np.float64)
        if w is None and not live.all():
            w = live.astype(np.float64)
        out = cls(R, kappa=kappa, roll_kappa=roll_kappa, weights=w)
        return out

    def at(self, ijk):
        return [(w, (R, tuple(self.kappa[ijk][k]), float(self.roll_kappa[ijk][k])))
                for k, (w, R) in enumerate(super().at(ijk))]

    def at_many(self, idx):
        """``(weights (N, K), (rotations (N, K, 3, 3), kappa (N, K, 2), roll_kappa (N, K)))``."""
        w, R = super().at_many(idx)
        return w, (R, self.kappa[tuple(idx.T)], self.roll_kappa[tuple(idx.T)])


def _convert_block(c, basis, legacy, lmax):
    """:meth:`FOD.from_sh`'s exact per-coefficient conversion, applied to a whole ``(..., n_c)`` block."""
    from ..replay.fod import _block
    out = np.array(c, np.float64, copy=True)
    if basis == "tournier07":
        if legacy:
            for l in range(0, lmax + 1, 2):
                blk = _block(l)
                for m in range(-l, l + 1):
                    if m != 0:
                        out[..., blk + l + m] /= np.sqrt(2.0)
    elif basis == "descoteaux07":
        for l in range(0, lmax + 1, 2):
            blk = _block(l)
            for m in range(-l, l + 1):
                out[..., blk + l + m] = ((-1) ** m if m > 0 else 1.0) * c[..., blk + l - m]
    else:
        raise ValueError(f"unknown basis {basis!r}")
    return out


def _with_population_axis(a, shape, trailing, name):
    """A per-voxel value broadcast over the population axis, whether or not the caller wrote one."""
    a = np.asarray(a, np.float64)
    for cand in (shape + trailing, shape[:3] + trailing):
        try:
            out = np.broadcast_to(a, cand)
        except ValueError:
            continue
        return out if cand == shape + trailing else np.broadcast_to(out[:, :, :, None, ...], shape + trailing)
    raise ValueError(f"{name} has shape {a.shape}, which is neither grid + {trailing} nor grid + (K,) + "
                     f"{trailing} for a {shape[3]}-population field")


def _lmax_of_n_coeffs(n_c):
    for l in range(0, 33, 2):
        if n_sh_coeffs(l) == n_c:
            return l
    raise ValueError(f"{n_c} coefficients is not an even-order real SH block")
