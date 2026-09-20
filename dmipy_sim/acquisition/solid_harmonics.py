"""The real solid harmonics a magnet's field is described in, in the standard shim nomenclature.

A magnetic field component in a current-free region satisfies Laplace's equation, so ``dB_z`` is a sum of
solid harmonics and can be nothing else. Which ones, and how they are normalised and named, is not a free
choice either: Romeo and Hoult (1984) fixed the description magnet builders and shim hardware have used ever
since, and a shim coil labelled "Z2" on a console is the term of that name here.

This is a BASIS, not a model. Nothing in it decides which coefficients a particular magnet has -- that is
what a catalogue entry is for, and what the published figures of a machine do or do not determine.

The polynomials are written out rather than generated, because the standard forms are what a reader will
want to check against a reference; and every one of them is verified harmonic by
:func:`check_harmonic`, so a typo cannot survive.
"""
from __future__ import annotations

import numpy as np

__all__ = ["TERMS", "names_through", "evaluate", "gradient", "check_harmonic"]

#: ``name -> (order, value(x, y, z), gradient(x, y, z))``. Orders 1 to 4, in the conventional shim order.
#: Order 0 is the constant and is omitted: a uniform offset of the static field is the centre frequency,
#: not a field shape, and re-centring removes it.
TERMS = {
    # ── order 1: the linear shims, which are the imaging gradients ──────────────────────────────────
    "Z":        (1, lambda x, y, z: z,
                 lambda x, y, z: (np.zeros_like(x), np.zeros_like(y), np.ones_like(z))),
    "X":        (1, lambda x, y, z: x,
                 lambda x, y, z: (np.ones_like(x), np.zeros_like(y), np.zeros_like(z))),
    "Y":        (1, lambda x, y, z: y,
                 lambda x, y, z: (np.zeros_like(x), np.ones_like(y), np.zeros_like(z))),
    # ── order 2 ────────────────────────────────────────────────────────────────────────────────────
    "Z2":       (2, lambda x, y, z: 2 * z ** 2 - x ** 2 - y ** 2,
                 lambda x, y, z: (-2 * x, -2 * y, 4 * z)),
    "ZX":       (2, lambda x, y, z: x * z,
                 lambda x, y, z: (z, np.zeros_like(y), x)),
    "ZY":       (2, lambda x, y, z: y * z,
                 lambda x, y, z: (np.zeros_like(x), z, y)),
    "X2Y2":     (2, lambda x, y, z: x ** 2 - y ** 2,
                 lambda x, y, z: (2 * x, -2 * y, np.zeros_like(z))),
    "XY":       (2, lambda x, y, z: x * y,
                 lambda x, y, z: (y, x, np.zeros_like(z))),
    # ── order 3 ────────────────────────────────────────────────────────────────────────────────────
    "Z3":       (3, lambda x, y, z: z * (2 * z ** 2 - 3 * x ** 2 - 3 * y ** 2),
                 lambda x, y, z: (-6 * x * z, -6 * y * z, 6 * z ** 2 - 3 * x ** 2 - 3 * y ** 2)),
    "Z2X":      (3, lambda x, y, z: x * (4 * z ** 2 - x ** 2 - y ** 2),
                 lambda x, y, z: (4 * z ** 2 - 3 * x ** 2 - y ** 2, -2 * x * y, 8 * x * z)),
    "Z2Y":      (3, lambda x, y, z: y * (4 * z ** 2 - x ** 2 - y ** 2),
                 lambda x, y, z: (-2 * x * y, 4 * z ** 2 - x ** 2 - 3 * y ** 2, 8 * y * z)),
    "ZX2Y2":    (3, lambda x, y, z: z * (x ** 2 - y ** 2),
                 lambda x, y, z: (2 * x * z, -2 * y * z, x ** 2 - y ** 2)),
    "XYZ":      (3, lambda x, y, z: x * y * z,
                 lambda x, y, z: (y * z, x * z, x * y)),
    "X3":       (3, lambda x, y, z: x * (x ** 2 - 3 * y ** 2),
                 lambda x, y, z: (3 * x ** 2 - 3 * y ** 2, -6 * x * y, np.zeros_like(z))),
    "Y3":       (3, lambda x, y, z: y * (3 * x ** 2 - y ** 2),
                 lambda x, y, z: (6 * x * y, 3 * x ** 2 - 3 * y ** 2, np.zeros_like(z))),
    # ── order 4, zonal and the two tesseral terms a yoked magnet is most likely to carry ────────────
    "Z4":       (4, lambda x, y, z: 8 * z ** 4 - 24 * z ** 2 * (x ** 2 + y ** 2) + 3 * (x ** 2 + y ** 2) ** 2,
                 lambda x, y, z: (-48 * z ** 2 * x + 12 * x * (x ** 2 + y ** 2),
                                  -48 * z ** 2 * y + 12 * y * (x ** 2 + y ** 2),
                                  32 * z ** 3 - 48 * z * (x ** 2 + y ** 2))),
    "Z3X":      (4, lambda x, y, z: x * z * (4 * z ** 2 - 3 * x ** 2 - 3 * y ** 2),
                 lambda x, y, z: (z * (4 * z ** 2 - 9 * x ** 2 - 3 * y ** 2), -6 * x * y * z,
                                  x * (12 * z ** 2 - 3 * x ** 2 - 3 * y ** 2))),
    "Z3Y":      (4, lambda x, y, z: y * z * (4 * z ** 2 - 3 * x ** 2 - 3 * y ** 2),
                 lambda x, y, z: (-6 * x * y * z, z * (4 * z ** 2 - 3 * x ** 2 - 9 * y ** 2),
                                  y * (12 * z ** 2 - 3 * x ** 2 - 3 * y ** 2))),
}


def names_through(order):
    """Every term name up to and including ``order``, in the conventional order."""
    return [n for n, (l, _f, _g) in TERMS.items() if l <= int(order)]


def evaluate(coeffs, r):
    """``dB/B0`` at ``r`` (``(..., 3)``, metres) from ``{name: coefficient}``."""
    r = np.atleast_2d(np.asarray(r, dtype=np.float64))
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    out = np.zeros(x.shape, dtype=np.float64)
    for name, c in coeffs.items():
        if c:
            out = out + c * TERMS[name][1](x, y, z)
    return out


def gradient(coeffs, r):
    """The spatial gradient of :func:`evaluate`, ``(..., 3)``, analytically."""
    r = np.atleast_2d(np.asarray(r, dtype=np.float64))
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    gx, gy, gz = (np.zeros(x.shape) for _ in range(3))
    for name, c in coeffs.items():
        if c:
            dx, dy, dz = TERMS[name][2](x, y, z)
            gx, gy, gz = gx + c * dx, gy + c * dy, gz + c * dz
    return np.stack([gx, gy, gz], axis=-1)


def check_harmonic(name, h=1e-4, points=None):
    """The largest ``|laplacian|`` of one term over a few points, scaled by its own curvature.

    Every entry in :data:`TERMS` must return ~0. This is what makes a written-out table safe: a mistyped
    coefficient stops being a harmonic and the check finds it, where an eye would not.
    """
    P = np.array([[0.031, -0.047, 0.023], [-0.019, 0.011, -0.053], [0.041, 0.037, 0.017]]) if points is None \
        else np.asarray(points, dtype=np.float64)
    f = lambda q: TERMS[name][1](q[:, 0], q[:, 1], q[:, 2])
    lap = sum(f(P + h * e) - 2 * f(P) + f(P - h * e) for e in np.eye(3)) / h ** 2
    scale = max(np.abs(sum(np.abs(f(P + h * e)) + np.abs(f(P - h * e)) for e in np.eye(3))).max(), 1e-12)
    return float(np.abs(lap).max() / (scale / h ** 2))
