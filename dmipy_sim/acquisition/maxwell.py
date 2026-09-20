"""Tier 0 of the scanner model's evaluation harness: the constraints Maxwell imposes for free.

These need no oracle, no reference machine and no measurement. They are properties any field the model
emits must have because of what a magnetic field IS, and each one is checkable at any point at the cost
of a few evaluations. A law that fails one is not an inaccurate field; it is not a field.

Three constraints live here (dmipy-sim#364 tier 0):

``require_harmonic``
    ``B_z`` satisfies Laplace's equation in a current-free region, so any expansion of the static field
    must too. This is the check that would have caught a ``c r^2`` inhomogeneity law on the day it was
    written: ``laplacian(r^2) = 6``, so the law places a forbidden interior minimum at isocentre, and
    every downstream number stays finite while being wrong.

``require_transverse``
    Only the component of B1 perpendicular to B0 excites, so a machine whose transmit axis lies along its
    field would produce no signal. Both axes are plausible unit vectors, so the confusion never announces
    itself.

``require_gradient_tensor_admissible``
    A DIAGONAL gradient-nonlinearity tensor that is not the identity is impossible. The proof is three
    lines: if ``L`` is diagonal throughout the volume then the z coil has ``d/dx B_z = d/dy B_z = 0``
    everywhere, so ``B_z = f(z)`` alone; Laplace then gives ``f'' = 0``, so ``f`` is linear and
    ``L_zz = 1``; likewise for x and y, hence ``L`` is the identity. So a spatially varying diagonal ``L``
    is the same class of error as ``c r^2`` -- an expression that is not a magnetic field -- and not a
    coarse approximation of one.

The numerical Laplacian is second-order accurate, so ``h`` trades truncation against cancellation; the
residual is reported RELATIVE to the curvature the same stencil sees, which makes it dimensionless and
independent of how the field is scaled.
"""
import numpy as np

DEFAULT_H = 1e-4                 # metres; the stencil width for the numerical Laplacian
HARMONIC_TOL = 1e-6              # a genuine harmonic lands near float64 noise, a wrong one near 1
#   the separation is the whole point: measured, the basis terms give <1e-8 and ``c r^2`` gives exactly 1.0


def harmonic_residual(field, points, h=DEFAULT_H):
    """How badly ``field`` fails Laplace's equation over ``points``: ``|sum d2| / sum|d2|`` per point.

    ``field`` maps ``(n, 3)`` positions in metres to ``(n,)`` values in any unit. The three per-axis second
    differences are compared against their own total size, not against the field's value, which is what
    makes the answer independent of ``h``, of the field's scale and of its units: a harmonic gives roundoff
    because the three differences CANCEL, and anything else gives an order-one number because they do not.
    ``c r^2`` gives exactly 1.

    Normalising by the value instead would hide a non-field behind a small stencil -- the ratio would fall
    as ``h^2`` and any tolerance could be met by shrinking ``h``.
    """
    P = np.atleast_2d(np.asarray(points, dtype=np.float64))
    f0 = np.asarray(field(P), dtype=np.float64)
    d2 = []
    for e in np.eye(3):
        g = [np.asarray(field(P + k * h * e), dtype=np.float64) for k in (-2, -1, 1, 2)]
        d2.append(-g[0] / 12.0 + 4.0 * g[1] / 3.0 - 2.5 * f0 + 4.0 * g[2] / 3.0 - g[3] / 12.0)
    d2 = np.stack(d2, axis=0)
    total = np.abs(np.sum(d2, axis=0))
    size = np.sum(np.abs(d2), axis=0)
    # A field with no measurable curvature along ANY axis -- a linear or bilinear one such as x y -- has
    # second differences that vanish identically, and the ratio would then divide roundoff by roundoff and
    # report a clean harmonic as a violation. Below the floor there is nothing to test and the function is
    # harmonic for the same reason it is smooth.
    floor = 1e-10 * float(np.max(np.abs(f0))) if f0.size else 0.0
    return float(np.max(np.where(size > floor, total / np.maximum(size, 1e-300), 0.0)))


def require_harmonic(field, points, what, tol=HARMONIC_TOL, h=DEFAULT_H):
    """Refuse a static-field law that does not solve Laplace's equation."""
    res = harmonic_residual(field, points, h=h)
    if not np.isfinite(res) or res > tol:
        raise ValueError(
            f"{what} is not a magnetic field: its Laplacian is {res:.2e} of its own curvature, where a "
            f"field in a current-free region must give 0. B_z solves Laplace's equation there, so an "
            f"expansion of it is a sum of solid harmonics and can be nothing else -- a law that fails this "
            f"puts extrema where the maximum principle forbids them and stays finite while doing it")
    return res


def require_transverse(b0_axis, b1_axis, what, tol=1e-6):
    """Refuse a machine whose transmit axis is not perpendicular to its field."""
    if b0_axis is None or b1_axis is None:
        return None
    b0 = np.asarray(b0_axis, dtype=np.float64)
    b1 = np.asarray(b1_axis, dtype=np.float64)
    dot = float(abs(b0 @ b1) / (np.linalg.norm(b0) * np.linalg.norm(b1)))
    if dot > tol:
        raise ValueError(
            f"{what} declares a transmit axis {tuple(b1)} that is not perpendicular to its field "
            f"{tuple(b0)} (|cos| = {dot:.3f}). Only the component of B1 perpendicular to B0 excites, so "
            f"this machine as described would not produce a signal -- the axes are confused")
    return dot


def require_gradient_tensor_admissible(tensor, points, what, tol=1e-9):
    """Refuse a gradient-nonlinearity tensor that is diagonal, varying, and therefore not a field.

    ``tensor`` maps ``(n, 3)`` positions to ``(n, 3, 3)``. A diagonal ``L`` forces each coil's ``B_z`` to
    depend on its own axis alone, which Laplace then forces to be linear -- so a diagonal ``L`` is either
    the identity everywhere or impossible. This refuses the impossible case by name rather than letting a
    plausible-looking eigenframe come out 30 degrees wrong.
    """
    P = np.atleast_2d(np.asarray(points, dtype=np.float64))
    L = np.asarray(tensor(P), dtype=np.float64)
    if L.shape[-2:] != (3, 3):
        raise ValueError(f"{what} must give a (n, 3, 3) tensor, got {L.shape}")
    off = np.abs(L[..., ~np.eye(3, dtype=bool)].reshape(len(L), 6)).max()
    dev = np.abs(np.einsum("nii->ni", L) - 1.0).max()
    if off <= tol and dev > tol:
        raise ValueError(
            f"{what} is diagonal ({off:.1e} off-diagonal) yet varies in space (|L_ii - 1| up to {dev:.2e}), "
            f"which no magnetic field can do. A diagonal L makes each coil's B_z depend on its own axis "
            f"alone; Laplace then forces that dependence to be linear, so a diagonal L is the identity or "
            f"it is nothing. The off-diagonals are not a refinement -- on a real coil they are the same "
            f"size as the diagonal departure and carry the eigenframe rotation entirely")
    return off, dev
