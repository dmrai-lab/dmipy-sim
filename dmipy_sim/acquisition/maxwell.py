"""Tier 0 of the scanner model's evaluation harness: the constraints Maxwell imposes for free.

These need no oracle, no reference machine and no measurement. They are properties any field the model
emits must have because of what a magnetic field IS, and each is checkable at the cost of a few
evaluations. A law that fails one is not an inaccurate field; it is not a field.

``require_harmonic``
    ``B_z`` satisfies Laplace's equation in a current-free region, so any expansion of the static field
    must too. This is the check that would have caught a ``c r^2`` inhomogeneity law on the day it was
    written: ``laplacian(r^2) = 6``, so the law places a forbidden interior minimum at isocentre while
    every downstream number stays finite.

``require_transverse``
    Only the component of B1 perpendicular to B0 excites, so a machine whose transmit axis lies along its
    field would produce no signal. Both axes are plausible unit vectors, so the confusion never announces
    itself.

``require_gradient_tensor_admissible``
    Column ``j`` of the gradient-nonlinearity tensor is ``grad Phi_j`` for a HARMONIC potential, and that
    is two conditions, both checked here: ``curl`` of the column vanishes (it is a gradient of something)
    and its ``divergence`` vanishes (that something is harmonic). Together they are necessary AND
    sufficient, which a test of the tensor's SHAPE is not. A diagonal ``L`` that is not the identity is
    impossible -- diagonal ``L`` makes each coil's ``B_z`` depend on its own axis alone and Laplace forces
    that dependence to be linear -- but refusing only that case leaves every other inadmissible tensor
    accepted, including one whose columns are not gradients at all.

WHAT THESE CHECKS CANNOT DO. They hold a field at the points they are given. A law non-harmonic everywhere
except on the probe set passes, and fields characterised on a DSV SHELL make that failure natural rather
than adversarial: ``r^4 - (10/3) R^2 r^2`` has a Laplacian vanishing exactly on ``r = R``. Probe a VOLUME,
not a surface, and prefer points with no relation to the model's own symmetries. The defaults here are a
generic scatter for that reason, and callers who can afford more points should pass more.
"""
import numpy as np

DEFAULT_H = 1e-4                 # metres; the stencil width
HARMONIC_TOL = 1e-6              # a genuine harmonic lands near roundoff, a violation near 1
TENSOR_TOL = 1e-4                # relative to the tensor's own variation; finite differences set the floor
_NOISE = 50.0 * np.finfo(np.float64).eps

#: A generic scatter in the unit ball. Deliberately not on an axis, a plane or a shell: a harmonic that is
#: odd in one coordinate has second differences that vanish identically on that coordinate's zero set, and
#: a field characterised on a shell can hide a Laplacian that vanishes only there.
UNIT_PROBE = np.array([
    [0.31, -0.47, 0.23], [-0.19, 0.11, -0.53], [0.41, 0.37, 0.17], [0.57, 0.29, -0.31],
    [-0.43, -0.22, 0.44], [0.13, 0.61, -0.19], [-0.51, 0.27, -0.09], [0.07, -0.33, 0.67],
    [0.62, -0.13, 0.41], [-0.28, 0.54, 0.36], [0.36, 0.19, -0.58], [-0.11, -0.64, -0.27]])


def probe_points(radius, centre=(0.0, 0.0, 0.0)):
    """A generic scatter of points filling a ball of ``radius`` -- the default place to test a law."""
    return np.asarray(centre, dtype=np.float64) + float(radius) * UNIT_PROBE


def _stencil(field, P, h, what):
    """``(f0, d2)`` with the fourth-order five-point second difference per axis, validated.

    Fourth order because a three-point stencil carries an error ``(h^2/12) sum d4/dxi4`` which does NOT
    vanish for a harmonic of order four: Z4 reported 1.4e-6 under it, and that was the stencil's failure
    rather than the field's. This is exact for polynomials through degree five.
    """
    n = len(P)
    f0 = np.asarray(field(P), dtype=np.float64)
    if f0.shape != (n,):
        raise ValueError(
            f"{what} must map (n, 3) positions to (n,) values; it gave {f0.shape} for {n} points. A field "
            f"returning a vector is summed against the stencil axes and scores a clean zero, so this is "
            f"checked rather than trusted")
    seen = [np.abs(f0)]
    d2 = []
    for e in np.eye(3):
        g = [np.asarray(field(P + k * h * e), dtype=np.float64) for k in (-2, -1, 1, 2)]
        if any(x.shape != (n,) for x in g):
            raise ValueError(f"{what} changed the shape of its output between points")
        seen.extend(np.abs(x) for x in g)
        d2.append(-g[0] / 12.0 + 4.0 * g[1] / 3.0 - 2.5 * f0 + 4.0 * g[2] / 3.0 - g[3] / 12.0)
    d2 = np.stack(d2, axis=0)
    if not np.all(np.isfinite(f0)) or not np.all(np.isfinite(d2)):
        raise ValueError(
            f"{what} is not finite at every point it was asked about, so it cannot be held to Laplace's "
            f"equation. A NaN compares False against every tolerance, so an unguarded check would report a "
            f"field that blows up as a clean one")
    return f0, d2, float(np.max([np.max(s) for s in seen]))


def harmonic_residual(field, points, h=DEFAULT_H):
    """How badly ``field`` fails Laplace's equation: ``|sum d2| / sum|d2|``, the worst over ``points``.

    ``field`` maps ``(n, 3)`` positions in metres to ``(n,)`` values in any unit. The three per-axis second
    differences are compared against their own total size, which makes the answer independent of the
    field's multiplicative scale and of its units: a harmonic gives roundoff because the three CANCEL,
    ``c r^2`` gives exactly 1.

    The magnitude scale is taken over EVERY evaluation, not over ``f0`` alone. Two failures follow from
    getting that wrong, and both were live. Scaling by ``max|f0|`` lets an additive offset raise the noise
    floor above a real violation, so ``1e3 + r^2`` -- the same defect written in absolute tesla rather than
    as ``dB/B0`` -- scored a clean 0. And where the law VANISHES on the probe set, ``max|f0|`` is zero, so
    the floor is zero and roundoff is divided by roundoff: a plain linear gradient probed on the bore axis
    was reported as not a magnetic field.

    A field whose variation is lost in its own offset is REFUSED rather than answered, because double
    precision cannot resolve it and "harmonic" would be a guess wearing a number's clothes.
    """
    P = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if P.ndim != 2 or P.shape[-1] != 3 or len(P) == 0:
        raise ValueError(f"points must be a non-empty (n, 3) array of positions in metres, got {P.shape}")
    f0, d2, mag = _stencil(field, P, float(h), "a field law")
    var = float(np.ptp(f0)) if len(f0) > 1 else 0.0
    if mag > 0.0 and var > 0.0 and var / mag < 1e-10:
        raise ValueError(
            f"this field's variation ({var:.2e}) is {var / mag:.1e} of its own magnitude ({mag:.2e}), so "
            f"differencing it loses the variation to roundoff and no verdict is possible. Pass the varying "
            f"part -- a uniform offset is not what Laplace constrains")
    total = np.abs(np.sum(d2, axis=0))
    size = np.sum(np.abs(d2), axis=0)
    # The sum of the three second differences means something only above the roundoff of forming them. A
    # field with no resolvable curvature at all -- a constant, or a bilinear one such as x y, whose pure
    # second derivatives vanish identically -- is harmonic for the same reason it is smooth.
    return float(np.max(np.where(total > _NOISE * mag, total / np.maximum(size, 1e-300), 0.0)))


def require_harmonic(field, points, what, tol=HARMONIC_TOL, h=DEFAULT_H):
    """Refuse a static-field law that does not solve Laplace's equation."""
    res = harmonic_residual(field, points, h=h)
    if res > tol:
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
    b0 = np.asarray(b0_axis, dtype=np.float64).ravel()
    b1 = np.asarray(b1_axis, dtype=np.float64).ravel()
    if b0.shape != (3,) or b1.shape != (3,):
        raise ValueError(f"{what} must declare its axes as 3-vectors, got {b0.shape} and {b1.shape}")
    n0, n1 = np.linalg.norm(b0), np.linalg.norm(b1)
    if not np.isfinite(n0) or not np.isfinite(n1) or n0 == 0.0 or n1 == 0.0:
        raise ValueError(
            f"{what} declares an axis that is not a direction ({tuple(b0)}, {tuple(b1)}). A zero or "
            f"non-finite axis makes every angle a NaN, and a NaN passes every tolerance it is compared to")
    dot = float(abs(b0 @ b1) / (n0 * n1))
    if dot > tol:
        raise ValueError(
            f"{what} declares a transmit axis {tuple(b1)} that is not perpendicular to its field "
            f"{tuple(b0)} (|cos| = {dot:.3f}). Only the component of B1 perpendicular to B0 excites, so "
            f"this machine as described would not produce a signal -- the axes are confused")
    return dot


def gradient_tensor_defects(tensor, points, h=DEFAULT_H):
    """``(max |div|, max |curl|, scale)`` over the columns of ``L`` -- what makes it a field, or not.

    Column ``j`` must be ``grad Phi_j``. A vector field is a gradient exactly when its curl vanishes, and
    that gradient's potential is harmonic exactly when its divergence vanishes. Both are finite-differenced
    from ``L`` itself and reported against the scale of its own Jacobian, so the answer is relative.
    """
    P = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if P.ndim != 2 or P.shape[-1] != 3 or len(P) == 0:
        raise ValueError(f"points must be a non-empty (n, 3) array, got {P.shape}")
    h = float(h)
    J = np.zeros((len(P), 3, 3, 3))                        # [point, column j, component i, derivative k]
    for k in range(3):
        d = np.zeros(3)
        d[k] = h
        hi = np.asarray(tensor(P + d), dtype=np.float64).reshape(len(P), 3, 3)
        lo = np.asarray(tensor(P - d), dtype=np.float64).reshape(len(P), 3, 3)
        J[..., k] = np.transpose((hi - lo) / (2.0 * h), (0, 2, 1))
    if not np.all(np.isfinite(J)):
        raise ValueError("the gradient tensor is not finite at every point it was asked about")
    div = np.abs(J[:, :, 0, 0] + J[:, :, 1, 1] + J[:, :, 2, 2]).max()
    curl = max(np.abs(J[:, :, 2, 1] - J[:, :, 1, 2]).max(),
               np.abs(J[:, :, 0, 2] - J[:, :, 2, 0]).max(),
               np.abs(J[:, :, 1, 0] - J[:, :, 0, 1]).max())
    return float(div), float(curl), float(np.max(np.abs(J)))


def require_gradient_tensor_admissible(tensor, points, what, tol=TENSOR_TOL, h=DEFAULT_H):
    """Refuse a gradient-nonlinearity tensor whose columns are not gradients of harmonic potentials.

    ``tensor`` maps ``(n, 3)`` positions to ``(n, 3, 3)``. This is the real condition and not a proxy for
    it. Testing instead that the tensor is not DIAGONAL-and-varying refuses one inadmissible tensor and
    accepts the rest: adding an off-diagonal of 1e-8 to that very defect made it pass, and a tensor whose
    columns are not gradients of anything passed untouched.
    """
    div, curl, scale = gradient_tensor_defects(tensor, points, h=h)
    if scale <= 0.0:
        return div, curl
    bad = []
    if div / scale > tol:
        bad.append(f"a column has divergence {div:.3e} ({div / scale:.1e} of the tensor's own variation), "
                   f"so its potential is not harmonic")
    if curl / scale > tol:
        bad.append(f"a column has curl {curl:.3e} ({curl / scale:.1e} of the tensor's own variation), so it "
                   f"is not the gradient of ANY potential")
    if bad:
        raise ValueError(
            f"{what} is not a magnetic field: " + "; and ".join(bad) + ". Column j of L is grad Phi_j for a "
            f"harmonic Phi_j, which is exactly these two conditions. A DIAGONAL L that is not the identity "
            f"fails them -- it makes each coil's B_z depend on its own axis alone, and Laplace then forces "
            f"that dependence to be linear, so a diagonal L is the identity or it is nothing")
    return div, curl
