"""Tier 1 of the scanner model's evaluation harness: a machine that exists only as geometry.

Every entry in the catalogue is a twin of a real machine, characterised from published measurements. This
is a twin of a machine nobody built, whose numbers are DERIVED rather than cited -- integrated from the
Biot-Savart law over an explicit wire -- and which therefore carries no measurement uncertainty at all.
That is what makes it an oracle: nothing here reads the catalogue, the fitted coefficients or the
solid-harmonic basis, so agreement with it is evidence and not bookkeeping (dmipy-sim#364).

It answers in the model's own currency. :meth:`GradientCoil.potential` returns ``{term: coefficient}`` in
the standard solid-harmonic basis, exactly what :meth:`ScannerLimits.gradient_potentials` returns, so the
two can be compared term by term rather than through a plot.

WHAT IT IS AN ORACLE FOR, and what it is not. This is magnetostatics in free space. It is exact for the
field of known currents, and from that it gives the harmonic content of a coil, the full gradient
nonlinearity tensor including the off-diagonals, and the concomitant field. It is silent about eddy
currents and the gradient impulse response (it contains no conducting structures), about passive shim iron
and magnet material (no permeable media), about RF above the quasi-static limit, and about anything
thermal.

The sharper limit is the one easiest to forget: it tests the model's MATHEMATICS, not its parameter
values. A Golay saddle is not a clinical gradient coil, which is stream-function designed and actively
shielded, so the absolute nonlinearity of anything built here is a property of the wire someone chose. Use
it to show that a diagonal ``L`` cannot happen; never to say what a particular magnet's ``L`` is.

Its own error budget, measured: 6.4e-6 against the analytic on-axis loop field, 1e-6 of scale on
``div B`` and ``curl B`` in the interior, and 6.3e-6 from wire discretisation at the default 720 segments.
"""
import numpy as np

from . import solid_harmonics

MU0 = 4.0e-7 * np.pi
SEGMENTS = 720                 # per closed turn; the discretisation error is 6.3e-6 here and falls as 1/n^2


def segment_field(a, b, points, current=1.0):
    """``B`` at ``points`` from a straight current segment running ``a -> b``, in tesla.

    The closed form of the Biot-Savart integral over a finite segment. Points ON the wire give zero rather
    than a singularity, which keeps a probe grid that happens to touch a turn from poisoning a whole field.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    P = np.atleast_2d(np.asarray(points, dtype=np.float64))
    r1, r2 = a - P, b - P
    n1 = np.linalg.norm(r1, axis=-1, keepdims=True)
    n2 = np.linalg.norm(r2, axis=-1, keepdims=True)
    denom = n1 * n2 * (n1 * n2 + np.sum(r1 * r2, axis=-1, keepdims=True))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (MU0 * current / (4.0 * np.pi)) * np.cross(r1, r2) * (n1 + n2) / denom
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def polyline_field(vertices, points, current=1.0):
    """``B`` at ``points`` from a current following a polyline, in tesla."""
    v = np.asarray(vertices, dtype=np.float64)
    B = np.zeros((len(np.atleast_2d(points)), 3))
    for a, b in zip(v[:-1], v[1:]):
        B += segment_field(a, b, points, current)
    return B


def circular_loop(radius, z, n=SEGMENTS):
    """One closed turn of radius ``radius`` in the plane ``z``, centred on the axis."""
    t = np.linspace(0.0, 2.0 * np.pi, int(n) + 1)
    return np.stack([radius * np.cos(t), radius * np.sin(t), np.full_like(t, float(z))], axis=-1)


class Coil:
    """A set of turns, each a polyline and a current. Its field is the sum of theirs."""

    def __init__(self, turns, name="coil"):
        self.turns = [(np.asarray(v, dtype=np.float64), float(i)) for v, i in turns]
        self.name = name

    def field(self, points):
        """``B`` at ``points``, ``(n, 3)`` in tesla."""
        B = np.zeros((len(np.atleast_2d(points)), 3))
        for v, i in self.turns:
            B += polyline_field(v, points, i)
        return B

    def bz(self, points):
        return self.field(points)[:, 2]


def maxwell_pair(radius=0.3, current=1.0, n=SEGMENTS):
    """Two coaxial turns of opposite current at the Maxwell separation -- a z gradient."""
    h = radius * np.sqrt(3.0) / 2.0
    return Coil([(circular_loop(radius, +h, n), +current),
                 (circular_loop(radius, -h, n), -current)], name="maxwell_pair")


def golay_saddle(radius=0.3, arc_deg=120.0, z_inner=0.15, z_outer=0.55, current=1.0, n=160):
    """Four saddles on a cylinder -- a transverse (x) gradient.

    A teaching geometry, not a clinical coil: it is unshielded and its winding was chosen for symmetry
    rather than for linearity, so its nonlinearity is a property of this choice and means nothing about any
    real machine. What it is good for is that it is genuinely a transverse gradient coil, so the STRUCTURE
    of what it produces -- which harmonics, off-diagonals comparable to the diagonal -- is a real coil's.
    """
    turns, half = [], np.deg2rad(arc_deg) / 2.0
    for z_lo, z_hi in ((z_inner, z_outer), (-z_outer, -z_inner)):
        for phi0, sign in ((0.0, +1.0), (np.pi, -1.0)):
            s = sign * (1.0 if z_lo > 0 else -1.0)
            t = np.linspace(phi0 - half, phi0 + half, 160)
            lo = np.stack([radius * np.cos(t), radius * np.sin(t), np.full_like(t, z_lo)], -1)
            hi = np.stack([radius * np.cos(t[::-1]), radius * np.sin(t[::-1]),
                           np.full_like(t, z_hi)], -1)
            turns.append((np.concatenate([lo, hi, lo[:1]]), s * current))
    return Coil(turns, name="golay_saddle")


class GradientCoil:
    """A coil read as a gradient axis: its field expressed the way the scanner model expresses one.

    ``axis`` is the direction the coil nominally encodes. ``potential`` is ``B_z`` divided by the gradient
    delivered at isocentre, in the standard solid-harmonic basis -- so it is directly comparable with
    :meth:`ScannerLimits.gradient_potentials`, and :meth:`gradient_tensor`'s columns are its gradients.
    """

    def __init__(self, coil, axis, radius=0.09):
        self.coil, self.axis, self.radius = coil, str(axis), float(radius)
        self._probe = self._sphere(800, seed=0)

    def _sphere(self, n, seed):
        rng = np.random.default_rng(seed)
        u = rng.normal(size=(n, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        return u * (rng.uniform(0.0, 1.0, (n, 1)) ** (1.0 / 3.0)) * self.radius

    def nominal_gradient(self, h=1e-3):
        """``dB_z/d(axis)`` at isocentre, in T/m -- what this coil calls unit gradient."""
        e = np.eye(3)["xyz".index(self.axis)]
        return float((self.coil.bz(h * e[None]) - self.coil.bz(-h * e[None]))[0] / (2.0 * h))

    def potential(self, order=4, points=None, significant=True):
        """``{term: coefficient}``: ``B_z / g_0`` least-squares projected onto the solid harmonics.

        The constant is fitted and discarded -- a uniform offset is B0's business, not a gradient's -- and
        every other term is kept, so the linear coefficient comes out near 1 for the nominal axis and the
        rest ARE the nonlinearity. Nothing here consults the catalogue.

        Two things this has to get right, or the numbers it reports mislead.

        The design matrix is COLUMN-SCALED before the solve. The terms carry units of ``1/m^(l-1)``, so at
        a 9 cm probe radius their columns span five orders of magnitude and the raw normal equations have a
        condition number of 7e4. Scaling brings it to 1.4 and the solve stops depending on how the basis
        happens to be normalised.

        And a coefficient is reported only if the term CONTRIBUTES more than the fit's own residual over
        the probe volume (``significant``). A coefficient is not a size: a Maxwell pair fits ``Z3X`` at
        +0.23, which looks like a gross violation of its own axial symmetry until one notices that ``Z3X``
        is order four, so at 9 cm it contributes 1.5e-5 against the linear term's 9e-2 -- a hundred times
        below the residual it is fitting. It is noise wearing a large number's clothes. Terms under the
        residual are dropped rather than reported, because an oracle that emits them invites exactly the
        comparison they cannot support.
        """
        P = self._probe if points is None else np.atleast_2d(np.asarray(points, dtype=np.float64))
        names = solid_harmonics.names_through(order)
        A = np.stack([solid_harmonics.evaluate({n: 1.0}, P) for n in names], axis=-1)
        A = np.concatenate([np.ones((len(P), 1)), A], axis=-1)
        scale = np.linalg.norm(A, axis=0)
        scale[scale == 0.0] = 1.0
        f = self.coil.bz(P) / self.nominal_gradient()
        c, *_ = np.linalg.lstsq(A / scale, f, rcond=None)
        c = c / scale
        out = {n: float(v) for n, v in zip(names, c[1:])}
        if not significant:
            return out
        varying = float(np.linalg.norm(f - np.mean(f)))
        floor = self.residual(order, P) * varying
        keep = {}
        for n, v in out.items():
            term = v * solid_harmonics.evaluate({n: 1.0}, P)
            if float(np.linalg.norm(term - np.mean(term))) > floor:
                keep[n] = v
        return keep

    def residual(self, order=4, points=None):
        """The share of the varying ``B_z`` the expansion does NOT reproduce.

        This is the honest report of where the basis stops. It falls as ``R^order``, so a residual that
        does not shrink with the probe radius is the coil's discretisation and not the truncation.
        """
        P = self._probe if points is None else np.atleast_2d(np.asarray(points, dtype=np.float64))
        f = self.coil.bz(P) / self.nominal_gradient()
        fit = solid_harmonics.evaluate(self.potential(order, P, significant=False), P)
        return float(np.linalg.norm(f - fit - np.mean(f - fit)) / np.linalg.norm(f - np.mean(f)))


def gradient_tensor(coils, points):
    """``L(r)`` from three :class:`GradientCoil`, ``{axis: coil}`` -- column ``j`` is ``grad Phi_j``.

    Derived from geometry alone, so it is the reference the catalogue's tensor is checked against. It has
    off-diagonals because a real coil has them, and no step here could have produced a diagonal tensor.
    """
    P = np.atleast_2d(np.asarray(points, dtype=np.float64))
    return np.stack([solid_harmonics.gradient(coils[j].potential(), P) for j in ("x", "y", "z")], axis=-1)


def concomitant_field(coil, points, b0_T, gradient_T_m=None):
    """The concomitant (Maxwell) field of ``coil`` at ``points``, in tesla, EXACTLY.

    A gradient coil cannot produce ``B_z`` alone: ``div B = 0`` and ``curl B = 0`` force transverse
    components, and to first order in ``1/B0`` the magnitude the spins see gains
    ``(B_x^2 + B_y^2) / (2 B0)``. That is the whole concomitant term, and here both components are
    integrated rather than expanded, so this needs no symmetry parameter and assumes no coil geometry --
    which is what lets it MEASURE the alpha that the catalogued expansion states.
    """
    P = np.atleast_2d(np.asarray(points, dtype=np.float64))
    B = coil.field(P)
    if gradient_T_m is not None:
        B = B * (float(gradient_T_m) / GradientCoil(coil, "z").nominal_gradient())
    return (B[:, 0] ** 2 + B[:, 1] ** 2) / (2.0 * float(b0_T))
