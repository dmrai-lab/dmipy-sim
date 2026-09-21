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
    cross = np.cross(r1, r2)
    denom = n1 * n2 * (n1 * n2 + np.sum(r1 * r2, axis=-1, keepdims=True))
    # ON the wire both the numerator and the denominator are rounding noise, and their ratio is a FINITE
    # number of plausible size -- up to six orders above the field being measured. It is never a NaN, so it
    # is never caught, and one such point in a probe set destroyed a whole fit. An oracle may not return a
    # wrong number that looks right, so a denominator that is not resolvably non-zero is refused.
    floor = 8.0 * np.finfo(np.float64).eps * (n1 * n2) ** 2
    if np.any(np.abs(denom) <= floor):
        bad = int(np.argmax(np.abs(denom) <= floor))
        raise ValueError(
            f"point {tuple(np.round(P[bad], 6))} lies on (or within rounding of) the segment "
            f"{tuple(np.round(a, 4))} -> {tuple(np.round(b, 4))}, where the Biot-Savart integrand diverges. "
            f"The field there is not defined; move the probe off the conductor")
    return (MU0 * current / (4.0 * np.pi)) * cross * (n1 + n2) / denom


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


def golay_saddle(radius=0.3, arc_deg=120.0, z_inner=0.15, z_outer=0.55, current=1.0, n=320):
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
            t = np.linspace(phi0 - half, phi0 + half, int(n))
            lo = np.stack([radius * np.cos(t), radius * np.sin(t), np.full_like(t, z_lo)], -1)
            hi = np.stack([radius * np.cos(t[::-1]), radius * np.sin(t[::-1]),
                           np.full_like(t, z_hi)], -1)
            turns.append((np.concatenate([lo, hi, lo[:1]]), s * current))
    return Coil(turns, name="golay_saddle")


def rectangular_loop(plane_coord, width, length, axis=1, n=2):
    """One rectangular turn in the plane ``axis = plane_coord``, spanning ``width`` and ``length``.

    ``n`` subdivides each side, and it buys NOTHING: the sides are straight and the segment field is closed
    form, so two points per side is already exact and 200 gave the same answer to ten digits at a hundred
    times the cost. It exists only so a caller can refine a side that has been bent; leaving it looking like
    a convergence knob invited the belief that there was something to converge.
    """
    a, b = float(width) / 2.0, float(length) / 2.0
    i, j = [k for k in range(3) if k != int(axis)]
    corners = []
    for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)):
        p = np.zeros(3)
        p[int(axis)] = float(plane_coord)
        p[i], p[j] = su * a, sv * b
        corners.append(p)
    out = []
    for p, q in zip(corners[:-1], corners[1:]):
        t = np.linspace(0.0, 1.0, int(n))[:, None]
        out.append(p + t * (q - p))
    return np.concatenate(out)


def biplanar_pair(half_gap=0.15, width=0.30, length=0.30, axis=1, current=1.0, n=2):
    """Two opposed rectangular plates -- the AXIAL gradient coil of an open, bi-planar magnet.

    ``axis`` is the normal of the plates and the direction of both B0 and the gradient, so this is the coil
    whose divergence must be shared transversely and therefore the one that HAS an ``alpha``. The Swoop's
    ``b0_axis`` is ``(0, 1, 0)``, hence the default.

    What sets ``alpha`` is the pair ``(width / gap, length / gap)`` and NOT the plate aspect ratio alone.
    Square plates are four-fold symmetric about B0, so the two transverse directions are equivalent and
    ``alpha`` is forced to 1/2 whatever the gap. Away from square, both knobs move it over essentially the
    whole range: at a fixed aspect ratio of three, ``alpha`` runs 0.02 to 0.43 as the gap grows, because a
    pair whose gap dwarfs both plate dimensions degenerates to a dipole pair, which is axially symmetric
    again. ``alpha`` approaches 0 only when one plate dimension greatly exceeds the gap, so that the
    geometry approaches translational invariance along it and the whole divergence is pushed into the one
    remaining transverse direction.

    The aspect ratio alone is therefore not the physics: a catalogued ``alpha`` needs BOTH ratios, not one.
    """
    return Coil([(rectangular_loop(+half_gap, width, length, axis, n), +current),
                 (rectangular_loop(-half_gap, width, length, axis, n), -current)], name="biplanar_pair")


class GradientCoil:
    """A coil read as a gradient axis: its field expressed the way the scanner model expresses one.

    ``axis`` is the direction the coil nominally encodes. ``potential`` is ``B_z`` divided by the gradient
    delivered at isocentre, in the standard solid-harmonic basis -- so it is directly comparable with
    :meth:`ScannerLimits.gradient_potentials`, and :meth:`gradient_tensor`'s columns are its gradients.
    """

    def __init__(self, coil, axis, radius=0.09, n_ang=100, n_shell=8):
        self.coil, self.axis, self.radius = coil, str(axis), float(radius)
        self._probe = self._ball(int(n_ang), int(n_shell))

    def _ball(self, n_ang, n_shell):
        """A deterministic near-uniform quadrature: a Fibonacci sphere per shell, shells uniform in volume.

        NOT a random cloud. Solid harmonics are orthogonal over the sphere, but a random sample makes them
        only approximately so, and the residual correlation lets a large odd coefficient leak into a small
        even one. On a symmetric Maxwell pair, whose even content is EXACTLY zero, 800 random points
        reported Z2 at 7e-4 and the Golay's Z2X moved 3.40 to 3.51 across seeds -- a 3 per cent spread on a
        number quoted to four figures. The same count arranged as a quadrature gives 1.5e-7 and 0.04 per
        cent. Each shell is rotated so the shells do not align.
        """
        i = np.arange(n_ang) + 0.5
        phi = np.arccos(1.0 - 2.0 * i / n_ang)
        theta = np.pi * (1.0 + 5.0 ** 0.5) * i
        u = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=-1)
        out = []
        for k in range(n_shell):
            t = 2.0 * np.pi * k / n_shell
            R = np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])
            out.append(self.radius * ((k + 0.5) / n_shell) ** (1.0 / 3.0) * (u @ R.T))
        return np.concatenate(out)

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

    def detection_floor(self, order=4, points=None):
        """The smallest coefficient of each term this fit could have distinguished from its own residual.

        :meth:`potential` drops terms contributing less than the residual, so a term below this is reported
        as ABSENT rather than as small -- and absence reads like a symmetry. A Maxwell pair whose two loops
        differ by 0.5 per cent has a genuine Z2 of -8e-3 and is reported as perfectly linear; the floor says
        why. The floor is set by where the basis stops, so it is a statement about the expansion and not
        about the coil.
        """
        P = self._probe if points is None else np.atleast_2d(np.asarray(points, dtype=np.float64))
        f = self.coil.bz(P) / self.nominal_gradient()
        bar = self.residual(order, P) * float(np.linalg.norm(f - np.mean(f)))
        out = {}
        for n in solid_harmonics.names_through(order):
            t = solid_harmonics.evaluate({n: 1.0}, P)
            spread = float(np.linalg.norm(t - np.mean(t)))
            out[n] = bar / spread if spread > 0.0 else float("inf")
        return out

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


def concomitant_field(coil, points, b0_T, b0_axis=(0.0, 0.0, 1.0), gradient_T_m=None):
    """The concomitant (Maxwell) field of ``coil`` at ``points``, in tesla, EXACTLY.

    A gradient coil cannot produce a field along B0 alone: ``div B = 0`` and ``curl B = 0`` force transverse
    components, and to first order in ``1/B0`` the magnitude the spins see gains ``|B_perp|^2 / (2 B0)``,
    where perpendicular means perpendicular to B0. Both components are INTEGRATED rather than expanded, so
    this needs no symmetry parameter and assumes no coil geometry -- which is what lets it MEASURE the alpha
    a catalogued expansion states.

    ``b0_axis`` is which way the field points, and it is not decoration. On a Halbach magnet B0 is
    TRANSVERSE to the bore, so the two components that count are not the two a cylindrical magnet's would
    be; taking them to be x and y there computes a real number for the wrong machine.
    """
    P = np.atleast_2d(np.asarray(points, dtype=np.float64))
    B = coil.field(P)
    n = np.asarray(b0_axis, dtype=np.float64)
    n = n / np.linalg.norm(n)
    if gradient_T_m is not None:
        # scale to a stated gradient, measured ALONG B0 -- the axial gradient this coil delivers. Reading it
        # along a fixed z instead divides by nearly zero for any machine whose B0 is not the bore axis.
        d = 2e-3 * n
        g0 = float((coil.field(d[None]) - coil.field(-d[None]))[0] @ n) / (2.0 * 2e-3)
        # relative, not absolute: g0 scales with the winding current, so a fixed 1e-18 floor let the SAME
        # transverse coil refuse at 1 A and return 1/noise at 10 A.
        probe = np.eye(3) * 2e-3
        scale = float(np.max(np.abs(coil.field(probe) - coil.field(-probe)))) / (2.0 * 2e-3)
        if abs(g0) < 1e-9 * max(scale, 1e-300):
            raise ValueError(
                f"this coil delivers no gradient along b0_axis ({g0:.2e} T/m against {scale:.2e} T/m "
                f"across its own axes), so a requested gradient_T_m cannot be referred to it -- scale a "
                f"TRANSVERSE coil by its own axis instead")
        B = B * (float(gradient_T_m) / g0)
    perp = B - np.outer(B @ n, n)
    return np.sum(perp ** 2, axis=-1) / (2.0 * float(b0_T))


def transverse_block(coil, b0_axis=(0.0, 0.0, 1.0), h=2e-3):
    """``(eigenvalues, axes, g)`` of the coil's TRANSVERSE gradient block, in the plane perpendicular to B0.

    An axial coil's ``div B = 0`` forces ``dB_u/du + dB_v/dv = -g``, but how that total is shared is a 2x2
    symmetric object, not a number: ``curl B = 0`` makes ``dB_u/dv = dB_v/du``, so the block has principal
    axes of its own and a cross term in any other frame. Everything invariant about the sharing lives here;
    a scalar alpha is a reading of it along a STATED direction.
    """
    n = np.asarray(b0_axis, dtype=np.float64)
    n = n / np.linalg.norm(n)
    u = np.cross(n, (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0))
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    def block(step):
        def slope(direction, component):
            d = float(step) * np.asarray(direction, dtype=np.float64)
            return float((coil.field(d[None]) - coil.field(-d[None]))[0] @ component) / (2.0 * float(step))
        gg = slope(n, n)
        MM = np.array([[slope(u, u), slope(u, v)], [slope(v, u), slope(v, v)]])
        return gg, 0.5 * (MM + MM.T)           # curl-free makes it symmetric; symmetrise off the noise

    # div B = 0 is EXACT, so the finite difference must reproduce it. Where it does not, the step is too
    # coarse for the geometry and the answer is truncation, not physics -- at a half gap of 2 cm the fixed
    # 2 mm step broke the identity by 4x alpha itself and the guard below then refused a perfectly good
    # axial coil, blaming its geometry. Refine until the identity holds, and say so if it never does.
    step = float(h)
    for _ in range(8):
        g, M = block(step)
        if g == 0.0 or abs(np.trace(M) + g) <= 1e-6 * abs(g):
            break
        step *= 0.25
    else:
        raise ValueError(
            f"this coil's field cannot be differenced consistently near isocentre: div B comes out "
            f"{abs(np.trace(M) + g) / max(abs(g), 1e-300):.1e} of the axial gradient at a step of {step:.1e} m, "
            f"where it must be zero. The transverse sharing read from it would be truncation error")
    w, V = np.linalg.eigh(M)
    order = np.argsort(-np.abs(w))            # the axis carrying most of the sharing first
    w = w[order]
    axes = np.stack([V[0, order[0]] * u + V[1, order[0]] * v,
                     V[0, order[1]] * u + V[1, order[1]] * v])
    return w, axes, g


def concomitant_alpha(coil, b0_axis=(0.0, 0.0, 1.0), transverse_axis=None, h=2e-3):
    """The symmetry parameter ``alpha`` of an AXIAL gradient coil, measured from its geometry.

    ``alpha`` is how the coil's divergence is shared between the two directions transverse to B0:
    ``dB_u/du = -alpha g`` and ``dB_v/dv = -(1 - alpha) g``. That statement is incomplete on its own, because
    the sharing is a 2x2 block and a scalar is a reading of it along ONE direction. Read along the block's
    own PRINCIPAL axes by default, which is the only choice that depends on the coil rather than on the
    frame the caller happens to be using; ``transverse_axis`` reads it along a stated direction instead.

    The default matters: deriving the transverse frame from a global axis makes ``alpha`` a property of the
    coordinate system. A bi-planar pair of aspect ratio three reports 0.05 at one azimuth and 0.95 rotated
    ninety degrees about its own B0, which is the same coil and the same physics.

    ``alpha = 1/2`` also does NOT imply a symmetric coil. Any block with equal eigenvalues reads 1/2, and so
    does an asymmetric coil read at 45 degrees to its own principal axes, where the concomitant field the
    alpha model predicts is wrong by orders of magnitude. :func:`transverse_block` is what to consult when
    that distinction matters; ``cross_term`` below reports it.

    Nothing about the MAGNET enters, only the direction of its field, because that is what defines
    transverse. A Halbach's B0 comes from magnetised blocks rather than free currents, so a route to alpha
    needing the magnet would leave this oracle's domain. This one does not.
    """
    w, axes, g = transverse_block(coil, b0_axis, h)
    if abs(g) < max(abs(w[0]), abs(w[1])):
        raise ValueError(
            f"alpha is a property of the AXIAL coil -- the one whose gradient lies along B0, so that its "
            f"divergence must be shared between the two TRANSVERSE directions. This coil's steepest "
            f"variation is transverse to the b0_axis given (along-axis {g:.3e} T/m against transverse "
            f"{max(abs(w[0]), abs(w[1])):.3e} T/m), so it is a transverse coil in this frame and has no "
            f"alpha. Reading one anyway returns a finite number that means nothing")
    if transverse_axis is None:
        return float(-w[0] / g)
    t = np.asarray(transverse_axis, dtype=np.float64)
    n = np.asarray(b0_axis, dtype=np.float64)
    n = n / np.linalg.norm(n)
    t = t - (t @ n) * n
    if np.linalg.norm(t) < 1e-12:
        raise ValueError("transverse_axis lies along b0_axis, so it names no transverse direction")
    t /= np.linalg.norm(t)
    c = axes @ t
    return float(-(w[0] * c[0] ** 2 + w[1] * c[1] ** 2) / g)


def cross_term(coil, b0_axis=(0.0, 0.0, 1.0), h=2e-3):
    """How far the transverse sharing is from being describable by a single ``alpha``, as a fraction of ``g``.

    Zero when the coil's principal axes are the ones alpha is read along. When it is not small the scalar
    alpha is an incomplete description however it is read, and a concomitant field built from alpha alone
    can be wrong by orders of magnitude rather than by a correction.
    """
    w, _axes, g = transverse_block(coil, b0_axis, h)
    return float(abs(w[0] - w[1]) / abs(g)) if g else float("inf")
