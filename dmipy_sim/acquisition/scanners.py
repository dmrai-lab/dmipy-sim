"""The scanner as a parameterised forward model of a machine, and the save-interval rule a persistent walk
derives its grid from.

:class:`ScannerLimits` is not a limits table with fields bolted on: it is the model, and the catalogue is a set of
parameter vectors for it -- one per machine there are citable numbers for. A consumer never asks which machine it
has and branches; it asks the model for a field, a gradient, a transmit scale or a drift at a point, and the model
answers from whatever parameters it was given. An entry resolved by name and one a site constructs from its own
measurements take the identical path, which is what makes this a catalogue of TWINS rather than of presets: the
published entries characterise a model of machine, and the same parameter vector filled in for one serial number
is a twin of that unit. See ``docs/scanner-model.md``.

Every number here is read from :mod:`dmipy_sim.acquisition.scanner_constants` (the cited catalogue); this module
carries none. A consumer takes it as a designer's constraint set, a Pulseq exporter's ``Opts``, the walk's save
grid, or a field law -- resolved from any of a scanner's names and a slew ``regime``.

The frames are three and are not conflated: the substrate's (the walk and the pack), the patient's (RAS: the
phantom grid), and the magnet's (+z along ``b0_axis``, where the field maths is written). ``b0_axis`` and
``b1_axis`` are parameters and not constants, since a Halbach magnet's B0 is not along its bore; ``magnet_frame``
is the rotation between the last two, and a catalogue entry whose transmit axis is not perpendicular to its B0 is
refused rather than expressed. A field law is evaluated only in the harmonic basis
(:mod:`dmipy_sim.acquisition.solid_harmonics`), because ``B_z`` satisfies Laplace's equation in a current-free
region and a law that does not is not a magnetic field. :data:`SCANNERS` is
the replay band-limit certificate's class table (dmipy-sim #143), the same classes that size K per scanner, as a
view of the catalogue: ``(G_max, slew_max)`` per class.

``save_interval`` is the rule for ``dt_save``: between two saves the walker moves and the replay's phase integral
treats it as sitting at the sample; for a Brownian path the phase error has variance
``(2/3) gamma^2 D dt^2 int |G|^2 dt`` and the relative signal error is half of it. The adversarial deliverable
waveform is ``G = Gmax`` for the whole echo time, so the scanner enters only through ``Gmax`` (as it enters the K
bound only through slew). Held to a fraction ``f`` of the Monte-Carlo floor ``1 / sqrt(N_w)``:

    dt <= sqrt( 3 f / ( sqrt(N_w) gamma^2 D Gmax^2 T ) )

Geometry sets nothing here: walls only shrink displacements, occupancy and wall contact are accumulated per save by
the walk itself. The field tier samples the field basis at the saves; its error is second order in ``dt`` and was
measured at 3e-4 in |E| at dt = 50 us in the harshest catalogued case (7 T, chi 1.06e-6, TE 40 ms), so a walk with
a field source is capped at ``FIELD_DT_CAP`` (#143). The codec needs only ``n_t >= 2K + 2``.
"""
from dataclasses import dataclass

import numpy as np

from ..constants import GAMMA
from . import maxwell, scanner_constants as scc, solid_harmonics

GAMMA_BAR = GAMMA / (2.0 * np.pi)

__all__ = ["ScannerLimits", "SCANNERS", "FIELD_DT_CAP", "scanner_limits", "save_interval"]



#: the patient-frame direction each letter names, in RAS -- the same language ``Grid.axes`` speaks, so a
#: scanner's axes and a grid's axes are comparable without a second convention to keep true.
_AXIS_LETTER = {"R": (1.0, 0.0, 0.0), "L": (-1.0, 0.0, 0.0),
                "A": (0.0, 1.0, 0.0), "P": (0.0, -1.0, 0.0),
                "S": (0.0, 0.0, 1.0), "I": (0.0, 0.0, -1.0)}


def _axis(letter, what, name):
    """A patient-frame unit vector from an axis letter, or ``None`` when the machine declares none."""
    if letter is None:
        return None
    key = str(letter).upper()
    if key not in _AXIS_LETTER:
        raise ValueError(
            f"{what} of {name!r} is an axis letter, one of {sorted(_AXIS_LETTER)}; got {letter!r}")
    return _AXIS_LETTER[key]


@dataclass(frozen=True)
class ScannerLimits:
    """What a scanner can deliver, in SI, resolved from the catalogue.

    ``None`` means the catalogue does not know (the leaf is absent or ``NEEDS VERIFICATION``); it is never a
    number standing in for one. ``slew_max`` is the ``regime``'s: ``"default"`` is the hardware figure,
    ``"diffusion"`` the PNS-derated figure a diffusion sequence is actually bound by where the catalogue has one
    (Connectom: 200 -> 62.5 T/m/s), else the hardware figure.
    """
    name: str                  # the resolved catalogue key
    kind: str                  # "scanner" (a machine) or "envelope" (a declared limit point)
    regime: str
    G_max: float               # T/m
    slew_max: float            # T/m/s
    grad_raster: float = None  # s
    rf_raster: float = None    # s
    adc_raster: float = None   # s
    rf_dead_time: float = None
    rf_ringdown_time: float = None
    adc_dead_time: float = None
    peak_B1: float = None      # T, body coil where the catalogue has it, else head coil
    field_T: float = None      # T, the static field; None for an envelope or an uncatalogued field
    b0_axis: tuple = None      # patient-frame unit vector along B0; the magnet frame's +z
    b1_axis: tuple = None      # patient-frame unit vector along the transmit coil, None for a birdcage
    gradient_axis_assignment: tuple = None  # which patient direction each vendor axis names; None = not known
    b0_harmonic_Z2: float = None      # 1/m^2, the standard shim term 2z^2-x^2-y^2, in magnet axes
    b0_harmonic_Z2X: float = None     # 1/m^3, the standard shim term x(4z^2-x^2-y^2): the R/L asymmetry
    b0_direction_spread_deg: float = None  # how far admissible laws disagree about the gradient direction
    b0_asymmetry_axis: tuple = None   # patient-frame unit vector the odd harmonic is odd along
    b0_validity_radius: float = None  # m, how far from isocentre that law is anchored
    b1_axial_falloff: float = None    # 1/m^2, the coefficient of kappa_B1 = 1 - a z^2 along the bore
    b1_calibration_offset: float = None  # a systematic transmit scale, 1 = nominal
    b0_temperature_coefficient: float = None  # 1/K, dB0/B0 per kelvin of magnet temperature
    f0_recentering_interval: float = None     # s, how long the field drifts before f0 is re-set
    f0_temperature_slope: float = None        # Hz/K, the same coefficient as published
    d_scale_x_dx: float = None    # 1/m, dL_xx/dx -- the gradient-nonlinearity tensor's x dependence
    d_scale_y_dx: float = None    # 1/m, dL_yy/dx
    d_scale_z_dx: float = None    # 1/m, dL_zz/dx
    gradient_completion: float = None   # share of the x coil's curvature borrowed from y (see
                                        # gradient_potentials); the even 1/2 when unstated

    @classmethod
    def of(cls, scanner, *, regime="default"):
        """Resolve ``scanner`` -- a certificate class (``"connectom"``), an alias (``"siemens_prisma"``), a model
        key, an envelope key, or an explicit ``(G_max, slew_max)`` pair -- to its limits."""
        if isinstance(scanner, cls):
            return scanner
        if not isinstance(scanner, str):
            g, sl = scanner
            return cls(name="explicit", kind="envelope", regime=regime, G_max=float(g), slew_max=float(sl))
        key, entry, kind = scc.resolve(scanner)
        grad = entry.get("gradient", {})
        slew_name = ("max_slew_rate_diffusion" if regime == "diffusion" and "max_slew_rate_diffusion" in grad
                     else "max_slew_rate")
        G_max, slew = scc.leaf_si(entry, "gradient", "max_amplitude"), scc.leaf_si(entry, "gradient", slew_name)
        if G_max is None or slew is None:
            raise ValueError(f"{key!r} has no verified gradient amplitude / slew in the catalogue")
        limits = cls(name=key, kind=kind, regime=regime, G_max=G_max, slew_max=slew,
                   grad_raster=scc.leaf_si(entry, "gradient", "gradient_raster_time"),
                   rf_raster=scc.leaf_si(entry, "rf", "rf_raster_time"),
                   adc_raster=scc.leaf_si(entry, "rf", "adc_dwell_raster_time"),
                   rf_dead_time=scc.leaf_si(entry, "rf", "rf_dead_time"),
                   rf_ringdown_time=scc.leaf_si(entry, "rf", "rf_ringdown_time"),
                   adc_dead_time=scc.leaf_si(entry, "rf", "adc_dead_time"),
                   peak_B1=(scc.leaf_si(entry, "rf", "peak_B1_body_coil")
                            if scc.leaf_si(entry, "rf", "peak_B1_body_coil") is not None
                            else scc.leaf_si(entry, "rf", "peak_B1_head_coil")),
                   field_T=(float(entry["field_T"]) if entry.get("field_T") is not None else None),
                   b0_harmonic_Z2=scc.leaf_si(entry, "homogeneity", "b0_harmonic_Z2"),
                   b0_harmonic_Z2X=scc.leaf_si(entry, "homogeneity", "b0_harmonic_Z2X"),
                   b0_direction_spread_deg=scc.leaf_si(entry, "homogeneity", "b0_direction_spread_deg"),
                   b0_asymmetry_axis=_axis(scc.leaf_raw(entry, "homogeneity", "b0_asymmetry_axis"),
                                          "b0_asymmetry_axis", key),
                   b0_validity_radius=scc.leaf_si(entry, "homogeneity", "b0_validity_radius"),
                   b1_axial_falloff=scc.leaf_si(entry, "rf", "b1_axial_falloff"),
                   b1_calibration_offset=scc.leaf_si(entry, "rf", "b1_calibration_offset"),
                   b0_temperature_coefficient=scc.leaf_si(entry, "thermal",
                                                          "b0_temperature_coefficient"),
                   f0_recentering_interval=scc.leaf_si(entry, "thermal", "f0_recentering_interval"),
                   f0_temperature_slope=scc.leaf_si(entry, "thermal", "f0_temperature_slope"),
                     d_scale_x_dx=scc.leaf_si(entry, "gradient_nonlinearity", "d_scale_x_dx"),
                     d_scale_y_dx=scc.leaf_si(entry, "gradient_nonlinearity", "d_scale_y_dx"),
                     d_scale_z_dx=scc.leaf_si(entry, "gradient_nonlinearity", "d_scale_z_dx"),
                     gradient_completion=scc.leaf_si(entry, "gradient_nonlinearity",
                                                     "gradient_completion"),
                     b0_axis=_axis(scc.leaf_raw(entry, "frame", "b0_axis"), "b0_axis", key),
                     b1_axis=_axis(scc.leaf_raw(entry, "frame", "b1_axis"), "b1_axis", key))
        limits._check_frame()
        return limits

    def _check_frame(self):
        """Tier 0: refuse a machine whose transmit coil is not perpendicular to its field, and a field law
        that does not solve Laplace's equation.

        Both are constraints Maxwell imposes for free (dmipy-sim#364) -- they need no oracle and no
        measurement, and a catalogue entry that fails one describes something that is not a machine. The
        law is checked at CONSTRUCTION rather than at first use, so a bad coefficient cannot reach a caller
        that happens never to evaluate the field off-axis.
        """
        maxwell.require_transverse(self.b0_axis, self.b1_axis, repr(self.name))
        law = self.harmonic_law()
        if law:
            P = maxwell.probe_points(0.5 * (self.b0_validity_radius or 0.1))
            maxwell.require_harmonic(lambda q: solid_harmonics.evaluate(law, q), P,
                                     f"the catalogued B0 law for {self.name!r}")

    def magnet_frame(self):
        """The rotation taking PATIENT axes to the MAGNET frame, whose +z is B0 by construction.

        Every piece of field physics -- the concomitant expansion, the EPG states, off-resonance, the
        susceptibility contraction -- is written for B0 along +z, and none of them says so in a way a caller
        can check. This is where that assumption becomes a value: the physics is evaluated in the frame this
        returns, and a machine whose field is not along the patient's head-foot axis gets a rotation instead
        of a silently wrong answer.

        On a cylindrical magnet this is the identity up to the choice of transverse axes, which is why the
        assumption has survived so long unexamined. On a bi-planar magnet with a vertical field it is not.

        ``None`` when the machine declares no field direction.
        """
        if self.b0_axis is None:
            return None
        z = np.asarray(self.b0_axis, dtype=np.float64)
        z = z / np.linalg.norm(z)
        # a transverse axis to build the frame on: the coil's if there is one, else any vector not parallel
        # to the field. Which one it is does not matter for anything axially symmetric about B0, and the
        # cases that are not -- the magnet's own R/L asymmetry -- are catalogued in PATIENT axes anyway.
        seed = self.b0_asymmetry_axis if self.b0_asymmetry_axis is not None else self.b1_axis
        seed = None if seed is None else np.asarray(seed, dtype=np.float64)
        if seed is None or abs(float(z @ (seed / np.linalg.norm(seed)))) > 1 - 1e-9:
            seed = np.eye(3)[int(np.argmin(np.abs(z)))]
        x = seed - (seed @ z) * z
        x = x / np.linalg.norm(x)
        return np.stack([x, np.cross(z, x), z])          # rows: the magnet frame's axes in patient RAS

    def _field_coords(self, offset_m):
        """``(zeta, xi, eta)`` -- a displacement resolved onto the field law's own axes: ``zeta`` along B0,
        ``xi`` along the magnet's asymmetry, ``eta`` completing a right-handed set.

        The harmonics are written in these coordinates and never in raw array indices, because an index means
        whatever the caller's frame happens to be and these mean something about the magnet.
        """
        d = np.atleast_2d(np.asarray(offset_m, dtype=np.float64))
        R = self.magnet_frame()
        if R is None:
            raise ValueError(f"{self.name!r} declares no b0_axis, so its field law has no frame to live in")
        m = d @ R.T                                   # rows of R are the magnet axes in patient coordinates
        return m[..., 2], m[..., 0], m[..., 1]

    def harmonic_law(self):
        """The catalogued field law as ``{term name: coefficient}`` in the standard solid-harmonic basis."""
        return {n: c for n, c in (("Z2", self.b0_harmonic_Z2), ("Z2X", self.b0_harmonic_Z2X)) if c}

    def _harmonics(self, offset_m):
        """``(value, gradient)`` of ``dB/B0`` in the magnet frame, from the catalogued solid harmonics.

        The terms and their gradients are :mod:`dmipy_sim.acquisition.solid_harmonics`' and are not written
        out again here: a law and the basis it is written in must be the same objects, or the two drift and
        only one of them is the one that was checked.
        """
        zeta, xi, eta = self._field_coords(offset_m)
        r = np.stack([xi, eta, zeta], axis=-1)          # the basis is in (x, y, z) = (xi, eta, zeta)
        law = self.harmonic_law()
        if not law:
            return np.zeros(zeta.shape), np.zeros(zeta.shape + (3,))
        return solid_harmonics.evaluate(law, r), solid_harmonics.gradient(law, r)

    def _refuse_outside(self, offset_m, what):
        r = np.linalg.norm(np.atleast_2d(np.asarray(offset_m, dtype=np.float64)), axis=-1)
        if self.b0_validity_radius is not None and float(np.max(r)) > self.b0_validity_radius:
            raise ValueError(
                f"the field law for {self.name!r} is a solid-harmonic expansion anchored at "
                f"{self.b0_validity_radius*100:.0f} cm from isocentre and something here is "
                f"{float(np.max(r))*100:.1f} cm out. Beyond it the truncation is an extrapolation in the "
                f"orders that were never constrained as well as in radius, so {what} is refused")

    def G_max_along(self, direction):
        """The strongest gradient this machine can deliver along a patient-frame ``direction``, in T/m.

        This exists to be REFUSED more often than answered, and the refusal is the point. A machine's per-axis
        amplitudes are only usable if you know which physical direction each vendor axis names, and for a
        bi-planar magnet that mapping is not in the public record: the regulatory filing and the peer-reviewed
        literature disagree about the outlier axis by a factor of 1.5, and the one primary document that ties
        a Hyperfine coil label to the field direction contradicts the inference every other source supports.

        So a machine that has not declared ``gradient_axis_assignment`` raises, naming the conflict, rather
        than picking the better-supported reading and returning a number that looks like a measurement. Use
        :attr:`G_max` for a deliverability check -- it is the WEAKEST axis and is therefore safe whatever the
        assignment turns out to be, which is why every builder already uses it.
        """
        if self.gradient_axis_assignment is None:
            raise ValueError(
                f"{self.name!r} does not declare which patient direction each of its gradient axes names, so "
                f"the strongest gradient along a direction cannot be stated. For this machine the mapping is "
                f"genuinely not public and the sources conflict -- see the catalogue's "
                f"per_axis_amplitude_fda and per_axis_amplitude_literature leaves, which disagree by a "
                f"factor of 1.5 on the third axis. Use G_max ({self.G_max*1e3:.1f} mT/m), the weakest axis, "
                f"which is safe under every candidate assignment")
        d = np.asarray(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        A = np.asarray(self.gradient_axis_assignment, dtype=np.float64)   # rows: patient direction per axis
        return float(np.min(self.G_max / np.abs(A @ d).clip(1e-12)))

    def b0_offset(self, offset_m):
        """The static field's departure from uniformity at a displacement from isocentre, in **tesla**.

        A magnet's field in the imaging volume solves Laplace's equation, so it is a sum of SOLID HARMONICS
        and can be nothing else. This evaluates the catalogued expansion: a zonal Z2 -- the bowl every magnet
        has -- plus an l=3, m=1 term odd along the magnet's declared asymmetry axis.

        Why the asymmetry sits at order THREE rather than one. The published homogeneity is a post-linear-shim
        residual, so the l=1 content has been nulled and cannot be fitted to it; but the magnet is still
        asymmetric, and an odd asymmetry with no l=1 lives at l=3. Order three is independently required by
        the two published figures, which no l<=2 expansion can reach.

        A consequence worth knowing, because it reverses what an inadmissible bowl suggested: every harmonic
        of order two or more has ZERO gradient at the origin, so a linearly shimmed magnet does not encode
        diffusion at isocentre. The background gradient grows from nothing.

        ``offset_m`` is ``(..., 3)`` in metres in PATIENT axes; the law resolves it onto its own frame
        itself. ``None`` when the machine publishes no profile, which is every machine but a permanent-magnet
        one. Refused beyond ``b0_validity_radius``.
        """
        if self.b0_harmonic_Z2 is None or self.field_T is None:
            return None
        self._refuse_outside(offset_m, "the field")
        val, _g = self._harmonics(offset_m)
        out = self.field_T * val
        return out if np.ndim(offset_m) > 1 else float(out[0])

    def b0_gradient(self, offset_m):
        """The SPATIAL GRADIENT of the static field at a displacement from isocentre, in **T/m** -- the
        magnet's own encoding gradient, on during every pulse and every dead time because a magnet does not
        switch off.

        The analytic derivative of :meth:`b0_offset`, returned in PATIENT axes. Being the gradient of a
        harmonic function it is divergence-free, which the r^2 bowl it replaces was not -- that one implied a
        monopole in the gradient field.

        It VANISHES at isocentre, and that is the linear shim rather than an accident: every l>=2 harmonic has
        zero gradient at the origin. ``None`` when the machine publishes no profile; refused beyond the
        anchor radius, where a truncated expansion extrapolates in order as well as in radius.
        """
        if self.b0_harmonic_Z2 is None or self.field_T is None:
            return None
        self._refuse_outside(offset_m, "its derivative")
        _v, g_mag = self._harmonics(offset_m)
        out = self.field_T * (g_mag @ self.magnet_frame())      # magnet axes back into patient axes
        return out.reshape(np.shape(offset_m)) if np.ndim(offset_m) > 1 else out[0]

    def gradient_potentials(self):
        """Each gradient coil's field as a SOLID-HARMONIC expansion, ``{axis: {term: coefficient}}``.

        ``Phi_j`` is coil ``j``'s ``B_z`` divided by the gradient it delivers at isocentre, so its leading
        term is the linear harmonic of that axis and ``grad Phi_j`` is column ``j`` of :meth:`gradient_tensor`.
        Writing the coils this way is what makes the tensor a FIELD: a harmonic expansion cannot express an
        inadmissible ``L``, where a tensor assembled element by element can and did.

        WHAT IS MEASURED AND WHAT IS CHOSEN. The NIST regression gives the DIAGONAL and nothing else:
        ``dL_jj/dx = a_j``. Harmonicity and ``L(0) = I`` do NOT then determine the rest. For each coil the
        admissible potentials are a four-parameter affine family in this basis (six in the full order-four
        space, which this basis does not carry): any two-dimensional harmonic in the OTHER two coordinates,
        with zero gradient at the origin, may be added without touching the measured diagonal -- for the y
        coil ``x^2 - z^2``, ``xz`` and their cubic partners. The freedom is the same for all three coils.

        What ships is the MINIMUM-NORM member of that family. That is not a tie-break dressed up: every free
        direction is a distinct solid harmonic and so is L2-orthogonal to the particular solution, which
        makes ``h = 0`` the unique minimiser of the added field energy rather than one of many. It is also
        the lowest-order member, and the only one that keeps each coil's own reflection parity -- ``B_z`` odd
        in the axis it encodes, the ideal Golay and Maxwell winding symmetry. Under that parity assumption
        the family collapses to a point and the off-diagonals ``L_xy = a_y y`` and ``L_xz = a_z z`` follow
        from the measured diagonal; without it they are a modelling choice, and the assumption is not in the
        data. Note the model already asserts that ONE ideal parity is broken, since a non-zero ``a_j`` is
        exactly a breach of parity in x.

        The choice is not small. An admissible alternative that breaks the y coil's parity by as much as the
        measured effect moves ``L`` by the whole modelled nonlinearity and a b value by a few per cent.

        ``gradient_completion`` is the one member of that freedom this catalogue EXPOSES rather than the only
        one that exists: the share of the x coil's curvature taken from y, at the least-committed 1/2, which
        is also that sub-family's minimum-norm point. It costs little -- over the validity radius the whole
        ``lam`` family spans 3.5 per cent of the nonlinearity and 0.001 per cent of a b value, because
        ``a_x`` is the smallest of the three coefficients. It is one line inside the four-dimensional kernel
        above, not a different kind of thing.
        """
        if self.d_scale_y_dx is None:
            return None
        ax = self.d_scale_x_dx or 0.0
        lam = 0.5 if self.gradient_completion is None else float(self.gradient_completion)
        # (a_x/2)(x^2 - y^2) = (a_x/2) X2Y2 ; (a_x/2)(x^2 - z^2) = (a_x/4)(X2Y2 - Z2)
        return {
            "x": {"X": 1.0, "X2Y2": 0.5 * ax * lam + 0.25 * ax * (1.0 - lam),
                  "Z2": -0.25 * ax * (1.0 - lam)},
            "y": {"Y": 1.0, "XY": self.d_scale_y_dx or 0.0},
            "z": {"Z": 1.0, "ZX": self.d_scale_z_dx or 0.0},
        }

    def gradient_tensor(self, offset_m):
        """The 3x3 tensor ``L`` taking a COMMANDED gradient vector to the one actually delivered at a
        displacement from isocentre: ``g_delivered = L(r) g_commanded``, with ``L(0) = I``.

        A gradient coil's field is only linear near isocentre. Away from it the delivered gradient is
        mis-scaled AND TILTED, which is the standard gradient-nonlinearity tensor, and it matters here
        because the b value a voxel receives is ``b_delivered = |L u|^2 b`` along a rotated direction -- so
        this is an encoding error, not a shading.

        Column ``j`` is ``grad Phi_j`` from :meth:`gradient_potentials`, so the tensor is a field by
        construction. That off-diagonals EXIST is not optional: a diagonal ``L`` that is not the identity is
        impossible, since a diagonal ``L`` makes each coil's ``B_z`` depend on its own axis alone and Laplace
        then forces that dependence to be linear. Their particular VALUES are a modelling choice --
        :meth:`gradient_potentials` says which one and why -- because the measurement constrains only the
        diagonal.

        The principal frame of ``L`` is not the scanner frame, which is the part a diagonal tensor is
        structurally incapable of expressing. A degree count for that rotation is easy to over-read: ``L`` is
        the identity plus a perturbation whose eigenvalue spread is under a tenth, so its eigenvectors are
        the ill-conditioned part and a large angle is arithmetic rather than physics. The bounded quantity is
        the delivered ``|L u|^2``, which moves by up to about seven per cent over the validity radius.

        One limit is the measurement's and not the representation's: only the TRACELESS part is OBSERVABLE,
        because normalising each axis by the trace is what removes the unknown true diffusivity. A common
        mode is perfectly expressible here -- three coils each carrying the same added curvature give
        ``tr L = 3(1 + e x)`` -- so what is missing is the data to set it, not the freedom to state it.

        ``None`` when the machine has no catalogued coefficients -- which is every machine here but one, and
        not because the others are linear. Vendors do publish spherical-harmonic coil descriptions to
        service channels; none of them is in the open literature.
        """
        phi = self.gradient_potentials()
        if phi is None:
            return None
        self._refuse_outside(offset_m, "the gradient-nonlinearity tensor")
        d = np.atleast_2d(np.asarray(offset_m, dtype=np.float64))
        L = np.stack([solid_harmonics.gradient(phi[j], d) for j in ("x", "y", "z")], axis=-1)
        return L.reshape(np.shape(offset_m)[:-1] + (3, 3)) if np.ndim(offset_m) > 1 else L[0]

    def b0_drift(self, delta_T_K):
        """The field a magnet this much warmer holds, minus the one it was tuned at, in **tesla**.

        A permanent magnet's field follows its temperature. The coefficient is large by MRI standards: on
        the Swoop it is a MEASURED -1400 Hz/K, so about two kelvin move the centre frequency as far as the
        whole spatial inhomogeneity of the same magnet. ``None`` when the machine has no catalogued
        coefficient, which is every superconducting one -- a magnet in liquid helium has no room temperature
        to drift with.

        The catalogued figure is the machine's, not its material's, and the distinction is worth a factor of
        two: bulk NdFeB falls about 0.1 % per kelvin, which would be 2725 Hz/K here, but a yoked magnet is
        not all permanent-magnet material and the measurement comes out at half that. Reaching for the
        materials constant would overstate every number downstream.

        Uniform in space, which is what separates it from :meth:`b0_offset`. The shape of the field is a
        function of position and this is not, so a whole image needs ONE offset and a replay pays for it
        once. That also means a scanner can cancel it by re-tuning, and the Swoop does: it re-centres f0
        after every two DWIs, so what reaches the data is not the drift but the drift accrued within
        ``f0_recentering_interval``. Ask for the residual, not the total, unless you mean the total.
        """
        if self.b0_temperature_coefficient is None or self.field_T is None:
            return None
        return self.field_T * self.b0_temperature_coefficient * np.asarray(delta_T_K, dtype=np.float64)

    def b0_drift_hz(self, delta_T_K):
        """The same drift as a frequency shift of the proton resonance, in hertz -- the unit a scanner's own
        calibration reports it in, and the one every published measurement of it is quoted in."""
        d = self.b0_drift(delta_T_K)
        return None if d is None else GAMMA_BAR * d

    def b1_scale(self, offset_m):
        """The transmit scale a pulse actually gets at a displacement from isocentre: 1 is nominal, and what
        multiplies every flip angle (``kappa_B1``). ``None`` when the machine's profile is not catalogued.

        Two separate things, and they multiply. ``b1_axial_falloff`` is SPATIAL -- a coil's field weakens
        toward its ends -- and ``b1_calibration_offset`` is SYSTEMATIC, applying at isocentre too, being the
        machine's own transmit calibration sitting off nominal.

        The spatial part is ``1 - a s^2 + a rho^2 / 2``, with ``s`` the distance along the coil and ``rho``
        the distance from its axis. The transverse term is NOT a second measurement and carries no second
        parameter: at 2.7 MHz the coil bore is quasi-static, so ``div B = 0`` and ``curl B = 0`` there, and
        the paraxial expansion of any such field is ``B(s) - (rho^2/4) B''(s)``. Asserting the axial
        behaviour therefore DETERMINES the transverse behaviour -- the field must rise off-axis at exactly
        half the rate it falls along the axis. Saying it is "flat across the bore" is not a simplification of
        a field but a statement about a field that cannot exist; the resulting scale is harmonic, as the
        axial component of a quasi-static field must be, and the flat version was not.

        What Laplace fixes is the SUM of the two transverse curvatures, not their split: it forces
        ``d2/dx2 + d2/dy2 = -d2/ds2 = 2a``, and an azimuthally symmetric coil divides that equally, which is
        the ``rho^2/2`` used here. The Swoop's transmit coil is oblong (205 x 240 mm), so its true split is
        not equal and is not published -- one transverse axis carries more than half and the other less, with
        the total pinned. That degeneracy is why the measurement can report "little appreciable inhomogeneity
        in the transverse plane" without contradicting the physics: it constrains one direction, and the
        constraint here is on the pair. An equal split is the symmetric choice, not a measured one.

        What this returns is the scale on the AXIAL component of the coil's field. What actually excites is
        the component of B1 perpendicular to B0; for a solenoid, whose axis is perpendicular to B0 by
        construction, the axial component is entirely transverse to the field and so is the whole story to
        this order. The radial component enters at the next one.

        That this is a property of the MACHINE at all is a low-field statement. At 2.7 MHz the RF wavelength
        in tissue is metres, so the profile is the coil's geometry rather than the subject's; the same claim
        must not be carried to 3 T, and emphatically not to 7 T.
        """
        if self.b1_axial_falloff is None and self.b1_calibration_offset is None:
            return None
        if self.b1_axial_falloff is not None and self.b1_axis is None:
            raise ValueError(
                f"{self.name!r} catalogues a transmit fall-off but declares no b1_axis for it to fall off "
                f"along. A quadrature birdcage has no single B1 axis -- it drives two orthogonal transverse "
                f"components -- so a fall-off law is meaningless for one and the catalogue should say which "
                f"kind of coil this is")
        d = np.asarray(offset_m, dtype=np.float64)
        scale = np.ones(d.shape[:-1]) if d.ndim > 1 else 1.0
        if self.b1_axial_falloff is not None:
            # the distance ALONG THE COIL, which is what the fall-off is a function of. Projecting rather
            # than indexing is the whole point: on a cylindrical magnet the coil axis is the patient's
            # head-foot direction and this is `d[..., 2]`, but on a bi-planar magnet with a vertical field it
            # is not, and an index cannot tell the difference while a projection can.
            axis = np.asarray(self.b1_axis, dtype=np.float64)
            axis = axis / np.linalg.norm(axis)
            along = d @ axis                                   # s, the distance along the coil
            across2 = np.sum(d * d, axis=-1) - along ** 2      # rho^2, the distance from its axis
            a = self.b1_axial_falloff
            scale = scale * (1.0 - a * along ** 2 + 0.5 * a * across2)
        if self.b1_calibration_offset is not None:
            scale = scale * self.b1_calibration_offset
        return scale

    @property
    def gradient_limits(self):
        """``(G_max, slew_max)`` in T/m, T/m/s."""
        return self.G_max, self.slew_max

    @property
    def safe_model(self):
        """The SAFE PNS coefficients per axis (``tau1_ms .. a3, stim_limit, g_scale``) of the REPRESENTATIVE example
        gradient system, or ``None``: per-coil calibrations are vendor-confidential, so this is the data every
        scanner shares and the solver in dmipy-design reads; it is not this scanner's own."""
        lf = scc.SCANNER_CONSTANTS["safety"]["safe_model"].get("example_coefficients")
        return None if lf is None else [dict(a) for a in lf["value"]]

    def pulseq_dict(self):
        """This scanner in Pulseq's ``Opts`` schema (``max_grad`` in mT/m, ``max_slew`` in T/m/s, rasters and
        dead times in s where the catalogue knows them): what :data:`~dmipy_sim.sequences.pulseq.PULSEQ_SYSTEMS`
        holds per preset."""
        out = dict(max_grad=self.G_max * 1e3, max_slew=self.slew_max, grad_unit="mT/m", slew_unit="T/m/s")
        for k, v in (("grad_raster_time", self.grad_raster), ("rf_raster_time", self.rf_raster),
                     ("adc_raster_time", self.adc_raster), ("rf_dead_time", self.rf_dead_time),
                     ("rf_ringdown_time", self.rf_ringdown_time), ("adc_dead_time", self.adc_dead_time)):
            if v is not None:
                out[k] = v
        return out


#: The certificate's class table, ``(peak gradient T/m, peak slew T/m/s)`` per class -- a view of the catalogue.
SCANNERS = {c: ScannerLimits.of(c).gradient_limits for c in scc.SCANNER_CONSTANTS["classes"]}

#: The save interval at which the field tier's save-grid error is < 0.2 of a 200k-walker floor in the harshest
#: catalogued case (measured, #143).
FIELD_DT_CAP = 5e-5


def scanner_limits(scanner):
    """``(G_max, slew_max)`` of any scanner name the catalogue resolves, or of an explicit ``(G_max, slew_max)``
    pair."""
    return ScannerLimits.of(scanner).gradient_limits


def save_interval(T_max, n_walkers, scanner="connectom", *, D=2e-9, floor_fraction=0.1, field=False):
    """The save interval ``dt`` a persistent walk of ``T_max`` with ``n_walkers`` needs so that the in-step path
    integration error of any waveform ``scanner`` can deliver stays below ``floor_fraction`` of the Monte-Carlo
    floor; capped at :data:`FIELD_DT_CAP` when the walk carries a field source. ``D`` is the fastest pool's
    diffusivity (free water is the worst case)."""
    G_max, _ = scanner_limits(scanner)
    T_max, n_walkers, D = float(T_max), float(n_walkers), float(D)
    if T_max <= 0 or n_walkers <= 0 or D <= 0 or G_max <= 0 or floor_fraction <= 0:
        raise ValueError("T_max, n_walkers, D, G_max and floor_fraction must be positive")
    dt = np.sqrt(3.0 * floor_fraction / (np.sqrt(n_walkers) * GAMMA ** 2 * D * G_max ** 2 * T_max))
    if field:
        dt = min(dt, FIELD_DT_CAP)
    # a whole number of saves over T_max, never coarser than the rule
    n_t = int(np.ceil(T_max / dt)) + 1
    return T_max / (n_t - 1)
