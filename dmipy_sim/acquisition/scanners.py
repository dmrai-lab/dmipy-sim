"""The scanner: its deliverability limits as one typed object, and the save-interval rule a persistent walk
derives its grid from.

Every number here is read from :mod:`dmipy_sim.acquisition.scanner_constants` (the cited catalogue); this module
carries none. :class:`ScannerLimits` is what a consumer takes -- a designer's constraint set, a Pulseq exporter's
``Opts``, the walk's save grid -- resolved from any of a scanner's names and a slew ``regime``. :data:`SCANNERS` is
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
from . import scanner_constants as scc

GAMMA_BAR = GAMMA / (2.0 * np.pi)

__all__ = ["ScannerLimits", "SCANNERS", "FIELD_DT_CAP", "scanner_limits", "save_interval"]


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
    b0_quadratic: float = None        # 1/m^2, the even term c of dB/B0 = a x + c r^2
    b0_asymmetry_rl: float = None     # 1/m, the odd term a, along the scanner's R/L axis
    b0_validity_radius: float = None  # m, how far from isocentre that law is anchored
    b1_axial_falloff: float = None    # 1/m^2, the coefficient of kappa_B1 = 1 - a z^2 along the bore
    b1_calibration_offset: float = None  # a systematic transmit scale, 1 = nominal
    b0_temperature_coefficient: float = None  # 1/K, dB0/B0 per kelvin of magnet temperature
    f0_recentering_interval: float = None     # s, how long the field drifts before f0 is re-set
    f0_temperature_slope: float = None        # Hz/K, the same coefficient as published

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
        return cls(name=key, kind=kind, regime=regime, G_max=G_max, slew_max=slew,
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
                   b0_quadratic=scc.leaf_si(entry, "homogeneity", "b0_quadratic"),
                   b0_asymmetry_rl=scc.leaf_si(entry, "homogeneity", "b0_asymmetry_rl"),
                   b0_validity_radius=scc.leaf_si(entry, "homogeneity", "b0_validity_radius"),
                   b1_axial_falloff=scc.leaf_si(entry, "rf", "b1_axial_falloff"),
                   b1_calibration_offset=scc.leaf_si(entry, "rf", "b1_calibration_offset"),
                   b0_temperature_coefficient=scc.leaf_si(entry, "thermal",
                                                          "b0_temperature_coefficient"),
                   f0_recentering_interval=scc.leaf_si(entry, "thermal", "f0_recentering_interval"),
                   f0_temperature_slope=scc.leaf_si(entry, "thermal", "f0_temperature_slope"))

    def b0_offset(self, offset_m):
        """The static field's departure from uniformity at a displacement from isocentre, in **tesla**:
        ``B0 * (a x + c r^2)`` for the catalogued shape. ``offset_m`` is ``(..., 3)`` in metres in the
        bore's frame (x is R/L, as the grid's ``axes`` name them), and the result has its leading shape.

        The two terms are different physics. ``c`` is the isotropic bowl every magnet has; ``a`` is the
        R/L asymmetry a SINGLE-YOKE magnet has because its yoke sits on one side, so the field is not
        mirror-symmetric about isocentre. Leaving the odd term out would be a visible error rather than a
        small one -- for the Swoop the field differs by 716 ppm between +8 and -8 cm.

        ``None`` when this machine's profile is not catalogued -- which is every machine but one, because a
        shimmed superconducting magnet's residual is parts per million and nobody publishes its shape. A
        permanent magnet's is parts per thousand and does get published, which is the case this exists for.

        Beyond ``b0_validity_radius`` the law is an extrapolation and is refused: the coefficient is
        anchored at one radius, and a magnet's profile steepens past the volume it was specified over.
        """
        if self.b0_quadratic is None or self.field_T is None:
            return None
        d = np.asarray(offset_m, dtype=np.float64)
        r = np.linalg.norm(d, axis=-1)
        if self.b0_validity_radius is not None and float(np.max(r)) > self.b0_validity_radius:
            raise ValueError(
                f"the field law for {self.name!r} is anchored at {self.b0_validity_radius*100:.0f} cm from "
                f"isocentre and something here is {float(np.max(r))*100:.1f} cm out. A magnet's profile "
                f"steepens beyond the volume it was specified over, so this is refused rather than "
                f"extrapolated")
        shape = self.b0_quadratic * r ** 2
        if self.b0_asymmetry_rl:
            shape = shape + self.b0_asymmetry_rl * d[..., 0]        # x is R/L
        return self.field_T * shape

    def b0_gradient(self, offset_m):
        """The SPATIAL GRADIENT of the static field at a displacement from isocentre, in **T/m** -- the
        magnet's own encoding gradient, which is on during every pulse and every dead time because a magnet
        does not switch off.

        It is the derivative of :meth:`b0_offset`, so it is the same law and not a second one:
        ``grad B0 (a x + c r^2) = B0 (a xhat + 2 c r)``. Feed it to
        :meth:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence.with_background_gradient` and the
        effective gradient, the b value and the cross term with the pulsed gradient all follow exactly.

        It does NOT vanish at isocentre, and that is the odd term rather than an error: a single-yoke magnet
        is not mirror-symmetric, so ``a`` survives differentiation where the bowl's ``2 c r`` does not. On
        the Swoop that residue is 0.29 mT/m at the origin, against 1.40 mT/m at 8 cm on the high side and
        0.83 mT/m on the low one. A magnet whose only term were the bowl would encode nothing at its centre;
        this one encodes something everywhere.

        ``None`` when the machine publishes no profile. Refused beyond ``b0_validity_radius``, for the
        reason :meth:`b0_offset` gives -- and more sharply here, since a derivative extrapolates worse than
        the quantity it came from.
        """
        if self.b0_quadratic is None or self.field_T is None:
            return None
        d = np.atleast_2d(np.asarray(offset_m, dtype=np.float64))
        r = np.linalg.norm(d, axis=-1)
        if self.b0_validity_radius is not None and float(np.max(r)) > self.b0_validity_radius:
            raise ValueError(
                f"the field law for {self.name!r} is anchored at {self.b0_validity_radius*100:.0f} cm from "
                f"isocentre and something here is {float(np.max(r))*100:.1f} cm out. A derivative "
                f"extrapolates worse than the law it came from, so this is refused")
        g = 2.0 * self.b0_quadratic * d
        if self.b0_asymmetry_rl:
            g[..., 0] += self.b0_asymmetry_rl                        # x is R/L
        out = self.field_T * g
        return out.reshape(np.shape(offset_m)) if np.ndim(offset_m) > 1 else out[0]

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
        toward its ends, so the scale falls as ``1 - a z^2`` along the bore and is flat across it, which is
        the anisotropy the measurement reports rather than a simplification. ``b1_calibration_offset`` is
        SYSTEMATIC: it applies at isocentre too, being the machine's own transmit calibration sitting off
        nominal.

        That this is a property of the MACHINE at all is a low-field statement. At 2.7 MHz the RF wavelength
        in tissue is metres, so the profile is the coil's geometry rather than the subject's; the same claim
        must not be carried to 3 T, and emphatically not to 7 T.
        """
        if self.b1_axial_falloff is None and self.b1_calibration_offset is None:
            return None
        d = np.asarray(offset_m, dtype=np.float64)
        scale = np.ones(d.shape[:-1]) if d.ndim > 1 else 1.0
        if self.b1_axial_falloff is not None:
            scale = scale * (1.0 - self.b1_axial_falloff * d[..., 2] ** 2)     # z is the bore
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
