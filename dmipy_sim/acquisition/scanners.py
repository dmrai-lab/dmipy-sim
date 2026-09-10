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
                            else scc.leaf_si(entry, "rf", "peak_B1_head_coil")))

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
