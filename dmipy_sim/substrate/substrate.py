"""Substrate: the physical ground-truth white-matter specification (sim-owned).

This describes the *physical tissue* the Monte-Carlo engine walks in: geometry
(axon-diameter Gamma law, g-ratio, volume fractions), diffusivity (intra / extra /
myelin / CSF), transverse relaxation (T2), and the membrane/surface properties
(surface relaxivity, permeability).  The engine consumes it directly; it *is* the
substrate definition.  Contract: there is ONE source for every physical constant,
and it is this dataclass (with the cited ``biophysical_constants`` catalogue).

Magnetisation is transverse throughout (ideal instantaneous pulses), so there is no
longitudinal (T1) or susceptibility (Delta_chi / B0) physics here.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

from .biophysical_constants import canonical_white_matter, get_default_value


def mean_inv_diameter_4(alpha, scale_diameter, volume_weighted=True):
    """<4/d> for the (volume- or number-weighted) Gamma distribution.

    number:  4 beta/(alpha-1);   volume (d^2 P): 4 beta/(alpha+1).
    Vendored from dmipy_fit.white_matter.surface.mean_inv_diameter_4.
    """
    a = alpha + 2.0 if volume_weighted else alpha
    beta = 1.0 / scale_diameter
    return 4.0 * beta / (a - 1.0)


@dataclass
class Substrate:
    """Physical ground truth for the canonical white-matter substrate.

    Conventions match the historical UnifiedWhiteMatterParameters:
    * ``f_axon`` is the total axon (lumen + myelin sheath) packing fraction; at
      g-ratio g the intra-axonal lumen is ``g**2 * f_axon`` and the sheath is
      ``(1 - g**2) * f_axon``; the remainder ``1 - f_axon - f_csf`` is extra-axonal.
    * ``D_extra`` is the *intrinsic* (pre-tortuosity) diffusivity -- the MC number.
    """

    # -- volume fractions --
    f_axon: float = 0.55
    f_csf: float = 0.0

    # -- geometry (canonical CC Gamma diameter law; Aboitiz 1992) --
    gamma_shape_diameter: float = 2.0
    gamma_scale_diameter: float = 0.304e-6
    g_ratio: float = 0.70

    # -- diffusivity (m^2/s; D_extra is intrinsic, pre-tortuosity) --
    # Myelin water is a single isotropic pool; it is trapped between the bilayers
    # with a short T2 and barely moves, so the canonical assumption is D_myelin = 0
    # (stuck). Set > 0 to let it diffuse.
    D_intra: float = 1.7e-9
    D_extra: float = 1.7e-9
    D_myelin: float = 0.0
    D_csf: float = 3.0e-9

    # -- transverse relaxation (s; field-matched apparent values) --
    T2_intra: float = 0.050
    T2_extra: float = 0.055
    T2_myelin: float = 0.010
    T2_csf: float = 2.0

    # -- membrane / surface --
    rho2: float = 1.16e-6         # transverse surface relaxivity (m/s)
    kappa: float = 1.0e-5         # axolemma permeability (m/s)

    # -- myelin water content: fraction of the myelin sheath VOLUME that is water
    #    (~0.40; West2018).  NOT the myelin water fraction / MWF signal (~0.15).
    #    Read from the catalogue so it cannot drift.
    myelin_water_proton_density: float = get_default_value('myelin_water_proton_density')

    # -- scale --
    M0: float = 100.0
    field_T: float = 3.0

    # -- extra-axonal tortuosity override (substrate-consistent MC tortuosity) --
    lambda_perp_extra_override: float = None

    # ── validation (the parameter audit) ──────────────────────────────────────
    def __post_init__(self):
        """Audit the physical parameters at construction (fail early & clearly).

        Every field has an implicit validity domain; a physically-impossible value
        otherwise produces a silently-wrong signal or a cryptic failure deep in a
        Monte-Carlo run.  Subclasses (e.g. dmipy-fit's UnifiedWhiteMatterParameters)
        call ``super().__post_init__()`` then audit their own extra fields.
        """
        from .validators import (positive, non_negative,
                                  in_open_interval, in_closed_interval)
        # strictly positive scales
        for n in ('gamma_shape_diameter', 'gamma_scale_diameter',
                  'D_intra', 'D_extra',
                  'D_csf', 'T2_intra', 'T2_extra', 'T2_myelin', 'T2_csf',
                  'field_T'):
            positive(n, getattr(self, n))
        # non-negative
        for n in ('D_myelin', 'rho2', 'kappa', 'M0'):
            non_negative(n, getattr(self, n))
        # bounded intervals
        in_open_interval('g_ratio', self.g_ratio, 0.0, 1.0)
        in_closed_interval('f_axon', self.f_axon, 0.0, 1.0)
        in_closed_interval('f_csf', self.f_csf, 0.0, 1.0)
        in_closed_interval('myelin_water_proton_density', self.myelin_water_proton_density, 0.0, 1.0)
        if self.lambda_perp_extra_override is not None:
            positive('lambda_perp_extra_override', self.lambda_perp_extra_override)
        # codependent: the extra-axonal fraction 1 - f_axon - f_csf must be physical
        if self.f_axon + self.f_csf > 1.0 + 1e-9:
            raise ValueError(
                "f_axon + f_csf must be <= 1: the extra-axonal fraction is "
                "1 - f_axon - f_csf and cannot be negative; got "
                f"f_axon={self.f_axon}, f_csf={self.f_csf} "
                f"(sum {self.f_axon + self.f_csf}).")

    # ── constructors ─────────────────────────────────────────────────────────
    @classmethod
    def canonical(cls, field_T: float = 3.0, **overrides) -> "Substrate":
        """Build from the cited constant set at the given field strength.

        ``overrides`` replace any field after the canonical defaults are pulled.
        """
        c = canonical_white_matter(field_T=field_T)
        kw = dict(
            gamma_shape_diameter=c['gamma_shape_diameter'],
            gamma_scale_diameter=c['gamma_scale_diameter'],
            g_ratio=c['g_ratio'],
            D_intra=c['D_intra'],
            D_extra=c['D_extra'],
            D_csf=c['D_csf'],
            T2_intra=c['T2_intra'], T2_extra=c['T2_extra'],
            T2_myelin=c['T2_myelin'], T2_csf=c['T2_csf'],
            rho2=c['rho2'], kappa=c['kappa'],
            field_T=field_T,
            # Substrate-consistent extra-axonal perpendicular diffusivity for the
            # canonical pack (MC-emergent ~0.96e-9), NOT the NODDI bridge.
            lambda_perp_extra_override=0.96e-9,
        )
        kw.update(overrides)
        return cls(**kw)

    # ── derived geometry ──────────────────────────────────────────────────────
    # The Gamma law (gamma_shape_diameter, gamma_scale_diameter) is the OUTER
    # (fibre) diameter distribution -- the quantity histology reports (Aboitiz 1992,
    # "Fiber composition of the human corpus callosum"). The inner (axon/lumen)
    # diameter follows as d_inner = g * d_outer.
    @property
    def mean_outer_radius(self) -> float:
        """Mean outer (fibre/myelin) radius (m): mean_diameter / 2 = alpha*scale / 2.

        The Gamma distribution is the OUTER (fibre) diameter (histology convention).
        """
        return 0.5 * self.gamma_shape_diameter * self.gamma_scale_diameter

    @property
    def std_outer_radius(self) -> float:
        return 0.5 * math.sqrt(self.gamma_shape_diameter) * self.gamma_scale_diameter

    @property
    def mean_inner_radius(self) -> float:
        """Mean inner (lumen/axon) radius (m): outer * g."""
        return self.mean_outer_radius * self.g_ratio

    @property
    def std_inner_radius(self) -> float:
        return self.std_outer_radius * self.g_ratio

    @property
    def mean_sq_outer_radius(self) -> float:
        """Second moment <b^2> of the outer (fibre) radius (m^2).

        The Gamma is the OUTER diameter (shape alpha, scale beta_d), b = d/2:
            <b^2> = alpha(alpha+1) beta_d^2 / 4 = <b>^2 (1 + 1/alpha).
        The susceptibility rate goes as b^2 and the population dephases the EA
        water as the AVERAGE of b^2 (carries the radius variance), not <b>^2.
        """
        a = self.gamma_shape_diameter
        bd = self.gamma_scale_diameter
        return a * (a + 1.0) * bd ** 2 / 4.0

    # ── derived volume fractions ──────────────────────────────────────────────
    @property
    def f_intra(self) -> float:
        """Intra-axonal lumen fraction = g**2 * f_axon."""
        return self.g_ratio ** 2 * self.f_axon

    @property
    def f_myelin(self) -> float:
        """Myelin sheath fraction = (1 - g**2) * f_axon."""
        return (1.0 - self.g_ratio ** 2) * self.f_axon

    @property
    def f_extra(self) -> float:
        """Extra-axonal fraction = 1 - f_axon - f_csf."""
        return 1.0 - self.f_axon - self.f_csf

    @property
    def spin_fractions(self) -> dict:
        """Equilibrium SPIN populations (normalised), the n(r0) weighting.

        The myelin sheath holds less water per unit volume
        (``myelin_water_proton_density`` ~0.40); these are the weights entering
        M0 = integral n(r0), for both the analytical sum and MC walker weighting.
        """
        s_i = self.f_intra
        s_m = self.f_myelin * self.myelin_water_proton_density
        s_e = self.f_extra
        s_c = self.f_csf
        tot = s_i + s_m + s_e + s_c
        return dict(intra=s_i / tot, myelin=s_m / tot,
                    extra=s_e / tot, csf=s_c / tot)

    @property
    def v_ic(self) -> float:
        """Intra-axonal volume fraction relative to non-CSF tissue (NODDI v_ic)."""
        tissue = 1.0 - self.f_csf
        return self.f_intra / tissue if tissue > 0 else 0.0

    @property
    def intra_surface_rate(self) -> float:
        """Intra-pore surface relaxation rate rho_int * <S/V>, s^-1.

        Uses the VOLUME-weighted Gamma <4/d> = 4*beta/(alpha+1) (the spin/water
        average, since water per cylinder scales as cross-sectional area d^2).
        The interior wall is the INNER (axon) diameter d_inner = g * d_outer, so the
        inner-diameter scale is g * gamma_scale_diameter.
        """
        sv = mean_inv_diameter_4(self.gamma_shape_diameter,
                                 self.g_ratio * self.gamma_scale_diameter,
                                 volume_weighted=True)
        return self.rho2 * sv

    @property
    def T2_intra_bulk(self) -> float:
        """Intra bulk T2 from apparent: 1/T2_app = 1/T2_bulk + rho*<S/V>."""
        rate = 1.0 / self.T2_intra - self.intra_surface_rate
        if rate <= 0.0:
            raise ValueError(
                f"intra surface rate {self.intra_surface_rate:.1f}/s >= apparent "
                f"1/T2_intra {1.0/self.T2_intra:.1f}/s: rho2={self.rho2:.2e} is too "
                f"large for T2_intra={self.T2_intra:.3f}s, so the implied bulk T2 is "
                f"non-positive. Lower rho2, raise T2_intra, or pass surface_relaxivity=False.")
        return 1.0 / rate

    @property
    def extra_surface_rate(self) -> float:
        """Extra-axonal exterior surface relaxation rate rho * S_ext/V_EA, s^-1."""
        return self.rho2 * self.S_ext_over_V_EA

    @property
    def T2_extra_bulk(self) -> float:
        """Extra bulk T2 from apparent via the exterior surface rate."""
        rate = 1.0 / self.T2_extra - self.extra_surface_rate
        if rate <= 0.0:
            raise ValueError(
                f"extra surface rate {self.extra_surface_rate:.1f}/s >= apparent "
                f"1/T2_extra {1.0/self.T2_extra:.1f}/s: rho2={self.rho2:.2e} is too "
                f"large for T2_extra={self.T2_extra:.3f}s, so the implied bulk T2 is "
                f"non-positive. Lower rho2, raise T2_extra, or pass surface_relaxivity=False.")
        return 1.0 / rate

    @property
    def S_ext_over_V_EA(self) -> float:
        """Extra-axonal exterior surface-to-volume ratio (1/m), for B_hat_EA.

        The Gamma is the OUTER (fibre) diameter, so with beta_d the OUTER-diameter
        scale the exterior surface density is (the g-ratio does not appear -- the outer
        wall is the fibre boundary itself):

            S_ext/V_EA = 4 f_axon / ((1-f_axon)(alpha+1) beta_d).

        (Identical to the analytical-fit ``exterior_surface_to_volume``; the earlier
        inner-scale form 4 f g/((1-f)(alpha+1) beta_inner) is this with beta_inner=g*beta_d.)
        """
        f, a, bd = self.f_axon, self.gamma_shape_diameter, self.gamma_scale_diameter
        return 4.0 * f / ((1.0 - f) * (a + 1.0) * bd)

    @property
    def lambda_perp_extra(self) -> float:
        """Tortuosity-hindered extra-axonal perpendicular diffusivity.

        ``lambda_perp_extra_override`` (substrate-consistent MC tortuosity) is used
        when set, else the NODDI estimate ``lambda_par * (1 - v_ic)``.
        """
        if self.lambda_perp_extra_override is not None:
            return self.lambda_perp_extra_override
        return self.D_extra * (1.0 - self.v_ic)

    def as_dict(self) -> dict:
        return asdict(self)
