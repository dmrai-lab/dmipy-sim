"""GreyMatterSubstrate: the physical ground-truth cortical grey-matter specification.

The grey-matter analogue of :class:`Substrate` (white matter).  It describes the
*physical tissue* the Monte-Carlo engine walks in — cell bodies (somata) packed in
extracellular space, their membrane permeability / surface relaxivity and bulk
transverse relaxation — and, uniquely for grey matter, the **static susceptibility
field of non-heme (ferritin) iron**, which in cortex is stored inside a glial
subpopulation (oligodendrocytes, microglia) rather than free in the interstitium.

Contract (identical to :class:`Substrate`): there is ONE source for every physical
constant, and it is this dataclass together with the cited ``biophysical_constants``
catalogue.  Every field is the experimental choice of the field, not ours, and every
canonical value is joined to a reference via :func:`canonical_grey_matter`.

Iron placement (the model)
--------------------------
Cortical non-heme iron is concentrated in glial cells, so the discrete-perturber
field (Schenck / Weisskoff / Yablonskiy static dephasing) is anchored to the glial
somata: each iron-bearing cell IS a magnetised sphere.  The tissue-average
susceptibility Δχ_tissue = c_iron · k_calib (Hallgren concentration × Langkammer
calibration) is reproduced by giving each glial perturber the LOCAL susceptibility

    Δχ_glia = Δχ_tissue / f_glia_volume

so the volume-weighted average over the voxel equals the measured tissue value while
the field the diffusing extracellular water samples has the correct cell-scale
clustering (that clustering, not the molecular ferritin, sets the diffusion-relevant
gradients).  Neuronal somata and ECS are treated as iron-free.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from ..constants import GAMMA
from ..geometries import PackedSpheres, pack_spheres
from ..susceptibility import SusceptibilitySources

# Yablonskiy-Haacke static-dephasing constant for randomly distributed spheres:
# R2' = (2π/3√3)·γ·B0·Δχ_eff, where Δχ_eff is the volume-average perturber
# susceptibility (Yablonskiy & Haacke 1994, 10.1002/mrm.1910320610).
_YABLONSKIY_STATIC_K = 2.0 * math.pi / (3.0 * math.sqrt(3.0))   # ≈ 1.209


@dataclass
class GreyMatterSubstrate:
    """Physical ground truth for a canonical cortical grey-matter substrate.

    All fields are REQUIRED (no hardcoded scientific defaults): construct via
    :meth:`canonical`, which pulls every value from the cited catalogue, or pass
    explicit values.  Lengths in metres, diffusivity in m²/s, times in seconds,
    iron concentration in µg(Fe)/g(wet tissue).
    """

    # -- volume fractions --
    f_cell: float                 # total cell-body (soma) packing fraction
    f_ecs: float                  # extracellular-space fraction

    # -- soma geometry (Gamma diameter law) --
    soma_diameter_mean: float     # mean soma diameter (m)
    soma_diameter_cv: float       # coefficient of variation of soma diameter

    # -- glia (the iron-bearing subpopulation) --
    glia_number_fraction: float   # fraction of somata that are glia (0..1)

    # -- diffusion / membrane --
    D_intra: float                # intrinsic intracellular diffusivity (m²/s)
    kappa: float                  # membrane permeability (m/s); 0 -> impermeable
    rho2: float                   # transverse surface relaxivity (m/s); 0 -> none

    # -- transverse / longitudinal relaxation (s) --
    T2: float
    T1: float

    # -- susceptibility (iron) --
    iron_concentration: float     # non-heme iron, µg(Fe)/g wet tissue (cortex)
    chi_per_iron: float           # Δχ (SI, dimensionless) per (µg/g)  [QSM calibration]

    # Fraction of the tissue iron that is aggregated at cell-body scale enough to
    # produce discrete-perturber (diffusion-relevant) dephasing; the rest is
    # mesoscopically smooth and refocused.  Calibrated so the analytical static
    # R2' matches the measured cortical R2' (item 17: most cortical ferritin is
    # diffuse).  None -> 1.0 = the mass-conservation upper bound (all iron clustered).
    iron_clustered_fraction: float = None

    # -- scale --
    field_T: float = 3.0

    # ── validation ────────────────────────────────────────────────────────────
    def __post_init__(self):
        from .validators import positive, non_negative, in_closed_interval
        for n in ('soma_diameter_mean', 'D_intra', 'T2', 'T1', 'field_T'):
            positive(n, getattr(self, n))
        for n in ('soma_diameter_cv', 'kappa', 'rho2', 'iron_concentration'):
            non_negative(n, getattr(self, n))
        if self.iron_clustered_fraction is not None:
            in_closed_interval('iron_clustered_fraction', self.iron_clustered_fraction, 0.0, 1.0)
        in_closed_interval('f_cell', self.f_cell, 0.0, 1.0)
        in_closed_interval('f_ecs', self.f_ecs, 0.0, 1.0)
        in_closed_interval('glia_number_fraction', self.glia_number_fraction, 0.0, 1.0)
        if self.f_cell + self.f_ecs > 1.0 + 1e-9:
            raise ValueError(
                "f_cell + f_ecs must be <= 1 (the remainder is neurite volume); got "
                f"f_cell={self.f_cell}, f_ecs={self.f_ecs}.")

    # ── constructor ───────────────────────────────────────────────────────────
    @classmethod
    def canonical(cls, field_T: float = 3.0, calibrate_r2prime: bool = True,
                  **overrides) -> "GreyMatterSubstrate":
        """Build from the cited grey-matter constant set at the given field.

        When ``calibrate_r2prime`` (default), the clustered-iron fraction is set so
        the analytical static-dephasing R2' equals the cited cortical R2'
        (``gm_R2prime_cortex``) at cortical iron — item 17: most cortical ferritin is
        mesoscopically smooth, so only this fraction produces discrete-perturber
        dephasing.  The fraction is fixed from the cortex reference and applies to
        any region's iron (R2' then scales with iron).  Set False for the
        all-iron-clustered upper bound.
        """
        from .biophysical_constants import canonical_grey_matter, get_value
        c = canonical_grey_matter(field_T=field_T)
        clustered = None
        if calibrate_r2prime:
            r2p_cortex = get_value('gm_R2prime_cortex', field_T, allow_nearest=True)
            chi_cortex = c['iron_concentration'] * c['chi_per_iron']   # cortex Δχ_tissue
            r2p_uncal = _YABLONSKIY_STATIC_K * GAMMA * field_T * chi_cortex
            clustered = float(r2p_cortex / r2p_uncal)
        kw = dict(
            f_cell=c['f_cell'], f_ecs=c['f_ecs'],
            soma_diameter_mean=c['soma_diameter_mean'],
            soma_diameter_cv=c['soma_diameter_cv'],
            glia_number_fraction=c['glia_number_fraction'],
            D_intra=c['D_intra'], kappa=c['kappa'], rho2=c['rho2'],
            T2=c['T2'], T1=c['T1'],
            iron_concentration=c['iron_concentration'],
            chi_per_iron=c['chi_per_iron'],
            iron_clustered_fraction=clustered,
            field_T=field_T,
        )
        kw.update(overrides)
        return cls(**kw)

    # ── derived susceptibility ────────────────────────────────────────────────
    @property
    def delta_chi_tissue(self) -> float:
        """Tissue-average susceptibility difference Δχ_tissue = c_iron · k_calib (SI)."""
        return self.iron_concentration * self.chi_per_iron

    @property
    def delta_chi_effective(self) -> float:
        """Diffusion-relevant susceptibility = clustered fraction × Δχ_tissue (SI).

        The clustered iron produces discrete-perturber dephasing; the smooth
        remainder is refocused.  Equals Δχ_tissue when no clustered fraction is set
        (the mass-conservation upper bound)."""
        f = 1.0 if self.iron_clustered_fraction is None else self.iron_clustered_fraction
        return f * self.delta_chi_tissue

    @property
    def static_r2prime(self) -> float:
        """Analytical static-dephasing R2' = (2π/3√3)·γ·B0·Δχ_eff [1/s].

        The MC walk additionally applies motional narrowing, so the simulated
        attenuation is at or below this static value."""
        return _YABLONSKIY_STATIC_K * GAMMA * self.field_T * self.delta_chi_effective

    # ── derived soma geometry ─────────────────────────────────────────────────
    @property
    def _gamma_shape(self) -> float:
        """Gamma shape α from the diameter CV (CV² = 1/α)."""
        return 1.0 / (self.soma_diameter_cv ** 2) if self.soma_diameter_cv > 0 else 1e6

    def sample_soma_radii(self, n_cells: int, seed: int = 0) -> np.ndarray:
        """Draw n_cells soma radii (m) from the Gamma diameter law (mean, CV)."""
        rng = np.random.default_rng(seed)
        a = self._gamma_shape
        scale = self.soma_diameter_mean / a
        return 0.5 * rng.gamma(a, scale, size=n_cells)

    # ── build the Monte-Carlo geometry ────────────────────────────────────────
    def build(self, n_cells: int = 60, L: float = None, seed: int = 0):
        """Return a :class:`PackedSpheres` of somata carrying the iron off-resonance field.

        Somata are packed (RSA) to ``f_cell``; a random ``glia_number_fraction`` of
        them are the iron-bearing perturbers, each a magnetised sphere of the cell's
        own radius with local susceptibility Δχ_glia = Δχ_tissue / f_glia_volume so
        the volume-averaged susceptibility equals ``delta_chi_tissue``.  Membrane
        permeability / surface relaxivity are baked onto the pack.

        Parameters
        ----------
        n_cells : int
            Number of somata to place.
        L : float, optional
            Cubic-domain side (m).  Default: derived from ``f_cell`` and the sampled
            radii (target volume fraction).
        seed : int
            RNG seed for the diameter sampling, packing and glia selection.

        Returns
        -------
        geom : PackedSpheres
            Soma pack with ``off_resonance`` set to the glial-iron field.
        info : dict
            Diagnostics: ``L``, ``achieved_vf``, ``n_glia``, ``f_glia_volume``,
            ``delta_chi_glia``, ``delta_chi_tissue``.
        """
        radii = self.sample_soma_radii(n_cells, seed=seed)
        if L is None:
            centers, L, vf = pack_spheres(radii, target_vf=self.f_cell, seed=seed)
        else:
            centers, L, vf = pack_spheres(radii, L=L, seed=seed)

        rng = np.random.default_rng(seed + 1)
        n_glia = int(round(self.glia_number_fraction * n_cells))
        glia_idx = rng.choice(n_cells, size=n_glia, replace=False) if n_glia else np.array([], int)

        off = None
        f_glia_vol = 0.0
        delta_chi_glia = 0.0
        if n_glia and self.delta_chi_effective != 0.0:
            g_radii = radii[glia_idx]
            f_glia_vol = float((4.0 / 3.0) * np.pi * np.sum(g_radii ** 3) / L ** 3)
            # local glial susceptibility so the voxel volume-average of the CLUSTERED
            # (diffusion-relevant) iron = Δχ_effective (see delta_chi_effective).
            delta_chi_glia = self.delta_chi_effective / f_glia_vol
            off = SusceptibilitySources(centers=centers[glia_idx], radii=g_radii,
                                        delta_chi=delta_chi_glia, B0=self.field_T)

        geom = PackedSpheres(
            radii=radii, centers=centers, L=L,
            surface_relaxivity_t2=(self.rho2 or None),
            permeability=(self.kappa or None),
            off_resonance=off,
        )
        info = dict(L=L, achieved_vf=vf, n_glia=n_glia, f_glia_volume=f_glia_vol,
                    delta_chi_glia=delta_chi_glia, delta_chi_tissue=self.delta_chi_tissue,
                    delta_chi_effective=self.delta_chi_effective,
                    iron_clustered_fraction=self.iron_clustered_fraction,
                    static_r2prime=self.static_r2prime)
        return geom, info

    def as_dict(self) -> dict:
        return asdict(self)
