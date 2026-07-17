"""Isotropic magnetic-susceptibility perturbers (grey-matter iron / vasculature).

Grey-matter susceptibility sources — non-heme iron in glia, deoxyhaemoglobin in the
microvasculature — are, to first order, ISOTROPIC magnetic inclusions: their magnetisation
lies along B0 regardless of shape, so each is a point/sphere dipole and the total off-resonance
field is a simple superposition. This is the classic discrete-perturber picture
(Weisskoff et al. 1994; Boxerman et al. 1995) with the uniformly-magnetised-sphere field
(Schenck 1996). It is deliberately the ISOTROPIC case only; the anisotropic myelin
hollow-cylinder model (Wharton & Bowtell 2012) is a separate, harder object and is not here.

A uniformly magnetised sphere of radius a and susceptibility difference Δχ in B0 produces,
OUTSIDE it (r > a),

    ΔBz(r) = (Δχ · B0 / 3) · a³ · (3 cos²θ − 1) / r³ ,   θ = angle(r − c, B0=z)

Inside the sphere the field is uniform; we clamp |r − c| ≥ a (avoids the r→0 dipole
singularity).  The established convention (Weisskoff 1994; Boxerman 1995; Murray/Zhong
2023) is that the susceptibility source is an IMPENETRABLE tissue structure — the
diffusing water samples only the exterior dipole and never enters, so the interior
value is not sampled.  The grey-matter substrate therefore makes the iron-bearing
somata impenetrable (an impermeable pack); the clamp is only a bounded approximation
in the rarer case where walkers are allowed to permeate the source.  Placing the
source ON a real tissue compartment (here the iron-bearing glial somata) rather than
as an independent random field is the mainstream, more-principled convention
(MCMRSimulator.jl, SpinWalk, and the vascular-BOLD lineage all co-locate source and
structure).

The field enters the walk as an extra transverse-plane phase, γ ΔBz dt, accrued ONLY while the
magnetisation is transverse (gated by χ_perp) and refocused by the sequence's echo (the ±1
refocusing sign, :func:`refocus_sign`). So a spin/stimulated echo refocuses the static part and
longitudinal storage pauses it — both handled by the sign×gate schedule passed to the step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from .constants import GAMMA  # noqa: F401  (re-exported convenience)


@dataclass
class SusceptibilitySources:
    """A set of isotropic magnetised-sphere perturbers producing an off-resonance field.

    Parameters
    ----------
    centers : array (P, 3)
        Perturber centres in metres (lab frame; B0 along +z).
    radii : array (P,)
        Perturber radii in metres.
    delta_chi : float or array (P,)
        Volume susceptibility difference Δχ (SI, dimensionless) of each perturber vs the
        surrounding medium. Sign follows the convention that paramagnetic (iron) Δχ > 0.
    B0 : float
        Static field in tesla (default 3.0).
    """
    centers: np.ndarray
    radii: np.ndarray
    delta_chi: float | np.ndarray = 1e-6
    B0: float = 3.0

    def __post_init__(self):
        self.centers = np.asarray(self.centers, dtype=np.float64).reshape(-1, 3)
        self.radii = np.asarray(self.radii, dtype=np.float64).reshape(-1)
        if self.centers.shape[0] != self.radii.shape[0]:
            raise ValueError("centers and radii must have the same length")
        dchi = np.asarray(self.delta_chi, dtype=np.float64)
        self.delta_chi = np.broadcast_to(dchi, self.radii.shape).copy()
        if not np.all(self.radii > 0):
            raise ValueError("radii must be positive")

    @property
    def n_perturbers(self) -> int:
        return int(self.radii.shape[0])

    def delta_bz_fn(self):
        """Return a JAX callable ``delta_bz(r) -> ΔBz`` (Tesla) for a single position r (3,).

        Sum of same-sign (∥B0=z) sphere dipoles; interior clamped to r = a."""
        c = jnp.asarray(self.centers, dtype=jnp.float32)          # (P, 3)
        # dipole coefficient per perturber: (Δχ·B0/3)·a³
        coeff = jnp.asarray((self.delta_chi * self.B0 / 3.0) * self.radii ** 3,
                            dtype=jnp.float32)                    # (P,)
        a2 = jnp.asarray(self.radii ** 2, dtype=jnp.float32)      # (P,)

        def delta_bz(r):
            d = r[None, :] - c                                   # (P, 3)
            dist2 = jnp.maximum(jnp.sum(d * d, axis=1), a2)      # (P,) clamp interior
            cos2 = (d[:, 2] ** 2) / dist2
            return jnp.sum(coeff * (3.0 * cos2 - 1.0) / dist2 ** 1.5)

        return delta_bz


def refocus_sign(waveform):
    """Per-timestep refocusing sign s(t) ∈ {+1, −1} for a static background field.

    A static off-resonance field is not carried in the (bipolar) gradient G, so the sequence's
    echo must be applied to it explicitly: the phase accrued before the refocusing/recall pulse
    is negated relative to after it. This matches the idealized single-pathway convention (the
    phase before the echo index is negated). The flip point is the LAST RF pulse in
    ``rf_events`` — the 180° for a spin echo (PGSE/OGSE), the recall-90° for a stimulated echo
    (PGSTE). For a spin echo with no rf_events, s ≡ +1 (no refocusing — a gradient-echo-like
    static-dephasing forward).

    Returns
    -------
    s : np.ndarray (n_t,) float32, values in {+1, -1}.
    """
    n_t = int(np.asarray(waveform.G).shape[1])
    rf = getattr(waveform, 'rf_events', None)
    s = np.ones(n_t, dtype=np.float32)
    if rf:
        t_flip = float(rf[-1]['t_s'])
        idx = int(round(t_flip / float(waveform.dt)))
        idx = max(0, min(idx, n_t))
        s[idx:] = -1.0
    return s
