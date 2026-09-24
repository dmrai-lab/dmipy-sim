"""Magnetic-susceptibility off-resonance fields for the forward Bloch engine.

A susceptibility source magnetises in the main field B0 and perturbs the local Larmor
frequency by an off-resonance field ``ΔBz(r)``.  In the forward vector-Bloch walk
(:mod:`dmipy_sim.engine.bloch`) this enters as an extra z-precession ``γ·ΔBz(r(t))·dt`` at every
step, accrued on the same spin as the gradient phase; the sequence's own 180° pulse
refocuses the static part of the field, exactly as in a real spin echo.

Three field providers, each exposing a pure-JAX ``delta_bz_fn() -> (r -> ΔBz)`` callable
that plugs straight into the Bloch step:

* :class:`SusceptibilitySources` — isotropic magnetised-sphere perturbers (grey-matter
  iron / vasculature): superposed uniformly-magnetised-sphere dipoles (Schenck 1996).
* :class:`MyelinSusceptibility` — anisotropic hollow-cylinder myelin field
  (Wharton & Bowtell 2012), closed form for packed parallel axons.
* :class:`GridSusceptibility` — an arbitrary 3-D susceptibility distribution on a regular
  grid, solved once by the Lorentz-corrected k-space dipole model (:func:`dipole_field`)
  and sampled along the walk; this is the mesh route (voxelise a substrate, build the χ
  tensor, solve, sample).

Units: lengths m, ``voxel_size`` m, ``B0`` tesla, susceptibilities dimensionless (SI
volume susceptibility, e.g. ``chi_aniso = -0.1e-6``).  ``ΔBz`` is returned in tesla.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from .susceptibility_field import _as_voxel_size, dipole_field, myelin_susceptibility_tensor
from ..replay.so3 import rotation_of


# Symmetric 3x3 tensor stored as 6 components in this fixed order.
_SYM6 = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


# =============================================================================== #
# Isotropic sphere perturbers (grey matter: iron / vasculature)
# =============================================================================== #
@dataclass
class SusceptibilitySources:
    """Isotropic magnetised-sphere perturbers producing an off-resonance field.

    A uniformly magnetised sphere of radius ``a`` and susceptibility difference ``Δχ`` in
    ``B0`` produces, outside it (r > a),

        ΔBz(r) = (Δχ·B0/3)·a³·(3cos²θ − 1)/r³ ,   θ = angle(r − c, B0=z)

    the interior is clamped to ``r = a`` (the source is treated as an impenetrable tissue
    structure; Weisskoff 1994, Boxerman 1995).

    Parameters
    ----------
    centers : (P, 3) array   perturber centres (m, lab frame, B0 along +z).
    radii : (P,) array       perturber radii (m).
    delta_chi : float or (P,) array   Δχ (SI); paramagnetic (iron) Δχ > 0.
    B0 : float               static field (T).
    """
    centers: np.ndarray
    radii: np.ndarray
    delta_chi: "float | np.ndarray" = 1e-6
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
        """JAX callable ``delta_bz(r) -> ΔBz`` (T); sum of ∥B0 sphere dipoles."""
        c = jnp.asarray(self.centers, dtype=jnp.float32)             # (P, 3)
        coeff = jnp.asarray((self.delta_chi * self.B0 / 3.0) * self.radii ** 3,
                            dtype=jnp.float32)                       # (P,)
        a2 = jnp.asarray(self.radii ** 2, dtype=jnp.float32)         # (P,)

        def delta_bz(r):
            d = r[None, :] - c                                       # (P, 3)
            dist2 = jnp.maximum(jnp.sum(d * d, axis=1), a2)          # clamp interior
            cos2 = (d[:, 2] ** 2) / dist2
            return jnp.sum(coeff * (3.0 * cos2 - 1.0) / dist2 ** 1.5)

        return delta_bz


# =============================================================================== #
# Anisotropic hollow-cylinder myelin field (white matter)
# =============================================================================== #
@dataclass
class MyelinSusceptibility:
    """Anisotropic hollow-cylinder (myelin) off-resonance field: the closed form of
    :mod:`dmipy_sim.fields.hollow_cylinder` summed over axons ``k`` (inner ``a_k``, outer ``b_k``, axis along
    z after ``R``) and their periodic images,

    ``dBz(r) = delta_chi_a * B0 * sum_k  H . M_A(r - c_k) . H``,

    with ``H`` the B0 direction given by ``theta`` (fibre-to-B0 angle) and ``alpha`` (its azimuth in the
    cross-section): the lumen field ``(1/2) delta_chi_a B0 sin^2 theta ln(1/g)``, uniform and zero with B0 along
    the fibre, the sheath's and the outside's per the module docstring. Closed form (no grid); one value
    ``delta_chi_a`` (the anisotropic susceptibility, the isotropic one being a replay knob on a pack).
    """
    centers: np.ndarray
    inner_radii: np.ndarray
    outer_radii: np.ndarray
    L: float
    delta_chi_a: float = -0.1e-6
    B0: float = 3.0
    theta: float = 0.0
    alpha: float = 0.0
    R: np.ndarray = None
    n_images: int = 2
    periodic: bool = True

    def __post_init__(self):
        self.centers = np.asarray(self.centers, dtype=np.float64).reshape(-1, 2)
        self.inner_radii = np.asarray(self.inner_radii, dtype=np.float64).reshape(-1)
        self.outer_radii = np.asarray(self.outer_radii, dtype=np.float64).reshape(-1)
        if not (self.centers.shape[0] == self.inner_radii.shape[0] == self.outer_radii.shape[0]):
            raise ValueError("centers, inner_radii, outer_radii must have the same length")
        if not np.all(self.outer_radii > self.inner_radii):
            raise ValueError("outer_radii must exceed inner_radii")

    @property
    def n_perturbers(self) -> int:
        return int(self.inner_radii.shape[0])

    @classmethod
    def from_geometry(cls, geom, delta_chi_a=-0.1e-6, B0=3.0, b0_dir=(0., 0., 1.),
                      n_images=2):
        """Build from a packed-myelinated-cylinder geometry + a lab B0 direction."""
        N = int(getattr(geom, 'N_actual', len(geom._inner_radii_np)))
        centers = np.asarray(geom._centers_np)[:N]
        inner = np.asarray(geom._inner_radii_np)[:N]
        outer = np.asarray(geom._outer_radii_np)[:N]
        L = float(geom._L_float)
        axis = np.asarray(getattr(geom, 'orientation', (0., 0., 1.)), float)
        axis = axis / np.linalg.norm(axis)
        b0 = np.asarray(b0_dir, float); b0 = b0 / np.linalg.norm(b0)
        theta = float(np.arccos(np.clip(abs(np.dot(axis, b0)), 0.0, 1.0)))
        b0_perp = rotation_of(axis).T @ b0                      # the axis to +z
        alpha = float(np.arctan2(b0_perp[1], b0_perp[0]))
        return cls(centers=centers, inner_radii=inner, outer_radii=outer, L=L,
                   delta_chi_a=delta_chi_a, B0=B0, theta=theta, alpha=alpha, R=None,
                   n_images=n_images)

    def delta_bz_fn(self):
        """JAX callable ``delta_bz(r) -> dBz`` (T) for a position r (3,)."""
        from .hollow_cylinder import hollow_cylinder_basis, q_of_H
        c = jnp.asarray(self.centers, jnp.float32)                   # (N, 2)
        a = jnp.asarray(self.inner_radii, jnp.float32)
        b = jnp.asarray(self.outer_radii, jnp.float32)
        ims = jnp.arange(-self.n_images, self.n_images + 1) * jnp.float32(self.L)
        ox, oy = jnp.meshgrid(ims, ims)
        ox = ox.ravel(); oy = oy.ravel()                             # (M,)
        H = np.array([np.sin(self.theta) * np.cos(self.alpha), np.sin(self.theta) * np.sin(self.alpha), np.cos(self.theta)])
        q = jnp.asarray(q_of_H(H), jnp.float32)                      # (6,)
        scale = jnp.float32(self.delta_chi_a * self.B0)
        u = jnp.asarray([0.0, 0.0, 1.0], jnp.float32)
        R = None if self.R is None else jnp.asarray(self.R, jnp.float32)
        Lf = jnp.float32(self.L)
        wrap = bool(self.periodic)

        def delta_bz(r):
            r_perp = r if R is None else jnp.matmul(R, r, precision=jax.lax.Precision.HIGHEST)   # no TF32 on a position
            x, y = r_perp[0], r_perp[1]
            if wrap:
                x = ((x + 0.5 * Lf) % Lf) - 0.5 * Lf
                y = ((y + 0.5 * Lf) % Lf) - 0.5 * Lf
            dx = x - (c[:, 0:1] + ox[None, :])                       # (N, M)
            dy = y - (c[:, 1:2] + oy[None, :])
            rho_vec = jnp.stack([dx, dy, jnp.zeros_like(dx)], axis=-1)            # (N, M, 3)
            C = hollow_cylinder_basis(rho_vec, jnp.broadcast_to(u, rho_vec.shape),
                                      jnp.broadcast_to(a[:, None], dx.shape), jnp.broadcast_to(b[:, None], dx.shape))
            return scale * jnp.sum(C[..., 7:13] @ q)

        return delta_bz


# =============================================================================== #
# Arbitrary 3-D distribution on a grid (the mesh route)
# =============================================================================== #
@dataclass
class GridSusceptibility:
    """Off-resonance field of an arbitrary χ distribution sampled from a solved grid.

    Build the field once with :func:`dipole_field` (or pass a precomputed ``dB`` grid),
    then sample it along the walk.  This is the mesh route: voxelise a substrate into a
    ``(mask, radial_dir)`` source, form the χ tensor, solve, sample.

    Parameters
    ----------
    dB : (Nx,Ny,Nz) array      solved off-resonance field (T).
    origin : (3,)              world coordinate of voxel (0,0,0) corner (m).
    voxel_size : float or (3,) voxel edge length(s) (m).
    periodic : bool            wrap sampling (periodic cell) vs clamp at the edge.
    """
    dB: np.ndarray
    origin: np.ndarray
    voxel_size: "float | np.ndarray"
    periodic: bool = False

    @classmethod
    def from_source(cls, mask, radial_dir, voxel_size, origin, b0_dir, B0,
                    chi_iso=0.0, chi_aniso=0.0, periodic=False):
        """Solve the field from a voxelised source and wrap it as a provider."""
        chi6 = myelin_susceptibility_tensor(mask, radial_dir, chi_iso, chi_aniso)
        dB = dipole_field(chi6, voxel_size, b0_dir, B0)
        return cls(dB=dB, origin=np.asarray(origin, float), voxel_size=voxel_size,
                   periodic=bool(periodic))

    def delta_bz_fn(self):
        """JAX callable ``delta_bz(r) -> ΔBz`` (T): trilinear sample of the grid."""
        from jax.scipy.ndimage import map_coordinates

        grid = jnp.asarray(self.dB, jnp.float32)
        org = jnp.asarray(np.asarray(self.origin, float).ravel(), jnp.float32)
        vs = jnp.asarray(_as_voxel_size(self.voxel_size, 3), jnp.float32)
        shape = jnp.asarray(self.dB.shape, jnp.float32)
        wrap = bool(self.periodic)
        mode = "wrap" if wrap else "nearest"

        def delta_bz(r):
            idx = (r - org) / vs - 0.5                           # fractional voxel coords
            if wrap:
                idx = jnp.mod(idx, shape)
            return map_coordinates(grid, [idx[0], idx[1], idx[2]], order=1, mode=mode)

        return delta_bz

