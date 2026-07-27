"""Magnetic-susceptibility off-resonance fields for the forward Bloch engine.

A susceptibility source magnetises in the main field B0 and perturbs the local Larmor
frequency by an off-resonance field ``ΔBz(r)``.  In the forward vector-Bloch walk
(:mod:`dmipy_sim.bloch`) this enters as an extra z-precession ``γ·ΔBz(r(t))·dt`` at every
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
import jax.numpy as jnp

from .constants import GAMMA  # noqa: F401  (re-exported convenience)

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
    """Anisotropic hollow-cylinder (myelin) off-resonance field (Wharton & Bowtell 2012).

    ``ΔBz(r) = Δχ_a·B0·[ (sin²θ/2 − 1/3)·Φ₀(r) + (sin²θ/2)·(cos2α·Φ_C + sin2α·Φ_S) ]``,
    summed over axons k (inner a, outer b, g=a/b) and periodic images, with, per axon,
    ``(dx,dy) = r⊥ − c_k``, ``r² = dx²+dy²``:

        extra  (r>b):   Φ_C += (b²−a²)(dx²−dy²)/r⁴ ,          Φ_S += (b²−a²)·2dxdy/r⁴
        sheath (a<r<b): Φ_C += (r⁴−a⁴)(dx²−dy²)/[r⁴(b²+a²)] , Φ_S += (r⁴−a⁴)·2dxdy/[…]
        Φ₀ = ln(b/r) in the sheath, ln(b/a)=ln(1/g) in the lumen, 0 outside.

    with the m=0 term carried at angular factor ``sin²θ/2``, so the intra field is
    ``½·Δχ_a·B0·sin²θ·ln(1/g)`` — uniform inside the lumen and zero when B0 ∥ fibre,
    matching the k-space dipole solver (:func:`dipole_field`).  θ is the fibre-to-B0
    angle, α the B0 azimuth in the cross-section.  Closed-form (no grid).
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
        b0_perp = _axis_to_z_rotation(axis) @ b0
        alpha = float(np.arctan2(b0_perp[1], b0_perp[0]))
        return cls(centers=centers, inner_radii=inner, outer_radii=outer, L=L,
                   delta_chi_a=delta_chi_a, B0=B0, theta=theta, alpha=alpha, R=None,
                   n_images=n_images)

    def delta_bz_fn(self):
        """JAX callable ``delta_bz(r) -> ΔBz`` (T) for a position r (3,)."""
        c = jnp.asarray(self.centers, jnp.float32)                   # (N, 2)
        a2 = jnp.asarray(self.inner_radii ** 2, jnp.float32)
        b2 = jnp.asarray(self.outer_radii ** 2, jnp.float32)
        a4 = jnp.asarray(self.inner_radii ** 4, jnp.float32)
        ln_ba = jnp.asarray(np.log(self.outer_radii / self.inner_radii), jnp.float32)  # ln(1/g)
        ims = jnp.arange(-self.n_images, self.n_images + 1) * jnp.float32(self.L)
        ox, oy = jnp.meshgrid(ims, ims)
        ox = ox.ravel(); oy = oy.ravel()                             # (M,)
        sin2 = jnp.float32(np.sin(self.theta) ** 2)
        s_l2 = jnp.float32(self.delta_chi_a * self.B0) * sin2 * 0.5
        s_m0 = jnp.float32(self.delta_chi_a * self.B0) * (sin2 * 0.5)
        c2a = jnp.float32(np.cos(2 * self.alpha)); s2a = jnp.float32(np.sin(2 * self.alpha))
        R = None if self.R is None else jnp.asarray(self.R, jnp.float32)
        Lf = jnp.float32(self.L)
        wrap = bool(self.periodic)

        def delta_bz(r):
            r_perp = r if R is None else R @ r
            x, y = r_perp[0], r_perp[1]
            if wrap:
                x = ((x + 0.5 * Lf) % Lf) - 0.5 * Lf
                y = ((y + 0.5 * Lf) % Lf) - 0.5 * Lf
            dx = x - (c[:, 0:1] + ox[None, :])                       # (N, M)
            dy = y - (c[:, 1:2] + oy[None, :])
            r2 = dx * dx + dy * dy
            r4 = r2 * r2
            extra = r2 > b2[:, None]
            myelin = (r2 > a2[:, None]) & (~extra)
            intra = r2 < a2[:, None]
            r4e = jnp.where(extra, r4, jnp.inf)
            PhiC = jnp.sum(jnp.where(extra, (b2 - a2)[:, None] * (dx * dx - dy * dy) / r4e, 0.0))
            PhiS = jnp.sum(jnp.where(extra, (b2 - a2)[:, None] * 2.0 * dx * dy / r4e, 0.0))
            r4m = jnp.where(myelin, r4, jnp.inf)
            sf = jnp.where(myelin, (r4 - a4[:, None]) / (r4m * (b2 + a2)[:, None]), 0.0)
            PhiC += jnp.sum(sf * (dx * dx - dy * dy))
            PhiS += jnp.sum(sf * 2.0 * dx * dy)
            r2s = jnp.where(myelin, r2, b2[:, None])
            # Phi0 = ln(b/r) in the sheath, ln(b/a)=ln(1/g) in the lumen, 0 outside.
            Phi0 = (jnp.sum(jnp.where(myelin, -0.5 * jnp.log(r2s / b2[:, None]), 0.0))
                    + jnp.sum(jnp.where(intra, ln_ba[:, None], 0.0)))
            return s_m0 * Phi0 + s_l2 * (c2a * PhiC + s2a * PhiS)

        return delta_bz


# =============================================================================== #
# Arbitrary 3-D distribution on a grid (the mesh route)
# =============================================================================== #
def _unit(v, axis=-1, eps=1e-30):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def _as_voxel_size(voxel_size, ndim=3):
    vs = np.atleast_1d(np.asarray(voxel_size, float))
    if vs.size == 1:
        vs = np.repeat(vs, ndim)
    if vs.size != ndim:
        raise ValueError(f"voxel_size must be a scalar or length-{ndim}, got {vs}")
    return vs


def _khat(shape, voxel_size):
    """Unit wave-vector component grids ``khat_i`` (zero at DC) and the DC mask."""
    vs = _as_voxel_size(voxel_size, len(shape))
    ks = [2.0 * np.pi * np.fft.fftfreq(n, d=d) for n, d in zip(shape, vs)]
    K = np.meshgrid(*ks, indexing="ij")
    k2 = sum(k ** 2 for k in K)
    dc = k2 == 0.0
    inv = np.where(dc, 0.0, 1.0 / np.sqrt(np.where(dc, 1.0, k2)))
    khat = np.stack([k * inv for k in K], axis=0)
    return khat, dc


def myelin_susceptibility_tensor(myelin_mask, radial_dir, chi_iso=0.0, chi_aniso=0.0):
    """Per-voxel symmetric χ tensor ``χ = χ_iso·I + χ_aniso·(n n^T − I/3)`` in the mask.

    ``myelin_mask`` (Nx,Ny,Nz) fraction in [0,1]; ``radial_dir`` (Nx,Ny,Nz,3) local
    membrane-normal (re-normalised).  Returns ``chi6`` (Nx,Ny,Nz,6): xx,yy,zz,xy,xz,yz.
    """
    m = np.asarray(myelin_mask, float)
    n = _unit(np.asarray(radial_dir, float), axis=-1)
    chi6 = np.zeros(m.shape + (6,), float)
    for c, (i, j) in enumerate(_SYM6):
        term = 0.0
        if chi_aniso != 0.0:
            term = term + chi_aniso * (n[..., i] * n[..., j] - (1.0 / 3.0 if i == j else 0.0))
        if chi_iso != 0.0 and i == j:
            term = term + chi_iso
        chi6[..., c] = m * term
    return chi6


def dipole_field(chi6, voxel_size, b0_dir, B0=1.0):
    """Off-resonance field ``ΔBz(r)`` (T) of a χ-tensor grid (one forward + inverse FFT).

    Lorentz-sphere-corrected k-space dipole (Salomir 2003; Marques & Bowtell 2005;
    Wharton & Bowtell 2012 for the anisotropic tensor):

        ΔB(k)/B0 = (1/3) H·χ(k)·H − (H·k̂)(k̂·χ(k)·H) ,

    with the k=0 term zeroed (Lorentz-sphere / zero-mean reference).  ``chi6`` is the
    symmetric tensor per voxel (see :func:`myelin_susceptibility_tensor`); ``voxel_size``
    m; ``b0_dir`` the main-field direction in the grid frame; ``B0`` tesla.
    """
    chi6 = np.asarray(chi6, float)
    shape = chi6.shape[:3]
    H = _unit(np.asarray(b0_dir, float).ravel())
    khat, dc = _khat(shape, voxel_size)
    chik = np.stack([np.fft.fftn(chi6[..., c]) for c in range(6)], axis=0)

    def full(t6):
        M = [[None] * 3 for _ in range(3)]
        for c, (i, j) in enumerate(_SYM6):
            M[i][j] = t6[c]
            M[j][i] = t6[c]
        return M

    C = full(chik)
    chiH = [sum(C[i][j] * H[j] for j in range(3)) for i in range(3)]
    HchiH = sum(H[i] * chiH[i] for i in range(3))
    khat_chiH = sum(khat[i] * chiH[i] for i in range(3))
    Hkhat = sum(H[i] * khat[i] for i in range(3))
    kernel = (1.0 / 3.0) * HchiH - Hkhat * khat_chiH
    kernel[dc] = 0.0
    return np.real(np.fft.ifftn(kernel)) * float(B0)


def radial_from_sdf(sdf, voxel_size):
    """Local radial (membrane-normal) unit field ``n = grad(sdf)/|grad(sdf)|``."""
    vs = _as_voxel_size(np.asarray(voxel_size, float), 3)
    g = np.gradient(np.asarray(sdf, float), *vs, edge_order=2)
    return _unit(np.stack(g, axis=-1), axis=-1)


def sample_grid(grid, positions, origin, voxel_size, periodic=False, order=1):
    """Host (scipy) trilinear sample of a field grid at continuous positions.

    ``origin`` is the world coordinate of the corner of voxel ``(0,0,0)`` (voxel centre
    at ``origin + 0.5*voxel_size``).  Used for analysis/tests; the Bloch walk uses the
    JAX sampler in :meth:`GridSusceptibility.delta_bz_fn`.
    """
    from scipy.ndimage import map_coordinates

    pos = np.asarray(positions, float)
    lead = pos.shape[:-1]
    vs = _as_voxel_size(voxel_size, 3)
    org = np.asarray(origin, float).ravel()
    idx = (pos.reshape(-1, 3) - org) / vs - 0.5
    if isinstance(periodic, bool):
        periodic = (periodic, periodic, periodic)
    coords = np.empty((3, idx.shape[0]), float)
    for a in range(3):
        coords[a] = np.mod(idx[:, a], grid.shape[a]) if periodic[a] else idx[:, a]
    mode = "grid-wrap" if any(periodic) else "nearest"
    return map_coordinates(grid, coords, order=order, mode=mode).reshape(lead)


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


def _axis_to_z_rotation(axis):
    """Rotation mapping unit ``axis`` -> +z (identity when already +z)."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(axis, z)
    s = np.linalg.norm(v)
    c = float(np.dot(axis, z))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
