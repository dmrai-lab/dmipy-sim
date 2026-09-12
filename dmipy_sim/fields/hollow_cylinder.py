"""The field basis of an infinite hollow cylinder in closed form: the per-segment building block of a strand
substrate's susceptibility field (:mod:`dmipy_sim.fields.strand_field`) and the analytic myelin provider of the
forward Bloch engine (:class:`dmipy_sim.fields.susceptibility.MyelinSusceptibility`).

The sheath ``a < r < b`` carries the susceptibility tensor ``chi_I I + chi_A (n n^T - I/3)`` with ``n`` the radial
(lipid) director. The along-B0 field, in the same 13-component basis the k-space route stores
(:func:`dmipy_sim.fields.susceptibility_field.field_basis`), is

    dB/B0 = chi_I [ m/3 - H.M_P.H ] + chi_A H.M_A.H

with ``m`` the sheath indicator, ``H`` the unit B0 direction and, in the point's own frame (``u`` the axis,
``e1`` the radial unit vector, ``e2 = u x e1``; ``P = I - u u^T``, ``S = e1 e1^T - e2 e2^T``):

    lumen  (r < a):      M_P = 0                              M_A = (1/2) ln(b/a) P
    sheath (a <= r < b): M_P = P/2 + (a^2 / 2 r^2) S          M_A = ((1/2) ln(b/r) - 5/18) P - u u^T / 9 - (1/12)(1 + a^2/r^2) S
    outside (r >= b):    M_P = -((b^2 - a^2) / 2 r^2) S       M_A = ((b^2 - a^2) / 12 r^2) S

so that ``H.P.H = sin^2 theta``, ``H.S.H = sin^2 theta cos 2 alpha`` (theta the fibre-to-B0 angle, alpha the azimuth
of B0 from the point's radial direction). The lumen field is the Wharton-Bowtell one, ``(1/2) chi_A sin^2 theta
ln(1/g)``, uniform and zero with B0 along the fibre; outside, ``(chi_I + chi_A/6) (b^2 - a^2) sin^2 theta cos 2
alpha / (2 r^2)`` (Wharton & Bowtell 2012, PNAS 109:18559, their eq. 2 with the anisotropic term in this tensor
convention). The forms were derived from the k-space dipole kernel itself (a 4096^2 two-dimensional solve of one
annulus at 0.02 um, fitted region by region) and agree with it on all 13 components. Every component is the bare
superposition value: the k-space route subtracts its domain mean (the Lorentz reference), a uniform offset that
a magnitude signal cannot see and that :meth:`StrandFieldBasis.channels` subtracts analytically.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

#: the 13 channels, in the order the pack's path channel stores them
CHANNEL_NAMES = ("iso_local",) + tuple(f"iso_P_{c}" for c in ("xx", "yy", "zz", "xy", "xz", "yz")) \
    + tuple(f"aniso_G_{c}" for c in ("xx", "yy", "zz", "xy", "xz", "yz"))


def _sym6(M):
    """``(..., 3, 3)`` symmetric -> ``(..., 6)`` in the (xx, yy, zz, xy, xz, yz) order."""
    return jnp.stack([M[..., 0, 0], M[..., 1, 1], M[..., 2, 2], M[..., 0, 1], M[..., 0, 2], M[..., 1, 2]], axis=-1)


def hollow_cylinder_basis(rho_vec, u, a, b):
    """The 13 basis channels of one infinite hollow cylinder at points given by ``rho_vec`` ``(..., 3)``, the
    vector from the axis to the point (perpendicular to the axis), ``u`` ``(..., 3)`` the unit axis, ``a`` and
    ``b`` ``(...)`` the inner and outer radii (metres). Returns ``(..., 13)``: ``iso_local``, the six components
    of ``M_P``, the six of ``M_A`` (see the module docstring). Traceable (jax.numpy); broadcasts over leading
    axes."""
    rho_vec = jnp.asarray(rho_vec); u = jnp.asarray(u)
    a = jnp.asarray(a, rho_vec.dtype); b = jnp.asarray(b, rho_vec.dtype)
    rho = jnp.linalg.norm(rho_vec, axis=-1)
    rho_safe = jnp.maximum(rho, jnp.asarray(1e-30, rho_vec.dtype))
    e1 = rho_vec / rho_safe[..., None]
    e2 = jnp.cross(u, e1)
    eye = jnp.eye(3, dtype=rho_vec.dtype)
    UU = u[..., :, None] * u[..., None, :]
    P = eye - UU
    S = e1[..., :, None] * e1[..., None, :] - e2[..., :, None] * e2[..., None, :]
    r2 = rho_safe ** 2
    a2 = a ** 2; b2 = b ** 2
    lumen = rho < a
    sheath = (rho >= a) & (rho < b)
    out = rho >= b
    m = sheath.astype(rho_vec.dtype)
    # isotropic
    cP_sheath = jnp.asarray(0.5, rho_vec.dtype)
    cS_P = jnp.where(sheath, a2 / (2.0 * r2), jnp.where(out, -(b2 - a2) / (2.0 * r2), 0.0))
    MP = jnp.where(sheath, cP_sheath, 0.0)[..., None, None] * P + cS_P[..., None, None] * S
    # anisotropic
    cP_A = jnp.where(lumen, 0.5 * jnp.log(b / a), jnp.where(sheath, 0.5 * jnp.log(b / rho_safe) - 5.0 / 18.0, 0.0))
    cU_A = jnp.where(sheath, -1.0 / 9.0, 0.0)
    cS_A = jnp.where(sheath, -(1.0 + a2 / r2) / 12.0, jnp.where(out, (b2 - a2) / (12.0 * r2), 0.0))
    MA = cP_A[..., None, None] * P + cU_A[..., None, None] * UU + cS_A[..., None, None] * S
    return jnp.concatenate([(m / 3.0)[..., None], _sym6(MP), _sym6(MA)], axis=-1)


def q_of_H(b0_dir):
    """``Q(H) = (Hx^2, Hy^2, Hz^2, 2 Hx Hy, 2 Hx Hz, 2 Hy Hz)``: the contraction weights of the six symmetric
    components, so that ``Q . sym6(M) = H . M . H``."""
    h = np.asarray(b0_dir, float).ravel(); h = h / np.linalg.norm(h)
    return np.array([h[0] ** 2, h[1] ** 2, h[2] ** 2, 2 * h[0] * h[1], 2 * h[0] * h[2], 2 * h[1] * h[2]])


def contract(channels, b0_dir, *, B0, chi_iso=0.0, chi_aniso=0.0):
    """``dB`` (Tesla) at each point from its 13 channels ``(..., 13)`` for one configuration: the same
    contraction :func:`dmipy_sim.fields.susceptibility_field.assemble_field` applies to grids."""
    c = np.asarray(channels, float)
    q = q_of_H(b0_dir)
    dB = np.zeros(c.shape[:-1], float)
    if chi_iso:
        dB = dB + float(chi_iso) * (c[..., 0] - c[..., 1:7] @ q)
    if chi_aniso:
        if c.shape[-1] < 13:
            raise ValueError("these channels carry no anisotropic basis (7 of 13); chi_aniso must be 0")
        dB = dB + float(chi_aniso) * (c[..., 7:13] @ q)
    return dB * float(B0)


def annulus_mean_log(a, b):
    """The area average of ``ln(b/r)`` over the annulus ``a < r < b``: ``1/2 - a^2 ln(b/a) / (b^2 - a^2)``."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    return 0.5 - a ** 2 * np.log(b / a) / (b ** 2 - a ** 2)
