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


def segment_basis(rho_vec, u, z1, z2, a, b):
    """The 13 channels of one FINITE segment of the hollow cylinder, seen from outside the sheath: the dipole kernel
    ``D = (I - 3 n n^T) / (4 pi r^3)`` integrated in closed form along the segment, whose ends are at axial
    coordinates ``z1 < z2`` from the point's foot on the axis (``c(z') = foot + z' u``, ``p - c = rho_vec - z' u``),
    times the sheath's cross-section ``pi (b^2 - a^2)``: ``M_P = A L``, ``M_A = -A sym(L T)`` with ``L`` the
    integrated tensor and ``T = P / 2 - I / 3`` the annulus-averaged director tensor. For an infinite line it is
    the outside formula of :func:`hollow_cylinder_basis` (``M_P = -A S / 2 pi rho^2``, ``M_A = A S / 12 pi rho^2``),
    far away the point dipole of the segment's moment, ``A L (I - 3 r r^T) / 4 pi r^3``; a line cut into segments
    sums to the line exactly. Within the sheath's radius ``rho`` is continued at the surface (``rho >= b``): the
    value a point inside another segment's extension sees. Traceless in ``M_P``, no local term."""
    rho_vec = jnp.asarray(rho_vec); u = jnp.asarray(u); dt = rho_vec.dtype
    a = jnp.asarray(a, dt); b = jnp.asarray(b, dt); z1 = jnp.asarray(z1, dt); z2 = jnp.asarray(z2, dt)
    # in units of the outer radius: the primitives are then of order one (in metres they reach 1e24 and a
    # float32 difference of two of them is noise)
    rho_vec = rho_vec / b[..., None]; z1 = z1 / b; z2 = z2 / b; a = a / b; b = jnp.ones_like(b)
    rho = jnp.linalg.norm(rho_vec, axis=-1)
    rho_c = jnp.maximum(rho, 1.0 + 1e-6)
    rv = rho_vec * (rho_c / jnp.maximum(rho, jnp.asarray(1e-30, dt)))[..., None]
    r2 = rho_c * rho_c

    def prim(z):                                       # the four primitives at z (r^2 = rho^2 + z^2)
        rr = jnp.sqrt(r2 + z * z); r3 = rr * rr * rr
        return (z / (r2 * rr), z * (2.0 * z * z + 3.0 * r2) / (3.0 * r2 * r2 * r3), -1.0 / (3.0 * r3), z * z * z / (3.0 * r2 * r3))
    p1, p2 = prim(z1), prim(z2)
    i0, j0, j1, j2 = (q2 - q1 for q1, q2 in zip(p1, p2))
    # the six components of L = (I i0 - 3 (rv rv^T j0 - (rv u^T + u rv^T) j1 + u u^T j2)) / 4 pi, no 3x3 intermediates
    # (a 1536 x 4096 block of 3x3 tensors is 226 MB per intermediate; this form moves a third of that)
    A = jnp.pi * (b * b - a * a)
    c = A / (4.0 * jnp.pi)
    rx, ry, rz = rv[..., 0], rv[..., 1], rv[..., 2]; ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    def L6(ri, rj, ui, uj, diag):
        return c * ((i0 if diag else 0.0) - 3.0 * (ri * rj * j0 - (ri * uj + ui * rj) * j1 + ui * uj * j2))
    Lxx = L6(rx, rx, ux, ux, True); Lyy = L6(ry, ry, uy, uy, True); Lzz = L6(rz, rz, uz, uz, True)
    Lxy = L6(rx, ry, ux, uy, False); Lxz = L6(rx, rz, ux, uz, False); Lyz = L6(ry, rz, uy, uz, False)
    MP = jnp.stack([Lxx, Lyy, Lzz, Lxy, Lxz, Lyz], axis=-1)                   # A L
    # M_A = -A sym(L T) with T = P/2 - I/3 = I/6 - u u^T/2:  -[A L / 6 - sym((A L u) u^T) / 2]
    wx = Lxx * ux + Lxy * uy + Lxz * uz; wy = Lxy * ux + Lyy * uy + Lyz * uz; wz = Lxz * ux + Lyz * uy + Lzz * uz   # (A L) u
    MA = jnp.stack([-(Lxx / 6.0 - wx * ux / 2.0), -(Lyy / 6.0 - wy * uy / 2.0), -(Lzz / 6.0 - wz * uz / 2.0),
                    -(Lxy / 6.0 - (wx * uy + wy * ux) / 4.0), -(Lxz / 6.0 - (wx * uz + wz * ux) / 4.0), -(Lyz / 6.0 - (wy * uz + wz * uy) / 4.0)], axis=-1)
    return jnp.concatenate([jnp.zeros(rho.shape + (1,), dt), MP, MA], axis=-1)
def q_of_H(b0_dir):
    """``Q(H) = (Hx^2, Hy^2, Hz^2, 2 Hx Hy, 2 Hx Hz, 2 Hy Hz)``: the contraction weights of the six symmetric
    components, so that ``Q . sym6(M) = H . M . H``."""
    h = np.asarray(b0_dir, float).ravel(); h = h / np.linalg.norm(h)
    return np.array([h[0] ** 2, h[1] ** 2, h[2] ** 2, 2 * h[0] * h[1], 2 * h[0] * h[2], 2 * h[1] * h[2]])


def field_terms(channels, b0_dir, axis=-1):
    """The two susceptibility terms of the channels along ``axis`` (``iso_local, iso_P (6), aniso_G (6)`` in that
    order, 7 or 13 of them) for a field direction: ``(iso, aniso)`` with ``iso = iso_local - Q . iso_P`` and
    ``aniso = Q . aniso_G`` (``None`` when the channels carry no anisotropic basis). ``dB = B0 (chi_iso iso +
    chi_aniso aniso)``: the one contraction of every route, grid, path and pack."""
    c = np.moveaxis(np.asarray(channels, float), axis, -1)
    q = q_of_H(b0_dir)
    iso = c[..., 0] - c[..., 1:7] @ q
    aniso = c[..., 7:13] @ q if c.shape[-1] >= 13 else None
    return iso, aniso


def contract(channels, b0_dir, *, B0, chi_iso=0.0, chi_aniso=0.0, axis=-1):
    """``dB`` (Tesla) at each point from its channels for one configuration (:func:`field_terms`)."""
    iso, aniso = field_terms(channels, b0_dir, axis=axis)
    dB = np.zeros(iso.shape, float)
    if chi_iso:
        dB = dB + float(chi_iso) * iso
    if chi_aniso:
        if aniso is None:
            raise ValueError("these channels carry no anisotropic basis (7 of 13); chi_aniso must be 0")
        dB = dB + float(chi_aniso) * aniso
    return dB * float(B0)


def annulus_mean_log(a, b):
    """The area average of ``ln(b/r)`` over the annulus ``a < r < b``: ``1/2 - a^2 ln(b/a) / (b^2 - a^2)``."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    return 0.5 - a ** 2 * np.log(b / a) / (b ** 2 - a ** 2)
