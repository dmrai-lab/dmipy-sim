"""Procedural myelin mesh + susceptibility-source builders.

Two kinds of output from one parametric description of a (possibly pathological)
myelinated axon:

* **surface meshes** ``(vertices, faces)`` for the diffusion walk -- inner (lumen) and
  outer (myelin) boundaries as z-periodic tubes; and
* **grid susceptibility sources** ``(mask, radial_dir)`` for the field solver
  (:mod:`dmipy_sim.susceptibility`), built analytically (exact, no voxelisation error) so
  the field validation isolates the physics.

The generators cover substrates that require a grid/mesh susceptibility treatment (no
straight-parallel-cylinder closed form): :func:`myelinated_cylinder` (the analytic
baseline), :func:`undulating_myelin` (sheath thickness modulated along z), and
:func:`half_bare_myelin` (sheath removed over a sector/segment).  :func:`voxelize_shell`
turns an arbitrary inner/outer mesh pair into ``(mask, radial)`` for real substrates.

Units: metres throughout.  Meshes are centred on the origin, ``z in [-length/2, length/2]``.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "icosphere", "cylinder_tube", "myelinated_cylinder", "undulating_myelin",
    "half_bare_myelin", "straight_myelin_source", "undulating_myelin_source",
    "half_bare_myelin_source", "grid_axes", "voxelize_shell",
]


# --------------------------------------------------------------------------- #
# Surface-mesh primitives
# --------------------------------------------------------------------------- #
def _tube(radius_fn, length, n_ang, n_ax, z0=None, theta_range=(0.0, 2.0 * np.pi),
          close_azimuth=True):
    """Open tube along z; ``radius_fn(z, theta)->r`` (m).  Returns (V, F)."""
    z0 = -length / 2.0 if z0 is None else z0
    zs = np.linspace(z0, z0 + length, n_ax + 1)
    full = close_azimuth and abs((theta_range[1] - theta_range[0]) - 2 * np.pi) < 1e-9
    thetas = np.linspace(theta_range[0], theta_range[1], n_ang, endpoint=not full) if full \
        else np.linspace(theta_range[0], theta_range[1], n_ang)
    nth = len(thetas)
    V = []
    for z in zs:
        for th in thetas:
            r = radius_fn(z, th)
            V.append((r * np.cos(th), r * np.sin(th), z))
    V = np.asarray(V, float)
    F = []
    for iz in range(n_ax):
        for it in range(nth if full else nth - 1):
            it2 = (it + 1) % nth if full else it + 1
            a = iz * nth + it
            b = iz * nth + it2
            c = (iz + 1) * nth + it
            d = (iz + 1) * nth + it2
            F.append((a, b, d))
            F.append((a, d, c))
    return V, np.asarray(F, np.int64)


def cylinder_tube(radius, length, n_ang=64, n_ax=32):
    """Straight open cylindrical tube of constant ``radius`` along z."""
    return _tube(lambda z, th: radius, length, n_ang, n_ax)


def icosphere(radius, subdivisions=2):
    """Geodesic icosphere of given radius (recursive icosahedron subdivision)."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], np.int64)
    for _ in range(subdivisions):
        Vl = list(V)
        mid = {}

        def _m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                mid[k] = len(Vl)
                Vl.append((V[a] + V[b]) / 2.0)
            return mid[k]

        nf = []
        for a, b, c in F:
            ab, bc, ca = _m(a, b), _m(b, c), _m(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        V = np.asarray(Vl, float)
        F = np.asarray(nf, np.int64)
    V = V / np.linalg.norm(V, axis=1, keepdims=True) * radius
    return V, F


# --------------------------------------------------------------------------- #
# Myelinated-axon surface meshes (inner lumen + outer myelin boundary)
# --------------------------------------------------------------------------- #
def myelinated_cylinder(inner_radius, outer_radius, length, n_ang=64, n_ax=32):
    """Straight hollow cylinder -- the analytic-baseline substrate.

    Returns ``{'inner': (V,F), 'outer': (V,F), 'g_ratio': a/b}``.
    """
    a, b = float(inner_radius), float(outer_radius)
    if not 0 < a < b:
        raise ValueError("require 0 < inner_radius < outer_radius")
    return {"inner": cylinder_tube(a, length, n_ang, n_ax),
            "outer": cylinder_tube(b, length, n_ang, n_ax),
            "g_ratio": a / b}


def undulating_myelin(inner_radius, outer_radius, length, amplitude, wavelength,
                      undulate="outer", n_ang=64, n_ax=96):
    """Myelinated cylinder whose sheath thickness varies along z."""
    a0, b0, A, lam = map(float, (inner_radius, outer_radius, amplitude, wavelength))

    def rb(z, th):
        return b0 * (1 + A * np.sin(2 * np.pi * z / lam)) if undulate in ("outer", "both") else b0

    def ra(z, th):
        return a0 * (1 + A * np.sin(2 * np.pi * z / lam)) if undulate in ("inner", "both") else a0

    return {"inner": _tube(ra, length, n_ang, n_ax),
            "outer": _tube(rb, length, n_ang, n_ax),
            "params": dict(inner_radius=a0, outer_radius=b0, amplitude=A,
                           wavelength=lam, undulate=undulate)}


def half_bare_myelin(inner_radius, outer_radius, length, bare_fraction=0.5,
                     mode="angular", n_ang=96, n_ax=48):
    """Myelinated cylinder with the outer sheath removed over part of the axon."""
    a, b = float(inner_radius), float(outer_radius)
    inner = cylinder_tube(a, length, n_ang, n_ax)
    if mode == "angular":
        th_bare = 2 * np.pi * float(bare_fraction)

        def rb(z, th):
            return a if (th % (2 * np.pi)) < th_bare else b
    elif mode == "axial":
        z_bare = float(bare_fraction) * length

        def rb(z, th):
            return a if z < (-length / 2.0 + z_bare) else b
    else:
        raise ValueError("mode must be 'angular' or 'axial'")
    outer = _tube(rb, length, n_ang, n_ax)
    return {"inner": inner, "outer": outer,
            "params": dict(inner_radius=a, outer_radius=b, bare_fraction=float(bare_fraction),
                           mode=mode)}


# --------------------------------------------------------------------------- #
# Grid susceptibility sources (analytic mask + radial director)
# --------------------------------------------------------------------------- #
def grid_axes(cell_size, n_grid, center=True):
    """Voxel-centre coordinates + ``(voxel_size, origin)`` for a cubic cell.

    Returns ``(X, Y, Z, voxel_size, origin)``; ``origin`` is the corner of voxel (0,0,0).
    """
    L = float(cell_size)
    vs = L / n_grid
    off = -L / 2.0 if center else 0.0
    ax = (np.arange(n_grid) + 0.5) * vs + off
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    origin = np.array([off, off, off], float)
    return X, Y, Z, vs, origin


def _radial_xy(X, Y):
    rho = np.sqrt(X ** 2 + Y ** 2)
    inv = np.where(rho > 0, 1.0 / np.maximum(rho, 1e-30), 0.0)
    return np.stack([X * inv, Y * inv, np.zeros_like(X)], axis=-1), rho


def straight_myelin_source(X, Y, Z, inner_radius, outer_radius):
    """Exact ``(mask, radial)`` for a straight hollow cylinder along z."""
    radial, rho = _radial_xy(X, Y)
    mask = ((rho >= inner_radius) & (rho <= outer_radius)).astype(float)
    return mask, radial


def undulating_myelin_source(X, Y, Z, inner_radius, outer_radius, amplitude,
                             wavelength, undulate="outer"):
    """Exact ``(mask, radial)`` for the undulating sheath."""
    radial, rho = _radial_xy(X, Y)
    b = outer_radius * (1 + amplitude * np.sin(2 * np.pi * Z / wavelength)) \
        if undulate in ("outer", "both") else np.full_like(Z, outer_radius)
    a = inner_radius * (1 + amplitude * np.sin(2 * np.pi * Z / wavelength)) \
        if undulate in ("inner", "both") else np.full_like(Z, inner_radius)
    mask = ((rho >= a) & (rho <= b)).astype(float)
    return mask, radial


def half_bare_myelin_source(X, Y, Z, inner_radius, outer_radius, bare_fraction=0.5,
                            mode="angular"):
    """Exact ``(mask, radial)`` for the half-bare sheath."""
    radial, rho = _radial_xy(X, Y)
    in_annulus = (rho >= inner_radius) & (rho <= outer_radius)
    if mode == "angular":
        phi = np.mod(np.arctan2(Y, X), 2 * np.pi)
        present = phi >= (2 * np.pi * bare_fraction)
    elif mode == "axial":
        z_bare = bare_fraction * (Z.max() - Z.min() + (Z[0, 0, 1] - Z[0, 0, 0]))
        present = Z >= (Z.min() + z_bare)
    else:
        raise ValueError("mode must be 'angular' or 'axial'")
    mask = (in_annulus & present).astype(float)
    return mask, radial


# --------------------------------------------------------------------------- #
# Arbitrary inner/outer mesh pair -> (mask, radial)
# --------------------------------------------------------------------------- #
def voxelize_shell(inner, outer, voxel_size, *, bbox=None, pad=1, origin=None,
                   compute_radial=True):
    """Voxelise an inner/outer mesh pair into ``(mask, radial, voxel_size, origin)``.

    ``mask`` is 1 inside the outer surface AND outside the inner surface (the myelin
    sheath).  ``radial`` (when ``compute_radial``) is the outward membrane normal from
    ``grad`` of the outer signed-distance field.  Uses :func:`trimesh.Trimesh.contains`
    (rtree-accelerated).  ``inner``/``outer`` are ``(vertices, faces)`` tuples in metres.
    """
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover
        raise ImportError("voxelize_shell requires trimesh (pip install 'dmipy-sim[mesh]')."
                          ) from exc
    from .susceptibility import radial_from_sdf, _as_voxel_size

    m_out = trimesh.Trimesh(vertices=np.asarray(outer[0], float),
                            faces=np.asarray(outer[1], np.int64), process=False)
    m_in = trimesh.Trimesh(vertices=np.asarray(inner[0], float),
                           faces=np.asarray(inner[1], np.int64), process=False)
    vs = _as_voxel_size(voxel_size, 3)
    lo, hi = (m_out.bounds if bbox is None else (np.asarray(bbox[0], float),
                                                 np.asarray(bbox[1], float)))
    org = (np.asarray(lo, float) - pad * vs) if origin is None else np.asarray(origin, float)
    dims = np.maximum(1, np.ceil((hi + pad * vs - org) / vs).astype(int))
    ax = [org[a] + (np.arange(dims[a]) + 0.5) * vs[a] for a in range(3)]
    X, Y, Z = np.meshgrid(ax[0], ax[1], ax[2], indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    ins_out = m_out.contains(pts).reshape(X.shape)
    ins_in = m_in.contains(pts).reshape(X.shape)
    mask = (ins_out & ~ins_in).astype(float)

    if compute_radial:
        sd_out = trimesh.proximity.signed_distance(m_out, pts).reshape(X.shape)
        radial = radial_from_sdf(-sd_out, vs)
    else:
        radial = None
    return mask, radial, vs, org
