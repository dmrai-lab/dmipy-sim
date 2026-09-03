"""Mesh/grid susceptibility off-resonance field (isotropic + anisotropic myelin).

The analytic ``PackedMyelinatedCylinders.compute_susceptibility_field_maps`` gives the
off-resonance field of *straight, parallel* hollow cylinders in closed form (the
2-D ``Phi_C/Phi_S/Phi_0`` maps).  That specialisation cannot represent a myelin
sheath whose geometry varies along the axon (undulating thickness, a bare sector,
a branch, or any PLY mesh), because there is no global cylinder axis to define the
perpendicular plane or the radial (lipid) director.

This module computes the field of an **arbitrary 3-D susceptibility distribution**
on a regular grid via the Lorentz-sphere-corrected k-space dipole model
(Salomir 2003; Marques & Bowtell 2005 for the isotropic case; Wharton & Bowtell
2012 for the anisotropic myelin tensor).  It handles

* an **isotropic** scalar susceptibility ``chi_I`` (special case), and
* an **anisotropic** uniaxial myelin tensor with the symmetry axis along the local
  radial (membrane-normal) direction ``n``:  ``chi = chi_I I + chi_A (n n^T - I/3)``.

The local director ``n`` comes from the geometry itself (the gradient of a signed
distance field, or the mesh's smooth surface normal), so the model generalises to
undulating / half-bare / meshed sheaths where an analytic normal does not exist.

Design mirroring the existing susceptibility phasor
---------------------------------------------------
Because the along-field component ``dB(r)`` is a **quadratic form in the unit main
field direction ``H``** and **linear in ``(chi_I, chi_A)``**, the geometry-only part
factorises into a small set of basis grids (:func:`field_basis`) computed once by
FFT.  Any main-field direction, field strength ``B0`` and susceptibility values are
then applied by a cheap real contraction (:func:`assemble_field`) — the 3-D
generalisation of applying ``(theta, alpha, dchi_a, B0)`` to ``Phi_C/Phi_S/Phi_0``.
Sampling the basis along a stored trajectory (:func:`sample_grid`) therefore lets a
single mesh walk replay an entire ``(B0-direction, B0, chi_I, chi_A, g)`` sweep.

Units: lengths m, ``voxel_size`` m, ``B0`` Tesla, susceptibilities dimensionless
(SI volume susceptibility; e.g. ``chi_A = -0.1e-6``).  ``dB`` is returned in Tesla.
The phase accrued by a spin is ``phi = GAMMA * integral eps(t) dB(r(t)) dt`` with
``GAMMA`` from :mod:`dmipy_sim.constants`, exactly as the analytic path.

Reference / zero level
----------------------
The k=0 (mean-field) term is set to zero — the Lorentz-sphere convention shared by
QSM forward models and by the analytic maps here.  The spatial *structure* (and
hence every intra/extra difference and the diffusion dephasing) is independent of
this global offset, and a spatially uniform offset is exactly refocused by a spin
echo, so this choice is physically inert for SE/STE signals.
"""

from __future__ import annotations

import warnings
import numpy as np

# Symmetric 3x3 tensor stored as 6 components in this fixed order.
_SYM6 = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


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
    """Unit wave-vector component grids ``khat_i`` and a DC mask.

    Returns ``(khat, dc)`` where ``khat`` has shape ``(3, *shape)`` (the components
    of ``k / |k|``, zero at k=0) and ``dc`` is the boolean k=0 location.
    """
    vs = _as_voxel_size(voxel_size, len(shape))
    ks = [2.0 * np.pi * np.fft.fftfreq(n, d=d) for n, d in zip(shape, vs)]
    K = np.meshgrid(*ks, indexing="ij")
    k2 = sum(k ** 2 for k in K)
    dc = k2 == 0.0
    inv = np.where(dc, 0.0, 1.0 / np.sqrt(np.where(dc, 1.0, k2)))
    khat = np.stack([k * inv for k in K], axis=0)     # (3, *shape); zero at DC
    return khat, dc


def myelin_susceptibility_tensor(myelin_mask, radial_dir, chi_iso=0.0, chi_aniso=0.0):
    """Per-voxel symmetric susceptibility tensor of a myelin distribution.

    ``chi(r) = chi_iso * I + chi_aniso * (n n^T - I/3)`` inside the myelin, zero
    elsewhere.  With ``chi_aniso = 0`` this is a purely isotropic source (the
    general special case); with ``chi_iso = 0`` a purely anisotropic one.

    Parameters
    ----------
    myelin_mask : (Nx,Ny,Nz) array
        Myelin volume fraction per voxel in ``[0, 1]`` (a 0/1 indicator, or a
        partial-volume fraction for a sub-voxel-accurate source).
    radial_dir : (Nx,Ny,Nz,3) array
        Local radial (membrane-normal / lipid-director) unit vectors.  Only used
        where ``myelin_mask > 0``; need not be normalised (it is re-normalised).
    chi_iso, chi_aniso : float
        Isotropic and anisotropic susceptibility (SI, dimensionless).

    Returns
    -------
    chi6 : (Nx,Ny,Nz,6) float64
        The 6 independent components (xx,yy,zz,xy,xz,yz) of ``chi(r)``.
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
    """Off-resonance field ``dB(r)`` (Tesla) of a susceptibility-tensor grid.

    Direct single-configuration solver (one forward + one inverse FFT).  Implements
    the along-field component

        dB(k)/B0 = (1/3) H . chi(k) . H  -  (H . khat)(khat . chi(k) . H) ,

    with the k=0 term zeroed (Lorentz-sphere / zero-mean reference).  ``H`` is the
    unit main-field direction ``b0_dir``.

    Parameters
    ----------
    chi6 : (Nx,Ny,Nz,6) array
        Symmetric susceptibility tensor per voxel (see
        :func:`myelin_susceptibility_tensor`), SI dimensionless.
    voxel_size : float or (3,) array
        Voxel edge length(s) in metres.
    b0_dir : (3,) array
        Main-field direction in the grid frame (need not be unit).
    B0 : float
        Field strength in Tesla.

    Returns
    -------
    dB : (Nx,Ny,Nz) float64
        Field offset along B0, in Tesla.
    """
    chi6 = np.asarray(chi6, float)
    shape = chi6.shape[:3]
    H = _unit(np.asarray(b0_dir, float).ravel())
    khat, dc = _khat(shape, voxel_size)

    # chi(k), 6 complex components
    chik = np.stack([np.fft.fftn(chi6[..., c]) for c in range(6)], axis=0)

    def full(t6):
        """Expand a 6-vector field into the symmetric 3x3 (indexed [i][j])."""
        M = [[None] * 3 for _ in range(3)]
        for c, (i, j) in enumerate(_SYM6):
            M[i][j] = t6[c]
            M[j][i] = t6[c]
        return M

    C = full(chik)
    # (chi . H)_i = sum_j chi_ij H_j ; H . chi . H = sum_i H_i (chi.H)_i
    chiH = [sum(C[i][j] * H[j] for j in range(3)) for i in range(3)]
    HchiH = sum(H[i] * chiH[i] for i in range(3))
    khat_chiH = sum(khat[i] * chiH[i] for i in range(3))          # khat . chi . H
    Hkhat = sum(H[i] * khat[i] for i in range(3))                 # H . khat
    kernel = (1.0 / 3.0) * HchiH - Hkhat * khat_chiH
    kernel[dc] = 0.0
    dB = np.real(np.fft.ifftn(kernel)) * float(B0)
    return dB


def field_basis(myelin_mask, radial_dir, voxel_size, include_aniso=True, kspace_lowpass=None):
    """Geometry-only field-basis grids for cheap ``(H, B0, chi_I, chi_A)`` replay.

    The along-field response factorises as

        dB(r; H, B0, chi_I, chi_A)/B0
            = chi_I * [ iso_local(r) - Q(H) . iso_P(r) ]
            + chi_A *   Q(H) . aniso_G(r)

    where ``Q(H) = (Hx^2, Hy^2, Hz^2, 2 HxHy, 2 HxHz, 2 HyHz)`` contracts the 6
    symmetric components (see :func:`assemble_field`).  Every returned grid depends
    only on the geometry (mask + director), so a single set serves any main-field
    direction, strength and susceptibility — the 3-D analogue of the analytic
    ``Phi_C/Phi_S/Phi_0`` phasor maps.  All grids are zero-mean (Lorentz reference).

    Parameters
    ----------
    myelin_mask : (Nx,Ny,Nz) array
        Myelin fraction per voxel in ``[0, 1]``.
    radial_dir : (Nx,Ny,Nz,3) array
        Local radial unit vectors (used only where ``include_aniso``).
    voxel_size : float or (3,) array
        Voxel edge length(s), metres.
    include_aniso : bool
        Compute the anisotropic basis (6 extra IFFTs).  Set False for a purely
        isotropic source to save work.

    Returns
    -------
    basis : dict
        ``iso_local`` (Nx,Ny,Nz), ``iso_P`` (6,Nx,Ny,Nz),
        ``aniso_G`` (6,Nx,Ny,Nz) or None, plus ``shape`` and ``voxel_size``.
    """
    m = np.asarray(myelin_mask, float)
    shape = m.shape
    khat, dc = _khat(shape, voxel_size)

    # Optional k-space low-pass (raised-cosine roll-off from `kspace_lowpass`·Nyquist to Nyquist).
    # The dipole kernel k̂ᵢk̂ⱼ does not decay with |k|, so grid-scale source roughness (a hard/thin
    # mask, a gradient-derived director) rings at the grid Nyquist — content the grid cannot
    # faithfully represent anyway. Damping it removes that ringing (standard in QSM forward models).
    W = 1.0
    if kspace_lowpass is not None:
        axf = [np.fft.fftfreq(n) * 2.0 for n in shape]          # -1..1 in Nyquist units
        r = np.sqrt(sum(g ** 2 for g in np.meshgrid(*axf, indexing="ij")) / len(shape))
        W = np.ones_like(r); f = float(kspace_lowpass)
        ramp = (r > f) & (r < 1.0)
        W[ramp] = 0.5 * (1.0 + np.cos(np.pi * (r[ramp] - f) / (1.0 - f)))
        W[r >= 1.0] = 0.0

    mk = np.fft.fftn(m)

    # isotropic local term (1/3) m(r), zero-meaned (DC removed for consistency).
    iso_local = (m - m.mean()) / 3.0

    # iso_P_ij = IFFT[ mk * khat_i khat_j ]  (khat=0 at DC -> already zero-mean).
    iso_P = np.empty((6,) + shape, float)
    for c, (i, j) in enumerate(_SYM6):
        iso_P[c] = np.real(np.fft.ifftn(mk * khat[i] * khat[j] * W))

    aniso_G = None
    if include_aniso:
        n = _unit(np.asarray(radial_dir, float), axis=-1)
        # anisotropic tensor field T_cd(r) = m (n_c n_d - delta_cd/3), FFT -> Tk.
        Tk = {}
        for (i, j) in _SYM6:
            Td = m * (n[..., i] * n[..., j] - (1.0 / 3.0 if i == j else 0.0))
            Tk[(i, j)] = np.fft.fftn(Td)
            Tk[(j, i)] = Tk[(i, j)]
        # K_ij(k) = (1/3) Tk_ij - sum_c khat_i khat_c Tk_cj ; symmetrise in (i,j).
        aniso_G = np.empty((6,) + shape, complex)
        for c, (i, j) in enumerate(_SYM6):
            Kij = (1.0 / 3.0) * Tk[(i, j)]
            Kji = (1.0 / 3.0) * Tk[(j, i)]
            sec_ij = sum(khat[i] * khat[cc] * Tk[(cc, j)] for cc in range(3))
            sec_ji = sum(khat[j] * khat[cc] * Tk[(cc, i)] for cc in range(3))
            Ksym = 0.5 * ((Kij - sec_ij) + (Kji - sec_ji))
            Ksym[dc] = 0.0
            aniso_G[c] = Ksym
        aniso_G = np.stack([np.real(np.fft.ifftn(aniso_G[c] * W)) for c in range(6)], axis=0)

    return {
        "iso_local": iso_local,
        "iso_P": iso_P,
        "aniso_G": aniso_G,
        "shape": shape,
        "voxel_size": _as_voxel_size(voxel_size, 3),
    }


def _q_of_H(b0_dir):
    """Symmetric-contraction weights ``Q(H)`` for the 6-component basis."""
    h = _unit(np.asarray(b0_dir, float).ravel())
    return np.array([h[0] ** 2, h[1] ** 2, h[2] ** 2,
                     2 * h[0] * h[1], 2 * h[0] * h[2], 2 * h[1] * h[2]])


def assemble_field(basis, b0_dir, B0=1.0, chi_iso=0.0, chi_aniso=0.0):
    """Assemble ``dB(r)`` (Tesla) from a :func:`field_basis` for one configuration.

    Cheap real contraction — no FFT.  Sweep ``b0_dir``/``B0``/``chi_*`` for free.
    """
    q = _q_of_H(b0_dir)
    dB = np.zeros(basis["shape"], float)
    if chi_iso != 0.0:
        contr = np.tensordot(q, basis["iso_P"], axes=(0, 0))     # Q . iso_P
        dB = dB + chi_iso * (basis["iso_local"] - contr)
    if chi_aniso != 0.0:
        if basis["aniso_G"] is None:
            raise ValueError("basis was built with include_aniso=False; cannot apply chi_aniso.")
        dB = dB + chi_aniso * np.tensordot(q, basis["aniso_G"], axes=(0, 0))
    return dB * float(B0)


def sample_grid(grid, positions, origin, voxel_size, periodic=False, order=1):
    """Sample a scalar grid at continuous positions (trilinear by default).

    Parameters
    ----------
    grid : (Nx,Ny,Nz) array
        Field grid (e.g. ``dB`` from :func:`assemble_field`).
    positions : (..., 3) array
        Query positions in metres (any leading shape, e.g. ``(n_walkers, n_t, 3)``).
    origin : (3,) array
        World coordinate of the voxel **corner** of index ``(0,0,0)`` (metres); the
        voxel spans ``[origin, origin+voxel_size)`` with its centre at
        ``origin + 0.5*voxel_size`` — the convention returned by
        :func:`dmipy_sim.mesh_shapes.grid_axes`.
    voxel_size : float or (3,) array
        Voxel edge length(s), metres.
    periodic : bool or (bool,bool,bool)
        Wrap sampling per axis (use for a periodic cell); else clamp at the edge.
    order : int
        Interpolation order for :func:`scipy.ndimage.map_coordinates` (1 = linear).

    Returns
    -------
    values : (...) array
        Interpolated values, same leading shape as ``positions``.
    """
    from scipy.ndimage import map_coordinates

    pos = np.asarray(positions, float)
    lead = pos.shape[:-1]
    vs = _as_voxel_size(voxel_size, 3)
    org = np.asarray(origin, float).ravel()
    # index of a point at voxel-centre i is i (map_coordinates samples at integer
    # = voxel centre); centre i sits at org + (i+0.5)*vs, hence the -0.5 shift.
    idx = (pos.reshape(-1, 3) - org) / vs - 0.5       # fractional voxel coordinates
    if isinstance(periodic, bool):
        periodic = (periodic, periodic, periodic)
    coords = np.empty((3, idx.shape[0]), float)
    for a in range(3):
        coords[a] = np.mod(idx[:, a], grid.shape[a]) if periodic[a] else idx[:, a]
    mode = "grid-wrap" if any(periodic) else "nearest"
    vals = map_coordinates(grid, coords, order=order, mode=mode)
    return vals.reshape(lead)


def radial_from_sdf(sdf, voxel_size):
    """Local radial (membrane-normal) unit field from a signed distance grid.

    ``n = grad(sdf) / |grad(sdf)|`` — robust for undulating / asymmetric sheaths
    where an analytic cylinder normal is unavailable.  ``sdf`` should increase
    outward across the membrane (sign convention only affects ``n``'s sign, which
    is immaterial to ``n n^T``).
    """
    vs = _as_voxel_size(np.asarray(voxel_size, float), 3)
    g = np.gradient(np.asarray(sdf, float), *vs, edge_order=2)
    return _unit(np.stack(g, axis=-1), axis=-1)


# ---------------------------------------------------------------------------
# Mesh -> field-grid: voxelise a myelin sheath (inner/outer surface meshes) and
# build its geometry-only susceptibility field basis. This is the producer side of
# the static field-grid replay channel (dmipy_sim.bank): store the basis once, then
# replay any (B0, direction, chi) by contracting + sampling along the walk.
# ---------------------------------------------------------------------------
def mesh_inside(V, F, pts, *, clip_axis=None, chunk=2_000_000):
    """Boolean "inside this surface" for arbitrary points, by **global nearest-triangle sidedness**.

    ``sign((p - centroid_nearest) . outward_normal_nearest) < 0`` — the same sidedness test the walk's
    reflection uses, but with an EXACT global nearest neighbour (scipy KD-tree) instead of a local
    cell-neighbourhood gather. That matters for a thin structure inside a large box: a gather-based
    test has no triangle in its neighbourhood for most of the box and silently returns an arbitrary
    side (which would seed walkers in free space and corrupt a voxel mask).

    ``clip_axis`` (e.g. 2) additionally requires the point to lie within the mesh's extent along that
    axis — needed for a surface with OPEN ends (a cropped axon tube), where points beyond the rim have
    no meaningful inside/outside. Degenerate (zero-area) faces are dropped.

    Accurate to about one triangle size at the boundary (nearest-*centroid* rather than nearest-point);
    for the axon meshes here the triangles (~0.15 um) are far finer than the field grid, and the
    resulting volumes match the closed-surface mesh volumes to within a few percent.

    NEAR-FIELD ONLY, because the triangle is selected by nearest CENTROID. Sidedness against the TRUE
    nearest triangle is correct at any distance, but the nearest centroid need not belong to the nearest
    triangle, and the further away the point is the more triangles sit at comparable centroid distance,
    so the wrong one is picked and its normal reports the wrong side. Measured on the Winther axon06
    inner surface over the full padded box: 13.6% of accepted points are false, at a median 10.2 um from
    a wall bounding a lumen of radius <= 1.24 um. Substituting a proper closest-point query (trimesh's
    BVH) resolves all 16 disputed points in a 6k sample, confirming the cause is triangle SELECTION and
    not the sidedness principle; taking more k-nearest centroids does not fix it (k=8 changed nothing).

    Close to the surface the selection is reliable and so is the result (99.8% agreement with ray casting,
    no false-outside), which is why this remains the right per-sample test for classifying field-grid
    voxels — they all sit on or near the sheath. It is NOT a containment test over a large box: use
    :func:`mesh_contains`, which calls this as a candidate filter and then verifies each candidate.
    """
    from scipy.spatial import cKDTree
    V = np.asarray(V, float); F = np.asarray(F, np.int64)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    good = ln > 0                                            # drop degenerate faces
    tri = tri[good]; n = n[good] / ln[good][:, None]
    cent = tri.mean(1)
    tree = cKDTree(cent)
    pts = np.asarray(pts, float)
    out = np.zeros(len(pts), bool)
    for i in range(0, len(pts), chunk):
        p = pts[i:i + chunk]
        _, idx = tree.query(p, workers=-1)
        out[i:i + chunk] = np.einsum("ij,ij->i", p - cent[idx], n[idx]) < 0
    if clip_axis is not None:
        a = int(clip_axis)
        out &= (pts[:, a] > V[:, a].min()) & (pts[:, a] < V[:, a].max())
    return out


# Separating "a defect" from "an open surface" has to scale with the mesh, because defects accumulate with
# component count while openness does not. Measured: a 366-strand axon bundle carries 22 boundary edges of
# 2.08M (1.1e-05) -- cleaner per strand than a single Winther axon that is accepted -- while half a tube,
# genuinely open, carries 64 of 192 (3.3e-01). An absolute cap cannot separate those as meshes grow.
#
# The threshold is set by what it must REJECT, with margin: an open surface exposes a rim, which is a
# fraction of order 1e-1 of its edges, so 1e-2 rejects that by ~30x while accepting defect rates several
# orders of magnitude below it. The absolute floor keeps small meshes (a one-triangle hole in a 144-edge
# cylinder) on the defect side, where a fraction alone would be too strict.
_MAX_BOUNDARY_EDGE_FRACTION = 1e-2
_MIN_REPAIRABLE_BOUNDARY_EDGES = 16


def _warn_if_bruteforce_ray_engine(mesh):
    """trimesh chooses its ray backend at import and says nothing either way.

    With the native ``embreex`` it uses a BVH; without it, a pure-NumPy engine that tests every
    ray against every triangle. The two differ by ~3 orders of magnitude, and the fallback is
    silent -- which is how a 650 ms/point containment test hid inside a walk for a day. It is not
    always fixable by installing the extension: ``embreex`` publishes no aarch64 wheels, so on ARM
    the slow engine is permanent. Say so, once, rather than let the caller discover it as an OOM.
    """
    if type(mesh.ray).__module__.endswith("ray_triangle"):
        warnings.warn(
            "trimesh has no native ray backend (embreex/pyembree), so `contains` is running its "
            "pure-NumPy engine: every ray is tested against EVERY triangle, costing ~O(points x "
            "triangles) in both time and memory (measured on a 1.6M-triangle bundle: ~650 ms and "
            "~84 MB per point). Prefer `mesh_contains(..., method='grid')`, which is exact and "
            "bins triangles by their xy footprint instead.", RuntimeWarning, stacklevel=3)


def mesh_contains(V, F, pts, *, method="grid", prefilter=False, chunk=2_000_000):
    """Exact "inside this CLOSED surface" for arbitrary points, by ray-crossing parity.

    The containment test to use when the points can be anywhere in a large box — seeding a compartment
    and measuring its volume fraction — where :func:`mesh_inside` is unreliable in the far field (see its
    docstring). Parity is a global test and has no such failure mode, but it costs far more per point,
    so this runs in two stages: :func:`mesh_inside` proposes candidates, and only those are ray cast.
    The prefilter is OFF by default, because that soundness argument does not hold in general. It rests on
    ``mesh_inside`` producing no false-OUTSIDE -- measured on single axons (0 of 51-142 interior points missed,
    at 3k/4k/6k sample sizes) and FALSE on a dense multi-body bundle: on the 366-fibre CACTUS outer surface it
    misses 1.67% of genuine interior points (33 of 1977). Because the proposal gate sits UPSTREAM of the ray
    cast, every one of those is inherited -- the cascade returned exactly the same 33 false-OUTSIDEs as the
    fast test alone, so the "exact" stage never saw them. The consequence is not subtle: seeding a compartment
    on that answer put 3.8% of a nominally extra-axonal pool inside a fibre, which after the fact is
    indistinguishable from walkers leaking through an impermeable wall.

    ``prefilter=True`` restores the cascade for the case it was written for -- a thin structure in a big box,
    where it is a large speed-up and its assumption holds. Spot-check it against a rescaled-parity reference
    before trusting it on a new substrate.

    Requires a closed surface: with open ends a ray can exit through the rim and parity is meaningless.
    A mesh whose boundary edges are a negligible FRACTION of its edges (``_MAX_BOUNDARY_EDGE_FRACTION``,
    with a small absolute floor for tiny meshes) is treated as defective rather than open: ``trimesh.repair.fill_holes`` is tried, and if the defect is not a fillable hole (one axon of
    the 29-axon Winther set has a two-edge slit) it proceeds with a ``RuntimeWarning``, since an opening
    that small perturbs parity only for rays threading it. More than that raises: a genuinely open surface
    has no inside, and silently returning the parity of a leaky mesh would be worse than failing. For a
    deliberately open surface use ``mesh_inside(..., clip_axis=...)`` and accept its near-field contract.

    Parity comes from :mod:`trimesh` (imported lazily, as elsewhere in the package) rather than being
    reimplemented here.
    """
    if method not in ("grid", "trimesh"):
        raise ValueError(f"method must be 'grid' or 'trimesh', got {method!r}")
    pts = np.asarray(pts, float)
    V = np.asarray(V, float); F = np.asarray(F, np.int64)
    import trimesh
    # Rescale so triangle edges are O(1) before handing the mesh to trimesh. Its geometry predicates
    # unitize against an absolute tolerance, so at SI scale (edges ~1e-7 m) face normals collapse to
    # zero and `contains` reports almost everything outside -- silently, and only in the direction that
    # looks like a clean, restrictive answer. mesh_inside is scale-free (it normalises its own normals),
    # so only this stage needs it. The factor cancels: points are scaled identically.
    s = 1.0 / max(float(np.median(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1))), 1e-300)
    m = trimesh.Trimesh(vertices=V * s, faces=F, process=False)
    n_open = len(trimesh.grouping.group_rows(m.edges_sorted, require_count=1))
    if n_open:
        # A handful of boundary edges is a defect (a missing triangle), not an open surface: repair and
        # continue. The cap is what separates the two -- a torn or genuinely open surface must still fail,
        # because fill_holes would happily span a wide rim and hand back confident nonsense.
        allowed = max(_MIN_REPAIRABLE_BOUNDARY_EDGES,
                      int(_MAX_BOUNDARY_EDGE_FRACTION * len(m.edges_sorted)))
        if n_open <= allowed:
            rep = m.copy()
            trimesh.repair.fill_holes(rep)
            if not len(trimesh.grouping.group_rows(rep.edges_sorted, require_count=1)):
                m = rep
                n_open = 0
            else:
                # Not every tiny defect is a fillable hole: one axon of the 29-axon Winther set has two
                # boundary edges meeting at a single vertex, which is a slit rather than a hole, and no
                # repair closes it without making that edge non-manifold. An opening this small is
                # negligible for parity -- only a ray threading the slit itself is affected -- so proceed,
                # but say so, and check the result against the mesh volume if the fraction matters.
                import warnings as _w
                _w.warn(f"mesh_contains: {n_open} boundary edges remain after trimesh.repair.fill_holes "
                        f"(a slit, not a hole). Proceeding: an opening this small changes parity only for "
                        f"rays passing through it. Cross-check the accepted fraction against the mesh "
                        f"volume if it matters.", RuntimeWarning, stacklevel=2)
                n_open = 0
        if n_open:
            raise ValueError(
                f"mesh_contains requires a closed surface; this one has {n_open} boundary edges "
                f"({n_open/max(len(m.edges_sorted),1):.2e} of all edges), too many to be a defect. Ray parity is undefined when a ray can leave through an open rim. Use "
                f"mesh_inside(..., clip_axis=...) for a deliberately open surface, noting it is a "
                f"near-field test only.")
    if method == "grid":
        # `m` is the surface the gate accepted (holes filled), and scaled by `s`; the grid path is
        # scale-free so either is fine, but it must see the REPAIRED faces, not the caller's.
        return mesh_contains_fast(np.asarray(m.vertices, float) / s,
                                  np.asarray(m.faces, np.int64), pts)

    _warn_if_bruteforce_ray_engine(m)
    out = np.zeros(len(pts), bool)
    cand = mesh_inside(V, F, pts) if prefilter else np.ones(len(pts), bool)
    if cand.any():
        sub = pts[cand]
        got = np.zeros(len(sub), bool)
        try:
            for i in range(0, len(sub), chunk):
                got[i:i + chunk] = m.contains(sub[i:i + chunk] * s)
        except ModuleNotFoundError as exc:      # trimesh's ray engine is an EXTRA, not a hard dependency
            raise ModuleNotFoundError(
                f"mesh_contains needs trimesh's ray-casting extras ({exc.name} is missing). A bare "
                f"'pip install trimesh' imports fine and only fails here, inside the containment call. "
                f"Install pip install 'dmipy-sim[mesh]' (or 'trimesh[easy]')."
            ) from exc
        out[cand] = got
    return out


def mesh_field_basis(inner, outer, box_min, box_max, *, res=0.1e-6, include_aniso=True,
                     mask_supersample=2, kspace_lowpass=0.5, clip_axis=2):
    """Geometry-only myelin susceptibility field basis on a voxel grid, from inner (axonal) and
    outer (myelin) surface meshes ``(V, F)`` (metres).

    Voxelises the myelin shell (inside outer, outside inner) via :meth:`Mesh.classify_position`
    and derives the per-voxel radial director from the signed distance to the inner surface, then
    calls :func:`field_basis`. ``mask_supersample`` (default 2) gives a partial-volume occupancy in
    ``[0,1]`` at the sheath boundary — essential because the dipole kernel ``k̂ᵢk̂ⱼ`` does not decay
    with ``|k|`` so a hard binary edge rings into the interior. Returns ``(basis, origin, voxel_size)``
    with ``origin`` = box corner (the :func:`sample_grid` convention)."""
    from scipy import ndimage

    box_min = np.asarray(box_min, float); box_max = np.asarray(box_max, float)
    side = box_max - box_min
    N = np.maximum(1, np.round(side / res).astype(int))
    vs = side / N
    axes = [box_min[a] + (np.arange(N[a]) + 0.5) * vs[a] for a in range(3)]
    XX, YY, ZZ = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

    in_in = mesh_inside(inner[0], inner[1], pts, clip_axis=clip_axis).reshape(N)   # for the SDF/director
    in_out = mesh_inside(outer[0], outer[1], pts, clip_axis=clip_axis).reshape(N)
    binary = (in_out & ~in_in)
    SS = max(1, int(mask_supersample))
    if SS == 1:
        myelin_mask = binary.astype(float)
    else:
        # Partial-volume (anti-aliased) occupancy. Essential: with a hard binary source the
        # non-decaying dipole kernel rings into the lumen (measured against the analytic hollow
        # cylinder: 2.6% of chi*B0 where the exact answer is 0; PV-4 reduces that to ~0.1% while
        # keeping the annulus amplitude at 1.00x analytic). Only voxels ON the boundary can have
        # fractional occupancy, so supersample just those -- exact, and ~100x cheaper than the
        # whole grid.
        myelin_mask = binary.astype(float)
        edge = ndimage.binary_dilation(binary) ^ ndimage.binary_erosion(binary)
        ei = np.flatnonzero(edge.ravel())
        if ei.size:
            sub = (np.arange(SS) + 0.5) / SS - 0.5                  # sub-voxel offsets in [-0.5,0.5)
            ox, oy, oz = np.meshgrid(sub * vs[0], sub * vs[1], sub * vs[2], indexing="ij")
            off = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=1)      # (SS^3, 3)
            base = pts[ei]                                                    # (n_edge, 3) voxel centres
            q = (base[:, None, :] + off[None, :, :]).reshape(-1, 3)
            occ = (mesh_inside(outer[0], outer[1], q, clip_axis=clip_axis)
                   & ~mesh_inside(inner[0], inner[1], q, clip_axis=clip_axis))
            myelin_mask.ravel()[ei] = occ.reshape(len(ei), -1).mean(axis=1)
    sdf = (ndimage.distance_transform_edt(~in_in, sampling=tuple(vs))
           - ndimage.distance_transform_edt(in_in, sampling=tuple(vs)))
    radial_dir = radial_from_sdf(sdf, vs)
    basis = field_basis(myelin_mask, radial_dir, vs, include_aniso=include_aniso,
                        kspace_lowpass=kspace_lowpass)
    return basis, box_min, vs


# ─────────────────────────────────────────────────────────────────────────────
def _xy_bins(tri, n_bins):
    """CSR (offsets, tri_ids, lo, inv) binning triangles by their xy footprint.

    A vertical (+z) ray from ``p`` can only hit a triangle whose xy AABB contains ``p.xy``, so
    binning in xy alone is both complete and conservative -- no intersection can be missed, which
    is what makes the acceleration exact rather than heuristic.
    """
    lo = tri[:, :, :2].min(axis=(0, 1))
    hi = tri[:, :, :2].max(axis=(0, 1))
    span = np.maximum(hi - lo, 1e-30)
    inv = n_bins / (span * (1.0 + 1e-9))
    t_lo = np.floor((tri[:, :, :2].min(axis=1) - lo) * inv).astype(np.int64)
    t_hi = np.floor((tri[:, :, :2].max(axis=1) - lo) * inv).astype(np.int64)
    np.clip(t_lo, 0, n_bins - 1, out=t_lo)
    np.clip(t_hi, 0, n_bins - 1, out=t_hi)

    wi = t_hi[:, 0] - t_lo[:, 0] + 1
    wj = t_hi[:, 1] - t_lo[:, 1] + 1
    counts = wi * wj
    total = int(counts.sum())
    tri_ids = np.repeat(np.arange(len(tri), dtype=np.int64), counts)
    start = np.cumsum(counts) - counts
    off = np.arange(total, dtype=np.int64) - np.repeat(start, counts)
    wj_r = np.repeat(wj, counts)
    bi = np.repeat(t_lo[:, 0], counts) + off // wj_r
    bj = np.repeat(t_lo[:, 1], counts) + off % wj_r
    key = bi * n_bins + bj

    order = np.argsort(key, kind="stable")
    key = key[order]; tri_ids = tri_ids[order]
    offsets = np.searchsorted(key, np.arange(n_bins * n_bins + 1))
    return offsets, tri_ids, lo, inv


def _parity_vertical(tri, pts, offsets, tri_ids, lo, inv, n_bins, tol):
    """Crossing parity of a +z ray, testing only the triangles in each point's xy bin.

    Returns ``(inside, ambiguous)``. A point is ambiguous when the ray passes within ``tol`` of a
    triangle edge in projection: the shared edge of two coplanar-adjacent faces is then counted
    twice or not at all, and parity is unreliable. Those points are re-cast from a jittered origin
    rather than silently trusted -- the failure is a flipped bit, not a small error.
    """
    n = len(pts)
    inside = np.zeros(n, bool)
    ambig = np.zeros(n, bool)
    b = np.floor((pts[:, :2] - lo) * inv).astype(np.int64)
    np.clip(b, 0, n_bins - 1, out=b)
    key = b[:, 0] * n_bins + b[:, 1]

    order = np.argsort(key, kind="stable")
    ks = key[order]
    edges = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    for e0, e1 in zip(edges, np.r_[edges[1:], len(ks)]):
        k = ks[e0]
        s, t = offsets[k], offsets[k + 1]
        if s == t:
            continue                                   # no triangle overhead: outside
        idx = order[e0:e1]
        T = tri[tri_ids[s:t]]                          # (m, 3, 3)
        P = pts[idx]                                   # (q, 3)
        ax, ay, az = T[:, 0, 0], T[:, 0, 1], T[:, 0, 2]
        bx, by, bz = T[:, 1, 0], T[:, 1, 1], T[:, 1, 2]
        cx, cy, cz = T[:, 2, 0], T[:, 2, 1], T[:, 2, 2]
        d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)      # 2 x signed xy area
        # relative to the typical face: an absolute floor cannot distinguish a vertical
        # face (never crossed by a vertical ray) from a merely small one.
        ok = np.abs(d) > 1e-12 * np.median(np.abs(d)) if len(d) else np.zeros(0, bool)
        dd = np.where(ok, d, 1.0)
        px = P[:, 0][:, None]; py = P[:, 1][:, None]; pz = P[:, 2][:, None]
        l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / dd
        l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / dd
        l3 = 1.0 - l1 - l2
        hit = (l1 >= 0.0) & (l2 >= 0.0) & (l3 >= 0.0) & ok[None, :]
        z = l1 * az + l2 * bz + l3 * cz
        cross = hit & (z > pz)
        inside[idx] = (cross.sum(axis=1) % 2) == 1
        near = (np.minimum(np.minimum(np.abs(l1), np.abs(l2)), np.abs(l3)) < tol) & ok[None, :]
        ambig[idx] = (near & (z > pz - tol)).any(axis=1)
    return inside, ambig


def mesh_contains_fast(V, F, pts, *, n_bins=256, tol=1e-9, max_retries=6, seed=0):
    """Exact "inside this CLOSED surface", accelerated by an xy bin index instead of a ray engine.

    Same ray-parity semantics as :func:`mesh_contains`, and the same answers -- what changes is
    only WHICH triangles are tested. trimesh picks its ray backend at import: the native
    ``embreex`` when present, else a pure-NumPy engine that tests every ray against every
    triangle, with no warning either way. On this 1.57M-triangle bundle that engine measured
    ~650 ms and ~84 MB PER POINT, so seeding 6,000 walkers would need ~4 h and ~2 TB. And it is
    not installable everywhere -- ``embreex`` publishes no aarch64 wheels -- so the cliff is
    silent, platform-dependent, and permanent on ARM.

    A +z ray from ``p`` can only meet a triangle whose xy footprint contains ``p.xy``, so binning
    triangles by that footprint is complete: no intersection is missed, and the acceleration is
    exact rather than approximate. Each point then tests the ~100 triangles over its own bin
    instead of all 1.57M.

    Points whose ray grazes an edge in projection are re-cast from a jittered origin. That case is
    not a rounding error -- a shared edge counted twice or zero times flips the parity bit outright
    -- so it is detected and retried rather than trusted.
    """
    V = np.asarray(V, float); F = np.asarray(F, np.int64)
    pts = np.asarray(pts, float)
    if len(pts) == 0:
        return np.zeros(0, bool)
    # Work in units of the median edge, exactly as `mesh_contains` does before handing trimesh a
    # mesh. Substrate coordinates are in METRES, so a triangle's doubled xy area is ~1e-14 there;
    # the barycentric divide then amplifies rounding, and an absolute degeneracy threshold cannot
    # tell a vertical face from a small one. Rescaling makes both O(1) and the predicates honest.
    scale = 1.0 / max(float(np.median(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1))), 1e-300)
    tri = V[F] * scale
    pts = pts * scale
    offsets, tri_ids, lo, inv = _xy_bins(tri, n_bins)

    inside, ambig = _parity_vertical(tri, pts, offsets, tri_ids, lo, inv, n_bins, tol)
    if ambig.any():
        rng = np.random.default_rng(seed)
        jit = float(np.median(np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)))
        for _ in range(max_retries):
            j = np.flatnonzero(ambig)
            if j.size == 0:
                break
            q = pts[j].copy()
            q[:, :2] += rng.normal(0.0, 1e-3 * jit, (len(j), 2))
            ins_j, amb_j = _parity_vertical(tri, q, offsets, tri_ids, lo, inv, n_bins, tol)
            inside[j] = ins_j
            ambig[j] = amb_j
        if ambig.any():
            warnings.warn(
                f"{int(ambig.sum())} of {len(pts)} points still graze a triangle edge after "
                f"{max_retries} jittered re-casts; their inside/outside is unreliable. That "
                f"usually means degenerate or duplicated faces rather than bad luck.",
                RuntimeWarning, stacklevel=2)
    return inside
