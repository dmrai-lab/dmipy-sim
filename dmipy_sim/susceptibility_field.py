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


# A missing triangle leaves ~3 boundary edges; a torn or open surface leaves many. Above this, refuse to
# repair rather than span a rim (see mesh_contains).
_MAX_REPAIRABLE_BOUNDARY_EDGES = 16


def mesh_contains(V, F, pts, *, prefilter=True, chunk=2_000_000):
    """Exact "inside this CLOSED surface" for arbitrary points, by ray-crossing parity.

    The containment test to use when the points can be anywhere in a large box — seeding a compartment
    and measuring its volume fraction — where :func:`mesh_inside` is unreliable in the far field (see its
    docstring). Parity is a global test and has no such failure mode, but it costs far more per point,
    so this runs in two stages: :func:`mesh_inside` proposes candidates, and only those are ray cast.
    The cascade is sound because ``mesh_inside`` produces no false-OUTSIDE (measured on the axon meshes:
    0 of 51-142 genuine interior points missed, at 3k/4k/6k sample sizes), so it never discards a point
    that is really inside; its errors are all false-inside, which the exact stage then removes. Set
    ``prefilter=False`` to ray cast every point and skip that assumption.

    Requires a closed surface: with open ends a ray can exit through the rim and parity is meaningless.
    A mesh with at most ``_MAX_REPAIRABLE_BOUNDARY_EDGES`` boundary edges is treated as defective rather
    than open: ``trimesh.repair.fill_holes`` is tried, and if the defect is not a fillable hole (one axon of
    the 29-axon Winther set has a two-edge slit) it proceeds with a ``RuntimeWarning``, since an opening
    that small perturbs parity only for rays threading it. More than that raises: a genuinely open surface
    has no inside, and silently returning the parity of a leaky mesh would be worse than failing. For a
    deliberately open surface use ``mesh_inside(..., clip_axis=...)`` and accept its near-field contract.

    Parity comes from :mod:`trimesh` (imported lazily, as elsewhere in the package) rather than being
    reimplemented here.
    """
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
        if n_open <= _MAX_REPAIRABLE_BOUNDARY_EDGES:
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
                f"mesh_contains requires a closed surface; this one has {n_open} boundary edges, too many "
                f"to be a defect. Ray parity is undefined when a ray can leave through an open rim. Use "
                f"mesh_inside(..., clip_axis=...) for a deliberately open surface, noting it is a "
                f"near-field test only.")
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
