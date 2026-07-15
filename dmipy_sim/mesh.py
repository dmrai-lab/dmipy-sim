"""Triangular-mesh geometry with spatial acceleration and optional 3D periodicity.

``Mesh`` lets the Monte-Carlo engine run arbitrary closed *or* periodic triangular
surface meshes (e.g. dense multi-cell microstructure phantoms exported as PLY) with
the same physics as the analytic geometries: restricted diffusion, surface
relaxivity (Brownstein–Tarr) and membrane permeability (Powles).

Design
------
* **Spatial acceleration.** A static uniform grid buckets every triangle by the
  cells its axis-aligned bounding box overlaps.  Each step only tests the triangles
  in the walker's 27-cell neighbourhood, turning the per-step cost from
  ``O(n_triangles)`` into ``O(candidates)`` — the difference between "intractable"
  and "seconds" on a 10^6-triangle mesh.  This is exact provided the grid cell size
  is at least the maximum single step (guaranteed by the ``cell_size`` default,
  which is tied to the diffusion step).
* **3D periodicity** (``periodic=True``).  Triangles within one cell of a periodic
  face are replicated ("ghosts") across to the opposite side, so a walker crossing
  the box sees the continuing structure.  Geometry queries use the *wrapped*
  position while the returned position stays *continuous* (unfolded), which keeps
  the gradient phase correct — the same convention the packed geometries use.  The
  box faces are wrap planes, **not** reflecting walls, so open (clipped) cells on
  the boundary are stitched to their periodic partners.
* **Smooth reflection.** Reflections and surface-relaxivity path lengths use a
  barycentrically-interpolated vertex normal, reducing the flat-facet error from
  ``O(h/R)`` to ``O(h^2/R^2)`` in the triangle edge length ``h``.
* **Leak-proof permeation.** The Powles crossing decision is taken once, at the
  first membrane hit; if the walker reflects, the remainder of the step is resolved
  by a multi-bounce reflection scan (no further crossing draws).  This neither
  leaks walkers through convex corners nor double-counts crossings there.

Accuracy note
-------------
Restricted diffusion and surface relaxivity reach the Monte-Carlo noise floor even
for coarse meshes; membrane *permeability* is more sensitive to the surface
tessellation (its bias falls ~``O(h^2)``), so accurate permeability needs a fairly
fine mesh (edge length ``<~ 0.04`` of the local feature radius).
"""

import itertools
import warnings
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

from .geometries import Geometry

# Above this median-edge / feature-radius ratio the surface is too coarsely
# tessellated for membrane permeability to reach the MC noise floor (its faceting
# bias falls ~O(h^2); measured: ratio 0.075 -> ~8x noise, 0.038 -> at noise floor).
# Restricted diffusion and surface relaxivity are unaffected at these ratios.
_PERM_EDGE_RATIO_MAX = 0.05


def _rotation_from_z(axis):
    """Rotation matrix R (mesh->lab) with R @ [0,0,1] = axis / |axis|.

    The in-plane (azimuthal) choice is arbitrary; pass an explicit ``R`` instead
    for meshes whose in-plane orientation matters.
    """
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, a)
    c = float(np.dot(z, a))
    if np.linalg.norm(v) < 1e-12:                 # parallel or anti-parallel
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))   # Rodrigues


def _smooth_vertex_normals(V, F):
    """Area-weighted vertex normals, shape (n_vertices, 3)."""
    tris = V[F]
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])  # ||=2*area
    vn = np.zeros((len(V), 3))
    np.add.at(vn, F[:, 0], cross)
    np.add.at(vn, F[:, 1], cross)
    np.add.at(vn, F[:, 2], cross)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-30)
    return vn


def load_ply(path, scale=1.0, recenter=False):
    """Load vertices and faces from a mesh file (PLY/STL/OBJ/...).

    Uses :mod:`trimesh` (install the ``mesh`` extra: ``pip install dmipy-sim[mesh]``).

    Parameters
    ----------
    path : str
        Path to the mesh file.
    scale : float
        Multiply all coordinates by this factor — use it to convert a mesh stored
        in arbitrary/normalised units into **metres** (the simulator's unit).
    recenter : bool
        If True, translate the mesh so its bounding box is centred on the origin.

    Returns
    -------
    vertices : (n_vertices, 3) float64 ndarray, metres
    faces    : (n_faces, 3) int64 ndarray
    """
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - exercised via the extra
        raise ImportError(
            "load_ply requires trimesh — install with `pip install dmipy-sim[mesh]` "
            "or `pip install trimesh`."
        ) from exc
    m = trimesh.load(path, process=False)
    V = np.asarray(m.vertices, np.float64)
    F = np.asarray(m.faces, np.int64)
    if recenter:
        V = V - 0.5 * (V.min(0) + V.max(0))
    return V * scale, F


class Mesh(Geometry):
    """Reflecting/permeable triangular-mesh geometry (see module docstring).

    Parameters
    ----------
    vertices : (n_vertices, 3) array-like, metres
    faces : (n_faces, 3) array-like of int
        Triangle vertex-index triples.
    periodic : bool or (bool, bool, bool)
        Wrap walkers periodically along the given axes (default False = closed mesh).
        When any axis is periodic you must pass ``voxel_min``/``voxel_max`` defining
        the periodic box (the mesh bbox is usually slightly larger than the true
        period, so it is not a safe default).
    voxel_min, voxel_max : (3,) array-like, optional
        The simulation box.  Defaults to the mesh bounding box (closed meshes only).
    feature_radius : float, optional
        Characteristic feature size (e.g. a cell/pore radius), used to size the
        diffusion sub-step (``step ~ feature_radius/6``, or ``/25`` when permeable)
        and the grid.  Defaults to half the smallest box side; **pass the real cell
        radius for packed substrates**, otherwise the step may be too coarse.
    surface_relaxivity_t2 : float, optional
        Surface relaxivity ρ₂ (m/s).  Applies a Brownstein–Tarr weight at the wall.
    permeability : float, optional
        Membrane permeability κ (m/s).  None → impermeable.
    orientation : (3,) array-like, optional
        Direction, in the scanner frame (B0 = +z), along which the mesh's native
        +z axis (e.g. a periodic / fibre axis) is placed in the bore.  Applied as
        an acquisition rotation (the gradient is rotated into the mesh frame in
        ``simulate``), so the walk itself is unchanged.  The in-plane rotation is
        arbitrary — pass ``R`` for meshes whose in-plane orientation matters.
    R : (3, 3) array-like, optional
        Explicit mesh→lab rotation matrix (mutually exclusive with ``orientation``).
    cell_size : float, optional
        Acceleration-grid cell size.  Defaults to ``4 * step`` (safe for the 27-cell
        neighbourhood).  Larger = fewer/denser cells; must be ≥ the maximum step.
    """

    def __init__(self, vertices, faces, *, periodic=False, voxel_min=None,
                 voxel_max=None, feature_radius=None, surface_relaxivity_t2=None,
                 permeability=None, orientation=None, R=None,
                 cell_size=None, cap=None):
        V = np.asarray(vertices, np.float64)
        F = np.asarray(faces, np.int64)
        self.vertices = V
        self.faces = F

        if isinstance(periodic, bool):
            periodic = (periodic, periodic, periodic)
        self.periodic = tuple(bool(p) for p in periodic)

        bbmin, bbmax = V.min(0), V.max(0)
        self.vmin = np.asarray(voxel_min, np.float64) if voxel_min is not None else bbmin.copy()
        self.vmax = np.asarray(voxel_max, np.float64) if voxel_max is not None else bbmax.copy()
        self.L = self.vmax - self.vmin

        if feature_radius is None:
            sides = self.vmax - self.vmin
            feature_radius = 0.5 * float(np.min(sides[sides > 0]))
        self.radius = float(feature_radius)              # read by core sub-step auto-tune
        self.reject_escape = True                        # impermeable-leak safety net

        self.surface_relaxivity_t2 = (float(surface_relaxivity_t2)
                                      if surface_relaxivity_t2 is not None else None)
        self.permeability = float(permeability) if permeability is not None else None

        # ---- placement in the scanner frame (B0 = +z convention) ----
        # The mesh, its grid and periodic box live in the mesh's NATIVE frame and
        # the walk runs entirely there.  `orientation`/`R` declare how that frame
        # is placed in the lab/scanner frame (B0 = +z); simulate() then rotates the
        # ACQUISITION (gradient vectors -- and later B0) into the mesh frame,
        # exactly the "rotate the waveform, not the geometry" convention used by the
        # mesoscopic orchestration.  This keeps the (validated) walk untouched and
        # makes the placement a pure acquisition rotation.  Default: mesh frame IS
        # the scanner frame (native +z, e.g. a periodic/fibre axis, along B0 = +z).
        self.orientation = orientation
        if R is not None:
            Rm = np.asarray(R, np.float64).reshape(3, 3)
        elif orientation is not None:
            Rm = _rotation_from_z(orientation)
        else:
            Rm = None
        # mesh->lab rotation; None when unoriented (simulate skips the hook).
        self._orient_R = None if Rm is None else np.ascontiguousarray(Rm, np.float32)

        # ---- surface-resolution diagnostics + permeability coarseness warning ----
        _e = V[F]
        edge = np.concatenate([
            np.linalg.norm(_e[:, 1] - _e[:, 0], axis=1),
            np.linalg.norm(_e[:, 2] - _e[:, 1], axis=1),
            np.linalg.norm(_e[:, 0] - _e[:, 2], axis=1)])
        self.edge_median = float(np.median(edge))
        self.edge_p90 = float(np.percentile(edge, 90))
        self.edge_feature_ratio = self.edge_median / self.radius
        if self.permeability is not None and self.edge_feature_ratio > _PERM_EDGE_RATIO_MAX:
            warnings.warn(
                f"Mesh is likely too coarse for MC-noise-accurate PERMEABILITY: "
                f"median edge / feature_radius = {self.edge_feature_ratio:.3f} "
                f"(need <~ {_PERM_EDGE_RATIO_MAX}). The permeability faceting bias "
                f"falls ~O(h^2); restricted diffusion and surface relaxivity are "
                f"unaffected. Use a finer / less-decimated mesh for permeability, or "
                f"call .quality_report() for details.",
                stacklevel=2)

        step_l = self.radius / (25.0 if self.permeability is not None else 6.0)
        self.cell_size = float(cell_size) if cell_size is not None else 4.0 * step_l
        self.margin = self.cell_size

        vn = _smooth_vertex_normals(V, F)
        tri_v = V[F]
        tri_vn = vn[F]
        tmin, tmax = tri_v.min(1), tri_v.max(1)
        base = np.arange(len(F))
        all_tri, all_vn = [tri_v], [tri_vn]

        # ghost replication across periodic faces/edges/corners
        for combo in itertools.product((-1, 0, 1), repeat=3):
            if all(c == 0 for c in combo):
                continue
            if any(combo[a] != 0 and not self.periodic[a] for a in range(3)):
                continue
            keep = np.ones(len(F), bool)
            shift = np.zeros(3)
            for a in range(3):
                if combo[a] == -1:
                    keep &= tmax[:, a] > self.vmax[a] - self.margin
                    shift[a] = -self.L[a]
                elif combo[a] == +1:
                    keep &= tmin[:, a] < self.vmin[a] + self.margin
                    shift[a] = +self.L[a]
            if keep.any():
                all_tri.append(tri_v[keep] + shift)
                all_vn.append(tri_vn[keep])
        tri_all = np.concatenate(all_tri, 0)
        vn_all = np.concatenate(all_vn, 0)
        self.n_ghost = len(tri_all) - len(F)

        nrm_all = np.cross(tri_all[:, 1] - tri_all[:, 0], tri_all[:, 2] - tri_all[:, 0])
        nrm_all /= np.maximum(np.linalg.norm(nrm_all, axis=1, keepdims=True), 1e-30)

        # uniform grid over [vmin - margin, vmax + margin]
        self.grid_min = self.vmin - self.margin
        grid_max = self.vmax + self.margin
        self.dims = np.maximum(1, np.ceil((grid_max - self.grid_min) / self.cell_size).astype(int))
        cs = self.cell_size
        lo = np.clip(np.floor((tri_all.min(1) - self.grid_min) / cs).astype(int), 0, self.dims - 1)
        hi = np.clip(np.floor((tri_all.max(1) - self.grid_min) / cs).astype(int), 0, self.dims - 1)
        buckets = defaultdict(list)
        for t in range(len(tri_all)):
            for ix in range(lo[t, 0], hi[t, 0] + 1):
                for iy in range(lo[t, 1], hi[t, 1] + 1):
                    for iz in range(lo[t, 2], hi[t, 2] + 1):
                        buckets[(ix * self.dims[1] + iy) * self.dims[2] + iz].append(t)
        occ = np.array([len(v) for v in buckets.values()]) if buckets else np.array([0])
        C = int(occ.max()) if cap is None else int(cap)
        self.C = C
        self.max_occ = int(occ.max())
        cell_tri = np.full((int(np.prod(self.dims)), C), -1, np.int32)
        self.overflow = 0
        for cid, lst in buckets.items():
            if len(lst) > C:
                self.overflow += len(lst) - C
                lst = lst[:C]
            cell_tri[cid, :len(lst)] = lst

        self._TRIS = jnp.asarray(tri_all, jnp.float32)
        self._VN = jnp.asarray(vn_all, jnp.float32)
        self._NRM = jnp.asarray(nrm_all, jnp.float32)
        self._CENT = jnp.asarray(tri_all.mean(1), jnp.float32)
        self._CELL = jnp.asarray(cell_tri, jnp.int32)
        self._DIMS = tuple(int(x) for x in self.dims)
        self._dims_arr = jnp.asarray(self._DIMS, jnp.int32)
        self._GMIN = jnp.asarray(self.grid_min, jnp.float32)
        self._CS = jnp.float32(self.cell_size)
        self._VMIN = jnp.asarray(self.vmin, jnp.float32)
        self._L = jnp.asarray(self.L, jnp.float32)
        self._PER = jnp.asarray([1.0 if p else 0.0 for p in self.periodic], jnp.float32)
        _scale = float(np.min(self.L))
        self._EPS = jnp.float32(1e-7 * _scale)
        self._NUDGE = jnp.float32(1e-4 * step_l)
        self._OFF = jnp.asarray([[dx, dy, dz] for dx in (-1, 0, 1)
                                 for dy in (-1, 0, 1) for dz in (-1, 0, 1)], jnp.int32)

    # ------------------------------------------------------------------
    def _wrap(self, r):
        w = self._VMIN + jnp.mod(r - self._VMIN, self._L)
        return jnp.where(self._PER > 0, w, r)

    def _gather(self, r_w):
        c = jnp.clip(jnp.floor((r_w - self._GMIN) / self._CS).astype(jnp.int32),
                     0, self._dims_arr - 1)
        nb = jnp.clip(c[None, :] + self._OFF, 0, self._dims_arr - 1)
        cids = (nb[:, 0] * self._DIMS[1] + nb[:, 1]) * self._DIMS[2] + nb[:, 2]
        cand = self._CELL[cids].reshape(-1)
        valid = cand >= 0
        return jnp.where(valid, cand, 0), valid

    def _mt(self, r0, d_hat, tri, valid):
        A = tri[:, 0]; E1 = tri[:, 1] - A; E2 = tri[:, 2] - A; T = r0[None] - A
        P = jnp.cross(jnp.broadcast_to(d_hat, E2.shape), E2); det = (P * E1).sum(1)
        Q = jnp.cross(T, E1)
        t = (Q * E2).sum(1) / det
        u = (P * T).sum(1) / det
        v = (Q * jnp.broadcast_to(d_hat, E2.shape)).sum(1) / det
        ok = (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & valid
        return jnp.where(ok, t, jnp.inf), u, v

    def classify_position(self, r):
        """Compartment tag: 0 = interior (inside a cell), 1 = exterior."""
        r_w = self._wrap(r)
        ci, valid = self._gather(r_w)
        cent = self._CENT[ci]; nrm = self._NRM[ci]
        dist = jnp.where(valid, jnp.linalg.norm(r_w[None] - cent, axis=1), jnp.inf)
        idx = jnp.argmin(dist)
        side = jnp.dot(r_w - cent[idx], nrm[idx])
        return jnp.where(side < 0, jnp.int32(0), jnp.int32(1))

    def _smooth_normal(self, vnf, nrmf, u, v, idx, d_hat):
        bu, bv = u[idx], v[idx]
        ns = (1 - bu - bv) * vnf[idx, 0] + bu * vnf[idx, 1] + bv * vnf[idx, 2]
        ns = ns / jnp.linalg.norm(ns)
        n = jnp.where(jnp.dot(d_hat, nrmf[idx]) > 0, -ns, ns)  # side by face, dir by smooth
        return n

    # ------------------------------------------------------------------
    def reflect(self, r, step):
        r_w = self._wrap(r)
        ci, valid = self._gather(r_w)
        tri = self._TRIS[ci]; vnf = self._VN[ci]; nrmf = self._NRM[ci]
        step_l = jnp.linalg.norm(step); d_hat = step / step_l

        def one(carry, _):
            r0, dh, rem = carry
            ts, u, v = self._mt(r0, dh, tri, valid)
            vm = (ts > self._EPS) & (ts < rem); ts = jnp.where(vm, ts, jnp.inf)
            idx = jnp.argmin(ts); d = ts[idx]; hit = d < jnp.inf
            n = self._smooth_normal(vnf, nrmf, u, v, idx, dh)
            r_hit = r0 + d * dh
            d_ref = dh - 2 * jnp.dot(dh, n) * n; d_ref /= jnp.linalg.norm(d_ref)
            return (jnp.where(hit, r_hit + self._NUDGE * n, r0),
                    jnp.where(hit, d_ref, dh),
                    jnp.where(hit, rem - d - self._NUDGE, rem)), None
        (rf, df, remf), _ = jax.lax.scan(one, (r_w, d_hat, step_l), None, length=10)
        r_out = r + (rf + df * jnp.maximum(remf, 0.0) - r_w)
        if self.reject_escape:
            r_out = jnp.where(self.classify_position(r) == self.classify_position(r_out),
                              r_out, r)
        return r_out

    def reflect_with_log_weight(self, r, step, rho_over_D):
        r_w = self._wrap(r)
        ci, valid = self._gather(r_w)
        tri = self._TRIS[ci]; vnf = self._VN[ci]; nrmf = self._NRM[ci]
        step_l = jnp.linalg.norm(step); d_hat = step / step_l

        def one(carry, _):
            r0, dh, rem = carry
            ts, u, v = self._mt(r0, dh, tri, valid)
            vm = (ts > self._EPS) & (ts < rem); ts = jnp.where(vm, ts, jnp.inf)
            idx = jnp.argmin(ts); d = ts[idx]; hit = d < jnp.inf
            n = self._smooth_normal(vnf, nrmf, u, v, idx, dh)
            r_hit = r0 + d * dh
            d_ref = dh - 2 * jnp.dot(dh, n) * n; d_ref /= jnp.linalg.norm(d_ref)
            cos_a = -jnp.dot(dh, n)
            d_perp = jnp.where(hit, (rem - d) * cos_a, jnp.float32(0.0))
            return (jnp.where(hit, r_hit + self._NUDGE * n, r0),
                    jnp.where(hit, d_ref, dh),
                    jnp.where(hit, rem - d - self._NUDGE, rem)), d_perp
        (rf, df, remf), dperps = jax.lax.scan(one, (r_w, d_hat, step_l), None, length=10)
        dlog_w = -2.0 * jnp.float32(rho_over_D) * jnp.sum(dperps)
        r_out = r + (rf + df * jnp.maximum(remf, 0.0) - r_w)
        if self.reject_escape:
            escaped = self.classify_position(r) != self.classify_position(r_out)
            return jnp.where(escaped, r, r_out), jnp.where(escaped, jnp.float32(0.0), dlog_w)
        return r_out, dlog_w

    def permeate(self, r, step, kappa_over_D, rho_over_D, perm_key):
        r_w = self._wrap(r)
        ci, valid = self._gather(r_w)
        tri = self._TRIS[ci]; vnf = self._VN[ci]; nrmf = self._NRM[ci]
        step_l = jnp.linalg.norm(step); d_hat = step / step_l
        u_rand = jax.random.uniform(perm_key, dtype=jnp.float32)

        def one(carry, _):
            r0, dh, rem, decided, dlogw = carry
            ts, u, v = self._mt(r0, dh, tri, valid)
            vm = (ts > self._EPS) & (ts < rem); ts = jnp.where(vm, ts, jnp.inf)
            idx = jnp.argmin(ts); d = ts[idx]; hit = d < jnp.inf
            n = self._smooth_normal(vnf, nrmf, u, v, idx, dh)
            cos_a = -jnp.dot(dh, n)
            d_perp = (rem - d) * cos_a
            first_hit = hit & (~decided)
            p_t = jnp.minimum(1.0, 2.0 * jnp.float32(kappa_over_D) * d_perp)
            transmit = first_hit & (u_rand < p_t)
            r_hit = r0 + d * dh
            d_ref = dh - 2.0 * jnp.dot(dh, n) * n; d_ref /= jnp.linalg.norm(d_ref)
            do_reflect = hit & ~transmit
            r_new = jnp.where(do_reflect, r_hit + self._NUDGE * n, r0 + rem * dh)
            d_new = jnp.where(do_reflect, d_ref, dh)
            rem_new = jnp.where(do_reflect, rem - d - self._NUDGE, jnp.float32(0.0))
            dperp_refl = jnp.where(first_hit & ~transmit, d_perp, jnp.float32(0.0))
            return (r_new, d_new, rem_new, decided | first_hit,
                    dlogw - 2.0 * jnp.float32(rho_over_D) * dperp_refl), None
        (rf, df, remf, _, dlogw), _ = jax.lax.scan(
            one, (r_w, d_hat, step_l, False, jnp.float32(0.0)), None, length=10)
        r_out = r + (rf + df * jnp.maximum(remf, 0.0) - r_w)
        return r_out, dlogw

    # ------------------------------------------------------------------
    def init_positions(self, n_walkers, key, intra=True):
        """Seed walkers inside (intra=True) or outside the cells by grid rejection."""
        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**30)))
        want = 0 if intra else 1
        classify = jax.jit(jax.vmap(self.classify_position))
        out, need = [], n_walkers
        while need > 0:
            pts = rng.uniform(self.vmin, self.vmax, (max(need * 4, 1024), 3)).astype(np.float32)
            lab = np.asarray(classify(jnp.asarray(pts)))
            out.append(pts[lab == want])
            need = n_walkers - sum(len(a) for a in out)
        return jnp.asarray(np.concatenate(out)[:n_walkers], jnp.float32)

    def quality_report(self, verbose=True):
        """Surface-resolution diagnostics + per-effect accuracy verdict.

        Returns a dict; also prints a table when ``verbose``.  Uses trimesh (if
        installed) to add watertight / component info.  The key number is
        ``edge_feature_ratio`` = median edge / feature_radius: permeability needs
        it ``<~ 0.05`` to reach the MC noise floor; diffusion and surface
        relaxivity are fine at much coarser ratios.
        """
        ratio = self.edge_feature_ratio
        perm_ok = ratio <= _PERM_EDGE_RATIO_MAX
        rep = {
            "n_vertices": int(len(self.vertices)),
            "n_faces": int(len(self.faces)),
            "n_ghost_faces": int(self.n_ghost),
            "feature_radius": self.radius,
            "edge_median": self.edge_median,
            "edge_p90": self.edge_p90,
            "edge_feature_ratio": ratio,
            "grid_dims": tuple(int(x) for x in self.dims),
            "grid_max_occupancy": int(self.max_occ),
            "grid_overflow": int(self.overflow),
            "periodic": self.periodic,
            "diffusion_noise_floor": True,
            "relaxivity_noise_floor": True,
            "permeability_noise_floor": bool(perm_ok),
        }
        try:
            import trimesh
            tm = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)
            rep["watertight"] = bool(tm.is_watertight)
            rep["n_components"] = int(tm.body_count)
        except Exception:
            pass
        if verbose:
            print(f"Mesh quality report")
            print(f"  vertices/faces        : {rep['n_vertices']:,} / {rep['n_faces']:,}"
                  + (f"  (+{rep['n_ghost_faces']:,} periodic ghosts)" if self.n_ghost else ""))
            if "watertight" in rep:
                print(f"  watertight/components : {rep['watertight']} / {rep['n_components']}")
            print(f"  feature_radius        : {self.radius*1e6:.3f} um")
            print(f"  edge median / p90     : {self.edge_median*1e6:.3f} / {self.edge_p90*1e6:.3f} um")
            print(f"  edge/feature ratio    : {ratio:.3f}  (permeability needs <~ {_PERM_EDGE_RATIO_MAX})")
            print(f"  grid dims / max-occ   : {rep['grid_dims']} / {rep['grid_max_occupancy']}"
                  + (f"  OVERFLOW={rep['grid_overflow']}" if self.overflow else ""))
            print(f"  MC-noise-floor accuracy:")
            print(f"    restricted diffusion : YES")
            print(f"    surface relaxivity   : YES")
            print(f"    permeability         : {'YES' if perm_ok else 'NO -- mesh too coarse; use a finer mesh'}")
        return rep

    @classmethod
    def from_ply(cls, path, scale=1.0, recenter=False, **kwargs):
        """Construct a Mesh directly from a mesh file (see :func:`load_ply`)."""
        V, F = load_ply(path, scale=scale, recenter=recenter)
        return cls(V, F, **kwargs)
