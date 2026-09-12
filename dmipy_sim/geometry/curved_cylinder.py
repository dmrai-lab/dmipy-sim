"""CurvedCylinder — a sphere-swept polyline geometry for curving fibres (e.g. DiSCo strands).

The intra-axonal space of a constant-radius fibre that follows an arbitrary curved
centerline is exactly ``{ r : dist(r, centerline_polyline) < R }`` — the Minkowski
sum of the polyline with a ball of radius R. This is smooth everywhere (cylindrical
along each segment, a spherical patch at every joint), so it has none of the
kink/gap/overlap artifacts of a chain of straight finite cylinders, and it captures
the local fibre orientation (which varies along the strand).

Reflection is specular off the local wall: the outward normal at a wall hit is the
radial direction from the nearest point on the centerline. Valid, as for the other
dmipy-sim geometries, in the single-reflection-per-step regime (step < R/6, enforced
by the core sub-step auto-tune via ``self.radius``).

Impermeable and analytic — no triangle mesh, no spatial grid — so it is ~orders of
magnitude cheaper than walking the equivalent triangulated tube.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from ._boundary import keep_side_radial, ray_quadric_t, specular, off_wall
from ._grid import bucket_by_bbox
import numpy as np

from .base import Geometry, LengthScales


class CurvedCylinder(Geometry):
    def __init__(self, centerline, radius: float, surface_relaxivity_t2=None):
        self.surface_relaxivity_t2 = None if surface_relaxivity_t2 is None else float(surface_relaxivity_t2)
        cl = np.asarray(centerline, np.float64)          # (P, 3) metres
        if cl.ndim != 2 or cl.shape[0] < 2:
            raise ValueError("centerline must be (P>=2, 3)")
        self.centerline = cl
        self.radius = float(radius)
        A = cl[:-1]
        B = cl[1:]
        self._A = jnp.asarray(A, jnp.float32)            # (M, 3)
        self._AB = jnp.asarray(B - A, jnp.float32)       # (M, 3)
        seglen2 = np.maximum(((B - A) ** 2).sum(1), 1e-30)
        self._AB2 = jnp.asarray(seglen2, jnp.float32)    # (M,)
        self._seglen = np.sqrt(seglen2)

    @property
    def length_scales(self):
        return LengthScales(min_feature=self.radius)

    # ---- containment / geometry ----
    def _nearest(self, r):
        """Nearest point on the centerline polyline to r, and the distance."""
        rA = r[None, :] - self._A                                   # (M,3)
        t = jnp.clip((rA * self._AB).sum(1) / self._AB2, 0.0, 1.0)  # (M,)
        Q = self._A + t[:, None] * self._AB                         # (M,3)
        dvec = r[None, :] - Q
        d2 = (dvec * dvec).sum(1)                                   # (M,)
        i = jnp.argmin(d2)
        return Q[i], jnp.sqrt(d2[i])

    def classify_position(self, r):
        """Compartment id: 1 inside the tube, 0 outside."""
        _, d = self._nearest(r)
        return jnp.where(d < jnp.float32(self.radius), jnp.int32(1), jnp.int32(0))

    def volume(self) -> float:
        return float(self._seglen.sum() * np.pi * self.radius ** 2)

    # ---- seeding: uniform inside the tube (arc-uniform x disk-uniform) ----
    def init_positions(self, n_walkers, key):
        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2 ** 30)))
        cl = self.centerline
        seg = cl[1:] - cl[:-1]
        L = self._seglen
        cumL = np.cumsum(L)
        total = float(cumL[-1])
        s = rng.uniform(0.0, total, n_walkers)
        idx = np.searchsorted(cumL, s, side="right")
        idx = np.clip(idx, 0, len(L) - 1)
        s0 = np.concatenate([[0.0], cumL])[idx]
        frac = np.clip((s - s0) / L[idx], 0.0, 1.0)
        C = cl[:-1][idx] + frac[:, None] * seg[idx]
        T = seg[idx] / L[idx][:, None]
        # arbitrary perpendicular frame per point
        ref = np.tile(np.array([0.0, 0.0, 1.0]), (n_walkers, 1))
        par = np.abs((T * ref).sum(1)) > 0.9
        ref[par] = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(T, ref); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(T, e1)
        rr = self.radius * np.sqrt(rng.uniform(0.0, 1.0, n_walkers))
        th = rng.uniform(0.0, 2 * np.pi, n_walkers)
        off = rr[:, None] * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
        return jnp.asarray(C + off, jnp.float32)

    # ---- specular reflection off the swept-tube wall ----
    def reflect(self, r, step):
        return self._reflect_contact(r, step)[0]

    def reflect_with_log_weight(self, r, step, rho_over_D):
        """The reflection with the surface-relaxation log-weight ``-2 (rho / D) d_perp`` of the wall contact:
        ``d_perp`` is the radial overshoot of the raw step past the wall, the perpendicular distance the base
        slab rule uses, read on the tube's own normal (single crossing per step, as there)."""
        r_out, d_perp = self._reflect_contact(r, step)
        return r_out, -2.0 * rho_over_D * d_perp

    def _reflect_contact(self, r, step):
        R = jnp.float32(self.radius)
        NUDGE = jnp.float32(1e-4 * self.radius)
        r_new = r + step
        Q, d = self._nearest(r_new)
        n = (r_new - Q) / (d + jnp.float32(1e-30))          # outward radial normal
        # mirror the radial overshoot back inside, then nudge just inside the wall
        r_ref = r_new - (2.0 * (d - R) + NUDGE) * n
        r_out = jnp.where(d > R, r_ref, r_new)
        d_perp = jnp.maximum(d - R, jnp.float32(0.0))           # the overshoot past the wall: the contact
        # safety clamp: if a sharp joint left it outside, put it back inside the wall.
        # `Q2 + (R - NUDGE) * n2` is exactly keep_side_radial's correction written out --
        # same rule, so it uses the same implementation and the same tie handling.
        Q2, _ = self._nearest(r_out)
        r_out, _ = keep_side_radial(r_out, r_out - Q2, R, True, NUDGE)
        return r_out, d_perp


class CurvedMyelinatedCylinder(CurvedCylinder):
    """A myelinated curved axon: concentric intra / myelin / extra shells swept along a
    curved centerline. Compartment by distance-to-centerline d:
      1 intra  (d < r_in),  2 myelin (r_in <= d < r_out),  0 extra (d >= r_out).
    Impermeable band-confined reflection keeps each walker in the shell it started in, so
    the three compartments are independent: seed a population with ``init_positions(...,
    shell=...)`` and walk it with that compartment's diffusivity (intra ~free, myelin
    ~stuck D->0, extra free). Because the local tangent varies along the strand, the
    myelin annulus carries the orientation-varying susceptibility source. (Single-pass
    per-walker-D, mirroring ``MyelinatedCylinder._is_myelinated``, is the later
    optimisation; impermeable shells make the separate-walk form exact.)
    """

    def __init__(self, centerline, r_in: float, r_out: float, pool="intra", surface_relaxivity_t2=None):
        super().__init__(centerline, r_out, surface_relaxivity_t2)   # base extent = outer radius
        if not (r_out > r_in > 0):
            raise ValueError("need r_out > r_in > 0")
        self.r_in = float(r_in)
        self.r_out = float(r_out)
        if pool not in ("intra", "myelin", "extra"):
            raise ValueError(f"pool must be 'intra', 'myelin' or 'extra', got {pool!r}")
        self.pool = pool                                   # the shell init_positions seeds
        self.radius = float(r_in)                      # auto-tune to the finest wall

    def classify_position(self, r):
        """Compartment id: 0 extra (d >= r_out), 1 intra (d < r_in), 2 myelin."""
        _, d = self._nearest(r)
        return jnp.where(d < jnp.float32(self.r_in), jnp.int32(1),
                         jnp.where(d < jnp.float32(self.r_out), jnp.int32(2), jnp.int32(0)))

    def reflect(self, r, step):
        return self._reflect_contact(r, step)[0]

    def reflect_with_log_weight(self, r, step, rho_over_D):
        """The band-confined reflection with the contact log-weight of the wall hit (either edge of the band)."""
        r_out, d_perp = self._reflect_contact(r, step)
        return r_out, -2.0 * rho_over_D * d_perp

    def _reflect_contact(self, r, step):
        r_in = jnp.float32(self.r_in); r_out = jnp.float32(self.r_out)
        NUDGE = jnp.float32(1e-4 * self.r_in)
        _, do = self._nearest(r)                       # band of the OLD position
        lo = jnp.where(do < r_in, jnp.float32(0.0), jnp.where(do < r_out, r_in, r_out))
        hi = jnp.where(do < r_in, r_in, jnp.where(do < r_out, r_out, jnp.float32(np.inf)))
        r_new = r + step
        Q, d = self._nearest(r_new)
        n = (r_new - Q) / (d + jnp.float32(1e-30))
        dt = d
        dt = jnp.where(d >= hi, 2.0 * hi - d - NUDGE, dt)  # mirror at the band's outer wall
        dt = jnp.where(d <= lo, 2.0 * lo - d + NUDGE, dt)  # mirror at the band's inner wall
        d_perp = jnp.maximum(jnp.where(jnp.isfinite(hi), d - hi, jnp.float32(-1.0)), jnp.float32(0.0)) \
            + jnp.where(lo > 0, jnp.maximum(lo - d, jnp.float32(0.0)), jnp.float32(0.0))       # the overshoot past either edge
        # Equality counts as the wrong side for BOTH neighbours (see _boundary): a walker
        # landing exactly on r_in or r_out belongs to neither band, and the strict `>` / `<`
        # used here previously left that tie unresolved -- the same defect that let walkers
        # change compartment without moving in the analytic geometries (#86). A mirror alone
        # also has no guarantee, so clamp the result into [lo, hi] explicitly.
        dt = jnp.clip(dt, lo + NUDGE, jnp.where(jnp.isfinite(hi), hi - NUDGE, dt))
        return Q + dt * n, d_perp

    def init_positions(self, n_walkers, key, pool=None, shell=None):
        if shell is not None:
            import warnings
            warnings.warn("init_positions(shell=...) is spelled pool=..., and the pool a driver seeds is the "
                          "constructor argument CurvedMyelinatedCylinder(pool=...)", DeprecationWarning, stacklevel=2)
            if pool is not None:
                raise ValueError("give pool= or shell=, not both")
            pool = shell
        shell = self.pool if pool is None else pool
        lo, hi = {"intra": (0.0, self.r_in),
                  "myelin": (self.r_in, self.r_out),
                  "extra": (self.r_out, 1.5 * self.r_out)}[shell]
        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2 ** 30)))
        cl = self.centerline; seg = cl[1:] - cl[:-1]; L = self._seglen
        cumL = np.cumsum(L); total = float(cumL[-1])
        s = rng.uniform(0.0, total, n_walkers)
        idx = np.clip(np.searchsorted(cumL, s, side="right"), 0, len(L) - 1)
        s0 = np.concatenate([[0.0], cumL])[idx]
        frac = np.clip((s - s0) / L[idx], 0.0, 1.0)
        C = cl[:-1][idx] + frac[:, None] * seg[idx]
        T = seg[idx] / L[idx][:, None]
        ref = np.tile(np.array([0.0, 0.0, 1.0]), (n_walkers, 1))
        ref[np.abs((T * ref).sum(1)) > 0.9] = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(T, ref); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(T, e1)
        rr = np.sqrt(rng.uniform(lo ** 2, hi ** 2, n_walkers))   # uniform-in-area radius
        th = rng.uniform(0.0, 2 * np.pi, n_walkers)
        off = rr[:, None] * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
        return jnp.asarray(C + off, jnp.float32)


class PackedCurvedCylinders(Geometry):
    """Extra-axonal diffusion around a pack of curved tubes, accelerated by a sparse grid
    over the tube *segments* (~O(#segments), not #triangles). An extra walker must not
    enter any tube: each step gathers the tube segments in the walker's 27-cell
    neighbourhood, and if the step would put it inside any tube (dist-to-segment < r_out)
    it reflects specularly off that tube's outer wall. This is the ~100x-lighter
    counterpart of a triangle-mesh grid for the same geometry.
    """

    def __init__(self, centerlines, radii, cell_size=None, interior=False, box=None, box_reflect=True,
                 surface_relaxivity_t2=None):
        self.surface_relaxivity_t2 = None if surface_relaxivity_t2 is None else float(surface_relaxivity_t2)
        # interior=False: extra-axonal (bounce off tube exteriors, stay outside all tubes)
        # interior=True : intra-axonal, all tubes at once (each walker confined inside its
        #                 own -- i.e. its nearest -- tube), one grid/one JIT for all tubes.
        # box=(lo, hi) : a finite, non-periodic voxel; with box_reflect its faces are mirrors that never
        #                carry a walker across a tube wall (the fold is refused, like an escaping step).
        self.interior = bool(interior)
        self.box = None if box is None else (np.asarray(box[0], float), np.asarray(box[1], float))
        self.box_reflect = bool(box_reflect) and self.box is not None
        if self.box_reflect:
            self._lo, self._hi = jnp.asarray(self.box[0], jnp.float32), jnp.asarray(self.box[1], jnp.float32)
        self.centerlines = [np.asarray(cl, np.float64) for cl in centerlines]     # what the pack was built from
        self.radii = np.asarray(radii, np.float64).reshape(-1)
        A, AB, rr = [], [], []
        for cl, R in zip(centerlines, radii):
            cl = np.asarray(cl, np.float64)
            A.append(cl[:-1]); AB.append(cl[1:] - cl[:-1]); rr.append(np.full(len(cl) - 1, float(R)))
        A = np.vstack(A); AB = np.vstack(AB); rout = np.concatenate(rr)
        self._seg_tube = jnp.asarray(np.concatenate([np.full(len(cl) - 1, k) for k, cl in enumerate(centerlines)]), jnp.int32)
        self._A = jnp.asarray(A, jnp.float32)
        self._AB = jnp.asarray(AB, jnp.float32)
        self._AB2 = jnp.asarray(np.maximum((AB ** 2).sum(1), 1e-30), jnp.float32)
        self._rout = jnp.asarray(rout, jnp.float32)
        self._Rmin = float(rout.min()); self._Rmax = float(rout.max())
        self.radius = self._Rmin                       # auto-tune to the finest wall
        cs = float(cell_size) if cell_size else (4.0 * self._Rmin / 6.0 + 2.0 * self._Rmax)
        self.cell_size = cs
        lo = np.minimum(A, A + AB) - self._Rmax
        hi = np.maximum(A, A + AB) + self._Rmax
        self.gmin = lo.min(0) - cs
        self.dims = np.maximum(1, np.ceil((hi.max(0) + cs - self.gmin) / cs).astype(int))
        loc = np.clip(np.floor((lo - self.gmin) / cs).astype(int), 0, self.dims - 1)
        hic = np.clip(np.floor((hi - self.gmin) / cs).astype(int), 0, self.dims - 1)
        cell, self.C, _max_occ, _overflow = bucket_by_bbox(loc, hic, self.dims, None)
        self._CELL = jnp.asarray(cell, jnp.int32)
        self._DIMS = tuple(int(x) for x in self.dims)
        self._dims_arr = jnp.asarray(self._DIMS, jnp.int32)
        self._GMIN = jnp.asarray(self.gmin, jnp.float32)
        self._CS = jnp.float32(cs)
        self._OFF = jnp.asarray([[dx, dy, dz] for dx in (-1, 0, 1)
                                 for dy in (-1, 0, 1) for dz in (-1, 0, 1)], jnp.int32)

    def _gather(self, r):
        c = jnp.clip(jnp.floor((r - self._GMIN) / self._CS).astype(jnp.int32), 0, self._dims_arr - 1)
        nb = jnp.clip(c[None, :] + self._OFF, 0, self._dims_arr - 1)
        cids = (nb[:, 0] * self._DIMS[1] + nb[:, 1]) * self._DIMS[2] + nb[:, 2]
        cand = self._CELL[cids].reshape(-1)
        valid = cand >= 0
        return jnp.where(valid, cand, 0), valid

    @property
    def length_scales(self):
        # a real tube radius (the R/6 rule applies) and a segment grid (the lookup rule applies)
        return LengthScales(min_feature=self._Rmin, lookup_cell=self.cell_size)

    classify_returns_object_id = True

    def classify_position(self, r):
        """Compartment id: ``k + 1`` inside tube ``k`` (1-indexed as every packed geometry; where tubes overlap or
        a thin tube runs close to a fat one, the tube the point is DEEPEST inside -- inside ANY tube, not only the
        nearest axis, which misread a walker in a fat tube close to a thin one's axis as outside), 0 outside every
        tube. The record of an interior walk carried pool 0 without this, and the pack then weighted every walker
        with the extra-cellular water fraction."""
        cand, valid = self._gather(r)
        A = self._A[cand]; AB = self._AB[cand]; AB2 = self._AB2[cand]; rr = self._rout[cand]
        t = jnp.clip(((r[None, :] - A) * AB).sum(1) / AB2, 0.0, 1.0)
        d = jnp.sqrt(((r[None, :] - (A + t[:, None] * AB)) ** 2).sum(1))
        depth = jnp.where(valid, rr - d, -jnp.inf)                   # positive inside a tube
        i = jnp.argmax(depth)
        return jnp.where(depth[i] > 0, self._seg_tube[cand[i]] + 1, 0).astype(jnp.int32)

    def inside_any(self, P, chunk=50000):
        """(n,3) → (n,) bool: is each point inside ANY tube (dist-to-segment < r_out)?
        Grid-accelerated (each point tests only its 27-cell segment neighbourhood) and
        GPU-vmapped in chunks — the fast primitive for seeding the extra-axonal space
        (rejection over ~O(#segments-per-cell), not the whole pack)."""
        P = np.asarray(P, np.float32)
        out = np.empty(P.shape[0], bool)

        # Built ONCE per instance, not per call. jax.jit caches compiled programs on the identity of
        # the function object it wraps, so a jit defined in this method body was a fresh object every
        # call and recompiled every time -- and `sample_outside` calls this in a rejection LOOP, so
        # that was a recompile per iteration. The closure captures this instance's segment arrays,
        # which are fixed at construction, so caching per instance (not module-wide) is the correct
        # scope. Measured on the same pattern in mesh.py: ~1.4 s per call -> 0.0002 s once hoisted.
        _batch = getattr(self, "_inside_any_batch", None)
        if _batch is None:
            @jax.jit
            def _batch(Pb):
                def one(p):
                    cand, valid = self._gather(p)
                    A = self._A[cand]; AB = self._AB[cand]; AB2 = self._AB2[cand]; rr = self._rout[cand]
                    t = jnp.clip(((p[None, :] - A) * AB).sum(1) / AB2, 0.0, 1.0)
                    d = jnp.linalg.norm(p[None, :] - (A + t[:, None] * AB), axis=1)
                    return (valid & (d < rr)).any()
                return jax.vmap(one)(Pb)
            self._inside_any_batch = _batch

        for i in range(0, P.shape[0], chunk):
            out[i:i + chunk] = np.asarray(_batch(jnp.asarray(P[i:i + chunk])))
        return out

    def radial_directors(self, P, chunk=50000):
        """``(n, 3)`` -> ``(n, 3)`` unit vectors from the nearest centerline point to each point: the sheath's
        radial (lipid) director at that point, exact from the geometry rather than from the gradient of a
        voxelised mask (dmipy-sim#213). Grid-accelerated like :meth:`inside_any`; a point with no segment in its
        neighbourhood gets the zero vector."""
        P = np.asarray(P, np.float32)
        out = np.zeros((P.shape[0], 3), np.float32)
        _batch = getattr(self, "_radial_batch", None)
        if _batch is None:
            @jax.jit
            def _batch(Pb):
                def one(p):
                    cand, valid = self._gather(p)
                    A = self._A[cand]; AB = self._AB[cand]; AB2 = self._AB2[cand]
                    t = jnp.clip(((p[None, :] - A) * AB).sum(1) / AB2, 0.0, 1.0)
                    Q = A + t[:, None] * AB
                    d2 = jnp.where(valid, ((p[None, :] - Q) ** 2).sum(1), jnp.inf)
                    i = jnp.argmin(d2)
                    v = p - Q[i]
                    n = jnp.sqrt((v * v).sum())
                    return jnp.where(valid.any() & (n > 0), v / jnp.maximum(n, 1e-30), jnp.zeros(3, v.dtype))
                return jax.vmap(one)(Pb)
            self._radial_batch = _batch
        for i in range(0, P.shape[0], chunk):
            out[i:i + chunk] = np.asarray(_batch(jnp.asarray(P[i:i + chunk])))
        return out

    def sample_outside(self, n_walkers, rng, bounds=None):
        """Uniformly sample `n_walkers` points in the extra-axonal space (outside all
        tubes). `bounds=(lo,hi)` overrides the tube bounding box (e.g. the voxel domain)."""
        lo = np.asarray(bounds[0]) if bounds else self.gmin
        hi = np.asarray(bounds[1]) if bounds else (self.gmin + self.dims * self.cell_size)
        acc = []; got = 0
        while got < n_walkers:
            P = rng.uniform(lo, hi, (max(n_walkers, 100000) * 2, 3))
            P = P[~self.inside_any(P)]
            acc.append(P); got += len(P)
        return np.concatenate(acc)[:n_walkers].astype(np.float32)

    def _inside_one(self, p):
        """Pure-JAX membership of one point in any tube."""
        cand, valid = self._gather(p)
        A = self._A[cand]; AB = self._AB[cand]; AB2 = self._AB2[cand]; rr = self._rout[cand]
        t = jnp.clip(((p[None, :] - A) * AB).sum(1) / AB2, 0.0, 1.0)
        d = jnp.linalg.norm(p[None, :] - (A + t[:, None] * AB), axis=1)
        return (valid & (d < rr)).any()

    def wall_scales(self, P, chunk=100_000):
        """``(n, 3) -> (d_wall (n,), R_near (n,))``: each point's distance to the nearest wall it can hit -- the
        nearest tube's surface (outside all tubes for the exterior pool, the confining tube's wall for the interior
        one) and, when the pack mirrors at a box, the nearest face -- and the radius of that nearest tube, the
        curvature scale of the wall. What an adaptive walk steps by (:mod:`dmipy_sim.engine.adaptive`): a walker
        farther from every wall than its round's excursion takes one free step, the rest step at the R/6 rule
        of THEIR tube rather than the pack's smallest. A point with no tube in reach gets ``inf`` and the
        pack's largest radius."""
        P = np.asarray(P, np.float32)
        out_d = np.empty(P.shape[0], np.float32); out_r = np.empty(P.shape[0], np.float32)
        _batch = self._wall_scales_device()
        for i in range(0, P.shape[0], chunk):
            d, r = _batch(jnp.asarray(P[i:i + chunk]))
            out_d[i:i + chunk] = np.asarray(d); out_r[i:i + chunk] = np.asarray(r)
        return out_d, out_r

    def _wall_scales_device(self):
        """The jitted ``(n, 3) -> (d_wall, R_near)`` of :meth:`wall_scales` on device arrays, built once."""
        _batch = getattr(self, "_wall_scales_batch", None)
        if _batch is None:
            interior = self.interior; box = self.box_reflect
            lo = self._lo if box else None; hi = self._hi if box else None
            R_max = jnp.float32(self._Rmax)

            @jax.jit
            def _batch(Pb):
                def one(p):
                    cand, valid = self._gather(p)
                    A = self._A[cand]; AB = self._AB[cand]; AB2 = self._AB2[cand]; rr = self._rout[cand]
                    t = jnp.clip(((p[None, :] - A) * AB).sum(1) / AB2, 0.0, 1.0)
                    d = jnp.linalg.norm(p[None, :] - (A + t[:, None] * AB), axis=1)
                    dd = jnp.where(valid, d, jnp.inf)
                    i = jnp.argmin(dd)
                    if interior:
                        d_wall = jnp.where(valid.any(), rr[i] - dd[i], jnp.inf)
                    else:
                        d_wall = jnp.where(valid, d - rr, jnp.inf).min()
                    R_near = jnp.where(valid.any(), rr[i], R_max)
                    if box:
                        d_wall = jnp.minimum(d_wall, jnp.minimum((p - lo).min(), (hi - p).min()))
                    return jnp.maximum(d_wall, 0.0), R_near
                return jax.vmap(one)(Pb)
            self._wall_scales_batch = _batch
        return _batch

    def _fold(self, r, r_new):
        """Mirror into the voxel, never across a tube wall."""
        if not self.box_reflect:
            return r_new
        span = self._hi - self._lo
        x = (r_new - self._lo) % (2.0 * span)
        folded = self._lo + jnp.where(x > span, 2.0 * span - x, x)
        ok = self._inside_one(folded) == self.interior
        return jnp.where(ok, folded, r)

    def reflect(self, r, step):
        return self._fold(r, self._reflect(r, step)[0])

    def reflect_with_log_weight(self, r, step, rho_over_D):
        """The reflection with the contact log-weight ``-2 (rho / D) d_perp``: inside, the overshoot past the
        tube's wall; outside, the radial part of the displacement left after the entry, on the entered tube's
        normal, as the exact packed cylinders read it."""
        r_new, d_perp = self._reflect(r, step)
        return self._fold(r, r_new), -2.0 * rho_over_D * d_perp

    def _reflect(self, r, step):
        NUDGE = jnp.float32(1e-4 * self._Rmin)
        r_new = r + step
        cand, valid = self._gather(r_new)
        A = self._A[cand]; AB = self._AB[cand]; AB2 = self._AB2[cand]; rr = self._rout[cand]
        t = jnp.clip(((r_new[None, :] - A) * AB).sum(1) / AB2, 0.0, 1.0)
        Q = A + t[:, None] * AB
        d = jnp.linalg.norm(r_new[None, :] - Q, axis=1)
        if self.interior:
            # confine each walker to its own (nearest) tube: reflect inward if it left it
            dd = jnp.where(valid, d, jnp.float32(np.inf))
            i = jnp.argmin(dd)
            Qh = Q[i]; dh = d[i]; rh = rr[i]
            n = (r_new - Qh) / (dh + jnp.float32(1e-30))
            r_ref = Qh + (2.0 * rh - dh - NUDGE) * n       # mirror back inside the tube
            r_int = jnp.where(dh >= rh, r_ref, r_new)      # equality is the wrong side
            # the mirror can still land outside off a sharp joint; the shared rule is the
            # guarantee, and it resolves the on-surface tie the same way everywhere
            r_int, _ = keep_side_radial(r_int, r_int - Qh, rh, True, NUDGE)
            return r_int, jnp.maximum(dh - rh, jnp.float32(0.0))
        # Proper specular reflection off the FIRST tube the step-ray enters: find the
        # entry point along the step, reflect the RADIAL component of the remaining
        # displacement (keeping the axial component), like the exact PackedCylinders.
        # (The old endpoint radial-mirror under-hindered dense packs by ~3%.)
        inside = valid & (d < rr)
        pen = jnp.where(inside, rr - d, -jnp.inf)           # the entered (deepest) tube
        i = jnp.argmax(pen)
        Ai = A[i]; ABi = AB[i]; rh = rr[i]
        u = ABi / jnp.sqrt(AB2[i] + jnp.float32(1e-30))     # segment axis (unit)
        rp = (r - Ai) - ((r - Ai) @ u) * u                  # start, radial to axis
        sp = step - (step @ u) * u                          # step, radial to axis
        aa = sp @ sp + jnp.float32(1e-30); bb = 2.0 * (rp @ sp); cc = rp @ rp - rh * rh
        # `aa t^2 + bb t + cc = 0` is the shared quadric with B = bb/2
        tau, _, _ = ray_quadric_t(aa, jnp.float32(0.5) * bb, cc)
        tau = jnp.clip(tau, 0.0, 1.0)                                   # first surface crossing
        entry = r + tau * step
        rem = (1.0 - tau) * step                            # displacement left after entry
        rem_ax = (rem @ u) * u                              # axial part continues
        rem_p = rem - rem_ax                                # radial part is reflected
        ne = (entry - Ai) - ((entry - Ai) @ u) * u
        nhat = ne / (jnp.linalg.norm(ne) + jnp.float32(1e-30))          # outward radial normal
        # shared rules: specular on the radial part, nudge off the wall on the outside
        r_ref = off_wall(entry, nhat, False, NUDGE) + rem_ax + specular(rem_p, nhat)
        # ... and the guarantee: an exterior walker must not end up inside the tube it
        # just bounced off. The mirror alone has no such guarantee near a joint.
        Qe = Ai + jnp.clip(((r_ref - Ai) @ u), 0.0, jnp.sqrt(AB2[i])) * u
        r_ref, _ = keep_side_radial(r_ref, r_ref - Qe, rh, False, NUDGE)
        hit = inside.any()
        d_perp = jnp.where(hit, jnp.abs(rem_p @ nhat), jnp.float32(0.0))       # the contact: the radial remainder on the normal
        return jnp.where(hit, r_ref, r_new), d_perp

    def sample_inside(self, n, rng):
        """``n`` points uniform by volume inside the tubes (segment volume, then the disc, then the length), the
        whole strand set: no box."""
        A = np.asarray(self._A); AB = np.asarray(self._AB); AB2 = np.asarray(self._AB2); rout = np.asarray(self._rout)
        L = np.sqrt(AB2)
        w = (rout ** 2) * L; w = w / w.sum()
        idx = rng.choice(len(A), size=int(n), p=w)
        C = A[idx] + rng.uniform(0.0, 1.0, int(n))[:, None] * AB[idx]
        T = AB[idx] / L[idx][:, None]
        ref = np.tile(np.array([0., 0., 1.]), (int(n), 1))
        ref[np.abs((T * ref).sum(1)) > 0.9] = np.array([1., 0., 0.])
        e1 = np.cross(T, ref); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(T, e1)
        rad = rout[idx] * np.sqrt(rng.uniform(0., 1., int(n)))
        th = rng.uniform(0., 2 * np.pi, int(n))
        return C + rad[:, None] * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)

    def init_positions(self, n_walkers, key):
        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2 ** 30)))
        if self.interior:
            pts = self.sample_inside(n_walkers, rng)
            if self.box is not None:                       # strands overrun the voxel: seed only inside it
                keep = ((pts >= self.box[0]) & (pts <= self.box[1])).all(1)
                pts = pts[keep]
                while len(pts) < n_walkers:
                    more = self.sample_inside(n_walkers, rng)
                    more = more[((more >= self.box[0]) & (more <= self.box[1])).all(1)]
                    pts = np.concatenate([pts, more])[:n_walkers]
            return jnp.asarray(pts, jnp.float32)
        # extra: rejection outside all tubes, grid-accelerated (see sample_outside/inside_any)
        return jnp.asarray(self.sample_outside(n_walkers, rng, bounds=self.box), jnp.float32)
