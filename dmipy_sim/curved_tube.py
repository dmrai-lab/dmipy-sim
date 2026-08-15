"""CurvedTube — a sphere-swept polyline geometry for curving fibres (e.g. DiSCo strands).

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

from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

from .geometries import Geometry


class CurvedTube(Geometry):
    def __init__(self, centerline, radius: float):
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
        _, d = self._nearest(r)
        return jnp.where(d < jnp.float32(self.radius), jnp.int32(0), jnp.int32(1))

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
        R = jnp.float32(self.radius)
        NUDGE = jnp.float32(1e-4 * self.radius)
        r_new = r + step
        Q, d = self._nearest(r_new)
        n = (r_new - Q) / (d + jnp.float32(1e-30))          # outward radial normal
        # mirror the radial overshoot back inside, then nudge just inside the wall
        r_ref = r_new - (2.0 * (d - R) + NUDGE) * n
        r_out = jnp.where(d > R, r_ref, r_new)
        # safety clamp: if a sharp joint left it outside, project onto the wall inside
        Q2, d2 = self._nearest(r_out)
        n2 = (r_out - Q2) / (d2 + jnp.float32(1e-30))
        r_out = jnp.where(d2 >= R, Q2 + (R - NUDGE) * n2, r_out)
        return r_out


class MultiShellCurvedTube(CurvedTube):
    """A myelinated curved axon: concentric intra / myelin / extra shells swept along a
    curved centerline. Compartment by distance-to-centerline d:
      0 intra  (d < r_in),  1 myelin (r_in <= d < r_out),  2 extra (d >= r_out).
    Impermeable band-confined reflection keeps each walker in the shell it started in, so
    the three compartments are independent: seed a population with ``init_positions(...,
    shell=...)`` and walk it with that compartment's diffusivity (intra ~free, myelin
    ~stuck D->0, extra free). Because the local tangent varies along the strand, the
    myelin annulus carries the orientation-varying susceptibility source. (Single-pass
    per-walker-D, mirroring ``MyelinatedCylinder._is_myelinated``, is the later
    optimisation; impermeable shells make the separate-walk form exact.)
    """

    def __init__(self, centerline, r_in: float, r_out: float):
        super().__init__(centerline, r_out)            # base extent = outer radius
        if not (r_out > r_in > 0):
            raise ValueError("need r_out > r_in > 0")
        self.r_in = float(r_in)
        self.r_out = float(r_out)
        self.radius = float(r_in)                      # auto-tune to the finest wall

    def classify_position(self, r):
        _, d = self._nearest(r)
        return jnp.where(d < jnp.float32(self.r_in), jnp.int32(0),
                         jnp.where(d < jnp.float32(self.r_out), jnp.int32(1), jnp.int32(2)))

    def reflect(self, r, step):
        r_in = jnp.float32(self.r_in); r_out = jnp.float32(self.r_out)
        NUDGE = jnp.float32(1e-4 * self.r_in)
        _, do = self._nearest(r)                       # band of the OLD position
        lo = jnp.where(do < r_in, jnp.float32(0.0), jnp.where(do < r_out, r_in, r_out))
        hi = jnp.where(do < r_in, r_in, jnp.where(do < r_out, r_out, jnp.float32(np.inf)))
        r_new = r + step
        Q, d = self._nearest(r_new)
        n = (r_new - Q) / (d + jnp.float32(1e-30))
        dt = d
        dt = jnp.where(d > hi, 2.0 * hi - d - NUDGE, dt)   # mirror at the band's outer wall
        dt = jnp.where(d < lo, 2.0 * lo - d + NUDGE, dt)   # mirror at the band's inner wall
        return Q + dt * n

    def init_positions(self, n_walkers, key, shell="intra"):
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


class PackedCurvedTubes(Geometry):
    """Extra-axonal diffusion around a pack of curved tubes, accelerated by a sparse grid
    over the tube *segments* (~O(#segments), not #triangles). An extra walker must not
    enter any tube: each step gathers the tube segments in the walker's 27-cell
    neighbourhood, and if the step would put it inside any tube (dist-to-segment < r_out)
    it reflects specularly off that tube's outer wall. This is the ~100x-lighter
    counterpart of a triangle-mesh grid for the same geometry.
    """

    def __init__(self, centerlines, radii, cell_size=None, interior=False):
        # interior=False: extra-axonal (bounce off tube exteriors, stay outside all tubes)
        # interior=True : intra-axonal, all tubes at once (each walker confined inside its
        #                 own -- i.e. its nearest -- tube), one grid/one JIT for all tubes.
        self.interior = bool(interior)
        A, AB, rr = [], [], []
        for cl, R in zip(centerlines, radii):
            cl = np.asarray(cl, np.float64)
            A.append(cl[:-1]); AB.append(cl[1:] - cl[:-1]); rr.append(np.full(len(cl) - 1, float(R)))
        A = np.vstack(A); AB = np.vstack(AB); rout = np.concatenate(rr)
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
        buckets = defaultdict(list)
        for s in range(len(A)):
            for ix in range(loc[s, 0], hic[s, 0] + 1):
                for iy in range(loc[s, 1], hic[s, 1] + 1):
                    for iz in range(loc[s, 2], hic[s, 2] + 1):
                        buckets[(ix * self.dims[1] + iy) * self.dims[2] + iz].append(s)
        self.C = max((len(v) for v in buckets.values()), default=1)
        cell = np.full((int(np.prod(self.dims)), self.C), -1, np.int32)
        for cid, lst in buckets.items():
            cell[cid, :len(lst)] = lst
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

    def inside_any(self, P, chunk=50000):
        """(n,3) → (n,) bool: is each point inside ANY tube (dist-to-segment < r_out)?
        Grid-accelerated (each point tests only its 27-cell segment neighbourhood) and
        GPU-vmapped in chunks — the fast primitive for seeding the extra-axonal space
        (rejection over ~O(#segments-per-cell), not the whole pack)."""
        P = np.asarray(P, np.float32)
        out = np.empty(P.shape[0], bool)

        @jax.jit
        def _batch(Pb):
            def one(p):
                cand, valid = self._gather(p)
                A = self._A[cand]; AB = self._AB[cand]; AB2 = self._AB2[cand]; rr = self._rout[cand]
                t = jnp.clip(((p[None, :] - A) * AB).sum(1) / AB2, 0.0, 1.0)
                d = jnp.linalg.norm(p[None, :] - (A + t[:, None] * AB), axis=1)
                return (valid & (d < rr)).any()
            return jax.vmap(one)(Pb)

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

    def reflect(self, r, step):
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
            return jnp.where(dh > rh, r_ref, r_new)
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
        disc = jnp.maximum(bb * bb - 4.0 * aa * cc, jnp.float32(0.0))
        tau = jnp.clip((-bb - jnp.sqrt(disc)) / (2.0 * aa), 0.0, 1.0)   # first surface crossing
        entry = r + tau * step
        rem = (1.0 - tau) * step                            # displacement left after entry
        rem_ax = (rem @ u) * u                              # axial part continues
        rem_p = rem - rem_ax                                # radial part is reflected
        ne = (entry - Ai) - ((entry - Ai) @ u) * u
        nhat = ne / (jnp.linalg.norm(ne) + jnp.float32(1e-30))          # outward radial normal
        r_ref = entry + rem_ax + (rem_p - 2.0 * (rem_p @ nhat) * nhat) + NUDGE * nhat
        return jnp.where(inside.any(), r_ref, r_new)

    def init_positions(self, n_walkers, key):
        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2 ** 30)))
        A = np.asarray(self._A); AB = np.asarray(self._AB)
        AB2 = np.asarray(self._AB2); rout = np.asarray(self._rout)
        if self.interior:
            # seed inside the tubes, volume-weighted over segments (pi R^2 * length)
            L = np.sqrt(AB2)
            w = (rout ** 2) * L; w = w / w.sum()
            idx = rng.choice(len(A), size=n_walkers, p=w)
            C = A[idx] + rng.uniform(0.0, 1.0, n_walkers)[:, None] * AB[idx]
            T = AB[idx] / L[idx][:, None]
            ref = np.tile(np.array([0., 0., 1.]), (n_walkers, 1))
            ref[np.abs((T * ref).sum(1)) > 0.9] = np.array([1., 0., 0.])
            e1 = np.cross(T, ref); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
            e2 = np.cross(T, e1)
            rad = rout[idx] * np.sqrt(rng.uniform(0., 1., n_walkers))
            th = rng.uniform(0., 2 * np.pi, n_walkers)
            off = rad[:, None] * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
            return jnp.asarray(C + off, jnp.float32)
        # extra: rejection outside all tubes, grid-accelerated (see sample_outside/inside_any)
        return jnp.asarray(self.sample_outside(n_walkers, rng), jnp.float32)
