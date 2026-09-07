"""Diffusion inside or outside a **union of overlapping spheres** -- the native geometry of sphere-grown
substrates (CATERPillar), with no meshing step.

A cell is a chain of overlapping spheres, so its surface is the *outer* boundary of their union: the piece
of a sphere that lies inside a neighbour is an internal seam and reflects nothing. The collision test is a
CSG boundary test rather than a per-sphere test. For a walker **inside** the union, the boundary along a
ray is the end of the connected interval of the union containing ``t = 0``: every candidate sphere's
entry/exit pair is merged onto the walker's own interval,

    t* <- max{ t_exit(i) : t_enter(i) <= t* },   iterated to a fixed point,

which ignores intersections buried inside another sphere without forming the per-hit product. For a
walker **outside**, the first entry into *any* sphere is the union boundary.

Broad phase: uniform grids, one per **radius band**, the band width chosen by price. One grid cannot serve
a substrate where a 0.15 um glial process tip and a 4 um soma coexist, and splitting costs a 27-cell gather
per level, so the candidate layouts are priced (bucket counts only) and the cheapest that fits the memory
budget is built. Each level's cell is at least the maximum step, which makes the 27-cell gather exact (the
:class:`~dmipy_sim.geometry.mesh.Mesh` contract).

Reflection is specular about the analytic normal, multi-bounce within a step, with a reject-escape guard: a
step that would end on the wrong side of the union is refused rather than allowed to leak. Surface
relaxivity is the Brownstein-Tarr per-collision weight every geometry here uses. A finite, non-periodic
voxel (``box=``) is mirrored at its faces, and a mirror that would land a walker across a membrane is
refused the same way.

The spec spelling is ``surface.kind = "sphere_union"`` (replay-pack-spec/SUBSTRATE.md); the producer is
:func:`dmipy_sim.spec.caterpillar_spec`.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .base import Geometry, LengthScales

_OFFSETS = np.array([[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)], np.int32)
# Plain numpy constants only at module scope: a module-level jnp scalar is a live DEVICE buffer, and
# `free_gpu_memory(aggressive=True)` deletes every live array in the process.
_INF = np.float32(np.inf)


class _Level(NamedTuple):
    """One radius octave's device buffers, passed as an ARGUMENT to the jitted kernels.

    Bundling them in a pytree keeps the (potentially large) grid out of the traced closure, so it
    is a runtime buffer rather than a constant baked into every compiled kernel -- the same fix the
    mesh engine needed to seed large substrates on GPU.
    """
    CEN: jnp.ndarray      # (n, 3) sphere centres
    RAD: jnp.ndarray      # (n,) sphere radii
    CELL: jnp.ndarray     # (n_cells, C) sphere ids, -1 padded
    DIMS: jnp.ndarray     # (3,) grid dimensions
    GMIN: jnp.ndarray     # (3,) grid origin
    CS: jnp.ndarray       # scalar cell size
    OFF: jnp.ndarray      # (27, 3) neighbour offsets


# --------------------------------------------------------------------------- kernels

def _gather(L, r):
    """Candidate sphere ids in the walker's 27-cell neighbourhood of one level + validity."""
    c = jnp.clip(jnp.floor((r - L.GMIN) / L.CS).astype(jnp.int32), 0, L.DIMS - 1)
    nb = jnp.clip(c[None, :] + L.OFF, 0, L.DIMS - 1)
    cid = (nb[:, 0] * L.DIMS[1] + nb[:, 1]) * L.DIMS[2] + nb[:, 2]
    cand = L.CELL[cid].reshape(-1)
    valid = cand >= 0
    return jnp.where(valid, cand, 0), valid


def _inside(levels, r):
    """Is ``r`` inside the union (strictly inside at least one sphere of any level)?"""
    out = jnp.bool_(False)
    for L in levels:
        cand, valid = _gather(L, r)
        d2 = jnp.sum((r[None, :] - L.CEN[cand]) ** 2, axis=1)
        out = out | jnp.any(valid & (d2 < L.RAD[cand] ** 2))
    return out


def _intervals(L, r, u):
    """Ray/sphere entry and exit parameters for one level's candidates."""
    cand, valid = _gather(L, r)
    cen, rad = L.CEN[cand], L.RAD[cand]
    oc = r[None, :] - cen
    b = jnp.sum(oc * u[None, :], axis=1)
    c = jnp.sum(oc * oc, axis=1) - rad ** 2
    disc = b * b - c
    hit = valid & (disc > 0.0)
    s = jnp.sqrt(jnp.maximum(disc, 0.0))
    return (-b - s), (-b + s), hit, cen, rad


def _first_boundary(levels, r, u, rem, interior, eps, n_merge=3):
    """Distance to the union boundary along ``u`` (``inf`` if none within ``rem``), plus the centre
    and radius of the sphere that owns it."""
    iv = [_intervals(L, r, u) for L in levels]

    if interior:
        # Seed the merge with the spheres that actually contain the walker, then chain on any
        # sphere whose entry is already covered. Each pass can only grow t*, so it is monotone.
        t = jnp.float32(0.0)
        for t_in, t_out, hit, _, _ in iv:
            t = jnp.maximum(t, jnp.max(jnp.where(hit & (t_in <= 0.0) & (t_out > 0.0), t_out, 0.0)))
        for _ in range(n_merge):
            for t_in, t_out, hit, _, _ in iv:
                t = jnp.maximum(t, jnp.max(jnp.where(hit & (t_in <= t), t_out, 0.0)))
        # The owning sphere is the one whose exit defines the merged interval.
        best_t, best_c, best_r = -_INF, jnp.zeros(3, jnp.float32), jnp.float32(1.0)
        for t_in, t_out, hit, cen, rad in iv:
            score = jnp.where(hit & (t_in <= t + eps), t_out, -_INF)
            k = jnp.argmax(score)
            take = score[k] > best_t
            best_t = jnp.where(take, score[k], best_t)
            best_c = jnp.where(take, cen[k], best_c)
            best_r = jnp.where(take, rad[k], best_r)
        t = jnp.where((t > eps) & (t < rem), t, _INF)
    else:
        # Outside: the first entry into ANY sphere is already the union boundary.
        t, best_c, best_r = _INF, jnp.zeros(3, jnp.float32), jnp.float32(1.0)
        for t_in, t_out, hit, cen, rad in iv:
            score = jnp.where(hit & (t_in > eps), t_in, _INF)
            k = jnp.argmin(score)
            take = score[k] < t
            t = jnp.where(take, score[k], t)
            best_c = jnp.where(take, cen[k], best_c)
            best_r = jnp.where(take, rad[k], best_r)
        t = jnp.where(t < rem, t, _INF)
    return t, best_c, best_r


def _normal(r, u, t, cen, rad, interior):
    """Unit normal at the hit point, pointing back towards the legal side."""
    n = (r + t * u - cen) / rad
    return -n if interior else n


def _walk_step(levels, r, step, interior, max_bounces, nudge, eps):
    """One diffusion step with multi-bounce reflection; returns (r_new, sum d_perp)."""
    step_l = jnp.linalg.norm(step)
    u0 = step / jnp.maximum(step_l, 1e-30)

    def one(carry, _):
        r0, u, rem = carry
        t, cen, rad = _first_boundary(levels, r0, u, rem, interior, eps)
        hit = jnp.isfinite(t)
        t_safe = jnp.where(hit, t, 0.0)
        n = _normal(r0, u, t_safe, cen, rad, interior)
        u_ref = u - 2.0 * jnp.dot(u, n) * n
        u_ref = u_ref / jnp.maximum(jnp.linalg.norm(u_ref), 1e-30)
        r_hit = r0 + t_safe * u + nudge * n
        # Brownstein-Tarr: the path that would have crossed the wall, projected on the normal.
        d_perp = jnp.where(hit, (rem - t_safe) * jnp.abs(jnp.dot(u, n)), 0.0)
        return ((jnp.where(hit, r_hit, r0),
                 jnp.where(hit, u_ref, u),
                 jnp.where(hit, rem - t_safe - nudge, rem)),
                (d_perp, hit))

    (rf, uf, remf), (dperp, hits) = jax.lax.scan(one, (r, u0, step_l), None, length=max_bounces)
    # Fly the remaining path only if the last bounce found nothing -- otherwise the bounce budget
    # ran out mid-step and the leftover is untested (the mesh engine's rule, same reasoning).
    r_out = rf + uf * jnp.where(hits[-1], 0.0, jnp.maximum(remf, 0.0))
    # Reject-escape: a step that ends on the wrong side of the union never happened.
    ok = _inside(levels, r_out) == interior
    return jnp.where(ok, r_out, r), jnp.where(ok, jnp.sum(dperp), 0.0)


# --------------------------------------------------------------------------- geometry

class SphereUnion(Geometry):
    """Union-of-spheres geometry (metres): one pool, inside (``pool="intra"``) or outside (``pool="extra"``).

    Because the cells of a substrate are disjoint, one interior geometry confines every walker to its own
    cell with no per-cell bookkeeping; one exterior geometry hinders the extra-cellular walk by every cell.

    Parameters
    ----------
    centers, radii : (n, 3), (n,) arrays
    pool : ``"intra"`` (confined inside the union) or ``"extra"`` (kept outside it)
    feature_radius : the scale the sub-step auto-tune divides (default: the smallest sphere radius)
    cell_size : force ONE grid with this cell (diagnostics); default: one grid per radius octave, priced
    octaves : the layouts priced against each other (``None`` = one grid; a number = one grid per that ratio)
    max_candidates, max_bytes : budgets a layout must fit; exceeding them raises rather than dropping walls
    surface_relaxivity_t2 : rho (m/s)
    max_bounces : reflections resolved within one step
    box : ``(lo, hi)`` voxel bounds; with ``box_reflect`` the faces are mirrors that never cross a membrane
    """

    def __init__(self, centers, radii, *, pool="intra", feature_radius=None, cell_size=None,
                 octaves=(None, 4.0, 2.0), max_candidates=8192, max_bytes=2e9, surface_relaxivity_t2=None,
                 max_bounces=4, box=None, box_reflect=True):
        centers = np.asarray(centers, np.float64).reshape(-1, 3)
        radii = np.broadcast_to(np.asarray(radii, np.float64).ravel(), (len(centers),)).copy()
        if len(centers) == 0:
            raise ValueError("SphereUnion needs at least one sphere")
        if np.any(radii <= 0):
            raise ValueError("all sphere radii must be positive")
        if pool not in ("intra", "extra"):
            raise ValueError(f"pool must be 'intra' or 'extra', got {pool!r}")
        self.centers, self.radii = centers, radii
        self.pool = pool
        self.interior = pool == "intra"
        self.max_bounces = int(max_bounces)
        self.surface_relaxivity_t2 = float(surface_relaxivity_t2) if surface_relaxivity_t2 is not None else None
        self.box = None if box is None else (np.asarray(box[0], float), np.asarray(box[1], float))
        self.box_reflect = bool(box_reflect) and self.box is not None
        if self.box_reflect:
            self._lo, self._hi = jnp.asarray(self.box[0], jnp.float32), jnp.asarray(self.box[1], jnp.float32)

        r_min, r_max = float(radii.min()), float(radii.max())
        self.radius = float(feature_radius) if feature_radius else r_min
        self._step_l = self.radius / 6.0                 # the sub-step the auto-tune will take
        # float32 is the walk's working precision: a nudge or an epsilon below one ulp at the substrate's
        # coordinate magnitude vanishes. Floor both against it.
        ulp = (float(np.abs(centers).max()) + r_max) * 2.0 ** -23
        self._eps = np.float32(max(1e-6 * r_min, 8 * ulp))
        self._nudge = np.float32(max(1e-4 * r_min, 8 * ulp))
        self.octave = None
        self._build(cell_size, octaves, int(max_candidates), float(max_bytes))

    @property
    def length_scales(self):
        return LengthScales(min_feature=self.radius, lookup_cell=min(r["cell_size"] for r in self._report))

    # ------------------------------------------------------------------- build
    def _split(self, octave):
        """Sphere indices per radius octave; levels holding <1% of the spheres merge upward."""
        r = self.radii
        if octave is None or len(r) == 0:
            return [np.arange(len(r))]
        k = np.floor(np.log2(r / r.min()) / np.log2(octave)).astype(int)
        groups = [np.flatnonzero(k == kk) for kk in np.unique(k)]
        merged, carry = [], None
        for i, g in enumerate(groups):
            g = g if carry is None else np.concatenate([carry, g])
            if len(g) < 0.01 * len(r) and i < len(groups) - 1:
                carry = g
                continue
            merged.append(g)
            carry = None
        if carry is not None:                                  # tail too small to stand alone
            if merged:
                merged[-1] = np.concatenate([merged[-1], carry])
            else:
                merged.append(carry)
        return merged

    def _cost(self, idx, cell_size, rule=None):
        """Price one level WITHOUT allocating it: bucket assignment, cell count, capacity.

        ``rule`` picks the cell size from the level's radii. Two are worth trying: twice the
        LARGEST radius (every sphere lands in ~3^3 cells, fewest cells, but a wide band then packs
        many small spheres per cell) and twice the MEDIAN (more cells, fewer candidates). Which
        wins depends on how wide the band is, so both get priced.
        """
        cen, rad = self.centers[idx], self.radii[idx]
        # At least the maximum step: that is what makes the 27-cell gather exact.
        cs = (float(cell_size) if cell_size
              else max(4.0 * self._step_l, float(rule(rad)) if rule else 2.0 * float(rad.max())))
        lo, hi = cen - rad[:, None], cen + rad[:, None]
        gmin = lo.min(axis=0) - cs
        dims = np.maximum(1, np.ceil((hi.max(axis=0) + cs - gmin) / cs).astype(int))
        n_cells = int(np.prod(dims))
        if n_cells > 500_000_000:                              # would not fit any allocation
            return None

        # Vectorised bucketing: expand each sphere over the cells its bbox overlaps, then sort.
        # (A Python loop over primitives is what makes this unusable at 1e6 spheres.)
        c0 = np.clip(np.floor((lo - gmin) / cs).astype(int), 0, dims - 1)
        c1 = np.clip(np.floor((hi - gmin) / cs).astype(int), 0, dims - 1)
        span = c1 - c0 + 1
        counts = span.prod(axis=1)
        n_ent = int(counts.sum())
        sid = np.repeat(np.arange(len(rad)), counts)
        k = np.arange(n_ent) - np.repeat(np.cumsum(counts) - counts, counts)
        sy, sz = span[sid, 1], span[sid, 2]
        ix = c0[sid, 0] + k // (sy * sz)
        iy = c0[sid, 1] + (k // sz) % sy
        iz = c0[sid, 2] + k % sz
        cid = (ix * dims[1] + iy) * dims[2] + iz
        order = np.argsort(cid, kind="stable")
        cid, sid = cid[order], sid[order]
        occ = np.bincount(cid, minlength=n_cells)
        C = int(occ.max())
        return dict(idx=idx, cid=cid, sid=sid, occ=occ, dims=dims, gmin=gmin, cs=cs,
                    C=C, n_cells=n_cells, n_ent=n_ent, bytes=4 * n_cells * C)

    def _materialize(self, plan):
        idx = plan["idx"]
        cen, rad = self.centers[idx], self.radii[idx]
        starts = np.cumsum(plan["occ"]) - plan["occ"]      # first slot of each cell, sorted order
        slot = np.arange(plan["n_ent"]) - starts[plan["cid"]]
        cell = np.full((plan["n_cells"], plan["C"]), -1, np.int32)
        cell[plan["cid"], slot] = plan["sid"]
        level = _Level(CEN=jnp.asarray(cen, jnp.float32), RAD=jnp.asarray(rad, jnp.float32),
                       CELL=jnp.asarray(cell, jnp.int32), DIMS=jnp.asarray(plan["dims"], jnp.int32),
                       GMIN=jnp.asarray(plan["gmin"], jnp.float32), CS=jnp.float32(plan["cs"]),
                       OFF=jnp.asarray(_OFFSETS, jnp.int32))
        report = dict(n_spheres=int(len(rad)), cells=plan["n_cells"], C=plan["C"],
                      entries=plan["n_ent"], cell_size=plan["cs"],
                      r_min=float(rad.min()), r_max=float(rad.max()))
        return level, report

    def _build(self, cell_size, octaves, max_candidates, max_bytes):
        """Pick the layout that tests the FEWEST candidates per step, then build only that one.

        Splitting by radius is what makes a mixed-scale substrate representable at all, but it is
        not free: every level costs its own 27-cell gather, so on a substrate whose radii nearly fit
        one octave a single grid is cheaper. Price the options first (bucket counts only, no dense
        allocation) and keep the cheapest that fits the memory budget.
        """
        options = [None] if cell_size else list(octaves)
        rules = (lambda rad: 2.0 * rad.max(), lambda rad: 2.0 * np.median(rad))
        best, best_cost = None, None
        for octave in options:
            plans, budget = [], max_bytes
            for idx in self._split(octave):
                priced = [p for p in (self._cost(idx, cell_size, rule) for rule in rules)
                          if p is not None and p["bytes"] <= budget and 27 * p["C"] <= max_candidates]
                if not priced:
                    plans = None
                    break
                p = min(priced, key=lambda q: (q["C"], q["bytes"]))
                budget -= p["bytes"]
                plans.append(p)
            if not plans:
                continue
            cand = int(sum(27 * p["C"] for p in plans))
            if best_cost is None or cand < best_cost:
                best, best_cost, self.octave = plans, cand, octave
        if best is None:
            raise MemoryError(
                "SphereUnion could not lay out this substrate within "
                f"max_candidates={max_candidates} and max_bytes={max_bytes / 1e9:.1f} GB. The cost "
                "is chain REDUNDANCY (CATERPillar packs ~20 spheres per micron of glial process), "
                "which no cell size fixes -- decimate the chains with a stated tolerance, or raise "
                "the budgets knowingly.")
        levels, report = [], []
        for plan in best:
            L, rep = self._materialize(plan)
            levels.append(L)
            report.append(rep)
        self.levels, self._report = tuple(levels), report

    def grid_report(self):
        """What the broad phase costs: one row per level, plus the total candidate budget."""
        return dict(n_spheres=int(len(self.radii)), octave=self.octave, levels=self._report,
                    candidates_per_step=int(sum(27 * r["C"] for r in self._report)),
                    bytes=int(sum(4 * r["cells"] * r["C"] for r in self._report)))

    # ------------------------------------------------------------------ queries
    def inside_any(self, P, chunk=200_000):
        """(n, 3) -> (n,) bool: is each point inside the union? Host-facing, chunked on device."""
        P = np.asarray(P, np.float32).reshape(-1, 3)
        f = jax.jit(jax.vmap(_inside, in_axes=(None, 0)))
        out = np.empty(len(P), bool)
        for lo in range(0, len(P), chunk):
            out[lo:lo + chunk] = np.asarray(f(self.levels, jnp.asarray(P[lo:lo + chunk])))
        return out

    def classify_position(self, r):
        """1 inside the union, 0 outside (pure JAX, per walker)."""
        return jnp.where(_inside(self.levels, r), 1, 0)

    def volume(self, n_probe=200_000, seed=0, bounds=None):
        """Monte-Carlo union volume (m^3). A sphere union has no closed form."""
        lo, hi = bounds if bounds is not None else (self.centers.min(axis=0) - self.radii.max(),
                                                    self.centers.max(axis=0) + self.radii.max())
        rng = np.random.default_rng(seed)
        p = rng.uniform(lo, hi, (int(n_probe), 3))
        return float(self.inside_any(p).mean() * np.prod(np.asarray(hi) - np.asarray(lo)))

    def surface_area(self, n_probe=400_000, seed=0):
        """Monte-Carlo union surface area (m^2): the fraction of each sphere's own surface that is
        not buried inside another sphere, summed."""
        rng = np.random.default_rng(seed)
        per = max(64, int(n_probe / len(self.radii)))
        v = rng.normal(size=(per, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        # Push the probe just off its own surface before asking "is this buried in a NEIGHBOUR?".
        # The offset has to clear float32 resolution at micrometre radii (~1e-7 relative), so 1e-4
        # relative -- sub-nanometre in absolute terms, invisible against any real feature.
        area = 0.0
        for i in range(len(self.radii)):                    # host loop: diagnostics, not hot path
            pts = self.centers[i] + self.radii[i] * v * (1.0 + 1e-4)
            area += 4.0 * np.pi * self.radii[i] ** 2 * (~self.inside_any(pts)).mean()
        return float(area)

    # ------------------------------------------------------------------ seeding
    def init_positions(self, n_walkers, key, bounds=None, oversample=6):
        """Uniform positions on the legal side of the union (rejection sampling) within ``bounds``, the box,
        or the union's bounding box."""
        if bounds is None and self.box is not None:
            bounds = self.box
        if bounds is None:
            lo = self.centers.min(axis=0) - self.radii.max()
            hi = self.centers.max(axis=0) + self.radii.max()
        else:
            lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
        out, need = [], int(n_walkers)
        while need > 0:
            key, sub = jax.random.split(key)
            m = max(1024, int(need * oversample))
            p = np.asarray(jax.random.uniform(sub, (m, 3), minval=jnp.asarray(lo, jnp.float32),
                                              maxval=jnp.asarray(hi, jnp.float32)))
            good = p[self.inside_any(p) == self.interior]
            if len(good):
                out.append(good[:need])
                need -= len(out[-1])
            oversample = min(64, oversample * 2)
        return jnp.asarray(np.concatenate(out, axis=0), jnp.float32)

    # ------------------------------------------------------------------- walk
    def _fold(self, r, r_new):
        """Mirror into the voxel, but never across a membrane: a fold near a face that lands inside a cell
        would be a compartment change with no wall crossing, so it is refused like an escaping step."""
        if not self.box_reflect:
            return r_new, jnp.bool_(True)
        span = self._hi - self._lo
        x = (r_new - self._lo) % (2.0 * span)
        folded = self._lo + jnp.where(x > span, 2.0 * span - x, x)
        ok = _inside(self.levels, folded) == self.interior
        return jnp.where(ok, folded, r), ok

    def reflect(self, r, step):
        r_new, _ = _walk_step(self.levels, r, step, self.interior, self.max_bounces, self._nudge, self._eps)
        return self._fold(r, r_new)[0]

    def reflect_with_log_weight(self, r, step, rho_over_D):
        r_new, dsum = _walk_step(self.levels, r, step, self.interior, self.max_bounces, self._nudge, self._eps)
        folded, ok = self._fold(r, r_new)
        return folded, jnp.where(ok, -2.0 * jnp.float32(rho_over_D) * dsum, jnp.float32(0.0))
