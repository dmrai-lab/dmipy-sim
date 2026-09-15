"""The susceptibility field basis of a strand substrate (sheathed swept polylines: DiSCo, the EPFL strand lists)
evaluated at points, without a grid: every strand within a cutoff of the point contributes the closed-form field
of the infinite hollow cylinder of its nearest segment (:mod:`dmipy_sim.fields.hollow_cylinder`), and the
contributions are summed. A pack's path channel samples this along each walker's path exactly as it samples a
:class:`~dmipy_sim.fields.susceptibility_field.FieldGrid`; the two are one protocol (``channels(points)``).

Why not the grid: DiSCo's 1 mm^3 at 0.2 um is 1.25e11 voxels; the superposition costs the strands within the
cutoff per point (~120 at 25 um in the densest DiSCo region). Why the superposition is the right field for a
finite, reflecting domain: the k-space route is periodic, and its images cost 6.5 % of the lumen field on a 6 um
box (1.4 % at 12 um), while the truncated superposition converges with the cutoff (measured on DiSCo: 2.5 % of
the isotropic and 0.8 % of the anisotropic channels' rms at 25 um against 100 um; 1.7 % / 0.5 % at 50 um) --
:meth:`StrandFieldBasis.cutoff_error` is that number for the substrate at hand, the certificate a producer
doubles the cutoff against. What the local-cylinder rule costs at a joint: a DiSCo strand with a 54-degree joint
reproduces the rasterised field to 3-6 % of the component's maximum in the lumen and sheath at the joint, 1-2 %
outside, 0.3 % beyond three radii. Overlapping strands (DiSCo admits residual overlap) are summed where the
rasterised mask takes their union: a second-order difference confined to the overlap volume.

Every channel is relative to the domain mean of the bare superposition, computed in closed form from the strands'
lengths inside the domain (the Lorentz reference the k-space route imposes by zeroing k = 0; a uniform offset a
magnitude signal cannot see).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from .hollow_cylinder import hollow_cylinder_basis, contract, annulus_mean_log, CHANNEL_NAMES
from ..geometry._grid import bucket_by_bbox


def _length_inside(A, AB, lo, hi):
    """The length of each segment ``A + t AB, t in [0, 1]`` that lies inside the box (Liang-Barsky clipping)."""
    t0 = np.zeros(len(A)); t1 = np.ones(len(A))
    for k in range(3):
        d = AB[:, k]
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (lo[k] - A[:, k]) / d; tb = (hi[k] - A[:, k]) / d
        tmin = np.where(d > 0, ta, tb); tmax = np.where(d > 0, tb, ta)
        par = d == 0
        t0 = np.where(par, np.where((A[:, k] >= lo[k]) & (A[:, k] <= hi[k]), t0, 1.0), np.maximum(t0, tmin))
        t1 = np.where(par, np.where((A[:, k] >= lo[k]) & (A[:, k] <= hi[k]), t1, 0.0), np.minimum(t1, tmax))
    return np.maximum(t1 - t0, 0.0) * np.linalg.norm(AB, axis=1)


@dataclass(frozen=True)
class FarGrid:
    """The far part of a strand substrate's field on a coarse grid: the superposition over every strand within the
    build cutoff, each strand's contribution weighted by the switch ``S(d)`` that rises from 0 at ``near_m - blend_m``
    to 1 at ``near_m`` with the distance ``d`` to its nearest segment -- smooth on the grid's scale by construction,
    so trilinear interpolation reads it -- and the complement ``1 - S`` is what the closed form sums over the few
    strands within ``near_m`` of a point (the particle-mesh split; issue #217). ``values`` ``(nx, ny, nz, 13)`` on the
    nodes ``origin + (i, j, k) * spacing_m``; ``cutoff_m`` the superposition cutoff the grid summed to."""
    origin_m: tuple
    spacing_m: float
    values: np.ndarray
    near_m: float
    blend_m: float
    cutoff_m: float

    @property
    def shape(self):
        return tuple(int(n) for n in self.values.shape[:3])

    @property
    def sha256(self):
        return hashlib.sha256(np.ascontiguousarray(self.values).tobytes()).hexdigest()

    @property
    def meta(self):
        return dict(spacing_m=float(self.spacing_m), near_m=float(self.near_m), blend_m=float(self.blend_m),
                    cutoff_m=float(self.cutoff_m), origin_m=[float(x) for x in self.origin_m], shape=list(self.shape),
                    dtype=str(np.dtype(self.values.dtype)), sha256=self.sha256)

    def save(self, path):
        """The grid as one ``.npz``."""
        np.savez(path, values=self.values, origin_m=np.asarray(self.origin_m, np.float64), spacing_m=self.spacing_m,
                 near_m=self.near_m, blend_m=self.blend_m, cutoff_m=self.cutoff_m)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(tuple(float(x) for x in z["origin_m"]), float(z["spacing_m"]), np.asarray(z["values"]), float(z["near_m"]),
                   float(z["blend_m"]), float(z["cutoff_m"]))


class StrandFieldBasis:
    """The field basis of sheathed strands, evaluated at points.

    ``centerlines``: a list of ``(P_i, 3)`` polylines (metres); ``inner_radii`` / ``outer_radii``: per strand
    (metres), the axolemma and the sheath's outer surface; ``cutoff_m``: strands whose nearest segment lies
    farther than this from a point are not summed; ``domain``: ``(lo, hi)`` of the walked box, over which the
    mean is taken (defaults to the strands' bounding box); ``strands_max``: the most strands a point may have
    within the cutoff (the closed form is evaluated for that many nearest candidates per point; a point with
    more is refused with the count, so the cutoff or this bound is raised knowingly).
    """

    #: two segments of one strand whose distances to a point differ by less than this fraction of the cutoff are
    #: a tie (a joint), resolved to the lower segment index: a rule, so that rounding does not choose
    TIE_TOL = 1e-4
    #: beyond a strand's FREE end the local infinite cylinder tapers out over this many outer radii of axial overshoot
    #: (a cubic smoothstep): a finite cylinder's field beyond its end falls off on the scale of its radius, and the
    #: extrapolated sheath would otherwise be a discontinuity in space (DiSCo's 24,392 ends sit inside its box)
    END_TAPER_RADII = 2.0

    def __init__(self, centerlines, inner_radii, outer_radii, *, cutoff_m, domain=None, certificate=None, strands_max=1024,
                 far=None):
        self.centerlines = [np.asarray(c, np.float64) for c in centerlines]
        self.inner_radii = np.asarray(inner_radii, np.float64).reshape(-1)
        self.outer_radii = np.asarray(outer_radii, np.float64).reshape(-1)
        if not (len(self.centerlines) == len(self.inner_radii) == len(self.outer_radii)):
            raise ValueError("centerlines, inner_radii and outer_radii must have one entry per strand")
        if np.any(self.outer_radii <= self.inner_radii):
            raise ValueError("every outer radius must exceed its inner radius")
        if any(c.ndim != 2 or c.shape[0] < 2 or c.shape[1] != 3 for c in self.centerlines):
            raise ValueError("every centerline must be (P >= 2, 3)")
        self.cutoff_m = float(cutoff_m)
        self.strands_max = int(strands_max)
        self.certificate = None if certificate is None else dict(certificate)
        if not self.cutoff_m > 0:
            raise ValueError("cutoff_m must be positive")
        self.far = far
        if far is not None:
            if not isinstance(far, FarGrid):
                raise TypeError("far must be a FarGrid (StrandFieldBasis.build_far_grid, or FarGrid.load)")
            if not (0 < far.blend_m < far.near_m <= self.cutoff_m):
                raise ValueError("a far grid's switch needs 0 < blend_m < near_m <= cutoff_m")
        # the radius the closed form is summed within: the switch's reach with a far grid, the cutoff without
        self.gather_radius_m = float(far.near_m) if far is not None else self.cutoff_m
        A, AB, sid, e0, e1 = [], [], [], [], []
        for k, c in enumerate(self.centerlines):
            n_s = len(c) - 1
            A.append(c[:-1]); AB.append(c[1:] - c[:-1]); sid.append(np.full(n_s, k))
            f0 = np.zeros(n_s, bool); f0[0] = True; f1 = np.zeros(n_s, bool); f1[-1] = True     # the strand's free ends
            e0.append(f0); e1.append(f1)
        A = np.vstack(A); AB = np.vstack(AB); sid = np.concatenate(sid)
        self._E0 = jnp.asarray(np.concatenate(e0)); self._E1 = jnp.asarray(np.concatenate(e1))
        self.n_strands = len(self.centerlines); self.n_segments = int(A.shape[0])
        self.domain = (tuple(np.minimum(A, A + AB).min(0) - self.outer_radii.max()), tuple(np.maximum(A, A + AB).max(0) + self.outer_radii.max())) \
            if domain is None else (tuple(np.asarray(domain[0], float)), tuple(np.asarray(domain[1], float)))
        self._A = jnp.asarray(A, jnp.float32); self._AB = jnp.asarray(AB, jnp.float32)
        self._AB2 = jnp.asarray(np.maximum((AB ** 2).sum(1), 1e-30), jnp.float32)
        self._sid = jnp.asarray(sid, jnp.int32)
        self._a = jnp.asarray(self.inner_radii[sid], jnp.float32); self._b = jnp.asarray(self.outer_radii[sid], jnp.float32)
        # the segment grid: cells of the cutoff, each segment in the cells its own box meets, so a point's 27-cell
        # neighbourhood holds every segment within the cutoff of it (a closer segment cannot be two cells away)
        lo = np.minimum(A, A + AB); hi = np.maximum(A, A + AB)
        cs = 1.01 * self.gather_radius_m                         # a hair wider than the reach: float32 cell edges
        self._gmin = lo.min(0) - cs
        dims = np.maximum(1, np.ceil((hi.max(0) + cs - self._gmin) / cs).astype(int))
        loc = np.clip(np.floor((lo - self._gmin) / cs).astype(int), 0, dims - 1)
        hic = np.clip(np.floor((hi - self._gmin) / cs).astype(int), 0, dims - 1)
        cell, self._cap, _, _ = bucket_by_bbox(loc, hic, dims, None)
        self._CELL = jnp.asarray(cell, jnp.int32); self._dims = tuple(int(x) for x in dims)
        self._dims_arr = jnp.asarray(self._dims, jnp.int32); self._GMIN = jnp.asarray(self._gmin, jnp.float32)
        self._CS = jnp.float32(cs)
        self._OFF = jnp.asarray([[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)], jnp.int32)
        self._mean = self._domain_mean()
        self._batch = None
        if far is not None:                                      # the far grid on the device as STORED (float16 halves the
            self._FAR = jnp.asarray(np.asarray(far.values))      # traffic of the 64-tap read); passed to every jitted kernel
            self._FAR_O = jnp.asarray(np.asarray(far.origin_m, np.float32)); self._FAR_H = jnp.float32(far.spacing_m)
            self._FAR_N = jnp.asarray(np.asarray(far.shape, np.int32))     # as an ARGUMENT, never a captured constant:
            # a 1.7 GB constant closed over by a jit is embedded in every executable it lowers (minutes per compile,
            # a copy per shape), which made a walk with the grid slower than one without
        logging.getLogger("dmipy_sim").info("StrandFieldBasis: %d strands, %d segments, cutoff %.1f um, grid %s, up to %d segments per cell "
                                            "(%d candidates per point), closed form on the nearest %d",
                                            self.n_strands, self.n_segments, self.cutoff_m * 1e6, self._dims, self._cap,
                                            27 * self._cap, min(self.strands_max + 1, 27 * self._cap))

    # ------------------------------------------------------------------ the mean over the domain (closed form)
    def _domain_mean(self):
        """The 13 channels averaged over the domain: per strand, its length inside the domain times the uniform
        parts of its lumen and sheath fields (the cos 2 alpha terms average to zero around every strand)."""
        lo, hi = np.asarray(self.domain[0]), np.asarray(self.domain[1])
        V = float(np.prod(hi - lo))
        mean = np.zeros(13)
        for c, a, b in zip(self.centerlines, self.inner_radii, self.outer_radii):
            seg = c[1:] - c[:-1]; L = np.linalg.norm(seg, axis=1)
            w = _length_inside(c[:-1], seg, lo, hi)                                    # per segment, the length in the box
            keep = w > 0
            if not keep.any():
                continue
            u = seg[keep] / L[keep][:, None]; w = w[keep]
            UU = np.einsum("k,ki,kj->ij", w, u, u); Pt = np.eye(3) * w.sum() - UU              # length-weighted
            sym6 = lambda M: np.array([M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]])
            f_l = np.pi * a ** 2 / V; f_s = np.pi * (b ** 2 - a ** 2) / V
            mean[0] += f_s * w.sum() / 3.0
            mean[1:7] += f_s * 0.5 * sym6(Pt)
            mean[7:13] += f_l * 0.5 * np.log(b / a) * sym6(Pt) + f_s * ((0.5 * annulus_mean_log(a, b) - 5.0 / 18.0) * sym6(Pt) - sym6(UU) / 9.0)
        return mean

    @property
    def mean(self):
        """The 13-channel domain mean that :meth:`channels` subtracts."""
        return self._mean.copy()

    channel_names = CHANNEL_NAMES

    @property
    def meta(self):
        """What a pack records about this field source (JSON-ready)."""
        return dict(kind="strand_superposition", cutoff_m=self.cutoff_m, strands_max=self.strands_max, n_strands=self.n_strands,
                    n_segments=self.n_segments, domain=[list(map(float, self.domain[0])), list(map(float, self.domain[1]))],
                    channels=list(CHANNEL_NAMES), mean_subtracted=True, certificate=self.certificate,
                    far_grid=(None if self.far is None else self.far.meta))

    # ------------------------------------------------------------------ evaluation
    def nearest_device(self, *, radius_m=None, k=None):
        """The jitted, vmapped ``p -> (segments (k,), keep (k,), n_within)``: for a point, the nearest segment of
        every strand within ``radius_m`` (the cutoff by default; a walk gathers with a margin and reuses the list
        over several save intervals, masking by the true distance when it evaluates) -- global segment indices,
        the nearest ``k`` of them by distance (``strands_max + 1`` by default) with ``keep`` marking the real
        entries, and how many strands were within the radius (more than ``k``: the list is short). The radius must
        not exceed the grid's reach (the cutoff the grid was built for, plus a cell). What a walk gathers and
        evaluates the field against at every round (:mod:`dmipy_sim.engine.adaptive`)."""
        radius = float(self.gather_radius_m if radius_m is None else radius_m)
        if radius > 2.0 * self.gather_radius_m:
            raise ValueError(f"radius_m={radius * 1e6:.0f} um is beyond what the {self.gather_radius_m * 1e6:.0f} um grid gathers (a cell)")
        k_max = int(min(self.strands_max + 1 if k is None else int(k), 27 * self._cap))
        cache = getattr(self, "_nearest_batches", None)
        if cache is None:
            cache = self._nearest_batches = {}
        f = cache.get((radius, k_max))
        if f is None:
            A, AB, AB2, sid = self._A, self._AB, self._AB2, self._sid
            CELL, OFF, GMIN, CS, dims_arr, DIMS = self._CELL, self._OFF, self._GMIN, self._CS, self._dims_arr, self._dims
            cutoff = jnp.float32(radius); n_strands = self.n_strands; n_seg = self.n_segments
            tie_tol = jnp.float32(self.TIE_TOL * self.gather_radius_m)

            def one(p):
                c = jnp.clip(jnp.floor((p - GMIN) / CS).astype(jnp.int32), 0, dims_arr - 1)
                nb = jnp.clip(c[None, :] + OFF, 0, dims_arr - 1)
                cids = (nb[:, 0] * DIMS[1] + nb[:, 1]) * DIMS[2] + nb[:, 2]
                raw = CELL[cids].reshape(-1); valid = raw >= 0; cand = jnp.where(valid, raw, 0)
                # a segment crossing several of the 27 cells is gathered once per cell: keep one copy (the padding,
                # -1, sorts first and is never valid, so a real segment is never taken for its duplicate)
                order = jnp.argsort(raw); rs = raw[order]
                dup = jnp.zeros_like(valid).at[order[1:]].set(rs[1:] == rs[:-1])
                As = A[cand]; ABs = AB[cand]
                t = jnp.clip(((p[None, :] - As) * ABs).sum(1) / AB2[cand], 0.0, 1.0)
                Q = As + t[:, None] * ABs
                d = jnp.linalg.norm(p[None, :] - Q, axis=1)
                near = valid & ~dup & (d < cutoff)
                # the nearest segment of EVERY strand among the candidates (a per-strand minimum over all of them,
                # not over a bounded prefix); at a joint two segments tie -- to a tolerance, so that float32 rounding
                # does not pick either -- and the lower segment index is the one
                s_c = sid[cand]
                dmin = jax.ops.segment_min(jnp.where(near, d, jnp.inf), s_c, num_segments=n_strands)
                tie = near & (d <= dmin[s_c] + tie_tol)
                imin = jax.ops.segment_min(jnp.where(tie, cand, n_seg), s_c, num_segments=n_strands)
                first = tie & (cand == imin[s_c])
                _, pick = jax.lax.top_k(jnp.where(first, -d, -jnp.inf), k_max)
                return cand[pick], first[pick], first.sum()
            f = cache[(radius, k_max)] = jax.jit(jax.vmap(one))
        return f

    def _switch(self, d):
        """The far switch ``S(d)`` of the split: 0 within ``near_m - blend_m`` of a strand, 1 beyond ``near_m``, a cubic
        smoothstep between (jnp)."""
        x = jnp.clip((d - jnp.float32(self.far.near_m - self.far.blend_m)) / jnp.float32(self.far.blend_m), 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _far_at(self, p, FAR):
        """Tricubic (Catmull-Rom) read of the far grid ``FAR`` at ``p`` (jnp, one point): the far part carries the 1/r^2
        tails of the strands beyond the switch, whose curvature a trilinear read resolves only at a spacing far below
        the switch radius; the cubic's error falls as the fourth power of spacing over radius. Border nodes repeat."""
        g = (p - self._FAR_O) / self._FAR_H
        i0 = jnp.floor(g).astype(jnp.int32)
        f = g - i0.astype(jnp.float32)

        def cr(t):                                   # Catmull-Rom weights for the nodes i0-1, i0, i0+1, i0+2
            t2 = t * t; t3 = t2 * t
            return jnp.stack([-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2])
        wx, wy, wz = cr(f[0]), cr(f[1]), cr(f[2])
        ix = jnp.clip(i0[0] + jnp.arange(-1, 3), 0, self._FAR_N[0] - 1)
        iy = jnp.clip(i0[1] + jnp.arange(-1, 3), 0, self._FAR_N[1] - 1)
        iz = jnp.clip(i0[2] + jnp.arange(-1, 3), 0, self._FAR_N[2] - 1)
        cube = FAR[ix[:, None, None], iy[None, :, None], iz[None, None, :]].astype(jnp.float32)   # (4, 4, 4, 13)
        return jnp.einsum("a,b,c,abcd->d", wx, wy, wz, cube)

    def _channels_kernel(self, weight):
        """``(p, segments, keep) -> (13,)`` summing the kept segments within the gather radius with ``weight(d)``
        (``d`` the distance to the segment): the whole field, the near part ``1 - S``, or the far part ``S``."""
        A, AB, AB2, ra, rb, E0, E1 = self._A, self._AB, self._AB2, self._a, self._b, self._E0, self._E1
        reach = jnp.float32(self.gather_radius_m); taper = jnp.float32(self.END_TAPER_RADII)

        def one(p, seg, keep):
            As = A[seg]; ABs = AB[seg]
            L = jnp.sqrt(AB2[seg]); u = ABs / L[:, None]
            t_raw = ((p[None, :] - As) * ABs).sum(1) / AB2[seg]
            t = jnp.clip(t_raw, 0.0, 1.0)
            q = p[None, :] - (As + t[:, None] * ABs)
            d = jnp.linalg.norm(q, axis=1)
            within = keep & (d < reach)
            rv = q - (q * u).sum(1, keepdims=True) * u
            C = hollow_cylinder_basis(rv, u, ra[seg], rb[seg])
            # beyond a strand's free end the infinite cylinder is a fiction: its field tapers out over END_TAPER_RADII
            # outer radii of axial overshoot (a finite cylinder's field beyond its end falls off on that scale)
            over = jnp.where(E1[seg] & (t_raw > 1.0), (t_raw - 1.0) * L, 0.0) + jnp.where(E0[seg] & (t_raw < 0.0), -t_raw * L, 0.0)
            x = jnp.clip(over / (taper * rb[seg]), 0.0, 1.0)
            w_end = 1.0 - x * x * (3.0 - 2.0 * x)
            return (C * (within * weight(d) * w_end)[:, None]).sum(0)
        return one

    def channels_at_device(self):
        """The jitted, vmapped ``(p, segments, keep) -> (13,)``: the bare channels at ``p`` from the given segments
        (their infinite hollow cylinders, the radial vector taken at ``p``), summed over the kept ones whose
        segment lies within the gather radius of ``p`` (a list gathered with a margin is masked here); with a far
        grid, the near part ``(1 - S)`` of those plus the grid's far part at ``p``. No mean is subtracted here."""
        f = getattr(self, "_channels_at_batch", None)
        if f is None:
            if self.far is None:
                f = self._channels_at_batch = jax.jit(jax.vmap(self._channels_kernel(lambda d: 1.0)))
            else:
                near = self._channels_kernel(lambda d: 1.0 - self._switch(d)); far_at = self._far_at
                g = jax.jit(jax.vmap(lambda p, seg, keep, FAR: near(p, seg, keep) + far_at(p, FAR), in_axes=(0, 0, 0, None)))
                FAR = self._FAR
                f = self._channels_at_batch = lambda p, seg, keep: g(p, seg, keep, FAR)     # the grid as an argument
        return f

    def far_channels_at_device(self, near_m, blend_m):
        """The jitted, vmapped far part ``S``-weighted superposition at a point, from a basis WITHOUT a far grid (the
        full cutoff gathered): what :meth:`build_far_grid` tabulates."""
        if self.far is not None:
            raise ValueError("the far part is built from the plain superposition: a basis without a far grid")
        n0, b0 = jnp.float32(near_m - blend_m), jnp.float32(blend_m)

        def S(d):
            x = jnp.clip((d - n0) / b0, 0.0, 1.0)
            return x * x * (3.0 - 2.0 * x)
        return jax.jit(jax.vmap(self._channels_kernel(S)))

    def build_far_grid(self, spacing_m, near_m, *, blend_m=None, dtype=np.float16, chunk=None, all_strands=False):
        """The :class:`FarGrid` of this substrate at ``spacing_m`` over the domain: the ``S``-weighted superposition
        (``near_m``, ``blend_m`` default ``2 spacing_m``) at every node, summed to this basis's cutoff. The switch
        must start beyond the largest sheath radius plus two cells (refused otherwise: the sheath surface is a
        discontinuity no grid reads); beyond it the far part is 1/r^2 tails, and the tricubic read is good to a
        percent at a spacing of a third of the switch's start. ``all_strands`` sums every strand at every node (no
        cutoff: the exact far part, so the truncation certificate is moot; ``chunk x n_segments`` device memory per
        step). A one-off per substrate (DiSCo at 2.5 um: 64M nodes; a few minutes to an hour on a GH200)."""
        if self.far is not None:
            raise ValueError("build the far grid from the plain superposition (a basis without a far grid)")
        h = float(spacing_m); near = float(near_m); blend = float(2.0 * h if blend_m is None else blend_m)
        if not (0 < blend < near <= self.cutoff_m):
            raise ValueError("need 0 < blend_m < near_m <= cutoff_m")
        r_min = float(self.outer_radii.max()) + 2.0 * h
        if near - blend < r_min:
            raise ValueError(f"the switch starts at {(near - blend) * 1e6:.1f} um; it must start beyond the largest sheath radius plus two "
                             f"cells ({r_min * 1e6:.1f} um), or the far part carries the sheath surface, which no grid reads")
        lo, hi = np.asarray(self.domain[0], float), np.asarray(self.domain[1], float)
        n = np.maximum(2, np.ceil((hi - lo) / h - 1e-6).astype(int) + 1)      # the nodes cover the domain (a divisible extent exactly)
        origin = lo
        ijk = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), -1).reshape(-1, 3)
        P = (origin[None, :] + ijk * h).astype(np.float32)
        out = np.empty((P.shape[0], 13), np.float32)
        log = logging.getLogger("dmipy_sim")
        if all_strands:                                          # every strand, no cutoff: the exact far part
            at_all = self.far_channels_all_device(near, blend)
            chunk = int(chunk or max(16, int(1.5e9 // max(self.n_segments * 4 * 3, 1))))
            log.info("StrandFieldBasis.build_far_grid: %d nodes at %.2f um (near %.1f um, blend %.1f um), every strand, chunks of %d",
                     P.shape[0], h * 1e6, near * 1e6, blend * 1e6, chunk)
            for i in range(0, P.shape[0], chunk):
                Pc = P[i:i + chunk]; m = Pc.shape[0]
                if m < chunk:
                    Pc = np.concatenate([Pc, np.repeat(Pc[-1:], chunk - m, axis=0)])
                out[i:i + m] = np.asarray(at_all(jnp.asarray(Pc)))[:m]
            cutoff = float(np.linalg.norm(hi - lo))                 # what it summed to: the whole domain
        else:
            nearest = self.nearest_device(); at = self.far_channels_at_device(near, blend)
            if chunk is None:
                k_max = min(self.strands_max + 1, 27 * self._cap)
                chunk = max(64, int(2e9 // max(27 * self._cap * 48 + self.n_strands * 16 + k_max * 9 * 4 * 24, 1)))
            log.info("StrandFieldBasis.build_far_grid: %d nodes at %.2f um (near %.1f um, blend %.1f um) in chunks of %d",
                     P.shape[0], h * 1e6, near * 1e6, blend * 1e6, chunk)
            for i in range(0, P.shape[0], chunk):
                Pc = P[i:i + chunk]; m = Pc.shape[0]
                if m < chunk:
                    Pc = np.concatenate([Pc, np.repeat(Pc[-1:], chunk - m, axis=0)])
                Pd = jnp.asarray(Pc)
                seg, keep, cnt = nearest(Pd)
                if int(np.asarray(cnt)[:m].max()) > self.strands_max:
                    raise ValueError(f"a node has more than strands_max={self.strands_max} strands within the cutoff")
                out[i:i + m] = np.asarray(at(Pd, seg, keep))[:m]
            cutoff = self.cutoff_m
        values = out.reshape(tuple(n) + (13,)).astype(dtype)
        return FarGrid(tuple(float(x) for x in origin), h, values, near, blend, cutoff)

    def far_channels_all_device(self, near_m, blend_m):
        """The jitted far part at a chunk of points summed over EVERY strand (its nearest segment found among all of
        them): ``(chunk, 3) -> (chunk, 13)``. Memory ``chunk x n_segments`` floats a few times over; what the exact
        far grid is built with."""
        A, AB, AB2, ra, rb, sid, E0, E1 = self._A, self._AB, self._AB2, self._a, self._b, self._sid, self._E0, self._E1
        n_strands, n_seg = self.n_strands, self.n_segments
        n0, b0 = jnp.float32(near_m - blend_m), jnp.float32(blend_m); taper = jnp.float32(self.END_TAPER_RADII)
        tie_tol = jnp.float32(self.TIE_TOL * self.gather_radius_m)                # the gather's own tie rule

        def one(p):
            t_raw = ((p[None, :] - A) * AB).sum(1) / AB2
            t = jnp.clip(t_raw, 0.0, 1.0)
            d = jnp.linalg.norm(p[None, :] - (A + t[:, None] * AB), axis=1)                    # (n_seg,)
            dmin = jax.ops.segment_min(d, sid, num_segments=n_strands)                         # per strand
            tie = d <= dmin[sid] + tie_tol
            imin = jax.ops.segment_min(jnp.where(tie, jnp.arange(n_seg), n_seg), sid, num_segments=n_strands)   # (n_strands,)
            seg = imin
            As = A[seg]; ABs = AB[seg]; L = jnp.sqrt(AB2[seg]); u = ABs / L[:, None]
            tr = ((p[None, :] - As) * ABs).sum(1) / AB2[seg]; tc = jnp.clip(tr, 0.0, 1.0)
            q = p[None, :] - (As + tc[:, None] * ABs); ds = jnp.linalg.norm(q, axis=1)
            rv = q - (q * u).sum(1, keepdims=True) * u
            C = hollow_cylinder_basis(rv, u, ra[seg], rb[seg])
            x = jnp.clip((ds - n0) / b0, 0.0, 1.0); S = x * x * (3.0 - 2.0 * x)
            over = jnp.where(E1[seg] & (tr > 1.0), (tr - 1.0) * L, 0.0) + jnp.where(E0[seg] & (tr < 0.0), -tr * L, 0.0)
            xe = jnp.clip(over / (taper * rb[seg]), 0.0, 1.0); w_end = 1.0 - xe * xe * (3.0 - 2.0 * xe)
            return (C * (S * w_end)[:, None]).sum(0)
        return jax.jit(jax.vmap(one))

    def with_far(self, far):
        """The same strands read through ``far`` (a :class:`FarGrid` built for this substrate at this cutoff): the
        closed form within ``far.near_m`` of a point, the grid beyond."""
        if abs(float(far.cutoff_m) - self.cutoff_m) > 1e-12 * self.cutoff_m:
            raise ValueError(f"the far grid was built to a {far.cutoff_m * 1e6:.0f} um cutoff, this basis has {self.cutoff_m * 1e6:.0f} um")
        return StrandFieldBasis(self.centerlines, self.inner_radii, self.outer_radii, cutoff_m=self.cutoff_m, domain=self.domain,
                                certificate=self.certificate, strands_max=self.strands_max, far=far)

    def _build(self):
        nearest = self.nearest_device(); at = self.channels_at_device()

        def batch(P):                                            # not jitted as a whole: the far grid stays an argument
            seg, keep, n = nearest(P)
            return at(P, seg, keep), n
        return batch

    def channels(self, points, *, chunk=None):
        """``(n, 13)`` channels at ``points`` ``(n, 3)`` (metres), domain mean subtracted."""
        P = np.asarray(points, np.float32).reshape(-1, 3)
        if self._batch is None:
            self._batch = self._build()
        if chunk is None:                                       # ~2 GB of device arrays per chunk: the candidates'
            k_max = min(self.strands_max + 1, 27 * self._cap)   # distances, the per-strand minima, the closed form
            chunk = max(64, int(2e9 // max(27 * self._cap * 48 + self.n_strands * 16 + k_max * 9 * 4 * 24, 1)))
        out = np.empty((P.shape[0], 13), np.float64)
        logging.getLogger("dmipy_sim").debug("StrandFieldBasis.channels: %d points in chunks of %d", P.shape[0], chunk)
        for i in range(0, P.shape[0], chunk):
            Pc = P[i:i + chunk]; m = Pc.shape[0]
            if m < chunk:                                        # one static shape per basis: the tail is padded
                Pc = np.concatenate([Pc, np.repeat(Pc[-1:], chunk - m, axis=0)])
            c, n = self._batch(jnp.asarray(Pc))
            n = np.asarray(n)[:m]
            if (n > self.strands_max).any():
                raise ValueError(f"{int((n > self.strands_max).sum())} point(s) have more than strands_max={self.strands_max} "
                                 f"strands within the {self.cutoff_m * 1e6:.0f} um cutoff; raise strands_max= (or lower the cutoff)")
            out[i:i + m] = np.asarray(c, np.float64)[:m]
        return out - self._mean[None, :]

    def field(self, points, b0_dir, *, B0, chi_iso=0.0, chi_aniso=0.0):
        """``dB`` (Tesla) at ``points`` for one configuration."""
        return contract(self.channels(points), b0_dir, B0=B0, chi_iso=chi_iso, chi_aniso=chi_aniso)

    def with_cutoff(self, cutoff_m, *, certificate=None):
        """The same strands at another cutoff (``certificate`` records how that cutoff was chosen)."""
        return StrandFieldBasis(self.centerlines, self.inner_radii, self.outer_radii, cutoff_m=cutoff_m, domain=self.domain,
                                certificate=certificate, strands_max=self.strands_max,
                                far=(self.far if (self.far is not None and abs(float(cutoff_m) - self.cutoff_m) < 1e-12 * self.cutoff_m) else None))

    def cutoff_error(self, points):
        """The relative rms change of the channels at ``points`` when the cutoff doubles, per group:
        ``{"iso": ..., "aniso": ...}`` -- what the truncation costs on this substrate at these points. A producer
        doubles the cutoff until it is under its tolerance."""
        c1 = self.channels(points); c2 = self.with_cutoff(2.0 * self.cutoff_m).channels(points)
        d = (c1 - c2) - (c1 - c2).mean(0); r = c2 - c2.mean(0)
        rel = lambda sl: float(np.sqrt((d[:, sl] ** 2).sum() / max((r[:, sl] ** 2).sum(), 1e-300)))
        return {"iso": rel(slice(0, 7)), "aniso": rel(slice(7, 13))}
