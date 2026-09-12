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


class StrandFieldBasis:
    """The field basis of sheathed strands, evaluated at points.

    ``centerlines``: a list of ``(P_i, 3)`` polylines (metres); ``inner_radii`` / ``outer_radii``: per strand
    (metres), the axolemma and the sheath's outer surface; ``cutoff_m``: strands whose nearest segment lies
    farther than this from a point are not summed; ``domain``: ``(lo, hi)`` of the walked box, over which the
    mean is taken (defaults to the strands' bounding box).
    """

    def __init__(self, centerlines, inner_radii, outer_radii, *, cutoff_m, domain=None, certificate=None):
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
        self.certificate = None if certificate is None else dict(certificate)
        if not self.cutoff_m > 0:
            raise ValueError("cutoff_m must be positive")
        A, AB, sid = [], [], []
        for k, c in enumerate(self.centerlines):
            A.append(c[:-1]); AB.append(c[1:] - c[:-1]); sid.append(np.full(len(c) - 1, k))
        A = np.vstack(A); AB = np.vstack(AB); sid = np.concatenate(sid)
        self.n_strands = len(self.centerlines); self.n_segments = int(A.shape[0])
        self.domain = (tuple(np.minimum(A, A + AB).min(0) - self.outer_radii.max()), tuple(np.maximum(A, A + AB).max(0) + self.outer_radii.max())) \
            if domain is None else (tuple(np.asarray(domain[0], float)), tuple(np.asarray(domain[1], float)))
        self._A = jnp.asarray(A, jnp.float32); self._AB = jnp.asarray(AB, jnp.float32)
        self._AB2 = jnp.asarray(np.maximum((AB ** 2).sum(1), 1e-30), jnp.float32)
        self._sid = jnp.asarray(sid, jnp.int32)
        self._a = jnp.asarray(self.inner_radii[sid], jnp.float32); self._b = jnp.asarray(self.outer_radii[sid], jnp.float32)
        # the segment grid: cells of the cutoff, so the 27-cell neighbourhood covers the cutoff ball of any point
        lo = np.minimum(A, A + AB) - self.cutoff_m; hi = np.maximum(A, A + AB) + self.cutoff_m
        cs = self.cutoff_m
        self._gmin = lo.min(0)
        dims = np.maximum(1, np.ceil((hi.max(0) - self._gmin) / cs).astype(int))
        loc = np.clip(np.floor((lo - self._gmin) / cs).astype(int), 0, dims - 1)
        hic = np.clip(np.floor((hi - self._gmin) / cs).astype(int), 0, dims - 1)
        cell, self._cap, _, _ = bucket_by_bbox(loc, hic, dims, None)
        self._CELL = jnp.asarray(cell, jnp.int32); self._dims = tuple(int(x) for x in dims)
        self._dims_arr = jnp.asarray(self._dims, jnp.int32); self._GMIN = jnp.asarray(self._gmin, jnp.float32)
        self._CS = jnp.float32(cs)
        self._OFF = jnp.asarray([[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)], jnp.int32)
        self._mean = self._domain_mean()
        self._batch = None

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
        return dict(kind="strand_superposition", cutoff_m=self.cutoff_m, n_strands=self.n_strands,
                    n_segments=self.n_segments, domain=[list(map(float, self.domain[0])), list(map(float, self.domain[1]))],
                    channels=list(CHANNEL_NAMES), mean_subtracted=True, certificate=self.certificate)

    # ------------------------------------------------------------------ evaluation
    def _build(self):
        A, AB, AB2, sid, ra, rb = self._A, self._AB, self._AB2, self._sid, self._a, self._b
        CELL, OFF, GMIN, CS, dims_arr, DIMS = self._CELL, self._OFF, self._GMIN, self._CS, self._dims_arr, self._dims
        cutoff = jnp.float32(self.cutoff_m); key_scale = jnp.float32(0.25 / self.cutoff_m)
        n_cand = 27 * self._cap

        def one(p):
            c = jnp.clip(jnp.floor((p - GMIN) / CS).astype(jnp.int32), 0, dims_arr - 1)
            nb = jnp.clip(c[None, :] + OFF, 0, dims_arr - 1)
            cids = (nb[:, 0] * DIMS[1] + nb[:, 1]) * DIMS[2] + nb[:, 2]
            cand = CELL[cids].reshape(-1); valid = cand >= 0; cand = jnp.where(valid, cand, 0)
            As = A[cand]; ABs = AB[cand]
            t = jnp.clip(((p[None, :] - As) * ABs).sum(1) / AB2[cand], 0.0, 1.0)
            Q = As + t[:, None] * ABs
            d = jnp.linalg.norm(p[None, :] - Q, axis=1)
            near = valid & (d < cutoff)
            # the nearest segment of each strand: sort by (strand, distance), keep the first of each run
            key = jnp.where(near, sid[cand].astype(jnp.float32) + d * key_scale, jnp.inf)
            order = jnp.argsort(key)
            ks = key[order]; ss = jnp.where(jnp.isfinite(ks), jnp.floor(ks), -1.0)
            first = jnp.concatenate([jnp.array([True]), ss[1:] != ss[:-1]]) & jnp.isfinite(ks)
            # duplicated cell entries of the same segment collapse too (the same key, the first survives)
            pick = order
            u = ABs[pick] / jnp.sqrt(AB2[cand][pick])[:, None]
            rv = p[None, :] - Q[pick]; rv = rv - (rv * u).sum(1, keepdims=True) * u
            C = hollow_cylinder_basis(rv, u, ra[cand][pick], rb[cand][pick])
            return (C * first[:, None]).sum(0)

        return jax.jit(jax.vmap(one))

    def channels(self, points, *, chunk=None):
        """``(n, 13)`` channels at ``points`` ``(n, 3)`` (metres), domain mean subtracted."""
        P = np.asarray(points, np.float32).reshape(-1, 3)
        if self._batch is None:
            self._batch = self._build()
        if chunk is None:
            chunk = max(256, int(4e8 // max(27 * self._cap * 64, 1)))
        out = np.empty((P.shape[0], 13), np.float64)
        for i in range(0, P.shape[0], chunk):
            out[i:i + chunk] = np.asarray(self._batch(jnp.asarray(P[i:i + chunk])), np.float64)
        return out - self._mean[None, :]

    def field(self, points, b0_dir, *, B0, chi_iso=0.0, chi_aniso=0.0):
        """``dB`` (Tesla) at ``points`` for one configuration."""
        return contract(self.channels(points), b0_dir, B0=B0, chi_iso=chi_iso, chi_aniso=chi_aniso)

    def with_cutoff(self, cutoff_m, *, certificate=None):
        """The same strands at another cutoff (``certificate`` records how that cutoff was chosen)."""
        return StrandFieldBasis(self.centerlines, self.inner_radii, self.outer_radii, cutoff_m=cutoff_m, domain=self.domain,
                                certificate=certificate)

    def cutoff_error(self, points):
        """The relative rms change of the channels at ``points`` when the cutoff doubles, per group:
        ``{"iso": ..., "aniso": ...}`` -- what the truncation costs on this substrate at these points. A producer
        doubles the cutoff until it is under its tolerance."""
        c1 = self.channels(points); c2 = self.with_cutoff(2.0 * self.cutoff_m).channels(points)
        d = (c1 - c2) - (c1 - c2).mean(0); r = c2 - c2.mean(0)
        rel = lambda sl: float(np.sqrt((d[:, sl] ** 2).sum() / max((r[:, sl] ** 2).sum(), 1e-300)))
        return {"iso": rel(slice(0, 7)), "aniso": rel(slice(7, 13))}
