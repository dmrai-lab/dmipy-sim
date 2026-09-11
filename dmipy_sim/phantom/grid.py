"""The voxel grid, placed in the scanner (RPH.md 3, 7)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Grid"]


@dataclass(frozen=True)
class Grid:
    """``shape`` voxels of ``voxel_size_m``; ``origin_m`` is the scanner coordinate of the **centre** of voxel
    ``(0, 0, 0)`` and ``isocenter_m`` the point the scanner is focused on. ``axes`` names the scanner axes the
    indices run along (``"RAS"``: i -> +x, j -> +y, k -> +z), and is the frame the acquisition's gradient and
    B0 directions are given in. Both positions default to a grid centred on the isocenter at the origin.

    ``attach`` says what the grid is welded to when a phantom is **partitioned** from one walk and then posed
    (RPH.md, spec#3): ``"substrate"`` -- the grid follows the tissue, so a pose changes the physics only and
    voxel membership is invariant; ``"lab"`` -- the grid is the bore's, so a pose reassigns walkers to voxels.
    The two coincide at the identity pose, and a composed phantom never poses, so it is inert there.

    Every argument is a keyword: ``Grid(shape=(40, 40, 1), voxel_size_m=(1.5e-3,) * 3)``.
    """

    shape: tuple
    voxel_size_m: tuple
    origin_m: tuple = None
    isocenter_m: tuple = None
    axes: str = "RAS"
    attach: str = "substrate"

    def __init__(self, *, shape, voxel_size_m, origin_m=None, isocenter_m=None, axes="RAS", attach="substrate"):
        sh = tuple(int(v) for v in shape)
        vs = tuple(float(v) for v in voxel_size_m)
        if len(sh) != 3 or len(vs) != 3:
            raise ValueError(f"a grid is three-dimensional: got shape {shape}, voxel_size_m {voxel_size_m}")
        if min(sh) < 1 or min(vs) <= 0:
            raise ValueError(f"shape must be positive integers and voxel_size_m positive lengths: {sh}, {vs}")
        org = tuple(-0.5 * (n - 1) * d for n, d in zip(sh, vs)) if origin_m is None else tuple(float(v) for v in origin_m)
        iso = tuple(o + 0.5 * (n - 1) * d for o, n, d in zip(org, sh, vs)) if isocenter_m is None \
            else tuple(float(v) for v in isocenter_m)
        if len(org) != 3 or len(iso) != 3:
            raise ValueError("origin_m and isocenter_m are scanner coordinates: three lengths each")
        ax = str(axes).upper()
        if len(ax) != 3 or any(c not in "RLAPSI" for c in ax):
            raise ValueError(f"axes names the scanner direction of each index, one of R/L, A/P, S/I each; got {axes!r}")
        object.__setattr__(self, "shape", sh); object.__setattr__(self, "voxel_size_m", vs)
        object.__setattr__(self, "origin_m", org); object.__setattr__(self, "isocenter_m", iso)
        object.__setattr__(self, "axes", ax)
        if attach not in ("substrate", "lab"):
            raise ValueError(f"attach is 'substrate' (the grid follows the tissue) or 'lab' (the grid is the bore's); got {attach!r}")
        object.__setattr__(self, "attach", attach)

    # ---- construction from elsewhere ---------------------------------------------------------------------
    @classmethod
    def from_affine(cls, affine, shape, *, isocenter_m=None):
        """The grid of a NIfTI image: ``affine`` maps a voxel index to the scanner coordinate of its centre, in
        **millimetres** (the NIfTI convention). An axis-aligned affine gives the voxel size, the origin and the
        axes; an oblique one is refused rather than resampled, because the phantom's voxels are the image's."""
        A = np.asarray(affine, np.float64)
        if A.shape != (4, 4):
            raise ValueError(f"an affine is 4 x 4; got {A.shape}")
        M = A[:3, :3]
        off = M - np.diag(np.diag(M))
        if np.abs(off).max() > 1e-6 * max(np.abs(np.diag(M)).max(), 1e-30):
            raise ValueError("the affine is oblique (its rotation block is not diagonal): a phantom's grid is "
                             "axis-aligned in the scanner, so resample the image first or declare the grid "
                             "yourself rather than have it silently straightened")
        d = np.diag(M)
        if np.any(d == 0):
            raise ValueError("the affine has a zero voxel size")
        axes = "".join(("R" if d[0] > 0 else "L", "A" if d[1] > 0 else "P", "S" if d[2] > 0 else "I"))
        return cls(shape=tuple(int(v) for v in shape)[:3], voxel_size_m=tuple(abs(float(v)) * 1e-3 for v in d),
                   origin_m=tuple(float(v) * 1e-3 for v in A[:3, 3]), isocenter_m=isocenter_m, axes=axes)

    @classmethod
    def from_prescription(cls, prescription, *, attach="lab"):
        """The voxels a prescribed acquisition images (:class:`~dmipy_sim.acquisition.prescription.Prescription`):
        its matrix, voxel size, origin, isocenter and axes, as a grid. The scanner's voxels are the bore's, so
        ``attach`` defaults to ``"lab"``."""
        p = prescription
        return cls(shape=p.matrix, voxel_size_m=p.voxel_size_m, origin_m=p.origin_m, isocenter_m=p.isocenter_m,
                   axes=p.axes, attach=attach)

    @classmethod
    def covering(cls, positions_m, *, voxel_size_m, isocenter_m=None, axes="RAS", attach="substrate", margin_voxels=0):
        """The smallest grid of ``voxel_size_m`` voxels whose faces enclose every position ``(N, 3)``, plus
        ``margin_voxels`` on every side: what a partition of one walk uses when no grid is prescribed."""
        P = np.asarray(positions_m, np.float64).reshape(-1, 3)
        vs = np.asarray(voxel_size_m, np.float64).reshape(3)
        lo, hi = P.min(axis=0), P.max(axis=0)
        n = np.maximum(1, np.ceil((hi - lo) / vs - 1e-9).astype(int)) + 2 * int(margin_voxels)
        centre = 0.5 * (lo + hi)
        origin = centre - 0.5 * (n - 1) * vs
        return cls(shape=tuple(int(v) for v in n), voxel_size_m=tuple(vs), origin_m=tuple(origin), isocenter_m=isocenter_m,
                   axes=axes, attach=attach)

    @classmethod
    def from_meta(cls, meta):
        g = meta["grid"] if "grid" in meta else meta
        return cls(shape=g["shape"], voxel_size_m=g["voxel_size_m"], origin_m=g.get("origin_m"),
                   isocenter_m=g.get("isocenter_m"), axes=g.get("axes", g.get("frame", "RAS")),
                   attach=g.get("attach", "substrate"))

    def to_meta(self):
        # RPH.md 0.4 spells the axes ``frame``; the reader accepts either
        return {"shape": list(self.shape), "voxel_size_m": list(self.voxel_size_m),
                "origin_m": list(self.origin_m), "isocenter_m": list(self.isocenter_m), "frame": self.axes,
                "attach": self.attach}

    # ---- geometry ---------------------------------------------------------------------------------------------
    @property
    def n_voxels(self):
        return int(np.prod(self.shape))

    def positions_m(self, voxel_index):
        """Scanner coordinates of the centres of the given voxels, ``(N, 3)``."""
        return np.asarray(self.origin_m) + np.asarray(voxel_index, np.float64) * np.asarray(self.voxel_size_m)

    def radius_m(self, voxel_index):
        """Distance of each voxel centre from the isocenter: what a macroscopic layer varies over."""
        return np.linalg.norm(self.positions_m(voxel_index) - np.asarray(self.isocenter_m), axis=-1)

    @property
    def corner_m(self):
        """The scanner coordinate of the low corner of voxel ``(0, 0, 0)``: the faces start here."""
        return tuple(o - 0.5 * d for o, d in zip(self.origin_m, self.voxel_size_m))

    @property
    def extent_m(self):
        return tuple(n * d for n, d in zip(self.shape, self.voxel_size_m))

    def with_voxel_size(self, voxel_size_m):
        """The same field of view (the same low corner, at least the same extent) on other voxels: an exact
        rebin of anything binned on this grid whenever the new size divides the old."""
        vs = tuple(float(v) for v in voxel_size_m)
        n = tuple(int(np.ceil(e / d - 1e-9)) for e, d in zip(self.extent_m, vs))
        origin = tuple(c + 0.5 * d for c, d in zip(self.corner_m, vs))
        return Grid(shape=n, voxel_size_m=vs, origin_m=origin, isocenter_m=self.isocenter_m, axes=self.axes,
                    attach=self.attach)

    def bin(self, positions_m):
        """The voxel each position falls in, ``(N, 3)`` int, and whether it is inside the grid, ``(N,)`` bool:
        ``floor((r - corner) / voxel_size)``, the derived partition of RPH.md."""
        P = np.asarray(positions_m, np.float64).reshape(-1, 3)
        ijk = np.floor((P - np.asarray(self.corner_m)) / np.asarray(self.voxel_size_m)).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < np.asarray(self.shape)), axis=1)
        return ijk, inside

    def check_volume(self, a, name):
        """``a`` as a float64 array of exactly this grid's shape, else a ValueError naming ``name``."""
        a = np.asarray(a, np.float64)
        if a.shape != tuple(self.shape):
            raise ValueError(f"{name} has shape {a.shape}, not the grid's {tuple(self.shape)}")
        return a
