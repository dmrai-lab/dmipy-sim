"""The imaging prescription: where in the bore the acquisition images, and on what voxels.

A :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence` is what the scanner does in **time**. The
prescription is its companion in **space**: the isocenter, the axes the gradient and B0 directions are given
in, the voxel size and the matrix. It derives nothing and nothing is derived from it (ACQUISITION.md 3.6); a
phantom whose voxels are the scanner's reads its grid from here (:meth:`dmipy_sim.phantom.Grid.from_prescription`),
and a phantom with its own grid checks that the two agree on the axes rather than rotating silently.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Prescription"]

_AXES = "RLAPSI"

#: the patient-frame direction each axis letter names, in RAS: the one language a prescription's axes, a
#: grid's axes and a scanner's catalogued axes are all spoken in.
_AXIS_LETTER = {"R": (1.0, 0.0, 0.0), "L": (-1.0, 0.0, 0.0),
                "A": (0.0, 1.0, 0.0), "P": (0.0, -1.0, 0.0),
                "S": (0.0, 0.0, 1.0), "I": (0.0, 0.0, -1.0)}


def index_directions(axes):
    """The ``3 x 3`` signed permutation whose column ``a`` is the unit direction index ``a`` runs along, for an
    axes string such as ``"LAS"`` (the first index runs along -x). A string that names one direction twice, or
    a letter that is no direction, is refused."""
    ax = str(axes).upper()
    if len(ax) != 3 or any(c not in _AXES for c in ax) or len({_AXES.index(c) // 2 for c in ax}) != 3:
        raise ValueError(f"axes names the scanner direction of each index, one of R/L, A/P, S/I each; got {axes!r}")
    return np.array([_AXIS_LETTER[c] for c in ax], dtype=np.float64).T


@dataclass(frozen=True)
class Prescription:
    """``matrix`` voxels of ``voxel_size_m`` along the scanner ``axes`` (``"RAS"``: the first index runs along
    +x, the second along +y, the third along +z), ``isocenter_m`` the point the scanner is focused on, and
    ``origin_m`` the scanner coordinate of the **centre** of voxel ``(0, 0, 0)`` -- by default the field of view
    is centred on the isocenter, whichever way each index runs. Every argument is a keyword."""

    isocenter_m: tuple
    voxel_size_m: tuple
    matrix: tuple
    axes: str = "RAS"
    origin_m: tuple = None

    def __init__(self, *, isocenter_m=(0.0, 0.0, 0.0), voxel_size_m, matrix, axes="RAS", origin_m=None):
        iso = tuple(float(v) for v in isocenter_m)
        vs = tuple(float(v) for v in voxel_size_m)
        mx = tuple(int(v) for v in matrix)
        if len(iso) != 3 or len(vs) != 3 or len(mx) != 3:
            raise ValueError("isocenter_m, voxel_size_m and matrix are three-dimensional")
        if min(vs) <= 0 or min(mx) < 1:
            raise ValueError(f"voxel_size_m must be positive lengths and matrix positive counts: {vs}, {mx}")
        ax = str(axes).upper()
        D = index_directions(ax)
        centred = np.asarray(iso) - 0.5 * D @ ((np.asarray(mx) - 1) * np.asarray(vs))
        org = tuple(float(v) for v in (centred if origin_m is None else origin_m))
        if len(org) != 3:
            raise ValueError("origin_m is a scanner coordinate: three lengths")
        for k, v in (("isocenter_m", iso), ("voxel_size_m", vs), ("matrix", mx), ("axes", ax), ("origin_m", org)):
            object.__setattr__(self, k, v)

    @property
    def fov_m(self):
        """The field of view along each index, ``matrix * voxel_size_m``."""
        return tuple(n * d for n, d in zip(self.matrix, self.voxel_size_m))

    def to_dict(self):
        return {"isocenter_m": list(self.isocenter_m), "voxel_size_m": list(self.voxel_size_m),
                "matrix": list(self.matrix), "axes": self.axes, "origin_m": list(self.origin_m)}

    @classmethod
    def from_dict(cls, d):
        return cls(isocenter_m=d["isocenter_m"], voxel_size_m=d["voxel_size_m"], matrix=d["matrix"],
                   axes=d.get("axes", "RAS"), origin_m=d.get("origin_m"))
