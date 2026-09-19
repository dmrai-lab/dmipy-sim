"""What the machine does to a phantom, as a function of where each voxel sits in the bore.

A magnet's field is not uniform, and the way it departs from uniformity is a property of the MACHINE while
where the sample sits in it is a property of the GRID. This module is the one place the two meet: it renders
a catalogued field law (:meth:`~dmipy_sim.acquisition.scanners.ScannerLimits.b0_offset`) onto a grid and
hands back what a replay already accepts as ``off_resonance``.

Nothing here is a new kind of input. A phantom has taken a callable of scanner coordinates for its
macroscopic layers all along; this supplies one that a machine, rather than a person, is the author of.
"""
from __future__ import annotations

import numpy as np

__all__ = ["b0_offset_map"]


def b0_offset_map(scanner, grid, *, to_scanner=None):
    """A callable giving the static field's departure from uniformity, in tesla, at each voxel of ``grid``.

    Pass the result straight to ``off_resonance=`` of any :meth:`~dmipy_sim.phantom.Phantom.replay`; it has
    the signature a phantom already evaluates for a layer, ``f(positions_m) -> (n,)``.

    ``to_scanner`` is the rotation taking the grid's axes to the scanner's -- the ``R`` that
    :meth:`~dmipy_sim.phantom.Grid.from_oblique_affine` returns. It is REQUIRED for an oblique grid and must
    be omitted for an axis-aligned one, because a field law is a function of position in the BORE and an
    oblique grid's coordinates are not the bore's. Leaving it out does not fail: it silently evaluates the
    law at the wrong place, and for a law with a preferred direction -- which a single-yoke magnet's is --
    it gets the sign of the asymmetry wrong on part of the volume.

    ``None`` when the machine publishes no profile, which is every machine but a permanent-magnet one;
    ``off_resonance=None`` is then exactly right, being what a replay already means by "no field offset".
    """
    if getattr(scanner, "b0_quadratic", None) is None:
        return None
    iso = np.asarray(grid.isocenter_m, dtype=np.float64)
    R = None if to_scanner is None else np.asarray(to_scanner, dtype=np.float64)
    if R is not None:
        if R.shape != (3, 3):
            raise ValueError(f"to_scanner is the 3x3 rotation taking grid axes to scanner axes; got {R.shape}")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
            raise ValueError("to_scanner is not a rotation: a grid's axes are orthonormal in the scanner")

    def field(positions_m):
        d = np.asarray(positions_m, dtype=np.float64).reshape(-1, 3) - iso
        if R is not None:
            d = d @ R.T                      # the grid's frame into the bore's, where the law is stated
        return scanner.b0_offset(d)

    return field
