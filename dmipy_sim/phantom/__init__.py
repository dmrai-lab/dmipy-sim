"""``dmipy_sim.phantom``: replay phantoms, built from volumes and replayed as volumes (RPH.md).

* :class:`Grid` -- the voxels, placed in the scanner.
* :class:`PackSubstrate`, :class:`FreeWater`, :class:`Inert` -- what a voxel may hold; each carries its own name
  and proton density, and a phantom is keyed by these objects.
* :class:`Peaks`, :class:`ODF`, :class:`Watson`, :class:`Frames`, :class:`Fan` -- a pose per voxel.
* :class:`Phantom` -- :meth:`~Phantom.compose` the above, :meth:`~Phantom.replay` under a ``ScannerSequence``,
  :meth:`~Phantom.write` / :meth:`~Phantom.read` the ``.rph`` file.
"""
from .grid import Grid
from .substrates import AnalyticSubstrate, FreeWater, Inert, PackSubstrate
from .orientation import ODF, Fan, Frames, Peaks, Watson, SH_BASES
from .phantom import Phantom

__all__ = ["Phantom", "Grid", "PackSubstrate", "FreeWater", "Inert", "AnalyticSubstrate",
           "Peaks", "ODF", "Watson", "Frames", "Fan", "SH_BASES"]
