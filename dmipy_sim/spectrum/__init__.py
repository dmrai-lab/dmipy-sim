"""The substrate spectrum (dmrai-lab/dmipy-sim#697, tessera#43's policy suite): generated substrates of every
engine class at controlled sizes -- strands (this package's first generator; analytic packs, label volumes,
sphere unions and meshes follow in later entries of ``GENERATORS``), rather than a bank collected one at a
time. Each generator writes one directory: a spec, a manifest + plan per walkable walk length
(:mod:`dmipy_sim.fill` reads these exactly as any other recipe), and a ``spectrum.json`` index of every file's
sha256 (:mod:`dmipy_sim.spectrum.common`). ``python -m dmipy_sim.spectrum`` is the CLI."""
from .common import EDGES_M, WALK_LENGTHS_S, SpectrumHaloError
from .strands import strands

#: ``{class name: generator(edge_m, seed, *, out_dir, **kw) -> dict}`` -- what ``--class`` picks from.
GENERATORS = {"strands": strands}

__all__ = ["EDGES_M", "WALK_LENGTHS_S", "SpectrumHaloError", "strands", "GENERATORS"]
