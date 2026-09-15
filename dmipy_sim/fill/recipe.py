"""The recipe of a fill: the dataset's ``manifest.json`` and what it names, resolved once from the hub.

A manifest holds (the DiSCo replay dataset is the reference):

- ``id``, ``license``, ``citation``: what every shard's pack declares;
- ``code.commit``: the dmipy-sim commit every worker runs;
- ``substrate``: ``kind`` (``"disco"``: the strands' ``tracks`` + inner ``diameters`` with their units and the
  domain's ``side_m``), an optional ``far_grid`` (``.npy`` with its ``.json`` beside it);
- ``variants``: named walks of the substrate (``field`` on or off), ``default_variant``;
- ``grid``: the voxel grid a shard's certificate is on (``shape``, ``voxel_size_m``, ``origin_m``);
- ``walk``: ``T_max_s``, ``dt_save_s``, ``scanner``, ``walker_batch_size``, ``adaptive_steps``, ``floor_fraction``,
  ``field_sample_every``, ``field_gather_every``, ``census_draws``;
- ``pack``: ``K``, ``K_path``, ``blt_K``, ``position_container``, ``blt_container``;
- ``plan``: ``file`` (an ``.npz`` of ``count_<pool>`` arrays on the grid), ``blocks`` (a JSON table of voxel
  boxes with their seeds), optional ``passes`` (each ``{pass, scale}``: a share of every block's counts, filled
  as its own shard from its own seed stream).

The spec cites the track file by its dataset path; the walk resolves it under ``DMIPY_SIM_SURFACE_DIR``, which
the recipe sets to the recipe's root (the hub snapshot, or the fake hub's directory)."""
import json
import logging
import os
import time

import numpy as np

FULL = {"pass": None, "scale": 1.0}   # the whole block in one shard: a plan without passes, and every certifying walk
log = logging.getLogger("dmipy_sim.fill")


class Recipe:
    """The manifest and what it names, resolved once; the block table; the plan's counts per voxel."""

    def __init__(self, hub, variant=None):
        self.hub = hub
        self.man = json.load(open(hub.get("manifest.json")))
        self.variant = variant or self.man["default_variant"]
        self.V = self.man["variants"][self.variant]
        self.table = json.load(open(hub.get(self.man["plan"]["blocks"])))["blocks"]
        self.passes = self.man["plan"].get("passes") or [FULL]
        self.plan = np.load(hub.get(self.man["plan"]["file"]))
        S = self.man["substrate"]
        if S.get("kind", "disco") != "disco":
            raise ValueError(f"substrate kind {S.get('kind')!r}: the fill knows 'disco' (strands from a track file)")
        self.tracks, self.diam = hub.get(S["tracks"]), hub.get(S["diameters"])
        fg = S.get("far_grid")
        self.far_grid = None
        if fg:
            self.far_grid = hub.get(fg); hub.get(fg[:-4] + ".json")           # the grid's .json beside its .npy
        os.environ["DMIPY_SIM_SURFACE_DIR"] = os.path.dirname(os.path.dirname(os.path.abspath(self.tracks)))
        self._spec = None; self._context = None

    @property
    def commit(self):
        return self.man["code"]["commit"]

    def spec(self):
        """The substrate spec of the recipe, parsed once."""
        if self._spec is None:
            from ..spec import disco_spec
            S = self.man["substrate"]
            self._spec = disco_spec(self.tracks, self.diam, coordinate_unit_m=S["coordinate_unit_m"], diameter_unit_m=S["diameter_unit_m"],
                                    side_m=S["side_m"], field=bool(self.V["field"]), cite_tracks_as=S["tracks"])
        return self._spec

    def context(self):
        """The walk context (:class:`~dmipy_sim.spec.WalkContext`): the spec's pool tests, boundaries, walking
        geometries and the strand-field basis with the far grid, built once per process and kept from block to
        block -- every block of a fill walks the same spec, so nothing but the seeds is built again."""
        if self._context is None:
            from ..spec import WalkContext
            spec = self.spec(); t0 = time.time()
            self._context = WalkContext(spec, field_far=(self.far_grid if spec.field_source_pools else None))
            log.info("walk context built in %.0f s: key %s", time.time() - t0, self._context.key)
        return self._context

    def grid(self):
        from ..phantom import Grid
        G = self.man["grid"]
        return Grid(shape=tuple(G["shape"]), voxel_size_m=tuple(G["voxel_size_m"]), origin_m=tuple(G["origin_m"]))

    def counts(self, row, budget=None, pass_scale=1.0):
        """The plan's counts per pool in the block's voxels at the pass's scale, then scaled down to ``budget``
        walkers when given: ``(want, planned, scale)`` with ``want`` ``{pool name: counts on the grid}``. A voxel
        the plan gives any walker keeps at least two at any scale."""
        grid = self.grid()
        inblock = np.zeros(grid.shape, bool)
        inblock[row["i"][0]:row["i"][1], row["j"][0]:row["j"][1], row["k"][0]:row["k"][1]] = True
        counts = {k[6:]: np.where(inblock, self.plan[k].astype(np.int64), 0) for k in self.plan.files if k.startswith("count_")}
        tot = sum(int(c.sum()) for c in counts.values())
        scale = pass_scale * (1.0 if budget is None else min(1.0, budget / max(tot * pass_scale, 1)))
        want = {n: np.where(c > 0, np.maximum(np.ceil(c * scale), 2), 0).astype(np.int64) for n, c in counts.items()}
        return want, tot, scale

    def rounds(self, row, budget, P, max_walkers):
        """How many rounds a block is walked in under ``max_walkers`` (one without a limit; never more than the
        smallest pool's count)."""
        want, _, _ = self.counts(row, budget, P["scale"])
        total = sum(int(v.sum()) for v in want.values())
        k = 1 if max_walkers is None else max(1, int(np.ceil(total / max_walkers)))
        return min(k, min(int(v.sum()) for v in want.values() if v.sum() > 0))

    def dt_save(self):
        from ..acquisition.scanners import save_interval
        spec = self.spec(); W = self.man["walk"]; dt = float(W["dt_save_s"])
        if spec.field_source_pools:
            dt = min(dt, save_interval(W["T_max_s"], 8, W["scanner"], D=max(float(p.D) for p in spec.pools if p.D), field=True))
        return dt
