# Changelog

## 2.1.0

**Triangular-mesh substrates.** The forward engine now walks arbitrary meshes with the same
noise-floor accuracy as the analytic geometries, plus observability tooling to see it.

### Added
- **Mesh geometry + PLY loading** (`dmipy_sim.Mesh`, `Mesh.from_ply` / `load_ply`; `[mesh]`
  extra). Arbitrary closed **or 3-D-periodic** triangular meshes run through the
  analytic-geometry Monte-Carlo engine:
  - **Uniform-grid broad phase** — each step tests only the walker's 27-cell triangle
    neighbourhood, so million-triangle meshes are tractable (exact when `cell_size ≥ max step`).
  - **3-D periodicity** via ghost-triangle replication; geometry queries use the wrapped
    position while the returned position stays continuous, so the gradient phase is correct.
  - **Smooth vertex-normal reflection** (`O(h²/R²)` faceting) and **leak-proof Powles
    permeation** (one crossing decision at the first hit, then multi-bounce reflection).
  - **Bore placement** (`orientation=`/`R=`) as an acquisition rotation (B0 = +z); the walk
    stays in the mesh frame.
  - **Coarseness guard** — `Mesh.quality_report()` and a construction warning flag a mesh too
    coarse for the requested effect (`edge/feature ≲ 0.04` for permeability).
- **Per-compartment wall & bulk properties** (`intra=` / `extra=` dicts):
  - **Side-dependent surface relaxivity** and **direction-dependent permeability**
    (`{"intra_to_extra": …, "extra_to_intra": …}`; scalar = symmetric, the default).
    *Caveat:* asymmetric κ breaks detailed balance — it is a pump, not passive exchange.
  - **Per-compartment bulk D / T2** — resolved per sub-step, so an aggregate step carries the
    fractional intra/extra occupancy. Unequal D across a permeable wall is rejected.
- **Trajectory export** — `simulate(..., return_positions='full', return_compartments='full')`
  returns full `(n_walkers, n_timesteps, 3)` paths and per-step compartment tags.
- **Mesh visualisation** (`dmipy_sim.viz`): `plot_mesh_section`, `plot_mesh_3d`,
  `plot_cell_surface`, `plot_walkers_3d`, `walk_paths` + `plot_trajectories`, and
  `save_rotation` (animated GIF). Rendered gallery in `examples/mesh_viz/`.
- **Agent guide** (`CLAUDE.md`) — the operational contract for driving the engine.

### Changed
- Public docstrings reference only the public engine (public-safety pass).
- CI publishes to PyPI only on `v*` release tags.

### Notes
- No mesh files ship in the repo — tests generate meshes on the fly (icosphere / open tube);
  large research PLYs are a manual stress test only.
- Per-compartment **T1** and the PGSTE coherence gating are **not** in this release (tracked on
  a separate branch for a later version).

## 2.0.0

First public release of the dmipy-sim forward Monte-Carlo engine.
