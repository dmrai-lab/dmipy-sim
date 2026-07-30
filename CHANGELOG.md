# Changelog

## Unreleased — Monte-Carlo replay engine, replay packs (`.rpk`) & substrate bank

**Walk once, replay any acquisition.** The walker trajectory depends only on
`(geometry, diffusivity, seed)`; the gradient waveform, B0/orientation, T2/T1, surface
relaxivity, susceptibility and bound-pool relaxation are all *replay knobs*. So a single walk
to `T_max` serves every shorter TE, every b-value/direction, and every relaxation/field/MT
setting — reconstructed post-hoc instead of re-simulated. This supersedes the earlier
"single pass; no replay" scope note below: replay is now a first-class public engine, with the
fused single-pass kernels retained as the validation oracle.

### Added
- **Walk-once producer** — `simulate_trajectories(...)` (`core.py`) records positions plus, with
  `save_relaxation_data=True`, the boundary-local-time and per-compartment channels (and a
  bound-pool occupancy channel when `kappa_MT>0`).
- **Replay operators** (`trajectories.py`) — `apply_waveform_to_trajectories`, `apply_waveform_jax`
  (differentiable w.r.t. `G`), `apply_waveform_with_relaxation[_jax]` (gradient + scalar/per-comp
  T2 + T1 + surface relaxivity + provider-driven susceptibility), `apply_waveform_bloch[_jax]`
  (vector-Bloch replay: emergent RF refocusing, off-resonance, B1⁺, slice profile, MT bound pool),
  plus `unwrap_periodic`, `pre_pulse_gradient_phase`, `finite_180_longitudinal_dwell`,
  `pathway_sign_se`.
- **MT replay** — `mt_walk.simulate_mt_trajectories` (binding walk recording per-save `bound_frac`),
  reusing the `mt` two-pool oracle.
- **`simulate(..., engine="auto"|"replay"|"fused")`** — `simulate()` now routes through the
  walk-once + replay backend by default (`"auto"`) for the geometries proven equivalent at the
  MC-noise floor, falling back to the byte-for-byte `"fused"` kernels for myelinated/permeable
  geometries, per-compartment-D meshes, and the `return_positions/compartments/walker_signals`
  single-pass internals. `engine="fused"` reproduces the pre-replay results exactly.
- **Replay-pack compression** (`compression.py`) — position codecs (`lowrank`, `temporal_dct`,
  `gaussian`, `marginal`), channel codecs (compartment RLE, quantized-RLE `bound_fraction`,
  density-aware `boundary_local_time`), split-half-floor fidelity (`measure_fidelity`,
  `auto_select_modes`), and mode-space replay (signal without reconstructing the trajectory).
- **`.rpk` creation** (`bank.py`) — `build_replay_pack`/`build_to_floor`, the `ReplayPack` class,
  `write_rpk`/`read_rpk` (safetensors container, spec schema `1.2`), and `master_from_walk`
  (a `simulate_trajectories(...)` result → pack). Conformant to the open
  [replay-pack-spec](https://github.com/dmrai-lab/replay-pack-spec) at capability tiers C0–C2 (+C4
  parametric MT).
- **Substrate bank** (`bank.py`, `bank_card.py`) — `stage_pack`/`pull`/`publish`/`publish_dir`,
  manifest + `SHA256SUMS` + Croissant catalog, and HuggingFace-renderable substrate cards
  (`substrate_card`). Behind the new **`[bank]`** extra (`safetensors`, `huggingface_hub`);
  `DEFAULT_REPO="dmrai-lab/substrate-bank"`.
- New `scipy` core dependency (DCT / low-rank KL compression + susceptibility field sampling).
- Examples: `examples/replay_pack_roundtrip.py` (walk → pack → write/read → replay == engine).

## Unreleased — magnetization transfer (mt-staging)

**Emergent magnetization transfer on a new forward vector-Bloch engine.** (Historical scope note:
this landed before the replay engine above; the "no replay / no susceptibility" framing it
originally carried is now superseded — both are public.)

### Added
- **`dmipy_sim.mt`** — host-side MT physics + analytic two-pool Bloch–McConnell oracle:
  impact-angle `stick_probability`, `(κ_MT,dwell)↔(f_b,k_f)` conversions, `mt_z_spectrum`.
- **`simulate_bloch`** (`bloch.py`) — forward vector-Bloch engine carrying `M=(Mx,My,Mz)`
  through RF pulses + gradient + T1/T2 in one scan, alongside the (unchanged) scalar
  engine. Emergent spin-echo / CPMG refocusing and off-resonance carrier; opt-in MT
  binding (stick / freeze / exponential dwell at the walls) with bound-pool
  `T2_bound`/`T1_bound` blended by occupancy → emergent Z-spectrum / saturation transfer,
  active during longitudinal storage; emergent voxel-scale crusher.
- **`pulse_sequence.py`** — `BlochSequence`, `gradient_echo`/`spin_echo` readouts,
  `prepend_mt_prep` (off-resonance MT-prep saturation block), `run_bloch_sequence`.
- **`Substrate`** MT config (`kappa_MT`, `dwell_time`, `T2_bound`, `T1_bound`,
  `off_resonance_bound`, `mt_side`) + `Substrate.with_mt`; plus the previously-missing
  per-compartment `T1_*` fields.
- `examples/three_observables.py` — the surface-relaxivity / MT / Z-spectrum figure.

## 2.2.0

**T1 relaxation & PGSTE coherence gating** on the direct `simulate(...)` path — the longitudinal
sibling of the in-scan T2 / surface-relaxivity pattern.

### Added
- **`simulate(..., T1=…)`** — longitudinal relaxation in the walk. Magnetisation is either
  transverse (T2 and surface relaxivity accrue) or stored along the field (only T1 acts).
- **PGSTE** (`pgste(...)` constructor) — a pulsed-gradient stimulated echo: `[+G | gradient-off
  mixing time TM | −G]`, storing magnetisation longitudinally during TM, so over TM there is **no**
  T2 loss and **no** surface-relaxivity loss, only T1. Ideal instantaneous perfect pulses only.
- **`Waveform`** gains `chi_perp` (a per-time-point binary transverse-coherence mask: 1 transverse,
  0 stored), `TM`, and `stimulated_echo`, propagated through `set_b` / `rotate_waveform` /
  `tile_waveform`.
- `physics.py` step functions accept `T1` and consume `(g_t, chi_t)` per step — adding
  `−(1−chi_t)·dt/T1` alongside `−chi_t·dt/T2` and gating surface relaxivity by `chi_t`.

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
