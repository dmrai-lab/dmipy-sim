# dmipy-sim — Agent Guide

**Read this file, not the whole tree.** dmipy-sim is built to be *operated by agents* (any
vendor); this guide is the operational contract — the mental model, entry points, copy-paste
tasks, and where to look for the rest.

JAX Monte-Carlo diffusion-MRI simulator: walkers random-walk through a geometry, a gradient
phase `φ = γ∫G·r dt` accumulates, and the signal is `mean(exp(log_w)·cos φ)`. It is the
**forward** model of the dmipy framework; the **analytical inverse** (model fitting) is
[dmipy-fit](https://github.com/dmrai-lab/dmipy-fit) (see its `CLAUDE.md`). **You describe the
tissue once**: both engines consume the same `AcquisitionScheme`, and `simulate()` accepts one
directly. The dependency is one-directional (**fit → sim**); sim never imports fit.

**Physics is the specification.** Correctness is defined by the test suite —
analytical solutions, eigenfunction series, Brownstein–Tarr relations, MISST reference
signals. Any refactor/backend change is fine as long as the suite stays green.

## Environment & GPU

Progress and diagnostics go to the `dmipy_sim.*` loggers (`logging.getLogger("dmipy_sim")`; silent unless you configure
logging), never `print`. A run that outlives ten seconds also leaves a record on disk (`run.py`, below): where it is, what it uses, how it ended. Warning categories: a physics-regime warning is a `UserWarning`, an environment / GPU / OOM warning a
`RuntimeWarning`, a retired spelling a `DeprecationWarning`.

Install: `pip install -e ".[dev]"` (add `[mesh]` for PLY loading, `[cuda12]` for GPU).
Large Monte-Carlo runs belong on GPU; use `float32` on GPU. If a CUDA jaxlib is
installed but `jax.devices()` shows only CPU, the loader path is usually missing —
export `LD_LIBRARY_PATH` to the venv's `nvidia/*/lib` dirs (see README).

## Tests — two tiers

```bash
JAX_PLATFORMS=cpu pytest tests/ -q -m "not slow and not gpu"   # fast: every PR
```

- Fast tier: primitives, geometry/waveform units, MC smoke — **~6 min on GPU** (405 tests,
  measured; CPU is several times slower, which is what CI pays). Runs on every push/PR
  (`.github/workflows/tests.yml`). Two things keep it there and are easy to undo by accident:
  the JAX **persistent compilation cache** (`tests/conftest.py`, cached across CI runs) — the
  suite is compile-bound, since every `jax.jit` is built inside a function body over a fresh
  closure so jit's in-memory cache never hits (#93) — and doing **no work at import time**.
  Anything built as a `parametrize` argument or by a module-level `importorskip` runs during
  COLLECTION, on every invocation, including runs that deselect it; one such import cost 87 s
  of the suite's 96 s collection (#91). Build fixtures on first USE.
- `@pytest.mark.slow`: heavy statistical MC validation (auto-marked per module in
  `tests/conftest.py::_SLOW_MC_MODULES`) — runs weekly / `workflow_dispatch`.
  Add a new heavy MC module's name to that set. `--heavy` bumps `N_WALKERS` to 1e6.

Tests mirror the package where a module has a home: `tests/geometry/` holds the substrate tests (walls, bounces,
packings, meshes, myelin, compartments, the per-geometry MC validations) — the test for `dmipy_sim/geometry/X` is under
`tests/geometry/`. `tests/physics/` and `tests/validation/` assert physics across modules and stay as they are.

When adding physics, assert against an **analytical** result or a **MISST** fixture
(`tests/fixtures/misst_*.npy`). Isolate faceting/discretisation bias by running a mesh
and the analytic geometry of the same shape through the identical waveform/seed/N.

## Common tasks (copy-paste)

**Forward signal** (b-values SI, s/m²; diffusivity m²/s; lengths m):
```python
from dmipy_sim import simulate, pgse, Cylinder
wf   = pgse([[1,0,0]], 0.01, 0.04, bvalues=[1e9], n_t=300)
geom = Cylinder(radius=5e-6, orientation=(0,0,1))
sig  = simulate(n_walkers=100_000, diffusivity=2e-9, waveform=wf, geometry=geom, seed=0)
```
`geometry=` on every driver (`simulate`, `simulate_cpmg`, `simulate_trajectories`, `simulate_bloch`,
`simulate_mt_trajectories`) also takes a `SubstrateSpec`, its dict or a `.sub.json` path (`spec.as_geometry`);
`geom.spec` writes out what the constructor left implicit. **The door is closed**: `as_geometry` computes and caches
`geometry.spec` (validated) on first use and refuses an object with no spec spelling; a `Mesh` built from arrays writes
its surface into `spec.build.surface_cache_dir()` (`$DMIPY_SIM_SURFACE_DIR`, else `~/.cache/dmipy-sim/surfaces`, named by
content hash) so it has one.

**Surface relaxivity / permeability** — substrate properties baked into the walk (one walk
per ρ/κ):
```python
Cylinder(radius=5e-6, orientation=(0,0,1), surface_relaxivity_t2=1e-6)  # ρ (m/s)
Cylinder(radius=5e-6, orientation=(0,0,1), permeability=2e-5)           # κ (m/s), Powles
```

**Load a mesh** (needs `[mesh]` extra):
```python
from dmipy_sim import Mesh
mesh = Mesh.from_ply("substrate.ply", scale=1e-5, periodic=True,
                     voxel_min=[-10e-6]*3, voxel_max=[10e-6]*3, feature_radius=1.7e-6)
mesh.quality_report()                       # per-effect resolution verdict
```

**Trajectory export → select walkers that permeated:**
```python
_, pos, origin, comp = simulate(N, D, wf, Mesh(V, F, permeability=2e-5), seed=0,
                                return_positions='full', return_compartments='full')
permeated = (comp != comp[:, :1]).any(axis=1)   # pos: (n_walkers, n_timesteps, 3)
```

**Visualise** a mesh + walkers (see `dmipy_sim.viz.viz`): `plot_mesh_3d`, `plot_mesh_section`,
`walk_paths` + `plot_trajectories`, `save_rotation` → gallery in `examples/mesh_viz/`.

**Cross-engine parity**: build a `dmipy_fit` `AcquisitionScheme` and pass it straight to
`simulate(..., waveform=scheme)` — the analytic model and this MC then see the identical
acquisition; assert to `max(0.02, 1/√N)`.

## Module map (`dmipy_sim/`)

Layered packages (#88): `geometry/` (substrates), `engine/` (`core`, `physics`, `bloch`, `pulse_sequence`, `mt`, `mt_walk`,
`gpu`, `_gpu_config`), `replay/` (`trajectories`, `compression`, `replay`, `bank`, `phantom`, `so3`, `fod`,
`_replay_kernel`), `acquisition/` (`waveforms`, `rf`, `noise`), `fields/`
(`susceptibility`, `susceptibility_field`), `viz/` (`viz`, `pedagogy`), plus `sequences/`, `substrate/`, `io/`, `math/` and the
level-0 modules `constants`, `compartments`, `persistent_walk`. `dmipy_sim.replay` and `dmipy_sim.viz` are packages that
re-export their same-named module. There are NO flat-path shims: import the packaged path or the public names from `dmipy_sim`.

| File | Role |
|------|------|
| `engine/core.py` | `simulate`, `simulate_mixture`, `simulate_cpmg`; sub-step auto-tune; `engine=` (`auto` default): replay when the geometry declares `replay_parity` (its producer walk IS the fused walk, test_replay_parity) and `_replay_gap` finds nothing fused-only, else fused — a capability on the geometry, never a class-name table; `return_positions` (`True`/`'full'`) and `return_compartments` (`'final'`/`'full'`). **Compile cache**: every fused scan and the replay producer keep their jitted batch function on the geometry (`geometry._batch_cache`, via `core.cached_batch`), keyed on the geometry's scalar state and the baked configuration; waveform samples, positions, keys and labels are traced arguments, so a sweep over b / seed / direction on one geometry compiles once, and a knob set on the geometry after a walk builds a new program. `simulate_trajectories` (the replay producer) returns a **`PersistentWalk`** and records every replay tier the geometry supports by default (`tiers="all"`: boundary local time + compartment occupancy, + bound fraction with `kappa_MT>0`); `tiers=()` is the cheaper positions-only walk |
| `engine/adaptive.py` | **`simulate_trajectories_adaptive`** (`field_sample_every=`: the strand field read at every that-many-th save only -- the path channel keeps a few modes over the walk and lives on its own grid, `PersistentWalk.field_sample_every`, the channel meta's `n_t` / `dt`, the replay gates it there (`replay._path_grid`); reading it at every one of DiSCo's 3349 saves cost 20x the diffusion walk): the trajectory producer for the curved tubes with adaptive stepping -- a save interval walked in rounds; at every round `PackedCurvedCylinders.wall_scales` places each walker by its distance to the nearest wall it can hit and that wall's radius; a walker beyond six sigma of the round's excursion takes one free Gaussian step truncated at its wall distance (it cannot cross), the rest run the wall-aware kernel at the R/6 step of THEIR nearest tube in radius classes, against the segments whose surface lies within the round's deterministic excursion, gathered once per round (`reach_candidates` / `_reflect_with`, the one interaction behind `_reflect`). The finest class is the fused producer's noise and rule; the channels are the fused producer's; the walk records how it stepped (`PersistentWalk.stepping`); `walk_spec(adaptive_steps=True)`. Tests: `tests/test_adaptive_steps.py` (finest class = fused to rounding, guarantees, statistics, cache = full gather) |
| `persistent_walk.py` | `PersistentWalk`: what both trajectory producers return — `positions`, `dt`, `sub_steps`, `dt_sim` always; `.save(path)` / `PersistentWalk.load(path)`: the walk as ONE safetensors file (its arrays as tensors, the scalars, the spec, the field basis's record `StrandFieldRecord` and the run's summary in the `walk` header; no geometry: the spec is the situation), from which `build_replay_pack` builds the same pack when the walk carries its field samples; `boundary_local_time`, `compartment`, `bound_frac` present or `None` by what was recorded (`has_surface`/`has_compartments`/`has_binding`); `illegal_crossings` on the object; `.bank_dict(**metadata)` is the bank's input dict. Never a tuple whose length depends on flags |
| `spec/` | the **substrate specification** (`SubstrateSpec`, replay-pack-spec/SUBSTRATE.md 0.1): domain + per-axis boundary, pools with bulk props / water fraction / susceptibility (the field-source pools), walls with per-direction permeability and per-side rho / MT, seeding, request vs realisation, validity, provenance; `load_spec` / `validate` refuse a spec that does not describe a walkable situation, naming the field. `geometry.spec` writes any analytic geometry's situation (open box of 4 radii for an isolated object, pool 0 present with zero water for a lone lumen, packed cells periodic in-plane with one wall entry and per-instance arrays), `spec.geometry_from_spec(spec)` builds it back; the two are a fixed point and the walks agree to the bit. A `Mesh` writes its spec too (its surface as a file: the one it was loaded from, or written into `surface_dir=`); a multi-surface bundle spec has no single Geometry and is walked pool by pool by **`spec.walk_spec(spec, n, T_max)`** (`dt_save` derived by `acquisition.scanners.save_interval` from the scanner class, N_w and T_max -- #143; `scanner=` names the class), the producer that reads a spec and nothing else (intra inside the inner surfaces, extra outside the outer ones with the domain's faces as walls, a stuck myelin pool frozen, seeding by measured volume with thinning or water-fraction weights, the field basis on the domain grid, or the per-segment closed form for a strand spec -- `field=True|False|"grid"`, `field_cutoff_m`, `field_cutoff_tol`; `seeding=StratifiedByVoxel(grid=, walkers_per_voxel=, census_draws=)` (`spec/seeding.py`) seeds the same count of every pool in each voxel it occupies with the weight `f_pool,v * water_fraction / n_pool,v`, the intra draw exact by segment volume (`fill_swept_by_voxel`) and every other pool drawn IN each wanted voxel and kept by membership (`fill_per_voxel`), so a block of the grid costs its own voxels and a distributed fill seeds one block per device; for a pack meant for `Phantom.partition`; returns a `PersistentWalk` carrying the spec, which the pack embeds). `build_replay_pack(walk, voxel_grid=Grid)` adds the **per-voxel certificate** (`bank.voxel_fidelity`: walkers binned by start voxel and pool, split-half floor and codec error per (voxel, pool) over the battery, arrays `voxel_ijk` / `voxel_certificate`, summary `fidelity["per_voxel"]`); `bank.voxel_fidelity_volumes(pack)` reads it back on the grid and `spec.plan_seeding(floors, counts, target_floor=, grid=)` turns a pilot's into the next walk's counts (`n = n_pilot (floor / target)^2`). `spec.cactus_spec(run_dir)`, `spec.winther_spec(inner, outer)`, `spec.caterpillar_spec(csv)` (sphere-grown cells: two `sphere_union` walls per axon population referencing the table by `column` / `cell_type`, a `glia` pool 3, a reflecting voxel) and `spec.strands_spec(txt)` (EPFL strand lists: one `swept_polyline` wall with per-instance centerlines and radii, a sheath at the g-ratio) and `spec.disco_spec(tracks, diameters, field=True)` (DiSCo's released `.tck` in 25 um voxel units + the strands' INNER diameters in mm, `io.strands.read_tck` / `read_diameters`; two tubes per strand as the paper meshed them, the axolemma at the listed radius and the sheath at the listed radius over `DISCO_G_RATIO` = 0.7, intra inside the first, extra outside the second, a DRY unseeded pool between them since DiSCo took its signal from the intra and extra particles only, both at DiSCo's own `DISCO_D` = 0.6e-9 m^2/s; the sheath is a susceptibility source by default (the myelin's chi is the sheath's, not its water's) so ONE walk carries C0-C3 for the two pools and DiSCo's own simulation is the replay with the field off; `field=False` leaves it inert; no myelin water in any pack of it; the 1 mm^3 reflecting domain; the walls CITE the track file -- `Surface(file=, format="tck", scale=, sha256=, instances={radii})`, `cite_tracks_as=` the path a dataset distributes it at, `build.polyline_arrays` / `resolve_surface_file` read it back from the working directory or `$DMIPY_SIM_SURFACE_DIR` -- since a spec embedded in every shard of a fill cannot carry 12,196 centerlines) are the dataset producers: they emit a spec and construct nothing. `walk_spec` walks ANY multi-surface spec pool by pool (`_walk_bundle`): a pool is what it is inside and outside of, a pool with D > 0 walks the interior of its inside-walls or the exterior of its outside-walls, a D = 0 shell is frozen, the field basis is rasterised from the same membership tests (`fields.susceptibility_field.predicate_field_basis`). Issue #130 is DONE: substrates enter as specs and nothing else constructs one (the bespoke `mesh_bundle_master` / `mesh_axon_master` builders and the `io.cactus` / `io.winther` / `io.mesh_substrate` loaders are deleted; `spec.cactus_spec` / `winther_spec` replace them), a pack embeds the spec and carries no T2 / rho / chi value, and `test_api_surface` fails on an `io` module that builds a geometry or an exported geometry without a spec |
| `geometry/` | the substrate package: `base` (ABC, `interact`, FreeDiffusion, Box1D), `analytic` (Sphere, Cylinder, Ellipsoid, PermeableSlab1D/Shell), `packed`, `myelin`, `packing`, `curved_cylinder` (`PackedCurvedCylinders(box=)` mirrors a finite voxel without crossing a tube wall), `sphere_union` (`SphereUnion`: the outer boundary of a union of overlapping spheres, one pool inside or outside, priced radius-band grids, seam-aware ray merge, reject-escape, the same voxel fold), `mesh`, `mesh_shapes`, and **`_boundary`** — the one implementation of each boundary rule. **Every geometry is walked in its own frame** (the cylinder kinds along +z): `orientation=` is the POSE, `_orient_R` (substrate -> lab), which `simulate` applies to the ACQUISITION, as the Mesh always did; nothing rotates a walker or a step (a float32 `@` on a CUDA device is a TF32 matmul, and the per-step frame change the analytic family used to do leaked a tilted cylinder's walkers through it). A pack from such a walk is in the substrate frame; the pose is a replay knob. |
| `geometry/curved_cylinder.py` | `CurvedCylinder`, `CurvedMyelinatedCylinder`, `PackedCurvedCylinders` — sphere-swept polyline fibres (curving strands, e.g. DiSCo). Intra-axonal space is the Minkowski sum of a centerline polyline with a ball, so it is smooth at every joint (no kink/gap/overlap of chained straight cylinders) and carries the local orientation along the strand. Analytic and impermeable — no mesh, no grid — so far cheaper than walking the equivalent triangulated tube. The wall interaction is specular at the crossing (interior: the exit from the walker's OWN tube, the chain of its segments' capsules; exterior: the entry into the first tube met), two bounces per step, then the shared side guard; an interior walker never changes tube (the strands are separate axons; DiSCo's inner tubes overlap on 0.33 % of their volume, the g-ratio-inflated outer ones on 3.3 %). `STEP_FRACTION` is the family's step rule, declared as `reflection_step_fraction` and `surface_substep_frac` and measured on the adversarial fixtures at one million walkers (the hairpin strand, the 0.1 R slot); the probes are `tests/geometry/test_curved_step_rule.py` |
| `geometry/mesh.py` | `Mesh` (grid-accelerated, closed or 3-D periodic triangular mesh) + `load_ply` |
| `io/` | file readers that construct nothing: `caterpillar.read_caterpillar` (CATERPillar `.csv` / `.swc` by header name, metres), `strands.read_strands` / `write_strands` (EPFL strand list: side, count, per strand `n` then `x y z r`); nothing under `io/` imports a geometry or the engine (test_api_surface locks it) |
| `fields/susceptibility.py` | off-resonance field providers (`SusceptibilitySources` iron/vasculature, `MyelinSusceptibility` hollow-cylinder, `GridSusceptibility` k-space dipole on a voxel source); each exposes a pure-JAX `delta_bz_fn()` that plugs into `simulate_bloch(..., susceptibility=)` as a per-step z-precession. **Field tier for an analytic substrate**: `fields.susceptibility_field.field_grid_of(MyelinatedCylinder | PackedMyelinatedCylinders, res=)` rasterises the sheath onto the field-basis grid (the packed cell is periodic and exact) and returns the `FieldGrid` that `build_replay_pack(field=)` takes, so a hollow-cylinder pack carries C3 like a mesh pack |
| `fields/hollow_cylinder.py` | **the closed form of the infinite hollow cylinder's field basis** (13 channels: `iso_local`, `M_P`, `M_A` per region, derived from the k-space kernel and matching it on every component; Wharton–Bowtell's lumen and outside fields in this tensor convention) -- the one building block of `MyelinSusceptibility` (the forward Bloch provider) and of the strand field; `contract` is the `(H, B0, chi)` contraction on channels |
| `fields/strand_field.py` | **`StrandFieldBasis`**: the field basis of a strand substrate at points without a grid -- every SEGMENT within `cutoff_m` contributes its infinite hollow cylinder's field times its finite-line factor `F = (z_A / sqrt(z_A^2 + rho^2) - z_B / sqrt(z_B^2 + rho^2)) / 2` (1 beside an infinite line, 1/2 on an end plane, the dipole tail `L rho^2 / 2 r^3` far away; telescopes to 1 along a straight strand), summed, minus the closed-form domain mean; within a strand's outer tube and one radius beyond (`NEAREST_GATE_RADII`) that strand's own term is its nearest segment's cylinder and the LOCAL terms (iso_local, the sheath's `P / 2` of `M_P`) are that segment's alone (membership, not a field: `tr M_P = 3 iso_local` holds and a pack stores 12 channels). Why not each strand's nearest segment: that rule jumps at every joint's bisector (DiSCo: 18 jumps per 100 um, the largest the far field's rms), which no far grid can read; the finite-line sum is continuous beyond the gate and matches the k-space route as well or better (54-degree joint: 8.9 / 8.4 / 3.6 / 1.0 % of the component's maximum in lumen / sheath / within three radii / beyond, against 9.0 / 9.3 / 4.6 / 1.4; 20 degrees: 4.8 / 3.5 / 1.3 / 0.9). `segments_max` caps the list per point (refused, never dropped). `StrandFieldRecord(meta)` is the record of a basis a walk sampled (a walk file, a loaded walk): meta only, it evaluates nothing; a pack of a walk with samples needs no more. The particle-mesh split (#217): `build_far_grid(spacing_m, near_m, blend_m=, all_strands=)` tabulates the switch-weighted far part (no local terms) on a coarse grid (`FarGrid`: a `.npy` of the values with a `.json` beside it, `.load` memory-maps it so the host keeps no copy of a grid that lives on the device; float16; `all_strands` sums every segment in `FAR_BLOCK`s, the exact far part), `with_far(grid)` then sums the closed form only within `near_m` of a point (`gather_radius_m`) and reads the grid tricubically beyond; the switch must start beyond the largest sheath's gate plus two cells; read error 0.1-0.2 % of the superposition on the free-end fixture and falls with the spacing; `walk_spec(field_far=)`. `channels(points)` is the one protocol a pack's path channel samples (`FieldGrid.channels` is the grid's side of it), `cutoff_error` the certificate `walk_spec` doubles the cutoff against; the default field of a strand spec (`field="grid"` rasterises within the budget as the cross-check). DiSCo's 1 mm^3 is 1.25e11 voxels at 0.2 um; ~120 strands within 25 um of a point |
| `geometry/mesh_shapes.py` | procedural myelin meshes + analytic grid sources (`myelinated_cylinder`, `undulating_myelin`, `half_bare_myelin`, `grid_axes`, `voxelize_shell`) — the susceptibility test/validation substrates |
| `engine/physics.py` | per-timestep `jax.lax.scan` bodies (`make_step_fn`, …) — boundary + phase + `log_w`, pure JAX |
| `replay/_replay_kernel.py` | **the** replay primitives: `effective_gradient` (a waveform's EXACT per-save weights against the piecewise-linear path through the saves, from the waveform's own grid: no resampling, an edge between saves carries its b; `se_gate` is the same reading of a spin-echo gate at any 180 time; `piece_moments` / `piece_phase_weights` the moments over any cut of the save grid; `bin_gate` the gate averaged over a save's ACCUMULATION interval, which is how the occupancy / contact channels are gated since a save's contact is accumulated over the step ending at it), `gradient_phase` / `phase_increments` (`γ dt Σ G·r`), `se_gate` (spin-echo sign), each with a `_jax` twin (`gradient_phase` is a chunked `einsum`, so the host replay never holds a float64 copy of the walk and does not pay BLAS's skinny-gemm path; the `_jax` twins pin `Precision.HIGHEST`, since TF32 on a GPU biased the Bloch replay signal by 10-20%). `trajectories.replay*`, `replay_bloch*`, `bank`, `replay.replay` read them; `trajectories.replay_bloch` (numpy, the reference) and `replay_bloch_jax` (the engine: `lax.scan`) take the same arguments and share `_bloch_replay_terms`, which cuts the walk into PIECES at every save and every RF instant (#147): per piece the gradient precession is exact for the piecewise-linear path, relaxation runs for the piece's duration, and the RF opening the piece flips exactly what came before it, so a gradient through a pulse and a pulse between saves are exact — finite/shaped pulses, carriers, slice-select, B1+, MT blend, weights, per-walker echoes all live there once; `compression.bridge_projection` is the one mode-space projection behind `mode_space_phi` and `replay.compile_scheme`; **gate extent** (dmipy-sim#225): a gate of `n` samples spans `(n - 1)` steps, its last sample is the readout and is held over nothing, and the first save of an accumulated channel (contact, occupancy, bound fraction) ends no step and weighs 0 -- so an acquisition of `TE = n dt` relaxes and accrues contact over exactly `n` saves, `relaxation_logweight(chi=None)` relaxes the walk over `(n_t - 1) dt`, and the longitudinal periods are `active - chi` with `active` the acquisition's own gate (T1 acts over a stored period, never over the walk beyond the echo) |
| `replay/bank.py` certificates | **what a pack certifies** (`build_replay_pack(fidelity=, fidelity_from=, device=)`, RPK.md 9.4 rule 4): `"measured"` replays the envelope battery on the raw and the decoded walk per tier (the dense oracle: 90 % of a block's pack time); a block of a FILL passes `fidelity="inherited", fidelity_from=<the certifying pack or its meta>` -- the codec error and per-tier terms are copied, the codec parameters (method, K, n_t, dt, containers, C2 K, path K/bits) must match or the build refuses, and the pack reads its OWN split-half floor, whole (`compression.measure_floor_coded`) and per voxel (`voxel_floor_coded`), from the stored coefficients (`coded_phases`: the bridge projection of the exact per-save weights), no path decoded. `device="auto"` runs the DST bands (`compression.dst_bands`, a full-precision matmul against the sine matrix, never TF32) and the coded floors on the JAX device when it is a GPU, in walker chunks; `"numpy"` / `"jax"` force one. Tests: `tests/test_fidelity_inherited.py` |
| `replay/compression.py` containers | **the integer band container** (registry 0.6, spec#10): `build_replay_pack(position_container="bands" | ((upto, bits), ...), blt_container="bands")` stores the bridge bands per band range at 16 / 8 bits with a per-band scale (`BAND_CONTAINER = ((16, 16), (None, 8))`: the first 16 bands at 16 bits, the rest at 8 -- a bridge's bands fall as 1/k, so the step stays nanometres; K = 128 positions 1560 -> 459 B per walker); keys `pos_x_ends`, `pos_x_b<i>`, `pos_band_scale (n_blocks, n_axes, K)`, `blt_b<i>`, `blt_band_scale (n_blocks, K)`, optional `band_block (N_w,)` (a merged pack stacks its shards' scale tables; `merge_packs` does it). `read_position_coeffs` / `bridge_bands` dequantise in ONE place, so every reader (the #242 contractions, the decoders, the certificate, `prefix`) takes either container; `position_container(arrays)` says which. C1 writes the **static** column (`comp_static` int8) by itself when no walker changes label; `has_c1` / `has_c2` / `c2_bands_K` are the predicates -- never test `pos_x` / `blt_bridge_dst` / `comp_rle_vals` by name. Defaults are still the float containers until the canonical set is re-certified |
| `replay/replay.py` | `ReplayPack`: **every knob is a contraction in the space its channel is stored in** (#241): the gradient with the bridge projection (`C @ W`), T2/T1 on the occupancy RUNS (`compression.relaxation_logweight_runs`: rate x the gate's on-time within the run from the gate's prefix sums; O(runs), one run per walker when nothing crosses), the gated surface term on the bridge itself (`compression.surface_logweight_bridge`: summation by parts + the orthonormal DST-I, two scalars and one (n_w, K) product; ungated = the exact endpoint), the path field on the cosine modes (`gate_hat = dct(field_gate)`, `Psi = gamma dt gate_hat . C`, then the Q(H) contraction of `susc_path_field`). No route through `walker_signals` decodes a track or a trajectory on a pack with the path channel (the grid-field route samples along the decoded path); the dense decoders are the oracles in `tests/test_replay_coefficient_space.py`. 120k walkers x 3560 saves: every row 0.2-0.6 s where T2 / rho / field took 138 / 22 / 113 s |
| `replay/bank.py` | **build a pack from a walk**: `build_replay_pack(walk, id=, license=, citation=)` assembles every tier the walk carries, reading the occupancy channel (C1), the water-fraction weights and the field basis (C3, myelinated geometries via `field_grid_of`) from the walk's spec / geometry; `field=FieldGrid(...)` supplies a mesh basis, `field=False` opts out. **A pack carries channels and the embedded `SubstrateSpec` (`pack.substrate`) and no physical value**: no T2 / T1 / rho / chi / MT parameter (`per_comp`, `mt`, `reference_chi_*` are gone). `has_relaxation` means the occupancy channel is present. `Substrate.canonical().request(n_fibres=, packing_fraction=, seed=, min_gap=)` realises the calibrated white matter as a **spec** (request and realisation recorded separately; a fraction above `RSA_LIMIT`, a failed placement or a violated gap is refused, never returned smaller); `.pack(...)` is that spec's geometry. `ReplayPack.load(path).replay(waveform)` is the one consume path and by default the **nominal replay**: the embedded spec's T2 / T1 per pool, wall rho, field-source chi and `nominal_field_T` as B0 (so a published pack reproduces its paper by firing a pulse at it); `tissue=False` is the bare diffusion signal, `tissue=Tissue(...)` a whole set of values, and every keyword (`T2=` by pool id or `{name: value}`, `T1=`, `rho=`, `D=`, `B0=`, `b0_dir=`, `chi_iso=`, `chi_aniso=`, `compartment=`) overrides one. **Pose knobs**: `orientation=` takes one pose (a 3x3 rotation, or the lab direction of `spec.frame.axis`; G and B0 rotate into the substrate frame together, exact by pose covariance) or a distribution of poses (a `replay.so3.Distribution`, or an `FOD` read as an axis density), which goes through `pack.pose_response(...)` -> `PoseResponse`: the response sampled once over Haar rotations and projected onto SO(3) (`lmax`, `nmax`), then composed by an inner product. `PoseResponse.asymmetry` is the share of the response that depends on the substrate's own azimuth, and the projection **refuses** rather than warns when the truncation misfits the response by more than the pack's Monte-Carlo floor. With a field it needs the susc_path channel; a multi-axis waveform is fine. A bare SH array is refused: `FOD.from_sh(coeffs, basis="tournier07"|"descoteaux07", legacy=)` converts per RPH.md 4.1, `FOD.native` checks the required (orthonormal, `real_sh_tournier(legacy=False)`) basis and unit integral, `FOD.watson` / `FOD.isotropic` are exact. `Substrate.request` writes the catalogue's myelin chi and its `field_T` into the spec; `winther_spec` keeps that dataset's +1.06e-6;  `merge_packs(shards, id=, out_path=)` joins the packs of DISJOINT voxel blocks of one walk (a distributed fill: each device seeds its block with `StratifiedByVoxel`, walks, packs with `voxel_grid=`): walker arrays concatenated, the per-voxel certificate the union (two shards holding one voxel are refused), fidelity the conservative aggregate, precision tiers recomputed and declared unshuffled, provenance listing the shards |
| `engine/mt.py` | magnetization-transfer host physics: impact-angle `stick_probability`, `(κ_MT,dwell)↔(f_b,k_f)` conversions, two-pool Bloch–McConnell oracle (`bloch_mcconnell_*`, `mt_z_spectrum`); **owns** `surface_to_volume`, `resolve_equilibrate_mode`, `equilibrate_burnin_plateau` for both MT drivers (`bloch.simulate_bloch`, `mt_walk.simulate_mt_trajectories`); the MT walk at `κ_MT = 0` is the plain walk to the bit and stores float32 |
| `engine/bloch.py` | **forward vector-Bloch engine** `simulate_bloch(n, D, seq, geometry, ...)` — reads the `ScannerSequence`'s physical `G`, `rf`, `crusher` and `readout` (the last sample, or every echo of a train) and carries `M=(Mx,My,Mz)` through RF + gradient + relaxation in ONE forward pass (no replay; no `rf_events=` / `echo_steps=` side channel -- #173 piece 6); opt-in MT binding + bound-pool blend + off-resonance + emergent voxel-scale crusher + **membrane permeability** (sub-stepped Powles crossing, so exchange across a longitudinal-storage mixing time is captured — e.g. FEXI) |
| `engine/pulse_sequence.py` | the Bloch-side builders: `bare_gradient_echo` / `bare_spin_echo` (no gradient), `fexi` (+ `fexi_b_detect`), `saturation_pulse` + `prepend_mt_prep` (the MT-prep block), `emergent_z_spectrum` (turnkey CW-saturation Z-spectrum sweep; emergent counterpart of `mt.mt_z_spectrum`) |
| `acquisition/rf.py` | the RF side of an acquisition: **`RFEvent`**, the one pulse dialect every builder emits and every reader reads (`t_s, flip_deg, label, axis_deg, duration_s, offset_hz, envelope`; `flip_split(nsub)` is the one place a pulse becomes what a grid does, read by both rasterisers), **`RFSchedule`**, the schedule itself — a tuple of `RFEvent` in time order, valid by construction (refuses anything else), and the ONE place the coherence mask (`.coherence`), the spin-echo sign / un-fold (`.sign`, RPK §6.6), `.refocus_time` and `.mixing_time` are derived from the pulses; `to_dicts`/`from_dicts` its serialised record — and `B1Pulse`, the played complex `B1(t)` that is an event's `envelope` (its length and `gamma * int |B1| dt` then ARE the event's duration and flip). A `.rf_events` is an `RFSchedule`; nothing reads an event as a dict and nothing re-derives from a list of events (`test_api_surface` locks it) |
| `acquisition/timing.py` | **`SequenceTiming`**, the spin-echo timing budget: where the gradient may not be on (`t_prep`, `t_excite`, the `t_refocus` window at TE/2, `t_readout_pre_echo`, native `TE`); `min_TE`, the off-`windows`, `on_mask`, `from_readout` (partial Fourier shortens the post-180 window), `from_pulseq`, and `rf_events(TE)`, the finite 90/180 it implies. A `ScannerSequence` built to a budget carries it as `timing` and `validate()` refuses gradient in its windows; every builder takes `timing=` and builds to one. Finite pulses make `chi_perp` a fractional pathway profile through `RFSchedule.coherence` (hard pulses stay binary) |
| `acquisition/scanner_constants.py` + `.json`, `acquisition/scanners.py` | **the one home of every scanner number**: the cited catalogue (per model `gradient` / `rf` leaves with `value, unit, source_key, confidence`; `envelopes` for declared limit points that are not a machine; `classes` and `aliases` for the short names) and `ScannerLimits.of(name, regime=)`, the typed SI view a consumer takes (`regime='diffusion'` gives the PNS-derated slew where one is catalogued). `SCANNERS` (the #143 certificate's class table) and `sequences.pulseq.PULSEQ_SYSTEMS` are views derived at import; `save_interval` is the save-grid rule. No Python module carries a scanner number (`test_api_surface` locks it) |
| `acquisition/scanner_sequence.py` | **`ScannerSequence`, the one acquisition object**: what the scanner does from t = 0 to the echo -- `G` (PHYSICAL), `dt`, `rf` (an `RFSchedule`), `readout` (the samples read; `echo_idx` the last), `timing` (a `SequenceTiming` budget or None), `encoding` (the per-measurement `Encoding` an analytical layer reads: b, directions, delta, Delta, TE, the family's parameters), `crusher` (the emergent voxel-scale crusher the vector-Bloch engine models), `family`. Derived, never stored: `G_eff = G * rf.sign(t)`, `chi_perp`, `TM`, `stimulated_echo`, `echoes`, `refocus_gap`, `b()`, `btensor()`; `validate()` on every build. The scalar engine, the b integrals and the pack's `replay` read `G_eff`; the vector-Bloch routes read `G` and apply `rf` themselves. `Protocol` is a tuple of these, one TE each, for a multi-TE scheme. `Waveform`, `Sequence` and `BlochSequence` no longer exist. Specified in replay-pack-spec `ACQUISITION.md` (the data model, the derivation rules, the invariants, the families) |
| `acquisition/waveforms.py` | **the** b / B-tensor integrals `b_from_gradient`, `btensor_from_gradient` (rectangular q, trapezoidal ∫; `calc_b` / `calc_btensor` read a sequence's `G_eff`), `btensor_invariants`, and the measurement-axis transforms `set_b`, `rotate_waveform`, `tile_waveform` (the `Encoding` follows). No builders live here |
| `sequences/` | **the builders, one per family, one mechanics** (#173 piece 5b). `assemble.py`: SHAPES (`trapezoid`, `trapezoid_train`, `cosine`, `bipolar`, `axis_pairs` -- unit blocks sampled mid-step, ramps `g / slew` on every edge, vertical at `np.inf`; a block that cannot reach its amplitude is refused, never clipped) × ASSEMBLERS (`SpinEcho`: the same block on both sides of the 180 at TE/2, every row centred on it, `q(TE) = 0` sample for sample; `StimulatedEcho`: block, store, TM, recall, block, `TE = 2 t_store + TM`; `GradientEcho`: a self-refocusing block after the lead-in; `EchoTrain`: one lobe per stretch the pulses and readouts leave, constant or alternating polarity, every echo refocused) and the driver `assemble()` (grid 0..TE, the amplitude iterated to an exact b or taken as given, the `Encoding`, `validate()`). `builders.py`: `pgse(dirs, delta, Delta, *, bvalues= | gradient_strengths=, TE=, n_t=, slew_rate=, timing=)`, `pgste(dirs, delta, TM, ...)`, `ogse(dirs, f, sigma, *, shape='trapezoid'|'cosine', Delta=, ...)`, `cpmg(n_echoes, TE, *, polarity='constant'|'alternate', beta_deg=, n_t_per_echo=, ...)`, `gre(TE, ...)`, `ste(duration, ...)`, `pte(normal, duration, ...)`, the readers `from_waveform`, `from_btensor_waveform`, `from_pgste_waveform`, and `instantaneous`, `to_gradient_array`; the top-level `dmipy_sim.pgse` etc. ARE these. Exactly one of `bvalues` / `gradient_strengths`; `TE=` beyond the minimum adds dead time symmetrically; `timing=` makes the pulses finite and keeps the gradient out of the lead-in, the pulse windows and the readout tails. `pulseq` import/export |
| `engine/gpu.py`, `engine/_gpu_config.py` | GPU guard/session, device-memory cap |
| `run.py` | **the record of a run** (#257): every long producer (`simulate`, `simulate_trajectories`, `simulate_trajectories_adaptive`, `walk_spec`, `build_replay_pack`, `merge_packs`) opens `with Run(producer, params=) as run:`; a caller never does. In memory from the first moment; PERSISTED under `$DMIPY_SIM_RUN_DIR` (default `~/.cache/dmipy-sim/runs/<start>-<producer>-<pid>/`, empty = off) the first time the run outlives `DMIPY_SIM_SAMPLE_S` (10 s) or at once with `run_dir=`: `manifest.json` first (params, code, host, devices, the memory ceiling, argv), `events.jsonl` append-only and flushed per row (`phase`, `progress` with rate/ETA -- throttled to `DMIPY_SIM_PROGRESS_S` here, so a producer reports every batch or save; `resource` from a sampler thread: host RSS, available, cgroup current/max, every device's bytes in use and peak; `artifact`; `warning`; `join` when a producer runs inside another, which shares the outer record; `end` with the traceback), `summary.json` at the end. A `PersistentWalk.run` is its Run; a pack's `provenance["run"]` carries the walk's summary and the pack run's id. **Spool and resume**: `run.spool(name, arrays, header)` writes a finished piece (a walk's batch) as `spool/<name>.safetensors` under the record at once, whole or absent; a `run_dir=` that already holds a record of the SAME producer and params is resumed (a `start` row with `resumed`, the events appended) and `run.spooled(name)` gives the pieces back -- the adaptive walk reads the batches it finds and walks the rest, so a killed walk costs the batch in progress (`walk_spec(run_dir=, spool=True)`; a different run in that directory is refused). **The budget guard**: the sampler holds the RSS against `DMIPY_SIM_BUDGET_FRACTION` (0.9) of the ceiling (`memory_ceiling()`: `DMIPY_SIM_MEMORY_CEILING_BYTES`, else the cgroup's, else MemTotal) and `run.batches` projects the growth over the last batch to the batches left; past either the producer raises `ResourceBudgetError` at the next batch boundary (the margin itself: at the next progress report) with the projection in the record -- a clean stop with the spool intact instead of the host's SIGKILL with nothing; `0` disables. `report(dir)` / `list_runs()` / `python -m dmipy_sim.run [list | <dir>]`: status `ok` / `error` / `running` / `stale` (heartbeat older than two intervals) / `killed`, the last progress and ETA, peaks per phase. Tests: `tests/test_run.py` (a SIGKILLed subprocess leaves a readable record; every producer's source contains `with Run(`) |
| `engine/tables.py` | **`jit_with_tables(obj, names, fn)`**: a jitted program that reads an object's device tables (segment arrays, cell tables, the far grid) as ARGUMENTS -- the attributes are swapped for the tracers during the trace and restored after, so `fn`'s body and every method it calls read traced values and no executable embeds a copy. A closure over a device array embeds it as a constant in EVERY program it lowers (a host copy per lowering, another in the runtime): a 20k-walker DiSCo rehearsal held 223 copies of the segment tables, 3.4 GB in Python and far more in XLA, 31 GB RSS for 2 GB of data (#257). `PackedCurvedCylinders.TABLES` and `StrandFieldBasis.TABLES` name theirs; every jitted program on them goes through the helper (`test_api_surface`-style rule: no `jax.jit` closing over `self._X` arrays) |
| `acquisition/noise.py` | Rician / nc-χ measurement noise |
| `replay/so3.py` | **the pose representation**: a pose is a rotation, so a pack's response for one measurement is a function on SO(3), expanded in the real Wigner basis (`so3_design`, truncated at `lmax` in the pose and `nmax` in the substrate's own azimuth) and composed with a voxel's orientation distribution by the Peter-Weyl inner product. `Distribution` carries those coefficients with what they mean -- `pose` (one rotation), `axis` (a direction, azimuth unstated: what a peak is), `axis_density` (an ODF, through a cached linear map), `watson`, `bingham` (two concentrations about a declared frame, so a fan is expressible), `uniform`. Nothing assumes axial symmetry on either side; a roll-uniform distribution has no `n != 0` coefficients, so that azimuth is integrated away exactly rather than sampled. The orthonormal real spherical harmonics (`real_sh`, RPH.md 4.1) live here too |
| `viz/viz.py` | waveform plots + **mesh observability** (below) |

## Geometry contract (duck-typed by `simulate`/`make_step_fn`)

A geometry subclasses `geometry.base.Geometry` and provides `init_positions(n, key)` (seeding the pool it declares: `Mesh(..., pool="intra"|"extra")`, `CurvedMyelinatedCylinder(..., pool=)`; the old `intra=`/`shell=` seeding flags warn),
`classify_position(r)` (compartment tag), `length_scales` (a `LengthScales` tuple:
`min_feature`, `surface_pore`, `lookup_cell`, `is_mesh_feature`, `min_gap` — what the sub-step
rules divide; read it via `physics.length_scales_of`, never by probing `radius`/`cell_size`),
and **one wall interaction**. Capability flags (`supports_permeability`, `carries_side`,
`classify_returns_object_id`, …) and wall/bulk attributes (`permeability`,
`surface_relaxivity_t2`, `_orient_R`, per-compartment `_D_comp_jax`…) are declared on the base
class with defaults, so the engine reads them directly; `tests/test_api_surface.py` fails on any
new `getattr(geometry, …)` probe.

**Compartment ids** follow one convention everywhere (the `.rpk` one): 0 is the extra-cellular /
free pool, enclosed pools are positive — 1 intra (the lumen / inside a closed surface), 2 myelin.
Packed geometries (`classify_returns_object_id`) return 1..N for the object a walker is in.
`geometry.classify_position(r)` is the one source; `comp_traj`, `return_compartments`, per-compartment
`T2_per_comp`/`intra=`/`extra=` arrays are all indexed by that id (a `Mesh(intra={"T2":…}, extra={"T2":…})`
stores `(T2_extra, T2_intra)`). `PackedMyelinatedCylinders` carries an encoded id (`k+1` lumen of
axon k, `N_max+k+1` its sheath) and maps it with `geometry.pool_of(...)` at the API boundary.
**Per-compartment properties have one spelling**: `compartments=Compartments(extra=Pool(T2=…, D=…), intra=Pool(…), myelin=Pool(…))`
(`dmipy_sim.compartments`; `pool_id(name)` is the only name→id map; `Substrate.compartments` builds one). `Mesh(intra=, extra=)`
dicts and the `T2_intra/T2_myelin/T2_extra` kwargs are the previous spelling and warn `DeprecationWarning`; per-axon arrays on
`PackedMyelinatedCylinders` stay kwargs because they are per object, not per pool.

**Myelinated substrates** (`MyelinatedCylinder`, `PackedMyelinatedCylinders`) are stepped by one
kernel, `physics.make_myelin_substep`, whose wall physics is
`geometry.myelin.concentric_wall_kernel`: ray-traced hits on the axon membrane and the sheath
boundary, `d_perp = remaining·|cos α|`, one Powles decision per step, multi-bounce reflection,
strict side sentinels. The bounce budget and the number of candidate axons are derived per kernel
build from the worst case (a step zig-zagging across the narrowest passage: `min_gap`, the thinnest
sheath, or the shortest nudged grazing chord), not fixed. Per-axon `T2_*`, `rho_inner/outer`,
`kappa_inner/outer` and `D_*` are indexed by the walker's axon; `rho` is applied per hit with the
diffusivity of the pool the walker is in.

**Packed substrates** (`PackedCylinders`, `PackedSpheres`) are stepped by
`geometry.packed.packed_wall_kernel`: the same ray-traced multi-bounce rule, tested against the
objects within reach of the step (`packed_candidate_count`), with the budget
`packed_bounce_budget` derived from the narrowest passage (`min_gap` or the nudged grazing
chord). **Every wall encounter is its own Powles trial** with an independent uniform: summed over
a step that is `κ/D` times the boundary local time the reflections record, so transmission does
not depend on how many walls a sub-step meets and no gap-based sub-step rule is needed. A
single-hit rule at outer packing 0.45 ended 90% of extra-axonal walkers inside a cylinder.

**Sub-steps** come from one dispatch, `physics.resolve_sub_steps(geometry, D, dt, surface=,
mt_dwell_time=, override=)` — the maximum of the reflection (R/6, R/25 permeable), collision-lookup,
surface-local-time (pore/8) and binding criteria that apply. Every driver (`make_step_fn`,
`simulate_trajectories`, `simulate_bloch`, `simulate_mt_trajectories`, the packed-myelin kernel)
calls it; the returned step functions carry the count as `.n_sub`. Do not pick a rule per call site.

```python
hit = geom.interact(r, step, kappa_over_D=0.0, rho_over_D=0.0, key=None, side=None)
hit.r  hit.dlog_w  hit.crossed  hit.illegal      # a WallHit NamedTuple (a pytree)
```

`interact` is defined once on `Geometry` and is the entry point callers should use.
`reflect(r, step)`, `reflect_with_log_weight(r, step, ρ/D)` and
`permeate(r, step, κ/D, ρ/D, key)` still exist, but they are **the same function at
different argument values** — `reflect` IS the κ=0 case — and each geometry now has a
single implementation behind them. They were three copies once, and the copies drifted
into four separate bugs (#88): packed geometries expelled intra-axonal walkers, analytic
ones absorbed exterior walkers, `Mesh.reflect` silently lost box reflection and adaptive
nudging, and mesh surface local time disagreed with itself by 0.07%. **Do not add a
per-geometry variant** — extend the one implementation.

Set `supports_permeability = True` on a geometry with a membrane; `interact` raises on
κ>0 otherwise rather than silently reflecting. Set `surface_relaxivity_t2=` /
`permeability=` on the geometry; they are baked into the walk (one walk per ρ/κ).

## Meshes (`mesh.py`)

`Mesh(vertices, faces, …)` / `Mesh.from_ply(path, scale=…)` runs arbitrary triangular
meshes:

- **Uniform-grid broad phase** — per step tests only the walker's 27-cell triangle
  neighbourhood → `O(candidates)` not `O(n_triangles)` (10⁶-triangle meshes are
  tractable). Exact when `cell_size ≥ max step`.
- **3-D periodicity** (`periodic=True`, `voxel_min/max`) via ghost-triangle
  replication; geometry queries use the wrapped position, the returned position stays
  continuous so the gradient phase is correct. Box faces are wrap planes, not walls.
- **Smooth vertex-normal reflection** (`O(h²/R²)` faceting) and **leak-proof
  permeation** (one Powles decision at the first hit, then a multi-bounce reflection).
- **`orientation=`/`R=`** place the mesh in the bore (B0 = +z) as an *acquisition
  rotation* — the walk stays in the mesh frame.
- **Compartment (intra/extra) wall properties.** The membrane can relax and permit
  crossing differently by side/direction — the side is known at the collision
  (`sign(step·outward_normal)`). `intra={"surface_relaxivity_t2": ρ_i}`,
  `extra={"surface_relaxivity_t2": ρ_e}` → side-dependent ρ; `permeability={
  "intra_to_extra": κ_out, "extra_to_intra": κ_in}` → direction-dependent κ (scalar
  = symmetric, the default). Stored as a nominal value × per-side/-direction
  multipliers applied in `reflect_with_log_weight` / `permeate` (per sub-step, so an
  aggregate step carries the fractional intra/extra occupancy); scalars reproduce
  the symmetric behaviour bit-for-bit. **Caveat:** asymmetric κ breaks detailed
  balance — it's a *pump* (net flux, non-equilibrium), not passive exchange.
- **Per-compartment bulk D / T2** via the same `intra=`/`extra=` dicts
  (`{"D":…, "T2":…}`) — a per-*step* effect resolved in `make_step_fn`: the step
  length uses the current compartment's D and the log-weight its 1/T2, indexed by
  the compartment id the scan carries (`_D_comp_jax` / `_inv_T2_comp_jax`; seeded by
  `classify_positions_exact`, advanced by `classify_position_carry`; absent →
  scalar path, unchanged). Both sides required if either is given. Unequal D across
  a **permeable** wall is rejected (diffusivity-discontinuity interface). T1 isn't
  applied in the forward walk, so per-compartment T1 is out of scope. There is ONE
  step builder — do **not** add a per-geometry copy; extend the resolvers.
- **Resolution:** diffusion & surface relaxivity hit the noise floor at coarse
  resolution; permeability needs `edge/feature ≲ 0.04`. `Mesh.quality_report()` and a
  construction warning flag a too-coarse mesh.
- **Collision-response flags** are constructor kwargs with the validated defaults (`reject_escape=True`,
  `box_reflect=True`, `adaptive_nudge=False`), documented in `Mesh.__init__`; nothing is set on the instance after
  construction. They are measurement switches for the engine's tests, not physics.
- **No mesh files in the repo** — tests generate meshes on the fly (icosphere / open
  tube); large research PLYs are a manual stress test only.

## Mesh visualisation (`viz.py`)

`plot_mesh_section` (slice inspector), `plot_walkers_3d`, `plot_cell_surface`,
`plot_mesh_3d` (transparent cells + paths — the honest confinement view for a 3-D
substrate), `walk_paths` + `plot_trajectories`, `save_rotation` (animated GIF).
matplotlib is a lazy/optional import; `trimesh` (the `[mesh]` extra) is only needed to
read files or split cells. Rendered gallery: `examples/mesh_viz/`.
