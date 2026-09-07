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
logging), never `print`. Warning categories: a physics-regime warning is a `UserWarning`, an environment / GPU / OOM warning a
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
from dmipy_sim import simulate, pgse, set_b, Cylinder
wf   = set_b(pgse(delta=0.01, DELTA=0.04, G_magnitude=0.2, bvecs=[[1,0,0]], n_t=300), 1e9)
geom = Cylinder(radius=5e-6, orientation=(0,0,1))
sig  = simulate(n_walkers=100_000, diffusivity=2e-9, waveform=wf, geometry=geom, seed=0)
```

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
`gpu`, `_gpu_config`), `replay/` (`trajectories`, `compression`, `replay`, `bank`, `phantom`, `sh_convolution`, `gaunt`,
`_replay_kernel`), `acquisition/` (`waveforms`, `rf`, `noise`), `fields/`
(`susceptibility`, `susceptibility_field`), `viz/` (`viz`, `pedagogy`), plus `sequences/`, `substrate/`, `io/`, `math/` and the
level-0 modules `constants`, `compartments`, `persistent_walk`. `dmipy_sim.replay` and `dmipy_sim.viz` are packages that
re-export their same-named module. There are NO flat-path shims: import the packaged path or the public names from `dmipy_sim`.

| File | Role |
|------|------|
| `engine/core.py` | `simulate`, `simulate_mixture`, `simulate_cpmg`; sub-step auto-tune; `engine=` (`auto` default): replay when the geometry declares `replay_parity` (its producer walk IS the fused walk, test_replay_parity) and `_replay_gap` finds nothing fused-only, else fused — a capability on the geometry, never a class-name table; `return_positions` (`True`/`'full'`) and `return_compartments` (`'final'`/`'full'`). **Compile cache**: every fused scan and the replay producer keep their jitted batch function on the geometry (`geometry._batch_cache`, via `core.cached_batch`), keyed on the geometry's scalar state and the baked configuration; waveform samples, positions, keys and labels are traced arguments, so a sweep over b / seed / direction on one geometry compiles once, and a knob set on the geometry after a walk builds a new program. `simulate_trajectories` (the replay producer) returns a **`PersistentWalk`** and records every replay tier the geometry supports by default (`tiers="all"`: boundary local time + compartment occupancy, + bound fraction with `kappa_MT>0`); `tiers=()` is the cheaper positions-only walk |
| `persistent_walk.py` | `PersistentWalk`: what both trajectory producers return — `positions`, `dt`, `sub_steps`, `dt_sim` always; `boundary_local_time`, `compartment`, `bound_frac` present or `None` by what was recorded (`has_surface`/`has_compartments`/`has_binding`); `illegal_crossings` on the object; `.bank_dict(**metadata)` is the bank's input dict. Never a tuple whose length depends on flags |
| `spec/` | the **substrate specification** (`SubstrateSpec`, replay-pack-spec/SUBSTRATE.md 0.1): domain + per-axis boundary, pools with bulk props / water fraction / susceptibility (the field-source pools), walls with per-direction permeability and per-side rho / MT, seeding, request vs realisation, validity, provenance; `load_spec` / `validate` refuse a spec that does not describe a walkable situation, naming the field. `geometry.spec` writes any analytic geometry's situation (open box of 4 radii for an isolated object, pool 0 present with zero water for a lone lumen, packed cells periodic in-plane with one wall entry and per-instance arrays), `spec.geometry_from_spec(spec)` builds it back; the two are a fixed point and the walks agree to the bit. A `Mesh` writes its spec too (its surface as a file: the one it was loaded from, or written into `surface_dir=`); a multi-surface bundle spec has no single Geometry and is walked pool by pool by **`spec.walk_spec(spec, n, T_max, dt_save)`**, the producer that reads a spec and nothing else (intra inside the inner surfaces, extra outside the outer ones with the domain's faces as walls, a stuck myelin pool frozen, seeding by measured volume with thinning or water-fraction weights, the field basis on the domain grid; returns a `PersistentWalk` carrying the spec, which the pack embeds). `spec.cactus_spec(run_dir)`, `spec.winther_spec(inner, outer)`, `spec.caterpillar_spec(csv)` (sphere-grown cells: two `sphere_union` walls per axon population referencing the table by `column` / `cell_type`, a `glia` pool 3, a reflecting voxel) and `spec.strands_spec(txt)` / `spec.disco_spec(txt)` (EPFL strand lists: one `swept_polyline` wall with per-instance centerlines and radii, a sheath at the g-ratio) are the dataset producers: they emit a spec and construct nothing. `walk_spec` walks ANY multi-surface spec pool by pool (`_walk_bundle`): a pool is what it is inside and outside of, a pool with D > 0 walks the interior of its inside-walls or the exterior of its outside-walls, a D = 0 shell is frozen, the field basis is rasterised from the same membership tests (`fields.susceptibility_field.predicate_field_basis`). Issue #130 is DONE: substrates enter as specs and nothing else constructs one (the bespoke `mesh_bundle_master` / `mesh_axon_master` builders and the `io.cactus` / `io.winther` / `io.mesh_substrate` loaders are deleted; `spec.cactus_spec` / `winther_spec` replace them), a pack embeds the spec and carries no T2 / rho / chi value, and `test_api_surface` fails on an `io` module that builds a geometry or an exported geometry without a spec |
| `geometry/` | the substrate package: `base` (ABC, `interact`, FreeDiffusion, Box1D), `analytic` (Sphere, Cylinder, Ellipsoid, PermeableSlab1D/Shell), `packed`, `myelin`, `packing`, `curved_tube` (`PackedCurvedTubes(box=)` mirrors a finite voxel without crossing a tube wall), `sphere_union` (`SphereUnion`: the outer boundary of a union of overlapping spheres, one pool inside or outside, priced radius-band grids, seam-aware ray merge, reject-escape, the same voxel fold), `mesh`, `mesh_shapes`, and **`_boundary`** — the one implementation of each boundary rule. |
| `geometry/curved_tube.py` | `CurvedTube`, `MultiShellCurvedTube`, `PackedCurvedTubes` — sphere-swept polyline fibres (curving strands, e.g. DiSCo). Intra-axonal space is the Minkowski sum of a centerline polyline with a ball, so it is smooth at every joint (no kink/gap/overlap of chained straight cylinders) and carries the local orientation along the strand. Analytic and impermeable — no mesh, no grid — so far cheaper than walking the equivalent triangulated tube |
| `geometry/mesh.py` | `Mesh` (grid-accelerated, closed or 3-D periodic triangular mesh) + `load_ply` |
| `io/` | file readers that construct nothing: `caterpillar.read_caterpillar` (CATERPillar `.csv` / `.swc` by header name, metres), `strands.read_strands` / `write_strands` (EPFL strand list: side, count, per strand `n` then `x y z r`); nothing under `io/` imports a geometry or the engine (test_api_surface locks it) |
| `fields/susceptibility.py` | off-resonance field providers (`SusceptibilitySources` iron/vasculature, `MyelinSusceptibility` hollow-cylinder, `GridSusceptibility` k-space dipole on a voxel source); each exposes a pure-JAX `delta_bz_fn()` that plugs into `simulate_bloch(..., susceptibility=)` as a per-step z-precession. **Field tier for an analytic substrate**: `fields.susceptibility_field.field_grid_of(MyelinatedCylinder | PackedMyelinatedCylinders, res=)` rasterises the sheath onto the field-basis grid (the packed cell is periodic and exact) and returns the `FieldGrid` that `build_replay_pack(field=)` takes, so a hollow-cylinder pack carries C3 like a mesh pack |
| `geometry/mesh_shapes.py` | procedural myelin meshes + analytic grid sources (`myelinated_cylinder`, `undulating_myelin`, `half_bare_myelin`, `grid_axes`, `voxelize_shell`) — the susceptibility test/validation substrates |
| `engine/physics.py` | per-timestep `jax.lax.scan` bodies (`make_step_fn`, …) — boundary + phase + `log_w`, pure JAX |
| `replay/_replay_kernel.py` | **the** replay primitives: `resample_gradient` (waveform → walk grid), `gradient_phase` / `phase_increments` (`γ dt Σ G·r`), `se_gate` (spin-echo sign), each with a `_jax` twin (`gradient_phase` is a chunked `einsum`, so the host replay never holds a float64 copy of the walk and does not pay BLAS's skinny-gemm path; the `_jax` twins pin `Precision.HIGHEST`, since TF32 on a GPU biased the Bloch replay signal by 10-20%). `trajectories.replay*`, `replay_bloch*`, `bank`, `sh_convolution` read them; `trajectories.replay_bloch` (numpy, the reference) and `replay_bloch_jax` (the engine: `lax.scan`, one measurement at a time) take the same arguments and share `_bloch_replay_terms` — finite/shaped pulses, carriers, slice-select, B1+, MT blend, weights, per-walker echoes all live there once; `compression.bridge_projection` is the one mode-space projection behind `mode_space_phi` and `replay.compile_scheme` |
| `replay/bank.py` | **build a pack from a walk**: `build_replay_pack(walk, id=, license=, citation=)` assembles every tier the walk carries, reading the occupancy channel (C1), the water-fraction weights and the field basis (C3, myelinated geometries via `field_grid_of`) from the walk's spec / geometry; `field=FieldGrid(...)` supplies a mesh basis, `field=False` opts out. **A pack carries channels and the embedded `SubstrateSpec` (`pack.substrate`) and no physical value**: no T2 / T1 / rho / chi / MT parameter (`per_comp`, `mt`, `reference_chi_*` are gone). `has_relaxation` means the occupancy channel is present. `Substrate.canonical().request(n_fibres=, packing_fraction=, seed=, min_gap=)` realises the calibrated white matter as a **spec** (request and realisation recorded separately; a fraction above `RSA_LIMIT`, a failed placement or a violated gap is refused, never returned smaller); `.pack(...)` is that spec's geometry. `ReplayPack.load(path).replay(waveform, tissue=, T2=, T1=, rho=, D=, B0=, b0_dir=, chi_iso=, chi_aniso=, compartment=)` is the one consume path: T2 / T1 per pool id (or `{name: value}` resolved through the embedded spec) are given at replay and require the occupancy channel; `spec.Tissue.from_spec(spec, B0=, **overrides)` bundles the spec's nominal T2 / T1 / rho / chi (chi is `None` for an analytic sheath: give it) |
| `engine/mt.py` | magnetization-transfer host physics: impact-angle `stick_probability`, `(κ_MT,dwell)↔(f_b,k_f)` conversions, two-pool Bloch–McConnell oracle (`bloch_mcconnell_*`, `mt_z_spectrum`); **owns** `surface_to_volume`, `resolve_equilibrate_mode`, `equilibrate_burnin_plateau` for both MT drivers (`bloch.simulate_bloch`, `mt_walk.simulate_mt_trajectories`); the MT walk at `κ_MT = 0` is the plain walk to the bit and stores float32 |
| `engine/bloch.py` | **forward vector-Bloch engine** `simulate_bloch` — carries `M=(Mx,My,Mz)` through RF + gradient + relaxation in ONE forward pass (no replay); opt-in MT binding + bound-pool blend + off-resonance + emergent voxel-scale crusher + **membrane permeability** (sub-stepped Powles crossing, so exchange across a longitudinal-storage mixing time is captured — e.g. FEXI) |
| `engine/pulse_sequence.py` | `BlochSequence`, `gradient_echo`/`spin_echo` readouts, `prepend_mt_prep` (off-resonance MT-prep saturation block), `run_bloch_sequence`, `emergent_z_spectrum` (turnkey CW-saturation Z-spectrum sweep; emergent counterpart of `mt.mt_z_spectrum`) |
| `acquisition/waveforms.py` | `Waveform`, `pgse/ogse/cpmg/…`, `set_b`; **the** b / B-tensor integrals `b_from_gradient`, `btensor_from_gradient` (rectangular q, trapezoidal ∫; `calc_b`/`calc_btensor` and `sequences` read them) |
| `sequences/` | `Sequence` = a `Waveform` (same readout attributes: `echo_idx`, `echo_indices`, `rf_events`, `chi_perp`, …) plus per-measurement encoding (`bvalues`, `gradient_directions`, `delta`, `Delta`, `TE`, family flags) for dmipy-fit; every `from_X` constructor ends with an exact numeric scaling so `bvalues == b_from_gradient(G, dt)`; `pulseq` import/export |
| `engine/gpu.py`, `engine/_gpu_config.py` | GPU guard/session, device-memory cap |
| `acquisition/noise.py` | Rician / nc-χ measurement noise |
| `replay/sh_convolution.py` | SH convolution for orientation distributions |
| `viz/viz.py` | waveform plots + **mesh observability** (below) |

## Geometry contract (duck-typed by `simulate`/`make_step_fn`)

A geometry subclasses `geometry.base.Geometry` and provides `init_positions(n, key)` (seeding the pool it declares: `Mesh(..., pool="intra"|"extra")`, `MultiShellCurvedTube(..., pool=)`; the old `intra=`/`shell=` seeding flags warn),
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
