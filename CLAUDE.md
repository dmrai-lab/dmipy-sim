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

**Visualise** a mesh + walkers (see `dmipy_sim.viz`): `plot_mesh_3d`, `plot_mesh_section`,
`walk_paths` + `plot_trajectories`, `save_rotation` → gallery in `examples/mesh_viz/`.

**Cross-engine parity**: build a `dmipy_fit` `AcquisitionScheme` and pass it straight to
`simulate(..., waveform=scheme)` — the analytic model and this MC then see the identical
acquisition; assert to `max(0.02, 1/√N)`.

## Module map (`dmipy_sim/`)

| File | Role |
|------|------|
| `core.py` | `simulate`, `simulate_mixture`, `simulate_cpmg`; sub-step auto-tune; `return_positions` (`True`/`'full'`) and `return_compartments` (`'final'`/`'full'`) |
| `geometry/` | the substrate package: `base` (ABC, `interact`, FreeDiffusion, Box1D), `analytic` (Sphere, Cylinder, Ellipsoid, PermeableSlab1D/Shell), `packed`, `myelin`, `packing`, `curved_tube`, `mesh`, `mesh_shapes`, and **`_boundary`** — the one implementation of each boundary rule. `dmipy_sim.geometries` is a compat shim. |
| `geometry/curved_tube.py` | `CurvedTube`, `MultiShellCurvedTube`, `PackedCurvedTubes` — sphere-swept polyline fibres (curving strands, e.g. DiSCo). Intra-axonal space is the Minkowski sum of a centerline polyline with a ball, so it is smooth at every joint (no kink/gap/overlap of chained straight cylinders) and carries the local orientation along the strand. Analytic and impermeable — no mesh, no grid — so far cheaper than walking the equivalent triangulated tube |
| `geometry/mesh.py` | `Mesh` (grid-accelerated, closed or 3-D periodic triangular mesh) + `load_ply` |
| `susceptibility.py` | off-resonance field providers (`SusceptibilitySources` iron/vasculature, `MyelinSusceptibility` hollow-cylinder, `GridSusceptibility` k-space dipole on a voxel source); each exposes a pure-JAX `delta_bz_fn()` that plugs into `simulate_bloch(..., susceptibility=)` as a per-step z-precession |
| `geometry/mesh_shapes.py` | procedural myelin meshes + analytic grid sources (`myelinated_cylinder`, `undulating_myelin`, `half_bare_myelin`, `grid_axes`, `voxelize_shell`) — the susceptibility test/validation substrates |
| `physics.py` | per-timestep `jax.lax.scan` bodies (`make_step_fn`, …) — boundary + phase + `log_w`, pure JAX |
| `mt.py` | magnetization-transfer host physics: impact-angle `stick_probability`, `(κ_MT,dwell)↔(f_b,k_f)` conversions, two-pool Bloch–McConnell oracle (`bloch_mcconnell_*`, `mt_z_spectrum`) |
| `bloch.py` | **forward vector-Bloch engine** `simulate_bloch` — carries `M=(Mx,My,Mz)` through RF + gradient + relaxation in ONE forward pass (no replay); opt-in MT binding + bound-pool blend + off-resonance + emergent voxel-scale crusher + **membrane permeability** (sub-stepped Powles crossing, so exchange across a longitudinal-storage mixing time is captured — e.g. FEXI) |
| `pulse_sequence.py` | `BlochSequence`, `gradient_echo`/`spin_echo` readouts, `prepend_mt_prep` (off-resonance MT-prep saturation block), `run_bloch_sequence`, `emergent_z_spectrum` (turnkey CW-saturation Z-spectrum sweep; emergent counterpart of `mt.mt_z_spectrum`) |
| `waveforms.py` | `Waveform`, `pgse/ogse/cpmg/…`, `set_b`, b-tensor helpers |
| `gpu.py`, `_gpu_config.py` | GPU guard/session, device-memory cap |
| `noise.py` | Rician / nc-χ measurement noise |
| `sh_convolution.py` | SH convolution for orientation distributions |
| `viz.py` | waveform plots + **mesh observability** (below) |

## Geometry contract (duck-typed by `simulate`/`make_step_fn`)

A geometry subclasses `geometry.base.Geometry` and provides `init_positions(n, key)`,
`classify_position(r)` (compartment tag), `length_scales` (a `LengthScales` tuple:
`min_feature`, `surface_pore`, `lookup_cell`, `is_mesh_feature`, `min_gap` — what the sub-step
rules divide; read it via `physics.length_scales_of`, never by probing `radius`/`cell_size`),
and **one wall interaction**. Capability flags (`supports_permeability`, `carries_side`,
`classify_returns_object_id`, …) and wall/bulk attributes (`permeability`,
`surface_relaxivity_t2`, `_orient_R`, per-compartment `_D_comp_jax`…) are declared on the base
class with defaults, so the engine reads them directly; `tests/test_api_surface.py` fails on any
new `getattr(geometry, …)` probe.

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
  length uses the current compartment's D and the log-weight its 1/T2 (via the
  geometry's `_D_comp_jax` / `_inv_T2_comp_jax` + `classify_position`; absent →
  scalar path, unchanged). Both sides required if either is given. Unequal D across
  a **permeable** wall is rejected (diffusivity-discontinuity interface). T1 isn't
  applied in the forward walk, so per-compartment T1 is out of scope. There is ONE
  step builder — do **not** add a per-geometry copy; extend the resolvers.
- **Resolution:** diffusion & surface relaxivity hit the noise floor at coarse
  resolution; permeability needs `edge/feature ≲ 0.04`. `Mesh.quality_report()` and a
  construction warning flag a too-coarse mesh.
- **No mesh files in the repo** — tests generate meshes on the fly (icosphere / open
  tube); large research PLYs are a manual stress test only.

## Mesh visualisation (`viz.py`)

`plot_mesh_section` (slice inspector), `plot_walkers_3d`, `plot_cell_surface`,
`plot_mesh_3d` (transparent cells + paths — the honest confinement view for a 3-D
substrate), `walk_paths` + `plot_trajectories`, `save_rotation` (animated GIF).
matplotlib is a lazy/optional import; `trimesh` (the `[mesh]` extra) is only needed to
read files or split cells. Rendered gallery: `examples/mesh_viz/`.
