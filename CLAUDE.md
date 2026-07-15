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

- Fast tier: primitives, geometry/waveform units, MC smoke — ~1 min on CPU, runs on
  every push/PR (`.github/workflows/tests.yml`).
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
| `geometries.py` | `FreeDiffusion`, `Box1D`, `Sphere`, `Cylinder`, `Ellipsoid`, `PackedCylinders/Spheres`, `MyelinatedCylinder`, `PackedMyelinatedCylinders`, packing helpers |
| `mesh.py` | `Mesh` (grid-accelerated, closed or 3-D periodic triangular mesh) + `load_ply` |
| `physics.py` | per-timestep `jax.lax.scan` bodies (`make_step_fn`, …) — boundary + phase + `log_w`, pure JAX |
| `waveforms.py` | `Waveform`, `pgse/ogse/cpmg/…`, `set_b`, b-tensor helpers |
| `gpu.py`, `_gpu_config.py` | GPU guard/session, device-memory cap |
| `noise.py` | Rician / nc-χ measurement noise |
| `sh_convolution.py` | SH convolution for orientation distributions |
| `viz.py` | waveform plots + **mesh observability** (below) |

## Geometry contract (duck-typed by `simulate`/`make_step_fn`)

A geometry provides `init_positions(n, key)` and `reflect(r, step)` (pure JAX), and
optionally `reflect_with_log_weight(r, step, ρ/D)` (surface relaxivity),
`permeate(r, step, κ/D, ρ/D, key)` (Powles crossing), `classify_position(r)`
(compartment tag), and a `radius`/feature scale for the sub-step auto-tune. Set
`surface_relaxivity_t2=` / `permeability=` on the geometry; they are baked into the
walk (one walk per ρ/κ).

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
