# dmipy-sim

A **JAX Monte-Carlo diffusion-MRI simulator** — the numerical ground-truth companion to
[dmipy-fit](https://github.com/dmrai-lab/dmipy-fit). Spins random-walk through geometric
substrates and accumulate phase under **arbitrary free gradient waveforms** `G(t)`, with
**surface relaxivity** and **membrane permeability** baked into the walk. Everything is
vmap/scan JAX and runs on CPU or any CUDA-12 GPU.

The free waveform is the base representation: `G(t)` of shape `(n_measurements, n_t, 3)` is
the ground truth, and PGSE/OGSE/CPMG/STE/PTE are factory constructors, not fundamental
types. Magnetisation is treated as fully transverse throughout (ideal instantaneous pulses).
dmipy-sim and dmipy-fit share **one** pulse-sequence and substrate interface — the
simulator and the analytical models eat the same `Waveform` and the same substrate, with no
conversion layer.

## Quickstart — a diffusion signal in a few lines

Free (Gaussian) diffusion under a PGSE waveform reproduces the textbook Stejskal–Tanner law
`S/S0 = exp(-b·D)` — a one-line sanity check that the install works:

```python
import numpy as np
from dmipy_sim import simulate, pgse, set_b, FreeDiffusion

D = 2e-9   # m^2/s
for b in [0, 1e9, 2e9]:                       # b in s/m^2  (1e9 s/m^2 = 1000 s/mm^2 = 1 ms/µm^2)
    wf = set_b(pgse(delta=0.01, DELTA=0.04, G_magnitude=0.05, bvecs=[[1, 0, 0]], n_t=300), b)
    S  = float(np.asarray(simulate(n_walkers=50_000, diffusivity=D, waveform=wf,
                                    geometry=FreeDiffusion(), seed=0, require_gpu=False)).ravel()[0])
    print(f"b={b/1e9:.0f} ms/µm²   S/S0={S:.3f}   (exp(-bD)={np.exp(-b*D):.3f})")
# b=0 → 1.000   b=1 → 0.131 (exp -bD 0.135)   b=2 → 0.021 (0.018)
```

Swap `FreeDiffusion()` for a `Cylinder`/`Sphere`/`PackedCylinders` to see restriction, or add
`surface_relaxivity_t2=` / `permeability=` to the geometry — see below. (`require_gpu=False`
just silences the CPU-fallback warning; drop it on a GPU.)

## What's here

| | Entry point | Notes |
|---|---|---|
| **Forward MC** | `simulate(waveform, geometry, diffusivity, ...)` | walker ensemble signal `Re⟨exp(iφ)⟩`, with optional baked-in T2, surface relaxivity, permeability — one walk per call |
| **Multi-echo** | `simulate_cpmg(n_walkers, D, cpmg_waveform, geometry, ...)` | full CPMG echo train from a **single** walk — signal sampled at each echo (ideal 180s); `(n_echoes, n_measurements)` |
| **Encodings** | `pgse`, `ogse`, `cpmg`, `ste`, `pte`, `trapezoidal_ogse` | factory constructors over the free waveform; `calc_b`, `calc_btensor`, `btensor_invariants` |
| **Noise** | `add_rician_noise`, `add_nc_chi_noise`, `estimate_sigma` | |
| **Sequence I/O** | `dmipy_sim.sequences` (incl. Pulseq `.seq` interop), `scanner_constants` | per-vendor gradient/RF/SAR deliverability limits |

### Geometries

| Geometry | Restriction | Surface relaxivity | Permeability |
|---|---|---|---|
| `FreeDiffusion` | none | — | — |
| `Box1D` | 1-D slab | ✓ | — |
| `Sphere`, `Cylinder`, `Ellipsoid` | closed wall | ✓ | ✓ |
| `PackedCylinders`, `PackedSpheres` | periodic ensemble | ✓ | ✓ |
| `MyelinatedCylinder`, `PackedMyelinatedCylinders` | multi-wall myelin geometry | ✓ | ✓ dual-wall |

## Surface relaxivity & permeability

`surface_relaxivity_t2=ρ` (m/s) — Brownstein–Tarr: each wall collision reduces the walker
weight by `exp(−2ρ·d⊥/D)`, so the ensemble signal decays as `exp(−TE·ρ·S/V)` (cylinder
`S/V=2/R`, sphere `3/R`). `permeability=κ` (m/s) — Powles (2004) bidirectional crossing with
`p=min(1, 2κ·d⊥/D)`. Both are **baked into the walk**: build the geometry with the property
and call `simulate()` (one walk per ρ/κ). The ρ weight applies on reflection only, never on
transmission. Intra↔extra exchange is the Kärger/NEXI path; lipid bilayers are impermeable,
so physiological exchange is at the nodes of Ranvier, not through the myelin sheath.

## ⚠️ Tracer self-diffusivity vs conductivity (read before benchmarking tortuosity)

The MC (and PGSE/dMRI) measure the **tracer self-diffusivity** `D_self = MSD/4t`. This is
**not** the Fickian/effective conductivity `σ_eff` — they differ by the porosity:
`σ_eff/σ0 = (1−f)·D_self/D0`. Maxwell–Garnett `(1−f)/(1+f)`, Rayleigh, and the
Hashin–Shtrikman / Wiener bounds are statements about `σ_eff`, **not** `D_self`. For a
square array of impermeable cylinders the exact *tracer* value is `D_self/D0 = 1/(1+f)`, and
the MC matches it to ≤1.4%. Comparing the MC tracer diffusivity against the conductivity
formula spuriously suggests a 2× error and an apparent bound violation — it is a
units/quantity mismatch, not a bug.

## One call, one walk

Each `simulate(waveform, geometry, ...)` runs a fresh spin walk and returns the ensemble
signal. Surface relaxivity and permeability are substrate properties baked into that walk —
set them on the geometry and call `simulate()`:

```python
from dmipy_sim import simulate, pgse, set_b, Cylinder

geom = Cylinder(radius=5e-6, orientation=(0, 0, 1), surface_relaxivity_t2=1e-6)
wf   = set_b(pgse(delta=0.01, DELTA=0.04, G_magnitude=0.2, bvecs=[[1, 0, 0]], n_t=300), 1e9)
sig  = simulate(n_walkers=100_000, diffusivity=2e-9, waveform=wf, geometry=geom, seed=0)
```

## Examples

- **[Flagship — canonical white matter](examples/canonical_wm_flagship.ipynb)**
  ([Open in Colab](https://colab.research.google.com/github/dmrai-lab/dmipy-sim/blob/main/examples/canonical_wm_flagship.ipynb))
  — build a histology-calibrated packed-myelinated-cylinder substrate, run the Monte-Carlo
  forward with surface relaxivity, and check it against the analytical model in dmipy-fit.
- **[Validation ladders](examples/validation/)** — surface relaxivity and permeability from 1-D
  to 3-D vs exact analytics, and the extra-axonal tortuosity scale sweep.

## Relationship to disimpy / MISST / Camino, and citation

The core engine — Brownian walk, specular reflection, phase accumulation — follows the same
physics as [disimpy](https://github.com/kerkelae/disimpy) (Kerkelä 2020), MISST, and Camino,
and is validated against analytical and MISST reference signals for sphere/cylinder.
Contributions here: membrane permeability (Powles), surface relaxivity (Brownstein–Tarr,
interior + exterior), B-tensor encoding, packed ensembles, the free-waveform pulse-sequence
interface shared with dmipy-fit, and the JAX (vmap/scan) backend.

> For basic MC functionality cite disimpy/MISST; for the extensions cite dmipy-sim / the
> dmrai ecosystem. Kerkelä L, Nery F, Hall M, Clark C (2020), *disimpy*, JOSS 5(52), 2527.

## Physics as the specification

dmipy-sim is written with the **physical test suite as the specification** — analytical
solutions, eigenfunction series, Brownstein–Tarr relations, and MISST reference signals
define correctness, and the code is written to pass them. Any refactor or backend change is
safe as long as the suite passes.

## Install

```bash
# CPU (any platform)
pip install -e ".[dev]" && pip install "jax[cpu]>=0.6.2"
# NVIDIA GPU (CUDA 12)
pip install -e ".[cuda12,dev]"
```
GPU note: after install, point the linker at the bundled CUDA libs in your venv's `activate`:
```bash
export LD_LIBRARY_PATH=$(find "$VIRTUAL_ENV/lib"/python*/site-packages/nvidia -name lib -type d | tr '\n' ':')$LD_LIBRARY_PATH
```

## Tests

```bash
pytest -q -m "not slow"   # fast suite
pytest -q -m slow         # heavy GPU battery (first-principles validation ladders)
```

The tests cover: free/box/sphere/cylinder/ellipsoid diffusion vs analytical + MISST;
baked-in T2; surface relaxivity (interior + exterior); B-tensor (LTE/STE/PTE); packed
cylinders/spheres; permeability (all closed surfaces); SH convolution; and periodic unwrap.
The `slow`-marked `tests/validation/` battery asserts the first-principles 1D→2D→3D
permeability and surface-relaxivity ladders against exact eigenvalues (see
`examples/validation/`).

## License

Dual-licensed: **GNU AGPL-3.0** for open-source use, or a **commercial license** for
proprietary/closed use. See [LICENSE](LICENSE) and [LICENSING.md](LICENSING.md)
(commercial: rutger.fick@dmrai-lab.org).
