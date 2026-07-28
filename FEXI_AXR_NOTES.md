# FEXI / AXR — status & testing notes (WIP branch `feat/bloch-permeation`)

Dev notes for the filter-exchange (FEXI) work. **Not for immediate merge** — the sequence and
the engine feature are solid; the packed-substrate AXR *demo figure* still needs a substrate with
real pool contrast (below). Parked here so we can pick it up later.

## What's implemented (and solid)
- **`dmipy_sim.bloch.simulate_bloch` — membrane permeation.** The vector-Bloch step now does the
  Powles crossing (`geometry.permeate`), sub-stepped to ~R/25, reusing the scalar engine's
  `permeable_sub_steps`. Reflecting path unchanged when `permeability is None`. Tests:
  `tests/test_bloch_permeation.py` (slow tier) — κ-monotone + high-κ→free. **This is the reusable
  win** (any Bloch exchange work needs it), independent of FEXI.
- **`dmipy_sim.pulse_sequence.fexi(...)` — the FEXI sequence.** Bipolar filter + 90 store +
  crusher/mixing + 90 recall + bipolar detection → a `BlochSequence` (+ per-measurement
  `.b_detect`). **Bipolar blocks, no 180** — matches Kiselev & Li (below). `Delta` sets the
  diffusion time. Tests: `tests/test_fexi.py` — free-diffusion mechanics (reads true D, ADC flat
  vs t_mix, filter attenuates).

## The AXR model (fit)
```
ADC'(t_m) = ADC_eq * (1 - sigma * exp(-AXR * t_m))      # Lasič 2011; params [ADC_eq, sigma, AXR]
sigma = 1 - ADC'(0)/ADC_eq                              # filter efficiency
AXR   = k_in + k_ex = 1/tau_ex   (two-site Kärger limit)
```
Measure `ADC'(t_m)` from the detection block at >=2 b-values (b=0 + one nonzero); apply the σ term
only to filtered acquisitions. Reference implementation: md-dmri `methods/fexi11/fexi11_1d_fit2data.m`
(Nilsson/Topgaard), https://github.com/markus-nilsson/md-dmri.

## Design decisions (hard-won — don't relitigate)
- **Bipolar (two inverted lobes, no 180), NOT PGSE-with-180.** Kiselev & Li, "What Does FEXI
  Measure in Neurons?", arXiv:2601.20657 (2026): gf=750 mT/m, δ=4 ms, Δ=5 ms, rectangular pulses,
  no refocusing pulse. At matched filter strength bipolar and PGSE give ~the same σ (0.22 vs 0.27
  on packed R=1.5 µm), so the 180 buys nothing. An earlier PGSE detour was reverted.
- **Needs a diffusion-time GAP** `Delta > delta`. A contiguous bipolar (Δ≈δ) has too short a
  time to feel restriction → σ≈0 (the original bug).
- **FEXI must go through the vector-Bloch engine.** The filter works via the stimulated-echo
  quadrature storage (90 store keeps cos φ) + the crusher selecting that pathway — a scalar
  `chi_perp` walk (`core.simulate`) can't represent it. Exchange = membrane permeation during the
  mixing time (needs the permeation feature above).

## The remaining blocker: SUBSTRATE contrast
FEXI needs a **>=2× inter-pool ADC contrast** or σ→0. Findings:
- **Packed cylinders (single bulk D):** weak. R=1.5 µm, Δ=15 ms gives σ~0.22 and a very low
  filtered signal (~0.04) → noisy ADC' even at 60k walkers × 3 seeds. Not clean.
- **Per-compartment D (D_intra≪D_extra) + permeability: NOT ALLOWED.** dmipy-sim rejects a
  diffusivity discontinuity across a permeable wall. Physically correct too — the FEXI contrast is
  *geometric restriction* (uniform bulk D), not a bulk-D difference. So don't try per-comp-D here.
- **Best in-repo bet: packed SPHERES** (uniform D; 3D confinement → intra ADC → ~0, strong
  contrast) + permeability. `examples/fexi_axr_demo.py` uses this. Smaller R / longer Δ = stronger
  contrast (but heavier: permeable MC sub-steps ~ R/25). NOT yet confirmed to give a clean curve —
  that's the next test.
- **Faithful substrate: realistic grey-matter meshes.** ConCeG (Aird-Rossiter, Şimşek, Jallais,
  Jones, Kanari, Palombo, arXiv:2607.03286, 2026) grows neuron/glia SWC → watertight meshes for MC
  (DiSimPy), explicitly for exchange-sensitive GM (NEXI et al.). Load via `Mesh.from_ply`. This is
  the GM-permeability regime the whole thing is aimed at; getting a substrate is an external
  (~hours) pipeline.
- **Note:** `simulate_bloch` currently walks a single bulk D — it does NOT yet resolve
  per-compartment D (the scalar engine does, via `_D_comp_jax`/`classify_position`). Only needed if
  a *non-permeable* multi-D substrate is ever wanted; irrelevant for the permeable FEXI case above.

## How to test (AXR sweep)
1. GPU (GH200): set the CUDA loader path (jaxlib needs it), modest mem fraction (shared box):
   ```
   NVLIBS=$(find ~/.local/lib/python3.11/site-packages/nvidia -name '*.so*' -path '*/lib/*' \
           | sed 's:/[^/]*$::' | sort -u | tr '\n' ':')
   LD_LIBRARY_PATH="$NVLIBS" XLA_PYTHON_CLIENT_PREALLOCATE=false \
       XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 JAX_PLATFORMS=cuda python examples/fexi_axr_demo.py
   ```
2. Expect: κ=0 → ADC'(t_m) flat below equilibrium; κ>0 → recovers toward equilibrium (fitted AXR
   > 0). Tune `G_FILTER` for σ≈0.3–0.6, `R`/`DELTA_SEP` for the contrast, and κ so AXR ~ 1/t_m,max.
3. Always include the κ=0 control (Khateri, NMR Biomed 2022: geometry alone can give a small
   pseudo-AXR even without membranes — characterize it).

## Open next steps
- Confirm packed spheres give a clean AXR recovery (tune R, Δ, g_f, κ); if marginal, go finer or
  to a ConCeG GM mesh.
- Optional: a fast analytic two-site Kärger check that the fit recovers AXR = k_in+k_ex (validates
  the pipeline with zero substrate ambiguity).
- Optional: add per-compartment D resolution to `simulate_bloch` (non-permeable only).
- Docs + a pedagogy figure once a clean curve exists.
