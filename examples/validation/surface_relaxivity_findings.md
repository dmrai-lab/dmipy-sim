# Surface relaxivity — first-principles validation findings (1D → 2D → 3D)

Investigation of surface relaxivity in dmipy-sim, from the geometry ladder 1D → 2D → 3D,
against **exact** analytics. Companion to `surface_relaxivity_1d_to_3d.py` and the sibling
`permeability_findings.md` — the same N-D template, for the other wall process.

**Bottom line: the surface-relaxivity weight rule delivers exactly the Robin boundary
condition.** A closed relaxing cell's magnetisation decays, at long time, as `exp(-t/τ₁)` with
`τ₁` set by the lowest Robin eigenvalue; the Monte-Carlo matches that exact finite-diffusion
`τ₁` to sub-percent in 1D/2D/3D. The fast-diffusion idealisation `1/τ = ρ·S/V` (Brownstein–Tarr
fast limit) is only the `ρR/D → 0` corner and is the *wrong* reference at finite `ρR/D` — exactly
as the well-mixed `V/(κS)` is for permeability.

Parameters throughout: D = 2×10⁻⁹ m²/s, ρ = 2×10⁻⁵ m/s, R = L/2 = 10 µm (ρR/D ≈ 0.1).

---

## The MC rule → the boundary condition
Each wall encounter multiplies the walker's magnetisation by `exp(dlog_w)` with

    dlog_w = −2·(ρ/D)·d_perp

(`d_perp` = perpendicular overshoot past the wall; `reflect_with_log_weight` in
`geometries.py`). Integrating this survival weight over the near-wall Gaussian excursion is the
standard construction whose continuum limit is the **Robin (partially-absorbing) boundary
condition** `D ∂c/∂n = −ρ c` (Brownstein & Tarr 1979). So the *only* correct reference is the
Robin-BC diffusion eigenproblem, whose lowest mode governs the long-time decay,
`τ₁ = 1/(D λ₁²)`:

| dim | geometry (relaxing wall) | lowest-Robin transcendental | fast limit `1/τ=ρS/V` |
|---|---|---|---|
| 1D | slab width L, **both** walls | `λ·tan(λL/2) = ρ/D` | `2ρ/L` |
| 2D | cylinder, wall at R | `D·λ·J₁(λR) = ρ·J₀(λR)` | `2ρ/R` |
| 3D | sphere, wall at R | `1 − λR·cot(λR) = ρR/D` | `3ρ/R` |

All three are the `ρ/D` analogue of the permeable-cell eigenvalues; the flat-slab form is the
same transcendental as the permeable slab with `ρ/D` in place of `2κ/D` (one absorbing face vs a
two-sided membrane).

## Eigenvalue-solver check — recovers the fast limit as ρR/D → 0 (no GPU)
Before touching the MC, the Robin solver was checked against the **only** regime with a
closed-form answer, the fast-diffusion limit `τ → V/(ρS)`:

| geometry | τ_exact (ρR/D≈0.1) | fast-limit ρS/V | finite-diffusion excess |
|---|---|---|---|
| 1D slab | 516.78 ms | 500.00 ms | +3.4% |
| 2D cylinder | 256.30 ms | 250.00 ms | +2.5% |
| 3D sphere | 170.03 ms | 166.67 ms | +2.0% |

Sphere ratio `τ_exact/τ_fast` as ρ is lowered: **1.0202 (ρ=20 µm/s) → 1.0020 (2 µm/s) →
1.0002 (0.2 µm/s)** — clean convergence to 1 as `ρR/D → 0`. The solver is correct, and the
finite-`ρR/D` excess (walkers must diffuse to the wall before relaxing — depletion) is real
physics that the fast limit omits.

## Ladder result — DONE (1D→2D→3D certified vs exact Robin eigenvalues)
Monte-Carlo relaxation time (long-time single-exponential fit of S(t), 500k walkers, step R/50)
vs the exact eigenvalue `τ₁`:

| rung | geometry | exact `τ₁` | MC vs exact | fast-limit `ρS/V` (the wrong ref) |
|---|---|---|---|---|
| 1D | `Box1D` (both walls relaxing) | 516.78 ms | **+0.00%** ✅ | 500.00 ms (−3.2%) |
| 2D | `Cylinder` (wall at R) | 256.30 ms | **+0.16%** ✅ | 250.00 ms (−2.5%) |
| 3D | `Sphere` (wall at R) | 170.03 ms | **+0.24%** ✅ | 166.67 ms (−2.0%) |

All three land <0.3% against the exact finite-diffusion eigenvalue — on par with the
permeability ladder. The 2–3% offset from `ρS/V` is the correct diffusion-limited near-wall
depletion (the compartment is not instantly well-mixed), *not* an engine error: it shrinks to
zero as `ρR/D → 0`, exactly where the fast limit becomes valid. Unlike the permeable crossing,
**no sub-stepping was needed** — the per-encounter survival weight `−2ρd⊥/D` is already accurate
at the R/50 step, confirming the two wall processes have different step sensitivities.



## How this ships (docs + heavy GPU test, one source)
- **Documentation / demo:** `examples/validation/surface_relaxivity_1d_to_3d.py` — runnable;
  prints the ladder (each rung MC vs exact eigenvalue, with the fast-limit shown as the wrong
  reference). This write-up is its prose.
- **Regression:** `tests/validation/test_surface_relaxivity_ladder.py` — `@pytest.mark.slow`
  GPU test asserting each rung <2% vs the exact Robin eigenvalue. Run with `pytest -m slow`;
  the fast suite skips it (`-m 'not slow'`).

## Relation to permeability
Same closed-cell / exact-eigenvalue template as `permeability_findings.md`, and the same lesson:
**validate against the exact finite-diffusion eigenvalue, never the fast/well-mixed shortcut.**
Note the two wall processes are step-sensitive to different degrees — the permeable *crossing*
needed sub-stepping to ~R/25 to avoid over-permeation, whereas the relaxivity *weight*
`−2ρd⊥/D` is a per-encounter survival factor that is already accurate at the R/50 step used
here (no sub-stepping); the ladder confirms it.
