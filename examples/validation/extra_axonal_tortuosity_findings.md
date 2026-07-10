# Extra-axonal tortuosity: same volume fraction, size-dependent D_perp

**Claim.** The extra-axonal perpendicular diffusivity is *not* fixed by the volume fraction
`v_ic` alone. At a fixed fibre fraction (f = 0.50), scaling the pack size changes the
emergent `D_perp` at a fixed diffusion time Δ:

| pack scale | mean fibre diam. | `t_c ~ R²/2D` | `D_perp / D_free` (Δ = 20 ms) |
|---|---|---|---|
| ×1  | 0.67 µm  | 0.03 ms | 0.514 |
| ×4  | 2.67 µm  | 0.52 ms | 0.536 |
| ×16 | 10.67 µm | 8.37 ms | 0.651 |

(N = 30 000 walkers, `PackedCylinders`, low b, step-resolved; small-N noise ~1–2 %.)

**Mechanism (Novikov–Fieremans structural disorder; Burcaw–Fieremans–Novikov 2015).**
Time-dependent diffusion `D(t) → D_∞` as walkers coarse-grain over the disorder. Two facts:

1. **The DC tortuosity limit `D_∞/D` is scale-invariant** — for hard cylinders it depends only
   on the area fraction `f` and the arrangement, not the absolute cylinder size. All three
   packs share the same `D_∞ ≈ 0.51·D` (the ×16 pack relaxes back to it at Δ ≫ `t_c`).
2. **The correlation time `t_c ~ ℓ_c²/2D ∝ R²` scales with size.** So at a *fixed* Δ, the
   dimensionless time `Δ/t_c` — hence where you sit on the crossover — depends on size.

**Consequence for modelling.**
- **Small pack (sub-micron axons), `t_c ≪ Δ`:** `D_perp` is on its DC plateau, essentially
  time-independent (Burcaw `A ≈ 0`). A *fixed* tortuosity `lambda_perp = D·(1 − f)` (a plain
  `G2Zeppelin`) is sufficient. **This is the white-matter-at-clinical-Δ regime** — our canonical
  substrate (`t_c ~ 0.03–0.5 ms` ≪ Δ ~ 20–60 ms).
- **Large pack (~10 µm), `t_c ~ Δ`:** `D_perp` is caught mid-crossover, elevated, and genuinely
  Δ-dependent (`A > 0`). Here a temporal Zeppelin (`G3TemporalZeppelin`,
  `D_perp(δ,Δ) = λ_inf + A·(ln(Δ/δ)+3/2)/(Δ−δ/3)`) is required.

So the choice between a fixed tortuosity constraint and a temporal Zeppelin is **not a
modelling taste** — it is set by axon calibre relative to Δ, and this sweep *measures* the
boundary. Observable radial time-dependence in real WM (Burcaw 2015 et seq.) therefore points
to **mesoscopic** correlation lengths (~µm: beading, undulation, caliber variation along the
axon, fascicle clustering) that push `t_c` into the clinical-Δ window — not the sub-micron
packing itself.
