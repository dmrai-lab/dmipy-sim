# Permeability — first-principles validation findings (1D → 2D → 3D)

Investigation of membrane permeability in dmipy-sim, from the geometry ladder 1D → 2D → 3D
→ mesh, against **exact** analytics. Companion to `permeability_1d_to_mesh.py`.

**Bottom line: the permeability crossing is correct — the transmission gives exactly κ.**
The apparent "failures" were all comparisons to *idealised* formulas (well-mixed
`τ = V/(κS)`, open-domain `exp(-t/τ)`) that omit two pieces of **correct** physics:
diffusion-limited near-membrane depletion, and (for a single object in an open domain)
exterior re-entry. Against the *exact* solution the Monte-Carlo matches to sub-percent.

Parameters throughout: D = 2×10⁻⁹ m²/s, κ = 2×10⁻⁵ m/s, R = L = 10–20 µm (κR/D ≈ 0.1–0.2).

---

## Symptom
Two single-sphere residence-time tests failed: fitted τ = 179 ms vs τ=R/(3κ)=167 ms (+7.5%);
f_inside(3τ)=0.066 vs exp(−3)=0.050 (+33%). Both = *under*-decay (walkers leave slower than
the idealised well-mixed rate).

## Experiment A — step-size convergence (is it discretisation?)
Swept the sub-step from R/25 → R/50 → R/100 for the sphere and cylinder residence time.
**Result: step-INDEPENDENT** — τ error flat at ~8% (sphere) / ~14% (cyl). Finer steps did
**not** help. → It is *not* a discretisation error. (This also disproves the old
`test_permeability.py` docstring, which attributed the cylinder bias to "step-size
discretisation ~20%".)

## Experiment B — open-domain initial slope (re-entry fingerprint)
Measured τ_eff from f_inside at short times for the single sphere/cylinder in an open domain.
**Result: the error GROWS with time** (sphere +2.7%→+4.5%, cyl +6.2%→+8.9% over 0.1–0.3 τ) and
is **larger for the 2D cylinder than the 3D sphere**. Growth-with-time + ordering by exterior
dimensionality (2D recurrent ≫ 3D transient) is the signature of **exterior re-entry**: a
single permeable object in an open domain is not a well-mixed reservoir, so f_inside decays
slower than `exp(-t/τ)`.

## Experiment C — t→0 efflux, high statistics (transmission prefactor)
Sphere/cylinder efflux at t = 2/4/8 ms (2×10⁶ walkers). τ_pt grows linearly from ~0 at t=0:
sphere 168.0→170.7→175.4, cyl 256.4→260.9→265.6. **Linear extrapolation to t→0: sphere
≈165.5 (−0.7%), cyl ≈253 (+1.3%)** — i.e. the *rate* is right; the finite-t excess is re-entry
building up. Analytic check: for a uniform starting density the flux onto the membrane is
exactly κ·c, so the t→0 efflux rate is exactly κ·S/V (derivable by integrating the
transmission `p = 2κ d⊥/D` over the Gaussian step distribution — it yields flux = κc to first
order). The prefactor is correct.

## Experiment D — 1D closed slab vs EXACT eigenvalue (the decisive test)
Added a `PermeableSlab1D` geometry (closed two-compartment slab: permeable membrane at L/2,
reflecting outer walls) — no curvature, no re-entry (a closed reservoir). Compared the MC
exchange time to two references:
- **well-mixed** `τ = L/(4κ) = 250 ms` — the idealisation;
- **exact finite-diffusion eigenvalue** `τ₁ = 1/(D λ₁²)` with `λ·tan(λL/2) = 2κ/D` → **266.9 ms**.

**Result (step-independent): MC = 265.4–266.5 ms → +6.6% vs well-mixed, but −0.14…−0.56% vs the
EXACT eigenvalue.** The +6.6% was entirely the well-mixed approximation being wrong at finite
κL/D (the compartment is not instantly well-mixed — near-membrane depletion). The engine
reproduces the exact theory. **This is the proof: permeability is correct to sub-percent.**

## Experiment E — closed-slab initial slope (√t short-time law)
The t→0 efflux rate of the closed slab: 1.94 (t=4 ms) → 1.90 → 1.86 (t=16 ms) per second.
These fit `rate(t) = 2κ/L − b·√t` (a Mitra-like √t correction), extrapolating to the exact
`2κ/L = 2.0 /s` at t→0. **Lesson:** the prefactor is exact, but the short-time approach is
√t, so a finite-t or through-origin fit under-reads the rate (this is why the first draft of
`permeability_1d_to_mesh.py` reported the initial-flux rungs ~8–11% low — a benchmark-
extraction error, not an engine error).

---

## Conclusions
1. **The transmission rule `p = min(1, 2κ d⊥/D)` is correct** — it delivers exactly permeability
   κ. Validated to −0.56% against the exact 1D eigenvalue, and the t→0 flux equals κS/V.
2. **It works in any regime and at any step size** (step-independent; no fast/slow-exchange
   caveat on the crossing itself).
3. **Deviations from well-mixed / open-domain-exponential idealisations are CORRECT physics:**
   diffusion-limited near-membrane depletion (√t short-time, finite-κL/D exchange slowdown) and
   exterior re-entry for open domains (stronger in lower-dimensional exteriors).
4. **Validate against EXACT analytics, never the well-mixed shortcut.** For exchange/residence
   time use closed reservoirs (a slab, or periodic packing) or the exact eigenvalue/effective-D;
   a single object in an open domain must be compared to the re-entry-inclusive solution.

## Experiment F — initial-flux rung is a fragile benchmark
Tried to validate 2D/3D/mesh with the t→0 efflux rate = κS/V, extracted by √t-extrapolation
of (1-f)/t over short times. **Result: too noisy to certify** — 1D −8.9%, 2D +6.2%, 3D +5.0%,
signs flipping run to run. The short-time window trades statistics (tiny 1−f) against the √t
depletion onset; the extrapolation is unstable. This is a **benchmark-extraction** limitation,
not an engine error (the 1D eigenvalue already proves the prefactor). **Takeaway: use the
closed-system exact eigenvalue, not initial-flux, as the certifying benchmark.**

## Experiment G — 3D closed sphere shell vs EXACT spherical-Bessel eigenvalue
Added `PermeableShell` (permeable inner membrane R_in + reflecting outer wall R_out, closed →
no re-entry). Exact reference = lowest spherical-Bessel exchange eigenvalue (solver validated:
ratio exact/well-mixed → 1.0005 as κR/D → 0). For R_out=2R_in, κR_in/D=0.1, exact τ₁=153.1 ms.
**MC over-permeates**, and it is TWO effects:
- step-resolution: τ error −4.2% (R_in/25) → −2.6% (R_in/50) → −2.5% (R_in/100) — curved
  membranes need a finer sub-step than the flat R_in/25 floor;
- a **residual over-permeation that plateaus** with finer steps.
At this point I hypothesised a genuine curvature bias in the flat-tangent transmission — **this
was wrong**; Experiment H shows it was a detailed-balance bug in the scaffold. (Kept here as the
honest trail: the right move was to check detailed balance, not to invent a correction.)

## Experiment H — the "curvature bias" was a DETAILED-BALANCE BUG, not physics
Ruled out `d⊥` as the lever (tangent −2.7%, radial −3.4%: both over-permeate, so penetration
depth isn't it). Then the decisive diagnostic — the **equilibrium partition** (must equal
V_A/V_total for a passive membrane): it plateaued at **f_A = 0.121, not 0.125 (−3%, 12σ)** →
**detailed balance was violated** → a *directional* transmission bias, i.e. an implementation
artifact in the two-boundary `PermeableShell`, not curved-membrane physics.

Root cause: reflected walkers were left *straddling* the curved membrane (no nudge), biasing
the next step's crossing. Adding the same **nudge** `Sphere.permeate` uses (push the reflected
walker 10⁻⁴·R onto its own side) restored it:
- equilibrium **0.121 → 0.124** (−3% → −0.7%);
- rate **148.9 → 152.96 ms vs exact 153.09 (−0.08%)** — the flat-tangent `d⊥` is *correct*
  (radial −0.83%, slightly worse).

**Conclusion: there is NO curvature-corrected transmission to derive.** The transmission physics
was always right (1D exact); the sphere shell now matches the exact spherical-Bessel eigenvalue
to −0.08%, on par with the 1D slab. The honest diagnostic path (detailed balance → nudge)
found a real bug in the validation scaffold, which is the far better outcome than a fabricated
"correction."

## Ladder status — DONE (1D→2D→3D certified vs exact eigenvalues)
| rung | geometry | exact benchmark | MC vs exact | equilibrium (detailed balance) |
|---|---|---|---|---|
| 1D | `PermeableSlab1D` | closed-slab `λ tan(λL/2)=2κ/D` | **−0.14%** ✅ | 0.500 (exact 0.500) ✅ |
| 2D | `PermeableShell(kind='cylinder')` | Bessel-J/Y transcendental | **−0.28%** ✅ | 0.250 (exact 0.250) ✅ |
| 3D | `PermeableShell(kind='sphere')` | spherical-Bessel transcendental | **−0.08%** ✅ | 0.124 (exact 0.125) ✅ |

**Two scaffold bugs were found and fixed along the way**, both in the new `PermeableShell`
(not the core transmission): (1) missing reflection **nudge** → detailed-balance violation
(sphere −3% → −0.08%); (2) the radial-intersection quadratic assumed `|d|=1`, wrong for the
cylinder's perpendicular projection → add `|perp(d)|²` (cylinder +574% → −0.28%). The
transmission rule `p=2κd⊥/D` (flat-tangent `d⊥`) was correct throughout.

## How this ships (docs + heavy GPU test, one source)
- **Documentation / demo:** `examples/validation/permeability_1d_to_3d.py` — runnable; prints
  the ladder (each rung MC vs exact eigenvalue + detailed balance). This write-up is its prose.
- **Regression:** `tests/validation/test_permeability_ladder.py` — `@pytest.mark.slow` GPU test
  that imports the example and **asserts** each rung <2% vs the exact eigenvalue + detailed
  balance. Run with `pytest -m slow` (a GPU box); the fast suite skips it (`-m 'not slow'`).

## Mesh — NOT in the public v1 (deferred, WIP)
The **public v1 is scoped to analytical/parametric geometries** (slab, cylinder, sphere,
ellipsoid, packed cyl/sph, myelinated). `Mesh` and `LabelMap2D` (numerical surfaces) are
**removed from the public release** because mesh permeability is not yet certified:
a `MeshShell` (mesh membrane + reflecting outer) vs the spherical-Bessel eigenvalue gives
**sub=3 (642 verts) −35%, sub=4 (2562 verts) −5.5% τ; equilibrium −9.7% → −2.8%** — it
*converges* with vertex count but is nowhere near the analytical <0.3%, and is partly
confounded by a crude `|r|<R_in` occupancy proxy. Mesh + `MeshShell` permeability is not yet
part of this release: the next step is to fix the occupancy test (true point-in-mesh), push
vertices, and re-run the same eigenvalue harness before mesh permeability lands here.
