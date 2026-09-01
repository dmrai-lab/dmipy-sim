# Susceptibility as a compressible tier: the `susc_path_dct` channel

## The problem it solves

A replay pack compresses positions because the gradient phase is **linear** in position:
`φ_G = γ∫G·r dt`, `G` is smooth, so by Parseval only the low temporal modes of `r` matter — that is what
`compile_scheme` exploits, and why `K ≪ n_t` is exact for gradients.

Susceptibility breaks that, because `φ_χ = γ∫s(t)·ΔB(r(t))dt` puts a **nonlinear** map `r → ΔB` between
the stored quantity and the observable. Truncating `r`'s spectrum perturbs `ΔB` unpredictably. Measured on
a real axon (n_t=1601): the susceptibility residual is `1.7e-1` at K=256 and `6.7e-2` at K=1024 against a
Monte-Carlo floor of `2.6e-2`, collapsing to `2.2e-4` only when the position transform is **lossless**.
That forced `K = n_t`, ~720 MB packs, and a Monte-Carlo floor 6–14× weaker than the canonical
cylinder/sphere/plane reference set — for which `ε = 1e-3` is affordable precisely *because* its positions
compress.

## The insight

The nonlinearity lives **only** in `r → ΔB`. Everything downstream is linear: the time integral, the
sequence gate `s(t)`, the `Q(H)` direction contraction, and the `χ` scaling. So evaluate the nonlinearity
once, at full walk resolution, during the build — then compress the **linear** representation.

Concretely: store the field basis **sampled along the path**, `b_c(t) = basis_c(r(t))`, temporally
compressed, instead of storing `r(t)` well enough to re-evaluate the field. The same Parseval argument
then applies verbatim, just to the right quantity.

Why truncation is then affordable is **not** that `b(t)` is smooth — it is not. Measured on a real axon,
truncating the stored channels to K=512 still leaves ~60% of the peak field unreproduced, and K=32 leaves
~90%: a walker crosses ~1 µm field structure every save step, so `b_c(t)` is broadband, and its DCT decays
slowly. (An earlier version of this document claimed the opposite; it was wrong.)

What *is* narrowband is the **observable**. The phase only ever sees `b(t)` integrated against the
sequence gate, `∫s(t)·b(t)dt`, and `s(t)` is a slow square wave — one sign flip for a spin echo, `n_p` for
a CPMG train. By Parseval that inner product depends only on the first ~`2·n_p` DCT modes of `b`, so the
field content truncation destroys is content no realisable gate can integrate. The consequence is that
**K is a declared capability, not a fidelity setting** (see *Choosing K*), and that field-space error is
the wrong metric to certify against — the gate-integrated signal is the right one.

## The channel

```
susc_path_dct       (N_w, n_ch, K) float16   temporal DCT-II coefficients of b_c(t), per walker
                    (walkers LEADING so a prefix read is one contiguous byte range -- see Precision tiers)
```
with channel metadata:
```
{ "channel": "susc_path_dct", "K": 32, "n_t": 1601, "n_ch": 6,
  "channels": ["iso_local", "iso_P_xx", "iso_P_yy", "iso_P_xy", "iso_P_xz", "iso_P_yz"],
  "iso_P_zz": "implied",        # = 3*iso_local - iso_P_xx - iso_P_yy, see Trace identity
  "chi_iso_reference": 1.06e-6, "delta_chi_a": 0.0 }
```
The channels are **geometry only** — `B0` direction, `B0` magnitude, `χ_iso` and `χ_aniso` all remain
replay knobs, applied by the same `Q(H)` contraction the grid form uses.

**f16 is sufficient** (measured, not assumed). Storing the K=32 coefficients as float16 instead of
float32 changes the replayed signal by 1.8e-08 (spin echo) and 7.9e-07 (CPMG-16), against this pack
family's Monte-Carlo floor of 9.4e-03 — four to five orders of margin. f16 halves the dominant term.

**Trace identity: 7 channels → 6 stored.** The dipole kernel is traceless in the sense that the six
`iso_P` components satisfy `P_xx + P_yy + P_zz = 3·iso_local` exactly, so `iso_P_zz` is redundant and is
reconstructed at read time. Verified to **4.7e-16** relative (machine precision) — *but only when the
source grid carries no k-space apodisation*. With `kspace_lowpass=0.5` the identity breaks (7.1e-02
relative) because the window is applied to `iso_P` and not to `iso_local`; the two are then no longer the
same field. `mesh_field_basis` therefore defaults to `kspace_lowpass=None`, which the oracle calibration
independently prefers (lumen null 0.073% of χB₀ vs 0.098%, identical sheath amplitude 0.999× analytic).
Dropping `iso_P_zz` saves 1/7 = 14% of the channel with no approximation.

> Provenance note: the 29 Winther packs in `packs_v1_weighted/` were built before this was understood,
> with the inherited `kspace_lowpass=0.5`. Their physics is unaffected (both settings clear the oracle),
> but they cannot use the trace identity and their metadata claims a setting that was not used. The
> pending rebuild picks up `kspace_lowpass=None` and the 6-channel form.

### Consuming it

*Scalar phase* — the gate enters by Parseval, exactly parallel to `compile_scheme`:
```
ŝ = dct(s)[:K]                     # compile the sequence gate once
φ_χ = γ·dt · Σ_c Q_c(H)·χ_c · (coeffs[:,c] @ ŝ)
```
*Vector-Bloch* (the primary path) — reconstruct `b(t)` and hand it to the existing per-step interface:
```
b = idct(coeffs, n=n_t, axis=-1)                   # (N_w, n_ch, n_t)
dB = χ_iso·(b[:,0] − Q(H)·b[:,1:7]) + χ_aniso·(Q(H)·b[:,7:13])     # (N_w, n_t)
replay_bloch(..., extra_phase_per_step = γ·dt·dB)
```

## Choosing K, and declaring it

K must resolve the **fastest RF modulation** in the sequences the pack is meant to serve: a CPMG train is
a comb filter that passes field fluctuations near its echo spacing, so the requirement scales with the
number of refocusing pulses (roughly `K ≳ 2×n_pulses`). Everything else — orientation, b-value, the
diffusion×susceptibility cross-term — is far less demanding.

Measured on `winther/G6/axon06` (n_t=1601, MC floor 9.4e-3), error of the **vector-Bloch** signal against
the full-resolution field:

| case | K=16 | **K=32** | K=64 |
|---|---|---|---|
| spin echo, b=0, worst over θ∈{0,30,45,60,90} | 4.1e-4 | **1.1e-4** | 9.4e-5 |
| b=1e9 ⊥, spin echo (cross-term) | 2.3e-4 | **3.4e-4** | 1.6e-4 |
| b=3e9 ⊥, spin echo (cross-term) | 3.2e-4 | **3.8e-4** | 1.3e-4 |
| b=1e9 ⊥, **CPMG 16 echoes** | 2.4e-2 ✗ | **1.5e-3** | 4.5e-4 |

**K=32 is the default**: ≥25× below the MC floor for single-refocus and gradient-on cases, 6× below it for
a 16-echo train. K=16 is *not* safe — it fails CPMG-16 outright. The pack therefore declares its
capability, alongside the acquisition envelope it already declares:
```
replay_envelope.field_modes           = 32
replay_envelope.max_refocusing_pulses = 16      # ~K/2; beyond this, rebuild with a larger K
```

## Two pack products

The scalar and Bloch paths have genuinely different needs, so one artefact serving both is bad at both.
Both are written from **one** walk and one field-basis solve, sharing provenance:

| | stores | per walker | serves |
|---|---|---|---|
| **light** | positions + gated moments `∫s·b_c dt` | ~460 B | PGSE/scalar-phase fitting, any B0/χ, declared gate family |
| **Bloch** | positions + `susc_path_dct` (6×K, f16) | **840 B** (K_pos=32, K_susc=32) / 1224 B (K_susc=64) | emergent RF: CPMG, stimulated echo, MT, any B0/χ |

Neither carries the **static field-basis grid**. That is not merely a size choice — it is a *correctness*
constraint. The grid route (`replay_susc` via `sample_grid`) evaluates the field at codec-**decoded**
positions, so it is only sound while the position codec is lossless; the path route samples the
full-resolution walk at build time, which is exactly what frees the positions to be lossy. Shipping both
next to lossy positions would advertise a replay route whose accuracy silently depends on a property the
pack no longer has, so `build_replay_pack` omits the grid arrays whenever `susc_path_K` is set and the
positions are not lossless, and records `replay_route` in the channel metadata. The grid ships as a
companion artefact (`<id>.field.rpk`, ~144 MB cropped f16, per-substrate not per-walker) for regenerating
the channel at a different K or for a new walk.

## What this buys

Measured, for one Winther axon at `n_t=1601`, `N_w=52,000`:

| | per walker | pack |
|---|---|---|
| lossless positions (K=n_t f32) + in-pack f16 field grid | 19.3 kB | **1100 MB** |
| positions K_pos=128 f32 + `susc_path_dct` 6×32 f16 + C2 + weights | 1.99 kB | **≤ 98.7 MB** |
| same at K_susc=64 (CPMG ≤ 32) | 2.37 kB | ≤ 117.7 MB |

`K_pos = 128` in that row is a conservative placeholder. The assembler auto-selects the position modes
against the gradient envelope, and it is now free to pick a small number because susceptibility no longer
constrains it. **Measured at production `n_t=1601`**, by decoding a v1 pack (whose positions are lossless,
so they give the true trajectory) and sweeping K against the floor the 52,000-walker ensemble will have
(6.34e-3, extrapolated from a 6,000-walker subsample and consistent with this family's measured 9.4e-3 at
31,000):

| K_pos | position-codec err | err / floor(52k) | pack at K_susc=32 | at K_susc=64 |
|---|---|---|---|---|
| 8 | 7.08e-03 | 1.12 | 27.4 MB | 46.4 MB |
| 16 | 1.60e-03 | 0.25 | 31.7 MB | 50.7 MB |
| **32** | **2.55e-04** | **0.04** | **41.7 MB** | **60.7 MB** |

So a production pack is **27--61 MB**, not the ~99 MB placeholder: **18--40x** smaller than v1's 1100 MB.

**Do not simply take the auto-selected K here.** With the default `tol = 2x floor` the selector passes
`K_pos = 8`, whose codec error (7.1e-3) *exceeds* the statistical floor (6.3e-3) -- the compression would
be the dominant error source. That is an acceptable trade for an ordinary pack and the wrong one for a
reference dataset, whose whole claim is that replay is indistinguishable from the walk. `K_pos = 32` puts
the codec 25x below the floor for ~288 B/walker, and is the recommended pin.

An end-to-end build on a synthetic myelinated cylinder (n_t=201) confirms the rest of the design:
`route=path`, `zz=implied` (6 channels), C2 present, prefix unbiased (23.9% myelin in the first 2,000 rows
vs 23.9% overall), weights collapsed to max/min=1.008, and `err_susc_path = 2.9e-4` at K=32 against a
3.0e-2 floor (103x below; 243x at K=64).

**The cross-term survives the position truncation**, which is the thing to check before accepting any of
this: the reported physics is diffusion × susceptibility, so it would be no good to compress positions if
that decorrelated the two phases. It does not, because the codec is *walker-preserving*: walker `w`'s
gradient phase and its field history carry the same index, so `exp(i(φ_G + φ_χ))` is formed per walker
before any ensemble average and their correlation is exact by construction. Position truncation perturbs
`φ_G` alone, by the amount the gradient tier already certifies against the MC floor — it cannot leak into
`φ_χ`, which no longer depends on the stored positions at all.

**11×** smaller. The saving is *not* mainly the field tier being cheap — it is that the susceptibility tier
no longer has a claim on the position codec. Lossless positions were never wanted for their own sake; they
were forced because grid-sampling needed exact `r(t)`. Remove that coupling and positions fall back to
`K ≪ n_t`, which is all the gradient tier ever needed (Parseval-exact).

Prefix reads then make the *typical* pull small: `ε=1e-2` is ~1 MB, `ε=3e-3` ~11 MB, and the full
`ε=1e-3` reference ~99 MB — so a user exploring 29 axons pulls tens of MB, not tens of GB.

## The C2 (surface / boundary-local-time) tier belongs in the budget

Easy to forget, because an impermeable membrane has no *exchange* tier -- but the intra-axonal walkers
strike the myelin inner wall constantly (measured contact density 12.1% of steps at n_t=1601), so the
**surface tier is real**: rho is a replay knob, not zero by construction. It is also the channel an
analytic MT tier is derived from (contact statistics / S:V at the myelin surface). The first generation of
axon packs discarded it -- `simulate_trajectories(save_relaxation_data=True)` was called and its
`dlog_boundary_unit` output dropped -- so those packs can do neither surface relaxivity nor MT.

Its cost depends entirely on the codec, and **the default picks the wrong one**:

| codec | B/walker at n_t=1601 | scales with n_t? | ensemble error (rho=1e-5..1e-4) |
|---|---|---|---|
| sparse CSR (**auto-selected**) | 574 | **yes** (cost ~ contact count) | exact |
| `boundary_dct` K=32, f16 | **66** | no | 6e-6 .. 7e-6 |
| `boundary_dct` K=64, f16 | 130 | no | 2e-6 .. 2e-5 |

against a split-half floor of 1.9e-2 -- 2 to 5 thousand times the truncation error. So **K=32 in f16 at
66 B/walker** is the right setting, and it keeps the tier flat in walk length.

Two defaulting problems produce the 574 B outcome and both need fixing:
1. `encode_boundary_local_time` chooses only between *sparse* and *dense*; `encode_boundary_bridge` is
   opt-in via `blt_temporal_K`. For a long mesh substrate the default therefore takes the expensive,
   n_t-scaling branch. The producer must pass `blt_temporal_K` (or the encoder should consider the bridge
   branch in its size comparison).
2. `encode_boundary_bridge` writes float32, while the canonical packs store `blt_bridge_dst` as float16 -- which is
   where the 2x difference between 130 B and 260 B comes from. f16 is demonstrably sufficient here.

### Revised per-walker budget

| component | f32 positions/field | all-f16 |
|---|---|---|
| positions, K=64 | 768 B | 384 B |
| `susc_path_dct`, 7x32 | 896 B | 448 B |
| **C2 `blt_bridge_dst`, K=32 f16** | **66 B** | **66 B** |
| compartment RLE + weights | ~18 B | ~18 B |
| **total** | **~1.75 KB** | **~0.92 KB** |

C2 is therefore ~4% of the pack, not the ~14% it is in the canonical shards and not the ~32% the auto
codec would make it. Whether the field channel itself survives f16 is untested -- the spin-echo
cancellation is precision-sensitive -- and that is the single remaining factor-of-two on the table.

## Precision tiers: one artefact, pull only what your accuracy needs

A 2--5 GB download per axon is unusable for anything but reference work, and it is also unnecessary. The
Monte-Carlo floor scales as `1/sqrt(N)`, and **walkers are the row dimension of every per-walker tensor**,
so a consumer who wants a coarser floor simply reads *fewer rows*. `safetensors` supports exactly this:
`safe_open(...).get_slice(name)[0:N]` reads only those bytes (measured: 1 000 rows = 19.2 MB in 0.002 s,
10 000 rows = 192 MB in 0.024 s, linear in N), and because the header carries per-tensor byte offsets a
remote client can range-GET a prefix instead of the file.

With the measured per-walker budget above (**840 B** at K_pos=32/K_susc=32, C2 and weights included),
anchored on the measured floor of `9.4e-3` at 31 000 walkers (`n(eps) = 31000*(9.4e-3/eps)^2`):

| tier | floor eps | walkers | bytes per axon |
|---|---|---|---|
| **browse / teaching** | 0.03 | ~3,044 | **~2 MB** |
| **working** | 0.01 | ~27,392 | **~22 MB** |
| high | 0.003 | ~304,351 | ~244 MB |
| reference | 0.0017 | ~947,806 | ~759 MB |
| canonical parity | 0.001 | ~2,739,160 | ~2.14 GB |

Three things must hold for a prefix read to be *valid*, and one of them was violated by the first
generation of packs:

1. **Walkers must be stored SHUFFLED.** The pools were stacked (all intra, then all myelin), so the first
   10 000 rows of an existing pack are 10 000 intra and *zero* myelin -- a prefix reader would silently get
   an axon with no myelin pool. Walkers are exchangeable, so a deterministic shuffle at build time fixes it;
   verified afterwards, a shuffled prefix is unbiased with error inside the `1/sqrt(N)` envelope
   (N=1 000 -> 1.2e-4, N=10 000 -> 5.4e-5 against the full-pack value).

   Implemented as: `mesh_axon_master` applies one permutation to every walker-indexed array
   (`traj`, `dlog_b`, `comp`, `comp0`, `w`) and sets `walkers_shuffled=True` on the master;
   `build_replay_pack` copies that into `compression.precision_tiers`, which also carries
   `bytes_per_walker`, `floor_at_full_n` and the per-eps walker counts so a consumer computes the
   prefix length rather than guessing it. **The producer asserts the property — the assembler cannot
   detect it.** A pack whose producer does not declare it gets `usable: false` and an explicit warning
   note instead of tiers that quietly invite a biased read.
2. **Walkers must be the leading axis** of every per-walker tensor, so a prefix is one contiguous byte
   range: `dct_coeffs` is `(N_w, K, 3)` already, and `susc_path_dct` is therefore specified as
   `(N_w, n_ch, K)` -- *not* `(n_ch, N_w, K)`, which would make a walker prefix a strided scatter.
   The per-walker compartment RLE is unaffected: each walker's label is constant in time, so it is one run
   per row whatever the row order.
3. **The tier map must be declared**, not left to the reader to guess:
   ```
   precision_tiers = [{"eps": 3e-2, "n_walkers": 3000}, {"eps": 1e-2, "n_walkers": 27000},
                      {"eps": 3e-3, "n_walkers": 304000}, {"eps": 1e-3, "n_walkers": 2740000}]
   ```
   derived from the pack's own measured floor, so `eps(N) = floor_measured * sqrt(N_full / N)`.

**The field grid ships as a companion artefact** (`<id>.field.rpk`), not inside the pack. Once the path
channel exists the grid is not needed for replay at all -- only for regenerating the channel at a different
K or for a new walk -- and at ~144 MB it would otherwise dominate a 45 MB working-tier pull.

This makes the size question a consumer choice rather than a dataset property: the bank stores one
reference-grade artefact per substrate, and a user pulling a single mesh replay for a fitting experiment
takes ~45 MB of it.

## Fidelity gate

The channel must be certified the way every other tier is: reconstruct at the stored K, compare against
the **full-resolution** field along the same walk, over a battery that **includes a CPMG train** (the
binding case), and require the error to sit below the pack's own split-half Monte-Carlo floor. The
existing `_susc_grid_fidelity` certifies grid quantisation and position-codec error; this one certifies
the temporal truncation, and both belong in the pack's `fidelity` block.

## Limitations

- The path channel is **tied to its walk**. Arbitrary spatial questions (re-seeding, sampling the field at
  new positions) need the grid, which is why the grid is retained.
- K is a **declared capability**, not a universal constant. A pack built for 16-echo trains is not valid
  for a 64-echo train; the envelope says so, and the grid allows a rebuild.
- Anisotropic (`aniso_G`) channels were not exercised in the measurements above — the packs measured were
  built isotropic (matching their source study). Enabling them raises `n_ch` 7 → 13 and needs the same
  K-sweep before being trusted.
- MT with emergent binding evolves a bound pool sequentially; it uses the same reconstructed `b(t)`, but
  its own mode requirement has not been measured separately.
