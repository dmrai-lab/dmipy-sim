# Susceptibility as a compressible tier: the `susc_path_dct` channel

## The problem it solves

A replay pack compresses positions because the gradient phase is **linear** in position:
`φ_G = γ∫G·r dt`, `G` is smooth, so by Parseval only the low temporal modes of `r` matter — that is what
`compile_scheme` exploits, and why `K ≪` the number of block ends is exact for gradients.

Susceptibility breaks that, because `φ_χ = γ∫s(t)·ΔB(r(t))dt` puts a **nonlinear** map `r → ΔB` between
the stored quantity and the observable. Truncating `r`'s spectrum perturbs `ΔB` unpredictably. Measured on
a real axon (1601 instants): the susceptibility residual is `1.7e-1` at K=256 and `6.7e-2` at K=1024 against a
Monte-Carlo floor of `2.6e-2`, collapsing to `2.2e-4` only when the position transform is **lossless**.
Evaluating the field from stored positions forces `K` to the full length of the walk, ~720 MB packs, and a
Monte-Carlo floor 6–14× weaker than the canonical cylinder/sphere/plane reference set — for which `ε = 1e-3`
is affordable precisely *because* its positions compress.

## The insight

The nonlinearity lives **only** in `r → ΔB`. Everything downstream is linear: the time integral, the
sequence gate `s(t)`, the `Q(H)` direction contraction, and the `χ` scaling. So evaluate the nonlinearity
once, **in the walk**, at the walker's own position — then compress the **linear** representation.

Concretely: the walk reads the field basis at its position every `field_read_interval` seconds, `b_c(t) =
basis_c(r(t))`, and accumulates the **cosine modes** of the trapezoid of the reads, `c_k = ∫ e_k b_c dt` with
`e_0 = 1/√T`, `e_k = √(2/T) cos(πkt/T)` over the storage window `T`. The pack stores those modes instead of
`r(t)` well enough to re-evaluate the field. The same Parseval argument then applies verbatim, just to the
right quantity. Nothing is stored along the walk: the accumulator holds `K_c3` modes per channel whatever
the number of reads.

Why truncation is then affordable is **not** that `b(t)` is smooth — it is not. Measured on a real axon,
truncating the stored channels to K=512 still leaves ~60% of the peak field unreproduced, and K=32 leaves
~90%: a walker crosses ~1 µm field structure between consecutive reads, so `b_c(t)` is broadband, and its
cosine spectrum decays slowly.

What *is* narrowband is the **observable**. The phase only ever sees `b(t)` integrated against the
sequence gate, `∫s(t)·b(t)dt`, and `s(t)` is a slow square wave — one sign flip for a spin echo, `n_p` for
a CPMG train. By Parseval that inner product depends only on the first ~`2·n_p` modes of `b`, so the
field content truncation destroys is content no realisable gate can integrate. The consequence is that
**K is a declared capability, not a fidelity setting** (see *Choosing K*), and that field-space error is
the wrong metric to certify against — the gate-integrated signal is the right one.

## How the walk reads the field

The read interval is derived, not chosen: `engine.accumulate.field_read_interval(D, T_e, grad_b, B0, chi,
floor)` holds the read error `v_read = (γ B0 χ)² 2 D T_e grad_b² dt_f²` under `FIELD_READ_FLOOR² = 1e-6`
(the floor is 1e-3) at `B0 = 7 T` and `χ = 1.06e-6` unless the envelope or the spec say otherwise, with
`grad_b` the basis's largest gradient over the sheath (`gradient_scale()`), `T_e` the storage window and `D`
the walk's diffusivity. The walk reads at the largest divisor of its steps per block within that interval;
when the rule asks for less than the physics step, **the read is the step** (a read along the chord between
two steps carries the same bridge deviation), and the pack's envelope states the `B0·χ` product the term
holds under (`B0_chi_max_T`). `CoefficientWalk.block["field_read"]` records the interval, the number of reads,
the rule and its term (`field_read_term`), and the certificate carries it as `fidelity.terms.field_read`.

A gridded basis (`FieldGrid`, from `fields.susceptibility_field.field_grid_of`) is read by
`simulate_trajectories(field_basis=)`; a strand basis (`StrandFieldBasis`) by the adaptive producer, at the
end of every `n_read`-th round. `walk_spec` builds and passes the basis whenever a pool of the spec is a field
source: there is no switch and no deferral.

## The channel

```
susc_path_dct       (N_w, n_ch, K) int8 | int16    cosine modes c_k of b_c(t), per walker
susc_path_scale     (n_ch, K) float32              per-(channel, mode) scale of the integer container
                    (walkers LEADING so a prefix read is one contiguous byte range -- see Precision tiers)
```
A container of several depths stores `susc_path_dct_b<i>[u<bits>]` per mode range instead (the same ranges
as the band containers of `compression`). The channel metadata, `compression.channels.susc_field_basis`:
```
{ "channel": "susc_field_basis", "K": 32, "T": 0.1, "n_ch": 6,
  "channels": ["iso_local", "iso_P_xx", "iso_P_yy", "iso_P_xy", "iso_P_xz", "iso_P_yz"],
  "iso_P_zz": "implied",        # = 3*iso_local - iso_P_xx - iso_P_yy, see Trace identity
  "bits": 8, "max_refocus_pulses": 16, "sampling": "accumulated_in_walk",
  "read_interval": 1.0e-4, "band": {...the pilot's record...}, "read": {...the walk's read record...} }
```
There is no `n_t` and no `dt`: the modes are a function on the window `[0, T]`, read at whatever instants a
consumer names. The channels are **geometry only** — `B0` direction, `B0` magnitude, `χ_iso` and `χ_aniso` all
remain replay knobs, applied by the same `Q(H)` contraction (`bank.susc_path_field`).

`compression.channels.susceptibility_grid` records only the field's placement (origin, voxel size, shape,
periodicity, `arrays_in_pack: false`, `replay_route: "path"`): the grid is not in the pack, and every
consumer reads the path.

**The integer container is sufficient** (certified, not assumed). The depth, 8 or 16 bits per mode with a
per-(channel, mode) scale, is chosen with the band on the pilot (*Choosing K*): the narrower container
is taken whenever its codec error on the certificate's battery is within the pilot's floor.

**Trace identity: 7 channels → 6 stored.** The dipole kernel is traceless in the sense that the six
`iso_P` components satisfy `P_xx + P_yy + P_zz = 3·iso_local` exactly, and the identity is linear, so it
holds in mode space: `iso_P_zz` is redundant and is reconstructed at read time. The encoder leaves it out
when the modes satisfy the identity to `atol_trace` (recorded as `trace_residual`) and stores all seven
otherwise. The identity holds to machine precision — *but only when the source grid carries no k-space
apodisation*: with `kspace_lowpass=0.5` it breaks (7.1e-02 relative) because the window is applied to
`iso_P` and not to `iso_local`; the two are then no longer the same field. Dropping `iso_P_zz` saves 1/7 =
14% of the channel with no approximation.

### Consuming it

*Scalar phase* — the gate enters by Parseval, exactly parallel to `compile_scheme`:
```
g = cosine_projection(gate, dt_gate, T, K)         # the gate's modes, once per acquisition (compression)
Psi = γ · Σ_k g_k c_k                              # bank.path_field_integral: (N_w, n_ch)
φ_χ = Σ_c Q_c(H)·χ_c · Psi[:, c]                   # bank.susc_path_field
```
*Vector-Bloch* (the primary path) — read the modes at the instants the route steps on and hand the field to
the existing per-step interface:
```
b, names = susc_path_decode(arrays, meta, n_t=n)   # (N_w, n_ch, n): the modes read at n instants of [0, T]
dB = χ_iso·(b[:,0] − Q(H)·b[:,1:7]) + χ_aniso·(Q(H)·b[:,7:13])     # (N_w, n)
replay_bloch(..., extra_phase_per_step = γ·dt·dB)
```

## Choosing K, and declaring it

K must resolve the **fastest RF modulation** in the sequences the pack is meant to serve: a CPMG train is
a comb filter that passes field fluctuations near its echo spacing, so the requirement scales with the
number of refocusing pulses (roughly `K ≳ 2×n_pulses`). Everything else — orientation, b-value, the
diffusion×susceptibility cross-term — is far less demanding.

Measured on `winther/G6/axon06` (1601 instants, MC floor 9.4e-3), error of the **vector-Bloch** signal against
the full-resolution field:

| case | K=16 | **K=32** | K=64 |
|---|---|---|---|
| spin echo, b=0, worst over θ∈{0,30,45,60,90} | 4.1e-4 | **1.1e-4** | 9.4e-5 |
| b=1e9 ⊥, spin echo (cross-term) | 2.3e-4 | **3.4e-4** | 1.6e-4 |
| b=3e9 ⊥, spin echo (cross-term) | 3.2e-4 | **3.8e-4** | 1.3e-4 |
| b=1e9 ⊥, **CPMG 16 echoes** | 2.4e-2 ✗ | **1.5e-3** | 4.5e-4 |

K=32 is ≥25× below the MC floor for single-refocus and gradient-on cases, 6× below it for a 16-echo train;
K=16 fails CPMG-16 outright.

**The band is fixed before the walk.** `bank.derive_field_band` walks a pilot of `PILOT_WALKERS` = 1024
walkers that keeps its block means (`raw_subset`), tries the rungs of `SUSC_PATH_LADDER` (16, 32, 64, …) in
ascending band, the narrower container first, and takes the first `(K_c3, bits)` whose codec error — the
truncated, quantised modes contracted for each field strength and direction of the envelope's battery,
decoded on the block grid and gated by GRE, spin echo, the CPMG train the rung serves and a stimulated echo
— is within the split-half floor of the pilot's own series. A refocusing depth the envelope declares
(`max_refocus_pulses`) puts a floor of `2×` that depth under the ladder; the position band is the point past
which the tier costs more than the walk it rides on, and the record says when the ladder went beyond it. The
production walk accumulates `K_c3` modes and nothing is re-derived on its reads afterwards; a fill's shards
take the certifying pack's `K_c3`. The pack declares its capability, alongside the acquisition envelope it
already declares:
```
replay_envelope.field_modes           = K_c3
replay_envelope.max_refocusing_pulses = K_c3 / 2     # beyond this, re-walk with a larger band
replay_envelope.field_read            = the read's term and the B0·χ it holds under
```
A pack stores `susc_path_K` modes, a prefix of the walk's `K_c3` (never more).

## What it costs

A walker's field channel is `n_ch × K` integers: 6 × 32 at 8 bit is 192 B, at 16 bit 384 B. Positions cost
`3 (K + 2)` coefficients at the position container's depth, and the field tier no longer has a claim on that
band: positions fall back to `K ≪` the number of block ends, which is all the gradient tier ever needed
(Parseval-exact), where a field evaluated from stored positions forced them lossless.

Measured at 1601 instants on a Winther axon, by sweeping the position band against the floor the
52,000-walker ensemble has (6.34e-3, extrapolated from a 6,000-walker subsample and consistent with this
family's measured 9.4e-3 at 31,000):

| K_pos | position-codec err | err / floor(52k) |
|---|---|---|
| 8 | 7.08e-03 | 1.12 |
| 16 | 1.60e-03 | 0.25 |
| **32** | **2.55e-04** | **0.04** |

A band whose codec error exceeds the statistical floor makes the compression the dominant error source, an
acceptable trade for an ordinary pack and the wrong one for a reference dataset, whose whole claim is that
replay is indistinguishable from the walk; K_pos = 32 puts the codec 25× below the floor.

**The cross-term survives the position truncation**, which is the thing to check before accepting any of
this: the reported physics is diffusion × susceptibility, so it would be no good to compress positions if
that decorrelated the two phases. It does not, because the codec is *walker-preserving*: walker `w`'s
gradient phase and its field history carry the same index, so `exp(i(φ_G + φ_χ))` is formed per walker
before any ensemble average and their correlation is exact by construction. Position truncation perturbs
`φ_G` alone, by the amount the gradient tier already certifies against the MC floor — it cannot leak into
`φ_χ`, which does not depend on the stored positions at all.

## The C2 (surface / boundary-local-time) tier belongs in the budget

Easy to forget, because an impermeable membrane has no *exchange* tier — but the intra-axonal walkers
strike the myelin inner wall constantly (measured contact density 12.1% of steps at 1601 instants), so the
**surface tier is real**: rho2 is a replay knob, not zero by construction. It is also the channel an
analytic MT tier is derived from (contact statistics / S:V at the myelin surface). A walk records it with
`tiers="all"` (the default).

The contact is stored in the same bridge form as the positions: the cumulative local time's two exact
endpoints (`blt_start`, `blt_endpoint`) and its `K` bands (`blt_bridge_dst`, or the integer container
`blt_b<i>[u<bits>]` with `blt_band_scale`; `build_replay_pack(blt_temporal_K=, blt_container=)`). Its cost
does not scale with the length of the walk: measured at K=32 in float16, 66 B/walker, with an ensemble error
of 6e-6 .. 7e-6 (K=64: 130 B, 2e-6 .. 2e-5) against a split-half floor of 1.9e-2 — 2 to 5 thousand times the
truncation error.

### Per-walker budget

| component | 8-bit field | 16-bit field |
|---|---|---|
| positions, K=64 (float32) | 768 B | 768 B |
| `susc_path_dct`, 6×32 | 192 B | 384 B |
| C2 bands, K=32 float16 | 66 B | 66 B |
| compartment (`comp_static`) + weights | ~5 B | ~5 B |

The compartment channel is one `int8` per walker (`comp_static`) when no walker changes pool. A walk whose
walkers cross walls writes the events of every change (`comp_events_offsets` / `_t` / `_label`, RPK.md 9.5.7)
instead, ragged per walker and priced by the number of events.

## Precision tiers: one artefact, pull only what your accuracy needs

A 2--5 GB download per axon is unusable for anything but reference work, and it is also unnecessary. The
Monte-Carlo floor scales as `1/sqrt(N)`, and **walkers are the row dimension of every per-walker tensor**,
so a consumer who wants a coarser floor simply reads *fewer rows*. `safetensors` supports exactly this:
`safe_open(...).get_slice(name)[0:N]` reads only those bytes (measured: 1 000 rows = 19.2 MB in 0.002 s,
10 000 rows = 192 MB in 0.024 s, linear in N), and because the header carries per-tensor byte offsets a
remote client can range-GET a prefix instead of the file.

Anchored on a measured floor of `9.4e-3` at 31 000 walkers (`n(eps) = 31000*(9.4e-3/eps)^2`) and a budget of
840 B per walker:

| tier | floor eps | walkers | bytes per axon |
|---|---|---|---|
| **browse / teaching** | 0.03 | ~3,044 | **~2 MB** |
| **working** | 0.01 | ~27,392 | **~22 MB** |
| high | 0.003 | ~304,351 | ~244 MB |
| reference | 0.0017 | ~947,806 | ~759 MB |
| canonical parity | 0.001 | ~2,739,160 | ~2.14 GB |

Three things must hold for a prefix read to be *valid*:

1. **Walkers must be stored SHUFFLED.** A walk that seeds pool by pool (all intra, then all myelin) has a
   prefix of one pool and *zero* of the other. Walkers are exchangeable, so `walk_spec` joins the pools'
   walks under one permutation of every walker-indexed array (`spec.walk._join_pool_walks`), after which a
   prefix is an unbiased subsample (N=1 000 -> 1.2e-4, N=10 000 -> 5.4e-5 against the full-pack value).
   `build_replay_pack` records `compression.precision_tiers` with `walkers_shuffled`, which also carries
   `bytes_per_walker`, `floor_at_full_n` and the per-eps walker counts so a consumer computes the prefix
   length rather than guessing it. **The builder cannot detect the property:** a pack whose walkers are not
   declared shuffled gets `usable: false` and an explicit note instead of tiers that quietly invite a biased
   read, and `merge_packs` declares the shards' concatenation unshuffled.
2. **Walkers must be the leading axis** of every per-walker tensor, so a prefix is one contiguous byte
   range: `pos_x` is `(N_w, K+2, 3)`-leading already, and `susc_path_dct` is therefore specified as
   `(N_w, n_ch, K)` -- *not* `(n_ch, N_w, K)`, which would make a walker prefix a strided scatter. The
   events columns are the exception: their rows are events, not walkers, so they are read whole and
   counted as fixed overhead (`precision_tiers.not_prefix_sliceable`).
3. **The tier map must be declared**, not left to the reader to guess, derived from the pack's own floor, so
   `eps(N) = floor_measured * sqrt(N_full / N)`.

The field basis is a function of the embedded spec (`spec.walk.field_grid_of_spec`, the strand basis from
the spec's strands), so the pack carries no field grid: only the channel, whose modes were read from it.

## Fidelity gate

The channel is certified the way every other tier is: the stored (truncated, quantised) modes decoded
against the walk's own reads, over a battery that **includes a CPMG train** at the declared depth (the
binding case), under the envelope's field strengths and directions, with the error required to sit below
the split-half Monte-Carlo floor.

* The default certificate (`fidelity="derived"`) carries the **pilot's** measurement of the pair it chose
  (`err_susc_path`, `floor_susc_path`, `susc_path_measured_on: "pilot"`) and the read's term
  (`fidelity.terms.field_read`), with the quadrature of the field's block accumulation under
  `fidelity.terms.quadrature.field`.
* `fidelity="measured"` re-measures it on the walk's raw subset (`raw_subset=n`): the scalar gates
  (`_susc_path_fidelity`) and the vector-Bloch path (`_susc_path_bloch_fidelity`), recorded as
  `err_susc_path` / `err_susc_bloch` with `susc_path_measured_on: "raw_subset"`.

## Limitations

- The path channel is **tied to its walk**. Arbitrary spatial questions (re-seeding, sampling the field at
  new positions) need the field basis, which the embedded spec regenerates.
- K is a **declared capability**, not a universal constant. A pack built for 16-echo trains is not valid
  for a 64-echo train; the envelope says so, and a larger band means a re-walk.
- A pack's field strength is bounded by its read: where the read is the step, the envelope's `B0_chi_max_T`
  is the largest `B0·χ` the field-read term holds under.
- Anisotropic (`aniso_G`) channels were not exercised in the measurements above — the packs measured were
  built isotropic (matching their source study). Enabling them raises `n_ch` 7 → 13 and needs the same
  K-sweep before being trusted.
- MT with emergent binding evolves a bound pool sequentially; it uses the same read `b(t)`, but its own
  mode requirement has not been measured separately.
