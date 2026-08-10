# Replay compression (IR basis)

The replay producer today stores the **raw** master walk — positions `(N_w, n_t, 3)` and,
for surface relaxivity / MT, dense per-save channels `(N_w, n_t)`. At the fine, high-`n_t`
fidelity we want to walk *once* and keep (sub-micron surface relaxivity needs `step ≈ R/6`,
so `n_t` reaches ~10⁵), that raw form does not fit host RAM: a 500k-walker survival to
0.6 s exhausted 573 GB and was OS-killed. This module compresses the walk into an **IR
basis** so the replay representation is memory-viable at arbitrary `n_t`/`N_w`.

## The two channels, one basis

Every replay observable is a functional of the stored walk, and both channels compress
under the temporal DCT:

| Channel | What replay needs | Codec | Fidelity / ratio (measured) |
|---|---|---|---|
| Positions | gradient phase `φ = γΔt Σ_t G·r` | `temporal_dct`: K DCT bands/path; `mode_space_signal` contracts them against `DCT(G)` — never reconstructs the trajectory | K=32 → `max|ΔS|=2.6e-4` (MC floor ~7e-3), **47×**; K=64 → 1.9e-5, 23× |
| Boundary local time | surface weight `(ρ/D) Σ_t χ_t ℓ_t`, at any TE truncation | `boundary_dct`: **detrend-then-DCT** of the cumulative `B(t)=cumsum(ℓ)` — store exact endpoint `B(T)` + K bands of `B(t) − (t/T)B(T)` | K=8 → per-TE `max|ΔE|=1.2e-4` (MC floor 1.1e-2), ratio ~`n_t/K` (**grows with n_t**) |
| Compartment / bound-fraction | per-comp T2/T1, MT occupancy | row RLE (piecewise-constant per walker) | lossless |

**Why detrend the boundary channel.** `B(t)` is ~linear (roughly constant contact rate). A
bare DCT of a ramp has a Gibbs endpoint error that *exceeds* the MC floor at small K —
fatal for the longest-TE truncation. Storing the endpoint exactly and DCT-ing only the
small, end-vanishing residual drops the per-TE error ~100× at the same K+1 storage. (A
T2 slope-fit over a mid-window hides this — the honest metric is per-TE signal error.)

**MT rides the boundary channel too (design decision).** Magnetization transfer is driven
by the same impact-angle / boundary-local-time estimate as surface relaxivity, and the
emergent bound fraction matches the two-pool oracle from that channel alone (`f_b=0.278`
vs `k_f/(k_f+k_r)=0.286`). The full emergent wall-sticking (freeze-during-dwell) has **no
observed signal impact** in the regimes tested (DW-attenuation coupling <0.2% across
`T2_bound` 5–200 ms) — it is a physics-completeness nicety parked for later, not a signal
requirement. So **MT is a separable boundary-channel log-weight**, same as surface
relaxivity: compressed replay applies it off `boundary_dct` with an MT saturation rate in
place of ρ/D — no bound-pool trajectory / vector-Bloch needed for the mean MT signal.

## What's here

`compression.py` — the codecs (`encode_temporal_dct`, `encode_boundary_dct`,
`encode_compartment`, …), mode-space replay (`mode_space_phi`, `mode_space_signal`), the
separable log-weights (`surface_logweight`, `relaxation_logweight`), and a self-contained
numpy fidelity scorer (`measure_fidelity`, `auto_select_modes`) — depends only on
numpy+scipy. `tests/test_compression.py` locks the algebraic identities and the real-walk
per-TE fidelity.

## Wiring status

1. **Producer streaming — DONE** (`simulate_trajectories(compress=K)`). DCTs each batch
   on-device and pulls only `(batch, K, 3)` positions + `(batch, K+1)` boundary; the raw
   trajectory never hits the host. `temporal_dct` is per-walker separable, so it streams per
   batch (unlike `lowrank`'s global SVD).
2. **Replay routing — DONE** (`replay()` dispatches a compressed master to
   `_replay_compressed`). Gradient phase via `mode_space_phi` (positions never
   reconstructed); ungated surface relaxivity from the stored endpoint `B(T)` (exact, no
   reconstruction); per-compartment T2/T1 from the carried compartment channel. Verified vs
   raw `replay()`: gradient `max|ΔS|=2.6e-4`, ungated surface exact to 8e-8.

## Remaining (next commits)

3. **K selection.** `auto_select_modes` against the acquisition envelope, stored in the pack
   metadata so the fidelity is self-certifying.
4. **Fused parity.** The ladder + S/V suites stay as the fused oracle; add a compressed-replay
   vs fused parity test across the envelope.

### Deferred within pieces 1–2 (documented, not blocking)

- **TE truncation from a compressed master** — evaluate `B(TE)` / the phase at `TE<T_max`
  from modes (single-index IDCT), so one compressed walk serves a TE sweep like the raw
  prewalk does. Today compressed replay is at the full walk length.
- **chi-gated surface without reconstruction** — the gated path currently decodes the
  boundary channel to `(N, n_t)` (one channel, 1/3 of positions); a mode-space contraction
  (`endpoint·(χ·Δramp) + modes·(χ·ΔΦ)`) would avoid it. Ungated already avoids it (endpoint).
- **Compartment-channel compression** — kept raw int8; RLE per batch is the follow-up.
- **`replay_jax` compressed path** — piece 2 wired the NumPy `replay()`; the JAX path
  (dmipy-design's differentiable gradient replay) mode-space kernel exists
  (`compression._jax_signal_kernels`) but isn't dispatched yet.
- **Susceptibility** — unsupported from a compressed master (nonlinear field sampling);
  `_replay_compressed` raises. Replay susceptibility from a raw walk.
