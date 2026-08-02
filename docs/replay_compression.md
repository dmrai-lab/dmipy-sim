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

## What's here

`compression.py` — the codecs (`encode_temporal_dct`, `encode_boundary_dct`,
`encode_compartment`, …), mode-space replay (`mode_space_phi`, `mode_space_signal`), the
separable log-weights (`surface_logweight`, `relaxation_logweight`), and a self-contained
numpy fidelity scorer (`measure_fidelity`, `auto_select_modes`) — depends only on
numpy+scipy. `tests/test_compression.py` locks the algebraic identities and the real-walk
per-TE fidelity.

## Remaining wiring (next commits)

1. **Producer streaming.** In `simulate_trajectories`' batch loop, when producing for
   replay, DCT each batch on-device and append `(batch, K, 3)` + boundary `(batch, K+1)` +
   compartment RLE, instead of the raw `(batch, n_t, 3)` float16. `temporal_dct` is
   per-walker separable, so it streams per batch (unlike `lowrank`'s global SVD). This is
   what removes the host-RAM blowup.
2. **Replay routing.** When a pack is compressed, `replay` / `replay_jax` compute the phase
   via `mode_space_signal` (positions modes) and add the surface/relaxation log-weight from
   the decoded boundary/compartment channels — no raw trajectory.
3. **K selection.** `auto_select_modes` against the acquisition envelope, stored in the pack
   metadata so the fidelity is self-certifying.
4. **Fused parity.** The ladder + S/V suites stay as the fused oracle; add a compressed-replay
   vs fused parity test across the envelope.
