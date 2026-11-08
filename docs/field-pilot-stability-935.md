# Field pilot stability (issue 935)

The selector is `derive_field_band` in `dmipy_sim/replay/bank.py`; the producer fixes `K_c3` through `derive_field_band_per_pool` before production.

The diagnosis fixture is absent from this repository. The diagnostic reconstructs its stated slab, permeability, diffusion, raster spacing, walker count, seed and time grid: 4 µm, 1e-5 m/s, 2e-9 m²/s, 400 × 1 × 1 nonperiodic 10 nm nodes, T=10 ms, K_position=32, 126 blocks × four physics steps, 20,000 production walkers, 1,024 pilot walkers, seed 43, B0=7 T, chi=1.06e-6, directions 0°/90°, no declared refocusing depth. The step basis is explicitly defined in the diagnostic. The original diagnostic did not supply its raster arrays; the reconstructed pilot is not a bitwise reproduction of its quoted ratios.

Each case uses 20 independent with-replacement walker resamples, NumPy default_rng seeds **93500–93519**, of that one CPU bank. These are independent bootstrap draws, not 20 new physical walks. Easy retains only the first two DCT modes of that bank. Hard uses the raster bank at four times the susceptibility and a 2,000,000-walker production threshold (ten times tighter). The legacy implementation is read from commit `e19d4175701a09fc3175e4938225c24092497800`; only its early exit is disabled for measuring later rungs.

| Case | Legacy K (count) | Conservative K (count) |
|---|---|---|
| easy | 16: 20 | 16: 20 |
| raster43 | 32: 1, 64: 19 | 126: 20 |
| hard | 126: 20 | 126: 20 |

Each entry below is min / median / max across the 20 draws. Legacy ratios are the recorded worst GRE/SE error/threshold; revised ratios and bounds include every train/stimulated-echo gate too. The JSON retains every seed and rung.

| Case | K | Legacy ratio | Revised ratio | Revised 99% upper bound |
|---|---:|---:|---:|---:|
| easy | 16 | 0.000296 / 0.000614 / 0.00194 | 0.00032 / 0.000589 / 0.00194 | 0.00378 / 0.00465 / 0.00757 |
| easy | 32 | 0.000363 / 0.00092 / 0.00204 | 0.000358 / 0.000937 / 0.00204 | 0.00391 / 0.0047 / 0.00754 |
| easy | 64 | 0.000221 / 0.000524 / 0.00125 | 0.000207 / 0.000524 / 0.00125 | 0.00268 / 0.00342 / 0.00469 |
| easy | 126 | 0.000207 / 0.000593 / 0.000846 | 0.000188 / 0.000386 / 0.000846 | 0.00232 / 0.00311 / 0.00391 |
| raster43 | 16 | 2.35 / 4.75 / 14.2 | 2.35 / 4.75 / 14.2 | 36.8 / 70.4 / 128 |
| raster43 | 32 | 0.606 / 2.22 / 4.48 | 1.04 / 2.22 / 4.48 | 20.9 / 35.2 / 70 |
| raster43 | 64 | 0.193 / 0.517 / 0.81 | 0.228 / 0.515 / 0.856 | 5.48 / 11.6 / 21.8 |
| raster43 | 126 | 0.00113 / 0.00365 / 0.0694 | 0.00404 / 0.00826 / 0.0694 | 0.0479 / 0.124 / 0.321 |
| hard | 16 | 25.1 / 48.1 / 144 | 25.1 / 48.1 / 144 | 368 / 703 / 1.29e+03 |
| hard | 32 | 7.39 / 22.2 / 44.7 | 10.4 / 22.2 / 44.7 | 211 / 352 / 700 |
| hard | 64 | 2.34 / 5 / 8.07 | 2.42 / 5.17 / 8.48 | 55 / 118 / 231 |
| hard | 126 | 0.0113 / 0.0235 / 0.11 | 0.0114 / 0.0235 / 0.126 | 0.12 / 0.178 / 0.778 |

## Statistical method

For paired per-walker raw/decoded signals z and z_dec, normalized weights W, and delta=z-z_dec, the error numerator is bounded by `abs(sum(W*delta)) + z_alpha*sqrt(sum(W²*abs(delta-mean_delta)²))`. Real/imaginary normal margins bound the complex norm by the triangle inequality. The raw-signal variance supplies the threshold, `2*floor_scale*sqrt(sum(W²*abs(z-mean_z)²))`. A log-variance delta-method margin supplies a positive lower threshold bound; its influence includes the weighted raw mean. The upper ratio divides the upper error by the lower threshold. An explicit target_floor instead has a fixed denominator.

The family error probability is 0.01, divided over every rung, setting, gate, both codec candidates and five component/variance tails (two tails for each complex component and one lower-variance tail). Per-pool selection additionally divides this error probability over populated pools. Bounding the whole family before taking its maximum accounts for the extreme-value effect and for choosing a rung after looking at the pilot. This is an **asymptotic normal/log-delta confidence approximation**, not a distribution-free finite-sample guarantee, particularly for rarely visited gates.

The single random split-half train floor is replaced with twice the per-walker standard error of each train signal. A derived codec that cannot be certified is retried at 16 bits, with both candidates included in the multiplicity correction. The smallest certified rung wins; an all-failing ladder selects the top rung, sets field_band_uncertified, and warns with its actual simultaneous upper bound and confidence level. A non-finite bound is refused by name. The record stores confidence_method, confidence_is_approximate and confidence_comparisons globally; the chosen rung and every visited rung store upper_bound, confidence_level, worst_ratio and the codec. The full-battery worst ratio is separate from the historical worst-echo diagnostic.

No pilot enlargement was necessary for the observed stability: all 20 revised draws chose the same band in each case with the existing 1,024 walkers. This observation does not establish a 99% probability of identical choices in future samples.

## Reference and scope

The regression test uses seed **93643**, 8,192 walkers (eight times the aggregate pilot path duration), the same 504-step/126-block grid, and the pilot’s actual K and bit ranges against independent raw reference means. Reference ratios are **0.00240–0.00341**, below every raster certificate (**0.0479–0.321**). It checks all 20 resamples and includes CPMG at the chosen band’s 63-pulse depth. The CPU stability and reference tests fit below two minutes.

The certificate concerns the block-grid replay domain, as the existing selector does. An exploratory twice-finer-grid walk (seed 93644, 8,192 walkers, 1,008 steps) gave ratio 0.484 for the first certificate versus its bound 0.192: temporal discretisation is outside this sampling bound, although that check remained below the error threshold. This change does not certify time-grid refinement or a subsequently rederived production codec.

Reproduction (offline, no new dependencies):

```sh
OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl DMIPY_SIM_RUN_DIR='' \
python -m diagnostics.field_pilot_stability --baseline-ref e19d4175701a09fc3175e4938225c24092497800 \
  --output /tmp/field-pilot-stability-935.json
```
