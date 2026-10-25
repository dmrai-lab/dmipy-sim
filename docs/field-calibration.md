# Scoped field calibration

Adaptive strand-field walks can reuse numerical field settings across seeds and shard sizes. The spatial geometry, material spec, diffusivity, immutable far grid, time window, stepping plan, scanner band, block projection, numerical field budget, query scope and backend must agree exactly. The worker process and its loaded source and far grid are assumed immutable. Backend library bytes are hashed when available; public Python source is hashed once per process.

The scanner class still sets C0. Calibration chooses the separate field read and C3 band within an explicit numerical budget, independent of production walker count. This does not expand the supported scanner/RF domain. Supply a query identity covering the actual protocol waveforms, RF/timing and tissue parameter domain, rather than an informal name. A hash identifies that domain; it does not demonstrate physical coverage.

## Producing and reusing a record

Pass query_scope with identity, metric="max_abs_complex_signal_error", target and confidence, and calibration_floor to simulate_trajectories_adaptive. After the reference/refinement work, construct FieldCalibration.from_walk(walk, accuracy=..., step=...). Accuracy comes from assess_accuracy; step comes from assess_step_certificate of the existing step-size diagnostic. Save/load the record as JSON and supply it through field_calibration on subsequent walks with the same query scope and numerical floor.

walk_spec accepts a mapping from every diffusing seeded pool name to its own FieldCalibration. Extract records from its joined walk using FieldCalibration.from_walk(..., pool="extra") and the corresponding per-pool evidence. All pools need a common field band. Incompatible bands are refused, requiring regeneration of the evidence at a common band. A pool's record cannot silently stand in for the joined substrate.

A supplied record is a snapshot, with a checksum and revalidated evidence. It refuses missing numerical terms, unresolved steps, scope changes and explicit read/band conflicts. The checksum detects accidental changes; it is not an authenticity signature. Evidence producers remain responsible for valid bounds and their scope.

## What the measurements establish

The native backend gathers a small every-round field-reference series using the native trajectory engine, including the same physical gather intervals as production. NumPy computes the shared statistics; the JAX field reader is not executed on this route. CPU JAX remains a host dependency. Explicit JAX reference walks remain available. The native backend still requires a single storage window.

Read candidates are charged at their actual measured rung, using the candidate's QV and a simultaneous one-sided normal jackknife ratio estimate across walkers. Windows belonging to one walker are not treated as independent samples. Aliasing with positive reference error and zero coarse QV is refused. The estimate is approximate, and finite samples can miss rare field events. The finest every-round reference cannot measure its own continuous-path error; the record explicitly labels that limitation. Independent trajectory/field-read refinement evidence must cover that residual.

The field-band battery is a diagnostic finite set, not certification of every possible human-scanner sequence. Explicit final-answer evidence must cover the declared query domain.

## Accuracy and remaining admission work

Numerical bias does not shrink with production walker count. assess_accuracy adds bounds in one answer metric and accounts for confidence without assuming independence. It keeps Monte Carlo uncertainty separate. Required numerical terms are trajectory, spatial_field, field_read, field_band and representation. The step diagnostic measures relative diffusion-spectrum RMS error, so it cannot substitute for any absolute complex-signal term or establish C2/C3 accuracy.

A reused calibration is not a certificate of a new shard's Monte Carlo precision. Assess production statistical uncertainty again from that run's outputs, and combine it with the applicable numerical bounds. In particular, changing the number or distribution of walkers can change statistical reliability even when the numerical settings remain reusable.

The current API consumes trusted bounds; it does not yet obtain all query-specific trajectory, spatial-field, C2/C3 refinement and representation evidence, nor wire Tessera production admission to those checks. Tests marked synthetic verify workflow mechanics only. Do not use them as DiSCo production evidence. Issue #765 remains open for that integration and physical validation.

## Native reuse benchmark

One bounded T4 run used the small strands_extra parity fixture, a 100 ms window,
C0 K=260, C3 K=32 and B0=0.1 T. The first 20k call took 13.07 s, including
8.72 s in the pilot phase. Reused calls measured:

| Walkers | Total seconds | Production backend calls | Calibration calls |
| ---: | ---: | ---: | ---: |
| 20,000 | 2.98 | 1 | 0 |
| 50,000 | 6.75 | 1 | 0 |
| 100,000 | 13.45 | 2 | 0 |

These are host-call times, including preparation and output handling. The first
call includes initialization and calibration; it is not a kernel-only comparison.
The workflow used explicitly synthetic accuracy evidence to exercise reuse, so it
does not establish physical accuracy or predict full DiSCo throughput. Native field
execution was checked by replacing the JAX field reader with a function that raises.
The bounded controller confirmed cleanup. Machine-readable measurements and source
identity are in field-calibration-benchmark.json.
