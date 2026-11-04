# FEXI on permeable cylinders: findings

The record of `fexi_axr.py`. One acquisition object runs from end to end:

- `dmipy_replay.fexi_protocol` builds the protocol;
- dmipy-sim walks and contracts it;
- dmipy-fit's `FEXIApparentExchange` fits it;
- dmipy-analytic's `karger_waveform` gives the two-site prediction, through dmipy-fit's Kärger model.

No step builds a waveform or fits a curve of its own.

**Bottom line.**

- The apparent exchange rate follows the membrane permeability over 1 to 15 s⁻¹, and every run agrees with the barrier-limited two-site prediction: |z| ≤ 0.8 in the base protocol, all within the family-wise band.
- With `kappa = 0` the geometry alone recovers at `0.05 ± 0.05 s⁻¹`, consistent with zero, so this lattice adds no pseudo-exchange of its own.
- A replay pack written from one walk replays the same FEXI `ScannerSequence` to `1.3e-8` of `simulate`'s signal of that walk. The pack's certificate is `5.0e-4`.

## Set-up

| | |
|---|---|
| substrate | cylinders of `R = 2 µm` on a square lattice (one per periodic cell), lumen area fraction `f = 0.5`, gap `1.01 µm`, `D = 1 µm²/ms` in both pools, the gradient across the cylinders |
| sequence | `fexi_protocol`: bipolar blocks of `delta = 4 ms`, `Delta = 10 ms`, square lobes, crushers of 1 ms; `b_f = 2.5 ms/µm²`; detection `b_d = 0, 0.5 ms/µm²`; filter off and on |
| mixing times | base: 10, 25, 50, 100, 200, 350, 600, 1000 ms; scaled (`--scaled`): eight from `0.1 / AXR` to `2.5 / AXR` |
| walk | per `kappa`, one walk of the longest span (1.1 s, 11 storage windows of 100 ms, `K = 40` bands per window at the 200 Hz the sequences read), every sequence contracted with it; 40,000 walkers; no bounce budget exhausted |
| pools (`kappa = 0`) | lumen `0.048 µm²/ms`, interstitium `0.629 µm²/ms` at the protocol's blocks: a 13-fold contrast |
| prediction | the two-site Kärger signal of those pools at `k_12 = 2 kappa / R`, `k_21 = k_12 f / (1 - f)` on the same protocol (`X0GeneralizedKarger`, the generalised Kärger equation along each sequence's own gradient), fitted by the same AXR analysis; the two-site rate `k_12 + k_21 = 2 kappa / (R (1 - f))` beside it |

**Standard errors.** Each AXR's standard error is the signals' covariance across the walkers, propagated through the fit's own sensitivities.

**Agreement band.** A run agrees when `|AXR - AXR_pred| <= z SE + (kappa R / (4 D)) AXR_pred`:

- `z` is the two-sided Bonferroni normal quantile at a family-wise false-alarm rate of `1e-3`;
- the second term is the first correction to the barrier-limited rate, from the lumen's own diffusion. It is derived from the slowest Robin mode of a disc, `D alpha^2 = (2 kappa / R)(1 - kappa R / (4 D) + ...)`.

## (B) AXR against the permeability, base protocol

| kappa (µm/s) | kappa R / D | AXR (s⁻¹) | SE | predicted | two-site rate | z | band | sigma | sigma predicted |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.051 | 0.046 | – | 0 | – | – | 0.483 | – |
| 0.5 | 1.0e-3 | 1.031 | 0.103 | 0.971 | 1.00 | +0.59 | 0.382 | 0.467 | 0.534 |
| 1.0 | 2.0e-3 | 1.938 | 0.168 | 1.950 | 2.00 | −0.07 | 0.624 | 0.470 | 0.526 |
| 2.0 | 4.0e-3 | 4.213 | 0.383 | 3.909 | 4.00 | +0.80 | 1.430 | 0.461 | 0.511 |
| 4.0 | 8.0e-3 | 7.886 | 1.005 | 7.835 | 8.00 | +0.05 | 3.752 | 0.400 | 0.482 |
| 7.5 | 1.5e-2 | 15.279 | 3.085 | 14.734 | 15.00 | +0.18 | 11.528 | 0.348 | 0.435 |

**Agreement.** Every run is within its band. The analysis' finite-b reading puts the prediction 1 to 3 % below the two-site rate, and the walks follow the prediction rather than the rate.

**Monotonicity.** The AXR increases at every step of `kappa`. The first two steps clear the family-wise band of their standard errors (0.91 against 0.72, 2.28 against 1.53). The last two do not (3.67 against 3.94, 7.39 against 11.88). The fixed mixing times of the base protocol hold only three points inside `1 / AXR` at 15 s⁻¹, so the standard error grows with the rate. The scaled protocol below resolves those steps.

**sigma.** The fitted `sigma` sits 13 to 20 % below the two-site value. The two-site pools are Gaussian at the protocol's ADCs. The walk's lumen is restricted, and its filter efficiency is lower. `sigma` is reported, not held to the two-site value.

**The `kappa = 0` control.** The geometric pseudo-exchange (Khateri et al. 2022) is `0.05 ± 0.05 s⁻¹`, with the filter efficiency 0.48. A square lattice's interstitium is nearly uniform, so the filter's selection does not relax by diffusion between its regions on these time scales.

**Refused steps.** The walks hold a walker in place on a step that ends in another pool without a granted crossing. At `kappa = 7.5 µm/s` that was 5,969 of about `1.1e10` walker-steps.

## (B) the scaled protocol

Each permeability walks its own eight mixing times from `0.1 / AXR` to `2.5 / AXR`; `kappa = 0` walks the base times again with another seed.

| kappa (µm/s) | kappa R / D | AXR (s⁻¹) | SE | predicted | two-site rate | z | band | sigma | sigma predicted |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.001 (the fit's lower bound) | – | – | 0 | – | – | 0.466 | – |
| 2.0 | 4.0e-3 | 4.941 | 0.435 | 3.912 | 4.00 | +2.37 | 1.563 | 0.493 | 0.512 |
| 4.0 | 8.0e-3 | 7.684 | 0.841 | 7.834 | 8.00 | −0.18 | 3.032 | 0.404 | 0.482 |
| 7.5 | 1.5e-2 | 15.734 | 1.661 | 14.720 | 15.00 | +0.61 | 6.015 | 0.372 | 0.434 |

**Agreement.** Every run is within its band.

**The fastest step.** Mixing times scaled to the rate resolve it: 7.68 to 15.73, a rise of 8.05 against a band of 6.48.

**The 4-to-8 s⁻¹ step.** It is positive in both independent runs, but neither clears the family-wise band at `1e-3`:

| run | rise | its own standard error | that standard error's z | family-wise band |
|---|---|---|---|---|
| base | 3.67 | 1.08 | 3.4 | 3.94 |
| scaled | 2.74 | 0.95 | 2.9 | 3.29 |

**Monotonicity overall.** The AXR is monotone in `kappa` in both protocols. Three of the four steps clear the family-wise band (1 to 2 and 2 to 4 in the base run, 8 to 15 in the scaled one). The fourth (4 to 8) is about three standard errors in each run, below that band.

**The `kappa = 0` control again.** Here the fit stops at its lower bound of `1e-3 s⁻¹`: no recovery.

## (C) the replay

`simulate`'s walk of the `t_m = 100 ms` sequence at `kappa = 1 µm/s` is used: 40,000 walkers over its own span.

- **Same walk.** The walk was written as a pack (`build_replay_pack(..., out_path=)`), read back with `dmipy_replay.read_rpk` and replayed on the same `ScannerSequence`. It equals `simulate`'s signal of that walk to `1.3e-8` (`tol=None`, every band). The pack's certificate is `fidelity.err_max = 5.0e-4`, with floor `5.0e-3`.
- **Second instance.** The scaled run repeats the check at `kappa = 4 µm/s`: `7.3e-9` against `err_max = 4.5e-4`.
- **Planned read.** The planned read (`tol` default: 93 of 333 bands, 19.0 of 40.0 MB) is within `1.1e-4`, against its stated planned error `1.05e-3`.
- **Two different walks.** The sweep's 1.1 s walk at the same `kappa`, contracted with that sequence, differs from `simulate`'s 0.13 s walk of it by 0.0016, 0.0045 and 0.0012 on the weighted rows. Those are z = 1.6, 2.1 and 0.5 against the two walks' combined standard error.

So a FEXI sequence needs nothing particular in a pack or its reader: the pathway, the storage gate and the crushers' selection are all read off the `ScannerSequence` itself.
