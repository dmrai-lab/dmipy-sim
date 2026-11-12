# Agent instructions

dmipy-sim is the JAX Monte-Carlo forward engine for diffusion MRI: it walks an
explicit substrate and accumulates acquisition-independent coefficients in a
`CoefficientWalk`, writes certified replay packs, and provides `simulate()` for
the walk plus signal contraction. It is not an analytical fitting engine or the
replay reader. Shared mathematical forms belong to dmipy-analytic; acquisitions,
`ReplayPack`, and reader operations belong to dmipy-replay. The dependency on a
fitting package is one-way: this package must not import it.

## Sources and layout

Read `CLAUDE.md` for the detailed physics and geometry contracts, `README.md` for
public usage, and `pyproject.toml` for dependencies and pytest configuration.
Check the current workflows and code before copying operational details.

- `dmipy_sim/engine/`: producer, integrators, accumulation, wall and MT physics.
- `dmipy_sim/geometry/`: substrates and their common geometry contract.
- `dmipy_sim/coefficient_walk.py`, `compartments.py`: walk and pool records.
- `dmipy_sim/spec/`, `substrate/`, `io/`: specs, realisations, data import.
- `dmipy_sim/replay/`: pack production, encoding, certification and re-encoding;
  its package exports no submodule names. Import the owning module directly.
- `dmipy_sim/fields/`, `math/`, `spectrum/`: fields and numerical forms;
  `acquisition/`, `sequences/`, `viz/`: producer-side helpers and visualisation.
- `dmipy_sim/run.py`: producer progress, resource budgets, spool and resume.
- `tests/geometry/` mirrors geometry modules; `tests/physics/` and
  `tests/validation/` check physics across modules; `tests/replay/`, `math/`,
  `backends/`, `spectrum/` cover their domains. Data live under `tests/data/`
  and `tests/fixtures/`. Root-level tests cover API and cross-module contracts.
- `docs/replay-guide/` Python blocks are executed by `tests/test_replay_guide.py`.
- `.github/workflows/tests.yml`, `tools/ci_groups.py`, `tests/conftest.py` define
  the CPU test run; `gpu-tier.yml` defines device/scale validation.

## Setup and verification

Use an isolated environment; Python 3.11 and 3.12 are tested in CI. From the
repository root, with that environment's Python and pytest on PATH:

```sh
python -m pip install -e ".[dev,interop]"
python -m pip install pytest pytest-cov
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For CI parity, first install the dmipy-analytic and dmipy-replay revisions named
by `DMIPY_ANALYTIC_REF` and `DMIPY_REPLAY_REF` in `tests.yml`, using its install
steps (including their extras). Released dependencies alone may lack new forms.
`dev` includes mesh and image support; `interop` supplies the independent basis
oracle. Do not silently treat an optional-dependency skip as verification.
Use CPU JAX locally; do not install GPU extras for the CPU suite.

Before handing back, run the test files your change touches and the files that
cover the modules you changed, with the CPU workflow's flags. The whole suite is
eight matrix groups of about 17 minutes each, run in parallel by CI; run them all
locally only on a machine with the memory and time for it, one fresh pytest
process per group:

```sh
python tools/ci_groups.py 8 --check
for group in 0 1 2 3 4 5 6 7; do
    PATHS=$(python tools/ci_groups.py 8 "$group") || exit
    JAX_PLATFORMS=cpu pytest $PATHS -q --tb=short -m "not gpu" --durations=0 --durations-min=5 || exit
done
JAX_PLATFORMS=cpu pytest tests --collect-only -q -m "not gpu" | tail -1
```

Inspect the collection exit status/output as well as its final line; the pipe
alone can hide a pytest failure. Preserve the persistent `.jax_cache` configured
by `conftest.py`. Do no expensive work at import/collection time: build fixtures
on first use, including optional imports and parametrization inputs.

There are two tiers by hardware, not by speed: CPU tests and tests marked `gpu`.
Device-dependent or large statistical checks belong in the bounded GPU workflow,
on a dedicated device, not a laptop-size box. An expensive CPU test must be
slimmed while preserving its claim, never parked behind a `slow` marker.
The million-walker `--heavy` mode is for deliberate large validation, not routine
local hand-back. Run only one local Python job at a time; allow test subprocesses.
Use the smallest free CPU/reference case first. Do not launch paid compute
without the lead's routing and budget approval; report expected cost and spend.
Shared guards and integration workflows also run in CI; releases are separate.

## Scientific and implementation rules

- Physics is the specification. Validate against analytical solutions or the
  checked-in MISST fixtures. Read a published model's primary source and implement
  its definitions; state exactly what you could not verify.
- Derive error bounds, walker counts and tolerances from the error model and
  comparison count. Preserve assertions for known defects; use
  `pytest.mark.xfail(strict=True, reason="<public issue>: ...")` when appropriate.
  Never change an expectation to match a defect or weaken coverage for speed.
- Tests must work offline in CI and on the device runner: no git, network, or
  files outside the repository. Use repository fixtures and public records.
- No legacy backdoors: remove the old spelling and convert every caller; no
  compatibility shims, aliases, converters or warning-only acceptance.
- No hidden fallbacks: a missing required value is an error, or `None` plus a
  named flag. Never substitute a numeric dictionary default, default argument,
  clamp, swallowed exception or guessed value for a required input.
- Never read another module's private attributes; use its public function or
  record. Use declared geometry capabilities and `physics.length_scales_of`,
  not attribute probes or class-name dispatch tables.
- Write declarative docstrings describing behavior, inputs, outputs, units and
  constraints. Keep implementation mechanics out of user-facing workflows.
- Use SI: lengths m, time s, diffusivity m²/s, b-values s/m², gradients T/m,
  permeability and surface relaxivity m/s, field T, susceptibility dimensionless.
  Convert explicitly at boundaries; distinguish ppm from dimensionless values.
- Keep reference calculations in float64 where their accuracy requires it;
  preserve established float32 walk/storage and device paths and their certified
  rounding terms. Do not globally enable x64: tests requiring it use the `x64`
  fixture, which restores the process flag. Preserve float64 reductions where
  cancellation or accumulated rounding requires them; justify precision changes
  with reference comparisons rather than switching dtype by hardware alone.
- Use `dmipy_sim.*` loggers, not prints. Physics-regime warnings are `UserWarning`;
  environment/resource warnings are `RuntimeWarning`. Preserve producer run
  records, named refusals and explicit capability limits.

## Boundaries and hand-back

Never push, merge, publish packs/pages/packages, create releases/tags, or touch
secrets. Leave review and commits to the lead unless explicitly instructed
otherwise; never amend or force-push a pushed branch. Do not work around a denial.
Public code, documentation, tests and review text must name no private work,
repositories, issues, methods, machines, hosts, providers or local infrastructure.
Wait only on recorded PIDs; never use process-name patterns to wait or kill.

Hand back the changed files and rationale, every command and its actual outcome,
test counts/skips/failures, exact unverified claims, and spend (including zero).
Number the brief's requirements with status and reason. Explicitly flag every
definition change (threshold, scope or units) as a decision for the lead. Include
friction in the first person, minutes lost, workaround and the top improvement.
Provide the structured hand-back required by the brief and validate it. Report
source discrepancies without editing unrelated guidance, code or workflows.
