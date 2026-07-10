# Contributing to dmipy-sim

Thanks for your interest. dmipy-sim is the forward-truth engine: a GPU Monte-Carlo of spins
under arbitrary `G(t)`. Physics is the specification — contributions are judged first on
physical correctness (analytical limiting cases, conservation laws, known closed forms), not
just on passing tests.

## Development setup

```bash
git clone https://github.com/dmrai-lab/dmipy-sim.git
cd dmipy-sim
pip install -e ".[examples]"   # JAX (CPU by default; see README for CUDA-12)
pytest -q                      # CPU-safe suite (CI runs CPU-only)
```

Test in **float64 first** (reference/correctness); float32 is production/GPU speed and is only
acceptable when the difference from float64 is below the physical noise floor. Mind step
resolution — the MC step must stay well below the smallest geometric feature (e.g. fibre
radius) or walkers tunnel through walls.

## Guidelines

- Match the surrounding code — naming, comment density, and idiom.
- One source of truth for every physical constant: tissue constants in the `Substrate` /
  `biophysical_constants` catalogue, scanner hardware/safety limits in
  `dmipy_sim.sequences.scanner_constants`. Read them via their accessors; do not hard-code.
- The free waveform `G(t)` is the base representation; PGSE/OGSE/etc. are factory
  constructors, not fundamental types.
- Add a validation example against an exact analytical result for any new physical effect.

## Contributor License Agreement

dmipy is **dual-licensed** (AGPL-3.0 OR commercial), so we need an explicit relicensing grant
from contributors — see the
[CLA](https://github.com/dmrai-lab/dmipy/blob/main/licensing/CLA.md). For now, add this line
to your first pull request:

> I have read the CLA and I agree to it on behalf of myself (and my employer if applicable).
> Signed, [your name] <[your email]>

You keep the copyright to your work. Please open an issue before starting anything large.
