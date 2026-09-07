# Contributing to dmipy-sim

dmipy-sim is maintained by one author with an AI pair, and features are cheap to add that way. So the
project does not take pull requests: **open an issue instead**. Say what you want to simulate or what
you found wrong, and we work out together how it fits the framework -- the substrate spec, the pack
format, the replay knobs -- before anything is written. That keeps the physics, the specifications and
the code in one hand, and it spares you a contributor license agreement.

## What makes a good issue

- **A bug**: the spec (`.sub.json`) or the geometry, the acquisition, the call, what you expected and
  why (an analytical limit, a reference simulator, a paper), and what came out. If a pack is involved,
  its `id` and `fidelity` block.
- **A feature**: the physics, with a reference, and the substrate or acquisition it needs. If it is a new
  substrate family, where the geometry comes from (a generator, a dataset) -- it enters as a spec
  producer, never as a loader that constructs geometry.
- **A question about a result**: the smallest script that reproduces it.

Physics is the specification. A change is right when it reproduces an analytical solution, an
eigenfunction series, a Brownstein-Tarr relation or a MISST reference to the Monte-Carlo noise floor,
not merely when the tests pass.

## Running it yourself

```bash
git clone https://github.com/dmrai-lab/dmipy-sim.git
cd dmipy-sim
pip install -e ".[dev]"                                   # add [mesh] for PLY loading, [cuda12] for GPU
JAX_PLATFORMS=cpu pytest tests/ -q -m "not slow and not gpu"   # the fast tier
```

`CLAUDE.md` is the operational guide to the code (engine, layout, conventions, traps); the formats live
in [replay-pack-spec](https://github.com/dmrai-lab/replay-pack-spec) (`SUBSTRATE.md`, `RPK.md`,
`RPH.md`).

## Licensing

dmipy-sim is AGPL-3.0 with a commercial license available; see `LICENSING.md`.
