# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

- **The MT walk binds at permeable walls, cross first** (#909): at each encounter that touches a permeable wall the
  crossing is decided by the transmission rule, then the walker, on the face it ends on, binds with the probability of
  the encounter's local time (crossed or reflected), freezes at the step's end and is released to the same face after
  its dwell. A permeability constant in time on a geometry that carries no side is walked, against
  `BlochTorrey1D(binding_reactivity=...)` within the standard error; a permeability over time, packed cylinders and a
  Mesh are still refused by name. Impermeable MT walks are unchanged to the bit.
- **The boundary local time counts every encounter that touches a wall**, a transmitted one as well as a reflected one,
  each by its own `2 d_perp` (a reflection also by its nudge's push): surface relaxivity on a permeable wall follows the
  continuum (a permeable slab at `kappa* = 0.5` against `BlochTorrey1D` with membrane relaxivity, where counting
  reflections only was ~5 % low). Impermeable walks are unchanged to the bit. A walk states the rule
  (`work["local_time_rule"]`, a pack's `provenance.walk.local_time_rule`: `every_touching_encounter`).
- **`walk` walks a permeable spec of one geometry**, with and without a `WalkContext`: a packed cell or an analytic
  pore whose walls carry a permeability per object and over time (`permeability_table`) is walked as its geometry is,
  bit for bit, on the JAX kernels and on a backend that declares it walks a permeability table. A walker keeps the
  pool it was seeded in as its label, a crossing is an event of its occupancy and no walker is excluded for it; every
  walk states the rule its pools followed (`work["pool_rule"]`, a pack's `provenance.walk.pool_rule`). A permeable
  multi-surface bundle, whose pools are walked one by one with no exchange, is refused by name.
- **FEXI is the acquisition layer's** (`dmipy_replay.fexi`, `fexi_protocol`, `fexi_bvalues`): the vector-Bloch
  constructor `engine.pulse_sequence.fexi` and `fexi_b_detect` are removed and refused by name; the one filter-exchange
  sequence is read by both engines -- the vector-Bloch oracle plays its pulses and its declared crushers, the walk
  contracts its effective gradient at the stimulated echo's amplitude -- and both hold the free-diffusion closed form
  (`tests/test_fexi.py`). `examples/validation/fexi_axr.py` walks the protocol on permeable packed cylinders across a
  permeability sweep, fits it with dmipy-fit's AXR analysis against the barrier-limited two-site prediction, and
  replays one walk from its written pack (`examples/validation/fexi_axr_findings.md`).

## [3.0.0] - 2026-11-02

dmipy-sim is the Monte-Carlo replay engine of the dmipy ecosystem. It walks a declared tissue substrate once and keeps
the walk as an acquisition-independent replay pack. [dmipy-replay](https://github.com/dmrai-lab/dmipy-replay) turns the
pack into a signal or an image.

- **The walk.** `walk(spec, ..., envelope=)` walks a `SubstrateSpec` (CACTUS meshes, CATERPillar sphere unions, DiSCo
  strands, label volumes, packed and analytic geometries, triangle meshes) and returns a `CoefficientWalk`: per walker,
  the exact endpoints and the sine-band projections of its path, its wall contact, its pool events and its
  susceptibility field modes, accumulated while the walk runs. Wall physics (surface relaxivity, membrane permeability,
  magnetization transfer), per-pool T2 and T1, and susceptibility fields are part of the walk. `simulate` is the walk and
  the replay in one call.
- **The envelope is physical.** `envelope=` is required; there is no default scanner class. An `Envelope` states the
  temporal band, the per-axis `G_max`, and the field and tissue bounds the pack is certified for, and
  `envelope_of(scanner)` (dmipy-replay) gives a scanner class's envelope. The walk accumulates the band the envelope
  states, and the pack built from it carries the envelope.
- **Stated inputs.** A certified or physical quantity comes from a stated input or is refused by name; no value stands
  in for a missing one (dmipy-sim#882). A pack written by an older producer is refused by name: re-walk it.
- **The pack.** `build_replay_pack` quantises and certifies a walk as a `.rpk` replay pack
  ([replay-pack-spec](https://github.com/dmrai-lab/replay-pack-spec)). `save_pack` writes it; `truncate_pack`,
  `rewindow_pack`, `prefix_pack`, `extend_pack` and `merge_packs` make another pack from one or more; `publish` sends it
  to [SubstrateCommons](https://huggingface.co/SubstrateCommons) on the Hugging Face Hub. `walk_from_positions` turns
  another engine's positions into a C0-only pack.
- **The signal side lives in dmipy-replay.** dmipy-sim produces packs and has no `ReplayPack`: the pack reader, the
  replay, the pose replay, the replay phantoms, the acquisition object (`ScannerSequence`), the sequence builders, the
  scanner catalogue and the substrate specification are dmipy-replay's. dmipy-sim re-exports none of them, and a name
  that moved is refused by name with its new spelling: `dmipy_sim.pgse` raises an `AttributeError` that says
  `from dmipy_replay import pgse`. dmipy-sim keeps the Pulseq writer (`to_pulseq`).
- **The forms are dmipy-analytic's.** The physical constants (`GAMMA`, `MU0`), the matrix method (`MatrixPore`, the
  Neumann roots), the field forms of the hollow cylinder and of a current segment, and the two-pool magnetization-transfer
  forms are [dmipy-analytic](https://github.com/dmrai-lab/dmipy-analytic)'s, and dmipy-sim imports them. The moved names
  are refused by name with their new spelling (`dmipy_sim.GAMMA`, the magnetization-transfer forms of `engine.mt`, the field forms of
  `fields.hollow_cylinder` and `acquisition.coil`), and `dmipy_sim.math.matrix_method`, which moved whole, is gone:
  `from dmipy_sim.math import matrix_method` says where it went.
- **Names.** `dmipy_sim.Pool` is the physics of one pool of a geometry's `Compartments`; the pool of a substrate spec is
  dmipy-replay's `PoolSpec`. A sequence's b-value is `seq.b()` and its b-tensor `seq.btensor()`; `calc_b` and
  `calc_btensor` are refused by name.
- **Runs on** JAX, on CPU or CUDA 12. Python >= 3.10, NumPy >= 2.0, SciPy >= 1.11, JAX >= 0.6.2. The replay guide
  ([docs/replay-guide](docs/replay-guide/README.md)) is the manual for the replay side, and every code block on its
  pages runs in the test suite.
- **Pairs with** [dmipy-replay](https://github.com/dmrai-lab/dmipy-replay) `>=0.2,<0.3` and dmipy-analytic
  (`[special,jax]`) `>=3,<4`, and with dmipy-fit and dmipy-design `>=3,<4`.
- **Licence** AGPL-3.0-only or commercial ([LICENSING.md](LICENSING.md)).
