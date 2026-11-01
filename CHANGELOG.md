# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### 3.0.0

dmipy-sim is the Monte-Carlo replay engine of the dmipy ecosystem. It walks a declared tissue substrate once and keeps
the walk as an acquisition-independent replay pack that answers any acquisition.

- **The walk.** `walk(spec, ...)` walks a `SubstrateSpec` (CACTUS meshes, CATERPillar sphere unions, DiSCo strands,
  label volumes, packed and analytic geometries, triangle meshes) and returns a `CoefficientWalk`: per walker, the
  exact endpoints and the sine-band projections of its path, its wall contact, its pool events and its susceptibility
  field modes, accumulated while the walk runs at the band the scanner class certifies. Wall physics (surface
  relaxivity, membrane permeability, magnetization transfer), per-pool T2 and T1 and susceptibility fields are part of
  the walk. `simulate` is the walk and the replay in one call; `backend=` selects another execution of the same physics.
- **The pack.** `build_replay_pack` quantises and certifies a walk as a `.rpk` replay pack
  ([replay-pack-spec](https://github.com/dmrai-lab/replay-pack-spec)); `walk_from_positions` turns another engine's
  positions into a C0-only pack. Packs are published to and read from [SubstrateCommons](https://huggingface.co/SubstrateCommons)
  on the Hugging Face Hub.
- **The replay.** A pack replays any gradient waveform, RF schedule (vector Bloch), tissue value, field strength and
  orientation, or a distribution of orientations (the pose response as an SO(3) expansion). A replay phantom tiles
  packs over a voxel grid (`Phantom`, `Grid`, `Pose`). The acquisition is one object, `ScannerSequence`, with sequence
  builders (`pgse`, `pgste`, `ogse`, `cpmg`, `splice`, `gre`, `ste`, `pte`), Pulseq import and export, and a cited
  scanner catalogue (`ScannerLimits`).
- **Runs on** JAX, on CPU or CUDA 12. Python >= 3.10, NumPy >= 2.0, JAX >= 0.6.2.
- **Pairs with** [dmipy-replay](https://github.com/dmrai-lab/dmipy-replay) `>=0.1,<0.3` (the NumPy-only reader: the
  acquisition object, builders, scanner catalogue and the pack reader), and with dmipy-fit and dmipy-design `>=3,<4`.
- **Licence** AGPL-3.0-only or commercial ([LICENSING.md](LICENSING.md)).

> **Pending, not yet landed (this entry is completed before the tag).**
> The physical `Envelope` ([#855](https://github.com/dmrai-lab/dmipy-sim/issues/855)): `Envelope(band_hz, G_max, field
> and tissue bounds)`, `envelope_of(scanner)` presets and `walk(spec, ..., envelope=...)`.
> The replay move to dmipy-replay: the pose replay, `pathways` and `ReplayPhantom` (with `fod` and `study`), the
> dmipy-sim `ReplayPack` subclass removed, and `truncate` / `extend` / `rewindow` / `prefix` / `publish` / `save` as
> dmipy-sim functions that take a pack.
