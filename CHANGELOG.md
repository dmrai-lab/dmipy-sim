# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

- **A walk states whether it starts at spatial equilibrium, and the step law's term for a start that does not**
  (#930). `physics.TRANSIENT_COEFFICIENT` = 0.3: a read at time `t` after walkers were placed out of spatial
  equilibrium carries up to `0.3 h^2 / (D t)` beyond the law's other terms, on the exchange across a membrane and on
  each face's bound pool (`physics.transient_error`). It is the non-smooth-data error of a first-order step: the law's
  wall and membrane terms hold each mode's rate to first order in that rate, and a start out of equilibrium puts weight
  on the fast modes, whose contributions sum to `O(dt / t)`. It is computed on the walk's exact chains from a one-sided
  start:
  - the exchanged fraction, at most `0.084 h^2 / (D t)` over `kappa l / D` from 0.05 to 50;
  - the far face's bound pool, `0.17` to `0.24 h^2 / (D t)` at `kappa h / D = 0.045`, the same at two binding
    reactivities.

  A face's local time read on its own is first order out of equilibrium (about `0.5 h` times the density's normal
  gradient over the density), which the transmission rule and the crossing share take out of the exchange and the
  pools. A walk records its start as `work["start"]`, from `physics.start_record`:
  - at equilibrium on a geometry with no permeable wall, or when its placement is uniform over every pool
    (`Geometry.seeds_every_pool`: packed cells seeded over the cell, the packed myelinated cell);
  - out of it when a permeable wall separates a seeded pool from an empty one;
  - not established for starts it was handed.

  A start that is not at equilibrium on a walk with a permeable wall is said at the walk (a `UserWarning` stating the
  term). `CoefficientWalk.start_error(t)` gives a read's term and flag, and a pack carries the record in
  `provenance.walk.start`. `MT_COEFFICIENT` stays 0.06: the per-face excess #926 measured from a one-sided start is
  this start term, together with the share's remainder, which is absent from a uniform start. The step is not changed.
- **Each face's MT bound pool is second order in the step** (#926): a walker that binds at an encounter it crossed
  binds on the near face or the far face, half each (`geometry._boundary.NEAR_FACE_SHARE`), where it bound on the far
  face (the face it ends on) every time. The crossing and the bind share the encounter's local time, the near face's
  before the crossing and the far face's after it, so given both the bind falls on either face with probability 1/2 to
  first order. A walker bound on the near face is seen at its step's start, on that face's side, so a bound walker's
  recorded position tells its face; it resumes from its step's end as before, and the face has a draw of its own, so
  the walk itself -- every free walker, the total bound pool, the exchange -- is unchanged to the bit. On the walk's
  exact chain against `BlochTorrey1D` from free walkers on one side of the membrane: each face's bound fraction was
  first order (4.7e-3 too much on the far face at `kappa h / D = 0.5`, `h = 0.09 R`, `t = 0.1 R^2/D`, halving with `h`)
  and is now second order (1e-5, falling by three to four with `h`); at `kappa h / D` of 5 and 50 each face carries
  half the total's error. A test holds each face's bound fraction and bound signal to the continuum's. A walker the
  burn-in leaves bound on the near face of a crossing is seen where it resumes until released (the burn-in ends at
  equilibrium, where as many are seen across either way). The packed myelinated walk and the forward vector-Bloch
  oracle are unchanged. An MT walk across a permeable wall records different positions (and so bands) for the walkers
  bound on a near face, and nothing else.
- **The MT burn-in stops at the bound pool's equilibrium** (#925): its plateau test (the occupancy changing by under 1 %
  between two chunks of one dwell) ended while the bound fraction was still climbing (0.088 against 0.091 at `0.3
  R^2/D` on the permeable slab), so an MT walk started out of equilibrium. Where the geometry gives the equilibrium,
  `P_eq = x / (1 + x)`, `x = kappa_MT tau S / V` with the geometry's own `S` and `V` (`mt.equilibrium_bound_fraction`:
  sealed sphere and cylinder, permeable slab, packed cylinders and spheres seeded over the cell, sealed pool by pool),
  the burn-in runs at least five relaxation times (the larger of the dwell and the pore's `1 / (D pore_rate)`) and stops
  with the ensemble's bound fraction within `0.01 P_eq` plus two standard errors; elsewhere it stops when the two halves
  of a window of eight dwells differ by under 1 % plus two standard errors of their difference. The criterion and where
  it stopped are `work["mt_equilibration"]`; one that does not hold within its cap warns. Walks without MT and walks
  with `equilibrate_binding` `'fast'` or `'off'` are unchanged; MT walks start nearer their equilibrium.
- **MT binding is second order in the step** (#921): a walker binds at an encounter with the binding clock's
  probability `1 - exp(-kappa_MT l / D)` (was `min(1, kappa_MT l / D)`), and the bind's dwell has the mean `tau
  exp(kappa_MT l / (2 D))` (`geometry._boundary.dwell_scale`): a released walker resumes the unspent half of the
  encounter it bound in and may rebind there, and those geminate rebinds join the dwell. The walker is still held, and
  released, at its step's end -- of the release positions compared (the step's end, the face, the face plus a
  reflected step, a uniform point of the step) it is the only one whose bound fraction is second order, and moving it
  cannot restore the rebinding bursts without unbalancing the bound fraction. On the walk's exact chain against
  `BlochTorrey1D` with bound pools: the bound pool's signal was first order (late signal -0.011 at `kappa_MT h/D =
  0.2`, `h = 0.09 R`, halving with `h`, the same at every permeability) and is now second order (+0.001, quartering),
  with the bound fraction and the exchange second order as before. The step law gains the binding's term,
  `physics.MT_COEFFICIENT = 0.06` (the bound pool's binding reactivity and dwell within `MT_COEFFICIENT (kappa_MT h / D
  + h / l)^2`; `binding_error`, `binding_step`, `binding_length`), applied with the binding rule when a walk binds; it
  asks nothing at qMT reactivities. The packed myelinated walk and the forward vector-Bloch oracle take the same law
  and dwell; their release step, held to its end, is second order (measured on the chain). MT walks are not
  bit-identical to before.
- **The MT walk binds at every permeable wall it can walk** (#909): geometries that carry the walker's side (packed
  cylinders and spheres, with a permeability per object, the side following the granted crossing), a permeability
  over time (the burn-in that equilibrates the bound pool reads the first interval), and the forward vector-Bloch
  oracle (a permeability constant in time, as its walk without binding takes it). Against `BlochTorrey1D` with binding
  on the slab (constant, over time, and the forward oracle) and, for the packed cylinder, against the isolated
  `Cylinder`. A Mesh with a permeable wall and a schedule in the forward oracle are still refused by name. Impermeable
  MT walks and the forward oracle are unchanged to the bit.
- **The MT dwell is counted continuously** (#918), not in whole steps: the encounter falls uniformly in its step, the
  dwell left at the step's end is the draw less the time from the encounter to the end, and a walker released inside a
  step walks the rest of it (the packed myelinated walk and the forward vector-Bloch oracle take the encounter's
  fraction of the step the same way, the Bloch spin evolving at the bound share of each sub-step). The bound time read
  at the block ends was too long by about `2 dt / tau` of itself (a late signal 0.018 low at a 17-step dwell, 5 se on
  30 000 walkers); it is unbiased now. MT walks are no longer bit-identical to before: an impermeable sphere's bound
  fraction moves by -1.4 % at a 33-step dwell, -1.5 % at 17, 0 % at 67 (inside the standard error, 1 %).
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
