# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

- **`Substrate()` and `Substrate.canonical()` default to a membrane permeability of 1.447e-6 m/s, Nedjati-Gilani et
  al. 2017's in-vivo human white matter, instead of 1e-5 m/s, which no cited paper gave.** The value is the paper's own
  cylinder relation `kappa = R / (2 tau_i)` (Section 2.2) at the splenium's `alpha` = 1.65 um, a volume-weighted radius
  index, and `tau_i` = 0.57 s (Table 3, row CC-S, 3 T): the paper does not print the permeability, the constants table
  derives it under `intracellular_residence_time` and stores it there once. `kappa_membrane` now holds no number of its
  own but a `value_from` reference to that entry, which `get_constant` resolves into a copy when it is called (a
  source holding both a `value` and a `value_from` is refused), and the `Substrate` dataclass reads it through a
  `default_factory`, so `get_value('kappa_membrane')`, `canonical_white_matter()['kappa']`, `Substrate.canonical()` and
  `Substrate()` all read that one entry at call time. The row's credit to Nilsson et al. 2013 is gone; its citation is
  Nedjati-Gilani 2017. The default substrate therefore exchanges water about seven times more slowly (1e-5 / 1.447e-6
  = 6.9; the intra-axonal residence time `R / (2 kappa)` grows by the same factor). Code that wants the old behaviour
  passes `kappa=1e-5`.
- **A walk records its membrane trials in a bounded number of slots per walker, one per (object, band) pair met.** The
  record of `record_trials=` holds, per walker, one row per (object, time band) pair the walker met, in the order it
  first met them (`TrialRecord.obj`, `band`, `n`, `N`, `S` of shape `(n_walkers, width)`, and `rows`, the pairs each
  walker met), instead of one cell per pair of the cell: its memory is `bytes_per_walker(slots, modes, itemsize)`,
  independent of the cell's object count, and a cell of thousands of objects is recorded. The numbers are those of
  one cell per pair, exactly: `TrialRecord.to_dense()` gives that form for a small cell (refused by name above
  `MAX_DENSE_BYTES_PER_WALKER` bytes per walker), its `n` and `N` equal and its `S` the same bits. The slots are
  `TrialRecordSpec(slots=)`, or, when None, the largest count of a pilot of the walk's own walkers with the events
  buffer's margin (`SLOTS_MARGIN`, `pilot_capacity`), stated in `work["trial_slots"]`; a walker that meets more pairs
  than the slots refuses the walk by name with the count it reached and a suggested capacity, no trial dropped.
  `MAX_BYTES_PER_WALKER` caps the slots' bytes. `take`, `bands`, `same_layout`, `concatenate`, `to_arrays` and
  `from_arrays` work on the rows; `totals()` and `walker(w)` read them. A walk file holds the rows as `trial_row_*`
  and `trial_rows`; a file holding the cells (`trial_n`, `trial_N`, `trial_S`) is refused by name. A backend that
  declares `TRIAL_RECORD` is handed `WalkRequest.trial_slots` and returns `(key, n, N, S, count, exceed)` per slot.
- **The field of a periodic cell of many parallel hollow cylinders is read from the near cylinders exactly and from a
  per-cell series for all the others, with a derived remainder bound.** `HollowCylinderField(read="near_far")` lays a
  grid of cells about the mean outer diameter over the cell (the shared periodic planar bucketing,
  `geometry._grid.bucket_periodic_plane`, which the concentric wall search now also uses); per cell the near list is
  every cylinder whose outer wall is within `near_cells` cell sides of the cell, summed in closed form at its nearest
  image exactly as the exact read sums it (lumen and sheath included), and the rest is the Taylor series of one
  analytic function of `z = x + i y` about the cell's centre -- every far cylinder's periodic field (all images) plus
  every cylinder's lattice regular part -- with coefficients summed on the host in float64, read by Horner's rule.
  Outside its outer wall a cylinder's six non-local channels are `-(b^2 - a^2)/2` or `(b^2 - a^2)/12` times the real or
  imaginary part of `1/z^2` and its other six are zero (tested against the closed form). The remainder bound,
  computed per cell, is the geometric tail `sum_j |b_j^2 - a_j^2| tail(rho / d_j, order) / d_j^2` over every far
  singularity (the nearest image of every far cylinder, a closed-form bound for every other image), `rho` the cell's
  half-diagonal. The order is the smallest whose largest bound is at most `tolerance` (1e-6) of the exact field's rms
  in its largest channel, the near set widened by a cell when no order up to 24 meets it, refused by name beyond four
  cells. `read=None` reads near / far in a periodic cell of at least 512 cylinders and exactly otherwise; the record's
  `summation` is `"lattice_near_far"` with `series_order`, `near_cells`, `series_cells_per_side`, `series_list_length`,
  `series_tolerance`, `series_tolerance_abs`, `series_bound_max` and `series_table_bytes`, a record is rebuilt as it
  was read and `from_meta(read=)` refuses the other read by name; `remainder_bound(points)` states the bound per point
  and channel. `device_arrays` (the exact read's six arrays) is refused for a near / far field.
- **The wall search of a packed cell of myelinated and plain cylinders reads a grid.** In a periodic cell,
  `concentric_wall_kernel` finds the nearest cylinder, the `n_cand` candidates and the end-of-step owner from the list
  of the grid cell holding the walker (a table built once on the host through the shared cell grid, `geometry._grid`),
  so neither of a step's two wall searches reads every cylinder. Each list holds every cylinder whose wall distance at
  the cell's centre is at most `D + 2 (h + pad)` -- `D` the `n_cand`-th smallest there, `h` the cell's half-diagonal,
  `pad` a float32 pad of 16 eps32 L, so twice the half-diagonal plus twice the pad -- which by the argument in the
  kernel's docstring contains the `n_cand` nearest at any point of the cell; the grid's integers are the dense search's at every position, and a walk is the same bits under either. The kernel
  states which search ran (`wall.search`, `"grid"` or `"dense"`) and the grid's layout (`wall.grid_report`: cells per
  side, list length, bytes, the priced options); the grid runs when the cell is periodic, at least `n_cand` cylinders
  are real and the priced list length is below the number of real cylinders within 64 MB, the dense search otherwise.
  `concentric_wall_kernel(search=)` and `PackedMyelinatedCylinders.wall_search` force either; a grid that cannot be
  built is refused by name.
- **Each walker of a walk carries its own membrane permeability, stated by the caller.**
  `_walk_geometry(permeability_factor=)` (and `walk(permeability_factor=)` on a spec of one geometry) takes one float32
  factor `f_w` in `[0, 1]` per walker, in walker order, formed into the walker's permeability in float32 by its cell's
  rule: a `PackedMyelinatedCylinders` walker meets plain cylinder `k` at `float32(kappa_outer_k * f_w)` (m/s), divided
  by the diffusivity of the pool it leaves, and the sheathed walls at their stated values; a `PackedCylinders` walker
  reads `float32(float32(kappa / D) * f_w)` (1/m). Its trials are decided by the walk's own words and uniforms. `None`
  and an array of ones are the walk at the stated permeability, bit for bit. The walk returns the array as
  `CoefficientWalk.permeability_factor` and states the cell's rule, the array's length and dtype in
  `work["permeability_factor"]`; a walk file whose array disagrees with that record is refused on load. Every other
  geometry, a cell with nothing to scale, a schedule in time, MT binding, another backend and a factor outside `[0, 1]`
  refuse it by name.
- **A walk accumulates the time integral of its displacement over declared time segments.** `_walk_geometry(segments=[(a, b), ...])`
  adds per walker and segment `int_a^b (x(t) - x(0)) dt`, the trapezoid of its positions at the walk's own step, as
  `CoefficientWalk.segment_rows` beside the unchanged bands, and `segment_rows` of the time-segment layout reads it as
  a source whose certificate carries no band truncation; off by default, refused by name off the block grid, across a
  storage window, empty, reversed or not increasing, and on any backend but the JAX kernels.
- **A backend may declare the packed myelinated class, a closed-form field read in the walk and the record of the
  membrane trials.** Three capability names beside `PERMEABILITY_TABLE`, `STORAGE_WINDOWS` and `MT_BINDING` in
  `engine.backends`, each in `JAX_CAPABILITIES`: `PACKED_MYELINATED` (a `PackedMyelinatedCylinders` walk is handed to the
  backend, which reads the cell through `PackedMyelinatedCylinders.cell_arrays()`, a read-only `MyelinatedCell` of the
  arrays and constants its wall kernel is built from), `FIELD_IN_WALK` (a closed-form basis such as a
  `HollowCylinderField` is handed as `WalkRequest.field`, its data public through `HollowCylinderField.device_arrays()`,
  and its modes come back as a raster's do) and `TRIAL_RECORD` (`record_trials=` hands the backend the spec and its
  resolution, `WalkRequest.trial_record`, `trial_steps`, `trial_radii`, `trial_u_max`, and takes back
  `CoefficientBatch.trial_record` per batch, joined into `CoefficientWalk.trial_record`). A capability decides only
  whether the existing refusal fires: a backend that does not declare it is refused by the same string as before, and
  the isolated `MyelinatedCylinder` stays a path of the JAX kernels alone. A declaring backend's missing or mis-shaped
  field modes or trial record are refused by name.
- **The realised acceptance of a membrane trial, exactly** (no change to any walk).
  `geometry._boundary.realised_acceptance(kappa_over_D, d_perp, dtype=numpy.float32)` returns the integer threshold
  `t` in `[0, 2^32]` of the plain walls' transmission trial: a 32-bit word `w` is granted if and only if `w < t`, so the
  trial's realised probability is exactly `t / 2^32`. It runs the kernel's own `transmit_probability` and the threshold
  `trial_granted` decides with (one definition, `_trial_threshold`), on float32 arguments or, under JAX's x64 mode, on
  float64 ones (refused by name without it). The `trial_granted` docstring states the derived distance from
  `x / (1 + x)`: `|r - p| < 2^-32 + eps p`, `eps = (3 u + u^2) / (1 - u)`, `u = 2^-24`, for float32 inputs each non-negative, zero
  or normal, and `kappa_over_D < 2^127`, and the behaviour outside that domain; it replaces the relative
  `2^-23`, which an input with a float32 `p` `1.46 x 2^-23` off the formula violates. `tools/realised_acceptance_bound.py`
  checks the bound over every float32 `x` of `[2^-34, 0.1124]` and `[2^-34, 2^26]` (records
  `docs/realised-acceptance-bound-{small,full}.json`); `tests/geometry/test_realised_acceptance.py` holds the
  function to the kernel's decisions on the edge table in both dtypes and on 100 000 random inputs.
- **Recorded bits per CPU family** (tests, CI). The recorded hashes of the all-myelinated reference walks are kept per
  family of compiled CPU arithmetic in `tests/recorded_bits.py` `FAMILIES`, keyed by the family's canary (the hash of a
  small fixed walk), and the `recorded_bits` fixture returns this host's family's values. The canary selects the family
  only; the recorded tests are the evidence. In the `recorded-bits` job (`DMIPY_SIM_RECORDED_BITS=1`) a canary of no
  recorded family fails naming it and another JAX build is an error; elsewhere both skip with that reason, and a host
  of a recorded family on the recorded build runs the tests. The job prints the runner's CPU model, flags, numpy's SIMD
  dispatch and canary on every run, and on a failure the runner's entry (`tools/record_recorded_bits.py`).
  `tools/recorded_bits_canary.py` is removed: `tests/recorded_bits.py` `canary` is the canary.
- **The field of a periodic cell of hollow cylinders is exactly periodic** (behaviour change: a walk of a
  `PackedMyelinatedCylinders` cell with `HollowCylinderField.of(cell)` reads a different field). Per sheath the field is
  the closed form at its nearest image plus every other image's tail through the regular part of the square lattice's
  Weierstrass function, `fields.hollow_cylinder.lattice_regular_part` (row-summed, DLMF 23.8.3): periodic exactly,
  harmonic outside the sources, the walls the closed form's. It replaces the closed form summed over six images per
  side, which jumped by up to 9e-4 of the field at the cell's faces and differed from its limit by up to 4.4e-4. The
  cell mean is not subtracted (as before) and is stated in closed form in the class docstring. `n_images` and
  `CELL_IMAGES` are removed; `device_read(dtype)` and `channels(points, dtype)` read in float32 (the walk's) or float64
  (the reference, under `jax_enable_x64`). The record states `summation` (`"lattice"` or `"none"`) and `lattice_rows`;
  a walk file whose field record names `n_images` for a cell is refused by name on load (re-walk it); an isolated
  cylinder's record reads as before. The constructor refuses non-finite input and a sheath whose outer radius reaches
  half the cell's side. The forward vector-Bloch provider `MyelinSusceptibility` keeps its own image sum.
- **A cell of myelinated and plain cylinders has an equilibrium bound fraction, per water domain** (behaviour
  change: the MT burn-in of such a cell stops by its equilibrium, and `equilibrate_binding='fast'` is accepted there).
  `engine.mt.equilibrium_domains(geometry, kappa_MT, dwell_time)` states, for a `PackedMyelinatedCylinders` cell whose
  sheaths are unseeded and sealed from every seeded pool, one record per water domain that holds seeded water
  (`pools`, `cylinders`, `volume`, `surface`, `surface_to_volume`, `weight`, `p_eq`; per unit length): the lumen of
  each myelinated cylinder (`S / V = 2 / a`), the lumen of each sealed plain cylinder (`2 / b`), and the exterior, the
  extra pool with the lumina of the open plain cylinders, whose surface holds the outer face of every sheath and
  sealed plain wall once and both faces of every open plain wall. Each domain is at `P = x / (1 + x)`,
  `x = kappa_MT dwell_time S / V`; its weight is its seeded area over the cell's. `equilibrium_bound_fraction` of the
  cell is the seeded-weight sum of the domains; a sheath that is seeded or reachable from a seeded pool through an
  open face has none (`EquilibriumDomains.reason`). The walk's `work["mt_equilibration"]` of such a cell carries
  `equilibrium_domains`, or `p_eq_reason` beside `p_eq: None`; every other geometry's record is unchanged.
  `'fast'` seeds each walker bound with its own start domain's `P` (by `classify_position` of its start) and refuses a
  cell without domains with the reason, and a walker that starts in a sheath. The cell's reach is one property,
  `PackedMyelinatedCylinders.reachable_pools`, which `placement_is_stationary` reads; its domains are
  `PackedMyelinatedCylinders.water_domains()`. `surface_to_volume` stays `None` for the cell.
- **A walk records its membrane trials at its plain walls** (`record_trials=`, off by default). `_walk_geometry` and
  `walk` take a `dmipy_sim.trial_record.TrialRecordSpec(modes=M, band_edges=...)` (seconds from 0 to `T_max`, at most
  64 bands, containing every storage-window boundary) and return `CoefficientWalk.trial_record`, a `TrialRecord`: per
  walker and per (object, time band) cell the trials `n` and the granted trials `N` (int32) and the sums `S_j = sum u
  T_j(2 u / u_max - 1)`, `j = 0 .. M - 1`, in the walk's accumulation dtype, `u = 2 d_perp / R` the trial's overshoot
  and `u_max = 2 LONGEST_STEP h / R` each object's bound (`h = sqrt(6 D dt)` at the larger diffusivity of the wall's
  two pools). Every encounter of a plain wall (`PackedCylinders`, `PackedSpheres`, the plain cylinders of
  `PackedMyelinatedCylinders`) is a trial, a sealed object's included; a trial belongs to its step and a step to the band
  whose left-open, right-closed interval `(edge_k, edge_k+1]` contains its end time (an end on an edge in the earlier
  band, so `window(i)` holds exactly its own steps); a step a bound walker discards
  records none. The record turns `count_walls` on (`sum n` over an all-plain cell is `work["n_hits"]`, `sum N` is
  `work["n_crossings"]`), draws nothing and rounds nothing, and enters none of the walk's arithmetic: one step of a wall kernel is the same
  bits with and without it, and the whole walk with and without it is the same bits on the recording build (another
  JAX build may compile the two programs to positions that differ in the last bits, which reflections can grow into
  another encounter: two valid walks of one law); a walk without it is the same program as before (the same optimised
  HLO, instruction for instruction). It is carried by `window`, `take`, `join_walks`, `save` / `load` and
  `nbytes`, and holds `n_objects * n_bands * (8 + M * itemsize)` bytes per walker. Refused by name: a geometry with no
  plain wall, a backend other than the JAX kernels, `equilibrate_binding='burnin'`, a field channel without a stated
  `K_c3` and `field_read_interval`, malformed edges, more than 65536 bytes per walker, and a walk with a trial beyond
  `u_max`. The wall kernels log the trials on `WallHit.trials` (`_boundary.TrialLog`) when the geometry's `log_trials`
  is set, which the walk does for its duration.
- **A plain wall's transmission trial is decided by a 32-bit word** (behaviour change: the walk's stream changes at
  the trials it decides). `packed_wall_kernel` (`PackedCylinders`, `PackedSpheres`) and the plain cylinders of
  `PackedMyelinatedCylinders` draw one `uint32` word per encounter (`jax.random.bits`, the word the float32 uniform of
  the same key was made of, so no other draw moves) and grant the crossing when the word is below `floor(p 2^32)`
  (`geometry._boundary.trial_granted`; always at `p >= 1`, never at `p <= 0`). The realised probability is
  `floor(p 2^32) / 2^32`, within `2^-32` of the float32 `p` and never above it, where the uniform realised
  `ceil(p 2^23) / 2^23`, above `p` by up to `2^-23` (at `p = 1e-4` a relative bias of `1.7e-4`, now `1.7e-6`). A
  decision differs from the old one only when its word lies between the two grids, with probability at most `2^-23`
  per trial and about `2^-24` on average: a walker of 4 570 trials (a 60 ms walk of a dense cell) changes path with
  probability about `2.7e-4`. A walk with no plain permeable wall is unchanged. Known cost of the correction: on a
  dense cell of six plain cylinders (64 walkers, 60 ms, CPU) the walk takes 6 % longer (71.3 s against 67.2 s, run
  without compilation, the median of three interleaved repeats); deriving the threshold from the float's bits in
  integers was slower (74.8 s). Known: the sheathed cylinder's one
  trial per step and the trials of the other permeable geometries (the analytic geometries, the mesh, the label
  volume) keep their float32 uniform.
- **A packed myelinated cell's spec states the pools it seeds, and a pool that holds no water is not seeded** (behaviour change). `PackedMyelinatedCylinders` seeds by default every pool whose `Pool.water_fraction` is not 0, by area among them (`seeded_pools`); its spec's `seeding.pools` lists exactly those pools, or the pools `seed_pools=` names, and `geometry_from_spec` reads that list literally, so a cell built from its spec places the same walkers in the same pools. A cell whose every pool holds water walks byte for byte as before. A default cell with a sheath (or another pool) of water fraction 0 no longer starts walkers in it: on the four-cylinder cell of the tests, 662 of 4000 walkers started in its dry sheaths and none does now, so its walks change; its spec's seeding list ([0, 1]) does not. `seed_pools=` naming a pool of water fraction 0 is refused by name; naming exactly the pools that hold water is the default (`seed_pools` is `None`). `UnstatedSeedingError` is removed: every seeding of the cell is stated by its spec, and the cell is walked through it (`geometry.spec`, `walk(cell)`, with the field). `Geometry.seeds_every_pool` is replaced by `Geometry.placement_is_stationary`, what the walk's start record reads: whether the placement's law is the walk's stationary law. For the packed myelinated cell it is True when every pool of positive area that is not seeded is sealed from every seeded pool (each face read per cylinder; a dry pool behind an open face fills like any other), so a cell whose unseeded sheaths are sealed records its start at spatial equilibrium and one with an unseeded sheath behind an open face does not; a cell that seeds every pool records it as before. Known limitation: within the seeded pools the placement is by area, not by area times water fraction as the specification's `uniform_by_volume` reads, for a pool whose water fraction lies between 0 and 1.
- **Myelinated and plain cylinders in one periodic cell.** `PackedMyelinatedCylinders` takes a g-ratio of 1 for a cylinder without a sheath: one wall at its radius between its lumen and the extra pool, with `kappa_outer` / `rho_outer`, a transmission trial at every encounter of a step as `PackedCylinders` makes (a myelinated cylinder keeps one trial per step, at the step's first encounter with any wall). A g-ratio outside (0, 1], a myelinated cylinder whose sheath has no thickness in float32 and an axolemma property given to a plain cylinder are refused by name. The field (`HollowCylinderField.of`, `field_grid_of`) is the myelinated cylinders' sheaths only; the spec writes a plain cylinder as one wall (`membrane`, lumen | extra) over its own instances and `geometry_from_spec` builds the cell back, myelinated cylinders first. `seed_pools=` names the pools the walkers start in. `simulate` on the cell takes `r0=` and a pinned `sub_steps`. A cell of myelinated cylinders only walks byte for byte as before.
- **The walk states the precision of its projection** (`block["projection_precision"]`): the dtype name of the product a ring's moments are projected with, read from `accumulate.projection_precision(dtype, K, m)` (the output of `ring_projection` itself), so a consumer reads the precision of the projection as data. `None` for a producer that states none (a backend that does not, a walk read from stored positions); a bundle walk joined from pools that differ states `"mixed"`. No array changes.
- **The walk's accumulation dtype is an argument** (`accumulation_dtype`, `"float32"` by default, or `"float64"`).
  `_walk_geometry`, `walk`, `simulate` (and `walk_cost`, `certify_bounce_budget`, `simulate_mt_trajectories` through
  their keywords) take it: the moments of each block, the ring projection against `block_table`, the level-2
  accumulator and every channel's quadratic variation are held in it, and `position_coeffs`, `qv_positions`, the
  contact and the field modes are returned in it. The walk itself (draws, steps, positions, walls) stays float32, so a
  float64 accumulation walks the same walkers as the float32 one and the difference of the two calls is the float32
  accumulation's rounding; `tools/accumulation_dtype.py` measures it. The float32 default returns the same bytes as
  before. `"float64"` needs JAX's x64 mode on in the calling process and is refused by name when it is off, on a
  backend other than the JAX kernels, on the adaptive walk and on the fused oracle; any other value is refused by name.
  `block["precision"]` is the dtype of the accumulators the walk returned; the module constant
  `accumulate.PRECISION`, which the arithmetic never read, is removed (`accumulate.ACCUMULATION` became
  `accumulate.accumulation_record(dtype)`). `block_table(xp=jax.numpy)` takes the table's `dtype`, required.
- **The MT walk records a pool label, and a backend may walk MT binding** (dmrai-lab/dmrai#72). The `"mt"` kind of
  `_walk_geometry` (`simulate_mt_trajectories`) carries, beside the position a walker is seen at, the compartment of
  the position it resumes from and the `_pool2` label of the seen position: the near face's pool while it is bound on
  the near face of an encounter it crossed, the resume position's otherwise. So every recorded time has the pool of
  the recorded position, bound, at the release and free; a permeable MT walk records the label's changes as C1 events
  (`comp_events`) and every MT walk its start labels (`compartment`). The MT walk's work record is its own (it read the
  seen position, so it reported 0 hits, crossings and exhausted steps), and a walker held through its dwell counts no
  encounter of the move it discards. `WalkRequest` gains `kappa_MT_over_D` and `dwell_time`; a backend that declares
  `backends.MT_BINDING` walks the `"mt"` kind, is handed each walker's dwell at the start (`walk_batch(dwell0=)`) and
  returns `bound_events`; the JAX burn-in is refused for a backend (`equilibrate_binding='fast'` or `'off'`).
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
