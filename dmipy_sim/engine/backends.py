"""Backends: another execution of the same physics, resolved by name (dmrai-lab/dmipy-sim#635).

A backend walks a batch of walkers the way the JAX kernels of :mod:`.core` walk it -- the same geometry, the same
starts and keys, the same sub-step and blocks -- and returns what they return: the walk's **coefficient form**
(:class:`CoefficientBatch`, dmrai-lab/dmipy-sim#740, #747), per walker the exact start and end and the bands of
every channel accumulated while the walk ran, the quadratic variation per channel, the events of a walker that
changes pool, and the work record. dmipy-sim stays the reference physics: a backend reads every constant from the
geometry it is handed and carries none of its own.

What a backend signs
--------------------
- **One physics.** The rule is dmipy-sim's; the backend executes it.
- **One form.** The producer takes a :class:`CoefficientBatch` as it takes its own batch -- the same
  :class:`~dmipy_sim.coefficient_walk.CoefficientWalk`, the same header -- and converts nothing: a backend that
  returns a walk of saves (a :class:`WalkBatch`) for the coefficient walk is refused by name. The only positions a
  backend returns are the block ends of the batch's ``raw_subset`` (:attr:`CoefficientBatch.saves`).
- **Parity, not identity.** A different RNG stream and FMA policy make bit-identity impossible, and it is not stable
  across XLA compilations of one program either (#623). A backend is admitted per engine class by the public
  validation of that class run on it and by the work record within Poisson noise: :func:`parity_report`, on the
  coefficient form. One case is a hard invariant rather than a rate (``strands_void``, #675): starts in the void
  beyond a tube's own gather reach, held to per-block containment of the raw subset -- a gather read as unbounded
  where it found nothing leaks a walker through the next wall it never actually tested.
- **Determinism per (seed, architecture).** A trajectory is a function of the key; the pack says which backend and
  architecture walked it (``work["backend"]`` and the run record).
- **Refusal over fallback.** A backend that does not implement an engine class, a tier or a wall says so by name
  (:meth:`Backend.refuses`); the producer never falls back to ``"jax"`` silently.

The public package ships one backend, ``"jax"`` (the kernels in :mod:`.core`), which is the default. Another is
installed as a package exposing the entry point group ``dmipy_sim.backends``, or passed as an object. A user who
has no other backend never sees the option do anything, and a name that is not installed is refused by name. How a backend takes a
:class:`WalkRequest` (in process, or serialised for a separate executable) is its own business; the package carries no
serialiser of a request.
"""
from __future__ import annotations

import dmipy_replay.replay.compression as _r_compression
import hashlib
import math
from typing import NamedTuple, Optional

import numpy as np

JAX = "jax"
ENTRY_POINT_GROUP = "dmipy_sim.backends"

#: A capability a backend declares in :attr:`Backend.capabilities`: it walks a permeability per object or over the
#: walk's time (the geometry's ``permeability_table``), reading the row of the step and the column of the object met.
PERMEABILITY_TABLE = "permeability_table"

#: A capability a backend declares in :attr:`Backend.capabilities`: it accumulates a walk of several storage windows
#: (:attr:`WalkRequest.n_windows`), each window's coefficient form its own -- the window's exact start and endpoint, its
#: bands on the window's own basis and the local time's form starting at 0 at the window's start (RPK.md 4.3).
STORAGE_WINDOWS = "storage_windows"

#: A capability a backend declares in :attr:`Backend.capabilities`: it walks MT binding at a walled geometry's walls
#: (:attr:`WalkRequest.kappa_MT_over_D` > 0, the ``"mt"`` kind of :func:`~dmipy_sim.engine.core._walk_geometry`): cross
#: first, then bind on the encounter's local time, the walker held through its dwell, its channels those of the position
#: it is seen at and that position's pool label, its bind / unbind events the batch's ``bound_events``. Its
#: :meth:`Backend.walk_batch` takes each walker's dwell at the start (``dwell0=``).
MT_BINDING = "mt_binding"

#: A capability a backend declares in :attr:`Backend.capabilities`: it walks the packed myelinated class
#: (:class:`~dmipy_sim.geometry.PackedMyelinatedCylinders`, a periodic cell of myelinated and plain cylinders), the cell
#: read from the request's geometry through its public description
#: (:meth:`~dmipy_sim.geometry.PackedMyelinatedCylinders.cell_arrays`, a
#: :class:`~dmipy_sim.geometry.myelin.MyelinatedCell`). The isolated
#: :class:`~dmipy_sim.geometry.MyelinatedCylinder` is not this class and is walked by the JAX kernels alone.
PACKED_MYELINATED = "packed_myelinated"

#: A capability a backend declares in :attr:`Backend.capabilities`: it reads a closed-form field basis in the walk (a
#: basis that is not a raster, such as :class:`~dmipy_sim.fields.hollow_cylinder.HollowCylinderField`, whose data a
#: backend reads through :meth:`~dmipy_sim.fields.hollow_cylinder.HollowCylinderField.device_arrays`), handed as
#: :attr:`WalkRequest.field`, ``K_c3`` cosine modes per channel every ``field_read_interval`` seconds, returned as the
#: batch's ``field_bands``, ``field_qv``, ``field_qv_increments`` and ``dt_f`` as for a raster. A backend that declares it
#: and is handed a basis class it does not implement refuses by name (:meth:`Backend.refuses`).
FIELD_IN_WALK = "field_in_walk"

#: A capability a backend declares in :attr:`Backend.capabilities`: it records the walk's membrane trials at its plain
#: walls (``record_trials=``, :mod:`dmipy_sim.trial_record`), handed the spec and what the producer resolved from it
#: (:attr:`WalkRequest.trial_record`, ``trial_steps``, ``trial_radii``, ``trial_u_max``, ``trial_slots``, with ``count_walls`` on) and
#: returning the batch's record per walker (:attr:`CoefficientBatch.trial_record`), which the producer joins into
#: :attr:`~dmipy_sim.coefficient_walk.CoefficientWalk.trial_record` as it joins the JAX kernels' record.
TRIAL_RECORD = "trial_record"

#: A capability a backend declares in :attr:`Backend.capabilities`: it walks each walker at its own fraction of the
#: geometry's stated permeability (``permeability_factor=``), handed the batch's own factors as
#: :attr:`WalkRequest.permeability_factor` and forming each trial's threshold by the geometry's
#: :meth:`~dmipy_sim.geometry.base.Geometry.permeability_factor_rule`.
PERMEABILITY_FACTOR = "permeability_factor"

#: What the JAX kernels declare: they walk a permeability table, several storage windows, MT binding, the packed
#: myelinated class, a closed-form field read in the walk, the record of the membrane trials and a permeability per
#: walker.
JAX_CAPABILITIES = frozenset({PERMEABILITY_TABLE, STORAGE_WINDOWS, MT_BINDING, PACKED_MYELINATED, FIELD_IN_WALK,
                              TRIAL_RECORD, PERMEABILITY_FACTOR})


class FieldSampling(NamedTuple):
    """In-walk sampling of a strand field (the paper's C3): what a backend evaluates along each walker's path."""
    basis: object             #: the :class:`~dmipy_sim.fields.strand_field.StrandFieldBasis` (its tables, switch, far grid, gate)
    sample_every: int         #: a read at every that-many-th step (read 0 the start's)
    reuse_intervals: int      #: the per-walker segment list is gathered every this many read intervals
    radius_m: float           #: the list's gather radius: the basis's reach plus the margin the walkers travel in between
    list_k: int               #: the list's width the producer sized from the start positions (a backend may widen)


class WalkRequest(NamedTuple):
    """What the producer has resolved before the walk; what a backend walks. A walk is ``n_blocks`` blocks of
    ``block_h`` seconds (``T = n_blocks block_h``), each of ``sub_steps`` physics steps, accumulated per channel as
    ``K`` bands through a block table of order ``m`` (dmipy-sim#740); there is no save grid in it."""
    geometry: object          #: the substrate, resolved (its constants are the physics)
    block_h: float            #: the block length ``h`` (s)
    n_blocks: int             #: blocks over the walk; ``T = n_blocks * block_h``
    sub_steps: int            #: physics steps per block
    dt_sim: float             #: the step's duration; the step length is ``sqrt(6 D dt_sim)`` times the law's atom (:mod:`dmipy_sim.math.step_law`)
    diffusivity: float        #: m^2/s
    record: bool              #: record the boundary local time and the compartment beside the positions
    kappa_over_D: float       #: the membrane's permeability over D (the largest over the objects and the walk's time); 0
                              #: for an impermeable walk. A permeability per object or over time is walked by a backend
                              #: that declares :data:`PERMEABILITY_TABLE`
    count_walls: bool         #: count hits and crossings (exhaustion is always reported)
    bounce_budget: Optional[int]   #: the budget the geometry's loop runs at for this walk
    seed: int                 #: the producer's master seed (the per-walker keys derive from it)
    K: int = 0                #: bands per axis and channel the walk accumulates; 0 is a walk of saves (the adaptive
                              #: producer's, until dmipy-sim#742 accumulates it)
    m: int = 3                #: the order of the block table (:func:`~dmipy_sim.replay.compression.block_table`)
    K_c3: int = 0             #: cosine modes per field channel accumulated in the walk; 0 accumulates no field
    field_read_interval: float = 0.0  #: seconds between reads of the field (``dt_f``, dmipy-sim#742); 0.0 reads none
    events: int = 0           #: the capacity of each walker's C1 event list, the pool changes a walk may record per
                              #: walker (:func:`event_capacity`); a walk whose walker changes pool more often, or at
                              #: all when the capacity is 0, is refused by name
    raw_subset: int = 0       #: the first that-many walkers of the batch whose block ends come back as saves
                              #: (:attr:`CoefficientBatch.saves`): the only positions a backend returns
    stepping: object = None   #: an adaptive walk's :class:`~dmipy_sim.engine.adaptive.AdaptivePlan`; None for the fused walk
    field: object = None      #: a :class:`FieldSampling` when the adaptive walk samples a strand field, the
                              #: :class:`~dmipy_sim.fields.susceptibility_field.FieldGrid` itself when the fixed-step walk
                              #: reads a raster, the closed-form basis itself (a backend that declares
                              #: :data:`FIELD_IN_WALK`) when it reads one (``K_c3`` modes every ``field_read_interval``);
                              #: None otherwise
    drop_pool_changes: bool = False   #: an impermeable walk returns its walkers that changed pool as
                                      #: :attr:`CoefficientBatch.pool_changed` instead of being refused
    n_windows: int = 1        #: the storage windows the walk accumulates separately: ``n_blocks`` is ``n_windows`` times
                              #: the blocks of a window, and each window's coefficient form is its own (RPK.md 4.3). A
                              #: walk of more than one is walked by a backend that declares :data:`STORAGE_WINDOWS`
    kappa_MT_over_D: float = 0.0  #: the MT binding reactivity over D (1/m): an encounter of local time ``l`` binds with
                                  #: ``1 - exp(-kappa_MT_over_D l)`` (:func:`~dmipy_sim.geometry._boundary.bind_probability`);
                                  #: 0 for a walk that binds nothing. A walk that binds is walked by a backend that
                                  #: declares :data:`MT_BINDING`
    dwell_time: float = 0.0   #: the bound pool's mean dwell (s) of a walk that binds (the geminate rebinds scale it per
                              #: bind, :func:`~dmipy_sim.geometry._boundary.dwell_scale`); 0 for a walk that binds nothing
    trial_record: object = None   #: the :class:`~dmipy_sim.trial_record.TrialRecordSpec` of a walk that records its
                                  #: membrane trials (a backend that declares :data:`TRIAL_RECORD`); None otherwise
    trial_steps: object = None    #: ``(n_bands + 1,)`` int64 the step-end index of every band edge
                                  #: (:func:`~dmipy_sim.trial_record.edge_steps`): band ``k`` holds the steps of index
                                  #: ``j`` with ``trial_steps[k] < j + 1 <= trial_steps[k + 1]``; None without a record
    trial_radii: object = None    #: ``(n_objects,)`` float64 the radius (m) of each object, in the geometry's object
                                  #: order; None without a record
    trial_u_max: object = None    #: ``(n_objects,)`` float64 each object's bound on a trial's overshoot
                                  #: (:func:`~dmipy_sim.trial_record.u_max_of`); None without a record
    trial_slots: object = None    #: the slots each walker's record holds (:func:`~dmipy_sim.trial_record.device_record`):
                                  #: the spec's, or the pilot's starting slots ``min(PILOT_SLOTS, pairs)`` (in the request the
                                  #: backend's refusal check receives and in the pilot's batches), then the capacity the
                                  #: pilot states; None without a record
    permeability_factor: object = None    #: ``(n,)`` float32 each walker's factor of the geometry's stated
                                          #: permeability for THIS batch, in ``r0``'s row order; None walks every
                                          #: walker at the stated permeability. A backend that declares
                                          #: :data:`PERMEABILITY_FACTOR` forms each trial's threshold by the
                                          #: geometry's :meth:`~dmipy_sim.geometry.base.Geometry.permeability_factor_rule`

    @property
    def T(self):
        """The walk's duration (s)."""
        return float(self.block_h) * int(self.n_blocks)

    @property
    def blocks_per_window(self):
        """The blocks of one storage window, ``n_blocks / n_windows``."""
        return int(self.n_blocks) // int(self.n_windows)

    @property
    def T_window(self):
        """One storage window's duration (s), the length of the basis its bands are on."""
        return float(self.block_h) * self.blocks_per_window


def capability_refusal(backend, name, request, *, permeability_table=False):
    """Why ``backend`` (named ``name``) does not walk ``request`` for want of a capability it does not declare in its
    ``capabilities``, by the capability's name, or None: a permeability per object or over time
    (``permeability_table``) needs :data:`PERMEABILITY_TABLE`, a walk of several storage windows
    :data:`STORAGE_WINDOWS`, a walk that binds (``request.kappa_MT_over_D > 0``) :data:`MT_BINDING`, a request
    carrying a permeability per walker (``request.permeability_factor``) :data:`PERMEABILITY_FACTOR`."""
    have = getattr(backend, "capabilities", ())
    if request.permeability_factor is not None and PERMEABILITY_FACTOR not in have:
        return (f"backend {name!r}: this walk walks each walker at its own permeability (permeability_factor), which the "
                f"backend does not declare it walks (capability {PERMEABILITY_FACTOR!r})")
    if float(request.kappa_MT_over_D) > 0.0 and MT_BINDING not in have:
        return (f"backend {name!r}: this walk binds at the walls (MT, kappa_MT/D = {float(request.kappa_MT_over_D):.4g} 1/m), "
                f"which the backend does not declare it walks (capability {MT_BINDING!r})")
    if permeability_table and PERMEABILITY_TABLE not in have:
        return (f"backend {name!r}: this walk uses a permeability per object or over time, which the backend does not "
                f"declare it walks (capability {PERMEABILITY_TABLE!r})")
    if int(request.n_windows) > 1 and STORAGE_WINDOWS not in have:
        return (f"backend {name!r}: this walk accumulates {int(request.n_windows)} storage windows, which the backend does "
                f"not declare it walks (capability {STORAGE_WINDOWS!r})")
    return None


def event_capacity(n_steps, p_max, *, sigma=6.0, floor=16):
    """The C1 event list's capacity per walker for a walk of ``n_steps`` physics steps on which no wall hit changes a
    walker's pool with a probability above ``p_max`` (:func:`~dmipy_sim.engine.physics.crossing_p_max` of the walk's
    step): the changes per walker are bounded by Binomial(``n_steps``, ``p_max``), and the capacity is its mean plus
    ``sigma`` standard deviations, at least ``floor``, and never more than ``n_steps``. A backend refuses by name a
    walker that exceeds it (the count returned), so the bound costs memory only: 9 bytes per entry on a GPU
    backend."""
    p = float(p_max)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p_max is a probability in [0, 1]; got {p_max!r}")
    mu = p * int(n_steps)
    cap = math.ceil(mu + float(sigma) * math.sqrt(mu * (1.0 - p)))
    return int(max(int(floor), min(int(n_steps), cap)))


class ParityCase(NamedTuple):
    """One substrate :func:`parity_cases` compares a backend on. ``case()`` builds the geometry -- the same call
    a bare builder answered, so every existing caller of ``parity_cases()[name]()`` is unaffected. A rate case
    is a case with nothing past ``build``; a hard-invariant case (``strands_void``) also carries ``starts``, the
    walk's own seeding (:func:`parity_report` draws none of its own when this is set), and the walk's own scale
    (``n_walkers``, ``T_max``, ``block_h``, ``seed`` -- each ``None`` falls back to :func:`parity_report`'s own
    keyword, except ``starts`` callers, which always get the case's own values)."""
    build: object             #: zero-arg callable returning the geometry
    starts: object = None     #: ``(n, seed) -> r0 (n, 3) float32``, or None: :func:`parity_report` draws its own
    n_walkers: object = None  #: overrides :func:`parity_report`'s ``n_walkers`` when ``starts`` is set
    T_max: object = None      #: overrides :func:`parity_report`'s ``T_max`` when ``starts`` is set
    block_h: object = None    #: overrides :func:`parity_report`'s ``block_h`` when ``starts`` is set
    seed: object = None       #: overrides :func:`parity_report`'s ``seed`` when ``starts`` is set

    def __call__(self):
        return self.build()


class WalkBatch(NamedTuple):
    """A batch of walkers as its saves -- what the adaptive producer (:mod:`.adaptive`) takes back from a backend
    until it accumulates (dmipy-sim#742): the JAX kernels' per-save outputs, as arrays. The coefficient producer
    (:func:`~dmipy_sim.engine.core._walk_geometry`) refuses a backend that returns one."""
    positions: np.ndarray             #: (n, n_t, 3) float32, save 0 the start
    boundary_local_time: object       #: (n, n_t) float32 at rho/D = 1, or None when not recorded
    compartment: object               #: (n, n_t) float32 fractional occupancy of pool 1 (int labels when impermeable), or None
    illegal: np.ndarray               #: (n,) int32 refused walker-steps per walker
    work: tuple                       #: (hits | None, crossings | None, exhausted) per walker, int32 arrays
    counters: object = None           #: what else the backend counted over the batch (an adaptive walk's
                                      #: ``dict(n_free=, n_kernel_steps=)``), or None when it reports none
    field_samples: object = None      #: (n, n_tf, n_ch) float32 the bare interval means of the field's channels at the
                                      #: sampled saves (sample 0 the start's reading; no mean subtracted), or None


class CoefficientBatch(NamedTuple):
    """A batch of walkers in its coefficient form (dmipy-sim#740, #747): what a backend returns for the coefficient
    walk, under the names of :class:`~dmipy_sim.coefficient_walk.CoefficientWalk` -- the producer concatenates the
    batches into one walk and converts nothing. The bands are the walk's own accumulation finished by the backend
    (the cosine projections of each channel's increments against :func:`~dmipy_sim.replay.compression.block_table`,
    turned into the bridge's bands on the continuous sine basis, :func:`~dmipy_sim.replay.compression.cosine_band_scale`),
    so a pack built from the walk is bit for bit the pack the backend's own writer builds from the same walk.

    Every array of the form carries the request's storage windows on its second axis (``n_windows``, 1 for a walk of
    one): window ``w`` is the walk over ``[w T_w, (w + 1) T_w]`` (``T_w`` = :attr:`WalkRequest.T_window`), its start and
    endpoint exact, its bands on the window's own basis, its local time starting at 0 at the window's start, its
    quadratic variations the window's own steps' (RPK.md 4.3). The walker's start pool, its events (on the walk's own
    clock), its counters and its saves are the walk's."""
    position_coeffs: np.ndarray       #: (n, n_windows, K + 2, 3) floats in the batch's ``precision``: per axis x(start),
                                      #: x(end) - x(start) (exact) and the K bands; the walk stores them float32
    boundary_start: object            #: (n, n_windows) the local time at each window's start (0), or None
    boundary_endpoint: object         #: (n, n_windows) at its end, exact, or None
    boundary_bands: object            #: (n, n_windows, K) its bridge's bands at rho/D = 1, or None
    compartment: object               #: (n,) int8 the walker's pool at the start (1 inside; a label volume's label), or None
    illegal: np.ndarray               #: (n,) int32 refused walker-steps per walker
    work: tuple                       #: (hits | None, crossings | None, exhausted) per walker, int32 arrays
    K: int                            #: bands per channel and window
    m: int                            #: the order of the block table
    sub_steps: int                    #: physics steps per block (h = sub_steps * dt_sim)
    precision: str                    #: the floating-point type the bands were accumulated in across blocks
    saves: object = None              #: the block ends of the batch's first ``request.raw_subset`` walkers as a
                                      #: :class:`WalkBatch` (save 0 the start), or None when none were asked for
    qv_positions: object = None       #: (n, n_windows, 3) each walker's sum dx^2 per axis over each window's steps
    qv_boundary: object = None        #: (n, n_windows) its sum dlog^2 (each step's contact squared), or None
                                      #: (the contact and the quadratic variations in the batch's ``precision`` too, stored
                                      #: float32 by the walk)
    events: object = None             #: C1's event list ``(offsets (n + 1,) int64, times (n_ev,) float64 s, labels
                                      #: (n_ev,) int8)``: walker w's pool changes are entries offsets[w] .. offsets[w+1],
                                      #: each the start of the step that ended in the pool entered; None unrecorded
    bound_events: object = None       #: an MT walk's bind / unbind events in the same form, or None
    field_bands: object = None        #: (n, n_windows, n_ch, K_c3) float64 the field's channels (less the domain mean)
                                      #: on the cosine basis of each window, or None
    field_qv: object = None           #: (n, n_windows, n_ch) float64 each channel's sum (b dt_f)^2 over its reads, or None
    field_qv_increments: object = None  #: (n, n_windows, n_ch) float64 each channel's read series' quadratic variation,
                                        #: the sum of (b_{i+1} - b_i)^2 over its reads (RPK.md 9.5.5), or None
    dt_f: object = None               #: the field's read interval (s); 0.0 a read after every step; None no field
    k0_residual: object = None        #: (n, n_windows, 3) float64 ``c_0 - (x(end) - x(start))``, the accumulation's
                                      #: measured residual on the k = 0 mode, when the backend reports it; None otherwise
    ring: object = None               #: blocks per projection window, when the backend projects in windows
    accumulation: object = None       #: how the backend summed the blocks (the header's ``block.accumulation``)
    projection_precision: object = None  #: the dtype name the backend forms its projection product in, or None when it states none
    pool_changed: object = None       #: (n,) bool the walkers of an impermeable walk whose label changed, when the request
                                      #: asked for them (``drop_pool_changes``); None otherwise
    n_windows: int = 1                #: the storage windows on the form's second axis: the request's
    trial_record: object = None       #: the record of the batch's membrane trials when the request asks for it
                                      #: (:attr:`WalkRequest.trial_record`), in the request's ``trial_slots`` slots
                                      #: per walker as :func:`~dmipy_sim.trial_record.device_record` holds them:
                                      #: ``(key, n, N, S, count, exceed)``, ``key`` ``(n_b, slots)`` each slot's pair
                                      #: ``object * n_bands + band`` in the order the walker first met it (-1 a free
                                      #: slot), ``n`` and ``N`` ``(n_b, slots)`` its trials and granted trials, ``S``
                                      #: ``(n_b, slots, modes)`` its Chebyshev sums, floating point in the backend's
                                      #: accumulation precision (the batch's ``precision``), ``count`` ``(n_b,)`` the
                                      #: distinct pairs the walker met (counting on past the slots) and ``exceed``
                                      #: ``(n_b,)`` the trials whose overshoot exceeded the object's ``trial_u_max`` --
                                      #: each count an integer array with values in ``[0, 2^31)``, held as int32 --
                                      #: joined into the walk's record as returned; None otherwise


class Backend:
    """The protocol. A backend subclasses or duck-types this: ``name``, :meth:`refuses`, :meth:`walk_batch`, and
    optionally :attr:`capabilities`."""
    name = "?"
    #: What this backend states it walks beyond the common path (a frozenset of the module's capability names, e.g.
    #: :data:`PERMEABILITY_TABLE`). A walk that needs a capability the backend does not state is refused by name; a
    #: backend that defines no ``capabilities`` states none.
    capabilities = frozenset()

    def refuses(self, request: WalkRequest) -> Optional[str]:
        """Why this backend cannot walk ``request`` (an engine class, a tier, a wall it does not implement), or
        None. The producer raises with the reason; it never falls back."""
        return None

    def walk_batch(self, request: WalkRequest, r0: np.ndarray, keys: np.ndarray, *, out=None, dwell0=None) -> CoefficientBatch:
        """Walk the ``(n, 3)`` float32 starts ``r0`` from their ``(n, 2)`` uint32 keys in the coefficient form
        ``request`` names (``K`` bands at order ``m`` per storage window, ``request.n_windows`` windows over
        ``n_blocks`` blocks of ``sub_steps`` steps): a :class:`CoefficientBatch`, with the block ends of the first
        ``request.raw_subset`` walkers as its ``saves``.
        A request with ``K = 0`` is the adaptive producer's walk of saves (a :class:`WalkBatch`; ``out``, when given,
        is a dict of that producer's own contiguous float32 arrays -- ``positions`` (n, n_t, 3),
        ``boundary_local_time`` and ``compartment`` (n, n_t), ``field_samples`` (n, n_tf, n_ch) -- a backend may write
        into instead of allocating its own). A walk that binds (``request.kappa_MT_over_D > 0``, a backend that declares
        :data:`MT_BINDING`) is handed ``dwell0``, ``(n,)`` float32 each walker's dwell left at the start in physics steps
        (0 free: the producer's equilibration), and returns ``bound_events``; every other walk is handed none."""
        raise NotImplementedError


def installed() -> dict:
    """The backends installed beside the JAX one, ``{name: entry point}``."""
    from importlib.metadata import entry_points
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:                                     # Python < 3.10 API
        eps = entry_points().get(ENTRY_POINT_GROUP, [])
    return {ep.name: ep for ep in eps}


def resolve(backend) -> Optional[Backend]:
    """``None`` for the JAX kernels (the default), else the backend object: by name through the entry point, or
    the object itself when one is passed. An unknown name is refused, naming what is installed."""
    if backend is None or backend == JAX:
        return None
    if isinstance(backend, str):
        eps = installed()
        if backend not in eps:
            raise ValueError(f"no backend {backend!r} is installed; installed: {[JAX] + sorted(eps)}")
        obj = eps[backend].load()
        obj = obj() if isinstance(obj, type) else obj
    else:
        obj = backend
    for attr in ("name", "refuses", "walk_batch"):
        if not hasattr(obj, attr):
            raise TypeError(f"a backend has `name`, `refuses(request)` and `walk_batch(request, r0, keys)`; "
                            f"{type(obj).__name__} lacks {attr!r}")
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Parity: how a backend is admitted. The same cases a backend's own tests run.
# ─────────────────────────────────────────────────────────────────────────────

def parity_cases():
    """The substrates a backend is compared on, each at its own permeability: ``{name: ParityCase}``. Every entry
    is a :class:`ParityCase` -- one spelling, whether the case is a plain rate comparison (``build`` alone) or
    carries its own starts and scale (``strands_void``) -- and ``case()`` builds the geometry exactly as a bare
    builder used to, so ``parity_cases()[name]()`` is unchanged for every existing caller."""
    import dmipy_sim as d
    R = 2e-6

    def packed(cls, **kw):
        c = np.array([[0.0, 0.0], [2.2 * R, 0.0], [0.0, 2.2 * R], [2.2 * R, 2.2 * R]])
        if cls is d.PackedSpheres:
            c = np.concatenate([c, np.zeros((4, 1))], axis=1)
        return cls([R, 0.8 * R, R, 0.9 * R], c, 4.4 * R, **kw)

    def label(**kw):
        from ..geometry.label_volume import LabelVolume
        lab = np.ones((30, 8, 8), np.uint8); lab[8:22] = 0
        return LabelVolume(lab, 0.5e-6, **kw)

    def mesh(**kw):
        from ..geometry import mesh_shapes
        V, F = mesh_shapes.icosphere(R, subdivisions=2)
        return d.Mesh(V, F, feature_radius=R, **kw)

    def strands(interior):
        # three curved tubes of 1 / 1.5 / 2 um in a reflecting 20 um box, the family's test fixture; impermeable
        cls = [np.array([[x, 0, -30e-6], [x, 0.5e-6, 0], [x, 0, 30e-6]]) for x in (-6e-6, 0, 6e-6)]
        return d.PackedCurvedCylinders(cls, [1.0e-6, 1.5e-6, 2.0e-6], interior=interior,
                                       box=(np.full(3, -10e-6), np.full(3, 10e-6)))

    VOID_R, VOID_L = 2e-6, 200e-6
    VOID_XT = VOID_YT = 100e-6

    def void_build():
        # one tube in a 200 um box: starts (below) sit in the void 2.5 cells from its wall, where a gather whose
        # 27-cell neighbourhood is empty must still bound the walk by the nearest wall it CAN reach, not by the
        # box.
        return d.PackedCurvedCylinders([np.array([[VOID_XT, VOID_YT, 0.0], [VOID_XT, VOID_YT, VOID_L]])], [VOID_R],
                                       interior=False, box=(np.zeros(3), np.full(3, VOID_L)))

    void_cell_size = void_build().cell_size   # the geometry's own cell-size rule, read once rather than re-derived

    def void_starts(n, seed):
        rng = np.random.default_rng(seed)
        return np.column_stack([np.full(n, VOID_XT - VOID_R - 2.5 * void_cell_size),
                                rng.uniform(VOID_YT - 5e-6, VOID_YT + 5e-6, n),
                                rng.uniform(20e-6, 180e-6, n)]).astype(np.float32)

    # kappa R / D = 2 at D = 2e-9: a membrane crossed at a quarter to a half of the hits, where the transmission's form
    # (x / (1 + x) against its linearisation) moves the crossing count by a tenth
    FAST = 2e-3
    return {
        "sphere": ParityCase(lambda: d.Sphere(R)),
        "sphere_perm": ParityCase(lambda: d.Sphere(R, permeability=1e-5)),
        "sphere_perm_fast": ParityCase(lambda: d.Sphere(R, permeability=FAST)),
        "cylinder": ParityCase(lambda: d.Cylinder(R, (0, 0, 1))),
        "cylinder_perm": ParityCase(lambda: d.Cylinder(R, (0, 0, 1), permeability=1e-5)),
        "ellipsoid": ParityCase(lambda: d.Ellipsoid((R, 0.7 * R, 0.5 * R))),
        "packed_cyl": ParityCase(lambda: packed(d.PackedCylinders)),
        "packed_cyl_perm": ParityCase(lambda: packed(d.PackedCylinders, permeability=2e-5)),
        "packed_cyl_perm_fast": ParityCase(lambda: packed(d.PackedCylinders, permeability=FAST)),
        "packed_sph": ParityCase(lambda: packed(d.PackedSpheres)),
        "packed_sph_perm": ParityCase(lambda: packed(d.PackedSpheres, permeability=2e-5)),
        "label_slab": ParityCase(lambda: label()),
        "label_slab_perm": ParityCase(lambda: label(permeability=1e-5, pools={0: "free", 1: "grain"})),
        "label_slab_perm_fast": ParityCase(lambda: label(permeability=FAST, pools={0: "free", 1: "grain"})),
        "mesh": ParityCase(lambda: mesh()),
        "mesh_perm": ParityCase(lambda: mesh(permeability=1e-5)),
        "mesh_perm_fast": ParityCase(lambda: mesh(permeability=FAST)),
        "strands_intra": ParityCase(lambda: strands(True)),
        "strands_extra": ParityCase(lambda: strands(False)),
        # a rate comparison at parity_report's default scale would barely move these walkers, so the case pins
        # its own walk scale (and its own seed, #675 review: never parity_report's, which would move the case
        # with it) rather than overriding only starts
        "strands_void": ParityCase(void_build, starts=void_starts, n_walkers=4096, T_max=40e-3, block_h=0.5e-3, seed=5),
    }


def void_invariant(g, w, *, wall_margin_m=0.5e-6, near_wall_fraction=0.02):
    """The void case's invariant (dmrai-lab/dmipy-sim#675): every block end of every walker of ``w``'s raw subset lies
    outside every tube of ``g`` -- tested by the geometry's own exact classifier
    (:meth:`~dmipy_sim.geometry.base.Geometry.classify_positions_exact` / :meth:`~dmipy_sim.geometry.base.Geometry.pool_of`),
    never a hand-rolled radial formula -- and more than ``near_wall_fraction`` of the walkers come within
    ``wall_margin_m`` of a wall at some block end (the geometry's own
    :meth:`~dmipy_sim.geometry.curved_cylinder.PackedCurvedCylinders.wall_scales`; the premise -- enough walkers
    reach the wall for the containment to mean something). Returns ``{check: {..., ok}}``; a failure of either is a
    hard violation of the backend under test. ``illegal`` is the count of refused steps (a step the engine held
    rather than let end in another pool) and the walker-steps they are of, with no verdict here: the reference's
    own kernels refuse such a step now and then too (a float32 tie at a wall), so :func:`parity_report` judges the
    count against the reference's measured rate."""
    n, n_t, _ = w.raw_positions.shape
    flat = np.asarray(w.raw_positions, np.float64).reshape(-1, 3)
    pool = np.asarray(g.pool_of(g.classify_positions_exact(flat))).reshape(n, n_t)
    n_inside = int((pool != 0).sum())
    illegal = int(np.asarray(w.illegal_crossings).sum())
    d_wall, _ = g.wall_scales(flat)
    n_near = int((np.asarray(d_wall).reshape(n, n_t).min(axis=1) < wall_margin_m).sum())
    return dict(
        contained=dict(n_inside=n_inside, ok=n_inside == 0),
        illegal=dict(n_illegal=illegal, walker_steps=int(w.work["walker_steps"])),
        reaches_wall=dict(n_near=n_near, fraction=n_near / n, ok=n_near > near_wall_fraction * n),
    )


def _mean_se(x, axis=0):
    """``(mean, standard error of the mean)`` of ``x`` along ``axis``."""
    x = np.asarray(x, np.float64)
    n = x.shape[axis]
    return x.mean(axis=axis), x.std(axis=axis, ddof=1) / np.sqrt(n)


def _z(a, ea, b, eb):
    """``|a - b|`` in units of the combined standard error (0 where both errors vanish and the values agree)."""
    den = np.sqrt(np.asarray(ea) ** 2 + np.asarray(eb) ** 2)
    diff = np.abs(np.asarray(a) - np.asarray(b))
    return np.where(den > 0.0, diff / np.where(den > 0.0, den, 1.0), np.where(diff > 0.0, np.inf, 0.0))


def coefficient_signal(walk, G, dt_wf):
    """The gradient signal of a coefficient walk's first window under the waveforms ``G`` ``(n_meas, n_wf, 3)`` on
    their own grid ``dt_wf``, read from the bands alone (the replay's own reading of a pack of the walk,
    :func:`~dmipy_sim.replay.compression.band_projection`): ``(signal (n_meas,), standard error (n_meas,))`` of the
    walkers' ``cos(gamma sum C W)`` at float32 coefficients, what the pack stores."""
    from dmipy_analytic.constants import GAMMA
    from ..replay import compression as _cx
    C = np.asarray(walk.position_coeffs, np.float32).astype(np.float64)[:, 0]              # (n, K + 2, 3)
    W = _r_compression.band_projection(np.asarray(G, np.float64), float(dt_wf), float(walk.T_window), int(C.shape[1]) - 2)
    phi = GAMMA * np.einsum("wkc,mkc->wm", C, W)
    return _mean_se(np.cos(phi))


#: the counts of a walk's work record a parity report compares as rates of rare events (a refused step, an exhausted
#: bounce budget): binomial against the reference's pooled count, since a few seeds' counts of 0-5 give no spread
RARE_COUNTS = ("exhausted_steps", "illegal")
#: the counts it compares against the reference's measured stream-to-stream spread (Student t, the sweep's dof)
SPREAD_COUNTS = ("n_hits", "n_crossings")


def _rate_p(k_b, n_b, k_a, n_a):
    """Two-sided p of ``k_b`` events in ``n_b`` trials against the reference's ``k_a`` in ``n_a`` at one rate: the
    conditional binomial ``k_b | k_a + k_b ~ Bin(k_a + k_b, n_b / (n_a + n_b))`` (1 when neither side has an event)."""
    from scipy.stats import binomtest
    k = int(k_a) + int(k_b)
    if k == 0:
        return 1.0
    return float(binomtest(int(k_b), k, float(n_b) / float(n_a + n_b)).pvalue)


def _spread_p(b, sweep):
    """``(t, p, sd)``: the backend's count ``b`` as a new draw of the reference's, from the sweep's counts at the same
    starts: ``t = (b - mean) / (sd sqrt(1 + 1/m))`` on ``m - 1`` degrees of freedom (the prediction interval of one more
    stream), two-sided. A sweep with no spread (an impermeable walk's crossings, all 0) admits its value alone."""
    from scipy.stats import t as student
    x = np.asarray(sweep, np.float64); m = x.size
    sd = float(x.std(ddof=1))
    if sd == 0.0:
        return (0.0, 1.0, 0.0) if float(b) == float(x[0]) else (float("inf"), 0.0, 0.0)
    tv = (float(b) - float(x.mean())) / (sd * np.sqrt(1.0 + 1.0 / m))
    return float(tv), float(2.0 * student.sf(abs(tv), m - 1)), sd


def case_seeds(seed, name, streams):
    """The walk seeds of one rate case of a report at ``seed``: ``streams`` integers in ``[0, 2^31)``, element ``i``
    the first four bytes (little-endian) of ``sha256(f"{seed}/{name}/{i}")`` with the sign bit cleared. A function of
    the report's seed, the case's name and the stream index alone, so two cases of one report walk different streams
    (two names collide on a seed with probability about ``(C streams)^2 / 2^32``; :func:`parity_report` refuses such
    a collision by name). Stream 0 is the case's walk -- the reference's starts, its stream-0 walk and the backend's
    walk -- and streams ``1 .. streams - 1`` are the sweep."""
    if int(streams) < 1:
        raise ValueError(f"case_seeds needs at least one stream; got streams={streams!r}")
    return tuple(int.from_bytes(hashlib.sha256(f"{int(seed)}/{name}/{i}".encode()).digest()[:4], "little") & 0x7FFFFFFF
                 for i in range(int(streams)))


def _normal_scores(t, dof):
    """The normal scores ``norm.ppf(student.cdf(t, dof))`` of Student ``t`` values (elementwise, broadcasting): a
    ``t`` on ``dof`` degrees of freedom maps to a standard normal ``z``."""
    from scipy.stats import norm, t as student
    return norm.ppf(student.cdf(np.asarray(t, np.float64), np.asarray(dof, np.float64)))


def pooled_bias(hits, false_failure):
    """The pooled bias of a report's hit counts over its cases, ``hits`` ``[(case, t, dof, relative, walk_seed,
    engine_class)]``: each ``t`` (:func:`_spread_p`) turned into a normal score through its own Student CDF
    (:func:`_normal_scores`), ``Z = sum z / sqrt(C)`` (Stouffer), ``ok`` when ``|Z|`` is within the two-sided normal
    threshold of ``false_failure`` (one test); the sign test of the same differences and the mean relative bias
    ``(backend - mean) / mean`` with its standard error beside it.

    **The null distribution.** Under a correct backend ``Z ~ N(0, 1)`` exactly when each pooled ``t`` is Student
    ``t`` on its ``dof`` (``sweep - 1``: the backend's count and the sweep's are independent draws of one law) and
    every ``t`` is a function of its own case's walks only, so the ``z`` are independent. The second premise is
    checked here as far as the record can show it: two entries that name the same ``walk_seed`` are refused by name
    (``ValueError`` naming both cases), never pooled.

    ``per_class`` is ``{engine_class: {"Z": sum z / sqrt(C_k), "cases": C_k}}`` over the entries of each engine class
    (the geometry's class name): reported, never judged -- only the overall ``Z`` decides ``ok``, so the report's
    false-failure rate is one test's."""
    from scipy.stats import norm, binomtest
    seen = {}
    for case, _, _, _, walk_seed, _ in hits:
        if int(walk_seed) in seen:
            raise ValueError(f"pooled_bias: cases {seen[int(walk_seed)]!r} and {case!r} name the same walk seed "
                             f"{int(walk_seed)}; their hit-count t values are not independent and are not pooled")
        seen[int(walk_seed)] = case
    zs = _normal_scores([h[1] for h in hits], [h[2] for h in hits])
    Z = float(zs.sum() / np.sqrt(zs.size))
    z_crit = float(norm.isf(float(false_failure) / 2.0))
    below = int(sum(1 for h in hits if h[1] < 0))
    rel = np.array([h[3] for h in hits], np.float64)
    per_class = {}
    for h, z in zip(hits, zs):
        per_class.setdefault(h[5], []).append(float(z))
    return dict(Z=Z, z_crit=z_crit, cases=len(hits), ok=bool(abs(Z) <= z_crit),
                sign_test=dict(below=below, of=len(hits), p=float(binomtest(below, len(hits), 0.5).pvalue)),
                relative_bias_mean=float(rel.mean()), relative_bias_se=float(rel.std(ddof=1) / np.sqrt(rel.size)),
                per_case={h[0]: dict(t=float(h[1]), relative=float(h[3])) for h in hits},
                per_class={k: dict(Z=float(np.sum(v) / np.sqrt(len(v))), cases=len(v)) for k, v in per_class.items()})


def parity_report(backend, cases=None, *, n_walkers=4000, T_max=2e-3, block_h=0.5e-3, seed=11, sweep=8, false_failure=1e-2):
    """Compare ``backend`` with the JAX kernels on ``cases`` (default :func:`parity_cases`) on the coefficient form:
    the same starts, the same sub-step and blocks, counting on, every walker's block ends kept as the positional
    oracle (``raw_subset``).

    **Every rate case walks its own streams.** A rate case's seeds are :func:`case_seeds` ``(seed, name, sweep + 1)``:
    stream 0 draws the starts and is walked by the JAX kernels and by the backend (from those starts); streams
    ``1 .. sweep`` are the sweep. No two cases of one report share a seed: the seeds of every rate case are collected
    before any walk and a collision of two ``(case, stream)`` pairs is refused by name (``ValueError``). Each rate
    record carries ``seeds = {"walk": s0, "sweep": [s1, ..]}`` and ``starts_sha256`` (the sha256 of the starts'
    float32 bytes). ``sweep`` must be at least 2.

    **The reference's own spread, measured, from streams the backend does not walk.** Per case the JAX kernels walk
    the backend's starts again with the ``sweep`` streams: the counts' stream-to-stream spread is the case's own, never
    a typed overdispersion (a walker near a wall hits it on consecutive steps, so the counts are clustered; measured
    2.4x Poisson on the sphere and 5.5x on the extra-axonal strands at 20k walkers). The hit and crossing counts are
    each a new draw of the reference's (:func:`_spread_p` over the sweep alone: Student t on ``sweep - 1`` degrees of
    freedom, the prediction interval of one more stream; the record's ``jax`` is stream 0's count, which the spread
    does not read); the exhausted and the refused walker-steps are rare events, judged as a rate against the
    reference's pooled count over the sweep (:func:`_rate_p`, binomial). Stream 0 is excluded from both, so the
    backend's count is independent of every stream it is compared with, whatever the backend does with the keys it is
    handed. The bands at the band level -- each
    channel's quadratic variation (``qv_positions`` per axis, ``qv_boundary``) and each band's mean square per axis --
    and the perpendicular PGSE signal replayed from each side's bands (:func:`coefficient_signal`, what a pack of the
    walk stores) against the two walks' combined standard errors of many walkers (normal); every walker on its side
    (the classifier at the last block end agrees with the start's pool unless a crossing was granted), a hard check.

    **One threshold, from the number of comparisons.** Every comparison has a two-sided p, and each fails below
    ``false_failure / M``, ``M`` the comparisons the report makes (Bonferroni): the chance that a correct backend fails
    the whole report is at most ``false_failure``. A check records its statistic, its p and the threshold.

    **The pooled bias.** A per-case test at the report's threshold cannot see a deficit of a percent that every case
    shares, so with two or more rate cases the report ends with a ``pooled`` record: each case's hit-count ``t``
    turned into a normal score (its ``t`` CDF), summed over the cases as ``Z = sum z / sqrt(C)`` (Stouffer), failed
    when ``|Z|`` is beyond the normal threshold of ``false_failure`` (one test), with the sign test of the same
    differences and the mean relative bias ``(backend - mean) / mean`` beside it (:func:`pooled_bias`). Its null
    distribution is ``Z ~ N(0, 1)``: each pooled ``t`` is a function of its own case's walks only (disjoint seeds
    across cases) and is Student ``t`` on ``sweep - 1`` degrees of freedom (a sweep that excludes the stream the
    backend walks). ``pooled.per_class`` states ``Z`` and the case count per engine class (the geometry's class name),
    reported and not judged: only the overall ``Z`` decides ``ok``.

    A case is a rate comparison unless it declares its own starts (:attr:`ParityCase.starts`, e.g.
    ``strands_void``): there the backend under test walks those starts at the case's own scale (its
    ``n_walkers``/``T_max``/``block_h``/``seed``, each falling back to this function's keyword of the same name when
    the case leaves it ``None``) and is held to :func:`void_invariant` on its block ends (containment and the premise:
    hard), its refused steps judged against the JAX kernels' rate on the same case over ``sweep + 1`` seeds. A case
    the backend refuses is recorded with its reason and not failed: refusal is the contract. Returns ``[{case, ok,
    ...}]`` (and the pooled record last); the list is what a backend's tests assert on."""
    from .core import _walk_geometry
    from dmipy_replay.sequences import pgse
    from scipy.stats import norm
    out, pending, hits = [], [], []          # pending: (record, check name, p) judged once M is known
    if int(sweep) < 2:
        raise ValueError(f"parity_report needs sweep >= 2 (the spread of the sweep's counts has sweep - 1 degrees of "
                         f"freedom); got sweep={sweep!r}")
    cases = parity_cases() if cases is None else cases
    # every rate case's seeds, before any walk: no (case, stream) pair may share a seed with another
    seeds_of, owner = {}, {}
    for name, case in cases.items():
        if case.starts is not None:
            continue
        seeds_of[name] = case_seeds(seed, name, int(sweep) + 1)
        for i, sv in enumerate(seeds_of[name]):
            if sv in owner:
                raise ValueError(f"parity_report: ({owner[sv][0]!r}, stream {owner[sv][1]}) and ({name!r}, stream {i}) "
                                 f"share the walk seed {sv}; two cases of one report must not share a walk")
            owner[sv] = (name, i)

    def judge(rec, key, p, **fields):
        rec["checks"][key] = dict(fields, p=float(p))
        pending.append((rec, key))

    for name, case in cases.items():
        g = case(); g.count_walls = True
        if case.starts is not None:
            nw = n_walkers if case.n_walkers is None else case.n_walkers
            Tm = T_max if case.T_max is None else case.T_max
            bh = block_h if case.block_h is None else case.block_h
            sd = seed if case.seed is None else case.seed
            try:
                w = _walk_geometry(nw, 2e-9, g, Tm, bh, seed=sd, tiers="all", walker_batch_size=nw,
                                          r0=case.starts(nw, sd), backend=backend, raw_subset=nw)
            except ValueError as e:
                if "refuses" in str(e) or "does not" in str(e):
                    out.append(dict(case=name, ok=None, refused=str(e))); continue
                raise
            rec = dict(case=name, ok=True, checks=void_invariant(g, w))
            gs = case(); gs.count_walls = True
            k_a = n_a = 0
            for s in range(sd + 1, sd + 2 + int(sweep)):                    # the reference's rate, sweep + 1 seeds
                ws = _walk_geometry(nw, 2e-9, gs, Tm, bh, seed=s, tiers="all", walker_batch_size=nw, r0=case.starts(nw, s))
                k_a += int(np.asarray(ws.illegal_crossings).sum()); n_a += int(ws.work["walker_steps"])
            il = rec["checks"]["illegal"]
            judge(rec, "illegal", _rate_p(il["n_illegal"], il["walker_steps"], k_a, n_a), n_illegal=il["n_illegal"],
                  walker_steps=il["walker_steps"], reference=dict(n_illegal=k_a, walker_steps=n_a, seeds=int(sweep) + 1))
            out.append(rec)
            continue
        s0, sweep_seeds = seeds_of[name][0], seeds_of[name][1:]
        ref = _walk_geometry(n_walkers, 2e-9, g, T_max, block_h, seed=s0, tiers="all", walker_batch_size=n_walkers,
                                    raw_subset=n_walkers)
        g2 = case(); g2.count_walls = True
        x0 = np.asarray(ref.position_coeffs[:, 0, 0, :], np.float32)                       # the exact starts
        try:
            w = _walk_geometry(n_walkers, 2e-9, g2, T_max, block_h, seed=s0, tiers="all", walker_batch_size=n_walkers,
                                      r0=x0, backend=backend, raw_subset=n_walkers)
        except ValueError as e:
            if "refuses" in str(e) or "does not" in str(e):
                out.append(dict(case=name, ok=None, refused=str(e))); continue
            raise
        rec = dict(case=name, ok=True, checks={}, seeds=dict(walk=int(s0), sweep=[int(v) for v in sweep_seeds]),
                   starts_sha256=hashlib.sha256(np.ascontiguousarray(x0, np.float32).tobytes()).hexdigest())
        if (w.K, w.m, w.sub_steps, w.n_blocks) != (ref.K, ref.m, ref.sub_steps, ref.n_blocks):
            raise ValueError(f"the two walks of {name!r} are not one resolution: jax (K, m, sub_steps, n_blocks) = "
                             f"{(ref.K, ref.m, ref.sub_steps, ref.n_blocks)}, backend {(w.K, w.m, w.sub_steps, w.n_blocks)}")
        # the work record against the reference's own spread at the same starts: the sweep's streams alone, never
        # stream 0, which the backend walks
        def counts_of(walk):
            c = {key: int(walk.work[key]) for key in ("n_hits", "n_crossings", "exhausted_steps")}
            c["illegal"] = int(np.asarray(walk.illegal_crossings).sum())
            return c
        gs = case(); gs.count_walls = True
        stream0 = counts_of(ref)
        runs = [counts_of(_walk_geometry(n_walkers, 2e-9, gs, T_max, block_h, seed=s, tiers="all",
                                         walker_batch_size=n_walkers, r0=x0))
                for s in sweep_seeds]
        mine, steps = counts_of(w), int(w.work["walker_steps"])
        for key in SPREAD_COUNTS:
            xs = [r[key] for r in runs]
            tv, p, sdv = _spread_p(mine[key], xs)
            judge(rec, key, p, jax=stream0[key], backend=mine[key], t=tv, dof=len(runs) - 1,
                  sweep=dict(mean=float(np.mean(xs)), sd=sdv, streams=len(runs)))
            if key == "n_hits" and sdv > 0:
                hits.append((name, tv, len(runs) - 1, (mine[key] - float(np.mean(xs))) / max(float(np.mean(xs)), 1.0),
                             int(s0), type(g).__name__))
        for key in RARE_COUNTS:
            k_a = sum(r[key] for r in runs)
            judge(rec, key, _rate_p(mine[key], steps, k_a, steps * len(runs)), jax=stream0[key], backend=mine[key],
                  reference=dict(count=k_a, streams=len(runs)))
        # the bands: each channel's quadratic variation and each band's mean square, walker means against their errors
        qa, ea = _mean_se(np.asarray(ref.qv_positions, np.float64).sum(axis=1))
        qb, eb = _mean_se(np.asarray(w.qv_positions, np.float64).sum(axis=1))
        z = _z(qa, ea, qb, eb)
        judge(rec, "qv_positions", min(1.0, 2.0 * norm.sf(z.max()) * z.size), jax=qa.tolist(), backend=qb.tolist(), z=z.tolist())
        if ref.qv_boundary is not None and w.qv_boundary is not None:
            qa, ea = _mean_se(np.asarray(ref.qv_boundary, np.float64).sum(axis=1))
            qb, eb = _mean_se(np.asarray(w.qv_boundary, np.float64).sum(axis=1))
            z = float(_z(qa, ea, qb, eb))
            judge(rec, "qv_boundary", 2.0 * norm.sf(z), jax=float(qa), backend=float(qb), z=z)
        ba, ea = _mean_se(np.asarray(ref.position_coeffs, np.float64)[:, 0, 2:, :] ** 2)      # (K, 3)
        bb, eb = _mean_se(np.asarray(w.position_coeffs, np.float64)[:, 0, 2:, :] ** 2)
        z = _z(ba, ea, bb, eb)
        judge(rec, "bands", min(1.0, 2.0 * norm.sf(z.max()) * z.size), z_max=float(z.max()))
        # confinement: the last block end's pool against the start's, where no crossing was granted (hard)
        pool0 = np.asarray(g2.classify_positions_exact(x0))
        poolT = np.asarray(g2.classify_positions_exact(np.asarray(w.raw_positions[:, -1])))
        crossings_allowed = g2.permeability is not None and float(g2.permeability) > 0
        leaked = int((g2.pool_of(poolT) != g2.pool_of(pool0)).sum())
        rec["checks"]["confinement"] = dict(changed_pool=leaked, ok=(crossings_allowed or leaked == 0))
        # the signal, from each side's bands
        seq = pgse([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 0.3e-3, 1.0e-3, bvalues=[2e9] * 3, TE=T_max, n_t=400, slew_rate=np.inf)
        sa, ea = coefficient_signal(ref, seq.G_eff, seq.dt); sb, eb = coefficient_signal(w, seq.G_eff, seq.dt)
        z = _z(sa, ea, sb, eb)
        judge(rec, "pgse", min(1.0, 2.0 * norm.sf(z.max()) * z.size), jax=sa.tolist(), backend=sb.tolist(), z=z.tolist())
        out.append(rec)
    # one threshold for every comparison: the report's false-failure rate over the number of comparisons
    M = max(len(pending), 1)
    alpha = float(false_failure) / M
    for rec, key in pending:
        c = rec["checks"][key]
        c["threshold"] = alpha
        c["ok"] = bool(c["p"] >= alpha)
    for rec in out:
        if rec.get("ok") is not None:
            rec["ok"] = all(c["ok"] for c in rec["checks"].values())
    if len(hits) >= 2:                                   # the pooled bias of the hit counts over the cases
        out.append(dict(case="pooled", checks=dict(n_hits_bias=pooled_bias(hits, false_failure))))
        out[-1]["ok"] = out[-1]["checks"]["n_hits_bias"]["ok"]
    return out
