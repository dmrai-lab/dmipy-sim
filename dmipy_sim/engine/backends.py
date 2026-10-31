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
has no other backend never sees the option do anything, and a name that is not installed is refused by name.
"""
from __future__ import annotations

import dmipy_replay.replay.compression as _r_compression
import math
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

JAX = "jax"
ENTRY_POINT_GROUP = "dmipy_sim.backends"


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
    kappa_over_D: float       #: the membrane's permeability over D; 0 for an impermeable walk
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
                              #: reads a raster (``K_c3`` modes every ``field_read_interval``); None otherwise
    drop_pool_changes: bool = False   #: an impermeable walk returns its walkers that changed pool as
                                      #: :attr:`CoefficientBatch.pool_changed` instead of being refused

    @property
    def T(self):
        """The walk's duration (s)."""
        return float(self.block_h) * int(self.n_blocks)


def event_capacity(n_steps, *, p_max=None, sigma=6.0, floor=16):
    """The C1 event list's capacity per walker for a permeable walk of ``n_steps`` physics steps: a crossing needs a
    hit and no hit crosses with a probability above ``p_max`` (the crossing rule's
    :data:`~dmipy_sim.engine.physics.CROSSING_P_MAX` by default, which sizes the step so), so the crossings per
    walker are bounded by Binomial(``n_steps``, ``p_max``); the capacity is its mean plus ``sigma`` standard
    deviations, at least ``floor``. A backend refuses by name a walker that exceeds it (the count returned), so the
    bound costs memory only: 9 bytes per entry on the CUDA backend."""
    if p_max is None:
        from .physics import CROSSING_P_MAX as p_max
    mu = float(p_max) * int(n_steps)
    return int(max(int(floor), math.ceil(mu + float(sigma) * math.sqrt(mu * (1.0 - float(p_max))))))


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
    so a pack built from the walk is bit for bit the pack the backend's own writer builds from the same walk."""
    position_coeffs: np.ndarray       #: (n, K + 2, 3) float64: per axis x(0), x(T) - x(0) (exact) and the K bands
    boundary_start: object            #: (n,) float64 the cumulative local time at the start (0), or None
    boundary_endpoint: object         #: (n,) float64 at the end, exact, or None
    boundary_bands: object            #: (n, K) float64 its bridge's bands at rho/D = 1, or None
    compartment: object               #: (n,) int8 the walker's pool at the start (1 inside; a label volume's label), or None
    illegal: np.ndarray               #: (n,) int32 refused walker-steps per walker
    work: tuple                       #: (hits | None, crossings | None, exhausted) per walker, int32 arrays
    K: int                            #: bands per channel
    m: int                            #: the order of the block table
    sub_steps: int                    #: physics steps per block (h = sub_steps * dt_sim)
    precision: str                    #: the floating-point type the bands were accumulated in across blocks
    saves: object = None              #: the block ends of the batch's first ``request.raw_subset`` walkers as a
                                      #: :class:`WalkBatch` (save 0 the start), or None when none were asked for
    qv_positions: object = None       #: (n, 3) float64 each walker's sum dx^2 per axis over every step
    qv_boundary: object = None        #: (n,) float64 its sum dlog^2 (each step's contact squared), or None
    events: object = None             #: C1's event list ``(offsets (n + 1,) int64, times (n_ev,) float64 s, labels
                                      #: (n_ev,) int8)``: walker w's pool changes are entries offsets[w] .. offsets[w+1],
                                      #: each the start of the step that ended in the pool entered; None unrecorded
    bound_events: object = None       #: an MT walk's bind / unbind events in the same form, or None
    field_bands: object = None        #: (n, n_ch, K_c3) float64 the field's channels (less the domain mean) on the
                                      #: cosine basis of [0, T], or None
    field_qv: object = None           #: (n, n_ch) float64 each channel's sum (b dt_f)^2 over its reads, or None
    field_qv_increments: object = None  #: (n, n_ch) float64 each channel's read series' quadratic variation, the sum
                                        #: of (b_{i+1} - b_i)^2 over its reads (RPK.md 9.5.5), or None
    dt_f: object = None               #: the field's read interval (s); 0.0 a read after every step; None no field
    k0_residual: object = None        #: (n, 3) float64 ``c_0 - (x(T) - x(0))``, the accumulation's measured residual
                                      #: on the k = 0 mode, when the backend reports it; None otherwise
    ring: object = None               #: blocks per projection window, when the backend projects in windows
    accumulation: object = None       #: how the backend summed the blocks (the header's ``block.accumulation``)
    pool_changed: object = None       #: (n,) bool the walkers of an impermeable walk whose label changed, when the request
                                      #: asked for them (``drop_pool_changes``); None otherwise


class Backend:
    """The protocol. A backend subclasses or duck-types this: ``name``, :meth:`refuses`, :meth:`walk_batch`."""
    name = "?"

    def refuses(self, request: WalkRequest) -> Optional[str]:
        """Why this backend cannot walk ``request`` (an engine class, a tier, a wall it does not implement), or
        None. The producer raises with the reason; it never falls back."""
        return None

    def walk_batch(self, request: WalkRequest, r0: np.ndarray, keys: np.ndarray, *, out=None) -> CoefficientBatch:
        """Walk the ``(n, 3)`` float32 starts ``r0`` from their ``(n, 2)`` uint32 keys in the coefficient form
        ``request`` names (``K`` bands at order ``m`` over ``n_blocks`` blocks of ``sub_steps`` steps): a
        :class:`CoefficientBatch`, with the block ends of the first ``request.raw_subset`` walkers as its ``saves``.
        A request with ``K = 0`` is the adaptive producer's walk of saves (a :class:`WalkBatch`; ``out``, when given,
        is a dict of that producer's own contiguous float32 arrays -- ``positions`` (n, n_t, 3),
        ``boundary_local_time`` and ``compartment`` (n, n_t), ``field_samples`` (n, n_tf, n_ch) -- a backend may write
        into instead of allocating its own, dmipy-sim-cuda#33)."""
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
        # box (dmipy-sim-cuda#36).
        return d.PackedCurvedCylinders([np.array([[VOID_XT, VOID_YT, 0.0], [VOID_XT, VOID_YT, VOID_L]])], [VOID_R],
                                       interior=False, box=(np.zeros(3), np.full(3, VOID_L)))

    void_cell_size = void_build().cell_size   # the geometry's own cell-size rule, read once rather than re-derived

    def void_starts(n, seed):
        rng = np.random.default_rng(seed)
        return np.column_stack([np.full(n, VOID_XT - VOID_R - 2.5 * void_cell_size),
                                rng.uniform(VOID_YT - 5e-6, VOID_YT + 5e-6, n),
                                rng.uniform(20e-6, 180e-6, n)]).astype(np.float32)

    return {
        "sphere": ParityCase(lambda: d.Sphere(R)),
        "sphere_perm": ParityCase(lambda: d.Sphere(R, permeability=1e-5)),
        "cylinder": ParityCase(lambda: d.Cylinder(R, (0, 0, 1))),
        "cylinder_perm": ParityCase(lambda: d.Cylinder(R, (0, 0, 1), permeability=1e-5)),
        "ellipsoid": ParityCase(lambda: d.Ellipsoid((R, 0.7 * R, 0.5 * R))),
        "packed_cyl": ParityCase(lambda: packed(d.PackedCylinders)),
        "packed_cyl_perm": ParityCase(lambda: packed(d.PackedCylinders, permeability=2e-5)),
        "packed_sph": ParityCase(lambda: packed(d.PackedSpheres)),
        "packed_sph_perm": ParityCase(lambda: packed(d.PackedSpheres, permeability=2e-5)),
        "label_slab": ParityCase(lambda: label()),
        "label_slab_perm": ParityCase(lambda: label(permeability=1e-5, pools={0: "free", 1: "grain"})),
        "mesh": ParityCase(lambda: mesh()),
        "mesh_perm": ParityCase(lambda: mesh(permeability=1e-5)),
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
    never a hand-rolled radial formula -- no walker was granted an illegal crossing, and more than
    ``near_wall_fraction`` of the walkers come within ``wall_margin_m`` of a wall at some block end (the geometry's own
    :meth:`~dmipy_sim.geometry.curved_cylinder.PackedCurvedCylinders.wall_scales`; the premise -- enough walkers
    reach the wall for the first two checks to mean something). Returns ``{check: {..., ok}}``; a failure here is
    a hard violation of the backend under test, never a rate comparison."""
    n, n_t, _ = w.raw_positions.shape
    flat = np.asarray(w.raw_positions, np.float64).reshape(-1, 3)
    pool = np.asarray(g.pool_of(g.classify_positions_exact(flat))).reshape(n, n_t)
    n_inside = int((pool != 0).sum())
    illegal = int(np.asarray(w.illegal_crossings).sum())
    d_wall, _ = g.wall_scales(flat)
    n_near = int((np.asarray(d_wall).reshape(n, n_t).min(axis=1) < wall_margin_m).sum())
    return dict(
        contained=dict(n_inside=n_inside, ok=n_inside == 0),
        illegal=dict(n_illegal=illegal, ok=illegal == 0),
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
    from dmipy_replay.constants import GAMMA
    from ..replay import compression as _cx
    C = np.asarray(walk.position_coeffs, np.float32).astype(np.float64)[:, 0]              # (n, K + 2, 3)
    W = _r_compression.band_projection(np.asarray(G, np.float64), float(dt_wf), float(walk.T_window), int(C.shape[1]) - 2)
    phi = GAMMA * np.einsum("wkc,mkc->wm", C, W)
    return _mean_se(np.cos(phi))


def parity_report(backend, cases=None, *, n_walkers=4000, T_max=2e-3, block_h=0.5e-3, seed=11, sigma=4.0, overdispersion=2.5):
    """Compare ``backend`` with the JAX kernels on ``cases`` (default :func:`parity_cases`) on the coefficient form:
    the same starts, the same sub-step and blocks, counting on, every walker's block ends kept as the positional
    oracle (``raw_subset``). Per case: the hit and crossing counts within ``sigma`` standard deviations, where the
    standard deviation is ``overdispersion * sqrt(a + b)``: a walker near a wall hits it on consecutive steps, so the
    per-walker counts are clustered and the total's spread is wider than Poisson (measured on the sphere and the
    ellipsoid at 20k walkers: seed-to-seed swings of 0.1-0.4 % against a Poisson 0.12 %, i.e. 2-3x); the exhausted
    walker-steps and the refused ones (``illegal_crossings``) compared the same way, since both are rates of the
    geometry at its budget and not of the backend (a permeable icosphere mesh of 320 facets exhausts 0-5 of 1.07 M
    walker-steps per seed on the JAX kernels themselves -- the wedge between a facet and the voxel face it touches);
    the bands at the band level -- each channel's measured quadratic variation (``qv_positions`` per axis,
    ``qv_boundary``) and each band's mean square per axis, every one within ``sigma`` of the two walks' combined
    standard errors; every walker on its side (the classifier at the last block end agrees with the start's pool
    unless a crossing was granted); and the perpendicular PGSE signal replayed from each side's bands
    (:func:`coefficient_signal`, what a pack of the walk stores) within ``sigma`` of their combined standard errors.
    A case is a rate comparison unless it declares its own starts (:attr:`ParityCase.starts`, e.g.
    ``strands_void``): there the backend under test walks those starts alone, at the case's own scale (its
    ``n_walkers``/``T_max``/``block_h``/``seed``, each falling back to this function's keyword of the same name when
    the case leaves it ``None``), and is held to :func:`void_invariant` on its block ends, a hard pass/fail with no
    JAX side to compare against. A case the backend refuses is recorded with its reason and not failed: refusal is
    the contract. Returns ``[{case, ok, ...}]``; the list is what a backend's tests assert on."""
    from .core import _walk_geometry
    from ..sequences import pgse
    out = []
    for name, case in (cases or parity_cases()).items():
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
            checks = void_invariant(g, w)
            out.append(dict(case=name, ok=all(c["ok"] for c in checks.values()), checks=checks))
            continue
        ref = _walk_geometry(n_walkers, 2e-9, g, T_max, block_h, seed=seed, tiers="all", walker_batch_size=n_walkers,
                                    raw_subset=n_walkers)
        g2 = case(); g2.count_walls = True
        x0 = np.asarray(ref.position_coeffs[:, 0, 0, :], np.float32)                       # the exact starts
        try:
            w = _walk_geometry(n_walkers, 2e-9, g2, T_max, block_h, seed=seed, tiers="all", walker_batch_size=n_walkers,
                                      r0=x0, backend=backend, raw_subset=n_walkers)
        except ValueError as e:
            if "refuses" in str(e) or "does not" in str(e):
                out.append(dict(case=name, ok=None, refused=str(e))); continue
            raise
        rec = dict(case=name, ok=True, checks={})
        if (w.K, w.m, w.sub_steps, w.n_blocks) != (ref.K, ref.m, ref.sub_steps, ref.n_blocks):
            raise ValueError(f"the two walks of {name!r} are not one resolution: jax (K, m, sub_steps, n_blocks) = "
                             f"{(ref.K, ref.m, ref.sub_steps, ref.n_blocks)}, backend {(w.K, w.m, w.sub_steps, w.n_blocks)}")
        # the work record
        counts = {key: (ref.work[key], w.work[key]) for key in ("n_hits", "n_crossings", "exhausted_steps")}
        counts["illegal"] = (int(np.asarray(ref.illegal_crossings).sum()), int(np.asarray(w.illegal_crossings).sum()))
        for key, (a, b) in counts.items():
            z = abs(a - b) / (overdispersion * max(np.sqrt(a + b), 1.0))
            rec["checks"][key] = dict(jax=int(a), backend=int(b), z=float(z), ok=bool(z <= sigma))
        # the bands: each channel's quadratic variation and each band's mean square, walker means against their errors
        qa, ea = _mean_se(np.asarray(ref.qv_positions, np.float64).sum(axis=1))
        qb, eb = _mean_se(np.asarray(w.qv_positions, np.float64).sum(axis=1))
        z = _z(qa, ea, qb, eb)
        rec["checks"]["qv_positions"] = dict(jax=qa.tolist(), backend=qb.tolist(), z=z.tolist(), ok=bool((z <= sigma).all()))
        if ref.qv_boundary is not None and w.qv_boundary is not None:
            qa, ea = _mean_se(np.asarray(ref.qv_boundary, np.float64).sum(axis=1))
            qb, eb = _mean_se(np.asarray(w.qv_boundary, np.float64).sum(axis=1))
            z = float(_z(qa, ea, qb, eb))
            rec["checks"]["qv_boundary"] = dict(jax=float(qa), backend=float(qb), z=z, ok=bool(z <= sigma))
        ba, ea = _mean_se(np.asarray(ref.position_coeffs, np.float64)[:, 0, 2:, :] ** 2)      # (K, 3)
        bb, eb = _mean_se(np.asarray(w.position_coeffs, np.float64)[:, 0, 2:, :] ** 2)
        z = _z(ba, ea, bb, eb)
        rec["checks"]["bands"] = dict(z_max=float(z.max()), ok=bool((z <= sigma).all()))
        # confinement: the last block end's pool against the start's, where no crossing was granted
        pool0 = np.asarray(g2.classify_positions_exact(x0))
        poolT = np.asarray(g2.classify_positions_exact(np.asarray(w.raw_positions[:, -1])))
        crossings_allowed = g2.permeability is not None and float(g2.permeability) > 0
        leaked = int((g2.pool_of(poolT) != g2.pool_of(pool0)).sum())
        rec["checks"]["confinement"] = dict(changed_pool=leaked, ok=(crossings_allowed or leaked == 0))
        # the signal, from each side's bands
        seq = pgse([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 0.3e-3, 1.0e-3, bvalues=[2e9] * 3, TE=T_max, n_t=400, slew_rate=np.inf)
        sa, ea = coefficient_signal(ref, seq.G_eff, seq.dt); sb, eb = coefficient_signal(w, seq.G_eff, seq.dt)
        z = _z(sa, ea, sb, eb)
        rec["checks"]["pgse"] = dict(jax=sa.tolist(), backend=sb.tolist(), z=z.tolist(), ok=bool((z <= sigma).all()))
        rec["ok"] = all(c["ok"] for c in rec["checks"].values())
        out.append(rec)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The request directory: dmipy-sim's side of the backend protocol (dmrai-lab/dmipy-sim#691). A backend
# executes a WalkRequest; the strands engine's request DIRECTORY (request.txt + the strand/adaptive/field
# tables) is serialised once, here, from the geometry's/plan's/basis's own arrays -- never read back by a
# backend's own ctypes struct, which is a different consumer of the same numbers (a live library call, not a
# file). dmipy_sim_cuda.cli.write_request (dmipy-sim-cuda#78) writes the same files from its ctypes structs, field
# by field under their names; a substrate kit (dmipy_sim.fill.kit) builds them from a template, never installing
# dmipy_sim_cuda to do it.
# ─────────────────────────────────────────────────────────────────────────────

#: the strand-walk tables of one pool's geometry, the file names a request directory writes them under
STRAND_ARRAY_NAMES = ("A.f32", "AB.f32", "AB2.f32", "rr.f32", "tube.i32", "cell_off.i32", "cell_ids.i32")
#: the field basis's own tables, shared by every pool that samples it
FIELD_ARRAY_NAMES = ("fA.f32", "fAB.f32", "fAB2.f32", "fa.f32", "fb.f32", "fsid.i32", "fcell_off.i32", "fcell_ids.i32", "fbox.i16")
#: the request-struct fields, in the backend's ``DscRequest`` field order: a kit's template holds them all, a batch
#: setting its own ``seed`` and its ``raw_subset`` (at most its walkers)
REQUEST_TEMPLATE_KEYS = ("n_blocks", "sub_steps", "step_l", "atom_short", "atom_long", "p_long", "kappa_over_D", "record",
                         "count", "budget", "seed", "block_h", "dt_sim", "K", "m", "ring", "events", "raw_subset")
#: the field-struct fields, in the backend's ``DscField`` field order (``far_dims`` / ``far_origin`` a batch's crop)
FIELD_STRUCT_KEYS = ("n_seg", "nnz", "dims", "gmin", "cs", "radius", "reach", "gate_radii", "has_far", "near_m", "blend_m",
                     "far_dims", "far_origin", "far_h", "reuse", "list_k", "segments_max", "K_c3", "read_every", "mean")
#: the terms the backend projects a window of blocks with: a ring of ``RING_TERMS // (m + 1)`` blocks by default
RING_TERMS = 256
#: the strands-struct fields, in the backend's ``DscStrands`` field order
STRANDS_STRUCT_KEYS = ("n_seg", "nnz", "dims", "gmin", "cs", "interior", "box_reflect", "lo", "hi", "nudge")


def _f32(x):
    """A value as a ``ctypes.c_float`` struct field would hold it after assignment: the float32 truncation a
    backend's request structs apply to every field named as such below."""
    return float(np.float32(x))


def format_request_value(v):
    """One ``request.txt`` value in the backend protocol's own spelling: an array comma-joined (a float
    ``repr``'d, an int as itself), a scalar float ``repr``'d, a scalar int, or a bare name (``"strands"``) as
    itself."""
    if isinstance(v, (list, tuple)):
        return ",".join(repr(float(x)) if isinstance(x, float) else str(int(x)) for x in v)
    if isinstance(v, float):
        return repr(float(v))
    if isinstance(v, str):
        return v
    return int(v)


def csr_cells(cell):
    """A geometry's padded ``(n_cells, C)`` cell table (ids ascending per cell, -1 padding) as CSR
    ``(offsets (n_cells + 1,), ids (nnz,))`` int32, the order kept: the one conversion every table with a
    bucketed cell grid (the strands, the field basis) goes through."""
    cell = np.asarray(cell)
    valid = cell >= 0
    counts = valid.sum(1, dtype=np.int64)
    offsets = np.zeros(cell.shape[0] + 1, np.int64)
    np.cumsum(counts, out=offsets[1:])
    return np.ascontiguousarray(offsets, np.int32), np.ascontiguousarray(cell[valid], np.int32)


def strands_struct_fields(geometry):
    """``(fields, arrays)`` of a ``PackedCurvedCylinders`` ``geometry``: the backend's ``DscStrands`` fields (in
    :data:`STRANDS_STRUCT_KEYS` order) and the tables a request directory names ``A.f32``/``AB.f32``/``AB2.f32``/
    ``rr.f32``/``tube.i32``/``cell_off.i32``/``cell_ids.i32`` (:data:`STRAND_ARRAY_NAMES`) -- built from the
    geometry's own arrays, with the float32 truncation a ``DscStrands`` ctypes field applies on assignment."""
    cell_off, cell_ids = csr_cells(np.asarray(geometry._CELL))
    lo, hi = ((np.asarray(geometry._lo), np.asarray(geometry._hi)) if geometry.box_reflect else (np.zeros(3), np.zeros(3)))
    fields = dict(n_seg=int(geometry._A.shape[0]), nnz=int(cell_ids.shape[0]), dims=[int(x) for x in geometry._DIMS],
                 gmin=[_f32(x) for x in np.asarray(geometry._GMIN)], cs=_f32(geometry._CS), interior=int(bool(geometry.interior)),
                 box_reflect=int(bool(geometry.box_reflect)), lo=[_f32(x) for x in lo], hi=[_f32(x) for x in hi],
                 nudge=_f32(geometry.nudge_m))
    arrays = {"A.f32": np.ascontiguousarray(np.asarray(geometry._A, np.float32)),
             "AB.f32": np.ascontiguousarray(np.asarray(geometry._AB, np.float32)),
             "AB2.f32": np.ascontiguousarray(np.asarray(geometry._AB2, np.float32)),
             "rr.f32": np.ascontiguousarray(np.asarray(geometry._rout, np.float32)),
             "tube.i32": np.ascontiguousarray(np.asarray(geometry._seg_tube, np.int32)),
             "cell_off.i32": cell_off, "cell_ids.i32": cell_ids}
    return fields, arrays


def field_tables(basis):
    """The field basis's own tables -- the segment tables, the CSR cell table and each segment's cell box, with
    a sanity check that the bucketing accounts for every CSR entry -- built from ``basis``'s own arrays."""
    fcell_off, fcell_ids = csr_cells(np.asarray(basis._CELL))
    A64 = np.vstack([c[:-1] for c in basis.centerlines]); AB64 = np.vstack([c[1:] - c[:-1] for c in basis.centerlines])
    lo64 = np.minimum(A64, A64 + AB64); hi64 = np.maximum(A64, A64 + AB64); cs64 = 1.01 * basis.gather_radius_m
    dims = np.asarray(basis._dims)
    loc = np.clip(np.floor((lo64 - basis._gmin) / cs64).astype(int), 0, dims - 1)
    hic = np.clip(np.floor((hi64 - basis._gmin) / cs64).astype(int), 0, dims - 1)
    fbox = np.ascontiguousarray(np.stack([loc[:, 0], hic[:, 0], loc[:, 1], hic[:, 1], loc[:, 2], hic[:, 2]], axis=1), np.int16)
    counts = (hic - loc + 1).prod(1)
    if int(counts.sum()) != int(fcell_ids.shape[0]):
        raise RuntimeError(f"the field's cell boxes ({int(counts.sum())} entries) do not match its cell table "
                           f"({fcell_ids.shape[0]}): the bucketing differs")
    return {"fA.f32": np.ascontiguousarray(np.asarray(basis._A, np.float32)), "fAB.f32": np.ascontiguousarray(np.asarray(basis._AB, np.float32)),
            "fAB2.f32": np.ascontiguousarray(np.asarray(basis._AB2, np.float32)), "fa.f32": np.ascontiguousarray(np.asarray(basis._a, np.float32)),
            "fb.f32": np.ascontiguousarray(np.asarray(basis._b, np.float32)), "fsid.i32": np.ascontiguousarray(np.asarray(basis._sid, np.int32)),
            "fcell_off.i32": fcell_off, "fcell_ids.i32": fcell_ids, "fbox.i16": fbox}


def cell_window(gmin, cs, dims, lo_m, hi_m, margin_m):
    """The cell-index window of a box ``[lo_m, hi_m]`` expanded by ``margin_m`` on a grid of origin ``gmin``,
    cell size ``cs`` and ``dims`` cells: ``(win_lo, win_hi)`` int64 ``(3,)`` -- inclusive/exclusive, each clipped
    to ``[0, dims]`` -- the one piece of arithmetic a block kit's cell window is (:func:`window_cell_table`
    crops a table to it; the request format's concern, not a block kit's own)."""
    gmin = np.asarray(gmin, np.float64); dims = np.asarray(dims, np.int64); cs = float(cs)
    lo = np.clip(np.floor((np.asarray(lo_m, np.float64) - margin_m - gmin) / cs).astype(np.int64), 0, dims - 1)
    hi = np.clip(np.floor((np.asarray(hi_m, np.float64) + margin_m - gmin) / cs).astype(np.int64), 0, dims - 1) + 1
    return lo, hi


def window_cell_table(cell, dims, win_lo, win_hi):
    """``cell`` (a geometry's or a field basis's padded ``(n_cells, C)`` table, x-major as :func:`bucket_by_bbox`
    builds it) restricted to the cell window ``[win_lo, win_hi)`` and turned into CSR with its ids renumbered
    DENSELY over the referenced primitives alone: ``(cell_off, cell_ids, uniq, window_dims)`` -- ``uniq`` selects
    a primitive table's rows in the new, dense order (``uniq[cell_ids] == the window's true ids``, a bijection on
    the referenced ids by construction of :func:`numpy.unique`), ``window_dims`` the window's own ``(3,)``. The
    window is a sub-block of the SAME x-major table (:func:`csr_cells` turns any such padded table into CSR), so
    cropping first and converting once is exactly the whole table's rule applied to fewer cells."""
    dims = tuple(int(x) for x in dims)
    cell = np.asarray(cell)
    cell3 = cell.reshape(dims + (cell.shape[-1],))
    window_dims = tuple(int(x) for x in (np.asarray(win_hi) - np.asarray(win_lo)))
    sub = cell3[win_lo[0]:win_hi[0], win_lo[1]:win_hi[1], win_lo[2]:win_hi[2], :]
    cell_off, cell_ids = csr_cells(np.ascontiguousarray(sub).reshape(-1, cell.shape[-1]))
    uniq, inv = np.unique(cell_ids, return_inverse=True)
    cell_ids = np.ascontiguousarray(inv.reshape(cell_ids.shape), np.int32)
    return cell_off, cell_ids, uniq, window_dims


def strands_struct_fields_window(geometry, lo_m, hi_m, margin_m):
    """``(fields, arrays)`` of a ``PackedCurvedCylinders`` ``geometry`` exactly as :func:`strands_struct_fields`,
    but cropped to the cell window of the box ``[lo_m, hi_m]`` expanded by ``margin_m`` (:func:`cell_window`):
    only the segments the window's cells reference, renumbered densely (:func:`window_cell_table`); ``dims`` and
    ``gmin`` are the WINDOW's, so the 27-cell gather a backend runs on this template reads the window alone and
    a walker beyond it reads a clamped border cell, by construction already clear of every segment the window
    does not carry. ``lo``/``hi`` (the box-reflect walls) are the geometry's own -- the substrate's domain edges,
    not the window's."""
    gmin = np.asarray(geometry._GMIN); dims = np.asarray(geometry._DIMS); cs = float(geometry._CS)
    win_lo, win_hi = cell_window(gmin, cs, dims, lo_m, hi_m, margin_m)
    cell_off, cell_ids, uniq, window_dims = window_cell_table(np.asarray(geometry._CELL), dims, win_lo, win_hi)
    window_gmin = gmin + win_lo * cs
    lo, hi = ((np.asarray(geometry._lo), np.asarray(geometry._hi)) if geometry.box_reflect else (np.zeros(3), np.zeros(3)))
    fields = dict(n_seg=int(uniq.shape[0]), nnz=int(cell_ids.shape[0]), dims=[int(x) for x in window_dims],
                 gmin=[_f32(x) for x in window_gmin], cs=_f32(cs), interior=int(bool(geometry.interior)),
                 box_reflect=int(bool(geometry.box_reflect)), lo=[_f32(x) for x in lo], hi=[_f32(x) for x in hi],
                 nudge=_f32(geometry.nudge_m))
    arrays = {"A.f32": np.ascontiguousarray(np.asarray(geometry._A, np.float32)[uniq]),
             "AB.f32": np.ascontiguousarray(np.asarray(geometry._AB, np.float32)[uniq]),
             "AB2.f32": np.ascontiguousarray(np.asarray(geometry._AB2, np.float32)[uniq]),
             "rr.f32": np.ascontiguousarray(np.asarray(geometry._rout, np.float32)[uniq]),
             "tube.i32": np.ascontiguousarray(np.asarray(geometry._seg_tube, np.int32)[uniq]),
             "cell_off.i32": cell_off, "cell_ids.i32": cell_ids}
    return fields, arrays


def field_tables_window(basis, lo_m, hi_m, margin_m):
    """``(dims_gmin, arrays)`` of a :class:`~dmipy_sim.fields.strand_field.StrandFieldBasis` exactly as
    :func:`field_tables`, but cropped to the cell window of the box ``[lo_m, hi_m]`` expanded by ``margin_m``
    (:func:`cell_window`): only the segments the window's cells reference, renumbered densely
    (:func:`window_cell_table`), with ``fbox`` rebuilt in the WINDOW's own cell coordinates (never the whole
    table's ``fbox`` shifted: a segment's box is clipped to ``[0, dims)`` at build time, so shifting a clipped
    box would carry the wrong clip -- recomputing against the window's ``gmin``/``dims`` is the same formula
    :func:`field_tables` uses, just smaller). ``dims_gmin`` is ``{"dims": [...], "gmin": [...]}`` of the window."""
    gmin = np.asarray(basis._gmin); dims = np.asarray(basis._dims); cs = float(basis._CS)
    win_lo, win_hi = cell_window(gmin, cs, dims, lo_m, hi_m, margin_m)
    cell_off, cell_ids, uniq, window_dims = window_cell_table(np.asarray(basis._CELL), dims, win_lo, win_hi)
    window_gmin = gmin + win_lo * cs
    A64 = np.vstack([c[:-1] for c in basis.centerlines]); AB64 = np.vstack([c[1:] - c[:-1] for c in basis.centerlines])
    lo64 = np.minimum(A64, A64 + AB64); hi64 = np.maximum(A64, A64 + AB64)
    wd = np.asarray(window_dims)
    loc = np.clip(np.floor((lo64 - window_gmin) / cs).astype(int), 0, wd - 1)
    hic = np.clip(np.floor((hi64 - window_gmin) / cs).astype(int), 0, wd - 1)
    fbox = np.stack([loc[:, 0], hic[:, 0], loc[:, 1], hic[:, 1], loc[:, 2], hic[:, 2]], axis=1)[uniq]
    dims_gmin = dict(dims=[int(x) for x in window_dims], gmin=[_f32(x) for x in window_gmin])
    arrays = {"fA.f32": np.ascontiguousarray(np.asarray(basis._A, np.float32)[uniq]),
             "fAB.f32": np.ascontiguousarray(np.asarray(basis._AB, np.float32)[uniq]),
             "fAB2.f32": np.ascontiguousarray(np.asarray(basis._AB2, np.float32)[uniq]),
             "fa.f32": np.ascontiguousarray(np.asarray(basis._a, np.float32)[uniq]),
             "fb.f32": np.ascontiguousarray(np.asarray(basis._b, np.float32)[uniq]),
             "fsid.i32": np.ascontiguousarray(np.asarray(basis._sid, np.int32)[uniq]),
             "fcell_off.i32": cell_off, "fcell_ids.i32": cell_ids, "fbox.i16": np.ascontiguousarray(fbox, np.int16)}
    return dims_gmin, arrays


def field_struct_fields(basis, radius_m, reuse, list_k, K_c3=0, read_every=0):
    """The backend's ``DscField`` fields in its field order (:data:`FIELD_STRUCT_KEYS`) -- built from ``basis``'s own
    arrays and the sampling settings, with no ``r0``: ``far_dims`` / ``far_origin`` (the crop, a batch's own) are
    zeros here for the batch to set. ``K_c3`` / ``read_every`` are the field accumulated in the walk
    (dmipy-sim#742): its cosine modes per channel and the rounds between reads (0: after every step); ``mean`` is
    the basis's domain mean the kernel subtracts at every read, written only when the field is accumulated (zeros
    otherwise, as the backend's own struct holds it)."""
    _, fcell_ids = csr_cells(np.asarray(basis._CELL))
    far = basis.far
    mean = [_f32(x) for x in np.asarray(basis.mean, np.float32)] if int(K_c3) else [0.0] * len(basis.channel_names)
    return dict(n_seg=int(basis.n_segments), nnz=int(fcell_ids.shape[0]), dims=[int(x) for x in basis._dims],
               gmin=[_f32(x) for x in np.asarray(basis._gmin)], cs=_f32(basis._CS), radius=_f32(radius_m),
               reach=_f32(basis.gather_radius_m), gate_radii=_f32(basis.NEAREST_GATE_RADII),
               has_far=int(far is not None), near_m=_f32(far.near_m if far is not None else 0.0),
               blend_m=_f32(far.blend_m if far is not None else 0.0), far_dims=[0, 0, 0], far_origin=[0.0, 0.0, 0.0],
               far_h=_f32(far.spacing_m if far is not None else 0.0), reuse=int(reuse), list_k=max(1, int(list_k)),
               segments_max=int(basis.segments_max), K_c3=int(K_c3), read_every=int(read_every), mean=mean)


def adaptive_struct_fields(plan, R_max):
    """The backend's ``DscAdaptive`` fields of an :class:`~dmipy_sim.engine.adaptive.AdaptivePlan` (``R_max`` is
    the WALKING geometry's own ``_Rmax``, which differs by pool: the axolemma's radius for intra, the sheath's
    for extra)."""
    n_classes = int(plan.n_classes)
    steps_c = [0] * 8; step_l_c = [0.0] * 8; reach_c = [0.0] * 8
    for c in range(n_classes):
        steps_c[c] = int(plan.steps_c[c]); step_l_c[c] = _f32(plan.step_l_c[c]); reach_c[c] = _f32(plan.reach_c[c])
    return dict(K=int(plan.steps_per_round), n_rounds=int(plan.n_rounds), n_classes=n_classes,
               sigma_round=_f32(plan.sigma_round), far_at=_f32(plan.far_at), R_min=_f32(plan.R_min), R_max=_f32(R_max),
               steps_c=steps_c, step_l_c=step_l_c, reach_c=reach_c)


def request_struct_fields(*, n_blocks, sub_steps, step_l, kappa_over_D, record, count, budget, block_h, dt_sim, K, m,
                          events, raw_subset, ring=None, seed=0):
    """The backend's ``DscRequest`` fields in its field order (:data:`REQUEST_TEMPLATE_KEYS`): the walk's ``n_blocks``
    blocks of ``sub_steps`` physics steps, the step length's scale ``step_l = sqrt(6 D dt_sim)`` with the step law's
    two atoms and the long one's probability (:mod:`dmipy_sim.math.step_law`: a kernel multiplies the scale by the
    atom it draws and defines no length of its own), the membrane, what is recorded and counted, the bounce budget,
    the ``seed`` (a batch's own), the block ``block_h`` and the step ``dt_sim`` (s, in double precision), the bands
    ``K`` at the block table's order ``m``, the ``ring`` of blocks projected at once (``RING_TERMS // (m + 1)`` by
    default), C1's event capacity (``events`` of a recorded walk, 0 otherwise) and the raw subset."""
    from ..math.step_law import PAIR_SHORT, PAIR_LONG, PAIR_P_LONG
    return dict(n_blocks=int(n_blocks), sub_steps=int(sub_steps), step_l=_f32(step_l),
               atom_short=_f32(PAIR_SHORT), atom_long=_f32(PAIR_LONG), p_long=_f32(PAIR_P_LONG),
               kappa_over_D=_f32(kappa_over_D), record=int(bool(record)), count=int(bool(count)), budget=int(budget),
               seed=int(seed) & 0xFFFFFFFFFFFFFFFF, block_h=float(block_h), dt_sim=float(dt_sim), K=int(K), m=int(m),
               ring=(RING_TERMS // (int(m) + 1) if ring is None else int(ring)), events=(int(events) if record else 0),
               raw_subset=max(int(raw_subset), 0))


def field_read_every(dt_f, plan, block):
    """The rounds between an adaptive walk's reads of the field for a read interval ``dt_f`` (s) and a block of
    ``block`` physics steps: the most rounds, at least one, that divide the block's rounds and last no longer than
    ``dt_f``; 0 (``dt_f = 0``) reads after every step."""
    if float(dt_f) == 0.0:
        return 0
    rounds = int(block) // int(plan.steps_per_round)
    best = 1
    for r in range(1, rounds + 1):
        if rounds % r == 0 and r * float(plan.dt_round) <= float(dt_f) * (1.0 + 1e-9):
            best = r
    return best


def block_table_of(request):
    """The block table a coefficient request is walked against, ``(n_blocks, K + 1, m + 1)`` float64 over the walk
    (:func:`~dmipy_sim.replay.compression.block_table`): one storage window, since a backend accumulates one window
    per walk (several are refused by name)."""
    from ..replay.compression import block_table
    return np.ascontiguousarray(block_table(float(request.T), int(request.n_blocks), int(request.K), int(request.m)), np.float64)


def crop_far_grid(values, origin_m, spacing_m, r0, T, diffusivity):
    """The far grid cropped to what a batch of starts ``r0`` can reach over a walk of ``T`` seconds: the starts'
    bounding box plus eight sigma of the walk's excursion, ``8 sqrt(2 D T)``, and three nodes, so a shard of a large
    substrate uploads a crop, never the whole grid. Returns ``(crop.view(uint16), far_dims, far_origin)``."""
    N = np.asarray(values.shape[:3])
    h = float(spacing_m)
    T = float(T)
    margin = 8.0 * math.sqrt(2.0 * float(diffusivity) * T) + 3.0 * h
    r0 = np.asarray(r0, np.float64)
    origin = np.asarray(origin_m, np.float64)
    lo = np.clip(np.floor((r0.min(0) - margin - origin) / h).astype(int) - 1, 0, N - 1)
    hi = np.clip(np.ceil((r0.max(0) + margin - origin) / h).astype(int) + 2, 1, N)
    hi = np.maximum(hi, lo + 1)
    crop = np.ascontiguousarray(np.asarray(values[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], np.float16))
    far_dims = [int(x) for x in crop.shape[:3]]
    far_origin = [_f32(x) for x in (origin + lo * h)]
    return crop.view(np.uint16), far_dims, far_origin


def write_request_directory(request: WalkRequest, r0, keys, out_dir) -> Path:
    """The request directory of a live, resolved ``WalkRequest`` whose geometry is the strands engine:
    ``request.txt`` (the engine's name, ``n``, the ``DscRequest`` fields -- the blocks, the step, the seed, the bands
    ``K`` at order ``m``, the ring, the event capacity, the raw subset --, the ``DscStrands`` fields, and when the
    request carries them the ``DscAdaptive`` and ``DscField`` fields, each prefixed ``adaptive_`` / ``field_``: the
    field's ``K_c3`` modes read every ``read_every`` rounds and its ``mean``), ``r0.f32``, ``keys.u32``, the block
    table ``table.f64`` (:func:`block_table_of`) and the strand / field tables beside it (the far grid CROPPED to
    what ``r0`` can reach, :func:`crop_far_grid`). dmipy-sim's side of the backend protocol: every value comes from
    the geometry's / plan's / basis's own arrays, with the float32 truncation the backend's ctypes structs apply on
    assignment (:func:`request_struct_fields`, :func:`strands_struct_fields`, :func:`adaptive_struct_fields`,
    :func:`field_struct_fields`) -- what ``dmipy_sim_cuda.cli.write_request`` writes for the strands engine
    (dmipy-sim-cuda#78), byte for byte, without that package installed. ``classify`` is the CUDA CLI's own flag and
    is not written here."""
    from ..geometry.curved_cylinder import PackedCurvedCylinders
    g = request.geometry
    if not isinstance(g, PackedCurvedCylinders):
        raise ValueError(f"write_request_directory serialises the strands engine only, not {type(g).__name__}")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    r0 = np.ascontiguousarray(r0, np.float32); n = int(r0.shape[0])
    struct, arrays = strands_struct_fields(g)
    lines = dict(engine="strands", n=n)
    lines.update(request_struct_fields(n_blocks=int(request.n_blocks), sub_steps=request.sub_steps,
                                       step_l=float(np.sqrt(6.0 * request.diffusivity * request.dt_sim)),
                                       kappa_over_D=request.kappa_over_D, record=request.record, count=request.count_walls,
                                       budget=(int(g.bounce_loop.budget) if g.bounce_loop is not None else 0),
                                       seed=request.seed, block_h=request.block_h, dt_sim=request.dt_sim, K=request.K, m=request.m,
                                       events=request.events, raw_subset=min(int(request.raw_subset), n)))
    lines.update(struct)
    if request.stepping is not None:
        lines["adaptive"] = 1
        lines.update({f"adaptive_{k}": v for k, v in adaptive_struct_fields(request.stepping, g._Rmax).items()})
    if request.stepping is not None and request.field is not None:
        f = request.field; basis = f.basis
        K_c3 = int(request.K_c3)
        every = field_read_every(float(request.field_read_interval), request.stepping, int(request.sub_steps)) if K_c3 else 0
        fields = field_struct_fields(basis, f.radius_m, f.reuse_intervals, f.list_k, K_c3=K_c3, read_every=every)
        arrays.update(field_tables(basis))
        if basis.far is not None:
            crop, fields["far_dims"], fields["far_origin"] = crop_far_grid(basis.far.values, basis.far.origin_m, basis.far.spacing_m,
                                                                           r0, request.T, request.diffusivity)
            arrays["far.u16"] = crop
        lines["field"] = 1
        lines.update({f"field_{k}": v for k, v in fields.items()})
    block_table_of(request).tofile(out / "table.f64")
    (out / "request.txt").write_text("".join(f"{k}={format_request_value(v)}\n" for k, v in lines.items()))
    r0.tofile(out / "r0.f32")
    np.ascontiguousarray(keys, np.uint32).reshape(n, -1)[:, :2].copy().tofile(out / "keys.u32")
    for name, a in arrays.items():
        np.ascontiguousarray(a).tofile(out / name)
    return out
