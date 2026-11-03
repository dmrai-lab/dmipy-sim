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
import math
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
    bound costs memory only: 9 bytes per entry on a GPU backend."""
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
        into instead of allocating its own)."""
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


def pooled_bias(hits, false_failure):
    """The pooled bias of a report's hit counts over its cases, ``hits`` ``[(case, t, dof, relative)]``: each ``t``
    (:func:`_spread_p`) turned into a normal score through its own Student CDF, ``Z = sum z / sqrt(C)`` (Stouffer),
    ``ok`` when ``|Z|`` is within the two-sided normal threshold of ``false_failure`` (one test); the sign test of the
    same differences and the mean relative bias ``(backend - mean) / mean`` with its standard error beside it."""
    from scipy.stats import norm, t as student, binomtest
    zs = np.array([norm.ppf(student.cdf(tv, dof)) for _, tv, dof, _ in hits])
    Z = float(zs.sum() / np.sqrt(zs.size))
    z_crit = float(norm.isf(float(false_failure) / 2.0))
    below = int(sum(1 for _, tv, _, _ in hits if tv < 0))
    rel = np.array([r for _, _, _, r in hits])
    return dict(Z=Z, z_crit=z_crit, cases=len(hits), ok=bool(abs(Z) <= z_crit),
                sign_test=dict(below=below, of=len(hits), p=float(binomtest(below, len(hits), 0.5).pvalue)),
                relative_bias_mean=float(rel.mean()), relative_bias_se=float(rel.std(ddof=1) / np.sqrt(rel.size)),
                per_case={n: dict(t=float(tv), relative=float(r)) for n, tv, _, r in hits})


def parity_report(backend, cases=None, *, n_walkers=4000, T_max=2e-3, block_h=0.5e-3, seed=11, sweep=8, false_failure=1e-2):
    """Compare ``backend`` with the JAX kernels on ``cases`` (default :func:`parity_cases`) on the coefficient form:
    the same starts, the same sub-step and blocks, counting on, every walker's block ends kept as the positional
    oracle (``raw_subset``).

    **The reference's own spread, measured.** Per case the JAX kernels walk the backend's starts again with ``sweep``
    further streams (seeds ``seed + 1 ..``): the counts' stream-to-stream spread is the case's own, never a typed
    overdispersion (a walker near a wall hits it on consecutive steps, so the counts are clustered; measured 2.4x
    Poisson on the sphere and 5.5x on the extra-axonal strands at 20k walkers). The hit and crossing counts are each
    a new draw of the reference's (:func:`_spread_p`: Student t on ``sweep`` degrees of freedom, the prediction
    interval of one more stream); the exhausted and the refused walker-steps are rare events, judged as a rate against
    the reference's pooled count over the sweep (:func:`_rate_p`, binomial). The bands at the band level -- each
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
    differences and the mean relative bias ``(backend - mean) / mean`` beside it.

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

    def judge(rec, key, p, **fields):
        rec["checks"][key] = dict(fields, p=float(p))
        pending.append((rec, key))

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
        # the work record against the reference's own spread at the same starts: sweep further streams
        def counts_of(walk):
            c = {key: int(walk.work[key]) for key in ("n_hits", "n_crossings", "exhausted_steps")}
            c["illegal"] = int(np.asarray(walk.illegal_crossings).sum())
            return c
        gs = case(); gs.count_walls = True
        runs = [counts_of(ref)] + [counts_of(_walk_geometry(n_walkers, 2e-9, gs, T_max, block_h, seed=s, tiers="all",
                                                            walker_batch_size=n_walkers, r0=x0))
                                   for s in range(seed + 1, seed + 1 + int(sweep))]
        mine, steps = counts_of(w), int(w.work["walker_steps"])
        for key in SPREAD_COUNTS:
            tv, p, sdv = _spread_p(mine[key], [r[key] for r in runs])
            judge(rec, key, p, jax=runs[0][key], backend=mine[key], t=tv, dof=len(runs) - 1,
                  sweep=dict(mean=float(np.mean([r[key] for r in runs])), sd=sdv, streams=len(runs)))
            if key == "n_hits" and sdv > 0:
                hits.append((name, tv, len(runs) - 1, (mine[key] - float(np.mean([r[key] for r in runs]))) / max(np.mean([r[key] for r in runs]), 1.0)))
        for key in RARE_COUNTS:
            k_a = sum(r[key] for r in runs)
            judge(rec, key, _rate_p(mine[key], steps, k_a, steps * len(runs)), jax=runs[0][key], backend=mine[key],
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
