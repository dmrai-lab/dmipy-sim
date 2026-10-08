"""Backends: another execution of the same physics, resolved by name (dmrai-lab/dmipy-sim#635).

A backend walks a batch of walkers the way the JAX kernels of :mod:`.core` walk it -- the same geometry, the same
starts and keys, the same sub-step and save grid -- and returns what they return. dmipy-sim stays the reference
physics: a backend reads every constant from the geometry it is handed and carries none of its own.

What a backend signs
--------------------
- **One physics.** The rule is dmipy-sim's; the backend executes it.
- **Parity, not identity.** A different RNG stream and FMA policy make bit-identity impossible, and it is not stable
  across XLA compilations of one program either (#623). A backend is admitted per engine class by the public
  validation of that class run on it and by the work record within Poisson noise: :func:`parity_report`. One
  case is a hard invariant rather than a rate (``strands_void``, #675): starts in the void beyond a tube's own
  gather reach, held to per-save containment -- a gather read as unbounded where it found nothing leaks a
  walker through the next wall it never actually tested.
- **Determinism per (seed, architecture).** A trajectory is a function of the key; the pack says which backend and
  architecture walked it (``work["backend"]`` and the run record).
- **Refusal over fallback.** A backend that does not implement an engine class, a tier or a wall says so by name
  (:meth:`Backend.refuses`); the producer never falls back to ``"jax"`` silently.

The public package ships one backend, ``"jax"`` (the kernels in :mod:`.core`), which is the default. Another is
installed as a package exposing the entry point group ``dmipy_sim.backends``, or passed as an object. A user who
has no other backend never sees the option do anything, and a name that is not installed is refused by name.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

JAX = "jax"
ENTRY_POINT_GROUP = "dmipy_sim.backends"


class FieldSampling(NamedTuple):
    """In-walk sampling of a strand field (the paper's C3): what a backend evaluates along each walker's path."""
    basis: object             #: the :class:`~dmipy_sim.fields.strand_field.StrandFieldBasis` (its tables, switch, far grid, gate)
    sample_every: int         #: the field's own save grid: a sample at every that-many-th save (sample 0 the start's)
    reuse_intervals: int      #: the per-walker segment list is gathered every this many save intervals
    radius_m: float           #: the list's gather radius: the basis's reach plus the margin the walkers travel in between
    list_k: int               #: the list's width the producer sized from the start positions (a backend may widen)


class WalkRequest(NamedTuple):
    """What the producer has resolved before the walk; what a backend walks."""
    geometry: object          #: the substrate, resolved (its constants are the physics)
    n_t: int                  #: saves, save 0 the start
    dt_save: float            #: seconds between saves
    sub_steps: int            #: fixed-length steps per save interval
    dt_sim: float             #: the step's duration; the step length is ``sqrt(6 D dt_sim)``
    diffusivity: float        #: m^2/s
    record: bool              #: record the boundary local time and the compartment beside the positions
    kappa_over_D: float       #: the membrane's permeability over D; 0 for an impermeable walk
    count_walls: bool         #: count hits and crossings (exhaustion is always reported)
    bounce_budget: Optional[int]   #: the budget the geometry's loop runs at for this walk
    seed: int                 #: the producer's master seed (the per-walker keys derive from it)
    stepping: object = None   #: an adaptive walk's :class:`~dmipy_sim.engine.adaptive.AdaptivePlan`; None for the fused walk
    field: object = None      #: a :class:`FieldSampling` when the walk samples a strand field; None otherwise


class ParityCase(NamedTuple):
    """One substrate :func:`parity_cases` compares a backend on. ``case()`` builds the geometry -- the same call
    a bare builder answered, so every existing caller of ``parity_cases()[name]()`` is unaffected. A rate case
    is a case with nothing past ``build``; a hard-invariant case (``strands_void``) also carries ``starts``, the
    walk's own seeding (:func:`parity_report` draws none of its own when this is set), and the walk's own scale
    (``n_walkers``, ``T_max``, ``dt_save``, ``seed`` -- each ``None`` falls back to :func:`parity_report`'s own
    keyword, except ``starts`` callers, which always get the case's own values)."""
    build: object             #: zero-arg callable returning the geometry
    starts: object = None     #: ``(n, seed) -> r0 (n, 3) float32``, or None: :func:`parity_report` draws its own
    n_walkers: object = None  #: overrides :func:`parity_report`'s ``n_walkers`` when ``starts`` is set
    T_max: object = None      #: overrides :func:`parity_report`'s ``T_max`` when ``starts`` is set
    dt_save: object = None    #: overrides :func:`parity_report`'s ``dt_save`` when ``starts`` is set
    seed: object = None       #: overrides :func:`parity_report`'s ``seed`` when ``starts`` is set

    def __call__(self):
        return self.build()


class WalkBatch(NamedTuple):
    """What a backend returns for a batch of walkers: the JAX kernels' outputs, as arrays."""
    positions: np.ndarray             #: (n, n_t, 3) float32, save 0 the start
    boundary_local_time: object       #: (n, n_t) float32 at rho/D = 1, or None when not recorded
    compartment: object               #: (n, n_t) float32 fractional occupancy of pool 1 (int labels when impermeable), or None
    illegal: np.ndarray               #: (n,) int32 refused walker-steps per walker
    work: tuple                       #: (hits | None, crossings | None, exhausted) per walker, int32 arrays
    counters: object = None           #: what else the backend counted over the batch (an adaptive walk's
                                      #: ``dict(n_free=, n_kernel_steps=)``), or None when it reports none
    field_samples: object = None      #: (n, n_tf, 13) float32 the bare interval means of the field's channels at the
                                      #: sampled saves (sample 0 the start's reading; no mean subtracted), or None


class Backend:
    """The protocol. A backend subclasses or duck-types this: ``name``, :meth:`refuses`, :meth:`walk_batch`."""
    name = "?"

    def refuses(self, request: WalkRequest) -> Optional[str]:
        """Why this backend cannot walk ``request`` (an engine class, a tier, a wall it does not implement), or
        None. The producer raises with the reason; it never falls back."""
        return None

    def walk_batch(self, request: WalkRequest, r0: np.ndarray, keys: np.ndarray, *, out=None) -> WalkBatch:
        """Walk the ``(n, 3)`` float32 starts ``r0`` from their ``(n, 2)`` uint32 keys. ``out``, when given, is a
        dict of the producer's own contiguous float32 arrays for this batch -- ``positions`` (n, n_t, 3),
        ``boundary_local_time`` and ``compartment`` (n, n_t), ``field_samples`` (n, n_tf, 13) -- that a backend
        may write into instead of allocating its own (the batch then holds those arrays); a backend that ignores
        ``out`` returns its own and the producer copies, at 7.6 GB per 100k DiSCo walkers (dmipy-sim-cuda#33)."""
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
        "strands_void": ParityCase(void_build, starts=void_starts, n_walkers=4096, T_max=40e-3, dt_save=0.5e-3, seed=5),
    }


def void_invariant(g, w, *, wall_margin_m=0.5e-6, near_wall_fraction=0.02):
    """The void case's invariant (dmrai-lab/dmipy-sim#675): every recorded save of every walker in ``w`` lies
    outside every tube of ``g`` -- tested by the geometry's own exact classifier
    (:meth:`~dmipy_sim.geometry.base.Geometry.classify_positions_exact` / :meth:`~dmipy_sim.geometry.base.Geometry.pool_of`),
    never a hand-rolled radial formula -- no walker was granted an illegal crossing, and more than
    ``near_wall_fraction`` of the walkers come within ``wall_margin_m`` of a wall at some save (the geometry's own
    :meth:`~dmipy_sim.geometry.curved_cylinder.PackedCurvedCylinders.wall_scales`; the premise -- enough walkers
    reach the wall for the first two checks to mean something). Returns ``{check: {..., ok}}``; a failure here is
    a hard violation of the backend under test, never a rate comparison."""
    n, n_t, _ = w.positions.shape
    flat = np.asarray(w.positions, np.float64).reshape(-1, 3)
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


def parity_report(backend, cases=None, *, n_walkers=4000, T_max=2e-3, dt_save=0.5e-3, seed=11, sigma=4.0, overdispersion=2.5):
    """Compare ``backend`` with the JAX kernels on ``cases`` (default :func:`parity_cases`): the same starts, the
    same sub-step and saves, counting on. Per case: the hit and crossing counts within ``sigma`` standard deviations,
    where the standard deviation is ``overdispersion * sqrt(a + b)``: a walker near a wall hits it on consecutive
    steps, so the per-walker counts are clustered and the total's spread is wider than Poisson (measured on the
    sphere and the ellipsoid at 20k walkers: seed-to-seed swings of 0.1-0.4 % against a Poisson 0.12 %, i.e. 2-3x);
    the exhausted walker-steps and the refused ones (``illegal_crossings``) compared the same way, since both are
    rates of the geometry at its budget and not of the backend (a permeable icosphere mesh of 320 facets exhausts
    0-5 of 1.07 M walker-steps per seed on the JAX kernels themselves -- the wedge between a facet and the voxel
    face it touches); every walker on its side (the classifier at the saves agrees with the start's pool unless a
    crossing was granted), and the perpendicular PGSE signal of the two walks within ``sigma`` of their combined
    standard errors. A case is a rate comparison unless it declares its own starts (:attr:`ParityCase.starts`,
    e.g. ``strands_void``): there the backend under test walks those starts alone, at the case's own scale (its
    ``n_walkers``/``T_max``/``dt_save``/``seed``, each falling back to this function's keyword of the same name
    when the case leaves it ``None``), and is held to :func:`void_invariant`, a hard pass/fail with no JAX side
    to compare against. A case the backend refuses is recorded with its reason and not failed: refusal is the
    contract. Returns ``[{case, ok, ...}]``; the list is what a backend's tests assert on."""
    import dmipy_sim as d
    from .core import simulate_trajectories
    from ..replay.trajectories import replay
    from ..sequences import pgse
    out = []
    for name, case in (cases or parity_cases()).items():
        g = case(); g.count_walls = True
        if case.starts is not None:
            nw = n_walkers if case.n_walkers is None else case.n_walkers
            Tm = T_max if case.T_max is None else case.T_max
            ds = dt_save if case.dt_save is None else case.dt_save
            sd = seed if case.seed is None else case.seed
            try:
                w = simulate_trajectories(nw, 2e-9, g, Tm, ds, seed=sd, tiers="all", walker_batch_size=nw,
                                          r0=case.starts(nw, sd), backend=backend)
            except ValueError as e:
                if "refuses" in str(e) or "does not" in str(e):
                    out.append(dict(case=name, ok=None, refused=str(e))); continue
                raise
            checks = void_invariant(g, w)
            out.append(dict(case=name, ok=all(c["ok"] for c in checks.values()), checks=checks))
            continue
        ref = simulate_trajectories(n_walkers, 2e-9, g, T_max, dt_save, seed=seed, tiers="all", walker_batch_size=n_walkers)
        g2 = case(); g2.count_walls = True
        try:
            w = simulate_trajectories(n_walkers, 2e-9, g2, T_max, dt_save, seed=seed, tiers="all", walker_batch_size=n_walkers,
                                      r0=np.asarray(ref.positions[:, 0]), backend=backend)
        except ValueError as e:
            if "refuses" in str(e) or "does not" in str(e):
                out.append(dict(case=name, ok=None, refused=str(e))); continue
            raise
        rec = dict(case=name, ok=True, checks={})
        # the work record
        steps = ref.work["walker_steps"]
        counts = {key: (ref.work[key], w.work[key]) for key in ("n_hits", "n_crossings", "exhausted_steps")}
        counts["illegal"] = (int(np.asarray(ref.illegal_crossings).sum()), int(np.asarray(w.illegal_crossings).sum()))
        for key, (a, b) in counts.items():
            z = abs(a - b) / (overdispersion * max(np.sqrt(a + b), 1.0))
            rec["checks"][key] = dict(jax=int(a), backend=int(b), z=float(z), ok=bool(z <= sigma))
        # confinement: the saves' pools against the start's, where no crossing was granted
        pool0 = np.asarray(g2.classify_positions_exact(np.asarray(w.positions[:, 0])))
        poolT = np.asarray(g2.classify_positions_exact(np.asarray(w.positions[:, -1])))
        crossings_allowed = g2.permeability is not None and float(g2.permeability) > 0
        leaked = int((g2.pool_of(poolT) != g2.pool_of(pool0)).sum())
        rec["checks"]["confinement"] = dict(changed_pool=leaked, ok=(crossings_allowed or leaked == 0))
        # the signal
        seq = pgse([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 0.3e-3, 1.0e-3, bvalues=[2e9] * 3, TE=T_max, n_t=400, slew_rate=np.inf)
        G = np.asarray(seq.G_eff)
        def sig(p):
            phi, log_w, _ = replay(np.asarray(p, np.float32), dt_save, G, seq.dt, return_walker_signals=True)
            ws = np.exp(np.asarray(log_w, np.float64)) * np.cos(np.asarray(phi, np.float64))
            return ws.mean(1), ws.std(1, ddof=1) / np.sqrt(ws.shape[1])
        sa, ea = sig(ref.positions); sb, eb = sig(w.positions)
        z = np.abs(sa - sb) / np.sqrt(ea ** 2 + eb ** 2)
        rec["checks"]["pgse"] = dict(jax=sa.tolist(), backend=sb.tolist(), z=z.tolist(), ok=bool((z <= sigma).all()))
        rec["ok"] = all(c["ok"] for c in rec["checks"].values())
        out.append(rec)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The request directory: dmipy-sim's side of the backend protocol (dmrai-lab/dmipy-sim#691). A backend
# executes a WalkRequest; the strands engine's request DIRECTORY (request.txt + the strand/adaptive/field
# tables) is serialised once, here, from the geometry's/plan's/basis's own arrays -- never read back by a
# backend's own ctypes struct, which is a different consumer of the same numbers (a live library call, not a
# file). dmipy_sim_cuda.cli.write_request calls write_request_directory for the strands engine; a substrate
# kit (dmipy_sim.fill.kit) builds the same files from a template, never installing dmipy_sim_cuda to do it.
# ─────────────────────────────────────────────────────────────────────────────

#: the strand-walk tables of one pool's geometry, the file names a request directory writes them under
STRAND_ARRAY_NAMES = ("A.f32", "AB.f32", "AB2.f32", "rr.f32", "tube.i32", "cell_off.i32", "cell_ids.i32")
#: the field basis's own tables, shared by every pool that samples it
FIELD_ARRAY_NAMES = ("fA.f32", "fAB.f32", "fAB2.f32", "fa.f32", "fb.f32", "fsid.i32", "fcell_off.i32", "fcell_ids.i32", "fbox.i16")
#: the request-struct fields that stay in a kit's template (everything but ``n`` and ``seed``, which are a batch's own)
REQUEST_TEMPLATE_KEYS = ("n_t", "sub_steps", "step_l", "kappa_over_D", "record", "count", "budget")
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


def field_struct_fields(basis, radius_m, sample_every, reuse, list_k, n_tf):
    """The backend's ``DscField`` fields EXCEPT ``far_dims``/``far_origin`` (the crop, a batch's own) -- built
    from ``basis``'s own arrays and the sampling settings, with no ``r0``."""
    _, fcell_ids = csr_cells(np.asarray(basis._CELL))
    far = basis.far
    return dict(n_seg=int(basis.n_segments), nnz=int(fcell_ids.shape[0]), dims=[int(x) for x in basis._dims],
               gmin=[_f32(x) for x in np.asarray(basis._gmin)], cs=_f32(basis._CS), radius=_f32(radius_m),
               reach=_f32(basis.gather_radius_m), gate_radii=_f32(basis.NEAREST_GATE_RADII),
               has_far=int(far is not None), near_m=_f32(far.near_m if far is not None else 0.0),
               blend_m=_f32(far.blend_m if far is not None else 0.0), far_h=_f32(far.spacing_m if far is not None else 0.0),
               sample_every=int(sample_every), reuse=int(reuse), list_k=max(1, int(list_k)),
               n_tf=int(n_tf), segments_max=int(basis.segments_max))


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


def request_struct_fields(*, n_t, sub_steps, step_l, kappa_over_D, record, count, budget):
    """The backend's ``DscRequest`` fields EXCEPT ``seed`` (a batch's own)."""
    return dict(n_t=int(n_t), sub_steps=int(sub_steps), step_l=_f32(step_l), kappa_over_D=_f32(kappa_over_D),
               record=int(bool(record)), count=int(bool(count)), budget=int(budget))


def crop_far_grid(values, origin_m, spacing_m, r0, n_t, dt_save, diffusivity):
    """The far grid cropped to what a batch of starts ``r0`` can reach: the starts' bounding box plus eight
    sigma of the walk's excursion and three nodes, so a shard of a large substrate uploads a crop, never the
    whole grid. Returns ``(crop.view(uint16), far_dims, far_origin)``."""
    values = np.asarray(values)
    N = np.asarray(values.shape[:3])
    h = float(spacing_m)
    T = float(dt_save) * (int(n_t) - 1)
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


def write_request_directory(request: WalkRequest, r0, keys, out_dir, *, far_crop: bool = True) -> Path:
    """The request directory of a live, resolved ``WalkRequest`` whose geometry is the strands engine:
    ``request.txt`` (the ``DscRequest`` fields, the engine's name, the ``DscStrands`` fields, and -- when the
    request carries them -- the ``DscAdaptive`` and ``DscField`` fields, each prefixed ``adaptive_``/``field_``),
    ``r0.f32``, ``keys.u32`` and the strand/field tables beside it (the far grid CROPPED to what ``r0`` can
    reach, :func:`crop_far_grid`). This is dmipy-sim's side of the backend protocol: every value comes from the
    geometry's/plan's/basis's own arrays, with the float32 truncation the backend's ctypes request structs apply
    on assignment (:func:`strands_struct_fields`, :func:`adaptive_struct_fields`, :func:`field_struct_fields`) --
    so this is exactly what ``dmipy_sim_cuda.cli.write_request`` writes for the strands engine, byte for byte,
    without that package installed. ``classify`` is the CUDA CLI's own flag and is not written here.
    ``far_crop=False`` leaves ``far_dims``/``far_origin`` at zero and writes no ``far.u16``, for a caller with
    no walkers yet to crop the far grid around."""
    from ..geometry.curved_cylinder import PackedCurvedCylinders
    g = request.geometry
    if not isinstance(g, PackedCurvedCylinders):
        raise ValueError(f"write_request_directory serialises the strands engine only, not {type(g).__name__}")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    r0 = np.ascontiguousarray(r0, np.float32); n = int(r0.shape[0])
    struct, arrays = strands_struct_fields(g)
    lines = dict(engine="strands", n=n)
    lines.update(request_struct_fields(n_t=request.n_t, sub_steps=request.sub_steps,
                                       step_l=float(np.sqrt(6.0 * request.diffusivity * request.dt_sim)),
                                       kappa_over_D=request.kappa_over_D, record=request.record, count=request.count_walls,
                                       budget=(int(g.bounce_loop.budget) if g.bounce_loop is not None else 0)))
    lines["seed"] = int(request.seed) & 0xFFFFFFFFFFFFFFFF
    lines.update(struct)
    if request.stepping is not None:
        lines["adaptive"] = 1
        lines.update({f"adaptive_{k}": v for k, v in adaptive_struct_fields(request.stepping, g._Rmax).items()})
    if request.field is not None:
        f = request.field; basis = f.basis
        n_tf = len(range(0, int(request.n_t), int(f.sample_every)))
        lines["field"] = 1
        lines.update({f"field_{k}": v for k, v in field_struct_fields(basis, f.radius_m, f.sample_every, f.reuse_intervals, f.list_k, n_tf).items()})
        arrays.update(field_tables(basis))
        far_dims, far_origin = [0, 0, 0], [0.0, 0.0, 0.0]
        if basis.far is not None and far_crop:
            crop, far_dims, far_origin = crop_far_grid(basis.far.values, basis.far.origin_m, basis.far.spacing_m,
                                                       r0, request.n_t, request.dt_save, request.diffusivity)
            arrays["far.u16"] = crop
        lines["field_far_dims"] = far_dims; lines["field_far_origin"] = far_origin
    (out / "request.txt").write_text("".join(f"{k}={format_request_value(v)}\n" for k, v in lines.items()))
    r0.tofile(out / "r0.f32")
    np.ascontiguousarray(keys, np.uint32).reshape(n, -1)[:, :2].copy().tofile(out / "keys.u32")
    for name, a in arrays.items():
        np.ascontiguousarray(a).tofile(out / name)
    return out
