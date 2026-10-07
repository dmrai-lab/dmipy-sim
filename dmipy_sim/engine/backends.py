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
    """The substrates a backend is compared on, each at its own permeability: ``{name: builder}``, ``builder()``
    a zero-arg callable returning the geometry. A case that needs starts and a walk scale of its own rather than
    :func:`parity_report`'s defaults (``strands_void``) sets them as attributes on the geometry it returns --
    ``parity_starts(n, seed) -> r0`` and, optionally, ``parity_overrides`` (``n_walkers``, ``T_max``, ``dt_save``,
    ``seed``) -- which :func:`parity_report` reads instead of drawing its own."""
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

    def void():
        # one tube in a 200 um box, starts in the void 2.5 cells from its wall: a gather whose 27-cell
        # neighbourhood is empty must still bound the walk by the nearest wall it CAN reach, not by the box
        # (dmipy-sim-cuda#36). The case carries its own starts (`parity_starts`) and walk scale
        # (`parity_overrides`): a rate comparison at `parity_report`'s default scale would barely move the
        # walkers, so the invariant needs its own T_max and walker count.
        R, L = 2e-6, 200e-6
        xt = yt = 100e-6
        g = d.PackedCurvedCylinders([np.array([[xt, yt, 0.0], [xt, yt, L]])], [R], interior=False,
                                    box=(np.zeros(3), np.full(3, L)))

        def starts(n, seed):
            rng = np.random.default_rng(seed)
            r0 = np.column_stack([np.full(n, xt - R - 2.5 * g.cell_size),
                                  rng.uniform(yt - 5e-6, yt + 5e-6, n),
                                  rng.uniform(20e-6, 180e-6, n)])
            return r0.astype(np.float32)

        g.parity_starts = starts
        g.parity_overrides = dict(n_walkers=4096, T_max=40e-3, dt_save=0.5e-3)
        return g

    return {
        "sphere": lambda: d.Sphere(R),
        "sphere_perm": lambda: d.Sphere(R, permeability=1e-5),
        "cylinder": lambda: d.Cylinder(R, (0, 0, 1)),
        "cylinder_perm": lambda: d.Cylinder(R, (0, 0, 1), permeability=1e-5),
        "ellipsoid": lambda: d.Ellipsoid((R, 0.7 * R, 0.5 * R)),
        "packed_cyl": lambda: packed(d.PackedCylinders),
        "packed_cyl_perm": lambda: packed(d.PackedCylinders, permeability=2e-5),
        "packed_sph": lambda: packed(d.PackedSpheres),
        "packed_sph_perm": lambda: packed(d.PackedSpheres, permeability=2e-5),
        "label_slab": lambda: label(),
        "label_slab_perm": lambda: label(permeability=1e-5, pools={0: "free", 1: "grain"}),
        "mesh": lambda: mesh(),
        "mesh_perm": lambda: mesh(permeability=1e-5),
        "strands_intra": lambda: strands(True),
        "strands_extra": lambda: strands(False),
        "strands_void": void,
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
    standard errors. A case is a rate comparison unless it declares its own starts (``geometry.parity_starts``,
    e.g. ``strands_void``): there the backend under test walks those starts alone (at ``geometry.parity_overrides``'
    scale, when given) and is held to :func:`void_invariant`, a hard pass/fail with no JAX side to compare against.
    A case the backend refuses is recorded with its reason and not failed: refusal is the contract. Returns
    ``[{case, ok, ...}]``; the list is what a backend's tests assert on."""
    import dmipy_sim as d
    from .core import simulate_trajectories
    from ..replay.trajectories import replay
    from ..sequences import pgse
    out = []
    for name, build in (cases or parity_cases()).items():
        g = build(); g.count_walls = True
        starts = getattr(g, "parity_starts", None)
        if starts is not None:
            overrides = getattr(g, "parity_overrides", {})
            nw = overrides.get("n_walkers", n_walkers); Tm = overrides.get("T_max", T_max)
            ds = overrides.get("dt_save", dt_save); sd = overrides.get("seed", seed)
            try:
                w = simulate_trajectories(nw, 2e-9, g, Tm, ds, seed=sd, tiers="all", walker_batch_size=nw,
                                          r0=starts(nw, sd), backend=backend)
            except ValueError as e:
                if "refuses" in str(e) or "does not" in str(e):
                    out.append(dict(case=name, ok=None, refused=str(e))); continue
                raise
            checks = void_invariant(g, w)
            out.append(dict(case=name, ok=all(c["ok"] for c in checks.values()), checks=checks))
            continue
        ref = simulate_trajectories(n_walkers, 2e-9, g, T_max, dt_save, seed=seed, tiers="all", walker_batch_size=n_walkers)
        g2 = build(); g2.count_walls = True
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
