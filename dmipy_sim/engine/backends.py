"""Backends: another execution of the same physics, resolved by name (dmrai-lab/dmipy-sim#635).

A backend walks a batch of walkers the way the JAX kernels of :mod:`.core` walk it -- the same geometry, the same
starts and keys, the same sub-step and save grid -- and returns what they return. dmipy-sim stays the reference
physics: a backend reads every constant from the geometry it is handed and carries none of its own.

What a backend signs
--------------------
- **One physics.** The rule is dmipy-sim's; the backend executes it.
- **Parity, not identity.** A different RNG stream and FMA policy make bit-identity impossible, and it is not stable
  across XLA compilations of one program either (#623). A backend is admitted per engine class by the public
  validation of that class run on it and by the work record within Poisson noise: :func:`parity_report`.
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


class WalkBatch(NamedTuple):
    """What a backend returns for a batch of walkers: the JAX kernels' outputs, as arrays."""
    positions: np.ndarray             #: (n, n_t, 3) float32, save 0 the start
    boundary_local_time: object       #: (n, n_t) float32 at rho/D = 1, or None when not recorded
    compartment: object               #: (n, n_t) float32 fractional occupancy of pool 1 (int labels when impermeable), or None
    illegal: np.ndarray               #: (n,) int32 refused walker-steps per walker
    work: tuple                       #: (hits | None, crossings | None, exhausted) per walker, int32 arrays


class Backend:
    """The protocol. A backend subclasses or duck-types this: ``name``, :meth:`refuses`, :meth:`walk_batch`."""
    name = "?"

    def refuses(self, request: WalkRequest) -> Optional[str]:
        """Why this backend cannot walk ``request`` (an engine class, a tier, a wall it does not implement), or
        None. The producer raises with the reason; it never falls back."""
        return None

    def walk_batch(self, request: WalkRequest, r0: np.ndarray, keys: np.ndarray) -> WalkBatch:
        """Walk the ``(n, 3)`` float32 starts ``r0`` from their ``(n, 2)`` uint32 keys."""
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
    """The substrates a backend is compared on, each at its own permeability: ``{name: builder}``."""
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
    }


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
    standard errors. A case the backend refuses is recorded with its reason and not failed: refusal is the
    contract. Returns ``[{case, ok, ...}]``; the list is what a backend's tests assert on."""
    import dmipy_sim as d
    from .core import simulate_trajectories
    from ..replay.trajectories import replay
    from ..sequences import pgse
    out = []
    for name, build in (cases or parity_cases()).items():
        g = build(); g.count_walls = True
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
