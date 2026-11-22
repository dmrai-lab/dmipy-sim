"""Myelinated geometries: concentric intra / myelin / extra compartments.

These are NOT stepped by `reflect`: their compartment is carried state, so the engine
routes them to the fused kernels in `physics` (see each class's `reflect`, which raises).
"""
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ._boundary import (vdot, keep_side_radial, specular, off_wall, ray_sphere_t, transmit_probability, trial_granted,
                        BounceLoop, TrialLog, wall_push, bounce_budget)

from .base import Geometry, LengthScales, acquisition_rotation, placement_seed
from ..compartments import Compartments, Pool

_TINY = 1e-30
#: The pool after a crossing of code 1..6 (index 0 unused): 1 lumen -> sheath and 2 sheath -> lumen at an axolemma,
#: 3 sheath -> extra and 4 extra -> sheath at a sheath's outer wall, 5 lumen -> extra and 6 extra -> lumen at a plain
#: cylinder's wall.
_POOL_AFTER_CROSSING = (0, 2, 1, 0, 2, 0, 1)


class ConcentricHit(NamedTuple):
    """Outcome of one step of :func:`concentric_wall_kernel`: the concentric substrates' `WallHit`, with the pool
    and axon the walker owns after the step and the four boundary local-time channels in place of one."""
    xy: jnp.ndarray        #: end position in the cross-section plane
    pool: jnp.ndarray      #: int32 pool after the step (0 extra, 1 intra, 2 myelin)
    k: jnp.ndarray         #: int32 owning / nearest axon after the step
    chan: jnp.ndarray      #: (4,) unit boundary local-time channels, ``-2 d_perp`` per wall and side met
    dlog_rho: jnp.ndarray  #: surface log-weight increment under the per-axon rho weights
    crossed: jnp.ndarray   #: bool -- a crossing was granted
    n_hits: object         #: int32 reflections plus the granted crossings; None unless the kernel counts
    exhausted: jnp.ndarray #: bool -- the bounce budget ran out with path untested
    trials: object = None  #: the step's plain-wall trials (:class:`~dmipy_sim.geometry._boundary.TrialLog`); None
                           #: unless the kernel logs them



class WaterDomain(NamedTuple):
    """One water domain of a :class:`PackedMyelinatedCylinders` cell (:meth:`PackedMyelinatedCylinders.water_domains`),
    per unit length of the cylinders."""
    pools: tuple           #: the pools whose water the domain holds, by name
    cylinders: tuple       #: the cylinders whose lumen is in the domain, by index
    volume: float          #: m^2: the domain's area in the cross-section
    surface: float         #: m: every face a walker of the domain meets, once per face and side
    surface_to_volume: float  #: 1/m: ``surface / volume``; a lumen's is the circle's ``2 / radius``
    seeded_area: float     #: m^2: the area of its seeded pools

def _refuse_schedule(owner, **kappas):
    """Refuse a :class:`~dmipy_sim.geometry.PermeabilitySchedule` on a myelinated substrate by name: its walls are
    stepped by its own three-compartment kernel, whose permeability per axon is constant in time."""
    from .base import PermeabilitySchedule
    for name, k in kappas.items():
        if isinstance(k, PermeabilitySchedule):
            raise NotImplementedError(f"{owner}({name}=) takes a permeability per axon constant in time; its walls are "
                                      f"stepped by the myelinated kernel, which does not read a PermeabilitySchedule")

#: How a walker's permeability factor ``f_w`` enters a :class:`PackedMyelinatedCylinders` walk (the walk's
#: ``work["permeability_factor"]["rule"]``): the product with the plain cylinder's stated ``kappa_outer`` in m/s, in float32,
#: then the division by the diffusivity of the pool the walker leaves (:func:`concentric_wall_kernel`).
PERMEABILITY_FACTOR_RULE = ("PackedMyelinatedCylinders: walker w meets plain cylinder k's wall at "
                            "kappa_w = float32(float32(kappa_outer[k]) * float32(permeability_factor[w])) (m/s); the "
                            "trial's kappa/D is float32(kappa_w / D), D the float32 diffusivity of the pool it leaves; "
                            "the sheathed cylinders' walls keep their stated kappa_inner and kappa_outer")


def pin_to_pool(xy, q, R_in, R_out, pool, nudge):
    """``xy`` pulled strictly onto the side of the concentric walls (radii ``R_in < R_out``) that ``pool`` (0 extra,
    1 lumen, 2 sheath) occupies, ``q`` its offset from the axon's centre: the side rule the fused step kernel applies
    at the end of every step, and the one the seeders apply to every accepted point."""
    xy, _ = keep_side_radial(xy, q, R_in, True, nudge, active=pool == 1)
    xy, _ = keep_side_radial(xy, q, R_in, False, nudge, active=pool == 2)
    xy, _ = keep_side_radial(xy, q, R_out, True, nudge, active=pool == 2)
    xy, _ = keep_side_radial(xy, q, R_out, False, nudge, active=pool == 0)
    return xy


def _pin_seeds(xy, centers, L, R_in, R_out, pool, nudge):
    """``(n, 2)`` float32 seeds ``xy`` pulled strictly onto their pool's side of their axon's walls.

    A seeder accepts a point by a float64 test and casts it to float32; a draw within half a float32 ulp of a wall
    can round onto or past it, where the first step reads it as the neighbouring pool for good. ``centers``,
    ``R_in``, ``R_out`` and ``pool`` are per point; ``L`` is the periodic cell edge (``None`` when there is none)."""
    f32 = jnp.float32

    def _one(p, c, ri, ro, pl):
        q = p - c
        if L is not None:
            q = q - f32(L) * jnp.floor(q / f32(L) + f32(0.5))
        return pin_to_pool(p, q, ri, ro, pl, f32(nudge))

    return np.asarray(jax.vmap(_one)(jnp.asarray(xy, f32), jnp.asarray(centers, f32), jnp.asarray(R_in, f32),
                                     jnp.asarray(R_out, f32), jnp.asarray(pool, jnp.int32)))


#: The byte cap of a wall-search grid's cell table (int32 entries): a layout priced above it is not built.
WALL_GRID_BYTE_CAP = 64 * 2 ** 20
#: The cells per side a wall-search grid is priced at: ``ceil(sqrt(N_real) f)`` for each ``f`` here.
WALL_GRID_CELL_FACTORS = (0.5, 1.0, 2.0)
#: The float32 pad of the inclusion rule, in units of ``eps32 L`` (``eps32 = 2^-23``, ``L`` the cell side).
WALL_GRID_PAD_EPS = 16.0


def _grid_reach(half_diagonal, pad):
    """The margin (m) of a grid cell's inclusion rule beyond the ``n_cand``-th smallest wall distance at its centre:
    twice the cell's half-diagonal plus the float32 pad (:func:`_wall_grid_lists`)."""
    return 2.0 * (half_diagonal + pad)


def _wall_grid_lists(c, R, ids, L, G, n_cand, gmin, cs, pad):
    """The cell table of a wall-search grid of ``G x G`` cells over the periodic cell of side ``L``.

    ``c`` ``(n, 2)`` and ``R`` ``(n,)`` are the real cylinders' centres and outer radii (float64 values of the
    kernel's float32 operands), ``ids`` their indices in the kernel's arrays (ascending); ``gmin`` and ``cs`` the
    float32 origin and cell size the kernel's lookup reads, ``pad`` the float32 pad (m). Cell ``(i, j)`` has the
    centre ``x0 = gmin + ((i, j) + 1/2) cs``; its list is every cylinder whose exact minimum-image wall distance
    ``d_k(x0) = |x0 - c_k|_per - R_k`` is at most ``D + _grid_reach(h, pad)``, ``D`` the ``n_cand``-th smallest
    ``d_k(x0)`` and ``h = cs / sqrt(2)`` the cell's half-diagonal, in ascending index.

    The cylinders within a reach ``r`` of the cell are found by the shared bucketing
    (:func:`._grid.bucket_periodic_plane`) of each cylinder's box ``c_k +- (R_k + r)``, split at the periodic seam
    into the intervals of :func:`._grid.periodic_spans`; ``r`` grows until every cell's ``D + _grid_reach`` is within it, so
    the bucket holds every cylinder the rule lists. Returns ``(table (G * G, M) int32, M)``, ``-1`` padding a row."""
    from ._grid import bucket_periodic_plane
    n = len(R)
    h = float(cs) / np.sqrt(2.0)
    extra = _grid_reach(h, pad)
    ii = np.arange(G)
    x0 = float(gmin) + (ii + 0.5) * float(cs)
    X, Y = np.meshgrid(x0, x0, indexing="ij")
    cells = np.stack([X.ravel(), Y.ravel()], 1)           # (G*G, 2), cell id = i * G + j
    slack = 1e-9 * L
    r = extra + np.sqrt(n_cand * L * L / (np.pi * n)) + float(np.max(R))
    while True:
        half = R + r + slack
        table = bucket_periodic_plane(c, half, L, G)
        C = table.shape[1]
        valid = table >= 0
        cand = np.where(valid, table, 0)
        q = cells[:, None, :] - c[cand]
        q -= L * np.round(q / L)
        d = np.where(valid, np.hypot(q[..., 0], q[..., 1]) - R[cand], np.inf)
        if C >= n_cand:
            D = np.partition(d, n_cand - 1, axis=1)[:, n_cand - 1]
        else:
            D = np.full(G * G, np.inf)
        need = D + extra
        if np.all(need <= r):
            break
        r = float(np.max(need)) if np.all(np.isfinite(need)) else 2.0 * r
    keep = d <= need[:, None]
    M = int(keep.sum(1).max())
    pos = np.argsort(~keep, axis=1, kind="stable")[:, :M]
    rows = np.arange(G * G)[:, None]
    out = np.where(keep[rows, pos], ids[cand[rows, pos]], -1).astype(np.int32)
    return out, M


def _wall_search(search, centers, outer, L, N, n_cand):
    """The candidate search :func:`concentric_wall_kernel` runs: ``(name, grid, report, reason)``.

    ``name`` is ``"grid"`` or ``"dense"``; ``grid`` the ``(table, gmin, cs, G)`` of the grid (None for the dense
    search), ``report`` its record (None for the dense search) and ``reason`` the sentence that states why the search
    is the one it is. The rule: no grid without a periodic cell (``L`` None), when the candidate count is the whole
    array (``n_cand == N``, where the dense search takes every slot in index order) or when fewer than ``n_cand``
    cylinders are real (the dense search's candidates then include padding slots). Otherwise the grid is priced at
    the cells per side of :data:`WALL_GRID_CELL_FACTORS` and the layout of the smallest list length ``M`` within
    :data:`WALL_GRID_BYTE_CAP` is taken (ties: the fewer bytes); ``search=None`` runs it when ``M`` is below the
    number of real cylinders (``outer > 0``), ``"grid"`` runs it whatever ``M``, ``"dense"`` never.
    ``"grid"`` where no grid can be built is refused, naming the reason."""
    if search not in (None, "dense", "grid"):
        raise ValueError(f"search is None (the stated rule), 'dense' or 'grid'; got {search!r}")
    outer64 = np.asarray(outer, np.float32).astype(np.float64)
    real = outer64 > 0
    n_real = int(real.sum())
    if L is None:
        reason = "no periodic cell (L is None): the grid covers a periodic cell"
    elif n_cand >= N:
        reason = f"the candidates are every one of the {N} slots (n_cand = N): there is nothing to search"
    elif n_real < n_cand:
        reason = (f"{n_real} real cylinders, fewer than the {n_cand} candidates: the dense search's candidates "
                  f"include padding slots, which no cell list holds")
    else:
        reason = None
    if reason is not None:
        if search == "grid":
            raise ValueError(f"concentric_wall_kernel(search='grid') is refused: {reason}")
        return "dense", None, None, reason
    if search == "dense":
        return "dense", None, None, "search='dense' was given"
    L32 = np.float32(L)
    L64 = float(L32)
    gmin = np.float32(-0.5) * L32
    pad = WALL_GRID_PAD_EPS * float(np.finfo(np.float32).eps) * L64
    c = np.asarray(centers, np.float32).astype(np.float64)[real]
    ids = np.flatnonzero(real)
    options, best = [], None
    for f in WALL_GRID_CELL_FACTORS:
        G = max(1, int(np.ceil(np.sqrt(n_real) * f)))
        cs = np.float32(L32 / np.float32(G))
        table, M = _wall_grid_lists(c, outer64[real], ids, L64, G, n_cand, gmin, cs, pad)
        nbytes = int(table.size * 4)
        options.append(dict(factor=float(f), cells_per_side=G, list_length=M, bytes=nbytes))
        if nbytes <= WALL_GRID_BYTE_CAP and (best is None or (M, nbytes) < (best[1], best[2])):
            best = (table, M, nbytes, gmin, cs, G)
    if best is None:
        reason = f"every priced layout is above the byte cap {WALL_GRID_BYTE_CAP} B: {options}"
        if search == "grid":
            raise ValueError(f"concentric_wall_kernel(search='grid') is refused: {reason}")
        return "dense", None, None, reason
    table, M, nbytes, gmin, cs, G = best
    report = dict(cells_per_side=G, list_length=M, bytes=nbytes, byte_cap=WALL_GRID_BYTE_CAP, pad_m=pad,
                  cell_size_m=float(cs), n_cand=int(n_cand), n_slots=int(N), n_real=n_real, options=options)
    if search is None and M >= n_real:
        return "dense", None, None, (f"the grid's list length {M} is not below the {n_real} real cylinders")
    reason = "search='grid' was given" if search == "grid" else (f"the grid's list length {M} is below the {n_real} "
                                                                 f"real cylinders, at {nbytes} B")
    return "grid", (table, gmin, cs, G), report, reason


def concentric_wall_kernel(centers, inner, outer, L, D_intra, D_myelin, D_extra,
                           kappa_inner, kappa_outer, rho_weights, eps, nudge, step_max, min_gap, count=False, budget=None,
                           trials=False, search=None):
    """The wall interaction of a concentric-cylinder substrate, in its cross-section plane.

    Cylinder ``k`` is the pair of coaxial circles ``|q - c_k| = inner_k`` (the axon membrane) and
    ``|q - c_k| = outer_k`` (the myelin / extra-axonal boundary); ``outer_k == 0`` marks a padding
    slot. A cylinder with ``inner_k == outer_k`` is PLAIN (g-ratio 1): it has no sheath, and its one wall, at
    ``outer_k``, separates its lumen from the extra pool with permeability ``kappa_outer_k``. A walker's pool --
    0 extra, 1 intra (lumen), 2 myelin (annulus) -- is carried state that changes only where a crossing is
    granted. The rule is the one every other geometry uses (:mod:`._boundary`): ray-traced hits,
    ``d_perp = remaining * |cos alpha|`` at the hit, specular multi-bounce reflection off the wall the ray
    actually meets, and a strict side sentinel at the start and end of the step.

    The transmission trials, per kind of wall: a SHEATHED cylinder's two interfaces make one Powles trial per
    step, at the step's first encounter with any wall, with the step's scalar uniform ``u``; a PLAIN cylinder's
    wall makes a trial at every encounter of the step, encounter ``i`` decided by the 32-bit word ``words[i]``
    (:func:`~dmipy_sim.geometry._boundary.trial_granted`), as
    :func:`~dmipy_sim.geometry.packed.packed_wall_kernel` does, whatever walls the step met before.

    The bounce budget and the candidate count are derived from the worst case a walker can meet. For the sheathed
    cylinders: a step ``step_max`` zig-zagging across the narrowest passage -- the smallest gap between two
    outer walls, the thinnest sheath, or the shortest chord a nudged grazing ray can make on the
    smallest lumen (``2 sqrt(2 nudge R)``) -- needs ``step_max / passage + 1`` reflections, with no upper cap: a
    packing whose outer walls nearly touch (``min_gap`` near zero) is a narrower passage than a typical one, and
    capping the count at a fixed value would silently leave some walker-steps' paths untested instead of raising
    the budget to the case at hand. For the plain cylinders: :func:`~dmipy_sim.geometry._boundary.bounce_budget`
    of the smallest radius, the plain packing's own. A cell holding both kinds takes the larger. The walls
    within reach of one step are those of at most ``pi (1 + step_max / R_min)`` disjoint disks, ``R_min`` the
    smallest lumen. ``budget=`` overrides the derived budget for one walk.

    Returns ``wall(xy, d_hat, step_l, pool, k, u, words=None, factor=None)`` giving a :class:`ConcentricHit`: the end position,
    the pool and owning / nearest cylinder after the step, the four boundary local-time channels ``-2 d_perp``
    (inner wall met from the lumen, inner wall met from the sheath, outer wall met from the sheath, outer wall
    met from outside; a plain cylinder's wall met from its lumen is the first channel, from outside the fourth),
    the surface log-weight increment with ``rho_weights[k] = (rho_in/D_in, rho_in/D_my, rho_out/D_my,
    rho_out/D_ex)`` applied per hit (a plain cylinder's first weight is its wall's ``rho/D_in``), whether a
    crossing was granted, the encounters when ``count``, and whether the budget ran out. ``words``, the ``(budget,)``
    ``uint32`` words of the plain walls' trials, is required when the cell holds a plain cylinder and is not read
    otherwise; ``wall.bounce_loop.budget`` is its length. With ``trials`` (a cell with a plain cylinder) the hit also
    carries the step's plain-wall trials (:class:`~dmipy_sim.geometry._boundary.TrialLog`: every encounter of a plain
    wall, the cylinder met, its overshoot ``2 d_perp / R`` and its outcome); a sheathed wall's encounters are not in it.
    ``factor`` (a float32 scalar, or None) is the walker's permeability factor: the walker meets plain cylinder ``k``'s
    wall at the permeability ``kappa_outer_k * factor`` (the float32 product), and a sheathed cylinder's two walls at
    their stated values; ``None`` is the factor 1 and the same program as before. A cell without a plain cylinder
    refuses a factor.

    The candidate search. A step searches the cylinders twice: at its start, for the nearest cylinder (``argmin``
    of the wall distance ``d_k = |_wrap(xy - c_k)| - outer_k`` in float32, a padding slot at infinity) and the
    ``n_cand`` candidates (``top_k`` of ``-d``), and at its end ``xy_f``, for the owner (``argmin`` of ``d`` there).
    The DENSE search computes ``d`` over all ``N`` slots. The GRID search reads ``d`` only on the list of the grid
    cell holding the (wrapped) position: a table built once on the host over ``G x G`` cells of side ``s`` covering
    the periodic cell, read through the shared cell grid (:func:`._grid.gather` with the walker's own cell as the one
    offset, :func:`._grid.wrap_periodic`), each list in ascending cylinder index and padded with an entry at
    infinite distance; the listed ``d`` are the dense search's values bit for bit (the same float32 operations on
    the same operands), and ``argmin`` / ``top_k`` break ties by the lower position, which in an ascending list is
    the lower cylinder index, as in the dense search. Both searches therefore return the same integers whenever
    each list holds every cylinder that can be among the ``n_cand`` nearest anywhere in its cell, which the
    inclusion rule guarantees:

    *Inclusion rule.* Cell ``x0`` (its centre) lists every real cylinder with ``d_k(x0) <= D + 2 (h + pad)``,
    ``d_k`` the exact minimum-image wall distance of the float32 operands, ``D`` the ``n_cand``-th smallest
    ``d_k(x0)``, ``h = s / sqrt(2)`` the cell's half-diagonal and ``pad = 16 eps32 L`` (:data:`WALL_GRID_PAD_EPS`,
    ``eps32 = 2^-23``). *Proof.* ``d_k`` is 1-Lipschitz (a minimum of distances to the images, less a constant).
    Let ``p`` be the position (any periodic image of it), ``e`` a bound on the float32 error of a computed ``d_k``
    and ``l`` a bound on how far the float32 lookup (the wrap, the division by ``s``, the clip into range) can place
    ``p`` outside its cell, so that ``|p - x0| <= h + l`` for the cell read. The ``n_cand`` cylinders of
    ``d_k(x0) <= D`` have ``d_k(p) <= D + h + l`` and computed values at most ``D + h + l + e``, so the
    ``n_cand``-th smallest computed value at ``p`` is at most that; any cylinder ``j`` among the ``n_cand`` nearest
    at ``p`` (or tied with the last of them) has a computed value at most that, so ``d_j(p) <= D + h + l + 2e`` and
    ``d_j(x0) <= D + 2h + 2l + 2e``: listed when ``l + e <= pad``. With the coordinates of the position and the
    centres within ``L`` of the origin (a walk wraps the position into the cell every step; a step's end is within
    one step of it), with ``u = eps32 / 2``: the difference ``xy - c`` is within ``2 u L`` per component, the image
    choice ``floor(q / L + 1/2)`` is decided within ``4.5 u`` of a half-integer, so a wrong image changes the
    component's magnitude by at most ``9 u L``, and the wrapped component's rounding adds ``u L / 2``: ``11.5 u L``
    per component, ``16.3 u L`` on the norm, and the norm's and the subtraction's own rounding ``2.2 u L``, so
    ``e <= 18.5 u L``. The lookup's wrap is within ``3 u L``, the cell index (the float32 origin and cell size the
    table was built with) within ``2 u L`` and the clip into range within ``u L`` per component, ``l <= 8.5 u L``.
    ``e + l <= 27 u L = 13.5 eps32 L <= pad``. The end-of-step search is the case
    ``n_cand = 1`` of the same lists (its ``D`` is no larger). Listing fewer than ``n_cand`` real cylinders cannot
    happen (the ``n_cand`` of ``d_k(x0) <= D`` are listed).

    Which search runs is the stated rule of :func:`_wall_search`, a cost choice since both return the same
    integers: the grid when the cell is periodic, at least ``n_cand`` cylinders are real, ``n_cand < N``, and the
    priced layout's list length ``M`` is below the number of real cylinders within :data:`WALL_GRID_BYTE_CAP`; the dense search otherwise
    (always without a periodic cell). ``search=`` forces ``"dense"`` or ``"grid"``; ``"grid"`` where the rule cannot
    build one is refused, naming why. The kernel carries ``wall.search`` (``"grid"`` or ``"dense"``),
    ``wall.grid_report`` (cells per side, list length ``M``, bytes, the priced options; None for the dense search),
    ``wall.search_reason`` and the two searches themselves, ``wall.nearest(xy) -> (k_near, idx)`` and
    ``wall.owner(xy_f) -> k_new``, which ``wall`` calls.
    """
    N = int(inner.shape[0])
    outer_np = np.asarray(outer, np.float64); inner_np = np.asarray(inner, np.float64)
    real_np = outer_np > 0
    plain_np = real_np & (inner_np >= outer_np)
    sheath_np = real_np & ~plain_np
    any_plain = bool(np.any(plain_np))
    R_in_min = float(np.min(inner_np[real_np]))
    chord_floor = 2.0 * np.sqrt(2.0 * float(nudge) * R_in_min)
    derived = 0
    if np.any(sheath_np):
        thick_min = float(np.min((outer_np - inner_np)[sheath_np]))
        passage = min(float(min_gap), thick_min, chord_floor)
        derived = int(max(2.0, np.ceil(float(step_max) / passage) + 1))
    if any_plain:
        derived = max(derived, bounce_budget(float(np.min(outer_np[real_np])), nudge, min_gap, step_max))
    if trials and not any_plain:
        raise ValueError("a cell without a plain cylinder has no plain-wall trials to log")
    loop = BounceLoop(budget or derived, count=count, log=trials)
    n_cand = int(min(N, max(8, np.ceil(np.pi * (1.0 + float(step_max) / R_in_min)) + 2)))

    search_name, grid, grid_report, search_reason = _wall_search(search, centers, outer, L, N, n_cand)

    real = outer > 0
    plain = jnp.asarray(plain_np)
    ar = jnp.arange(n_cand)
    f32 = jnp.float32
    nudge = f32(nudge); eps = f32(eps)
    pool_after = jnp.asarray(_POOL_AFTER_CROSSING, jnp.int32)

    def _wrap(q):
        return q if L is None else q - L * jnp.floor(q / L + f32(0.5))

    def _pin(xy, q, R_in, R_out, pool):
        return pin_to_pool(xy, q, R_in, R_out, pool, nudge)

    if grid is not None:
        from ._grid import gather, wrap_periodic
        table, gmin, cs, G = grid
        CELL = jnp.asarray(table)
        OFF = jnp.zeros((1, 3), jnp.int32)                                     # the walker's own cell, inflated
        GMIN = jnp.asarray([gmin, gmin, 0.0], f32)
        CS = f32(cs)
        DIMS = jnp.asarray([G, G, 1], jnp.int32)
        VMIN = jnp.asarray([gmin, gmin], f32)
        PER = jnp.ones(2, jnp.int32)

    def _listed(xy):
        """The listed cylinders of the cell holding ``xy`` (wrapped into the cell) and their wall distances, the
        unlisted padding at infinity: ``(ids (M,), dist (M,))``."""
        p = wrap_periodic(VMIN, L, PER, xy)
        ids, valid = gather(CELL, OFF, GMIN, CS, DIMS, jnp.concatenate([p, jnp.zeros(1, f32)]))
        q = _wrap(xy[None, :] - centers[ids])
        return ids, jnp.where(valid, jnp.sqrt(jnp.sum(q * q, -1)) - outer[ids], jnp.inf)

    def nearest(xy):
        """``(k_near, idx)`` at ``xy``: the cylinder of the smallest wall distance ``|_wrap(xy - c_k)| - outer_k``
        (int32; the lowest index on a tie) and the ``n_cand`` candidates, the cylinders of the smallest wall
        distances in ``top_k``'s order (every slot in index order when ``n_cand == N``)."""
        if grid is not None:
            ids, dist = _listed(xy)
            _, pos = jax.lax.top_k(-dist, n_cand)
            return ids[jnp.argmin(dist)].astype(jnp.int32), ids[pos]
        q_all = _wrap(xy[None, :] - centers)                                   # (N, 2)
        dist_wall = jnp.where(real, jnp.sqrt(jnp.sum(q_all * q_all, -1)) - outer, jnp.inf)
        if n_cand < N:
            _, idx = jax.lax.top_k(-dist_wall, n_cand)                         # walls within reach
        else:
            idx = ar
        return jnp.argmin(dist_wall).astype(jnp.int32), idx

    def owner(xy_f):
        """The cylinder of the smallest wall distance at the step's end ``xy_f`` (int32): the owner, or the
        nearest."""
        if grid is not None:
            ids, dist = _listed(xy_f)
            return ids[jnp.argmin(dist)].astype(jnp.int32)
        q_all_f = _wrap(xy_f[None, :] - centers)
        dist_f = jnp.where(real, jnp.sqrt(jnp.sum(q_all_f * q_all_f, -1)) - outer, jnp.inf)
        return jnp.argmin(dist_f).astype(jnp.int32)

    def wall(xy, d_hat, step_l, pool, k, u, words=None, factor=None):
        if factor is not None and not any_plain:
            raise ValueError("a permeability factor scales the plain cylinders' walls; this cell has no plain cylinder")
        if any_plain and words is None:
            raise ValueError("a cell with a plain cylinder (g-ratio 1) takes the 32-bit words of its plain walls' trials, "
                             f"words of shape ({loop.budget},), uint32")
        k_near, idx = nearest(xy)
        k0 = jnp.where(pool == 0, k_near, k)
        xy = _pin(xy, _wrap(xy - centers[k0]), inner[k0], outer[k0], pool)   # the carried pool rules

        c_c, in_c, out_c = centers[idx], inner[idx], outer[idx]
        real_c = out_c > 0
        plain_c = plain[idx]
        ki_c, ko_c = kappa_inner[idx], kappa_outer[idx]
        if factor is not None:                                                 # the walker's own plain-wall permeability
            ko_c = jnp.where(plain_c, ko_c * factor, ko_c)
        Di_c, Dm_c, De_c = D_intra[idx], D_myelin[idx], D_extra[idx]
        w_c = rho_weights[idx]                                                 # (n_cand, 4)

        def rule(p, d, rem, state, i):
            decided, last = state if any_plain else (state, None)
            q = _wrap(p[None, :] - c_c)
            d2 = jnp.sum(q * q, -1)
            pen = jnp.where(real_c, d2 / jnp.maximum(out_c * out_c, _TINY), jnp.inf)
            kc = jnp.argmin(pen)
            in_out = pen[kc] < f32(1.0)                                        # inside an outer wall
            R_in = in_c[kc]
            in_lum = in_out & (d2[kc] < R_in * R_in)
            if any_plain:                                                      # a plain cylinder: all lumen
                pk = plain_c[kc]
                in_out = jnp.where(pk, d2[kc] < out_c[kc] * out_c[kc], in_out)
                in_lum = jnp.where(pk, in_out, in_lum)
            pool_c = jnp.where(in_lum, 1, jnp.where(in_out, 2, 0))
            # outer walls: the owner's from inside (exit root), every other from outside (entry)
            t_en, t_ex, disc_o = ray_sphere_t(q, d, out_c)
            own = (ar == kc) & in_out
            t_o = jnp.where(own, t_ex, t_en)
            t_o = jnp.where(real_c & (disc_o > 0) & (t_o > eps) & (t_o < rem), t_o, jnp.inf)
            ko = jnp.argmin(t_o)
            t_o_min = t_o[ko]
            # the owner's inner wall: exit root from the lumen, entry root from the sheath (a plain cylinder has none)
            t_en_i, t_ex_i, disc_i = ray_sphere_t(q[kc], d, R_in)
            t_i = jnp.where(in_lum, t_ex_i, t_en_i)
            has_inner = in_out & (R_in > 0) & (disc_i > 0) & (t_i > eps) & (t_i < rem)
            if any_plain:
                has_inner = has_inner & (~pk)
            t_i = jnp.where(has_inner, t_i, jnp.inf)
            hit_in = t_i < t_o_min
            t_hit = jnp.minimum(t_i, t_o_min)
            any_hit = jnp.isfinite(t_hit) & (rem > 0)
            t_safe = jnp.where(any_hit, t_hit, f32(0.0))
            kh = jnp.where(hit_in, kc, ko)
            R_hit = jnp.where(hit_in, R_in, out_c[kh])
            disc_h = jnp.where(hit_in, disc_i, disc_o[kh])
            raw = q[kh] + t_safe * d
            n_out = raw / jnp.maximum(jnp.linalg.norm(raw), _TINY)
            rem_a = rem - t_safe
            cos_a = jnp.sqrt(jnp.maximum(disc_h, f32(0.0))) / jnp.maximum(R_hit, _TINY)
            d_perp = jnp.where(any_hit, rem_a * cos_a, f32(0.0))
            stay_in = jnp.where(hit_in, in_lum, own[kh])
            kap = jnp.where(hit_in, ki_c[kh], ko_c[kh])
            D_leave = jnp.where(pool_c == 1, Di_c[kc], jnp.where(pool_c == 2, Dm_c[kc], De_c[kh]))
            first = any_hit & (~decided)
            p_cross = transmit_probability(kap / jnp.maximum(D_leave, _TINY), d_perp)
            transmit = first & (u < p_cross)
            if any_plain:                                                      # a plain wall: a trial at every encounter
                plain_hit = (~hit_in) & plain_c[kh]
                transmit = jnp.where(plain_hit, any_hit & trial_granted(words[i], p_cross), transmit)
            d_refl = specular(d, n_out)
            d_refl = d_refl / jnp.maximum(jnp.linalg.norm(d_refl), _TINY)
            q_off = off_wall(R_hit * n_out, n_out, stay_in, nudge)
            q_off, _ = keep_side_radial(q_off, q_off, R_hit, stay_in, nudge, active=~transmit)
            reflecting = any_hit & (~transmit)
            p_new = jnp.where(reflecting, p + (q_off - q[kh]), p + rem * d)
            d_new = jnp.where(reflecting, d_refl, d)
            rem_new = jnp.where(reflecting, jnp.maximum(rem_a - nudge, f32(0.0)), f32(0.0))
            from_lumen = any_hit & hit_in & (pool_c == 1)
            if any_plain:
                from_lumen = any_hit & (hit_in | plain_hit) & (pool_c == 1)
            sel = jnp.stack([from_lumen, any_hit & hit_in & (pool_c == 2),
                             any_hit & (~hit_in) & (pool_c == 2),
                             any_hit & (~hit_in) & (pool_c == 0)]).astype(f32)
            chan = -wall_push(transmit, d_perp, nudge, cos_a, rem_a) * sel      # the encounter's local time, reflected or transmitted
            dlog_rho = jnp.sum(chan * w_c[kh])
            code = jnp.where(transmit, jnp.where(hit_in, jnp.where(in_lum, f32(1.0), f32(2.0)),
                                                 jnp.where(pool_c == 2, f32(3.0), f32(4.0))), f32(0.0))
            if not any_plain:
                dlw = jnp.concatenate([chan, dlog_rho[None], code[None]])
                return p_new, d_new, rem_new, decided | first, dlw, transmit, reflecting
            code = jnp.where(plain_hit, jnp.where(pool_c == 1, f32(5.0), f32(6.0)), code)
            last = jnp.where(transmit, jnp.round(code).astype(jnp.int32), last)   # the pool follows the last crossing
            dlw = jnp.concatenate([chan, dlog_rho[None], transmit.astype(f32)[None]])
            if trials:
                entry = TrialLog(any_hit & plain_hit, idx[kh].astype(jnp.int32), f32(2.0) * d_perp / R_hit,
                                 transmit & plain_hit)
                return p_new, d_new, rem_new, (decided | first, last), dlw, transmit, reflecting, entry
            return p_new, d_new, rem_new, (decided | first, last), dlw, transmit, reflecting

        if any_plain:
            b = loop(rule, xy, d_hat, step_l, dlog_init=jnp.zeros(6, f32),
                     state_init=(jnp.zeros((), bool), jnp.zeros((), jnp.int32)))
            code, n_cross = b.state[1], jnp.round(b.dlog_w[5]).astype(jnp.int32)
        else:
            b = loop(rule, xy, d_hat, step_l, dlog_init=jnp.zeros(6, f32))
            code, n_cross = jnp.round(b.dlog_w[5]).astype(jnp.int32), b.crossed.astype(jnp.int32)
        xy_f, dlw, crossed = b.r, b.dlog_w, b.crossed
        pool_new = jnp.where(crossed, pool_after[code], pool)
        k_new = owner(xy_f)                                                    # owner, or nearest
        xy_f = _pin(xy_f, _wrap(xy_f - centers[k_new]), inner[k_new], outer[k_new], pool_new)
        n_hits = None if b.n_reflections is None else b.n_reflections + n_cross
        return ConcentricHit(xy_f, pool_new, k_new, dlw[:4], dlw[4], crossed, n_hits, b.exhausted, b.log)

    wall.bounce_loop = loop
    wall.n_cand = n_cand
    wall.has_plain = any_plain
    wall.search = search_name
    wall.grid_report = grid_report
    wall.search_reason = search_reason
    wall.nearest = nearest
    wall.owner = owner
    return wall



def _pool_t2(compartments):
    """``(Compartments, {pool: T2})`` of the ``compartments`` argument, ``None`` where a pool or its T2 is not given."""
    comps = Compartments.coerce(compartments)
    return comps, {n: (comps[n].T2 if n in comps else None) for n in ("extra", "intra", "myelin")}


_POOL_NAMES = ("extra", "intra", "myelin")


def _wet_pools(comps):
    """The pools of ``comps`` that hold water, in :data:`_POOL_NAMES` order: every pool whose ``water_fraction`` is not
    0 (a pool that states none holds water, as its spec writes it, 1.0)."""
    return tuple(n for n in _POOL_NAMES if not (n in comps and comps[n].water_fraction == 0.0))


def _seed_pools(seed_pools, wet):
    """The pools a packed cell seeds as given: ``None`` (every pool that holds water, ``wet``) or a tuple / list of
    distinct names of :data:`_POOL_NAMES`, returned in that order, and ``None`` when they are exactly the pools that
    hold water (one spelling of the default). A name outside :data:`_POOL_NAMES`, a repeated name or another type is a
    ValueError naming the spelling; a pool that holds no water is a ValueError naming the pool."""
    if seed_pools is None:
        return None
    names = list(seed_pools) if isinstance(seed_pools, (tuple, list)) else None
    if not names or any(n not in _POOL_NAMES for n in names) or len(set(names)) != len(names):
        raise ValueError(f"seed_pools is None (every pool that holds water, by area) or distinct names among "
                         f"{_POOL_NAMES}; got {seed_pools!r}")
    dry = [n for n in _POOL_NAMES if n in names and n not in wet]
    if dry:
        raise ValueError(f"seed_pools={tuple(seed_pools)!r}: pool {dry[0]!r} is seeded but has water_fraction 0")
    names = tuple(n for n in _POOL_NAMES if n in names)
    return None if names == wet else names


def _pool_D(comps, who):
    """``{pool: D}`` (m^2/s) of the three pools from ``comps``. No pool has a default: every pool that holds walkers
    states ``Pool(D=)`` (a frozen myelin pool states ``D=0.0``); a missing one is a ValueError naming the pool. Only a
    myelin pool that states ``water_fraction=0`` holds none and may omit it."""
    out = {}
    for n in ("extra", "intra", "myelin"):
        pool = comps[n] if n in comps else None
        v = pool.D if pool is not None else None
        if v is None:
            if n == "myelin" and pool is not None and pool.water_fraction == 0.0:
                v = 0.0
            else:
                raise ValueError(f"{who} needs the diffusivity of its {n} pool: "
                                 f"compartments=Compartments({n}=Pool(D=...), ...)"
                                 + ("; a frozen myelin pool states D=0.0" if n == "myelin" else ""))
        out[n] = float(v)
    return out


class MyelinatedCylinder(Geometry):
    """Three-compartment myelinated cylinder: intra-axonal, myelin sheath, extra-axonal.

    The geometry consists of two concentric cylinders (inner radius R_inner,
    outer radius R_outer) along a given orientation axis:
      - Compartment 0 (intra-axonal): r_xy < R_inner
      - Compartment 1 (myelin sheath): R_inner <= r_xy <= R_outer
      - Compartment 2 (extra-axonal):  r_xy > R_outer

    Each compartment has a single **isotropic** diffusivity. Myelin water is trapped
    between the lipid bilayers with a short T2 and barely moves on any realistic
    diffusion time, so ``D_myelin`` defaults to 0 (a stuck pool that only carries
    ``T2_myelin``; the analytical counterpart is a stationary ``S1Dot``). Set it to any
    value > 0 to let myelin water diffuse. Two boundaries have independent permeability.

    Parameters
    ----------
    inner_radius : float
        Inner cylinder radius in metres (axon radius).
    outer_radius : float
        Outer cylinder radius in metres (outer myelin boundary).
    orientation : array-like of shape (3,)
        The axis's direction in the lab: the POSE of the substrate for :func:`~dmipy_sim.simulate`. The walk runs
        in the substrate's own frame (the axis along +z) and the engine rotates the acquisition into it
        (``_orient_R``, as for a Mesh), so the walkers and the wall rules never see the pose and a tilted
        substrate costs what an upright one does. A pack built from a walk of this geometry is in the substrate
        frame; a pose is given at replay.
    kappa_inner : float, optional
        Permeability at inner boundary (m/s). Default None (impermeable).
    kappa_outer : float, optional
        Permeability at outer boundary (m/s). Default None (impermeable).
    compartments : Compartments
        The per-pool physics, ``Compartments(intra=Pool(D=..., T2=..., water_fraction=...), myelin=Pool(...),
        extra=Pool(...))``. ``D`` (m^2/s, isotropic) is required on every pool that holds walkers; a frozen myelin
        pool states ``D=0.0``. ``T2`` (s) is optional. ``water_fraction`` is the relative water content (proton density)
        per pool: set on all three pools or none; none uses the biophysical table, myelin =
        ``myelin_water_proton_density`` (~0.40, the water per unit myelin VOLUME), intra and extra = 1.0. Do NOT pass
        the MWF signal value (~0.15) -- that is a measured signal fraction, not a per-volume weight.
    """

    # Marker: this geometry provides its own step function (not make_step_fn)
    _is_myelinated = True
    has_walls = False                 #: stepped by its own kernel, never by `_wall`

    @property
    def length_scales(self):
        return LengthScales(min_feature=self.inner_radius)

    def __init__(self, inner_radius, outer_radius, orientation,
                 kappa_inner=None, kappa_outer=None, compartments=None):
        if outer_radius <= inner_radius:
            raise ValueError("outer_radius must be > inner_radius")
        self.compartments, t2 = _pool_t2(compartments)
        T2_extra, T2_intra, T2_myelin = t2["extra"], t2["intra"], t2["myelin"]
        D = _pool_D(self.compartments, "MyelinatedCylinder")
        D_intra, D_extra, D_myelin = D["intra"], D["extra"], D["myelin"]
        wf_pools = [self.compartments[n].water_fraction if n in self.compartments else None
                    for n in ("intra", "myelin", "extra")]
        water_fractions = None
        if any(w is not None for w in wf_pools):
            if any(w is None for w in wf_pools):
                raise ValueError("Pool(water_fraction=) must be set on all three pools or none")
            water_fractions = tuple(wf_pools)

        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self.D_intra = float(D_intra)
        self.D_myelin = float(D_myelin)
        self.D_extra = float(D_extra)
        _refuse_schedule("MyelinatedCylinder", kappa_inner=kappa_inner, kappa_outer=kappa_outer)
        self.kappa_inner = float(kappa_inner) if kappa_inner is not None else None
        self.kappa_outer = float(kappa_outer) if kappa_outer is not None else None
        self.T2_intra = float(T2_intra) if T2_intra is not None else None
        self.T2_myelin = float(T2_myelin) if T2_myelin is not None else None
        self.T2_extra = float(T2_extra) if T2_extra is not None else None
        if water_fractions is None:
            from ..substrate.biophysical_constants import get_default_value
            water_fractions = (1.0, float(get_default_value('myelin_water_proton_density')), 1.0)
        self.water_fractions = tuple(float(w) for w in water_fractions)
        #: proton-density weight indexed by POOL id (0 extra, 1 intra, 2 myelin)
        self._water_fraction_by_pool = (self.water_fractions[2], self.water_fractions[0],
                                        self.water_fractions[1])

        orientation = np.asarray(orientation, dtype=np.float64)
        self.orientation = (orientation / np.linalg.norm(orientation)).astype(np.float32)
        self._orient_R = acquisition_rotation(self.orientation)

        # The same per-axon array layout as PackedMyelinatedCylinders, with one axon at the
        # origin and no periodic cell, so both are stepped by the one kernel in `physics`.
        self.N_max = 1
        self._centers_jax = jnp.zeros((1, 2), jnp.float32)
        self._inner_radii_jax = jnp.array([self.inner_radius], jnp.float32)
        self._outer_radii_jax = jnp.array([self.outer_radius], jnp.float32)
        self._D_intra_jax = jnp.array([self.D_intra], jnp.float32)
        self._D_myelin_jax = jnp.array([self.D_myelin], jnp.float32)
        self._D_extra_jax = jnp.array([self.D_extra], jnp.float32)
        self._kappa_inner_jax = jnp.array([self.kappa_inner or 0.0], jnp.float32)
        self._kappa_outer_jax = jnp.array([self.kappa_outer or 0.0], jnp.float32)
        self._eps = jnp.float32(1e-7 * self.inner_radius)
        self._nudge = jnp.float32(1e-4 * self.inner_radius)
        self.min_gap = float('inf')

    def init_positions(self, n_walkers, key):
        """Distribute walkers proportional to volume * water_fraction per compartment.

        Extra-axonal volume is set to the annulus between R_outer and 2*R_outer
        for allocation purposes (actual walkers are placed uniformly).
        """
        R_in = self.inner_radius
        R_out = self.outer_radius

        # Volumes (per unit length)
        vol_intra = np.pi * R_in**2
        vol_myelin = np.pi * (R_out**2 - R_in**2)
        # For extra-axonal, use an annulus up to 2*R_out for allocation
        R_extra = 2.0 * R_out
        vol_extra = np.pi * (R_extra**2 - R_out**2)

        # Weighted volumes
        # Homogeneous placement (by volume). The per-compartment water proton density
        # (water_fractions) is applied as a SIGNAL weight in simulate(), not here.
        w_intra = vol_intra
        w_myelin = vol_myelin
        w_extra = vol_extra
        w_total = w_intra + w_myelin + w_extra

        n_intra = int(round(n_walkers * w_intra / w_total))
        n_myelin = int(round(n_walkers * w_myelin / w_total))
        n_extra = n_walkers - n_intra - n_myelin

        rng = np.random.default_rng(placement_seed(key))

        positions = np.zeros((n_walkers, 3), dtype=np.float32)
        compartments = np.zeros(n_walkers, dtype=np.int32)

        idx = 0

        # Intra-axonal: uniform in circle of radius R_in
        if n_intra > 0:
            pts = []
            while sum(len(p) for p in pts) < n_intra:
                xy = rng.uniform(-R_in, R_in, (n_intra * 3, 2))
                pts.append(xy[np.linalg.norm(xy, axis=1) < R_in])
            xy_intra = np.concatenate(pts, axis=0)[:n_intra].astype(np.float32)
            positions[idx:idx + n_intra, 0] = xy_intra[:, 0]
            positions[idx:idx + n_intra, 1] = xy_intra[:, 1]
            compartments[idx:idx + n_intra] = 1
            idx += n_intra

        # Myelin: uniform in annulus R_in <= r < R_out
        if n_myelin > 0:
            pts = []
            while sum(len(p) for p in pts) < n_myelin:
                xy = rng.uniform(-R_out, R_out, (n_myelin * 3, 2))
                r_xy = np.linalg.norm(xy, axis=1)
                pts.append(xy[(r_xy >= R_in) & (r_xy < R_out)])
            xy_myelin = np.concatenate(pts, axis=0)[:n_myelin].astype(np.float32)
            positions[idx:idx + n_myelin, 0] = xy_myelin[:, 0]
            positions[idx:idx + n_myelin, 1] = xy_myelin[:, 1]
            compartments[idx:idx + n_myelin] = 2
            idx += n_myelin

        # Extra-axonal: uniform in annulus R_out <= r < R_extra
        if n_extra > 0:
            pts = []
            while sum(len(p) for p in pts) < n_extra:
                xy = rng.uniform(-R_extra, R_extra, (n_extra * 3, 2))
                r_xy = np.linalg.norm(xy, axis=1)
                pts.append(xy[(r_xy >= R_out) & (r_xy < R_extra)])
            xy_extra = np.concatenate(pts, axis=0)[:n_extra].astype(np.float32)
            positions[idx:idx + n_extra, 0] = xy_extra[:, 0]
            positions[idx:idx + n_extra, 1] = xy_extra[:, 1]
            compartments[idx:idx + n_extra] = 0
            idx += n_extra

        # the float64 accept tests above can round a draw onto a wall in float32: pull every seed onto its pool's side
        positions[:, :2] = _pin_seeds(positions[:, :2], np.zeros((n_walkers, 2)), None, np.full(n_walkers, R_in),
                                      np.full(n_walkers, R_out), compartments, self._nudge)

        r_lab = positions                                   # the substrate frame: xy the cross-section, z the axis

        self._init_compartments = jnp.array(compartments, dtype=jnp.int32)
        return jnp.array(r_lab, dtype=jnp.float32)

    def _wall(self, r, step, kappa_over_D, rho_over_D, key):
        """Not a usable boundary rule for this geometry -- raises.

        A three-compartment geometry cannot be stepped by a single-surface reflect: the
        walker's compartment is STATE that changes only at a granted crossing, which is why
        the real kernel (`physics.make_myelin_step_fn`) carries `compartment_id` and clamps
        the position to match it.  This method used to clamp every walker to the inner
        cylinder instead, so a `_walk_geometry` call on a MyelinatedCylinder returned
        a silently single-compartment walk -- myelin and extra-axonal water simply absent,
        with no error.  Failing loudly is the point: a boundary interaction that does not
        happen is exactly the class of defect that produced dmrai-lab/dmipy-sim#86.
        """
        raise NotImplementedError(
            "MyelinatedCylinder has no single-surface reflect: its three compartments are "
            "stepped by the fused kernel physics.make_myelin_step_fn, which carries the "
            "compartment id. Use simulate(...) (which dispatches to that kernel) rather "
            "than a generic reflect/trajectory walk.")
    def classify_position(self, r: jnp.ndarray) -> jnp.ndarray:
        """Compartment id from the radial distance in the cross-section: 1 intra
        (|r_xy| < R_inner), 2 myelin (R_inner <= |r_xy| < R_outer), 0 extra."""
        R_in  = jnp.float32(self.inner_radius)
        R_out = jnp.float32(self.outer_radius)
        r_xy_sq = vdot(r[:2], r[:2])
        in_intra  = r_xy_sq < R_in  * R_in
        in_myelin = (r_xy_sq >= R_in * R_in) & (r_xy_sq < R_out * R_out)
        return jnp.where(in_intra, jnp.int32(1),
               jnp.where(in_myelin, jnp.int32(2), jnp.int32(0)))

    def volume(self, compartment: str, L: float = 1.0) -> float:
        """Volume of a compartment per unit length L (m³).

        Compartment 'extra' uses the annulus between R_outer and 2·R_outer as
        its bounding region (matching the convention used in init_positions).

        Parameters
        ----------
        compartment : str
            One of 'intra', 'myelin', or 'extra'.
        L : float, optional
            Cylinder length in metres. Default 1.0 (per-unit-length).
        """
        L = float(L)
        R_in  = self.inner_radius
        R_out = self.outer_radius
        if compartment == 'intra':
            return np.pi * R_in ** 2 * L
        elif compartment == 'myelin':
            return np.pi * (R_out ** 2 - R_in ** 2) * L
        elif compartment == 'extra':
            R_extra = 2.0 * R_out
            return np.pi * (R_extra ** 2 - R_out ** 2) * L
        else:
            raise ValueError(
                f"compartment must be 'intra', 'myelin', or 'extra'; got {compartment!r}")

    def surface_area(self, compartment: str, L: float = 1.0) -> float:
        """Lateral surface area bounding a compartment per unit length L (m²).

        Returns the area of the cylindrical wall(s) that bound the compartment:
          - 'intra':  inner wall only, area = 2π·R_inner·L
          - 'myelin': both walls, area = 2π·(R_inner + R_outer)·L
          - 'extra':  outer wall only (inner boundary of extra-axonal space),
                      area = 2π·R_outer·L

        Parameters
        ----------
        compartment : str
            One of 'intra', 'myelin', or 'extra'.
        L : float, optional
            Cylinder length in metres. Default 1.0 (per-unit-length).
        """
        L = float(L)
        R_in  = self.inner_radius
        R_out = self.outer_radius
        if compartment == 'intra':
            return 2.0 * np.pi * R_in * L
        elif compartment == 'myelin':
            return 2.0 * np.pi * (R_in + R_out) * L
        elif compartment == 'extra':
            return 2.0 * np.pi * R_out * L
        else:
            raise ValueError(
                f"compartment must be 'intra', 'myelin', or 'extra'; got {compartment!r}")

    def volume_fraction(self, compartment: str) -> float:
        """Volume fraction of a compartment within the bounding cylinder.

        The bounding cylinder has radius 2·R_outer (matching init_positions).
        Volume fractions sum to 1 over {'intra', 'myelin', 'extra'}.

        Parameters
        ----------
        compartment : str
            One of 'intra', 'myelin', or 'extra'.
        """
        R_in   = self.inner_radius
        R_out  = self.outer_radius
        R_extra = 2.0 * R_out
        total = np.pi * R_extra ** 2  # bounding cylinder cross-section area
        if compartment == 'intra':
            return np.pi * R_in ** 2 / total
        elif compartment == 'myelin':
            return np.pi * (R_out ** 2 - R_in ** 2) / total
        elif compartment == 'extra':
            return np.pi * (R_extra ** 2 - R_out ** 2) / total
        else:
            raise ValueError(
                f"compartment must be 'intra', 'myelin', or 'extra'; got {compartment!r}")


class MyelinatedCell(NamedTuple):
    """What a backend walks of a :class:`PackedMyelinatedCylinders` cell (:meth:`PackedMyelinatedCylinders.cell_arrays`):
    the arrays and constants its wall kernel (:func:`concentric_wall_kernel`) is built from, read-only, in the
    substrate frame (the cylinders along +z, the cross-section ``[-L/2, L/2)^2``), SI units. The per-cylinder arrays are
    padded to ``N_max`` (zero radii past ``N_actual``), as the kernel pads them.

    Compartments are encoded as the kernel encodes them (:meth:`PackedMyelinatedCylinders.pool_of`): 0 the extra pool,
    ``k + 1`` the lumen of cylinder ``k`` (``1 .. N_max``), ``N_max + k + 1`` its sheath (``N_max + 1 .. 2 N_max``,
    empty for a plain cylinder). A walker in pool ``p`` of cylinder ``k`` steps ``sqrt(6 D_p[k] dt_sim)`` times the
    step law's atom, ``D_p`` the pool's array below and ``dt_sim`` the request's (:mod:`dmipy_sim.math.step_law`).

    Every input of :func:`concentric_wall_kernel` as the coefficient walk builds it
    (:func:`~dmipy_sim.engine.physics.make_packed_myelin_traj_step_fn`) is here or is a property of the walk:

    - ``centers``, ``inner``, ``outer``, ``L`` (``cell_size`` in float32), ``D_intra``, ``D_myelin``, ``D_extra``,
      ``kappa_inner``, ``kappa_outer``, ``eps``, ``nudge``, ``min_gap`` and ``budget`` (``bounce_budget``) are fields;
    - ``step_max``, the longest step the loop is sized for, depends on the walk's step: :meth:`step_max` of
      :attr:`~dmipy_sim.engine.backends.WalkRequest.dt_sim`;
    - ``rho_weights`` is zero in the coefficient walk, which applies no surface relaxivity: the contact is recorded as
      the four boundary local-time channels at ``rho / D = 1`` (:attr:`~dmipy_sim.engine.backends.WalkRequest.record`),
      a relaxivity being a replay knob. The cell's ``rho_inner`` / ``rho_outer`` are read only by the forward-signal
      step (:func:`~dmipy_sim.engine.physics.make_packed_myelin_step_fn`), which no backend walks;
    - ``count`` is :attr:`~dmipy_sim.engine.backends.WalkRequest.count_walls` and ``trials`` whether the request
      carries a :attr:`~dmipy_sim.engine.backends.WalkRequest.trial_record`;
    - ``search`` is the cell's ``wall_search``, the choice of the candidate search, which returns the same candidates
      either way: nothing a backend walks depends on it."""
    centers: np.ndarray       #: (N_max, 2) float32 the cylinders' centres (m)
    inner_radii: np.ndarray   #: (N_max,) float32 the lumen radii (m)
    outer_radii: np.ndarray   #: (N_max,) float32 the outer radii (m); a plain cylinder's equals its inner radius
    cell_size: float          #: the cell's side ``L`` (m)
    D_intra: np.ndarray       #: (N_max,) float32 the lumen's diffusivity per cylinder (m^2/s)
    D_myelin: np.ndarray      #: (N_max,) float32 the sheath's diffusivity per cylinder (m^2/s)
    D_extra: np.ndarray       #: (N_max,) float32 the extra pool's diffusivity seen from each cylinder (m^2/s)
    kappa_inner: np.ndarray   #: (N_max,) float32 the axolemma's permeability (m/s); 0 at a plain cylinder
    kappa_outer: np.ndarray   #: (N_max,) float32 the outer wall's permeability (m/s); a plain cylinder's one wall
    plain_mask: np.ndarray    #: (N_max,) bool the plain cylinders (g-ratio 1); False past ``N_actual``
    N_actual: int             #: the cylinders in the cell
    N_max: int                #: the padded length of every per-cylinder array
    eps: float                #: the kernel's distance tolerance (m), ``1e-7`` of the smallest lumen radius
    nudge: float              #: the kernel's push off a wall after a reflection (m), ``1e-4`` of the smallest lumen radius
    min_gap: float            #: the narrowest gap between two outer walls in the periodic cell (m)
    bounce_budget: Optional[int]  #: the budget the kernel's loop runs at: the cell's ``bounce_budget``, which is the
                                  #: :attr:`~dmipy_sim.engine.backends.WalkRequest.bounce_budget` of a walk of it;
                                  #: None for the budget the kernel derives

    def step_max(self, dt_sim):
        """The longest step (m) the kernel's loop is sized for at a physics step of ``dt_sim`` seconds:
        ``sqrt(6 D_max dt_sim)``, ``D_max`` the largest diffusivity of the three pools, times
        :data:`~dmipy_sim.math.step_law.LONGEST_STEP` when the cell holds a plain cylinder."""
        return concentric_step_max(self.D_intra, self.D_myelin, self.D_extra, dt_sim, bool(np.any(self.plain_mask)))


def concentric_step_max(D_intra, D_myelin, D_extra, dt, has_plain):
    """The longest step (m) :func:`concentric_wall_kernel`'s loop is sized for at a physics step of ``dt`` seconds:
    ``sqrt(6 D_max dt)``, ``D_max`` the largest of the three pools' per-cylinder diffusivities (m^2/s), times
    :data:`~dmipy_sim.math.step_law.LONGEST_STEP` when ``has_plain`` (a cell holding a plain cylinder sizes its loop
    for the longest step the law draws, as the plain packing does).

    A cell of sheathed cylinders only is sized for ``sqrt(6 D_max dt)``, and a draw can be up to ``LONGEST_STEP`` times
    longer. The bounce budget derived from the sizing can then be exhausted by such a draw: on the four-cylinder test
    cell at ``dt`` 1e-6 s it is 5, where the longest draw would size 7. A step that exhausts its budget stops the walker
    at its last collision: the step is shorter than drawn, it is reported as ``exhausted`` (the flag a walk sums into
    ``exhausted_steps``), and the walker stays in its pool. In ``tests/geometry/test_myelin_longest_step.py``, 5.9, 3.6,
    1.5, 2.9 and 0 % of 90 300 near-tangent lanes of the smallest lumen exhaust at ``dt`` 2e-7, 5e-7, 1e-6, 2e-6 and
    4e-6 s, a step is shortened by at most 1.9e-8 m, and in every searched lane the unflown remainder ends in the
    walker's pool (an enumeration of those lanes, not a bound): also in the sheath, where the budget runs out on the
    outer wall at ``dt`` 2e-7, and in the extra-axonal pool, where no lane exhausts it. The candidate count
    ``min(N, max(8, ceil(pi (1 + step_max / R_min)) + 2))`` is the one a kernel sized at the longest draw has."""
    from ..math.step_law import LONGEST_STEP
    D_max = float(max(np.max(np.asarray(D_intra)), np.max(np.asarray(D_myelin)), np.max(np.asarray(D_extra))))
    step_max = float(np.sqrt(6.0 * D_max * dt))
    if has_plain:
        step_max = float(LONGEST_STEP) * step_max
    return step_max


class PackedMyelinatedCylinders(Geometry):
    """A periodic square cell of parallel cylinders, each myelinated or plain: one packing of axons.

    Each cylinder k (k = 0..N_actual-1) is placed by its OUTER radius ``R_outer_k = R_inner_k / g_k`` at the centre
    ``(cx_k, cy_k)`` of the periodic cell ``[-L/2, L/2)^2`` and is one of two kinds, set by its g-ratio ``g_k``:

    - **myelinated** (``0 < g_k < 1``): a lumen (``r < R_inner_k``), a sheath (``R_inner_k <= r < R_outer_k``) and
      two walls, the axolemma at ``R_inner_k`` (``kappa_inner``, ``rho_inner``) and the sheath's outer wall at
      ``R_outer_k`` (``kappa_outer``, ``rho_outer``); its sheath is a source of the hollow-cylinder field;
    - **plain** (``g_k = 1``): a lumen and ONE wall at ``R_outer_k = R_inner_k`` between the lumen and the extra
      pool, with permeability ``kappa_outer`` and relaxivity ``rho_outer``; no sheath and no field.

    The extra-axonal pool is the rest of the cell. Every wall reflects specularly or, where permeable, is crossed by
    the Powles rule: a sheathed cylinder's two walls make one trial per step, at the step's first encounter with
    any wall; a plain cylinder's wall makes a trial at every encounter, as the plain packing's walls do
    (:func:`concentric_wall_kernel`). A cell of plain cylinders only is a packing of permeable cylinders; a cell of
    myelinated cylinders only is a packing of myelinated axons; a cell of both is a white-matter packing with
    myelinated and unmyelinated axons.

    Example: a cell of two myelinated axons (impermeable) and two unmyelinated ones (permeable)::

        g = [0.7, 1.0, 0.7, 1.0]
        cell = PackedMyelinatedCylinders(outer * g, g, centers, L, N_max=4,
                                         compartments=Compartments(intra=Pool(D=2e-9), myelin=Pool(D=0.0),
                                                                   extra=Pool(D=2e-9)),
                                         kappa_outer=[0.0, 4e-5, 0.0, 4e-5])

    Compartment numbering
    ~~~~~~~~~~~~~~~~~~~~~
    - 0              : extra-axonal
    - 1 .. N_max     : intra_k  (cylinder k, k+1-th slot)
    - N_max+1 .. 2*N_max : myelin_k  (sheath of cylinder k, k+1-th slot; empty for a plain cylinder)

    Zero-radius padding
    ~~~~~~~~~~~~~~~~~~~
    Arrays are padded to length ``N_max`` with zeros.  Dummy cylinders (r=0)
    receive zero walkers (area = 0) and have SDF = inf in the step function
    (they never win ``argmin``), so different N_actual with the same N_max
    compile to the same JAX program.

    Parameters
    ----------
    inner_radii : array-like, shape (N_actual,)
        Lumen radii in metres.
    g_ratios : array-like, shape (N_actual,) or scalar
        g-ratio per cylinder, finite and in ``(0, 1]``: below 1 a myelinated cylinder, 1 a plain one. A g-ratio
        outside that range, and a myelinated cylinder whose sheath has no positive thickness in float32, are
        refused by name.
    centers : np.ndarray, shape (N_actual, 2)
        Cylinder centre positions in metres, [-L/2, L/2) convention.
    cell_size : float
        Side-length of the periodic square cell in metres.
    N_max : int, optional
        Fixed JIT padding length (>= N_actual).  Default 128.
    orientation : array-like, shape (3,), optional
        The axis's direction in the lab: the POSE of the substrate for :func:`~dmipy_sim.simulate`. The walk runs
        in the substrate's own frame (the axis along +z) and the engine rotates the acquisition into it
        (``_orient_R``, as for a Mesh), so the walkers and the wall rules never see the pose and a tilted
        substrate costs what an upright one does. A pack built from a walk of this geometry is in the substrate
        frame; a pose is given at replay.
        Default [0, 0, 1].
    compartments : Compartments, optional
        The per-pool physics shared by all cylinders: ``D`` (m^2/s, isotropic; required on every pool that holds
        walkers, a frozen myelin pool states ``D=0.0``) and ``T2`` (s), ``Compartments(intra=Pool(D=..., T2=...), ...)``.
    T2_intra, T2_myelin, T2_extra : array-like (N_actual,) or None
        Per-cylinder T2 relaxation times in seconds, a per-object property. A pool-wide T2 is a pool
        property and is given through ``compartments=``; a scalar here is refused. None = no T2.
    kappa_inner, kappa_outer : float or array-like (N_actual,) or None
        Inner/outer wall permeabilities in m/s.  None / 0.0 = impermeable. ``kappa_inner`` is the axolemma of the
        myelinated cylinders: a scalar applies to them, and an array holds 0 at every plain cylinder (a plain
        cylinder's one wall is its outer wall, ``kappa_outer``; a nonzero entry there is refused).
    rho_inner, rho_outer : float or array-like (N_actual,) or None
        Surface relaxivity at inner/outer walls in m/s.  0.0 = no relaxivity; ``rho_inner`` as ``kappa_inner``.
    seed_pools : sequence of str or None
        The pools the walkers start in, by area among them: ``None`` (every pool that holds water, i.e. whose
        ``Pool.water_fraction`` is not 0) or names among ``"extra"``, ``"intra"``, ``"myelin"`` (``("extra",
        "intra")``: no walker starts in a sheath). A pool of water fraction 0 holds no walker: naming one is a
        ValueError. Naming exactly the pools that hold water is the default, kept as ``None``. The spec states the
        seeded pools (:attr:`seeded_pools`) as its ``seeding.pools``.
    """

    _is_packed_myelinated = True
    has_walls = False                 #: stepped by its own kernel, never by `_wall`

    @property
    def seeded_pools(self):
        """The pools the walkers start in, in ``("extra", "intra", "myelin")`` order: ``seed_pools`` when given, else
        every pool that holds water."""
        return self.seed_pools if self.seed_pools is not None else _wet_pools(self.compartments)

    @property
    def reachable_pools(self):
        """The pools a seeded walker can reach, in ``("extra", "intra", "myelin")`` order: the seeded pools
        (:attr:`seeded_pools`) and every pool joined to one of them by a chain of open faces. A face between two pools is
        the axolemma (lumen | sheath, ``kappa_inner``) or the sheath's outer face (sheath | extra, ``kappa_outer``) of a
        myelinated cylinder, or the one wall (lumen | extra, ``kappa_outer``) of a plain cylinder; it is open when the
        permeability of any cylinder that has it is above 0. The permeabilities are per cylinder and constant in time
        (a schedule is refused at construction), so these are every face the walk can open."""
        N = self.N_actual
        sheathed = ~np.asarray(self.plain_mask[:N], bool)
        k_in = np.asarray(self._kappa_inner_jax)[:N]; k_out = np.asarray(self._kappa_outer_jax)[:N]
        faces = []                                                     # the pairs of pools a walker can cross between
        if np.any(k_in[sheathed] > 0):
            faces.append(("intra", "myelin"))
        if np.any(k_out[sheathed] > 0):
            faces.append(("myelin", "extra"))
        if np.any(k_out[~sheathed] > 0):
            faces.append(("intra", "extra"))
        reach = set(self.seeded_pools)
        while True:
            more = {q for f in faces for p, q in (f, f[::-1]) if p in reach and q not in reach}
            if not more:
                break
            reach |= more
        return tuple(n for n in _POOL_NAMES if n in reach)

    @property
    def sheath_in_reach(self):
        """Whether a sheath of positive area is among the :attr:`reachable_pools`: seeded, or behind an open face of a
        seeded pool. Without one, every sheath is unseeded and sealed from every seeded pool."""
        return bool("myelin" in self.reachable_pools and np.any(self.sheathed))

    @property
    def placement_is_stationary(self):
        """Whether the placement's law is the walk's stationary law. The placement is the uniform law over the union of
        the seeded pools (:attr:`seeded_pools`, by area among them), so every pair of seeded pools that exchange is
        seeded in proportion to its area; the law is stationary exactly when no pool of positive area outside the
        seeded pools is among the :attr:`reachable_pools`. A pool that holds no water is no exception: a walker
        crosses into it as into any pool (the crossing reads the diffusivity of the side it leaves), so an unseeded
        dry pool behind an open face fills during the walk."""
        N = self.N_actual
        inner = self._inner_radii_np[:N]; outer = self._outer_radii_np[:N]
        area = {"extra": self._L_float ** 2 - float(np.sum(np.pi * outer ** 2)),
                "intra": float(np.sum(np.pi * inner ** 2)), "myelin": float(np.sum(np.pi * (outer ** 2 - inner ** 2)))}
        return not any(area[n] > 0.0 for n in set(self.reachable_pools) - set(self.seeded_pools))

    def water_domains(self):
        """The water domains of the seeded walkers, when no sheath is among the :attr:`reachable_pools`.

        A water domain is a connected component of the water through the open faces; with every sheath unseeded and
        sealed from the seeded pools they are, per unit length of the cylinders:

        * the lumen of each myelinated cylinder ``k``, ``V = pi a_k^2``, ``S = 2 pi a_k`` (its axolemma, from inside),
          and of each sealed plain cylinder (``kappa_outer = 0``), ``V = pi b_k^2``, ``S = 2 pi b_k`` (its wall, from
          inside);
        * the exterior: the extra pool and the lumina of the open plain cylinders (``kappa_outer > 0``),
          ``V = L^2 - sum pi b_k^2`` over the myelinated and the sealed plain cylinders, ``S = sum 2 pi b_k`` over them
          (the sheath's outer face and the sealed plain wall, each from the extra side, once) ``+ sum 2 (2 pi b_k)``
          over the open plain cylinders (both faces of an open wall: a walker of the domain meets it from either side).

        ``a_k`` is the lumen radius, ``b_k`` the outer radius, ``L`` the cell's side. A domain is listed when it holds
        seeded water: its ``seeded_area`` is the area of its seeded pools (the placement is uniform by area over the
        seeded pools), and a domain with none is left out. Lumina first, in the cylinders' order, then the exterior.

        Returns
        -------
        tuple of WaterDomain

        Raises
        ------
        ValueError
            When a sheath is among the :attr:`reachable_pools` (seeded, or behind an open face of a seeded pool): its
            water joins a domain whose walls this form does not count.
        """
        if self.sheath_in_reach:
            raise ValueError(f"the myelin pool is reachable from the seeded pools {self.seeded_pools!r}: the water "
                             f"domains are stated for a cell whose sheaths are unseeded and sealed from every seeded pool")
        N = self.N_actual
        a = np.asarray(self._inner_radii_np[:N], np.float64)
        b = np.asarray(self._outer_radii_np[:N], np.float64)
        plain = np.asarray(self.plain_mask[:N], bool)
        open_plain = plain & (np.asarray(self._kappa_outer_jax, np.float64)[:N] > 0.0)
        seeded = self.seeded_pools
        domains = []
        if "intra" in seeded:
            for k in np.flatnonzero(~open_plain):                     # a sealed lumen: its own wall, from inside
                V = float(np.pi * a[k] ** 2)
                domains.append(WaterDomain(("intra",), (int(k),), V, float(2.0 * np.pi * a[k]), float(2.0 / a[k]), V))
        closed = ~open_plain
        V = float(self._L_float ** 2 - np.sum(np.pi * b[closed] ** 2))
        S = float(2.0 * np.pi * np.sum(b[closed]) + 2.0 * (2.0 * np.pi * np.sum(b[open_plain])))
        seeded_area = ((float(self._L_float ** 2 - np.sum(np.pi * b ** 2)) if "extra" in seeded else 0.0)
                       + (float(np.sum(np.pi * b[open_plain] ** 2)) if "intra" in seeded else 0.0))
        if seeded_area > 0.0:
            pools = ("extra", "intra") if np.any(open_plain) else ("extra",)
            domains.append(WaterDomain(pools, tuple(int(k) for k in np.flatnonzero(open_plain)), V, S, S / V,
                                       seeded_area))
        return tuple(domains)

    @property
    def length_scales(self):
        """Smallest lumen for the step rules; the extra-axonal pore ``A_ext / P_ext`` for the
        surface-relaxivity rule, since that is the pore the fast, grazing extra-axonal walkers
        occupy."""
        N = self.N_actual
        inner = self._inner_radii_np[:N]; outer = self._outer_radii_np[:N]
        area_ext = self._L_float ** 2 - float(np.sum(np.pi * outer ** 2))
        perim = float(np.sum(2.0 * np.pi * outer))
        pore = area_ext / perim if (perim > 0 and area_ext > 0) else None
        nz = inner[inner > 0]
        return LengthScales(min_feature=float(np.min(nz)) if nz.size else None,
                            surface_pore=pore, min_gap=self.min_gap)

    def __init__(
        self,
        inner_radii,
        g_ratios,
        centers,
        cell_size,
        N_max=128,
        orientation=(0., 0., 1.),
        T2_intra=None,
        T2_myelin=None,
        T2_extra=None,
        kappa_inner=0.0,
        kappa_outer=0.0,
        rho_inner=0.0,
        rho_outer=0.0,
        compartments=None,
        seed_pools=None,
    ):
        self.compartments, t2 = _pool_t2(compartments)
        self.seed_pools = _seed_pools(seed_pools, _wet_pools(self.compartments))
        #: The wall kernel's candidate search (:func:`concentric_wall_kernel` ``search=``): None for its stated rule,
        #: ``"dense"`` or ``"grid"`` to force one. Both return the same integers; a program built after a change is
        #: a new one (a string of the instance is part of the compiled-program key).
        self.wall_search = None
        for name, arr in (("extra", T2_extra), ("intra", T2_intra), ("myelin", T2_myelin)):
            if arr is not None and np.ndim(arr) == 0:
                raise ValueError(f"T2_{name}= is per cylinder (an array over the objects); a pool-wide T2 is spelled "
                                 f"compartments=Compartments({name}=Pool(T2=...))")
            if arr is not None and t2[name] is not None:
                raise ValueError(f"T2 of {name} given both through compartments= and T2_{name}=")
        T2_extra, T2_intra, T2_myelin = [t2[n] if a is None else a
                                         for n, a in (("extra", T2_extra), ("intra", T2_intra), ("myelin", T2_myelin))]
        D = _pool_D(self.compartments, "PackedMyelinatedCylinders")
        D_intra, D_myelin, D_extra = D["intra"], D["myelin"], D["extra"]
        inner_radii = np.asarray(inner_radii, dtype=np.float64).ravel()
        N_actual = len(inner_radii)
        g_ratios = np.broadcast_to(
            np.asarray(g_ratios, dtype=np.float64).ravel(), (N_actual,)).copy()
        centers = np.asarray(centers, dtype=np.float64)
        if centers.shape != (N_actual, 2):
            raise ValueError(
                f"centers shape {centers.shape} must be ({N_actual}, 2)")
        if N_max < N_actual:
            raise ValueError(f"N_max={N_max} < N_actual={N_actual}")

        bad = ~(np.isfinite(g_ratios) & (g_ratios > 0.0) & (g_ratios <= 1.0))
        if np.any(bad):
            raise ValueError(f"PackedMyelinatedCylinders: every g-ratio is finite and in (0, 1] (below 1 a myelinated "
                             f"cylinder, 1 a plain one); cylinders {np.flatnonzero(bad).tolist()} have "
                             f"{g_ratios[bad].tolist()}")
        outer_radii = inner_radii / g_ratios
        plain = g_ratios == 1.0
        thin = (~plain) & ~(np.float32(outer_radii) > np.float32(inner_radii))
        if np.any(thin):
            raise ValueError(f"PackedMyelinatedCylinders: a myelinated cylinder (g-ratio below 1) has a sheath of positive "
                             f"thickness; cylinders {np.flatnonzero(thin).tolist()} have inner radii "
                             f"{inner_radii[thin].tolist()} m and outer radii {outer_radii[thin].tolist()} m, which do not "
                             f"differ in float32 (a cylinder without a sheath has g-ratio 1)")
        self.N_actual = N_actual
        self.N_max = N_max
        self._cell_size = float(cell_size)
        from ..substrate.biophysical_constants import get_default_value as _gdv
        self._myelin_proton_density = float(_gdv('myelin_water_proton_density'))

        # Broadcast per-cylinder physics parameters to (N_actual,)
        def _bcast(x, name):
            x = np.asarray(x, dtype=np.float64).ravel()
            if x.size == 1:
                return np.broadcast_to(x, (N_actual,)).copy()
            if x.size != N_actual:
                raise ValueError(
                    f"{name} must be scalar or length {N_actual}, got {x.size}")
            return x.copy()

        def _bcast_kappa(x, name):
            if x is None:
                x = 0.0
            _refuse_schedule("PackedMyelinatedCylinders", **{name: x})
            return _bcast(x, name)

        def _axolemma(x, name):
            """A property of the myelinated cylinders' axolemma, 0 at every plain cylinder (which has none)."""
            per_cylinder = np.ndim(x) > 0 and np.asarray(x).size > 1
            arr = _bcast_kappa(x, name)
            if per_cylinder and np.any(arr[plain] != 0.0):
                raise ValueError(f"{name} is the axolemma of a myelinated cylinder; cylinders "
                                 f"{np.flatnonzero(plain & (arr != 0.0)).tolist()} are plain (g-ratio 1), whose one wall "
                                 f"is the outer wall: give their value in {name.replace('inner', 'outer')} and 0 here")
            return np.where(plain, 0.0, arr)

        def _bcast_opt(x, name):
            if x is None:
                return None
            return _bcast(x, name)

        D_intra_arr   = _bcast(D_intra,  'D_intra')
        D_myelin_arr  = _bcast(D_myelin, 'D_myelin')
        D_extra_arr   = _bcast(D_extra,  'D_extra')
        kappa_inner_arr = _axolemma(kappa_inner, 'kappa_inner') if plain.any() else _bcast_kappa(kappa_inner, 'kappa_inner')
        kappa_outer_arr = _bcast_kappa(kappa_outer, 'kappa_outer')
        rho_inner_arr   = _axolemma(rho_inner, 'rho_inner') if plain.any() else _bcast_kappa(rho_inner, 'rho_inner')
        rho_outer_arr   = _bcast_kappa(rho_outer,   'rho_outer')
        T2_intra_arr  = _bcast_opt(T2_intra,  'T2_intra')
        T2_myelin_arr = _bcast_opt(T2_myelin, 'T2_myelin')
        T2_extra_arr  = _bcast_opt(T2_extra,  'T2_extra')

        # Pad to N_max with zeros
        def _pad(arr):
            out = np.zeros(N_max, dtype=np.float64)
            out[:N_actual] = arr
            return out

        def _pad_opt(arr):
            return None if arr is None else _pad(arr)

        inner_p   = _pad(inner_radii)
        outer_p   = _pad(outer_radii)
        cx_np     = np.zeros(N_max, dtype=np.float64)
        cy_np     = np.zeros(N_max, dtype=np.float64)
        cx_np[:N_actual] = centers[:, 0]
        cy_np[:N_actual] = centers[:, 1]
        centers_padded = np.stack([cx_np, cy_np], axis=1)  # (N_max, 2)

        D_intra_p  = _pad(D_intra_arr)
        D_myelin_p = _pad(D_myelin_arr)
        D_extra_p  = _pad(D_extra_arr)
        T2_intra_p  = _pad_opt(T2_intra_arr)
        T2_myelin_p = _pad_opt(T2_myelin_arr)
        T2_extra_p  = _pad_opt(T2_extra_arr)
        kappa_inner_p = _pad(kappa_inner_arr)
        kappa_outer_p = _pad(kappa_outer_arr)
        rho_inner_p   = _pad(rho_inner_arr)
        rho_outer_p   = _pad(rho_outer_arr)

        #: (N_max,) bool: the plain cylinders (g-ratio 1, no sheath); padding slots are False
        self.plain_mask = np.zeros(N_max, dtype=bool)
        self.plain_mask[:N_actual] = plain

        # Store numpy arrays for init_positions
        self._inner_radii_np = inner_p
        self._outer_radii_np = outer_p
        self._centers_np     = centers_padded  # (N_max, 2)

        # JAX constants (baked at construction, one JIT per N_max)
        self._L_jax            = jnp.float32(cell_size)
        self._L_float          = float(cell_size)
        self._inner_radii_jax  = jnp.array(inner_p,        dtype=jnp.float32)
        self._outer_radii_jax  = jnp.array(outer_p,        dtype=jnp.float32)
        self._centers_jax      = jnp.array(centers_padded, dtype=jnp.float32)
        self._D_intra_jax      = jnp.array(D_intra_p,      dtype=jnp.float32)
        self._D_myelin_jax     = jnp.array(D_myelin_p,     dtype=jnp.float32)
        self._D_extra_jax      = jnp.array(D_extra_p,      dtype=jnp.float32)

        has_t2 = (T2_intra is not None or T2_myelin is not None or
                  T2_extra is not None)
        self._has_t2 = has_t2
        _BIG = np.float32(1e6)
        if has_t2:
            self._T2_intra_jax = jnp.array(
                T2_intra_p  if T2_intra_p  is not None else np.full(N_max, _BIG),
                dtype=jnp.float32)
            self._T2_myelin_jax = jnp.array(
                T2_myelin_p if T2_myelin_p is not None else np.full(N_max, _BIG),
                dtype=jnp.float32)
            self._T2_extra_jax = jnp.array(
                T2_extra_p  if T2_extra_p  is not None else np.full(N_max, _BIG),
                dtype=jnp.float32)

        self._kappa_inner_jax = jnp.array(kappa_inner_p, dtype=jnp.float32)
        self._kappa_outer_jax = jnp.array(kappa_outer_p, dtype=jnp.float32)
        self._rho_inner_jax   = jnp.array(rho_inner_p,   dtype=jnp.float32)
        self._rho_outer_jax   = jnp.array(rho_outer_p,   dtype=jnp.float32)

        # Rotation matrix (shared cylinder axis)
        orientation = np.asarray(orientation, dtype=np.float64)
        self.orientation = (orientation / np.linalg.norm(orientation)).astype(np.float32)
        self._orient_R = acquisition_rotation(self.orientation)

        # EPS/NUDGE — scale by smallest non-zero inner radius
        nonzero = inner_p[inner_p > 0]
        ref_r = float(np.min(nonzero)) if len(nonzero) > 0 else 1e-6
        self._eps   = jnp.float32(1e-7 * ref_r)
        self._nudge = jnp.float32(1e-4 * ref_r)

        # Minimum gap (diagnostic, uses outer radii)
        self.min_gap = self._compute_min_gap()
        if self.min_gap < 0.0:
            raise ValueError(
                f"PackedMyelinatedCylinders: sheaths overlap (closest outer walls are {-self.min_gap:.3g} m "
                "inside each other). Place the axons with pack_myelinated_cylinders(inner_radii, g_ratios, "
                "...), which packs the OUTER radii, rather than pack_cylinders on the inner radii.")

    def permeability_factor_gap(self):
        """``None`` when the cell holds a plain cylinder stated permeable (``kappa_outer > 0``): a walker's factor
        scales the plain cylinders' ``kappa_outer`` (a sealed one stays sealed) and leaves the sheathed cylinders'
        walls at their stated values. Else the reason a permeability per walker is refused: nothing to scale."""
        N = self.N_actual
        plain = np.asarray(self.plain_mask[:N], bool)
        if not plain.any():
            return ("PackedMyelinatedCylinders: a permeability per walker scales the plain cylinders' walls, and this "
                    "cell has no plain cylinder (g-ratio 1); the sheathed cylinders' walls are walked at their stated "
                    "values")
        if not np.any(np.asarray(self._kappa_outer_jax, np.float64)[:N][plain] > 0.0):
            return ("PackedMyelinatedCylinders: a permeability per walker scales the plain cylinders' kappa_outer, and "
                    "every plain cylinder of this cell is stated sealed (kappa_outer = 0): there is nothing to scale")
        return None

    def permeability_factor_rule(self):
        """:data:`PERMEABILITY_FACTOR_RULE`."""
        return PERMEABILITY_FACTOR_RULE

    def trial_objects(self, diffusivity):
        """The cell's cylinders, in their order, when it holds a plain one: ``(radii, D)``, each cylinder's outer
        radius and the larger of its lumen's and the extra pool's diffusivity (a plain wall separates those two; a
        myelinated cylinder's cells of the record stay empty, its trials being the sheathed kind). None for a cell of
        myelinated cylinders only. ``diffusivity`` is not read: each pool steps at its own."""
        if not self.has_plain:
            return None
        N = self.N_actual
        D = np.maximum(np.asarray(self._D_intra_jax, np.float64)[:N], np.asarray(self._D_extra_jax, np.float64)[:N])
        return np.asarray(self._outer_radii_np[:N], np.float64).copy(), D

    def cell_arrays(self):
        """The cell as a backend walks it: a :class:`MyelinatedCell` of read-only copies of the arrays and constants the
        wall kernel is built from (``bounce_budget`` the cell's at the call, which a walk sets to its own)."""
        def ro(x, dtype):
            a = np.array(x, dtype=dtype)
            a.flags.writeable = False
            return a
        return MyelinatedCell(
            centers=ro(self._centers_jax, np.float32), inner_radii=ro(self._inner_radii_jax, np.float32),
            outer_radii=ro(self._outer_radii_jax, np.float32), cell_size=float(self._cell_size),
            D_intra=ro(self._D_intra_jax, np.float32), D_myelin=ro(self._D_myelin_jax, np.float32),
            D_extra=ro(self._D_extra_jax, np.float32), kappa_inner=ro(self._kappa_inner_jax, np.float32),
            kappa_outer=ro(self._kappa_outer_jax, np.float32), plain_mask=ro(self.plain_mask, bool),
            N_actual=int(self.N_actual), N_max=int(self.N_max), eps=float(self._eps), nudge=float(self._nudge),
            min_gap=float(self.min_gap), bounce_budget=(None if self.bounce_budget is None else int(self.bounce_budget)))

    @property
    def has_plain(self):
        """Whether the cell holds a plain cylinder (g-ratio 1)."""
        return bool(self.plain_mask.any())

    @property
    def sheathed(self):
        """``(N_actual,)`` bool: the myelinated cylinders (g-ratio below 1), the sources of the sheath's field."""
        return ~self.plain_mask[:self.N_actual]

    def _compute_min_gap(self):
        """Minimum clear gap between outer boundaries (actual cylinders only)."""
        from .packing import periodic_min_gap
        N = self.N_actual
        return periodic_min_gap(self._centers_np[:N], self._outer_radii_np[:N], self._L_float)

    def volume_fraction(self, compartment: str) -> float:
        """Volume fraction of a named compartment within the periodic cell.

        Parameters
        ----------
        compartment : str
            One of 'intra', 'myelin', or 'extra'.
        """
        L = self._L_float
        N = self.N_actual
        inner = self._inner_radii_np[:N]
        outer = self._outer_radii_np[:N]
        cell_area = L * L
        if compartment == 'intra':
            return float(np.pi * np.sum(inner ** 2) / cell_area)
        elif compartment == 'myelin':
            return float(np.pi * np.sum(outer ** 2 - inner ** 2) / cell_area)
        elif compartment == 'extra':
            total_cyl = np.pi * np.sum(outer ** 2)
            return float((cell_area - total_cyl) / cell_area)
        else:
            raise ValueError(
                f"compartment must be 'intra', 'myelin', or 'extra'; got {compartment!r}")

    def init_positions(self, n_walkers, key):
        """Distribute walkers proportional to compartment area in the periodic cell.

        Walker allocation (area-weighted):
          - Extra-axonal : area = L^2 - sum(pi*R_outer_k^2)
          - Intra_k      : area = pi*R_inner_k^2  (zero for dummy cylinders)
          - Myelin_k     : area = pi*(R_outer_k^2 - R_inner_k^2)

        Dummy cylinders (R=0) and the sheaths of plain cylinders (zero area) receive zero walkers. The walkers are
        placed by area among the seeded pools (:attr:`seeded_pools`) only: a pool that holds no water, or one that
        ``seed_pools`` does not name, holds no walker at the start.
        """
        L = self._L_float
        N = self.N_actual
        N_max = self.N_max
        inner = self._inner_radii_np[:N]
        outer = self._outer_radii_np[:N]
        centers = self._centers_np[:N]

        cell_area = L * L
        area_intra  = np.pi * inner ** 2                  # (N,)
        area_myelin = np.pi * (outer ** 2 - inner ** 2)   # (N,)
        area_extra  = cell_area - np.pi * np.sum(outer ** 2)

        seeded = self.seeded_pools
        if seeded != _POOL_NAMES:                          # the pools not seeded hold no walker at the start
            area_intra = area_intra * ("intra" in seeded)
            area_myelin = area_myelin * ("myelin" in seeded)
            area_extra = area_extra * ("extra" in seeded)
        total_area = area_extra + np.sum(area_intra) + np.sum(area_myelin)
        if not total_area > 0.0:
            raise ValueError(f"the seeded pools {seeded!r} (seed_pools={self.seed_pools!r}) hold no pool of positive "
                             f"area in this cell")

        n_intra  = np.array([int(round(n_walkers * a / total_area))
                             for a in area_intra], dtype=int)
        n_myelin = np.array([int(round(n_walkers * a / total_area))
                             for a in area_myelin], dtype=int)

        # Extra fills remainder (handles rounding); without an extra pool to seed, the largest seeded lumen or sheath does
        n_extra = n_walkers - int(np.sum(n_intra)) - int(np.sum(n_myelin))
        if not area_extra > 0.0 and n_extra != 0:            # largest remainders: whole counts summing to n_walkers
            share = n_walkers * np.concatenate([area_intra, area_myelin]) / total_area
            counts = np.floor(share).astype(int)
            short = n_walkers - int(counts.sum())
            counts[np.argsort(-(share - counts), kind="stable")[:short]] += 1
            n_intra, n_myelin = counts[:N].copy(), counts[N:].copy()
            n_extra = 0
        if n_extra < 0:
            excess = -n_extra
            for k in range(N):
                trim = min(excess, n_intra[k])
                n_intra[k] -= trim
                excess -= trim
                if excess == 0:
                    break
            n_extra = 0

        rng = np.random.default_rng(placement_seed(key))

        positions    = np.zeros((n_walkers, 3), dtype=np.float32)
        compartments = np.zeros(n_walkers, dtype=np.int32)
        idx = 0

        # Intra-axonal: compartment_id = k+1  (1-based, slot 1..N_max)
        for k in range(N):
            nk = int(n_intra[k])
            if nk == 0:
                continue
            r_k = float(inner[k])
            cx_k, cy_k = float(centers[k, 0]), float(centers[k, 1])
            pts = []
            while sum(len(p) for p in pts) < nk:
                batch = max(nk * 4, 64)
                xy = rng.uniform(-r_k, r_k, (batch, 2))
                pts.append(xy[np.linalg.norm(xy, axis=1) < r_k])
            xy_k = np.concatenate(pts, axis=0)[:nk].astype(np.float32)
            # Shift to cylinder centre
            positions[idx:idx + nk, 0] = xy_k[:, 0] + cx_k
            positions[idx:idx + nk, 1] = xy_k[:, 1] + cy_k
            compartments[idx:idx + nk] = k + 1
            idx += nk

        # Myelin: compartment_id = N_max + k + 1  (slots N_max+1..2*N_max)
        for k in range(N):
            nk = int(n_myelin[k])
            if nk == 0:
                continue
            r_in  = float(inner[k])
            r_out = float(outer[k])
            cx_k, cy_k = float(centers[k, 0]), float(centers[k, 1])
            pts = []
            while sum(len(p) for p in pts) < nk:
                batch = max(nk * 4, 64)
                xy = rng.uniform(-r_out, r_out, (batch, 2))
                d = np.linalg.norm(xy, axis=1)
                pts.append(xy[(d >= r_in) & (d < r_out)])
            xy_k = np.concatenate(pts, axis=0)[:nk].astype(np.float32)
            positions[idx:idx + nk, 0] = xy_k[:, 0] + cx_k
            positions[idx:idx + nk, 1] = xy_k[:, 1] + cy_k
            compartments[idx:idx + nk] = N_max + k + 1
            idx += nk

        # Extra-axonal: compartment_id = 0
        n_extra_actual = n_walkers - idx
        if n_extra_actual > 0:
            from .packed import _seed_periodic
            xy_ex = _seed_periodic(n_extra_actual, None, L, outer[:N], centers[:N], 2, "extra", rng=rng)
            positions[idx:idx + n_extra_actual, 0] = xy_ex[:, 0]
            positions[idx:idx + n_extra_actual, 1] = xy_ex[:, 1]
            compartments[idx:idx + n_extra_actual] = 0

        # the float64 accept tests above can round a draw onto a wall in float32: pull every lumen and sheath seed
        # onto its pool's side of its own axon's walls (the extra seeds were pulled by `_seed_periodic`)
        own = np.concatenate([np.repeat(np.arange(N), n_intra), np.repeat(np.arange(N), n_myelin)]).astype(int)
        m = len(own)
        pool = np.where(compartments[:m] <= N_max, 1, 2)
        positions[:m, :2] = _pin_seeds(positions[:m, :2], centers[own], L, inner[own], outer[own], pool, self._nudge)

        r_lab = positions                                   # the substrate frame: xy the cross-section, z = 0

        self._init_compartments = jnp.array(compartments, dtype=jnp.int32)
        return jnp.array(r_lab, dtype=jnp.float32)

    def pool_of(self, encoded):
        """Pool id (0 extra, 1 intra, 2 myelin) of an encoded compartment id
        (0 extra, ``1..N_max`` lumen of axon k, ``N_max+1..2N_max`` its sheath)."""
        enc = jnp.asarray(encoded)
        return jnp.where(enc == 0, jnp.int32(0),
                         jnp.where(enc <= jnp.int32(self.N_max), jnp.int32(1), jnp.int32(2)))

    def classify_position(self, r: jnp.ndarray) -> jnp.ndarray:
        """Encoded compartment id from position: 0 = extra, ``k+1`` = lumen of axon ``k``,
        ``N_max+k+1`` = sheath of axon ``k`` (minimum image in the periodic cell); ``k`` is the cylinder of the
        smallest ``|q|^2 / R_outer^2``, the one whose outer wall holds the point, as the wall kernel reads it.
        :meth:`pool_of` collapses it to the pool id."""
        L = self._L_jax
        q = r[None, :2] - self._centers_jax                          # (N_max, 2)
        q = q - L * jnp.floor(q / L + jnp.float32(0.5))
        d2 = jnp.sum(q * q, axis=1)
        out2 = self._outer_radii_jax * self._outer_radii_jax                  # the cylinder whose outer wall holds r
        k = jnp.argmin(jnp.where(self._outer_radii_jax > 0, d2 / jnp.maximum(out2, _TINY), jnp.inf)).astype(jnp.int32)
        in_lumen = d2[k] < self._inner_radii_jax[k] ** 2
        in_sheath = (~in_lumen) & (d2[k] < self._outer_radii_jax[k] ** 2)
        return jnp.where(in_lumen, k + jnp.int32(1),
                         jnp.where(in_sheath, jnp.int32(self.N_max) + k + jnp.int32(1),
                                   jnp.int32(0)))

    def _wall(self, r, step, kappa_over_D, rho_over_D, key):
        """Not a usable boundary rule for this geometry -- raises.

        This returned `r + step`: free diffusion, no boundaries at all.  Reached through
        `_walk_geometry` it produced an unrestricted walk on a packed axon substrate
        (measured: RMS displacement 1.02x free through 1 um axons) with no error and no
        warning.  The real kernel is `physics.make_packed_myelin_traj_step_fn`, which
        carries the compartment.
        """
        raise NotImplementedError(
            "PackedMyelinatedCylinders has no single-surface reflect: it is stepped by the "
            "fused kernel physics.make_packed_myelin_traj_step_fn, which carries the "
            "compartment id. Use simulate(...) or _walk_geometry(..., "
            "tiers=\"all\"), which dispatch to that kernel.")
