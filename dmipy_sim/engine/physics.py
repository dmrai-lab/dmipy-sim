"""Pure-JAX scan body for Monte Carlo phase accumulation.

make_step_fn returns a closure suitable for jax.lax.scan that captures
the geometry, diffusivity, and dt. The returned function is JIT-compiled
on first call via jax.jit applied in core.py.
"""

import jax
import jax.numpy as jnp
from ..geometry._boundary import bind_probability, dwell_left, dwell_scale
import warnings

import numpy as np

from dmipy_analytic.constants import GAMMA
from ..math.step_law import PAIR_SHORT, PAIR_LONG, PAIR_P_LONG, LONGEST_STEP


def length_scales_of(geometry):
    """The geometry's :class:`~dmipy_sim.geometry.base.LengthScales`: what it declares as ``length_scales``.
    Every geometry the engine walks declares them; there is no other way to read them."""
    ls = getattr(geometry, 'length_scales', None)
    if ls is None:
        raise TypeError(f"{type(geometry).__name__} declares no length_scales; a geometry the engine walks declares "
                        "them (LengthScales(min_feature=..., lookup_cell=..., is_mesh_feature=...))")
    return ls


def _geometry_radius(geometry):
    """Smallest confining length scale (m), or None -- ``length_scales_of(geometry).min_feature``."""
    return length_scales_of(geometry).min_feature


WALKER_INDEX_LIMIT = 2 ** 32
"""Walkers a seed indexes: walker ``i``'s key folds ``i`` in as one 32-bit word."""


def walker_keys(walker_key, first_walker, n_walkers):
    """The step streams of the walkers ``first_walker .. first_walker + n_walkers - 1``: ``fold_in(walker_key, i)``
    for walker ``i``, a function of the key and the index alone (counter-based threefry: the same key words as
    ``split(walker_key, n)[i]``)."""
    first, n = int(first_walker), int(n_walkers)
    if first < 0 or n < 0 or first + n > WALKER_INDEX_LIMIT:
        raise ValueError(f"the walkers {first} .. {first + n - 1} are not within the {WALKER_INDEX_LIMIT} a seed indexes")
    idx = jnp.asarray(np.arange(first, first + n, dtype=np.uint32))
    return jax.vmap(jax.random.fold_in, in_axes=(None, 0))(walker_key, idx)


def seed_keys(seed):
    """``(master_key, position_key, walker_key)`` of a seed: the master key split once into the key the placement
    folds its blocks off and the key the walkers' step streams fold their indices off."""
    master_key = jax.random.PRNGKey(int(seed))
    pos_key, walker_key = jax.random.split(master_key)
    return master_key, pos_key, walker_key


def seed_walkers(geometry, n_walkers, seed, r0=None, *, first_walker=0):
    """The walk's random state from its seed: ``(master_key, r0, walker_keys, labels)`` of the walkers
    ``first_walker .. first_walker + n_walkers - 1`` of the seed.

    The master key splits once into the position key and the walker key. Walker ``i``'s step stream is
    :func:`walker_keys`' and its start the placement's (:func:`~dmipy_sim.geometry.base.placement`; unused when ``r0``
    is given), each a function of the seed and ``i`` alone, so the walkers of a seed are the same walkers however
    they are partitioned into walks. ``labels`` are the placement's seeding labels (a myelinated geometry's), ``None``
    otherwise or when ``r0`` is given. Every engine entry point draws its state here, which is what makes their walks
    one walk; a driver that needs a further independent stream folds it off the master key."""
    from ..geometry.base import initial_positions, placement
    master_key, pos_key, walker_key = seed_keys(seed)
    keys = walker_keys(walker_key, first_walker, n_walkers)
    labels = None
    if r0 is None:
        r0, labels = placement(geometry, n_walkers, pos_key, first_walker)
        if labels is not None:
            geometry._init_compartments = labels
    else:
        r0 = initial_positions(geometry, n_walkers, pos_key, r0)
    return master_key, r0, keys, labels


def pair_step(key):
    """One step of the walk in units of ``sqrt(6 D dt)``, as float32: a direction uniform over the sphere (a standard
    Gaussian triple from ``key``, normalised) times the length the step law draws -- ``PAIR_LONG`` with probability
    ``PAIR_P_LONG``, else ``PAIR_SHORT`` (:mod:`dmipy_sim.math.step_law`: the two lengths that give the step the
    Gaussian step's second, fourth and sixth moments, so the walk's bulk error is third order in ``dt`` and its
    longest step is bounded by ``LONGEST_STEP``)."""
    k_dir, k_len = jax.random.split(key)
    noise = jax.random.normal(k_dir, (3,), dtype=jnp.float32)
    long = jax.random.uniform(k_len, dtype=jnp.float32) < jnp.float32(PAIR_P_LONG)
    length = jnp.where(long, jnp.float32(PAIR_LONG), jnp.float32(PAIR_SHORT))
    return noise * (length / jnp.linalg.norm(noise))


MEMBRANE_COEFFICIENT = 0.33
"""The membrane's first-order coefficient of a permeable walk's exchange, in the step law's units: with the
transmission :func:`~dmipy_sim.geometry._boundary.transmit_probability` the slowest exchange mode of a permeable pore
decays at a rate within ``MEMBRANE_COEFFICIENT * lambda_1 * dt / 5`` of the exact one, ``lambda_1`` the pore's slowest
Neumann rate (``LengthScales.pore_rate``) and ``dt`` the physics step, at EVERY permeability. In ``h = sqrt(6 D dt)``
that is second order, and it vanishes both as ``kappa -> 0`` (no exchange) and as ``kappa -> inf`` (no membrane).

Computed, not fitted: the walk's normal component on a lattice of ``h / 32`` (the pair's two lengths, the membrane on
a cell edge, so reflection maps the lattice to itself) is an exact Markov matrix, and its slowest exchange mode on the
two-compartment slab of half-width ``R`` is compared with the root of ``k tan(k R) = 2 kappa R / D``. Over
``kappa R / D`` from 0.01 to 50 at ``h / R`` = 0.18, 0.09, 0.045 the relative error is negative (the walk exchanges
slightly slower) and largest at ``kappa R / D`` near 1: 0.292, 0.308, 0.317 of ``lambda_1 dt / 5`` (``lambda_1 =
pi^2 D / R^2``), 0.33 at zero step. A permeable wall's step is the step law's pore term with this added to the wall's
own coefficient (:func:`tolerance_sub_steps`).

It bounds every permeability a geometry states, so the step depends on the largest permeability over the objects and
over the walk's time only through whether it is positive. Several membranes of their own permeability: the walk's
error is local to each membrane, its first-order terms add over the membranes, each weighted by the mode's share at
it, with one sign (the walk exchanges slower), and the sum stays under the one-membrane bound -- on three layers
(:class:`~dmipy_sim.math.step_law.LayeredLattice`, widths up to 2:1, each ``kappa w_min / D`` from 0.03 to 30) at most
0.14 of ``lambda_1 dt / 5`` with ``lambda_1 = pi^2 D / w_min^2`` the narrowest layer's, its declared ``pore_rate``;
the bound is reached only as the other membranes seal and one membrane between two layers is left. A permeability
piecewise constant in time: each interval is walked as a constant permeability, and a breakpoint on a step boundary
composes the intervals' one-step operators as the exact propagators compose, so the error is the intervals' own; a
breakpoint inside a step is taken at the nearest step boundary (:func:`membrane_kappa`), an error first order in the
step's duration, second in its length.

The term bounds the slowest mode, which is all the exchange of a walk whose walkers start at spatial equilibrium reads.
From a start out of it the fast modes add :data:`TRANSIENT_COEFFICIENT`'s term while they decay."""


def membrane_kappa(geometry, diffusivity, dt_step):
    """``(kappa_at, record)``: the membrane permeability over ``D`` that each step of a walk reads, from the geometry's
    ``permeability`` and ``permeability_table`` (:func:`~dmipy_sim.geometry.base.permeability_of`); ``(None, None)``
    for walls that never let a walker cross.

    ``kappa_at(step)`` is the float32 ``kappa / D`` of the walk's step ``step`` (the walk-wide index from 0, an int32):
    one number, or one per object when the geometry states a permeability per object. A step of ``dt_step`` seconds
    reads the permeability at its MIDPOINT, ``(step + 1/2) dt_step``: the interval of the schedule that holds that
    time. For a schedule whose breakpoints lie on step boundaries this is the stated schedule exactly, and every
    breakpoint is half a step from every time read, so the rounding of a time cannot move it. A breakpoint inside a
    step takes effect at the step boundary nearest to it (the start rule would take it at the next boundary, a shift
    of up to a whole step, always late): the walk then exchanges ``|Delta kappa| |delta|`` more or less per unit membrane
    area and unit concentration jump than the stated schedule, ``|delta| <= dt_step / 2``, which is first order in the
    step's duration and so second order in its length ``h = sqrt(6 D dt)`` -- the order of the walk's own exchange error
    (:data:`MEMBRANE_COEFFICIENT`) -- and it is warned of and recorded. ``kappa_at`` of a permeability constant in time
    ignores ``step`` (``None`` is then allowed); one that varies refuses ``None``.

    ``record`` is ``None`` for one permeability of every object at every time (the walk is the scalar one), else the
    walk's record of the permeability it walked: ``rule`` (``"midpoint"``), ``breakpoints_s`` and ``values_m_per_s``
    as stated (``(n_intervals, n)``, ``n`` one or the object count), ``step_s``, and ``taken_at_s`` / ``shift_s``:
    the step boundary at which each breakpoint took effect and its distance from the stated time."""
    k = geometry.permeability
    if k is None or not float(k) > 0.0:
        return None, None
    D = float(diffusivity)
    table = geometry.permeability_table
    if table is None:
        c = jnp.float32(float(k) / D)
        return (lambda step: c), None
    vals = np.asarray(table.values, np.float64)
    n = int(vals.shape[1])
    rows = jnp.asarray(vals / D if n > 1 else vals[:, 0] / D, jnp.float32)          # (P, n) or (P,)
    brk = np.asarray(table.breakpoints, np.float64)
    dt = float(dt_step)
    first = np.ceil(brk / dt - 0.5).astype(np.int64)        # the first step whose midpoint is at or past the breakpoint
    taken = first * dt
    shift = taken - brk
    record = dict(rule="midpoint", breakpoints_s=brk.tolist(), values_m_per_s=vals.tolist(), step_s=dt,
                  taken_at_s=taken.tolist(), shift_s=shift.tolist())
    off = np.abs(shift) > 1e-9 * dt
    if np.any(off):
        import warnings
        j = int(np.argmax(np.abs(shift)))
        warnings.warn(
            f"the permeability breakpoint at {brk[j]!r} s falls inside a step of {dt!r} s; the walk takes it at the "
            f"nearest step boundary, {taken[j]!r} s (a shift of {shift[j]:+.3g} s, at most half a step), and so "
            f"exchanges |Delta kappa| |shift| more or less per unit membrane area and unit concentration jump than the "
            f"stated schedule -- second order in the step, the order of the walk's own error. A breakpoint on a step "
            f"boundary (a whole number of the walk's blocks) is taken exactly. See CoefficientWalk.work['permeability'].",
            UserWarning, stacklevel=3)
    if brk.size == 0:
        c = rows[0]
        return (lambda step: c), record
    first_j = jnp.asarray(first, jnp.int32)

    def kappa_at(step):
        if step is None:
            raise ValueError("this permeability varies in time: a step of the walk reads it at the step's index, and "
                             "this caller passed none")
        piece = jnp.sum((jnp.asarray(step, jnp.int32) >= first_j).astype(jnp.int32))
        return rows[piece]
    return kappa_at, record


def has_membrane(geometry) -> bool:
    """Whether a walker on ``geometry`` can cross a wall: a permeability on the geometry, or on a myelinated
    substrate's inner or outer boundary."""
    if geometry.permeability is not None and float(geometry.permeability) > 0.0:
        return True
    if geometry._is_myelinated or geometry._is_packed_myelinated:
        return bool(np.any(np.asarray(geometry._kappa_inner_jax) > 0) or np.any(np.asarray(geometry._kappa_outer_jax) > 0))
    return False


START_OUT_OF_EQUILIBRIUM = "start_out_of_spatial_equilibrium"
"""The flag of a read whose walk started out of spatial equilibrium (:func:`start_record`)."""
START_NOT_ESTABLISHED = "start_not_established"
"""The flag of a read whose walk was handed its starts, so it cannot tell whether they are at equilibrium."""


def start_record(geometry, r0_given: bool) -> dict:
    """What a walk records of its start (``work["start"]``).

    - ``spatial_equilibrium``:
      - True when the walkers start at the walk's stationary density, uniform over every pool they can reach. That holds
        for a placement on a geometry with no permeable wall (each pool it seeds is uniform and keeps its walkers), and
        on one whose placement is uniform over every pool (``Geometry.seeds_every_pool``).
      - False when a permeable wall separates a pool the placement fills from one it leaves empty.
      - None when the walk was handed its starts (``r0``) and cannot tell.
    - ``reason``: why.
    - ``transient_coefficient``: the term a start out of it adds (:data:`TRANSIENT_COEFFICIENT`) when a wall is
      permeable. None without one: no pool exchanges, and the term has nothing to read."""
    membrane = has_membrane(geometry)
    if r0_given:
        eq, why = None, "the walk was handed its starts (r0), whose density it cannot tell"
    elif not membrane:
        eq, why = True, "no wall is permeable: each pool the placement seeds is uniform and keeps its walkers"
    elif bool(geometry.seeds_every_pool):
        eq, why = True, "the placement is uniform over every pool a walker can reach"
    else:
        eq, why = False, "the placement leaves a pool empty that a walker reaches across a permeable wall"
    return dict(spatial_equilibrium=eq, reason=why, transient_coefficient=(TRANSIENT_COEFFICIENT if membrane else None))


def start_error(start, diffusivity: float, dt: float, t: float) -> dict:
    """The start's term of a read at time ``t`` (s since the walkers were placed) of a walk at the physics step ``dt``
    whose start record is ``start`` (:func:`start_record`): ``{"relative_error", "flag"}``.

    - 0 and no flag when the start is at spatial equilibrium, or no wall is permeable.
    - Otherwise :func:`transient_error`, flagged :data:`START_OUT_OF_EQUILIBRIUM` or (a walk handed its starts)
      :data:`START_NOT_ESTABLISHED`.
    - A walk with no start record: ``relative_error`` None, flag ``"start_unrecorded"``."""
    if not start:
        return dict(relative_error=None, flag="start_unrecorded")
    if start.get("transient_coefficient") is None or start.get("spatial_equilibrium") is True:
        return dict(relative_error=0.0, flag=None)
    flag = START_OUT_OF_EQUILIBRIUM if start.get("spatial_equilibrium") is False else START_NOT_ESTABLISHED
    return dict(relative_error=transient_error(diffusivity, dt, t), flag=flag)


def crossing_p_max(geometry, diffusivity: float, dt_sim: float) -> float:
    """The largest probability that one wall hit of a physics step ``dt_sim`` changes a walker's pool: the
    transmission of a hit as deep as the longest step the law draws, ``LONGEST_STEP * sqrt(6 D dt_sim)``, at the
    geometry's largest permeability; 0 without a membrane."""
    if not has_membrane(geometry):
        return 0.0
    kappas = [] if geometry.permeability is None else [float(geometry.permeability)]
    if geometry._is_myelinated or geometry._is_packed_myelinated:
        kappas += [float(np.max(np.asarray(geometry._kappa_inner_jax))), float(np.max(np.asarray(geometry._kappa_outer_jax)))]
    x = 2.0 * max(kappas) / float(diffusivity) * LONGEST_STEP * float(np.sqrt(6.0 * float(diffusivity) * float(dt_sim)))
    return x / (1.0 + x)


def _surface_char_radius(geometry):
    """Characteristic pore radius (m) that sets surface-relaxivity convergence.

    The boundary-local-time (overshoot) estimator is biased at coarse step by
    step_l relative to the pore the RELAXING walkers occupy. This is NOT the
    smallest axon (permeability's scale): confined intra-axonal walkers fully
    sample the inner wall and are accurate at any step, so the binding scale is
    the LARGER extra-axonal pore, ~ 1 / (S_ext/V), declared by the geometry as
    ``length_scales.surface_pore``; otherwise the confining scale is used.
    """
    ls = length_scales_of(geometry)
    return ls.surface_pore if ls.surface_pore is not None else ls.min_feature


def collision_sub_steps(geometry, diffusivity: float, dt: float, frac: float = 0.9) -> int:
    """Sub-steps so one displacement cannot outrun the collision candidate lookup.

    Collision detection is an exact segment-triangle test, but only against the triangles in the 27-cell
    gather around the step's START. A step longer than a cell can therefore leave that box and cross a
    triangle that was never a candidate -- the walker passes through a wall it was never tested against.

    Measured on a closed cylinder at the default waveform resolution (step 1.039 um): at cell 0.533 um
    (step/cell 1.95) 2.0% of walkers leak per step, and refining the mesh to cell 0.250 um (step/cell 4.16)
    takes it to 8.7%. Compounded over a walk that is 45% and 90% of the ensemble. The failure is invisible
    without an independent containment check and gets WORSE as the mesh improves, since the cell size
    scales with the triangle size while the step does not.

    Step length falls as 1/sqrt(n), so n = (L / (frac*cell))^2 keeps each sub-step inside the gather, ``L`` the
    LONGEST step the law draws (``LONGEST_STEP * sqrt(6 D dt)``). ``frac`` under 1 leaves margin for a walker
    sitting at the far edge of its own cell.

    Returns 1 for geometries with no cell grid, and self-limits to 1 once dt already resolves the cell.
    """
    cs = length_scales_of(geometry).lookup_cell
    if cs is None or not np.isfinite(cs) or cs <= 0:
        return 1
    L = LONGEST_STEP * float(np.sqrt(6.0 * float(diffusivity) * float(dt)))
    ratio = L / (float(frac) * float(cs))
    return int(max(1, np.ceil(ratio ** 2))) if ratio > 1.0 else 1


MT_COEFFICIENT = 0.06
"""The binding's coefficient in the step law: with the binding clock and its geminate dwell
(:func:`~dmipy_sim.geometry._boundary.bind_probability`, :func:`~dmipy_sim.geometry._boundary.dwell_scale`), the walk's
bound pool is that of a continuum whose binding reactivity and dwell are each within ``MT_COEFFICIENT (kappa_MT h / D +
h / l)^2`` (relative) of the stated ones, ``h = sqrt(6 D dt)`` and ``l`` the pore's length (:func:`binding_length`).
Second order in the step, as the walk's exchange is (:data:`MEMBRANE_COEFFICIENT`); the ``h / l`` part is the
diffusion's own wall term seen through the rebinding, the ``kappa_MT h / D`` part the binding's.

Computed, not fitted: the walk's normal component on a lattice of ``h / 32`` with a bound copy of each cell near the
membrane is an exact Markov chain of the MT walk (cross first, the binding clock, the dwell counted continuously and
carrying the geminate rebinds, a released walker walking the rest of its step), compared with
``dmipy_analytic.bloch_torrey.BlochTorrey1D`` with bound pools on two layers and one membrane. For each case the error
of the bound pool's complex signal ``E[exp(-(R2_b + i omega_b) t_b)]`` (at two times) and of its stationary fraction
is divided by ``|dX/d ln k_on| + |dX/d ln tau_b|`` (the reference's sensitivities): the largest ratio to ``(kappa_MT h
/ D + h / l)^2`` over 63 cases is 0.056 -- ``kappa h / D`` from 0.1 to 1, ``kappa_MT h / D`` from 0.0125 to 0.4,
dwells of 17 to 66 steps, ``l / h`` from 11 to 44, two steps a factor 2 apart, a walker released inside a step walking
the rest of it or held to the step's end (the fixed-length kernels). The linear law ``min(1, x)`` with a plain dwell,
which this replaces, is first order in the bound pool's signal: its ratio at ``h = 0.09 l`` and ``kappa_MT h / D =
0.0125`` is 9x this one's and halves, rather than quarters, with the step (dmrai-lab/dmipy-sim#921).

Each face's pool (#926), a crossing's bind on either face half each
(:data:`~dmipy_sim.geometry._boundary.NEAR_FACE_SHARE`), measured on the same chain from free walkers on one side of the
membrane, the far face filled only through it, ``kappa_MT h / D = 0.2``, at ``h`` and ``h / 2`` (``h = 0.09 l``) and
``t`` of 0.1 and 0.3 ``l^2 / D``: at ``kappa h / D`` of 0.5, 5 and 50 each face is within this term (a coefficient of
0.04 would do; from 5 on each face carries half the total's error). At ``kappa h / D = 0.045`` the far face's pool needs
up to 0.18 at ``0.1 l^2 / D`` and 0.06 at ``0.3 l^2 / D``, from two parts. One is the start's transient, a relative
error of about ``0.2 h^2 / (D t)`` that does not depend on ``kappa_MT``: :data:`TRANSIENT_COEFFICIENT`'s term, which a
walk states for a start out of spatial equilibrium. The other is the share's second-order remainder -- at a low
permeability the binding clock, ringing early in the encounter, tilts a crossing's binds towards the near face --
about 1.5 times this term at ``0.1 l^2 / D`` and within it by ``0.3 l^2 / D``. It too is a start effect. From walkers
uniform over both layers the crossings each way are as many and alike, and the two faces' pools are the same: at
``0.1 l^2 / D`` they need 0.008. The coefficient covers a walk that starts at equilibrium, spatially and in its pools
(the burn-in's); a start out of spatial equilibrium adds the transient term."""


def binding_length(geometry):
    """The length the binding term of the step law reads (:data:`MT_COEFFICIENT`): ``pi / sqrt(pore_rate)``, the width
    of the slab whose slowest rate is the pore's (a slab's layer itself; 1.51 R for a sphere, 1.70 R for a cylinder),
    else the geometry's ``min_feature``; None for a geometry without either."""
    ls = length_scales_of(geometry)
    if ls.pore_rate:
        return float(np.pi / np.sqrt(float(ls.pore_rate)))
    return None if ls.min_feature is None else float(ls.min_feature)


def binding_error(diffusivity: float, kappa_MT: float, dt: float, length) -> float:
    """The relative error the step law states for the walk's binding at the physics step ``dt``: ``MT_COEFFICIENT
    (kappa_MT h / D + h / l)^2``, ``h = sqrt(6 D dt)``, ``l`` the pore's length (:func:`binding_length`; None drops its
    term). The bound pool's binding reactivity and its dwell are each within it, so an observable ``X`` of the pool is
    within ``eps (|dX/d ln k_on| + |dX/d ln tau_b|)``."""
    h = float(np.sqrt(6.0 * float(diffusivity) * float(dt)))
    a = float(kappa_MT) * h / float(diffusivity) + (h / float(length) if length else 0.0)
    return MT_COEFFICIENT * a * a


def binding_step(diffusivity: float, kappa_MT: float, eps_D: float, length) -> float:
    """The largest physics step (s) at which :func:`binding_error` stays within ``eps_D``: ``h (kappa_MT / D + 1 / l) =
    sqrt(eps_D / MT_COEFFICIENT)``; infinite without binding."""
    if not kappa_MT or kappa_MT <= 0:
        return float("inf")
    D = float(diffusivity)
    per_h = float(kappa_MT) / D + (1.0 / float(length) if length else 0.0)
    h = float(np.sqrt(float(eps_D) / MT_COEFFICIENT)) / per_h
    return h * h / (6.0 * D)


TRANSIENT_COEFFICIENT = 0.3
"""The step law's term for a start out of spatial equilibrium: a read at time ``t`` after the walkers were placed carries,
beyond the law's other terms, a relative error of up to ``TRANSIENT_COEFFICIENT h^2 / (D t)`` (:func:`transient_error`) on
the exchange across a membrane and on each face's bound pool, ``h = sqrt(6 D dt)``.

**Where it comes from.** The law's wall and membrane terms hold a mode's rate to first order in that rate: mode ``n``
decays at a rate within ``c lambda_n dt / 5`` of the exact ``lambda_n``, and the step is sized for the slowest mode
(:data:`MEMBRANE_COEFFICIENT`, the pore term) and the band. A start at spatial equilibrium -- walkers uniform over every
pool they can reach -- puts no weight on the fast modes of the occupancy, and the law's terms are the whole error. A start
out of it, walkers on one side of a membrane, puts a weight ``a_n`` on each mode. At time ``t`` mode ``n`` then contributes
``a_n lambda_n t delta_n exp(-lambda_n t) <= a_n (c dt / (5 t)) (lambda_n t)^2 exp(-lambda_n t)``. Each mode's share is
at most ``4 / e^2`` of ``a_n c dt / (5 t)``, so the sum over the fast modes is ``O(dt / t)``: the non-smooth-data error of
a first-order one-step scheme. In the step's length it is ``C h^2 / (D t)``, second order, and it decays as the transient
does.

**Computed, not fitted.** On the walk's exact chains, from walkers placed uniformly on one side of a membrane between two
layers of width ``l`` with reflecting outer walls, at ``t`` from 0.005 to 0.3 ``l^2 / D``:

- **The exchanged fraction** (:class:`~dmipy_sim.math.step_law.LayeredLattice` against the two-layer eigenfunction series):
  ``|error| D t / h^2`` is at most 0.084 over ``kappa l / D`` from 0.05 to 50 at ``h / l`` of 0.09, 0.045 and 0.0225. That
  includes the slowest mode's own term.
- **Each face's bound pool** (the binding term's chain, :data:`MT_COEFFICIENT`, against
  ``dmipy_analytic.bloch_torrey.BlochTorrey1D``). At ``kappa h / D = 0.045`` the far face's is 0.17 at ``h = 0.09 l``
  and 0.24 at ``h = 0.011 l`` for ``t <= 0.1 l^2 / D``, the same at ``kappa_MT h / D`` of 0.05 and 0.2. It is at most 0.10
  at ``kappa h / D = 0.5`` and 0.04 at 5. The near face's and the free pool's are at most 0.06.

The coefficient is their envelope.

**A face's local time on its own is first order out of equilibrium.** The walk's local time samples the density over a
step's reach. A density gradient ``g`` normal to a face therefore moves the face's local-time rate by about
``0.5 h g / c`` (on the same lattice, at ``h / 64`` as at ``h / 32``). At a membrane the exchange sets
``g = kappa [c] / D``. From a one-sided start:

- the far face's rate is low by about ``0.48 h / sqrt(D t)``;
- the near face's is high by about ``0.5 kappa h / D``.

At an impermeable wall, and at equilibrium, ``g = 0``. Two rules take this term out of the observables above:

- the exchange's transmission ``x / (1 + x)``;
- each face's binding: a crossing binds on either face half each (:data:`~dmipy_sim.geometry._boundary.NEAR_FACE_SHARE`),
  which makes up the far face's local-time deficit.

So a read of one face's local time from a start out of equilibrium is not covered by this term.

**The regime.** The step is sized when a walk starts, before any read's time is known, so this term is stated, not
stepped for. A walk records whether its start is at spatial equilibrium (:func:`start_record`, its ``work["start"]``).
A read of a walk whose start is not carries this term and a flag
(:meth:`~dmipy_sim.coefficient_walk.CoefficientWalk.start_error`). ``t`` is the time since the walkers were placed: a
binding walk's burn-in precedes its ``t = 0``, so a ``t`` counted from 0 overstates the term."""


def transient_error(diffusivity: float, dt: float, t: float) -> float:
    """The relative error :data:`TRANSIENT_COEFFICIENT` states for a read at time ``t`` (s since the walkers were placed)
    of a walk at the physics step ``dt`` from a start out of spatial equilibrium: ``TRANSIENT_COEFFICIENT h^2 / (D t)``,
    ``h^2 = 6 D dt``; infinite at ``t <= 0``."""
    del diffusivity                                 # h^2 / (D t) = 6 dt / t
    t = float(t)
    return float("inf") if t <= 0.0 else TRANSIENT_COEFFICIENT * 6.0 * float(dt) / t


def mt_sub_steps(geometry, diffusivity: float, dt: float, dwell_time: float,
                 frac: float = 8.0, dwell_frac: float = 20.0, kappa_MT: float = None, eps_D: float = None) -> int:
    """Sub-steps for an emergent-MT (surface-binding) walk.

    MT's free->bound rate is not imposed; it emerges from the boundary local time accumulated at wall
    encounters, and the per-encounter probability is ``1 - exp(-(kappa_MT/D) * local_time)``. So what the
    step size has to resolve is *the local time*, and -- on a mesh -- the *encounters themselves*. It does
    NOT have to resolve a crossing.

    The geometry criterion is therefore not a pore-size rule (``R/25`` binds the step to a length the physics does
    not set): the linearisation needs ``p_stick << 1`` (``p_stick ~ 1e-5`` at canonical parameters), a mesh's
    ``_geometry_radius`` is its ``feature_radius``, a mesh-resolution parameter and not a pore, and on the canonical
    well-mixed sphere the emergent equilibrium bound fraction sits within 0.43% of the analytic ``k_f/(k_f+k_r)``
    at every setting from 1 to 38 sub-steps. MCMRSimulator, which implements the same emergent model, derives its
    binding timestep from the binding rate and the dwell time alone.

    The geometry criterion is dispatched to whichever one the engine already uses for this class of geometry:

    * **Mesh-like** (anything with a ``cell_size``): :func:`collision_sub_steps`. A step that outruns the
      27-cell collision lookup misses wall encounters outright, and a missed encounter contributes no
      local time -- so the binding rate is under-counted, not merely noisy, and it fails SILENTLY to a
      plausible-looking number rather than raising. Measured on an R=2um mesh sphere (cell 0.10um,
      f_b=0.3333): n_sub=4 gives a bound fraction of exactly 0.0000 -- at step_l=0.245um the search misses
      the wall entirely, so nothing ever binds -- n_sub=8 is 12.1% low, and it converges from n_sub=16
      (-0.36%, -0.71% at the 30 this rule picks, -0.06% at 40). It is the collision criterion, not R/25,
      that sets where this converges, which is also why the rule must never fall below it.
    * **Analytic**: ``step_l = R/frac`` with ``frac=8``, the same boundary-local-time accuracy target
      surface relaxivity uses (~0.1 pp bias) -- reflection is exact at any step, so only the local-time
      estimator's bias matters. Written out here rather than delegating to
      :func:`surface_sub_steps` so that disabling surface sub-stepping (``surface_substep_frac=0``, used
      for long qualitative CPMG forwards) cannot silently disable MT's.

    Plus a floor so the rule is physics-aware and not purely geometric: release is tested once per
    sub-step, so the dwell must span ``dwell_frac`` of them. It does not bind at realistic parameters
    (~15,000 sub-steps per dwell for a CACTUS-scale mesh) and costs nothing when it does not.

    And the step law's binding term (:data:`MT_COEFFICIENT`, :func:`binding_step`) when ``kappa_MT`` is given: the
    binding's own error, second order in the step, held within ``eps_D`` (default :data:`STEP_TOLERANCE`).
    """
    if length_scales_of(geometry).lookup_cell:
        n = collision_sub_steps(geometry, diffusivity, dt)
    else:
        R = _geometry_radius(geometry)
        if R is None:
            n = 1
        else:
            dt_phys_max = (R / float(frac)) ** 2 / (6.0 * diffusivity)
            n = max(1, int(np.ceil(dt / dt_phys_max)))
    if dwell_time and dwell_time > 0:
        n = max(n, int(np.ceil(dt * float(dwell_frac) / float(dwell_time))))
    if kappa_MT:
        dt_b = binding_step(diffusivity, kappa_MT, STEP_TOLERANCE if eps_D is None else eps_D, binding_length(geometry))
        n = max(n, int(np.ceil(float(dt) / dt_b * (1.0 - 1e-12))))
    return max(1, int(n))


STEP_TOLERANCE = 1e-3
"""The relative error of the walk's diffusion spectrum ``D(w)`` the physics step is sized for, up to the band the
walk serves (:func:`tolerance_sub_steps`): every mode a measurement reads decays at a rate within this of the
exact one, so a row's signal error is below ``STEP_TOLERANCE * |S ln S| <= STEP_TOLERANCE / e``, under the
Monte-Carlo floor of every pack of the substrate bank (dmrai-lab/dmipy-sim#714, #715)."""


def tolerance_sub_steps(geometry, diffusivity: float, dt: float, *, eps_D: float = STEP_TOLERANCE,
                        f_band: float = None) -> int:
    """Sub-steps so the walk's diffusion spectrum is within ``eps_D`` up to the band ``f_band`` (Hz).

    The walk's step law (:mod:`dmipy_sim.math.step_law`) leaves two first-order terms and one bound, and the
    physics step ``dt_p`` is the smallest of the three:

    * the curved wall, ``dt_p = 5 eps_D / (c lambda_1)`` (:func:`~dmipy_sim.math.step_law.pore_step`), ``lambda_1``
      the pore's slowest rate and ``c`` its wall coefficient, both declared by the geometry (``LengthScales.pore_rate``
      and ``wall_coefficient``; exact for the analytic pores, the envelope of a sphere of ``min_feature`` and 0.85 otherwise);
      a wall a walker can cross (:func:`has_membrane`) adds the membrane's coefficient, :data:`MEMBRANE_COEFFICIENT`,
      the two first-order errors of one mode adding;
    * the band, the largest step at which the pair's bulk error on the mode a band of ``f_band`` reads stays within
      ``eps_D`` (:func:`~dmipy_sim.math.step_law.band_step`); ``f_band`` is the Nyquist frequency of the step ``dt``,
      ``1 / (2 dt)``, when a driver states none (the fused drivers state the waveform's band, the band accumulation
      ``K / (2 T)``, ``walk`` the scanner class's);
    * the first-order regime, ``delta_0 <= min_feature / 2`` (the longest step 0.82 of it), where the law is measured;
      a geometry that declares its own ceiling (``reflection_step_fraction``: the curved tubes' two-bounce wall) keeps it.

    A geometry whose ``min_feature`` is a resolution and not a pore declares ``pore_rate = 0`` (a label volume's
    voxel) and has no pore term; the band term and the ceiling size its step, and the pilot of a build measures its
    error.

    For a MESH it is :func:`collision_sub_steps` instead, because a pore rule cannot be written on a meshing
    parameter: ``min_feature`` is ``feature_radius``, a MESH-RESOLUTION parameter, so a pore rule on it tightened as a
    mesh was refined while the pore stayed the size it always was. What bounds a mesh step is the 27-cell collision
    lookup -- outrun it and the wall is missed entirely. Measured on the 366-fibre CACTUS bundle (feature_radius
    0.186um, cell 0.124um, block 0.1ms, D=2e-9): a pore rule on the feature asked for 1256 sub-steps, the
    collision criterion for 97, and across 97 -> 314 the apparent perpendicular diffusivity scatters +/-1.3% with no
    trend, the accumulated boundary local time moves +0.16%, and containment is flat (97.45% -> 97.02%). A mesh's
    step error is what the two-step pilot of its build measures (#715).
    """
    from ..math.step_law import pore_step, band_step, WALL_COEFFICIENT_ENVELOPE, PORE_RATE_ENVELOPE
    ls = length_scales_of(geometry)
    R = ls.min_feature
    n_coll = 1
    if ls.lookup_cell:
        n_coll = collision_sub_steps(geometry, diffusivity, dt)
        # A cell grid is not the same thing as a mesh. For a MESH the collision criterion REPLACES the pore rule,
        # because `min_feature` there is `feature_radius` -- a meshing parameter, not a pore size. For an ANALYTIC
        # geometry that merely carries a spatial index (PackedCurvedCylinders buckets tube SEGMENTS so a step tests
        # ~50 candidates instead of millions), R is a real physical radius and the two criteria bound DIFFERENT
        # failures, so both apply: the cell (a step must not outrun the 27-cell candidate gather) and the pore rule.
        # Keying this branch on `cell_size` alone silently dropped the pore rule for PackedCurvedCylinders: on the
        # DiSCo substrate (R_min 0.72um, cell 6.5um) it asked for 1 sub-step where the pore rule asks for 106,
        # i.e. step/R ~ 1.7 -- walkers stepping straight through tube walls.
        if ls.is_mesh_feature or R is None:
            return n_coll
    if R is None:
        # No scale found means "nothing to resolve", which is right for free diffusion and WRONG for anything
        # with walls -- and it fails silently, at one sub-step, with the boundary channel garbled. That is how
        # a dropped `length` clause went unnoticed for a whole release (see `_geometry_radius`). A confined
        # geometry advertises finite volume and surface area, so say so rather than guessing.
        try:
            sa, vol = geometry.surface_area, geometry.volume
        except AttributeError:                               # a geometry that states no walls
            sa = vol = None
        confined = False
        if sa is not None and vol is not None:               # one that states them is read; a failure to read them raises
            sa = sa() if callable(sa) else sa
            vol = vol() if callable(vol) else vol
            confined = float(sa) > 0 and 0 < float(vol) < float('inf')
        if confined:
            warnings.warn(
                f"{type(geometry).__name__} exposes walls (finite surface_area and volume) but no length "
                f"scale that tolerance_sub_steps recognises, so the walk runs at ONE sub-step "
                f"(step_l = sqrt(6*D*dt)). Boundary local time, and any surface T2 fitted from it, will be "
                f"wrong if that step is comparable to the pore. Expose `radius` (or `length`) on the "
                f"geometry, or pass sub_steps explicitly.", UserWarning, stacklevel=3)
        return 1
    D = float(diffusivity)
    R = float(R)
    rate = ls.pore_rate if ls.pore_rate is not None else PORE_RATE_ENVELOPE / R ** 2
    c = ls.wall_coefficient if ls.wall_coefficient is not None else WALL_COEFFICIENT_ENVELOPE
    if has_membrane(geometry):
        c = c + MEMBRANE_COEFFICIENT
    pore = pore_step(rate, D, eps_D, c)                      # infinite at a declared rate of 0: no confining mode
    if f_band is None:
        f_band = 0.5 / float(dt)
    frac = getattr(geometry, 'reflection_step_fraction', None)
    delta_max = R / float(frac) if frac else R / 2.0
    dt_p = min(pore, band_step(f_band, eps_D), delta_max ** 2 / (6.0 * D))
    return max(n_coll, max(1, int(np.ceil(float(dt) / dt_p * (1.0 - 1e-12)))))      # an exact multiple is not one more

def regime_sub_steps(geometry, diffusivity: float, dt: float) -> int:
    """Sub-steps so the step stays in the first-order regime the step law is measured in: ``delta_0 <= min_feature / 2``,
    or the ceiling a geometry declares (``reflection_step_fraction``); 1 for a geometry without a scale. The ceiling
    alone, which :func:`tolerance_sub_steps` folds in; the step pilot of a build reads it to keep its coarse walk in
    the regime (dmrai-lab/dmipy-sim#715)."""
    ls = length_scales_of(geometry)
    if ls.min_feature is None or ls.is_mesh_feature:
        return 1
    frac = getattr(geometry, 'reflection_step_fraction', None)
    delta_max = float(ls.min_feature) / (float(frac) if frac else 2.0)
    return max(1, int(np.ceil(float(dt) / (delta_max ** 2 / (6.0 * float(diffusivity))) * (1.0 - 1e-12))))


def surface_sub_steps(geometry, diffusivity: float, dt: float, frac: float = 8.0) -> int:
    """Fine sub-steps so a surface-relaxivity walk resolves the boundary local time.

    Targets step_l ≈ R_char / ``frac`` with R_char the extra-axonal pore
    (:func:`_surface_char_radius`) — the pore the relaxing walkers occupy, which is
    coarser than permeability's min-axon R/25 (the confined intra lumen is already
    exact). ``frac=8`` gives a ${\\sim}0.1$-pp boundary-local-time bias; ``n_sub`` is
    self-limiting (→1 once the waveform dt already resolves the pore).

    The resolution is controllable per geometry via the ``surface_substep_frac``
    attribute (overrides ``frac``); ``0`` removes this criterion (the reflection and lookup
    criteria of :func:`resolve_sub_steps` still apply). A mesh declares no pore -- its
    ``min_feature`` is a meshing parameter -- so this criterion is 1 there and the lookup
    criterion bounds the step.
    """
    g_frac = getattr(geometry, 'surface_substep_frac', None)
    if g_frac is not None:
        frac = g_frac
    if not frac or frac <= 0:
        return 1
    ls = length_scales_of(geometry)
    if ls.is_mesh_feature and ls.surface_pore is None:
        # a meshing parameter is not a pore; the lookup criterion bounds a mesh step
        return 1
    Rc = _surface_char_radius(geometry)
    if Rc is None:
        return 1
    step_target = Rc / frac
    dt_phys_max = step_target ** 2 / (6.0 * diffusivity)   # step_l = sqrt(6 D dt)
    return max(1, int(np.ceil(dt / dt_phys_max)))


def _warn_if_step_outruns_the_lookup(geometry, diffusivity, dt, n_sub, what):
    """Warn when a sub-step is longer than the collision lookup can serve on a mesh.

    The candidate lookup gathers only the 27 cells around a step's START, so a step longer than a cell crosses
    triangles that were never candidates and the wall is missed outright. On the reflecting path
    `reject_escape` catches most of that; on the PERMEABLE path nothing does, because a compartment change is
    legitimate there.

    Measured on a permeable mesh sphere (R=5 um, subdivisions=4, cell 30.2 nm) at kappa=1e-14, where no walker
    may legitimately cross, per 1 ms: step/cell 0.25 -> 0.00% escaped, 0.89 -> 0.00%, 1.79 -> 0.05%,
    3.31 -> 4.85%, 6.63 -> 26.2%. The engine's own rules stay below 1 cell, so this fires only when a caller
    overrides `sub_steps` (or sets `cell_size`) into unsound territory -- which is silent otherwise, and cost
    a long detour to diagnose once (dmrai-lab/dmipy-sim#65).
    """
    cell = length_scales_of(geometry).lookup_cell
    if not cell:
        return
    step_l = LONGEST_STEP * float(np.sqrt(6.0 * diffusivity * dt / max(n_sub, 1)))
    ratio = step_l / float(cell)
    if ratio > 0.9:
        warnings.warn(
            f"{what}: the longest sub-step {step_l:.3e} m is {ratio:.2f} x the collision-lookup cell "
            f"({float(cell):.3e} m). A step longer than a cell crosses triangles that were never gathered as "
            f"candidates, so walls are missed: measured 4.85% of walkers lost per ms at 3.3 cells and 26% at "
            f"6.6 cells on a permeable mesh sphere. Increase sub_steps (or do not override it).",
            UserWarning, stacklevel=3)



#: Sub-steps per block / waveform step above which the resolver refuses: a substrate whose smallest
#: feature demands more than this is degenerate far more often than it is intended, and a walk at that
#: cost blocks a user for hours before telling them. ``sub_steps=`` overrides deliberately.
MAX_SUB_STEPS = 100_000

def sub_step_rules(geometry, diffusivity: float, dt: float, *, surface: bool = False, mt_dwell_time=None,
                   mt_kappa=None, eps_D: float = STEP_TOLERANCE, f_band: float = None) -> dict:
    """The sub-step count each criterion that applies to a walk on ``geometry`` asks for, by name -- what
    :func:`resolve_sub_steps` takes the maximum of, and what a walk records as its ``step_rule``."""
    rules = {"step law": tolerance_sub_steps(geometry, diffusivity, dt, eps_D=eps_D, f_band=f_band),
             "regime": regime_sub_steps(geometry, diffusivity, dt),
             "collision lookup": collision_sub_steps(geometry, diffusivity, dt)}
    if surface:
        rules["surface local time"] = surface_sub_steps(geometry, diffusivity, dt)
    if mt_dwell_time is not None:
        rules["binding"] = mt_sub_steps(geometry, diffusivity, dt, mt_dwell_time, kappa_MT=mt_kappa, eps_D=eps_D)
    return rules


def resolve_sub_steps(geometry, diffusivity: float, dt: float, *, surface: bool = False,
                      mt_dwell_time=None, mt_kappa=None, override=None, eps_D: float = STEP_TOLERANCE,
                      f_band: float = None) -> int:
    """Fine sub-steps per waveform step (a walk's block) for a walk on ``geometry``.

    Every driver -- the fused scan (:func:`make_step_fn`, :func:`make_packed_myelin_step_fn`),
    the trajectory producer (:func:`~dmipy_sim.engine.core._walk_geometry`), the vector-Bloch
    engine and the MT walker -- takes its count from here, so one substrate is walked at one
    resolution whichever way it is driven. The count is the maximum over the criteria that apply
    (:func:`sub_step_rules`):

    * the step law, the walk's diffusion spectrum within ``eps_D`` up to the band ``f_band``
      (:func:`tolerance_sub_steps`: the curved-wall term on the pore's slowest mode, with the membrane's
      coefficient added when a wall is permeable, the pair's bulk term on the band's mode, the first-order regime);
      ``f_band`` is the step's Nyquist frequency when a driver states none; not applied when ``min_feature`` is a
      meshing parameter;
    * collision lookup, ``LONGEST_STEP * step_l <= 0.9 * lookup_cell`` (:func:`collision_sub_steps`), for a
      spatially indexed geometry;
    * surface local time, ``step_l <= surface_pore / 8`` (:func:`surface_sub_steps`), when
      ``surface`` -- the walk records or applies a boundary local time;
    * binding, :func:`mt_sub_steps`, when ``mt_dwell_time`` is given, with the step law's binding term when
      ``mt_kappa`` (the binding reactivity, m/s) is.

    ``override`` pins the count. An unbounded walk resolves to 1.
    """
    if override:
        return int(override)
    rules = sub_step_rules(geometry, diffusivity, dt, surface=surface, mt_dwell_time=mt_dwell_time, mt_kappa=mt_kappa,
                           eps_D=eps_D, f_band=f_band)
    name, n = max(rules.items(), key=lambda kv: kv[1])
    n = max(1, int(n))
    if n > MAX_SUB_STEPS:
        ls = length_scales_of(geometry)
        feat = ls.min_feature
        gap = ls.min_gap
        step = float(np.sqrt(6.0 * float(diffusivity) * float(dt) / n))
        raise ValueError(
            f"{type(geometry).__name__} needs {n:,} sub-steps per {dt:.3g} s step (the {name} rule: a "
            f"{step:.2g} m step for a smallest feature of {feat * 1e9:.1f} nm"
            + (f", narrowest gap {gap * 1e9:.1f} nm" if gap is not None and np.isfinite(gap) else "") +
            f"). That is {n / MAX_SUB_STEPS:.0f}x the cap of {MAX_SUB_STEPS:,} and almost always a degenerate "
            f"substrate -- a radius drawn far below histology (floor the diameter law, e.g. outer diameter "
            f">= 0.4 um) or a packing with a vanishing gap. Fix the substrate, or pass sub_steps= to walk it "
            f"anyway at that cost.")
    return n


def make_step_fn(geometry, diffusivity: float, dt: float, T2: float = None,
                 T1: float = None, sub_steps: int = None, track_compartment: bool = False, f_band: float = None):
    """Return ``(step_fn, has_weight)``: the fused scan body for one waveform timestep.

    Each step consumes ``(g_t, chi_t)`` or ``(g_t, chi_t, k_t)``: the gradient sample, a binary
    transverse-coherence flag and the sample's index from 0. When ``chi_t == 1`` the magnetisation is
    transverse (T2 decay and surface relaxivity act); when ``chi_t == 0`` it is stored longitudinally
    (only T1 acts). A plain spin echo passes ``chi_t == 1`` throughout. A permeability that varies in
    time needs ``k_t``: sub-step ``j`` of sample ``k_t`` is the walk's step ``k_t n_sub + j``, which
    reads the permeability at its midpoint (:func:`membrane_kappa`).

    The carry is ``(r, phi, log_w, key, comp)`` for every geometry and effect::

        step_fn(carry, (g_t, chi_t[, k_t])) -> (carry, None)

    ``comp`` is the walker's compartment id (see ``Geometry.classify_position``). It is advanced
    with ``geometry.classify_position_carry`` when the geometry has per-compartment bulk
    properties or when ``track_compartment`` is set, and passed through untouched otherwise, so a
    walk that needs no label pays nothing for it. Per-compartment D is taken from the compartment
    the walker is in when the sub-step starts; per-compartment 1/T2 and 1/T1 from the one it ends
    in. Surface relaxivity, permeability and relaxation all accrue in ``log_w`` per fine sub-step.

    Parameters
    ----------
    geometry : Geometry
        Provides ``reflect`` and, when it has a surface relaxivity, ``reflect_with_log_weight``;
        when it has a permeability, ``permeate``.
    diffusivity : float
        Diffusion coefficient in m²/s (``None`` for a geometry with per-compartment D).
    dt : float
        ScannerSequence time step in seconds.
    T2, T1 : float, optional
        Bulk relaxation times in seconds. ``-chi_t * dt / T2`` and ``-(1 - chi_t) * dt / T1``
        accrue in ``log_w`` each step.
    sub_steps : int, optional
        Pin the sub-step count; otherwise :func:`resolve_sub_steps` chooses it. The count is
        exposed as ``step_fn.n_sub``.
    track_compartment : bool
        Advance ``comp`` every sub-step even when no per-compartment property needs it.

    Returns
    -------
    step_fn : callable
    has_weight : bool
        True when something writes ``log_w`` (surface relaxivity, permeability, T2 or T1); a
        caller may skip ``exp(log_w)`` otherwise.
    """
    _D_arr     = geometry._D_comp_jax        # per-compartment D, indexed by pool id, or None
    _invT2_arr = geometry._inv_T2_comp_jax
    _invT1_arr = geometry._inv_T1_comp_jax
    per_comp   = any(a is not None for a in (_D_arr, _invT2_arr, _invT1_arr))
    carry_comp = per_comp or track_compartment
    _D0 = diffusivity if diffusivity is not None else geometry._D_comp_max

    has_surf = geometry.surface_relaxivity_t2 is not None
    has_perm = geometry.permeability          is not None
    has_t2   = (T2 is not None) or (_invT2_arr is not None)
    has_t1   = (T1 is not None) or (_invT1_arr is not None)
    has_weight = has_surf or has_perm or has_t2 or has_t1

    _inv_T2 = jnp.float32(1.0 / T2) if T2 is not None else jnp.float32(0.0)
    _inv_T1 = jnp.float32(1.0 / T1) if T1 is not None else jnp.float32(0.0)

    n_sub = resolve_sub_steps(geometry, float(_D0), dt, surface=has_surf, override=sub_steps, f_band=f_band)
    _warn_if_step_outruns_the_lookup(geometry, float(_D0), dt, n_sub, 'walk')
    dt_sub       = dt / n_sub
    gamma_dt_sub = jnp.float32(GAMMA * dt_sub)
    dt_sub_f32   = jnp.float32(dt_sub)
    one          = jnp.float32(1.0)

    def _D_at(comp):
        return _D_arr[comp] if _D_arr is not None else jnp.float32(_D0)

    def _t2_decrement(comp):
        return dt_sub_f32 * (_invT2_arr[comp] if _invT2_arr is not None else _inv_T2)

    def _t1_decrement(comp):
        return dt_sub_f32 * (_invT1_arr[comp] if _invT1_arr is not None else _inv_T1)

    if has_perm:
        # D is single across a permeable wall (unequal D is rejected at construction), so κ/D and
        # ρ/D use the single diffusivity; the permeability of a step is the one membrane_kappa reads.
        kappa_at, _ = membrane_kappa(geometry, float(_D0), dt / n_sub)
        rho_over_D   = (jnp.float32(geometry.surface_relaxivity_t2 / float(_D0))
                        if has_surf else jnp.float32(0.0))
        interact = geometry.interact               # absorbs a geometry that also reports crossing / refusal

        def _move(r, step, comp, key, step_index):
            hit = interact(r, step, kappa_over_D=kappa_at(step_index), rho_over_D=rho_over_D, key=key)
            return hit.r, hit.dlog_w

    elif has_surf:
        rho_nom = jnp.float32(geometry.surface_relaxivity_t2)
        reflect_with_log_weight = geometry.reflect_with_log_weight

        def _move(r, step, comp, key, step_index):
            return reflect_with_log_weight(r, step, rho_nom / _D_at(comp))

    else:
        reflect = geometry.reflect

        def _move(r, step, comp, key, step_index):
            return reflect(r, step), jnp.float32(0.0)

    carry_fn = geometry.classify_position_carry

    def step_fn(carry, inputs):
        g_t, chi_t = inputs[0], inputs[1]
        k_t = inputs[2] if len(inputs) > 2 else None

        def _sub(c, j):
            r, phi, log_w, key, comp = c
            if has_perm:
                key, k_step, k_wall = jax.random.split(key, 3)
            else:
                key, k_step = jax.random.split(key)
                k_wall = None
            step = pair_step(k_step) * jnp.sqrt(6.0 * _D_at(comp) * dt_sub_f32)

            r_new, dlog_w = _move(r, step, comp, k_wall, None if k_t is None else k_t * n_sub + j)
            comp_new = carry_fn(r_new, comp) if carry_comp else comp

            # surface relaxivity accrues only while transverse
            dlog_w = dlog_w * chi_t
            if has_t2:
                dlog_w = dlog_w - _t2_decrement(comp_new) * chi_t
            if has_t1:
                dlog_w = dlog_w - _t1_decrement(comp_new) * (one - chi_t)

            phi_new = phi + gamma_dt_sub * jnp.dot(g_t, r_new)
            return (r_new, phi_new, log_w + dlog_w, key, comp_new), None

        carry_out, _ = jax.lax.scan(_sub, carry, jnp.arange(n_sub, dtype=jnp.int32))
        return carry_out, None

    step_fn.n_sub = n_sub
    return step_fn, has_weight


def _pool_and_axon(geometry, comp_id):
    """Decode an encoded compartment id into ``(pool, k)``: 0 extra, ``k+1`` lumen of axon k,
    ``N_max+k+1`` sheath of axon k. Extra walkers carry the nearest axon as ``k``."""
    pool = geometry.pool_of(comp_id)
    k = jnp.where(pool == 1, comp_id - 1,
                  jnp.where(pool == 2, comp_id - jnp.int32(geometry.N_max) - 1, jnp.int32(0)))
    return pool, jnp.maximum(k, 0).astype(jnp.int32)


def _encode_compartment(geometry, pool, k):
    return jnp.where(pool == 0, jnp.int32(0),
                     jnp.where(pool == 1, k + 1, jnp.int32(geometry.N_max) + k + 1)).astype(jnp.int32)


def _add_work(work, w, keep=None):
    """``work + w`` elementwise for the carried ``(hits, crossings, exhausted)``; an entry that is None (hits and
    crossings when not counting) stays None, exhaustion is always summed. ``keep`` masks the step's work out where
    False (a bound walker's step that did not happen)."""
    if work is None or w is None:
        return work
    out = []
    for a, b in zip(work, w):
        if a is None or b is None:
            out.append(None); continue
        out.append(a + (jnp.where(keep, b, jnp.int32(0)) if keep is not None else b))
    return tuple(out)


def make_myelin_substep(geometry, dt: float, rho_weights=None, f_band: float = None):
    """The one displacement-and-wall rule for the concentric-cylinder substrates.

    Serves :class:`MyelinatedCylinder` (one axon, open extra-axonal space) and
    :class:`PackedMyelinatedCylinders` (a periodic pack); the wall physics is
    :func:`dmipy_sim.geometry.myelin.concentric_wall_kernel`. Each pool steps with its own
    diffusivity (the axon's, indexed by the carried compartment); the axial coordinate is free.

    Returns ``sub(r, step_key, u, comp_id) -> (r_new, comp_id_new, chan, dlog_rho, work)`` where
    ``chan`` holds the four unit boundary local-time channels of the wall kernel, ``dlog_rho``
    the surface log-weight increment under ``rho_weights`` (``(N_max, 4)``; zero when None), and ``work``
    the step's ``(hits, crossings, exhausted)`` as int32 -- the first two None unless the geometry counts
    (``count_walls``), the exhaustion always.
    """
    dt_f32 = jnp.float32(dt)
    N_max = geometry.N_max
    L = geometry._L_jax if geometry._is_packed_myelinated else None
    D_i, D_m, D_e = geometry._D_intra_jax, geometry._D_myelin_jax, geometry._D_extra_jax
    step_i = jnp.sqrt(jnp.float32(6.0) * D_i * dt_f32)
    step_m = jnp.sqrt(jnp.float32(6.0) * D_m * dt_f32)
    step_e = jnp.sqrt(jnp.float32(6.0) * D_e * dt_f32)
    D_max = float(max(np.max(np.asarray(D_i)), np.max(np.asarray(D_m)), np.max(np.asarray(D_e))))
    step_max = float(np.sqrt(6.0 * D_max * dt))
    if rho_weights is None:
        rho_weights = jnp.zeros((N_max, 4), jnp.float32)
    from ..geometry.myelin import concentric_wall_kernel
    wall = concentric_wall_kernel(
        geometry._centers_jax, geometry._inner_radii_jax, geometry._outer_radii_jax, L,
        D_i, D_m, D_e, geometry._kappa_inner_jax, geometry._kappa_outer_jax, rho_weights,
        geometry._eps, geometry._nudge, step_max, geometry.min_gap, count=geometry.count_walls,
        budget=geometry.bounce_budget)
    def sub(r, step_key, u, comp_id):
        pool, k = _pool_and_axon(geometry, comp_id)
        unit = pair_step(step_key)
        step_l = jnp.where(pool == 1, step_i[k], jnp.where(pool == 2, step_m[k], step_e[k]))
        r_c = r
        s_c = unit * step_l
        l_xy = jnp.linalg.norm(s_c[:2])
        d_hat = jnp.where(l_xy > 0, s_c[:2] / jnp.maximum(l_xy, jnp.float32(1e-30)),
                          jnp.zeros(2, jnp.float32))
        h = wall(r_c[:2], d_hat, l_xy, pool, k, u)
        xy_new = h.xy
        if L is not None:
            xy_new = xy_new - L * jnp.floor(xy_new / L + jnp.float32(0.5))      # stay in the cell
        r_new = jnp.stack([xy_new[0], xy_new[1], r_c[2] + s_c[2]])
        counting = h.n_hits is not None
        work = (h.n_hits, h.crossed.astype(jnp.int32) if counting else None, h.exhausted.astype(jnp.int32))
        return r_new, _encode_compartment(geometry, h.pool, h.k), h.chan, h.dlog_rho, work

    sub.bounce_loop = wall.bounce_loop
    sub.n_cand = wall.n_cand
    return sub


def make_myelin_traj_step_fn(geometry, dt: float):
    """Trajectory step for the isolated :class:`MyelinatedCylinder`: the open-domain sibling of
    :func:`make_packed_myelin_traj_step_fn`, stepped by the same substep
    (:func:`make_myelin_substep`).

    Carry ``(r, r_unwrapped, key, dlog_accum, comp_id, work)`` -- the same shape the packed trajectory
    step carries, so :func:`~dmipy_sim.engine.core._walk_geometry` records both through one
    recording branch; ``work`` is the carried ``(hits, crossings, exhausted)`` or None. There is no periodic cell to unwrap: ``r`` already is the continuous
    position, so ``r_unwrapped`` repeats it rather than tracking a minimum-image wrap.
    ``dlog_accum`` accumulates the unit boundary local time (``rho/D = 1``), as the packed step
    does. No magnetization transfer: binding is wired to the packed myelin walk only
    (:func:`make_packed_myelin_traj_step_fn`).
    """
    sub = make_myelin_substep(geometry, dt)

    def step_fn(carry, _):
        r, r_uw, key, dlog_accum, comp_id, work = carry
        key, k_step, k_perm = jax.random.split(key, 3)
        u = jax.random.uniform(k_perm, dtype=jnp.float32)
        r_new, comp_new, chan, _, w = sub(r, k_step, u, comp_id)
        dlog_boundary = jnp.sum(chan)
        return (r_new, r_new, key, dlog_accum + dlog_boundary, comp_new, _add_work(work, w)), None

    step_fn.bounce_loop = sub.bounce_loop
    return step_fn


def make_myelin_step_fn(geometry, dt: float, T1: float = None, sub_steps: int = None, f_band: float = None):
    """Fused forward step for :class:`MyelinatedCylinder`.

    Carry ``(r, phi, log_w, compartment_id, key)`` with the compartment a pool id (0 extra,
    1 intra, 2 myelin); inputs ``(g_t, chi_t)``. Per-pool T2 accrues while transverse
    (``chi_t == 1``), T1 while stored. Sub-stepped by :func:`resolve_sub_steps`; the count is
    exposed as ``step_fn.n_sub``.
    """
    D_max = float(max(geometry.D_intra, geometry.D_myelin, geometry.D_extra))
    n_sub = resolve_sub_steps(geometry, D_max, dt, override=sub_steps, f_band=f_band)
    dt_sub = dt / n_sub
    gamma_dt_sub = jnp.float32(GAMMA * dt_sub)
    dt_sub_f32 = jnp.float32(dt_sub)
    sub = make_myelin_substep(geometry, dt_sub)

    has_t2 = any(t is not None for t in (geometry.T2_intra, geometry.T2_myelin, geometry.T2_extra))
    if has_t2:
        _big = 1e6
        inv_t2_by_pool = jnp.array([1.0 / (geometry.T2_extra or _big), 1.0 / (geometry.T2_intra or _big),
                                    1.0 / (geometry.T2_myelin or _big)], jnp.float32)
    has_t1 = T1 is not None
    inv_T1 = jnp.float32(1.0 / T1) if has_t1 else jnp.float32(0.0)

    def step_fn(carry, inputs):
        g_t, chi_t = inputs

        def _sub(c, _):
            r, phi, log_w, comp, key = c
            key, k_step, k_perm = jax.random.split(key, 3)
            u = jax.random.uniform(k_perm, dtype=jnp.float32)
            r_new, comp_new, _, _, _w = sub(r, k_step, u, comp)
            dlog = jnp.float32(0.0)
            if has_t2:
                dlog = dlog - dt_sub_f32 * inv_t2_by_pool[comp_new] * chi_t
            if has_t1:
                dlog = dlog - dt_sub_f32 * inv_T1 * (jnp.float32(1.0) - chi_t)
            return (r_new, phi + gamma_dt_sub * jnp.dot(g_t, r_new), log_w + dlog, comp_new, key), None

        carry_out, _ = jax.lax.scan(_sub, carry, None, length=n_sub)
        return carry_out, None

    step_fn.n_sub = n_sub
    return step_fn


def _unwrapped_step(r_unwrapped, r_incell, r_incell_new, L):
    """The continuous position after one sub-step of a walker in a periodic ``(x, y)`` cell of side ``L``.

    ``r_incell -> r_incell_new`` is the step as the cell holds it, wrapped into ``[-L/2, L/2)`` in-plane; its
    in-plane displacement is the minimum image (a sub-step is far shorter than ``L/2``), and the axial coordinate,
    which the cell never wraps, is the in-cell one. Elementwise arithmetic, no matmul, so the position is exact in
    float32 on any device.
    """
    dxy = r_incell_new[:2] - r_incell[:2]
    dxy = dxy - L * jnp.round(dxy / L)
    return jnp.array([r_unwrapped[0] + dxy[0], r_unwrapped[1] + dxy[1], r_incell_new[2]], dtype=jnp.float32)


def make_packed_myelin_traj_step_fn(geometry, dt: float,
                                    kappa_MT: float = 0.0, dwell_time: float = 0.0,
                                    mt_side_intra: float = 1.0, mt_side_extra: float = 1.0):
    """Trajectory step for PackedMyelinatedCylinders: geometry + permeability, no relaxation.

    Carry ``(r_incell, r_unwrapped, key, dlog_accum, comp_id, work)`` -- or ``(r_incell, r_unwrapped, key,
    dlog_accum, comp_id, bound_rem, bound_acc, work)`` when ``kappa_MT > 0``; ``step_fn(carry, None) -> (carry,
    None)``; ``work`` is the carried ``(hits, crossings, exhausted)`` or None (a bound walker's step adds nothing). ``r_incell`` is the position in the cell ``[-L/2, L/2)`` that the wall kernel reads;
    ``r_unwrapped`` is the continuous position (:func:`_unwrapped_step`), the one a trajectory records. ``dlog_accum``
    accumulates the unit boundary local time (``-2 d_perp`` per reflection, i.e. ``rho/D = 1``) so
    a replay can apply any surface relaxivity. ``comp_id`` is the encoded compartment (0 extra,
    ``1..N_max`` lumen of axon k, ``N_max+1..2N_max`` its sheath).

    Magnetization transfer (``kappa_MT > 0``): free water that reflects off a myelin wall binds
    with probability ``1 - exp(-(kappa_MT/D) * local_time)``, freezes for an exponential dwell of
    mean ``dwell_time`` and is released; ``bound_acc`` is the time bound, in steps. An intra walker
    only ever meets the inner wall and an extra walker the outer one, so the walker's pool selects
    the side reactivity (``mt_side_intra`` / ``mt_side_extra``); myelin water cannot bind. A bound
    or binding encounter contributes nothing to ``dlog_accum``.
    """
    mt_on = kappa_MT > 0.0
    kappa_intra_f = jnp.float32(kappa_MT * mt_side_intra)
    kappa_extra_f = jnp.float32(kappa_MT * mt_side_extra)
    dwell_steps_mean = jnp.float32(dwell_time / dt) if dwell_time > 0 else jnp.float32(0.0)
    sub = make_myelin_substep(geometry, dt)
    D_intra, D_extra = geometry._D_intra_jax, geometry._D_extra_jax

    L = jnp.float32(geometry._cell_size)

    def step_fn(carry, _):
        if mt_on:
            r, r_uw, key, dlog_accum, comp_id, bound_rem, bound_acc, work = carry
            key, k_step, k_perm, stick_key, dwell_key = jax.random.split(key, 5)
        else:
            r, r_uw, key, dlog_accum, comp_id, work = carry
            key, k_step, k_perm = jax.random.split(key, 3)
        u = jax.random.uniform(k_perm, dtype=jnp.float32)
        r_new, comp_new, chan, _, w = sub(r, k_step, u, comp_id)
        r_uw_new = _unwrapped_step(r_uw, r, r_new, L)
        dlog_boundary = jnp.sum(chan)
        if not mt_on:
            return (r_new, r_uw_new, key, dlog_accum + dlog_boundary, comp_new, _add_work(work, w)), None

        pool, k = _pool_and_axon(geometry, comp_id)
        is_intra, is_extra = pool == 1, pool == 0
        is_bound = bound_rem > jnp.float32(0.0)
        local_time = jnp.where(is_intra, -chan[0], jnp.where(is_extra, -chan[3], jnp.float32(0.0)))
        D_bind = jnp.where(is_intra, D_intra[k], jnp.where(is_extra, D_extra[k], jnp.float32(1.0)))
        kappa_bind = jnp.where(is_intra, kappa_intra_f,
                               jnp.where(is_extra, kappa_extra_f, jnp.float32(0.0)))
        kod_bind = kappa_bind / jnp.maximum(D_bind, jnp.float32(1e-30))
        p_stick = bind_probability(kod_bind, local_time)
        newly = (~is_bound) & (jax.random.uniform(stick_key, dtype=jnp.float32) < p_stick)
        u_dwell, u_when = jax.random.uniform(dwell_key, (2,), dtype=jnp.float32)
        dwell_draw = (-jnp.log(jnp.maximum(u_dwell, jnp.float32(1e-20))) * dwell_steps_mean
                      * dwell_scale(kod_bind, local_time))             # with the encounter's geminate rebinds
        # the encounter falls uniformly in the step: the dwell left at its end is the draw less the time from the
        # encounter to the end (a dwell shorter than that is over within the step)
        dwell_left_ = dwell_left(dwell_draw, jnp.float32(1.0), u_when)
        r_out = jnp.where(is_bound, r, r_new)
        r_uw_out = jnp.where(is_bound, r_uw, r_uw_new)
        comp_out = jnp.where(is_bound, comp_id, comp_new)
        dlog_contrib = jnp.where(is_bound | newly, jnp.float32(0.0), dlog_boundary)
        bound_rem_out = jnp.where(is_bound, bound_rem - jnp.float32(1.0),
                                  jnp.where(newly, dwell_left_, jnp.float32(0.0)))
        # the time bound within the step: the dwell left, up to the step, of a bound walker; from the encounter on,
        # for one that binds
        bound_acc_out = bound_acc + jnp.where(is_bound, jnp.minimum(bound_rem, jnp.float32(1.0)),
                                              jnp.where(newly, jnp.minimum(dwell_draw, u_when), jnp.float32(0.0)))
        return (r_out, r_uw_out, key, dlog_accum + dlog_contrib, comp_out, bound_rem_out, bound_acc_out,
                _add_work(work, w, keep=~is_bound)), None

    step_fn.bounce_loop = sub.bounce_loop
    return step_fn


def make_packed_myelin_step_fn(geometry, dt: float, T1: float = None, f_band: float = None):
    """Fused forward signal step for PackedMyelinatedCylinders.

    The walk of :func:`make_packed_myelin_traj_step_fn` with, in the same scan, the gradient
    phase on the continuous (periodic-unwrapped) position, per-axon per-pool T2, surface
    relaxivity from the per-axon inner / outer wall values, and T1 on the stored intervals.

    Carry ``(r_incell, r_unwrapped, phi, log_w, compartment_id, key)``; inputs ``(g_t, chi_t)``.
    Sub-stepped by :func:`resolve_sub_steps`; the count is ``step_fn.n_sub``.
    """
    L = jnp.float32(geometry._cell_size)
    D_i, D_m, D_e = geometry._D_intra_jax, geometry._D_myelin_jax, geometry._D_extra_jax
    rho_i, rho_o = geometry._rho_inner_jax, geometry._rho_outer_jax
    has_rho = bool(np.max(np.asarray(rho_i)) > 0 or np.max(np.asarray(rho_o)) > 0)

    def _over(rho, D):
        return jnp.where(D > 0, rho / jnp.maximum(D, jnp.float32(1e-30)), jnp.float32(0.0))
    rho_weights = (jnp.stack([_over(rho_i, D_i), _over(rho_i, D_m), _over(rho_o, D_m), _over(rho_o, D_e)],
                             axis=1) if has_rho else None)                       # (N_max, 4)

    has_t2 = geometry._has_t2
    if has_t2:
        inv_t2 = [jnp.float32(1.0) / geometry._T2_extra_jax, jnp.float32(1.0) / geometry._T2_intra_jax,
                  jnp.float32(1.0) / geometry._T2_myelin_jax]                    # by pool, per axon
    has_t1 = T1 is not None
    inv_T1 = jnp.float32(1.0 / T1) if has_t1 else jnp.float32(0.0)

    D_ref = float(max(np.max(np.asarray(D_i)), np.max(np.asarray(D_e))))
    n_sub = resolve_sub_steps(geometry, D_ref, dt, surface=has_rho, f_band=f_band)
    dt_sub = dt / n_sub
    sub = make_myelin_substep(geometry, dt_sub, rho_weights=rho_weights)
    gamma_dt_sub = jnp.float32(GAMMA * dt_sub)
    dt_sub_f32 = jnp.float32(dt_sub)

    def step_fn(carry, inputs):
        g_t, chi_t = inputs

        def _sub(c, _):
            r_ic, r_uw, phi, log_w, cid, key = c
            key, k_step, k_perm = jax.random.split(key, 3)
            u = jax.random.uniform(k_perm, dtype=jnp.float32)
            r_ic_new, cid_new, _, dlog_rho, _w = sub(r_ic, k_step, u, cid)
            r_uw_new = _unwrapped_step(r_uw, r_ic, r_ic_new, L)
            phi_new = phi + gamma_dt_sub * (g_t @ r_uw_new)

            dlog = jnp.float32(0.0)
            if has_t2:
                pool, k = _pool_and_axon(geometry, cid_new)
                inv = jnp.where(pool == 0, inv_t2[0][k], jnp.where(pool == 1, inv_t2[1][k], inv_t2[2][k]))
                dlog = dlog - dt_sub_f32 * inv * chi_t
            if has_t1:
                dlog = dlog - dt_sub_f32 * inv_T1 * (jnp.float32(1.0) - chi_t)
            if has_rho:
                dlog = dlog + dlog_rho * chi_t
            return (r_ic_new, r_uw_new, phi_new, log_w + dlog, cid_new, key), None

        carry_out, _ = jax.lax.scan(_sub, carry, None, length=n_sub)
        return carry_out, None

    step_fn.n_sub = n_sub
    step_fn.bounce_loop = sub.bounce_loop
    return step_fn
