"""Pure-JAX scan body for Monte Carlo phase accumulation.

make_step_fn returns a closure suitable for jax.lax.scan that captures
the geometry, diffusivity, and dt. The returned function is JIT-compiled
on first call via jax.jit applied in core.py.
"""

import jax
import jax.numpy as jnp
from .geometry._boundary import transmit_probability, bind_probability
import warnings

import numpy as np

from .constants import GAMMA


def length_scales_of(geometry):
    """The geometry's :class:`~dmipy_sim.geometry.base.LengthScales`.

    Shipped geometries declare them. An object that is not a :class:`Geometry` is read through
    the legacy attributes (``radius``, ``sphere_radius``, ``length``, ``_radii_np``,
    ``_inner_radii_np``, ``cell_size``, ``radius_is_mesh_feature``), here and nowhere else.
    """
    ls = getattr(geometry, 'length_scales', None)
    if ls is not None:
        return ls
    from .geometry.base import LengthScales
    R = getattr(geometry, 'radius', None)
    if R is None:
        R = getattr(geometry, 'sphere_radius', None)
    if R is None:
        R = getattr(geometry, 'length', None)
    if R is None:
        radii = getattr(geometry, '_radii_np', None)
        if radii is not None and len(radii) > 0:
            R = float(np.min(radii))
    if R is None:
        inner = getattr(geometry, '_inner_radii_np', None)
        if inner is not None and len(inner) > 0 and np.any(inner > 0):
            R = float(np.min(inner[inner > 0]))
    cell = getattr(geometry, 'cell_size', None)
    return LengthScales(min_feature=None if R is None else float(R),
                        lookup_cell=None if not cell else float(cell),
                        is_mesh_feature=bool(getattr(geometry, 'radius_is_mesh_feature', False)))


def _geometry_radius(geometry):
    """Smallest confining length scale (m), or None -- ``length_scales_of(geometry).min_feature``."""
    return length_scales_of(geometry).min_feature


def permeable_sub_steps(geometry, diffusivity: float, dt: float) -> int:
    """Number of fine sub-steps so a permeable walk resolves membrane crossing.

    Impermeable reflection is exact at any step (step_l = R/6 suffices), but
    membrane *crossing* over-permeates at coarse steps — the transmission needs
    the near-membrane motion spatially resolved (step_l ≈ R/25 for <1% bias).
    Returns 1 when
    no radius scale is available (free diffusion).
    """
    R = _geometry_radius(geometry)
    if R is None:
        return 1
    dt_phys_max = R ** 2 / (3750.0 * diffusivity)   # step_l = R/25 (6·25²)
    return max(1, int(np.ceil(dt / dt_phys_max)))


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

    Step length falls as 1/sqrt(n), so n = (L / (frac*cell))^2 keeps each sub-step inside the gather.
    ``frac`` under 1 leaves margin for a walker sitting at the far edge of its own cell.

    Returns 1 for geometries with no cell grid, and self-limits to 1 once dt already resolves the cell.
    """
    cs = length_scales_of(geometry).lookup_cell
    if cs is None or not np.isfinite(cs) or cs <= 0:
        return 1
    L = float(np.sqrt(6.0 * float(diffusivity) * float(dt)))
    ratio = L / (float(frac) * float(cs))
    return int(max(1, np.ceil(ratio ** 2))) if ratio > 1.0 else 1


def mt_sub_steps(geometry, diffusivity: float, dt: float, dwell_time: float,
                 frac: float = 8.0, dwell_frac: float = 20.0) -> int:
    """Sub-steps for an emergent-MT (surface-binding) walk.

    MT's free->bound rate is not imposed; it emerges from the boundary local time accumulated at wall
    encounters, and the per-encounter probability is ``min(1, (kappa_MT/D) * local_time)``. So what the
    step size has to resolve is *the local time*, and -- on a mesh -- the *encounters themselves*. It does
    NOT have to resolve a crossing, which is what permeability's much finer rule is for.

    The rule this replaces was ``step_l = R/25``, justified as "binding freezes walkers
    (trajectory-altering, like permeability)". Three things were wrong with that:

    * It is geometric, while the binding physics is not. The linearisation needs ``p_stick << 1``; at
      canonical parameters ``p_stick ~ 1e-5``, four orders below where it would matter. MCMRSimulator --
      which implements the same emergent model -- derives its binding timestep purely from the binding
      rate and the dwell time, with no length scale, and does NOT tighten its geometric term when MT is
      enabled (its permeability term is likewise geometry-free, so the heritage was geometry-free at
      source).
    * For a Mesh, ``_geometry_radius`` returns ``feature_radius`` -- a MESH-RESOLUTION parameter, not a
      pore. The same physical sphere therefore demanded 38 sub-steps as an analytic geometry and 6610 as a
      mesh, and refining the mesh multiplied the cost quadratically for no physical reason.
    * Measured, it bought nothing on the analytic side: on the canonical well-mixed sphere the emergent
      equilibrium bound fraction sits within 0.43% of the analytic ``k_f/(k_f+k_r)`` at EVERY setting from
      1 to 38 sub-steps, with no trend, at 8.9x the wall time.

    So the geometry criterion is dispatched to whichever one the engine already uses for this class of
    geometry, rather than inventing a third:

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

    Residual known bias, deliberately not chased: a fractional dwell remainder is rounded up to a whole
    sub-step, biasing ``bound_frac`` high by ~``0.5*dt_sim/dwell_time`` per binding event -- 0.002% to
    0.02% at realistic parameters, 0.33% with a 0.1 ms dwell. MCMRSimulator avoids it by releasing on a
    continuous fraction of a step; not worth the complexity here at that magnitude.
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
    return max(1, int(n))


def walk_sub_steps(geometry, diffusivity: float, dt: float) -> int:
    """Sub-steps for a plain diffusion walk (no MT, no surface tier of its own).

    For an ANALYTIC pore this is the historical ``step_l = R/6`` (``R/25`` when the wall is permeable, since
    the crossing probability is step-size sensitive and over-permeates at coarse steps). Reflection off an
    analytic surface is exact at any step, so the criterion only has to keep a step from skipping the pore.

    For a MESH it is :func:`collision_sub_steps` instead, because ``R/6`` is not a physical criterion there:
    ``_geometry_radius`` returns ``feature_radius``, a MESH-RESOLUTION parameter, so the rule tightened as a
    mesh was refined while the pore stayed the size it always was. What actually bounds a mesh step is the
    27-cell collision lookup -- outrun it and the wall is missed entirely. This is the same substitution
    :func:`mt_sub_steps` makes for the binding walk, for the same reason.

    Measured on the 366-fibre CACTUS bundle (feature_radius 0.186um, cell 0.124um, dt_save 0.1ms, D=2e-9):
    the old rule asked for 1256 sub-steps, the collision criterion for 97, and across 97 -> 314 the apparent
    perpendicular diffusivity scatters +/-1.3% with no trend, the accumulated boundary local time moves
    +0.16%, and containment is flat (97.45% -> 97.02%). So the observables are converged at 97 and the extra
    13x was buying nothing.

    A PERMEABLE mesh deliberately keeps the fine analytic rule: the crossing probability
    ``p = 2(kappa/D) d_perp`` is step-size sensitive in a way the collision criterion says nothing about, and
    that regime has not been measured here.
    """
    has_perm = getattr(geometry, 'permeability', None) is not None
    ls = length_scales_of(geometry)
    R = ls.min_feature
    n_coll = 1
    if ls.lookup_cell and not has_perm:
        n_coll = collision_sub_steps(geometry, diffusivity, dt)
        # A cell grid is not the same thing as a mesh. For a MESH the collision criterion
        # REPLACES R/6, because `_geometry_radius` there returns `feature_radius` -- a
        # meshing parameter, not a pore size (that is the point of the substitution above).
        # For an ANALYTIC geometry that merely carries a spatial index (PackedCurvedTubes
        # buckets tube SEGMENTS so a step tests ~50 candidates instead of millions), R is a
        # real physical radius and the two criteria bound DIFFERENT failures, so both apply:
        #   * cell   -- a step must not outrun the 27-cell candidate gather (wall never tested)
        #   * R/6    -- a step must not break single-reflection-per-step (the analytic
        #               reflection math itself; see CurvedTube's class docstring)
        # Keying this branch on `cell_size` alone silently dropped the R/6 rule for
        # PackedCurvedTubes: on the DiSCo substrate (R_min 0.72um, cell 6.5um) it asked for
        # 1 sub-step where the analytic rule asks for 106, i.e. step/R ~ 1.7 -- walkers
        # stepping straight through tube walls.
        if ls.is_mesh_feature or R is None:
            return n_coll
    if R is None:
        # No scale found means "nothing to resolve", which is right for free diffusion and WRONG for anything
        # with walls -- and it fails silently, at one sub-step, with the boundary channel garbled. That is how
        # a dropped `length` clause went unnoticed for a whole release (see `_geometry_radius`). A confined
        # geometry advertises finite volume and surface area, so say so rather than guessing.
        try:
            sa, vol = geometry.surface_area, geometry.volume
            sa = sa() if callable(sa) else sa
            vol = vol() if callable(vol) else vol
            confined = float(sa) > 0 and 0 < float(vol) < float('inf')
        except Exception:
            confined = False
        if confined:
            warnings.warn(
                f"{type(geometry).__name__} exposes walls (finite surface_area and volume) but no length "
                f"scale that walk_sub_steps recognises, so the walk runs at ONE sub-step "
                f"(step_l = sqrt(6*D*dt)). Boundary local time, and any surface T2 fitted from it, will be "
                f"wrong if that step is comparable to the pore. Expose `radius` (or `length`) on the "
                f"geometry, or pass sub_steps explicitly.", UserWarning, stacklevel=3)
        return 1
    divisor = 3750.0 if has_perm else 216.0
    dt_phys_max = float(R) ** 2 / (divisor * diffusivity)
    return max(n_coll, max(1, int(np.ceil(dt / dt_phys_max))))


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
    step_l = float(np.sqrt(6.0 * diffusivity * dt / max(n_sub, 1)))
    ratio = step_l / float(cell)
    if ratio > 0.9:
        warnings.warn(
            f"{what}: sub-step length {step_l:.3e} m is {ratio:.2f} x the collision-lookup cell "
            f"({float(cell):.3e} m). A step longer than a cell crosses triangles that were never gathered as "
            f"candidates, so walls are missed: measured 4.85% of walkers lost per ms at 3.3 cells and 26% at "
            f"6.6 cells on a permeable mesh sphere. Increase sub_steps (or do not override it).",
            UserWarning, stacklevel=3)


def resolve_sub_steps(geometry, diffusivity: float, dt: float, *, surface: bool = False,
                      mt_dwell_time=None, override=None) -> int:
    """Fine sub-steps per waveform (or save) step for a walk on ``geometry``.

    Every driver -- the fused scan (:func:`make_step_fn`, :func:`make_packed_myelin_step_fn`),
    the trajectory producer (:func:`~dmipy_sim.core.simulate_trajectories`), the vector-Bloch
    engine and the MT walker -- takes its count from here, so one substrate is walked at one
    resolution whichever way it is driven. The count is the maximum over the criteria that apply:

    * reflection, ``step_l <= min_feature / 6`` (:func:`walk_sub_steps`), or ``min_feature / 25``
      when the wall is permeable, since the crossing probability is step-size sensitive; not applied
      when ``min_feature`` is a meshing parameter;
    * collision lookup, ``step_l <= 0.9 * lookup_cell`` (:func:`collision_sub_steps`), for a
      spatially indexed geometry;
    * surface local time, ``step_l <= surface_pore / 8`` (:func:`surface_sub_steps`), when
      ``surface`` -- the walk records or applies a boundary local time;
    * binding, :func:`mt_sub_steps`, when ``mt_dwell_time`` is given.

    ``override`` pins the count. An unbounded walk resolves to 1.
    """
    if override:
        return int(override)
    n = walk_sub_steps(geometry, diffusivity, dt)
    n = max(n, collision_sub_steps(geometry, diffusivity, dt))
    if surface:
        n = max(n, surface_sub_steps(geometry, diffusivity, dt))
    if mt_dwell_time is not None:
        n = max(n, mt_sub_steps(geometry, diffusivity, dt, mt_dwell_time))
    return max(1, int(n))


def make_step_fn(geometry, diffusivity: float, dt: float, T2: float = None,
                 T1: float = None, sub_steps: int = None, track_compartment: bool = False):
    """Return ``(step_fn, has_weight)``: the fused scan body for one waveform timestep.

    Each step consumes ``(g_t, chi_t)``: the gradient sample and a binary transverse-coherence
    flag. When ``chi_t == 1`` the magnetisation is transverse (T2 decay and surface relaxivity
    act); when ``chi_t == 0`` it is stored longitudinally (only T1 acts). A plain spin echo
    passes ``chi_t == 1`` throughout.

    The carry is ``(r, phi, log_w, key, comp)`` for every geometry and effect::

        step_fn(carry, (g_t, chi_t)) -> (carry, None)

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
        Waveform time step in seconds.
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

    n_sub = resolve_sub_steps(geometry, float(_D0), dt, surface=has_surf, override=sub_steps)
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
        # ρ/D use the single diffusivity.
        kappa_over_D = jnp.float32(geometry.permeability / float(_D0))
        rho_over_D   = (jnp.float32(geometry.surface_relaxivity_t2 / float(_D0))
                        if has_surf else jnp.float32(0.0))
        permeate = geometry.permeate

        def _move(r, step, comp, key):
            r_new, dlog_w = permeate(r, step, kappa_over_D, rho_over_D, key)
            return r_new, dlog_w

    elif has_surf:
        rho_nom = jnp.float32(geometry.surface_relaxivity_t2)
        reflect_with_log_weight = geometry.reflect_with_log_weight

        def _move(r, step, comp, key):
            return reflect_with_log_weight(r, step, rho_nom / _D_at(comp))

    else:
        reflect = geometry.reflect

        def _move(r, step, comp, key):
            return reflect(r, step), jnp.float32(0.0)

    carry_fn = geometry.classify_position_carry

    def step_fn(carry, inputs):
        g_t, chi_t = inputs

        def _sub(c, _):
            r, phi, log_w, key, comp = c
            if has_perm:
                key, k_step, k_wall = jax.random.split(key, 3)
            else:
                key, k_step = jax.random.split(key)
                k_wall = None
            noise = jax.random.normal(k_step, (3,), dtype=jnp.float32)
            step = (noise / jnp.linalg.norm(noise)) * jnp.sqrt(6.0 * _D_at(comp) * dt_sub_f32)

            r_new, dlog_w = _move(r, step, comp, k_wall)
            comp_new = carry_fn(r_new, comp) if carry_comp else comp

            # surface relaxivity accrues only while transverse
            dlog_w = dlog_w * chi_t
            if has_t2:
                dlog_w = dlog_w - _t2_decrement(comp_new) * chi_t
            if has_t1:
                dlog_w = dlog_w - _t1_decrement(comp_new) * (one - chi_t)

            phi_new = phi + gamma_dt_sub * jnp.dot(g_t, r_new)
            return (r_new, phi_new, log_w + dlog_w, key, comp_new), None

        carry_out, _ = jax.lax.scan(_sub, carry, None, length=n_sub)
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


def make_myelin_substep(geometry, dt: float, rho_weights=None):
    """The one displacement-and-wall rule for the concentric-cylinder substrates.

    Serves :class:`MyelinatedCylinder` (one axon, open extra-axonal space) and
    :class:`PackedMyelinatedCylinders` (a periodic pack); the wall physics is
    :func:`dmipy_sim.geometry.myelin.concentric_wall_kernel`. Each pool steps with its own
    diffusivity (the axon's, indexed by the carried compartment); the axial coordinate is free.

    Returns ``sub(r, step_key, u, comp_id) -> (r_new, comp_id_new, chan, dlog_rho)`` where
    ``chan`` holds the four unit boundary local-time channels of the wall kernel and ``dlog_rho``
    the surface log-weight increment under ``rho_weights`` (``(N_max, 4)``; zero when None).
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
    from .geometry.myelin import concentric_wall_kernel
    wall = concentric_wall_kernel(
        geometry._centers_jax, geometry._inner_radii_jax, geometry._outer_radii_jax, L,
        D_i, D_m, D_e, geometry._kappa_inner_jax, geometry._kappa_outer_jax, rho_weights,
        geometry._eps, geometry._nudge, step_max, geometry.min_gap)
    R_mat, R_inv = geometry._R, geometry._R_inv
    ident = bool(np.allclose(np.array(R_mat), np.eye(3)))

    def sub(r, step_key, u, comp_id):
        pool, k = _pool_and_axon(geometry, comp_id)
        noise = jax.random.normal(step_key, (3,), dtype=jnp.float32)
        unit = noise / jnp.linalg.norm(noise)
        step_l = jnp.where(pool == 1, step_i[k], jnp.where(pool == 2, step_m[k], step_e[k]))
        r_c = r if ident else R_mat @ r
        s_c = unit * step_l
        l_xy = jnp.linalg.norm(s_c[:2])
        d_hat = jnp.where(l_xy > 0, s_c[:2] / jnp.maximum(l_xy, jnp.float32(1e-30)),
                          jnp.zeros(2, jnp.float32))
        xy_new, pool_new, k_new, chan, dlog_rho, _ = wall(r_c[:2], d_hat, l_xy, pool, k, u)
        if L is not None:
            xy_new = xy_new - L * jnp.floor(xy_new / L + jnp.float32(0.5))      # stay in the cell
        r_c_new = jnp.stack([xy_new[0], xy_new[1], r_c[2] + s_c[2]])
        r_new = r_c_new if ident else R_inv @ r_c_new
        return r_new, _encode_compartment(geometry, pool_new, k_new), chan, dlog_rho

    sub.max_bounces = wall.max_bounces
    sub.n_cand = wall.n_cand
    return sub


def make_myelin_step_fn(geometry, dt: float, T1: float = None, sub_steps: int = None):
    """Fused forward step for :class:`MyelinatedCylinder`.

    Carry ``(r, phi, log_w, compartment_id, key)`` with the compartment a pool id (0 extra,
    1 intra, 2 myelin); inputs ``(g_t, chi_t)``. Per-pool T2 accrues while transverse
    (``chi_t == 1``), T1 while stored. Sub-stepped by :func:`resolve_sub_steps`; the count is
    exposed as ``step_fn.n_sub``.
    """
    D_max = float(max(geometry.D_intra, geometry.D_myelin, geometry.D_extra))
    n_sub = resolve_sub_steps(geometry, D_max, dt, override=sub_steps)
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
            r_new, comp_new, _, _ = sub(r, k_step, u, comp)
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


def make_packed_myelin_traj_step_fn(geometry, dt: float,
                                    kappa_MT: float = 0.0, dwell_time: float = 0.0,
                                    mt_side_intra: float = 1.0, mt_side_extra: float = 1.0):
    """Trajectory step for PackedMyelinatedCylinders: geometry + permeability, no relaxation.

    Carry ``(r, key, dlog_accum, comp_id)`` -- or ``(r, key, dlog_accum, comp_id, bound_rem,
    bound_acc)`` when ``kappa_MT > 0``; ``step_fn(carry, None) -> (carry, None)``. ``dlog_accum``
    accumulates the unit boundary local time (``-2 d_perp`` per reflection, i.e. ``rho/D = 1``) so
    a replay can apply any surface relaxivity. ``comp_id`` is the encoded compartment (0 extra,
    ``1..N_max`` lumen of axon k, ``N_max+1..2N_max`` its sheath).

    Magnetization transfer (``kappa_MT > 0``): free water that reflects off a myelin wall binds
    with probability ``min(1, (kappa_MT/D) * local_time)``, freezes for an exponential dwell of
    mean ``dwell_time`` and is released; ``bound_acc`` counts frozen sub-steps. An intra walker
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

    def step_fn(carry, _):
        if mt_on:
            r, key, dlog_accum, comp_id, bound_rem, bound_acc = carry
            key, k_step, k_perm, stick_key, dwell_key = jax.random.split(key, 5)
        else:
            r, key, dlog_accum, comp_id = carry
            key, k_step, k_perm = jax.random.split(key, 3)
        u = jax.random.uniform(k_perm, dtype=jnp.float32)
        r_new, comp_new, chan, _ = sub(r, k_step, u, comp_id)
        dlog_boundary = jnp.sum(chan)
        if not mt_on:
            return (r_new, key, dlog_accum + dlog_boundary, comp_new), None

        pool, k = _pool_and_axon(geometry, comp_id)
        is_intra, is_extra = pool == 1, pool == 0
        is_bound = bound_rem > jnp.float32(0.0)
        local_time = jnp.where(is_intra, -chan[0], jnp.where(is_extra, -chan[3], jnp.float32(0.0)))
        D_bind = jnp.where(is_intra, D_intra[k], jnp.where(is_extra, D_extra[k], jnp.float32(1.0)))
        kappa_bind = jnp.where(is_intra, kappa_intra_f,
                               jnp.where(is_extra, kappa_extra_f, jnp.float32(0.0)))
        p_stick = bind_probability(kappa_bind / jnp.maximum(D_bind, jnp.float32(1e-30)), local_time)
        newly = (~is_bound) & (jax.random.uniform(stick_key, dtype=jnp.float32) < p_stick)
        u_dwell = jax.random.uniform(dwell_key, dtype=jnp.float32)
        dwell_draw = -jnp.log(jnp.maximum(u_dwell, jnp.float32(1e-20))) * dwell_steps_mean
        r_out = jnp.where(is_bound, r, r_new)
        comp_out = jnp.where(is_bound, comp_id, comp_new)
        dlog_contrib = jnp.where(is_bound | newly, jnp.float32(0.0), dlog_boundary)
        bound_rem_out = jnp.where(is_bound, bound_rem - jnp.float32(1.0),
                                  jnp.where(newly, dwell_draw, jnp.float32(0.0)))
        bound_acc_out = bound_acc + jnp.where(is_bound, jnp.float32(1.0), jnp.float32(0.0))
        return (r_out, key, dlog_accum + dlog_contrib, comp_out, bound_rem_out, bound_acc_out), None

    step_fn.max_bounces = sub.max_bounces
    return step_fn


def make_packed_myelin_step_fn(geometry, dt: float, T1: float = None):
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
    n_sub = resolve_sub_steps(geometry, D_ref, dt, surface=has_rho)
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
            r_ic_new, cid_new, _, dlog_rho = sub(r_ic, k_step, u, cid)
            # continuous displacement: remove the periodic wrap jump in the (x, y) cell plane
            dr = r_ic_new - r_ic
            dxy = dr[:2] - L * jnp.round(dr[:2] / L)
            r_uw_new = r_uw + jnp.array([dxy[0], dxy[1], dr[2]], dtype=jnp.float32)
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
    step_fn.max_bounces = sub.max_bounces
    return step_fn
