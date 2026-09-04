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


def _geometry_radius(geometry):
    """Smallest geometric radius (m) of a geometry, or None if not applicable.

    Used to size the walk's sub-steps against the geometry's smallest
    length scale (see make_step_fn).
    """
    R = getattr(geometry, 'radius', None)
    if R is None:
        R = getattr(geometry, 'sphere_radius', None)
    if R is None:
        # A slab (Box1D) confines over its WIDTH and exposes `length`, not `radius`. Without this clause the
        # search falls through to None, the caller takes it as "no scale to resolve" and runs ONE sub-step at
        # step_l = sqrt(6 D dt_save) -- 6 um for a 2 um slab at dt_save=3 ms -- which garbles the recorded
        # boundary local time and inflates a fitted surface T2 by 42% (2 um slab, rho=1e-6: 1.42 s against a
        # Brownstein-Tarr 1.0 s). This clause existed in the core.py auto-tune before it was refactored here
        # in 6d585fc and was dropped in the move; `tests/test_compression.py` caught it.
        R = getattr(geometry, 'length', None)
    if R is None:
        radii = getattr(geometry, '_radii_np', None)
        if radii is not None and len(radii) > 0:
            R = float(np.min(radii))
    if R is None:
        inner = getattr(geometry, '_inner_radii_np', None)
        if inner is not None and len(inner) > 0 and np.any(inner > 0):
            R = float(np.min(inner[inner > 0]))
    return float(R) if R is not None else None


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
    the LARGER extra-axonal pore, ~ 1 / (S_ext/V). For a packed substrate we take
    that pore; otherwise fall back to the geometric radius.
    """
    outer = getattr(geometry, '_outer_radii_np', None)
    cell = getattr(geometry, '_cell_size', None)
    if outer is not None and cell is not None:
        outer = np.asarray(outer, float); outer = outer[outer > 0]
        area_ext = float(cell) ** 2 - float(np.sum(np.pi * outer ** 2))
        perim = float(np.sum(2.0 * np.pi * outer))
        if perim > 0 and area_ext > 0:
            return area_ext / perim                 # 1 / (S_ext/V) = extra-axonal pore
    return _geometry_radius(geometry)


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
    cs = getattr(geometry, "cell_size", None)
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
    if getattr(geometry, 'cell_size', None):
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
    R = _geometry_radius(geometry)
    n_coll = 1
    if getattr(geometry, 'cell_size', None) and not has_perm:
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
        if getattr(geometry, 'radius_is_mesh_feature', False) or R is None:
            return n_coll
    if R is None:
        # No scale found means "nothing to resolve", which is right for free diffusion and WRONG for anything
        # with walls -- and it fails silently, at one sub-step, with the boundary channel garbled. That is how
        # a dropped `length` clause went unnoticed for a whole release (see `_geometry_radius`). A confined
        # geometry advertises finite volume and surface area, so say so rather than guessing.
        try:
            confined = float(geometry.surface_area) > 0 and 0 < float(geometry.volume) < float('inf')
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
    attribute (overrides ``frac``); set it to ``0`` (or None) to DISABLE sub-stepping
    (single step, fast). Disabling is appropriate for a qualitative long-echo-time
    forward (e.g. a full CPMG train), where resolving the pore over the whole train is
    prohibitively expensive and the surface rate is validated separately.
    """
    g_frac = getattr(geometry, 'surface_substep_frac', None)
    if g_frac is not None:
        frac = g_frac
    if not frac or frac <= 0:
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
    cell = getattr(geometry, 'cell_size', None)
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


def make_step_fn(geometry, diffusivity: float, dt: float, T2: float = None,
                 T1: float = None, sub_steps: int = None):
    """Return (step_fn, has_weight) for one simulation timestep.

    Each step consumes ``(g_t, chi_t)``: the gradient sample and a binary
    transverse-coherence flag.  When ``chi_t == 1`` the magnetisation is
    transverse (T2 decay and surface relaxivity act); when ``chi_t == 0`` it is
    stored longitudinally (only T1 acts — no T2 loss, no surface-relaxivity
    loss).  A plain spin echo passes ``chi_t ≡ 1``.

    Parameters
    ----------
    sub_steps : int, optional
        Override the per-branch sub-step auto-tune with this exact count. Each branch otherwise picks its own
        rule -- ``permeable_sub_steps`` (R/25, because the crossing probability is step-size sensitive),
        ``surface_sub_steps`` (pore/8) or ``walk_sub_steps`` (R/6) -- so two runs of the SAME geometry that
        differ only in whether permeability is set are also run at different time resolutions, and their
        difference is not purely physics. Pass this to compare like with like, or to run a convergence sweep.
    geometry : Geometry instance
        Provides reflect(r, step).  If geometry.surface_relaxivity_t2 is set,
        also provides reflect_with_log_weight(r, step, rho_over_D).
        If geometry.permeability is set, also provides
        permeate(r, step, kappa_over_D, rho_over_D, perm_key).
    diffusivity : float
        Diffusion coefficient in m²/s.
    dt : float
        Time step in seconds.
    T2 : float, optional
        Transverse relaxation time in seconds. When set, accumulates
        ``-chi_t * dt / T2`` into log_weight each step.
    T1 : float, optional
        Longitudinal relaxation time in seconds. When set, accumulates
        ``-(1 - chi_t) * dt / T1`` into log_weight each step (only the stored,
        longitudinal intervals relax by T1).

    Returns
    -------
    step_fn : callable
        Without weight (no surface relaxation, no permeability, no T2, no T1):
            carry = (r, phi, key);  step_fn(carry, (g_t, chi_t)) -> (carry, None)
        With weight (surface relaxation, permeability, T2, or T1 set):
            carry = (r, phi, log_weight, key);
            step_fn(carry, (g_t, chi_t)) -> (carry, None)
    has_weight : bool
        True when geometry has surface_relaxivity_t2, permeability, T2, or T1 set.
    """
    gamma_dt = jnp.float32(GAMMA * dt)
    dt_f32   = jnp.float32(dt)

    # Optional per-compartment bulk properties (a Mesh may carry per-compartment D
    # and/or T2). They are None for ordinary geometries, in which case the resolvers
    # below collapse to the single-diffusivity / single-T2 scalars (identical path).
    # Optional per-compartment bulk properties (a Mesh may carry per-compartment D,
    # T2 and/or T1). None for ordinary geometries -> the resolvers collapse to the
    # single-diffusivity / single-T2 / single-T1 scalars (identical path).
    _D_arr     = getattr(geometry, '_D_comp_jax', None)        # (2,) or None
    _invT2_arr = getattr(geometry, '_inv_T2_comp_jax', None)   # (2,) or None
    _invT1_arr = getattr(geometry, '_inv_T1_comp_jax', None)   # (2,) or None
    _classify  = (geometry.classify_position
                  if any(a is not None for a in (_D_arr, _invT2_arr, _invT1_arr)) else None)
    _D0 = diffusivity if diffusivity is not None else getattr(geometry, '_D_comp_max', None)

    has_surf = getattr(geometry, 'surface_relaxivity_t2', None) is not None
    has_perm = getattr(geometry, 'permeability',          None) is not None
    has_t2   = (T2 is not None) or (_invT2_arr is not None)   # per-compartment T2 also needs log_w
    has_t1   = (T1 is not None) or (_invT1_arr is not None)   # per-compartment T1 also needs log_w
    has_weight = has_surf or has_perm or has_t2 or has_t1

    _inv_T2 = jnp.float32(1.0 / T2) if T2 is not None else jnp.float32(0.0)
    _inv_T1 = jnp.float32(1.0 / T1) if T1 is not None else jnp.float32(0.0)

    def _step_l(r, dt_local):
        """Step length at r over dt_local — per-compartment D if present, else single."""
        if _D_arr is not None:
            return jnp.sqrt(6.0 * _D_arr[_classify(r)] * dt_local)
        return jnp.sqrt(6.0 * _D0 * dt_local)

    def _t2_decrement(r, dt_local):
        """T2 log-weight decrement for a step ending at r (per-compartment if present).
        The caller gates this by chi_t (only accrues while transverse)."""
        if _invT2_arr is not None:
            return dt_local * _invT2_arr[_classify(r)]
        return dt_local * _inv_T2

    def _t1_decrement(r, dt_local):
        """T1 log-weight decrement for a step ending at r (per-compartment if present).
        The caller gates this by (1 - chi_t) (only accrues during longitudinal storage)."""
        if _invT1_arr is not None:
            return dt_local * _invT1_arr[_classify(r)]
        return dt_local * _inv_T1

    if has_perm:
        # D is single when permeable (unequal-D across a permeable wall is rejected
        # at Mesh construction), so κ/D uses the single diffusivity _D0.
        kappa_over_D = jnp.float32(geometry.permeability / float(_D0))
        rho_over_D   = (jnp.float32(geometry.surface_relaxivity_t2 / float(_D0))
                        if has_surf else jnp.float32(0.0))
        permeate = geometry.permeate

        # Membrane crossing is step-size sensitive (over-permeates at coarse
        # steps), so sub-step the permeable walk to step_l ≈ R/25 even when the
        # waveform dt is large.  Phase + relaxation accumulate per fine sub-step
        # (more accurate than one big step); G is held fixed across the group.
        n_sub        = sub_steps if sub_steps else permeable_sub_steps(geometry, float(_D0), dt)
        _warn_if_step_outruns_the_lookup(geometry, float(_D0), dt, n_sub, 'permeable walk')
        dt_sub       = dt / n_sub
        gamma_dt_sub = jnp.float32(GAMMA * dt_sub)
        dt_sub_f32   = jnp.float32(dt_sub)

        def step_fn(carry, inputs):
            g_t, chi_t = inputs

            def _sub(c, _):
                r, phi, log_weight, key = c
                key, subkey_step, subkey_perm = jax.random.split(key, 3)
                noise = jax.random.normal(subkey_step, (3,), dtype=jnp.float32)
                unit_noise = noise / jnp.linalg.norm(noise)
                step = unit_noise * _step_l(r, dt_sub_f32)

                r_new, dlog_w = permeate(r, step, kappa_over_D,
                                         rho_over_D, subkey_perm)

                # Surface relaxivity accrues only while transverse (chi_t == 1).
                dlog_w = dlog_w * chi_t
                if has_t2:
                    dlog_w = dlog_w - _t2_decrement(r_new, dt_sub_f32) * chi_t
                if has_t1:
                    dlog_w = dlog_w - _t1_decrement(r_new, dt_sub_f32) * (jnp.float32(1.0) - chi_t)

                phi_new = phi + gamma_dt_sub * jnp.dot(g_t, r_new)
                return (r_new, phi_new, log_weight + dlog_w, key), None

            carry_out, _ = jax.lax.scan(_sub, carry, None, length=n_sub)
            return carry_out, None

    elif has_surf:
        rho_nom = jnp.float32(geometry.surface_relaxivity_t2)
        reflect_with_log_weight = geometry.reflect_with_log_weight

        def _rho_over_D(r):
            Dc = _D_arr[_classify(r)] if _D_arr is not None else _D0
            return rho_nom / Dc

        # Surface relaxivity accrues via the boundary local time (accumulated reflection
        # overshoot). A single coarse step under-counts grazing wall contact, so sub-step
        # to step_l ~ pore/8 (the extra-axonal pore, coarser than permeability's R/25 since
        # the confined intra lumen is already exact). n_sub -> 1 once dt already resolves it;
        # phase / T2 / local-time accumulate per fine sub-step.
        n_sub        = sub_steps if sub_steps else surface_sub_steps(geometry, float(_D0), dt)
        _warn_if_step_outruns_the_lookup(geometry, float(_D0), dt, n_sub, 'surface walk')
        dt_sub       = dt / n_sub
        gamma_dt_sub = jnp.float32(GAMMA * dt_sub)
        dt_sub_f32   = jnp.float32(dt_sub)

        def step_fn(carry, inputs):
            g_t, chi_t = inputs

            def _sub(c, _):
                r, phi, log_weight, key = c
                key, subkey = jax.random.split(key)
                noise = jax.random.normal(subkey, (3,), dtype=jnp.float32)
                unit_noise = noise / jnp.linalg.norm(noise)
                step = unit_noise * _step_l(r, dt_sub_f32)

                r_new, dlog_w = reflect_with_log_weight(r, step, _rho_over_D(r))
                # Surface relaxivity accrues only while transverse (chi_t == 1).
                dlog_w = dlog_w * chi_t
                if has_t2:
                    dlog_w = dlog_w - _t2_decrement(r_new, dt_sub_f32) * chi_t
                if has_t1:
                    dlog_w = dlog_w - _t1_decrement(r_new, dt_sub_f32) * (jnp.float32(1.0) - chi_t)
                phi_new = phi + gamma_dt_sub * jnp.dot(g_t, r_new)
                return (r_new, phi_new, log_weight + dlog_w, key), None

            carry_out, _ = jax.lax.scan(_sub, carry, None, length=n_sub)
            return carry_out, None

    elif has_t2 or has_t1:
        # No surface relaxation, no permeability — but T2/T1 (incl. per-compartment)
        # require the log_weight carry.
        reflect = geometry.reflect
        # Sub-step so a displacement cannot outrun the collision candidate lookup (see
        # collision_sub_steps). Without it a step spanning several grid cells crosses triangles that were
        # never candidates and the walker leaves an impermeable mesh silently.
        # `walk_sub_steps`, NOT `collision_sub_steps`. The collision criterion is keyed to `cell_size` and
        # so returns 1 for every ANALYTIC geometry, meaning a fused impermeable Sphere/Cylinder/PackedSpheres
        # /Box1D walk did not sub-step at all -- while the permeable branch used R/25 and the replay backend
        # used R/6 for the same substrate. Measured on PackedSpheres R=5 um: fused 0.02857 vs replay 0.01975
        # at b=2000, a 0.0088 gap against 0.0004 for the permeable cases. `walk_sub_steps` delegates to the
        # collision rule for a mesh and applies R/6 for an analytic pore, which is what both other paths use.
        n_col = sub_steps if sub_steps else walk_sub_steps(geometry, float(_D0), dt)
        _warn_if_step_outruns_the_lookup(geometry, float(_D0), dt, n_col, 'mesh walk')
        dt_col = jnp.float32(dt / n_col)
        gamma_dt_col = jnp.float32(GAMMA * dt / n_col)

        def step_fn(carry, inputs):
            g_t, chi_t = inputs

            def _sub(c, _):
                r, phi, log_weight, key = c
                key, subkey = jax.random.split(key)
                noise = jax.random.normal(subkey, (3,), dtype=jnp.float32)
                unit_noise = noise / jnp.linalg.norm(noise)
                step = unit_noise * _step_l(r, dt_col)
                r_new = reflect(r, step)
                dlog = jnp.float32(0.0)
                if has_t2:
                    dlog = dlog - _t2_decrement(r_new, dt_col) * chi_t
                if has_t1:
                    dlog = dlog - _t1_decrement(r_new, dt_col) * (jnp.float32(1.0) - chi_t)
                return (r_new, phi + gamma_dt_col * jnp.dot(g_t, r_new),
                        log_weight + dlog, key), None

            carry_out, _ = jax.lax.scan(_sub, carry, None, length=n_col)
            return carry_out, None

    else:
        # No weight at all. (A Mesh with only per-compartment D lands here — the
        # step length is still resolved per compartment via _step_l.)
        reflect = geometry.reflect
        # Same collision-lookup constraint as the branch above: a step longer than a grid cell can cross a
        # triangle that was never a candidate, and the walker leaves an impermeable mesh with nothing raised.
        # `walk_sub_steps`, NOT `collision_sub_steps`. The collision criterion is keyed to `cell_size` and
        # so returns 1 for every ANALYTIC geometry, meaning a fused impermeable Sphere/Cylinder/PackedSpheres
        # /Box1D walk did not sub-step at all -- while the permeable branch used R/25 and the replay backend
        # used R/6 for the same substrate. Measured on PackedSpheres R=5 um: fused 0.02857 vs replay 0.01975
        # at b=2000, a 0.0088 gap against 0.0004 for the permeable cases. `walk_sub_steps` delegates to the
        # collision rule for a mesh and applies R/6 for an analytic pore, which is what both other paths use.
        n_col = sub_steps if sub_steps else walk_sub_steps(geometry, float(_D0), dt)
        _warn_if_step_outruns_the_lookup(geometry, float(_D0), dt, n_col, 'mesh walk')
        dt_col = jnp.float32(dt / n_col)
        gamma_dt_col = jnp.float32(GAMMA * dt / n_col)

        def step_fn(carry, inputs):
            g_t, _chi_t = inputs

            def _sub(c, _):
                r, phi, key = c
                key, subkey = jax.random.split(key)
                noise = jax.random.normal(subkey, (3,), dtype=jnp.float32)
                unit_noise = noise / jnp.linalg.norm(noise)
                step = unit_noise * _step_l(r, dt_col)
                r_new = reflect(r, step)
                return (r_new, phi + gamma_dt_col * jnp.dot(g_t, r_new), key), None

            carry_out, _ = jax.lax.scan(_sub, carry, None, length=n_col)
            return carry_out, None

    return step_fn, has_weight


def make_myelin_step_fn(geometry, dt: float, T1: float = None):
    """Return step_fn for MyelinatedCylinder geometry.

    Carry state: (r, phi, log_w, compartment_id, key)
    All compartment branching uses jnp.where (JAX-compatible).

    Each step consumes ``(g_t, chi_t)``: when ``chi_t == 1`` the magnetisation is
    transverse (per-compartment T2 acts); when ``chi_t == 0`` it is stored
    longitudinally (only T1 acts).

    Handles:
      - Anisotropic diffusion in myelin (radial vs tangential)
      - Dual-boundary permeability (inner + outer)
      - Per-compartment T2 relaxation folded into log_w (transverse intervals)
      - Longitudinal T1 relaxation folded into log_w (stored intervals)

    Parameters
    ----------
    geometry : MyelinatedCylinder
    dt : float
        Time step in seconds.
    T1 : float, optional
        Longitudinal relaxation time in seconds. When set, accumulates
        ``-(1 - chi_t) * dt / T1`` into log_w on the stored intervals.

    Returns
    -------
    step_fn : callable
        (carry, (g_t, chi_t)) -> (carry, None)
        carry = (r, phi, log_w, compartment_id, key)
    """
    gamma_dt = jnp.float32(GAMMA * dt)
    dt_f32 = jnp.float32(dt)
    has_t1 = T1 is not None
    if has_t1:
        inv_T1 = jnp.float32(1.0 / T1)

    # Pre-compute step lengths per compartment
    D_intra = jnp.float32(geometry.D_intra)
    D_myelin = jnp.float32(geometry.D_myelin)
    D_extra = jnp.float32(geometry.D_extra)

    step_l_intra = jnp.sqrt(jnp.float32(6.0) * D_intra * dt_f32)
    step_l_extra = jnp.sqrt(jnp.float32(6.0) * D_extra * dt_f32)
    # Myelin diffuses isotropically (single D_myelin; 0 -> stuck pool, canonical default).
    step_l_myelin = jnp.sqrt(jnp.float32(6.0) * D_myelin * dt_f32)

    R_in = jnp.float32(geometry.inner_radius)
    R_out = jnp.float32(geometry.outer_radius)
    EPS = jnp.float32(1e-7 * geometry.inner_radius)
    NUDGE = jnp.float32(1e-4 * geometry.inner_radius)

    R_mat = geometry._R
    R_inv = geometry._R_inv
    # GPU batch-matmul bug: vmap(R_mat @ r) with R_mat == I gives wrong
    # results on GPU (XLA dot_general identity-matrix bug).  Resolve at
    # closure-creation time so the buggy path is never compiled.
    _is_identity_R = bool(np.allclose(np.array(R_mat), np.eye(3)))

    # Permeability
    has_perm_inner = geometry.kappa_inner is not None
    has_perm_outer = geometry.kappa_outer is not None

    # D values per compartment for permeability formula: D of compartment being LEFT
    D_arr = jnp.array([D_intra, D_myelin, D_extra], dtype=jnp.float32)

    if has_perm_inner:
        kappa_inner = jnp.float32(geometry.kappa_inner)
    else:
        kappa_inner = jnp.float32(0.0)

    if has_perm_outer:
        kappa_outer = jnp.float32(geometry.kappa_outer)
    else:
        kappa_outer = jnp.float32(0.0)

    # T2 per compartment (magnetisation fully transverse throughout)
    has_t2 = (geometry.T2_intra is not None or
              geometry.T2_myelin is not None or
              geometry.T2_extra is not None)
    if has_t2:
        t2_intra  = jnp.float32(geometry.T2_intra  if geometry.T2_intra  is not None else 1e6)
        t2_myelin = jnp.float32(geometry.T2_myelin if geometry.T2_myelin is not None else 1e6)
        t2_extra  = jnp.float32(geometry.T2_extra  if geometry.T2_extra  is not None else 1e6)
        T2_arr = jnp.array([t2_intra, t2_myelin, t2_extra], dtype=jnp.float32)

    def step_fn(carry, inputs):
        g_t, chi_t = inputs
        r, phi, log_w, compartment_id, key = carry

        key, subkey_step, subkey_perm = jax.random.split(key, 3)
        noise = jax.random.normal(subkey_step, (3,), dtype=jnp.float32)

        # Transform to cylinder frame (skip matmul for identity — GPU bug)
        r_c = r if _is_identity_R else R_mat @ r

        # --- Compartment-dependent step generation ---

        # Intra-axonal: isotropic
        unit_noise_iso = noise / jnp.linalg.norm(noise)
        step_intra_c = unit_noise_iso * step_l_intra

        # Extra-axonal: isotropic
        step_extra_c = unit_noise_iso * step_l_extra

        # Myelin: isotropic step (D_myelin; 0 -> no displacement, a stuck pool).
        step_myelin_c = unit_noise_iso * step_l_myelin

        # Select step based on compartment
        # compartment_id: 0=intra, 1=myelin, 2=extra
        step_c = jnp.where(compartment_id == 0, step_intra_c,
                    jnp.where(compartment_id == 1, step_myelin_c, step_extra_c))

        # --- Proposed new position in cylinder frame ---
        r_new_c = r_c + step_c

        # --- Dual-boundary reflection and permeability ---
        r_new_xy = r_new_c[:2]
        r_new_xy_norm = jnp.linalg.norm(r_new_xy)

        new_compartment_id = compartment_id
        dlog_w = jnp.float32(0.0)

        # D of compartment being LEFT (for permeability formula)
        D_leaving = D_arr[compartment_id]

        # --- Inner boundary check ---
        # Walker in intra (0) crossing outward past R_in -> could enter myelin
        # Walker in myelin (1) crossing inward past R_in -> could enter intra
        crosses_inner_outward = (compartment_id == 0) & (r_new_xy_norm >= R_in)
        crosses_inner_inward = (compartment_id == 1) & (r_new_xy_norm < R_in)
        crosses_inner = crosses_inner_outward | crosses_inner_inward

        # --- Outer boundary check ---
        # Walker in myelin (1) crossing outward past R_out -> could enter extra
        # Walker in extra (2) crossing inward past R_out -> could enter myelin
        crosses_outer_outward = (compartment_id == 1) & (r_new_xy_norm >= R_out)
        crosses_outer_inward = (compartment_id == 2) & (r_new_xy_norm < R_out)
        crosses_outer = crosses_outer_outward | crosses_outer_inward

        # --- Permeability at inner boundary ---
        # d_perp approximation: distance past the boundary
        d_perp_inner = jnp.abs(r_new_xy_norm - R_in)
        kappa_over_D_inner = kappa_inner / jnp.maximum(D_leaving, jnp.float32(1e-30))
        p_inner = transmit_probability(kappa_over_D_inner, d_perp_inner)

        # --- Permeability at outer boundary ---
        d_perp_outer = jnp.abs(r_new_xy_norm - R_out)
        kappa_over_D_outer = kappa_outer / jnp.maximum(D_leaving, jnp.float32(1e-30))
        p_outer = transmit_probability(kappa_over_D_outer, d_perp_outer)

        # Split perm_key for inner and outer draws
        perm_key1, perm_key2 = jax.random.split(subkey_perm)
        u_inner = jax.random.uniform(perm_key1, dtype=jnp.float32)
        u_outer = jax.random.uniform(perm_key2, dtype=jnp.float32)

        transmit_inner = crosses_inner & (u_inner < p_inner)
        transmit_outer = crosses_outer & (u_outer < p_outer)

        # --- Handle inner boundary crossing ---
        # If transmit: walker passes through -> update compartment
        # If reflect: push walker back to its side of R_in
        safe_new_xy_norm = jnp.maximum(r_new_xy_norm, jnp.float32(1e-20))
        r_new_xy_hat = r_new_xy / safe_new_xy_norm

        # Inner boundary: SPECULAR reflection (mirror across R_in to 2*R_in - d),
        # matching make_packed_myelin_traj_step_fn and Cylinder.reflect.  A clamp to
        # R_in +- NUDGE lets walkers hug the wall and under-hinders transport; the
        # mirror works for both crossing directions (d>R_in -> back inside;
        # d<R_in -> back outside) without a direction branch.
        reflect_inner_r = jnp.float32(2.0) * R_in - r_new_xy_norm
        r_reflected_inner_xy = r_new_xy_hat * reflect_inner_r

        # Inner transmit: new compartment
        new_comp_inner_transmit = jnp.where(crosses_inner_outward,
                                             jnp.int32(1),   # intra -> myelin
                                             jnp.int32(0))   # myelin -> intra

        # Apply inner boundary decision
        inner_reflect = crosses_inner & ~transmit_inner
        r_new_xy = jnp.where(inner_reflect, r_reflected_inner_xy, r_new_xy)
        new_compartment_id = jnp.where(transmit_inner, new_comp_inner_transmit,
                                        new_compartment_id)

        # --- Handle outer boundary crossing ---
        # Recalculate r_new_xy_norm after potential inner reflection
        r_new_xy_norm2 = jnp.linalg.norm(r_new_xy)
        safe_new_xy_norm2 = jnp.maximum(r_new_xy_norm2, jnp.float32(1e-20))
        r_new_xy_hat2 = r_new_xy / safe_new_xy_norm2

        # Outer boundary: SPECULAR reflection (mirror across R_out to 2*R_out - d),
        # matching the trajectory path; a clamp to R_out +- NUDGE under-hinders the
        # (dominant) extra-axonal pool.
        reflect_outer_r = jnp.float32(2.0) * R_out - r_new_xy_norm2
        r_reflected_outer_xy = r_new_xy_hat2 * reflect_outer_r

        # Outer transmit: new compartment
        new_comp_outer_transmit = jnp.where(crosses_outer_outward,
                                             jnp.int32(2),   # myelin -> extra
                                             jnp.int32(1))   # extra -> myelin

        outer_reflect = crosses_outer & ~transmit_outer
        r_new_xy = jnp.where(outer_reflect, r_reflected_outer_xy, r_new_xy)
        new_compartment_id = jnp.where(transmit_outer, new_comp_outer_transmit,
                                        new_compartment_id)

        # --- Reconstruct 3D position ---
        r_new_c = jnp.array([r_new_xy[0], r_new_xy[1], r_new_c[2]], dtype=jnp.float32)

        # Safety clamp: ensure walker is in correct compartment region
        final_r_xy_norm = jnp.linalg.norm(r_new_c[:2])
        safe_final = jnp.maximum(final_r_xy_norm, jnp.float32(1e-20))
        final_xy_hat = r_new_c[:2] / safe_final

        # Compartment 0: must be inside R_in
        r_new_c = r_new_c.at[:2].set(
            jnp.where((new_compartment_id == 0) & (final_r_xy_norm >= R_in),
                      final_xy_hat * (R_in - NUDGE), r_new_c[:2]))

        # Compartment 1: must be between R_in and R_out
        final_r_xy_norm2 = jnp.linalg.norm(r_new_c[:2])
        safe_final2 = jnp.maximum(final_r_xy_norm2, jnp.float32(1e-20))
        final_xy_hat2 = r_new_c[:2] / safe_final2
        r_new_c = r_new_c.at[:2].set(
            jnp.where((new_compartment_id == 1) & (final_r_xy_norm2 < R_in),
                      final_xy_hat2 * (R_in + NUDGE), r_new_c[:2]))
        final_r_xy_norm3 = jnp.linalg.norm(r_new_c[:2])
        safe_final3 = jnp.maximum(final_r_xy_norm3, jnp.float32(1e-20))
        final_xy_hat3 = r_new_c[:2] / safe_final3
        r_new_c = r_new_c.at[:2].set(
            jnp.where((new_compartment_id == 1) & (final_r_xy_norm3 >= R_out),
                      final_xy_hat3 * (R_out - NUDGE), r_new_c[:2]))

        # Compartment 2: must be outside R_out
        final_r_xy_norm4 = jnp.linalg.norm(r_new_c[:2])
        safe_final4 = jnp.maximum(final_r_xy_norm4, jnp.float32(1e-20))
        final_xy_hat4 = r_new_c[:2] / safe_final4
        r_new_c = r_new_c.at[:2].set(
            jnp.where((new_compartment_id == 2) & (final_r_xy_norm4 < R_out),
                      final_xy_hat4 * (R_out + NUDGE), r_new_c[:2]))

        # Transform back to lab frame (skip matmul for identity — GPU bug)
        r_new = r_new_c if _is_identity_R else R_inv @ r_new_c

        # --- Per-compartment transverse (T2) relaxation, gated by chi_t ---
        if has_t2:
            dlog_w = dlog_w - dt_f32 / T2_arr[new_compartment_id] * chi_t
        # --- Longitudinal (T1) relaxation on the stored intervals ---
        if has_t1:
            dlog_w = dlog_w - dt_f32 * inv_T1 * (jnp.float32(1.0) - chi_t)

        # --- Phase accumulation ---
        dphi = gamma_dt * jnp.dot(g_t, r_new)
        phi_new = phi + dphi

        return (r_new, phi_new, log_w + dlog_w, new_compartment_id, key), None

    return step_fn


def make_packed_myelin_traj_step_fn(geometry, dt: float,
                                    kappa_MT: float = 0.0, dwell_time: float = 0.0,
                                    mt_side_intra: float = 1.0, mt_side_extra: float = 1.0):
    """Stripped PackedMyelinatedCylinders step for trajectory saving.

    Runs geometry + permeability only (no T2/T1, rho=1 at all walls).
    Carry: (r, key, dlog_accum, comp_id, bound_rem, bound_acc)
    Returns: (carry, None)
    dlog_accum accumulates -2*d_perp per boundary hit (rho/D=1).

    Magnetization transfer (``kappa_MT`` > 0): free water (intra/extra) that hits a
    myelin wall STICKS with p = min(1, (kappa_MT/D)*(-dlog_boundary)) -- the same
    impact-angle boundary-local-time rule as everywhere else -- then FREEZES for an
    exponential dwell (mean ``dwell_time``) and is released.  ``bound_acc`` counts
    frozen sub-steps -> per-save bound occupancy.  Mutually exclusive with surface
    relaxivity (a stuck encounter is removed from the rho channel, dlog_accum).
    Myelin water (D=0) never contacts a wall, so it cannot bind.  **kappa_MT = 0
    reproduces the pre-MT walk bit-for-bit (RNG stream and positions unchanged).**
    """
    dt_f32 = jnp.float32(dt)
    N_max  = geometry.N_max
    mt_on = kappa_MT > 0.0
    kappa_intra_f = jnp.float32(kappa_MT * mt_side_intra)   # inner-wall reactivity (intra water)
    kappa_extra_f = jnp.float32(kappa_MT * mt_side_extra)   # outer-wall reactivity (extra water)
    dwell_steps_mean = jnp.float32(dwell_time / dt) if dwell_time > 0 else jnp.float32(0.0)

    # Pre-extract JAX arrays (same geometry setup as the generic packed step fns)
    L          = geometry._L_jax
    inner_r    = geometry._inner_radii_jax    # (N_max,)
    outer_r    = geometry._outer_radii_jax    # (N_max,)
    centers_2d = geometry._centers_jax        # (N_max, 2)
    D_intra    = geometry._D_intra_jax        # (N_max,)
    D_myelin   = geometry._D_myelin_jax       # (N_max,)
    D_extra    = geometry._D_extra_jax        # (N_max,)
    kappa_inner = geometry._kappa_inner_jax   # (N_max,)
    kappa_outer = geometry._kappa_outer_jax   # (N_max,)

    R_mat    = geometry._R
    R_inv    = geometry._R_inv
    _is_identity_R = bool(np.allclose(np.array(R_mat), np.eye(3)))
    NUDGE    = geometry._nudge

    # Step-size arrays: sqrt(6*D*dt)
    step_intra_arr  = jnp.sqrt(jnp.float32(6.0) * D_intra  * dt_f32)
    step_extra_arr  = jnp.sqrt(jnp.float32(6.0) * D_extra  * dt_f32)
    step_myelin_arr = jnp.sqrt(jnp.float32(6.0) * D_myelin * dt_f32)

    def step_fn(carry, _):
        # Carry contract is 4-element by default (unchanged for every non-MT caller) and
        # 6-element only when MT is on — so kappa_MT=0 stays byte-for-byte the pre-MT walk.
        if mt_on:
            r, key, dlog_accum, compartment_id, bound_rem, bound_acc = carry
            key, subkey_step, subkey_perm, stick_key, dwell_key = jax.random.split(key, 5)
        else:
            r, key, dlog_accum, compartment_id = carry
            key, subkey_step, subkey_perm = jax.random.split(key, 3)
        noise = jax.random.normal(subkey_step, (3,), dtype=jnp.float32)
        unit_noise = noise / jnp.linalg.norm(noise)

        # ── Compartment classification ────────────────────────────────────────
        is_extra  = compartment_id == jnp.int32(0)
        is_intra  = (compartment_id >= jnp.int32(1)) & (compartment_id <= jnp.int32(N_max))
        is_myelin = compartment_id > jnp.int32(N_max)

        k_intra  = compartment_id - jnp.int32(1)
        k_myelin = compartment_id - jnp.int32(N_max + 1)
        k_cyl    = jnp.where(is_intra, k_intra,
                   jnp.where(is_myelin, k_myelin, jnp.int32(0)))
        k_cyl    = jnp.maximum(k_cyl, jnp.int32(0))

        # ── Step length selection ─────────────────────────────────────────────
        sl_intra  = step_intra_arr[k_cyl]
        sl_myelin = step_myelin_arr[k_cyl]
        sl_extra  = step_extra_arr[k_cyl]
        step_l    = jnp.where(is_intra, sl_intra,
                    jnp.where(is_myelin, sl_myelin, sl_extra))

        # ── Transform to cylinder frame ───────────────────────────────────────
        r_c     = r if _is_identity_R else R_mat @ r
        step_c  = unit_noise * step_l
        r_new_c = r_c + step_c

        r_new_xy = r_new_c[:2]
        step_z   = step_c[2]

        # ── Cylinder-specific geometry ────────────────────────────────────────
        c_k   = centers_2d[k_cyl]
        R_in  = inner_r[k_cyl]
        R_out = outer_r[k_cyl]
        kap_i = kappa_inner[k_cyl]
        kap_o = kappa_outer[k_cyl]

        # ── Min-image position relative to cylinder centre ────────────────────
        q_new  = r_new_xy - c_k
        q_new  = q_new - L * jnp.floor(q_new / L + jnp.float32(0.5))
        r_new_xy_norm = jnp.linalg.norm(q_new)

        new_compartment_id = compartment_id
        dlog_boundary      = jnp.float32(0.0)

        # ── Boundary crossing detection ───────────────────────────────────────
        crosses_inner_out = is_intra  & (r_new_xy_norm >= R_in)
        crosses_inner_in  = is_myelin & (r_new_xy_norm  < R_in)
        crosses_outer_out = is_myelin & (r_new_xy_norm >= R_out)
        crosses_outer_in  = is_extra  & (r_new_xy_norm  < R_out)

        crosses_inner = crosses_inner_out | crosses_inner_in
        crosses_outer = crosses_outer_out | crosses_outer_in

        # ── Permeability (actual kappa; rho/D=1 for dlog accumulation) ───────
        D_leaving = jnp.where(is_intra,  D_intra[k_cyl],
                    jnp.where(is_myelin, D_myelin[k_cyl], D_extra[k_cyl]))

        d_perp_inner = jnp.abs(r_new_xy_norm - R_in)
        d_perp_outer = jnp.abs(r_new_xy_norm - R_out)

        kappa_over_D_inner = kap_i / jnp.maximum(D_leaving, jnp.float32(1e-30))
        kappa_over_D_outer = kap_o / jnp.maximum(D_leaving, jnp.float32(1e-30))

        p_inner = transmit_probability(kappa_over_D_inner, d_perp_inner)
        p_outer = transmit_probability(kappa_over_D_outer, d_perp_outer)

        perm_key1, perm_key2 = jax.random.split(subkey_perm)
        u_i = jax.random.uniform(perm_key1, dtype=jnp.float32)
        u_o = jax.random.uniform(perm_key2, dtype=jnp.float32)

        transmit_inner = crosses_inner & (u_i < p_inner)
        transmit_outer = crosses_outer & (u_o < p_outer)

        # ── Inner boundary handling ───────────────────────────────────────────
        safe_norm = jnp.maximum(r_new_xy_norm, jnp.float32(1e-20))
        r_hat     = q_new / safe_norm

        refl_r_inner = jnp.float32(2.0) * R_in - r_new_xy_norm
        q_reflected_inner = r_hat * refl_r_inner

        new_comp_inner = jnp.where(crosses_inner_out,
                                    jnp.int32(N_max + k_cyl + 1),   # intra -> myelin
                                    jnp.int32(k_cyl + 1))            # myelin -> intra

        inner_reflect = crosses_inner & ~transmit_inner
        q_new = jnp.where(inner_reflect, q_reflected_inner, q_new)
        new_compartment_id = jnp.where(transmit_inner, new_comp_inner, new_compartment_id)

        # dlog with rho/D = 1 (unit boundary log-weight)
        dlog_boundary = dlog_boundary + jnp.where(
            inner_reflect,
            -jnp.float32(2.0) * d_perp_inner,
            jnp.float32(0.0))

        # ── Outer boundary handling ───────────────────────────────────────────
        r_new_xy_norm2 = jnp.linalg.norm(q_new)
        safe_norm2     = jnp.maximum(r_new_xy_norm2, jnp.float32(1e-20))
        r_hat2         = q_new / safe_norm2

        refl_r_outer = jnp.float32(2.0) * R_out - r_new_xy_norm2
        q_reflected_outer = r_hat2 * refl_r_outer

        new_comp_outer = jnp.where(crosses_outer_out,
                                    jnp.int32(0),                    # myelin -> extra
                                    jnp.int32(N_max + k_cyl + 1))   # extra -> myelin

        outer_reflect = crosses_outer & ~transmit_outer
        q_new = jnp.where(outer_reflect, q_reflected_outer, q_new)
        new_compartment_id = jnp.where(transmit_outer, new_comp_outer, new_compartment_id)

        dlog_boundary = dlog_boundary + jnp.where(
            outer_reflect,
            -jnp.float32(2.0) * d_perp_outer,
            jnp.float32(0.0))

        # ── Reconstruct absolute position + periodic wrap ─────────────────────
        xy_abs = q_new + c_k
        xy_abs = xy_abs - L * jnp.floor(xy_abs / L + jnp.float32(0.5))

        # ── Extra-axonal safety clamp against ALL cylinders ───────────────────
        q_f  = xy_abs[None, :] - centers_2d
        q_f  = q_f - L * jnp.floor(q_f / L + jnp.float32(0.5))
        d2_f = jnp.sum(q_f ** 2, axis=1)
        pen  = jnp.where(outer_r > jnp.float32(0.0),
                         jnp.where(d2_f < outer_r ** 2,
                                   d2_f / (outer_r ** 2 + jnp.float32(1e-30)),
                                   jnp.float32(1.0)),
                         jnp.float32(1.0))
        k_cl       = jnp.argmin(pen)
        inside_any = pen[k_cl] < jnp.float32(1.0)

        c_cl   = centers_2d[k_cl]
        R_cl   = outer_r[k_cl]
        q_cl   = xy_abs - c_cl
        q_cl   = q_cl - L * jnp.floor(q_cl / L + jnp.float32(0.5))
        d_cl   = jnp.linalg.norm(q_cl)
        c_near = xy_abs - q_cl
        # Specular reflection off the nearest cylinder's outer wall (mirror to
        # 2*R_cl - d_cl); this is the extra-axonal reflection (the cyl-0 outer
        # logic above only handles cylinder 0).  Reflecting matches the
        # interior/outer reflection geometry, so the -2*d_perp surface-local-time
        # estimator below keeps the Brownstein-Tarr calibration.
        d_refl = jnp.float32(2.0) * R_cl - d_cl
        xy_reflected = c_near + q_cl * d_refl / jnp.maximum(d_cl, NUDGE)
        xy_abs = jnp.where(is_extra & inside_any, xy_reflected, xy_abs)

        # Exterior surface local time: record the outer-wall contact for the
        # reflected extra walker.
        dlog_boundary = dlog_boundary + jnp.where(
            is_extra & inside_any,
            -jnp.float32(2.0) * jnp.maximum(R_cl - d_cl, jnp.float32(0.0)),
            jnp.float32(0.0))

        # ── Safety clamps for intra and myelin walkers ────────────────────────
        q_eff = xy_abs - c_k
        q_eff = q_eff - L * jnp.floor(q_eff / L + jnp.float32(0.5))
        d_eff = jnp.linalg.norm(q_eff)
        safe_d_eff = jnp.maximum(d_eff, jnp.float32(1e-20))
        q_eff_hat  = q_eff / safe_d_eff
        xy_abs = jnp.where(is_intra & (new_compartment_id == compartment_id) & (d_eff >= R_in),
                           c_k + q_eff_hat * (R_in - NUDGE),
                           xy_abs)

        q_myl = xy_abs - c_k
        q_myl = q_myl - L * jnp.floor(q_myl / L + jnp.float32(0.5))
        d_myl = jnp.linalg.norm(q_myl)
        safe_d_myl = jnp.maximum(d_myl, jnp.float32(1e-20))
        q_myl_hat  = q_myl / safe_d_myl
        xy_abs = jnp.where(is_myelin & (new_compartment_id == compartment_id) & (d_myl < R_in),
                           c_k + q_myl_hat * (R_in + NUDGE), xy_abs)
        q_myl2 = xy_abs - c_k
        q_myl2 = q_myl2 - L * jnp.floor(q_myl2 / L + jnp.float32(0.5))
        d_myl2 = jnp.linalg.norm(q_myl2)
        safe_d_myl2 = jnp.maximum(d_myl2, jnp.float32(1e-20))
        q_myl2_hat  = q_myl2 / safe_d_myl2
        xy_abs = jnp.where(is_myelin & (new_compartment_id == compartment_id) & (d_myl2 >= R_out),
                           c_k + q_myl2_hat * (R_out - NUDGE), xy_abs)

        # ── Reconstruct 3D and rotate back ────────────────────────────────────
        z_final = r_c[2] + step_z
        r_c_new = jnp.stack([xy_abs[0], xy_abs[1], z_final])
        r_new   = r_c_new if _is_identity_R else R_inv @ r_c_new

        # ── Update comp_id from new position ─────────────────────────────────
        # Use the same cylinder k_cyl as the step for comp_id assignment
        r_c_new_xy   = (r_new if _is_identity_R else R_mat @ r_new)[:2]
        q_new_abs    = r_c_new_xy - c_k
        q_new_abs    = q_new_abs - L * jnp.floor(q_new_abs / L + jnp.float32(0.5))
        dist_sq      = jnp.dot(q_new_abs, q_new_abs)
        inner_r_sq_k = inner_r[k_cyl] ** 2
        outer_r_sq_k = outer_r[k_cyl] ** 2
        new_intra    = dist_sq < inner_r_sq_k
        new_myelin   = (~new_intra) & (dist_sq < outer_r_sq_k)
        # Extra-axonal walkers have NO owning cylinder: k_cyl is a dummy 0, so
        # reclassifying them against cylinder 0's annulus spuriously absorbs
        # near-wall extra walkers into "myelin of cylinder 0" (where D=0 freezes
        # them permanently -- they then carry the short myelin T2 and stop
        # diffusing).  An extra walker's compartment changes ONLY through the
        # explicit transmit_outer permeation above; with the canonical
        # impermeable myelin (kappa=0) it stays extra.  Guard the position-based
        # reclassification to intra/myelin walkers, whose k_cyl IS meaningful.
        comp_id_new  = jnp.where(is_extra, new_compartment_id,
                       jnp.where(new_intra,  k_cyl + 1,
                       jnp.where(new_myelin, geometry.N_max + k_cyl + 1,
                                             new_compartment_id)))

        if mt_on:
            # ── MT surface binding at the myelin walls ────────────────────────
            is_bound   = bound_rem > jnp.float32(0.0)
            local_time = -dlog_boundary            # >= 0: myelin-wall local time this step
            # free-water diffusivity of the walker's pool (myelin water can't bind)
            D_bind = jnp.where(is_intra, D_intra[k_cyl],
                     jnp.where(is_extra, D_extra[k_cyl], jnp.float32(1.0)))
            # SIDE-dependent reactivity: an intra walker only ever contacts the INNER
            # myelin wall, an extra walker only the OUTER wall, so the walker's pool
            # selects the side (myelin water -> 0, cannot bind).
            kappa_bind = jnp.where(is_intra, kappa_intra_f,
                         jnp.where(is_extra, kappa_extra_f, jnp.float32(0.0)))
            p_stick = bind_probability(
                kappa_bind / jnp.maximum(D_bind, jnp.float32(1e-30)), local_time)
            u_stick = jax.random.uniform(stick_key, dtype=jnp.float32)
            newly   = (~is_bound) & (u_stick < p_stick)
            u_dwell = jax.random.uniform(dwell_key, dtype=jnp.float32)
            dwell_draw = -jnp.log(jnp.maximum(u_dwell, jnp.float32(1e-20))) * dwell_steps_mean
            # frozen while bound: hold position + compartment; mutual exclusivity ->
            # a stuck (or bound) encounter contributes NO surface-relaxivity dlog.
            r_out    = jnp.where(is_bound, r, r_new)
            comp_out = jnp.where(is_bound, compartment_id, comp_id_new)
            dlog_contrib  = jnp.where(is_bound | newly, jnp.float32(0.0), dlog_boundary)
            bound_rem_out = jnp.where(is_bound, bound_rem - jnp.float32(1.0),
                                      jnp.where(newly, dwell_draw, jnp.float32(0.0)))
            bound_acc_out = bound_acc + jnp.where(is_bound, jnp.float32(1.0), jnp.float32(0.0))
            return (r_out, key, dlog_accum + dlog_contrib, comp_out,
                    bound_rem_out, bound_acc_out), None
        return (r_new, key, dlog_accum + dlog_boundary, comp_id_new), None

    return step_fn


def make_packed_myelin_step_fn(geometry, dt: float, T1: float = None):
    """Fused forward SIGNAL step for PackedMyelinatedCylinders (transverse, instant pulses).

    Wraps the validated per-compartment walk (:func:`make_packed_myelin_traj_step_fn`) and adds,
    in the SAME forward scan (no trajectory storage / replay):

      * gradient phase ``phi += GAMMA*dt * (G(t) . r)`` accumulated on the CONTINUOUS lab-frame
        position -- the packed cell is periodic, so the in-cell walk is unwrapped here on the fly
        via the per-step min-image displacement (identical to ``unwrap_periodic``);
      * per-compartment transverse relaxation ``log_w += -dt / T2[intra|myelin|extra]``;
      * surface relaxivity ``log_w += (rho/D) * dlog_unit`` (rho from the geometry walls; the
        walk returns the unit ``rho/D = 1`` boundary local-time term).

    Carry: ``(r_incell, r_unwrapped, phi, log_w, compartment_id, key)``; inputs: ``g_t`` (n_meas, 3).
    Phase, per-compartment T2 and surface-relaxivity conventions match the rest of the forward
    model, so the signal is consistent across the engine by construction.
    """
    L = jnp.float32(geometry._cell_size)
    N_max = geometry.N_max

    has_t2 = getattr(geometry, '_has_t2', False)
    if has_t2:
        t2_intra = jnp.float32(np.asarray(geometry._T2_intra_jax).ravel()[0])
        t2_myelin = jnp.float32(np.asarray(geometry._T2_myelin_jax).ravel()[0])
        t2_extra = jnp.float32(np.asarray(geometry._T2_extra_jax).ravel()[0])

    has_t1 = T1 is not None
    if has_t1:
        inv_T1 = jnp.float32(1.0 / T1)

    rho_i = float(np.max(np.asarray(geometry._rho_inner_jax))) \
        if hasattr(geometry, '_rho_inner_jax') else 0.0
    rho_o = float(np.max(np.asarray(geometry._rho_outer_jax))) \
        if hasattr(geometry, '_rho_outer_jax') else 0.0
    rho = max(rho_i, rho_o)
    # Surface relaxivity accumulates the boundary local time (wall-contact overshoot); a
    # single coarse step under-counts grazing contact for the fast extra-axonal walkers, so
    # sub-step to the extra-axonal pore scale (step_l ~ pore/8; n_sub -> 1 once the waveform
    # dt already resolves it). Phase / T2 / local-time accumulate per fine sub-step.
    if rho > 0.0:
        D_ref = float(max(np.max(np.asarray(geometry._D_intra_jax)),
                          np.max(np.asarray(geometry._D_extra_jax))))
        rho_over_D = jnp.float32(rho / D_ref)
        n_sub = surface_sub_steps(geometry, D_ref, dt)
    else:
        n_sub = 1
    dt_sub = dt / n_sub
    traj_step = make_packed_myelin_traj_step_fn(geometry, dt_sub)
    gamma_dt_sub = jnp.float32(GAMMA * dt_sub)
    dt_sub_f32 = jnp.float32(dt_sub)

    def step_fn(carry, inputs):
        g_t, chi_t = inputs

        def _sub(c, _):
            r_ic, r_uw, phi, log_w, cid, key = c
            (r_ic_new, key_new, dlog_step, cid_new), _ = traj_step(
                (r_ic, key, jnp.float32(0.0), cid), None)
            # continuous displacement: remove the periodic wrap jump in the (x, y) cell plane
            dr = r_ic_new - r_ic
            dxy = dr[:2] - L * jnp.round(dr[:2] / L)
            dr = jnp.array([dxy[0], dxy[1], dr[2]], dtype=jnp.float32)
            r_uw_new = r_uw + dr
            phi_new = phi + gamma_dt_sub * (g_t @ r_uw_new)          # (n_meas,)

            dlog = jnp.float32(0.0)
            if has_t2:
                is_extra = cid_new == jnp.int32(0)
                is_myelin = cid_new > jnp.int32(N_max)
                T2 = jnp.where(is_extra, t2_extra, jnp.where(is_myelin, t2_myelin, t2_intra))
                # Transverse decay only while chi_t == 1 (stored intervals: no T2 loss).
                dlog = dlog - dt_sub_f32 / T2 * chi_t
            if has_t1:
                # Longitudinal decay only on the stored intervals (chi_t == 0).
                dlog = dlog - dt_sub_f32 * inv_T1 * (jnp.float32(1.0) - chi_t)
            if rho > 0.0:
                # Surface relaxivity accrues only while transverse (chi_t == 1).
                dlog = dlog + rho_over_D * dlog_step * chi_t
            return (r_ic_new, r_uw_new, phi_new, log_w + dlog, cid_new, key_new), None

        carry_out, _ = jax.lax.scan(_sub, carry, None, length=n_sub)
        return carry_out, None

    return step_fn
