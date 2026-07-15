"""Pure-JAX scan body for Monte Carlo phase accumulation.

make_step_fn returns a closure suitable for jax.lax.scan that captures
the geometry, diffusivity, and dt. The returned function is JIT-compiled
on first call via jax.jit applied in core.py.
"""

import jax
import jax.numpy as jnp
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


def make_step_fn(geometry, diffusivity: float, dt: float, T2: float = None,
                 T1: float = None):
    """Return (step_fn, has_weight) for one simulation timestep.

    Each step consumes ``(g_t, chi_t)``: the gradient sample and a binary
    transverse-coherence flag.  When ``chi_t == 1`` the magnetisation is
    transverse (T2 decay and surface relaxivity act); when ``chi_t == 0`` it is
    stored longitudinally (only T1 acts — no T2 loss, no surface-relaxivity
    loss).  A plain spin echo passes ``chi_t ≡ 1``.

    Parameters
    ----------
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
        n_sub        = permeable_sub_steps(geometry, float(_D0), dt)
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
        n_sub        = surface_sub_steps(geometry, float(_D0), dt)
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

        def step_fn(carry, inputs):
            g_t, chi_t = inputs
            r, phi, log_weight, key = carry

            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, (3,), dtype=jnp.float32)
            unit_noise = noise / jnp.linalg.norm(noise)
            step = unit_noise * _step_l(r, dt_f32)

            r_new = reflect(r, step)

            dlog_w = jnp.float32(0.0)
            if has_t2:
                dlog_w = dlog_w - _t2_decrement(r_new, dt_f32) * chi_t
            if has_t1:
                dlog_w = dlog_w - _t1_decrement(r_new, dt_f32) * (jnp.float32(1.0) - chi_t)
            dphi    = gamma_dt * jnp.dot(g_t, r_new)
            phi_new = phi + dphi

            return (r_new, phi_new, log_weight + dlog_w, key), None

    else:
        # No weight at all. (A Mesh with only per-compartment D lands here — the
        # step length is still resolved per compartment via _step_l.)
        reflect = geometry.reflect

        def step_fn(carry, inputs):
            g_t, _chi_t = inputs
            r, phi, key = carry

            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, (3,), dtype=jnp.float32)
            unit_noise = noise / jnp.linalg.norm(noise)
            step = unit_noise * _step_l(r, dt_f32)

            r_new = reflect(r, step)

            dphi    = gamma_dt * jnp.dot(g_t, r_new)
            phi_new = phi + dphi

            return (r_new, phi_new, key), None

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
        p_inner = jnp.minimum(jnp.float32(1.0),
                              jnp.float32(2.0) * kappa_over_D_inner * d_perp_inner)

        # --- Permeability at outer boundary ---
        d_perp_outer = jnp.abs(r_new_xy_norm - R_out)
        kappa_over_D_outer = kappa_outer / jnp.maximum(D_leaving, jnp.float32(1e-30))
        p_outer = jnp.minimum(jnp.float32(1.0),
                              jnp.float32(2.0) * kappa_over_D_outer * d_perp_outer)

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


def make_packed_myelin_traj_step_fn(geometry, dt: float):
    """Stripped PackedMyelinatedCylinders step for trajectory saving.

    Runs geometry + permeability only (no T2/T1, rho=1 at all walls).
    Carry: (r, key, dlog_accum, comp_id)
    Returns: (carry, None)
    dlog_accum accumulates -2*d_perp per boundary hit (rho/D=1).
    """
    dt_f32 = jnp.float32(dt)
    N_max  = geometry.N_max

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

        p_inner = jnp.minimum(jnp.float32(1.0),
                              jnp.float32(2.0) * kappa_over_D_inner * d_perp_inner)
        p_outer = jnp.minimum(jnp.float32(1.0),
                              jnp.float32(2.0) * kappa_over_D_outer * d_perp_outer)

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
    Mirrors the phase / T2 / rho conventions of ``apply_waveform_with_relaxation`` exactly, so the
    fused forward and the (private) trajectory-replay path are the same signal by construction.
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
