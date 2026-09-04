"""Periodic packings of many objects, with minimum-image geometry.

The walker interacts only with the object it already borders, which is what lets the
boundary tests stay O(1) per step rather than O(N objects).
"""
import jax
import jax.numpy as jnp
import numpy as np

from ._boundary import (keep_side_radial, ray_sphere_t, specular,
                        transmit_probability, off_wall, step_off_wall)
from .base import Geometry, _rotation_to_z


class PackedCylinders(Geometry):
    """Extra-axonal diffusion in a periodic square domain packed with cylinders.

    Walkers are initialised in the interstitial space between cylinders and
    are reflected specularly when they would enter any cylinder.  The
    cross-section boundary is periodic (walkers wrap around the square box);
    diffusion along the shared cylinder axis is unrestricted.

    All N cylinders are parallel and share the same ``orientation`` axis.
    Use ``pack_cylinders()`` to generate collision-free centre positions.

    Parameters
    ----------
    radii : array-like, shape (N,)
        Cylinder radii in metres.
    centers : np.ndarray, shape (N, 2)
        Cylinder centre positions in the cross-section plane, metres.
        Must come from ``pack_cylinders()`` (or otherwise be non-overlapping).
    L : float
        Side-length of the periodic square domain in metres.
    orientation : array-like, shape (3,), optional
        Shared cylinder axis direction (normalised internally).  Default [0,0,1].

    Attributes
    ----------
    min_gap : float
        Minimum clear gap between any two cylinder surfaces (including periodic
        images), metres.  Use this to verify the single-reflection-per-step
        approximation::

            sigma = np.sqrt(6 * D * dt)   # typical step length
            assert sigma < 0.1 * geom.min_gap

        At D = 2e-9 m²/s and n_t = 1000 over a 50 ms experiment,
        dt ≈ 50 µs and σ ≈ 0.25 µm.  For min_gap = 1 µm, σ/δ_min ≈ 0.25,
        which is borderline.  Use n_t ≥ 2000 for VF > 0.4.

    Notes
    -----
    Reflection algorithm
    ~~~~~~~~~~~~~~~~~~~~
    One specular reflection per timestep is applied against the nearest
    cylinder (minimum ray-entry time across all N cylinders).  After the
    reflection the walker travels the remaining step in the reflected
    direction without checking for further collisions.  This is exact when
    σ ≪ min_gap (the walker cannot traverse the full gap in one step) and
    introduces negligible error when σ < 0.1 · min_gap.

    Periodic boundary conditions
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    A minimum-image convention is applied when computing ray-circle
    intersections, so cylinders near the box edge correctly interact with
    walkers near the opposite edge.  The final position is wrapped into
    [-L/2, L/2)² after every timestep.
    """

    supports_permeability = True   #: has a membrane a walker can cross
    carries_side = True            #: `permeate` accepts the walker's own side

    # `classify_position` returns 0=extra and 1..N = the cylinder the walker is in,
    # i.e. an OBJECT id, not a pool id. core.simulate_trajectories collapses it to a
    # two-pool label for `comp_traj` (0=extra, 1=intra) -- both because relaxation is
    # per-pool, not per-cylinder, and because an object id overflows the int8 channel
    # above 127 cylinders.
    classify_returns_object_id = True

    def __init__(self, radii, centers, L, orientation=(0., 0., 1.),
                 surface_relaxivity_t2=None, permeability=None):
        radii   = np.asarray(radii,   dtype=np.float64).ravel()
        centers = np.asarray(centers, dtype=np.float64)
        if centers.shape != (len(radii), 2):
            raise ValueError(
                f"centers shape {centers.shape} does not match "
                f"({len(radii)}, 2) for {len(radii)} cylinders.")
        if np.any(radii <= 0):
            raise ValueError("All radii must be positive.")
        self.surface_relaxivity_t2 = (
            float(surface_relaxivity_t2)
            if surface_relaxivity_t2 is not None else None
        )
        self.permeability = (
            float(permeability) if permeability is not None else None
        )

        self._L_float  = float(L)
        self._radii_np = radii.copy()

        orientation = np.asarray(orientation, dtype=np.float64)
        self.orientation = (orientation / np.linalg.norm(orientation)).astype(
            np.float32)
        _R_np = _rotation_to_z(self.orientation)
        self._R     = jnp.array(_R_np, dtype=jnp.float32)
        self._R_inv = jnp.array(_R_np.T, dtype=jnp.float32)
        self._is_identity_rotation = bool(np.allclose(_R_np, np.eye(3)))

        # JAX-side constants baked in at construction time
        self._L_jax       = jnp.float32(L)
        self._radii_jax   = jnp.array(radii,   dtype=jnp.float32)   # (N,)
        self._centers_jax = jnp.array(centers, dtype=jnp.float32)   # (N, 2)

        min_r = float(np.min(radii))
        self._eps_detect = jnp.float32(1e-7 * min_r)
        self._nudge      = jnp.float32(1e-4 * min_r)

        self.min_gap = self._compute_min_gap(centers, radii, float(L))

    @staticmethod
    def _compute_min_gap(centers, radii, L):
        """Minimum clear gap between any two cylinder surfaces (periodic)."""
        N       = len(radii)
        min_gap = float('inf')
        # Between distinct cylinder pairs
        for i in range(N):
            for j in range(i + 1, N):
                dq  = centers[i] - centers[j]
                dq -= L * np.round(dq / L)
                gap = np.linalg.norm(dq) - radii[i] - radii[j]
                min_gap = min(min_gap, gap)
        # Each cylinder vs its own periodic images (nearest image is at distance L)
        for i in range(N):
            min_gap = min(min_gap, L - 2.0 * radii[i])
        return float(min_gap)

    def init_positions(self, n_walkers, key):
        """Uniform placement in the periodic box, outside all cylinder cross-sections."""
        L       = self._L_float
        radii   = self._radii_np
        centers = np.array(self._centers_jax)  # (N, 2)
        rng = np.random.default_rng(
            int(jax.random.randint(key, (), 0, 2 ** 30)))

        accepted = []
        n_have   = 0
        while n_have < n_walkers:
            batch = max(n_walkers * 4, 1024)
            xy    = rng.uniform(-L / 2.0, L / 2.0, (batch, 2))
            outside = np.ones(batch, dtype=bool)
            for k in range(len(radii)):
                dxy     = xy - centers[k]
                dxy    -= L * np.round(dxy / L)   # minimum-image
                outside &= np.sum(dxy ** 2, axis=1) > radii[k] ** 2
            accepted.append(xy[outside])
            n_have = sum(len(a) for a in accepted)

        xy_out = np.concatenate(accepted, axis=0)[:n_walkers].astype(np.float32)
        # z = 0; walkers are free along the cylinder axis
        r_cyl = np.concatenate(
            [xy_out, np.zeros((n_walkers, 1), dtype=np.float32)], axis=1)
        R_inv = np.array(self._R_inv)
        r_lab = (R_inv @ r_cyl.T).T
        return jnp.array(r_lab, dtype=jnp.float32)

    def reflect(self, r, step):
        """Impermeable wall interaction -- the kappa = 0 case of :meth:`permeate`.

        NOT a separate algorithm. It used to be one, and the copies drifted: this method
        expelled 100% of intra-cylinder walkers while `permeate(kappa=0)` confined them, and
        the two were bit-identical on the extra side (#88). At kappa = 0 nothing may cross,
        so a walker reflects on whichever side of the wall it starts -- one rule, one
        implementation. XLA folds the constant and drops the dead transmit branch, so this
        costs exactly what the hand-written version did (0.11 ms / 40k walkers, measured).

        The key is unused: at kappa = 0 the transmit probability is identically zero, so the
        draw cannot change the outcome.
        """
        return self.permeate(r, step, jnp.float32(0.0), jnp.float32(0.0),
                             jax.random.PRNGKey(0))[0]

    def reflect_with_log_weight(self, r, step, rho_over_D):
        """Impermeable wall interaction that also accrues surface relaxation.

        The kappa = 0 case of :meth:`permeate` with rho > 0 -- see :meth:`reflect`.
        """
        return self.permeate(r, step, jnp.float32(0.0), rho_over_D,
                             jax.random.PRNGKey(0))[:2]

    def permeate(self, r, step, kappa_over_D, rho_over_D, perm_key, side=None):
        """Probabilistic membrane crossing (Powles 2004) + optional relaxivity.

        Bidirectional: walkers start in the extra-axonal space and may enter
        (and re-exit) cylinders when κ > 0.  At each timestep the nearest
        cylinder wall is tested; if a crossing occurs the walker transmits
        with probability p = min(1, 2κ·d_perp/D) or reflects otherwise.

        Side detection: walker is inside cylinder k if |r2 − c_k|² < R_k².
        For inside walkers the exit root is used (t = −dp + √disc);
        for outside walkers the entry root is used (t = −dp − √disc).
        Only one permeability event per timestep (single-event approximation).

        Parameters
        ----------
        r          : (3,) float32, current position (lab frame)
        step       : (3,) float32, proposed displacement (lab frame)
        kappa_over_D : float32, κ/D
        rho_over_D   : float32, ρ/D  (0.0 if no surface relaxivity)
        perm_key   : JAX PRNGKey

        Returns
        -------
        r_new  : (3,) float32
        dlog_w : float32
        """
        L          = self._L_jax
        centers_2d = self._centers_jax    # (N, 2)
        radii_arr  = self._radii_jax      # (N,)
        EPS        = self._eps_detect
        NUDGE      = self._nudge

        if self._is_identity_rotation:
            r_c    = r
            step_c = step
        else:
            r_c    = self._R @ r
            step_c = self._R @ step
        r2      = r_c[:2]
        step_xy = step_c[:2]
        step_z  = step_c[2]

        step_l_xy = jnp.linalg.norm(step_xy)
        d_hat_xy  = jnp.where(
            step_l_xy > jnp.float32(0.0),
            step_xy / step_l_xy,
            jnp.zeros(2, dtype=jnp.float32))

        # ── Minimum-image positions relative to each cylinder ─────────────────
        q_all  = r2[None, :] - centers_2d                          # (N, 2) raw
        q_all  = q_all - L * jnp.floor(q_all / L + jnp.float32(0.5))

        dist2_all = jnp.sum(q_all ** 2, axis=1)                    # (N,)
        inside_k  = dist2_all < radii_arr ** 2                     # (N,) bool

        # ── Vectorised ray-circle intersection ────────────────────────────────
        t_entry_all, t_exit_all, disc_all = ray_sphere_t(q_all, d_hat_xy, radii_arr)
        # Inside cylinders: use exit root; outside: use entry root
        t_all = jnp.where(inside_k, t_exit_all, t_entry_all)      # (N,)

        valid    = (
            (disc_all > jnp.float32(0.0))
            & (t_all  > EPS)
            & (t_all  < step_l_xy)
            & (step_l_xy > jnp.float32(0.0))
        )
        t_valid  = jnp.where(valid, t_all, jnp.float32(jnp.inf))

        i_min        = jnp.argmin(t_valid)
        t_min        = t_valid[i_min]
        any_hit      = jnp.isfinite(t_min)
        t_safe       = jnp.where(any_hit, t_min, jnp.float32(0.0))
        hit_is_inside = inside_k[i_min]            # walker was inside hit cylinder

        c_hit = centers_2d[i_min]
        R_hit = radii_arr[i_min]

        q_c       = r2 - c_hit
        q_c       = q_c - L * jnp.floor(q_c / L + jnp.float32(0.5))
        # Raw hit point in local (cylinder-relative) frame.  Float32 rounding
        # on r2 + t_safe*d_hat_xy can place the hit point slightly outside the
        # cylinder.  When the walker is inside and we nudge with -n_out, a hit
        # point that is already outside makes the nudge insufficient to push the
        # walker back inside — leaving it outside on the next step.
        # Fix: snap the local hit point to the boundary (R_hit * n_out) so the
        # nudge always starts from exactly |q_hit| = R_hit.
        q_hit_raw = q_c + t_safe * d_hat_xy
        q_hit_len = jnp.linalg.norm(q_hit_raw)
        n_out     = q_hit_raw / jnp.maximum(q_hit_len, jnp.float32(1e-30))
        q_hit     = R_hit * n_out               # snapped: |q_hit| = R_hit exactly

        remaining = step_l_xy - t_safe

        # cos(α) = √disc / R  (same formula as reflect_with_log_weight)
        disc_hit  = disc_all[i_min]
        cos_alpha = jnp.sqrt(jnp.maximum(disc_hit, jnp.float32(0.0))) / R_hit
        d_perp    = jnp.where(any_hit, remaining * cos_alpha, jnp.float32(0.0))

        # ── Permeability decision ─────────────────────────────────────────────
        p_transmit = transmit_probability(kappa_over_D, d_perp)
        u        = jax.random.uniform(perm_key, dtype=jnp.float32)
        transmit = any_hit & (u < p_transmit)

        # ── Reflected: specular, nudge to same side ───────────────────────────
        d_refl      = specular(d_hat_xy, n_out)
        d_refl_norm = jnp.linalg.norm(d_refl)
        d_refl      = jnp.where(
            d_refl_norm > jnp.float32(0.0),
            d_refl / jnp.maximum(d_refl_norm, jnp.float32(1e-30)),
            jnp.zeros(2, dtype=jnp.float32)
        )
        # LOCAL frame (relative to c_hit), so |q| = R_hit ± NUDGE exactly
        q_refl = step_off_wall(q_hit, n_out, hit_is_inside, d_refl, remaining, NUDGE)

        # ── Safety clamp in local frame: keep q_refl on the correct side ─────
        # Same fix as Cylinder.permeate(): tangential steps can push |q_refl|
        # past R_hit.  Clamp to R_hit-NUDGE (inside) or R_hit+NUDGE (outside).
        q_refl_norm      = jnp.linalg.norm(q_refl)
        q_refl_norm_safe = jnp.maximum(q_refl_norm, jnp.float32(1e-30))
        target_q         = jnp.where(hit_is_inside, R_hit - NUDGE, R_hit + NUDGE)
        wrong_side_q     = jnp.where(hit_is_inside, q_refl_norm >= R_hit,
                                                     q_refl_norm <= R_hit)
        q_refl = jnp.where(wrong_side_q,
                            q_refl * target_q / q_refl_norm_safe,
                            q_refl)

        # Convert local reflected position back to lab frame.
        # r2 = c_hit_eff + q_c  where c_hit_eff = r2 - q_c (effective centre).
        # r2_refl = c_hit_eff + q_refl = r2 + (q_refl - q_c)
        r2_refl = r2 + (q_refl - q_c)

        # ── Transmitted: straight through ─────────────────────────────────────
        r2_straight = r2 + step_xy

        # ── Combine (no periodic wrap — keep unfolded position) ───────────────
        r2_hit_result = jnp.where(transmit, r2_straight, r2_refl)
        xy_final      = jnp.where(any_hit, r2_hit_result, r2 + step_xy)

        # ── Relaxivity weight on reflection only ──────────────────────────────
        dlog_w = jnp.where(
            any_hit & ~transmit,
            -jnp.float32(2.0) * rho_over_D * d_perp,
            jnp.float32(0.0))

        # ── Reconstruct lab-frame position ───────────────────────────────────
        # Build absolute cylinder-frame position then rotate to lab frame.
        # For the identity-rotation case (_is_identity_rotation=True) we skip
        # the _R_inv matmul: the GPU batch-matmul for _R==I gives wrong r_c
        # values (XLA dot_general identity-matrix bug), so both _R @ r at
        # input and _R_inv @ r_c_new at output are bypassed by Python-level
        # branching resolved at trace time.
        # ── Sentinel: a compartment change is legal ONLY where a crossing was granted ──
        # Adapted from MC/DC's deportation check (dynamicsSimulation.finalPositionCheck): the
        # walker CARRIES its side, and any disagreement between that and the geometry is illegal
        # by definition rather than something to be inferred from a strict inequality.
        #
        # This is what makes the failure non-expressible. Deriving the compartment from the
        # position each step means a walker whose step ROUNDS onto |q| = R changes label without
        # moving: measured on this substrate, `t_exit` exceeded the step by ~0.006 nm out of 18,
        # so no collision fired (correctly -- it does not reach the wall), the straight step was
        # taken, and float32 put it exactly on the surface where `dist2 < R^2` reads "outside".
        # 0.675% of intra walkers left their axon that way over 30k steps at kappa = 0.
        #
        # MC/DC re-runs the offending walker (`w--`). A vmapped walk cannot re-run one lane, so
        # the equivalent here is to eject it back to its own side; `crossed` is returned so the
        # driver can advance the carried label only on legal events.
        crossed = any_hit & transmit
        if side is not None:
            # Only ONE cylinder can matter. A sub-step is orders of magnitude smaller than
            # the inter-cylinder gap, so the walker can only be on the wrong side of the
            # cylinder it already borders -- the nearest one at the START of the step, which
            # `dist2_all` above has already given us. Re-testing all N at the endpoint costs
            # +40% on the step for no extra coverage; this is O(1).
            k_near = jnp.argmin(dist2_all - radii_arr ** 2)
            q_n    = xy_final - centers_2d[k_near]
            q_n    = q_n - L * jnp.floor(q_n / L + jnp.float32(0.5))
            R_n    = radii_arr[k_near]
            want_in  = side < jnp.int8(0)          # side < 0 == intra, >= 0 == extra
            want_in  = jnp.where(crossed, ~want_in, want_in)   # a granted crossing flips it
            # A walker exactly ON the surface belongs to NEITHER compartment, and in float32
            # that tie is real: |q| - R evaluates to exactly 0 for these endpoints, and two
            # equivalent spellings of "is it inside" (this one and classify_position's) then
            # disagree depending on op fusion. So equality is illegal for both sides -- an
            # intra walker must be STRICTLY inside, an extra one STRICTLY outside. Testing
            # `geo_in != want_in` instead leaves the tie unresolved and let 0.4% through.
            xy_final, illegal = keep_side_radial(xy_final, q_n, R_n, want_in, NUDGE)
            # `illegal` is the diagnostic worth counting: at kappa = 0 no crossing is ever
            # granted, so a count of granted-crossings-at-kappa-0 is identically zero and
            # says nothing. What actually fires is this disagreement between the carried
            # label and the geometry -- i.e. the rounding-onto-the-surface event.
        else:
            # UNCONDITIONAL sentinel, for callers that do not carry a side. Without this
            # PackedCylinders was the only geometry whose protection was opt-in, and it
            # leaked 0.067% at kappa = 0 when `permeate` was called with the plain 5-argument
            # contract -- which is what `physics`, `bloch` and `pedagogy` all use. Every other
            # geometry derives its side from the START of the step (`inside`, `inside_k`) and
            # needs no caller cooperation; this makes PackedCylinders match. The carried
            # `side` above remains the stronger guarantee, because it also survives a walker
            # that legitimately crossed earlier in the same walk.
            k_near = jnp.argmin(dist2_all - radii_arr ** 2)
            q_n2   = xy_final - centers_2d[k_near]
            q_n2   = q_n2 - L * jnp.floor(q_n2 / L + jnp.float32(0.5))
            xy_final, _ = keep_side_radial(xy_final, q_n2, radii_arr[k_near],
                                           inside_k[k_near], NUDGE, active=~transmit)

        z_final   = r_c[2] + step_z
        r_c_new   = jnp.stack([xy_final[0], xy_final[1], z_final])
        # Backwards compatible: without a carried `side` this is the old two-value contract,
        # so existing callers (physics, bloch, pedagogy) are untouched. A caller that opts into
        # carried-compartment bookkeeping passes `side` and gets the crossing flag back.
        r_out = r_c_new if self._is_identity_rotation else self._R_inv @ r_c_new
        if side is None:
            return r_out, dlog_w
        return r_out, dlog_w, crossed, illegal

    def classify_position(self, r: jnp.ndarray) -> jnp.ndarray:
        """Compartment ID: 0=extra-axonal, 1..N = intra_k (inside cylinder k).

        Walkers in the periodic extra-axonal space return 0.  Walkers inside
        cylinder k (k = 1..N in 1-indexed) return k.

        Parameters
        ----------
        r : (3,) float32, position in lab frame.
        """
        L          = self._L_jax
        centers_2d = self._centers_jax    # (N, 2)
        radii_arr  = self._radii_jax      # (N,)

        r_c = r if self._is_identity_rotation else self._R @ r
        r2  = r_c[:2]

        # Minimum-image distances to each cylinder centre
        q_all = r2[None, :] - centers_2d                              # (N, 2)
        q_all = q_all - L * jnp.floor(q_all / L + jnp.float32(0.5))
        dist2 = jnp.sum(q_all ** 2, axis=1)                          # (N,)

        # For each cylinder, 1-indexed ID if inside, else 0
        inside_k = dist2 < radii_arr ** 2                             # (N,) bool
        ids      = jnp.arange(1, radii_arr.shape[0] + 1, dtype=jnp.int32)  # 1..N
        # Pick the first (smallest index) cylinder the walker is inside; 0 if none.
        # Use a reduction: if inside_k[i] then ids[i] else 0; max gives intra ID
        # (only one cylinder should contain the walker at a time).
        intra_id = jnp.max(jnp.where(inside_k, ids, jnp.int32(0)))
        return intra_id

    def volume(self, L: float = 1.0) -> float:
        """Total intra-cylindrical volume: Σ π·Rk²·L (m³).

        Parameters
        ----------
        L : float, optional
            Cylinder length in metres. Default 1.0 (per-unit-length).
        """
        return float(np.pi * np.sum(self._radii_np ** 2) * float(L))

    def surface_area(self, L: float = 1.0) -> float:
        """Total lateral surface area of all cylinders: Σ 2π·Rk·L (m²).

        Parameters
        ----------
        L : float, optional
            Cylinder length in metres. Default 1.0 (per-unit-length).
        """
        return float(2.0 * np.pi * np.sum(self._radii_np) * float(L))

    def volume_fraction(self) -> float:
        """Intra-cylindrical volume fraction: Σ π·Rk² / L² (dimensionless).

        Returns the fraction of the periodic square cross-section area
        (side L) occupied by the cylinder cross-sections.
        """
        return float(np.pi * np.sum(self._radii_np ** 2) / self._L_float ** 2)


class PackedSpheres(Geometry):
    """Extra-axonal diffusion in a periodic cubic domain packed with spheres.

    Walkers are initialised in the interstitial space between spheres and are
    reflected (or permeated) when they would enter any sphere.  Periodic
    boundary conditions are applied via minimum-image convention; positions are
    kept unfolded for correct phase accumulation.

    Parameters
    ----------
    radii : array-like, shape (N,)
        Sphere radii in metres.
    centers : np.ndarray, shape (N, 3)
        Sphere centre positions in metres.
        Must come from ``pack_spheres()`` (or otherwise be non-overlapping).
    L : float
        Side-length of the periodic cubic domain in metres.
    surface_relaxivity_t2 : float, optional
        Surface relaxivity ρ₂ in m/s.  Brownstein-Tarr weight on each
        reflection.  Default None (no surface relaxation).
    permeability : float, optional
        Membrane permeability κ in m/s.  Bidirectional exchange via Powles
        (2004): p = min(1, 2κ·d_perp/D).  Default None (fully reflecting).

    Attributes
    ----------
    min_gap : float
        Minimum clear gap between any two sphere surfaces (including periodic
        images), metres.
    """

    supports_permeability = True   #: has a membrane a walker can cross

    # `classify_position` returns 0=extra and 1..N = the sphere the walker is in,
    # i.e. an OBJECT id, not a pool id. core.simulate_trajectories collapses it to a
    # two-pool label for `comp_traj` (0=extra, 1=intra) -- both because relaxation is
    # per-pool, not per-sphere, and because an object id overflows the int8 channel
    # above 127 spheres.
    classify_returns_object_id = True

    def __init__(self, radii, centers, L,
                 surface_relaxivity_t2=None, permeability=None):
        radii   = np.asarray(radii,   dtype=np.float64).ravel()
        centers = np.asarray(centers, dtype=np.float64)
        if centers.shape != (len(radii), 3):
            raise ValueError(
                f"centers shape {centers.shape} does not match "
                f"({len(radii)}, 3) for {len(radii)} spheres.")
        if np.any(radii <= 0):
            raise ValueError("All radii must be positive.")
        self.surface_relaxivity_t2 = (
            float(surface_relaxivity_t2)
            if surface_relaxivity_t2 is not None else None
        )
        self.permeability = (
            float(permeability) if permeability is not None else None
        )

        self._L_float   = float(L)
        self._radii_np  = radii.copy()
        self._centers_np = centers.copy()

        self._L_jax       = jnp.float32(L)
        self._radii_jax   = jnp.array(radii,   dtype=jnp.float32)   # (N,)
        self._centers_jax = jnp.array(centers, dtype=jnp.float32)   # (N, 3)

        min_r = float(np.min(radii))
        self._eps_detect = jnp.float32(1e-7 * min_r)
        self._nudge      = jnp.float32(1e-4 * min_r)

        self.min_gap = self._compute_min_gap(centers, radii, float(L))

    @staticmethod
    def _compute_min_gap(centers, radii, L):
        """Minimum clear gap between any two sphere surfaces (periodic, 3D)."""
        N       = len(radii)
        min_gap = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                dq  = centers[i] - centers[j]
                dq -= L * np.round(dq / L)
                gap = np.linalg.norm(dq) - radii[i] - radii[j]
                min_gap = min(min_gap, gap)
        for i in range(N):
            min_gap = min(min_gap, L - 2.0 * radii[i])
        return float(min_gap)

    def init_positions(self, n_walkers, key):
        """Uniform placement in the periodic cube, outside all spheres."""
        L       = self._L_float
        radii   = self._radii_np
        centers = self._centers_np
        rng = np.random.default_rng(
            int(jax.random.randint(key, (), 0, 2 ** 30)))

        accepted = []
        n_have   = 0
        while n_have < n_walkers:
            batch = max(n_walkers * 4, 1024)
            pts   = rng.uniform(-L / 2.0, L / 2.0, (batch, 3))
            outside = np.ones(batch, dtype=bool)
            for k in range(len(radii)):
                dq      = pts - centers[k]
                dq     -= L * np.round(dq / L)   # minimum-image
                outside &= np.sum(dq ** 2, axis=1) > radii[k] ** 2
            accepted.append(pts[outside])
            n_have = sum(len(a) for a in accepted)

        pts_out = np.concatenate(accepted, axis=0)[:n_walkers].astype(np.float32)
        return jnp.array(pts_out, dtype=jnp.float32)

    def reflect(self, r, step):
        """Impermeable wall interaction -- the kappa = 0 case of :meth:`permeate`.

        NOT a separate algorithm. It used to be one, and the copies drifted: this method
        expelled 100% of intra-sphere walkers while `permeate(kappa=0)` confined them, and
        the two were bit-identical on the extra side (#88). At kappa = 0 nothing may cross,
        so a walker reflects on whichever side of the wall it starts -- one rule, one
        implementation. XLA folds the constant and drops the dead transmit branch, so this
        costs exactly what the hand-written version did (0.11 ms / 40k walkers, measured).

        The key is unused: at kappa = 0 the transmit probability is identically zero, so the
        draw cannot change the outcome.
        """
        return self.permeate(r, step, jnp.float32(0.0), jnp.float32(0.0),
                             jax.random.PRNGKey(0))[0]

    def reflect_with_log_weight(self, r, step, rho_over_D):
        """Impermeable wall interaction that also accrues surface relaxation.

        The kappa = 0 case of :meth:`permeate` with rho > 0 -- see :meth:`reflect`.
        """
        return self.permeate(r, step, jnp.float32(0.0), rho_over_D,
                             jax.random.PRNGKey(0))[:2]

    def permeate(self, r, step, kappa_over_D, rho_over_D, perm_key):
        """Probabilistic membrane crossing (Powles 2004) + optional relaxivity.

        Bidirectional: walkers may start inside or outside any sphere.  At each
        timestep the nearest sphere wall is tested.  Inside walkers use the exit
        root; outside walkers use the entry root.  Transmit with
        p = min(1, 2κ·d_perp/D); reflect otherwise.

        Parameters
        ----------
        r            : (3,) float32, current position
        step         : (3,) float32, proposed displacement
        kappa_over_D : float32, κ/D
        rho_over_D   : float32, ρ/D  (0.0 if no surface relaxivity)
        perm_key     : JAX PRNGKey

        Returns
        -------
        r_new  : (3,) float32
        dlog_w : float32
        """
        L         = self._L_jax
        centers   = self._centers_jax    # (N, 3)
        radii_arr = self._radii_jax      # (N,)
        EPS       = self._eps_detect
        NUDGE     = self._nudge

        step_l = jnp.linalg.norm(step)
        d_hat  = jnp.where(
            step_l > jnp.float32(0.0),
            step / step_l,
            jnp.zeros(3, dtype=jnp.float32))

        # ── Minimum-image displacements and side detection ─────────────────────
        q_all     = r[None, :] - centers                             # (N, 3)
        q_all     = q_all - L * jnp.floor(q_all / L + jnp.float32(0.5))
        dist2_all = jnp.sum(q_all ** 2, axis=1)                     # (N,)
        inside_k  = dist2_all < radii_arr ** 2                      # (N,) bool

        # ── Vectorised ray-sphere intersection ─────────────────────────────────
        t_entry_all, t_exit_all, disc_all = ray_sphere_t(q_all, d_hat, radii_arr)
        t_all = jnp.where(inside_k, t_exit_all, t_entry_all)        # (N,)

        valid    = (
            (disc_all > jnp.float32(0.0))
            & (t_all  > EPS)
            & (t_all  < step_l)
            & (step_l > jnp.float32(0.0))
        )
        t_valid  = jnp.where(valid, t_all, jnp.float32(jnp.inf))

        i_min        = jnp.argmin(t_valid)
        t_min        = t_valid[i_min]
        any_hit      = jnp.isfinite(t_min)
        t_safe       = jnp.where(any_hit, t_min, jnp.float32(0.0))
        hit_is_inside = inside_k[i_min]

        c_hit = centers[i_min]
        R_hit = radii_arr[i_min]

        q_c       = r - c_hit
        q_c       = q_c - L * jnp.floor(q_c / L + jnp.float32(0.5))
        q_hit_raw = q_c + t_safe * d_hat
        q_hit_len = jnp.linalg.norm(q_hit_raw)
        n_out     = q_hit_raw / jnp.maximum(q_hit_len, jnp.float32(1e-30))
        q_hit     = R_hit * n_out              # snapped to exact boundary

        remaining = step_l - t_safe

        # cos(α) = √disc / R
        disc_hit  = disc_all[i_min]
        cos_alpha = jnp.sqrt(jnp.maximum(disc_hit, jnp.float32(0.0))) / R_hit
        d_perp    = jnp.where(any_hit, remaining * cos_alpha, jnp.float32(0.0))

        # ── Permeability decision ──────────────────────────────────────────────
        p_transmit = transmit_probability(kappa_over_D, d_perp)
        u        = jax.random.uniform(perm_key, dtype=jnp.float32)
        transmit = any_hit & (u < p_transmit)

        # ── Reflected: specular, nudge to same side ────────────────────────────
        d_refl      = specular(d_hat, n_out)
        d_refl_norm = jnp.linalg.norm(d_refl)
        d_refl      = jnp.where(
            d_refl_norm > jnp.float32(0.0),
            d_refl / jnp.maximum(d_refl_norm, jnp.float32(1e-30)),
            jnp.zeros(3, dtype=jnp.float32)
        )
        # LOCAL frame (relative to c_hit), so |q| = R_hit ± NUDGE exactly
        q_refl = step_off_wall(q_hit, n_out, hit_is_inside, d_refl, remaining, NUDGE)

        # Safety clamp in local frame (same as PackedCylinders.permeate)
        q_refl_norm      = jnp.linalg.norm(q_refl)
        q_refl_norm_safe = jnp.maximum(q_refl_norm, jnp.float32(1e-30))
        target_q         = jnp.where(hit_is_inside, R_hit - NUDGE, R_hit + NUDGE)
        wrong_side_q     = jnp.where(hit_is_inside, q_refl_norm >= R_hit,
                                                     q_refl_norm <= R_hit)
        q_refl = jnp.where(wrong_side_q,
                            q_refl * target_q / q_refl_norm_safe,
                            q_refl)

        r_refl    = r + (q_refl - q_c)

        # ── Transmitted: straight through ─────────────────────────────────────
        r_straight = r + step

        # ── Combine (unfolded position — no periodic wrap) ────────────────────
        r_hit_result = jnp.where(transmit, r_straight, r_refl)
        r_out        = jnp.where(any_hit, r_hit_result, r + step)

        # ── Compartment sentinel: no granted crossing => no change of side ────────
        # The 3-D twin of the PackedCylinders sentinel (dmrai-lab/dmipy-sim#86). A step
        # landing exactly on |q| = R fires no collision and the strict `dist2 < R**2` test
        # then reads the other compartment, so the walker changes compartment without
        # moving. Only ONE sphere can matter -- a sub-step is far smaller than the gap, so
        # the walker can only be on the wrong side of the sphere it already borders, which
        # `dist2_all` has already given us. O(1), no extra all-N pass.
        _k      = jnp.argmin(dist2_all - radii_arr ** 2)   # nearest surface at step start
        _q      = r_out - centers[_k]
        _q      = _q - L * jnp.floor(_q / L + jnp.float32(0.5))   # minimum image, as above
        _qn     = jnp.linalg.norm(_q)
        _R_k    = radii_arr[_k]
        # `inside_k` is the strict test at the START of the step. A walker sitting exactly
        # on the surface belongs to neither side, so equality is wrong for both -- see the
        # PackedCylinders sentinel for why the tie is real in float32.
        _side0 = inside_k[_k]                               # side BEFORE the step
        r_out, _ = keep_side_radial(r_out, _q, _R_k, _side0, NUDGE, active=~transmit)

        # ── Relaxivity weight on reflection only ───────────────────────────────
        dlog_w = jnp.where(
            any_hit & ~transmit,
            -jnp.float32(2.0) * rho_over_D * d_perp,
            jnp.float32(0.0))

        return r_out, dlog_w

    def classify_position(self, r: jnp.ndarray) -> jnp.ndarray:
        """Compartment ID: 0=extra-axonal, 1..N = inside sphere k (1-indexed)."""
        L         = self._L_jax
        centers   = self._centers_jax
        radii_arr = self._radii_jax

        q_all = r[None, :] - centers
        q_all = q_all - L * jnp.floor(q_all / L + jnp.float32(0.5))
        dist2 = jnp.sum(q_all ** 2, axis=1)

        inside_k = dist2 < radii_arr ** 2
        ids      = jnp.arange(1, radii_arr.shape[0] + 1, dtype=jnp.int32)
        intra_id = jnp.max(jnp.where(inside_k, ids, jnp.int32(0)))
        return intra_id

    def volume(self) -> float:
        """Total intra-sphere volume: Σ (4/3)·π·Rk³ (m³)."""
        return float((4.0 / 3.0) * np.pi * np.sum(self._radii_np ** 3))

    def surface_area(self) -> float:
        """Total sphere surface area: Σ 4·π·Rk² (m²)."""
        return float(4.0 * np.pi * np.sum(self._radii_np ** 2))

    def volume_fraction(self) -> float:
        """Intra-sphere volume fraction: Σ (4/3)·π·Rk³ / L³."""
        return float(
            (4.0 / 3.0) * np.pi * np.sum(self._radii_np ** 3)
            / self._L_float ** 3
        )
