"""Geometry classes for boundary conditions.

Each geometry implements two methods:
  init_positions(n_walkers, key) -> (n_walkers, 3) float32
  reflect(r, step)               -> (3,) float32   [pure JAX, no Python branching]
"""

from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import numpy as np


class Geometry(ABC):
    @abstractmethod
    def init_positions(self, n_walkers: int, key: jax.Array) -> jnp.ndarray:
        """Return initial walker positions of shape (n_walkers, 3), float32."""

    @abstractmethod
    def reflect(self, r: jnp.ndarray, step: jnp.ndarray) -> jnp.ndarray:
        """Apply boundary conditions. Pure JAX — no Python control flow.

        Parameters
        ----------
        r : (3,) float32, current position
        step : (3,) float32, proposed displacement

        Returns
        -------
        (3,) float32, new position after boundary enforcement
        """


class FreeDiffusion(Geometry):
    """Unbounded free diffusion — walkers move without any reflection."""

    def init_positions(self, n_walkers, key):
        return jnp.zeros((n_walkers, 3), dtype=jnp.float32)

    def reflect(self, r, step):
        return r + step

    def classify_position(self, r: jnp.ndarray) -> jnp.ndarray:
        """Compartment ID: always 0 (single compartment)."""
        return jnp.int32(0)


class Box1D(Geometry):
    """1D reflecting slab with walls at x=0 and x=length.

    Diffusion is unrestricted along y and z. Used for step 4 (eigenfunction
    series validation) and surface-relaxivity physics tests.

    Parameters
    ----------
    length : float
        Slab thickness in metres.
    surface_relaxivity_t2 : float, optional
        Surface relaxivity ρ₂ in m/s.  When set, each wall collision reduces
        the walker magnetisation weight by exp(-2·ρ₂·d_perp/D) where d_perp
        is the perpendicular overshoot depth at the wall.
        T2_surface = d / (2·ρ)  (V/S = d/2 for a slab).  Default None.
    """

    def __init__(self, length: float, surface_relaxivity_t2=None):
        self.length = float(length)
        self.surface_relaxivity_t2 = (
            float(surface_relaxivity_t2) if surface_relaxivity_t2 is not None else None
        )

    def volume(self) -> float:
        """Volume per unit cross-section area = slab thickness (m)."""
        return self.length

    def surface_area(self) -> float:
        """Surface area per unit cross-section area = 2 walls (dimensionless)."""
        return 2.0

    def classify_position(self, r: jnp.ndarray) -> jnp.ndarray:
        """Compartment ID: 0=intra (0 <= x <= length), 1=extra (outside).

        For the Box1D geometry walkers are always inside the slab (reflecting
        walls), so this always returns 0.
        """
        return jnp.int32(0)

    def init_positions(self, n_walkers, key):
        x = jax.random.uniform(key, (n_walkers, 1), dtype=jnp.float32,
                                minval=0.0, maxval=self.length)
        yz = jnp.zeros((n_walkers, 2), dtype=jnp.float32)
        return jnp.concatenate([x, yz], axis=1)

    def reflect(self, r, step):
        L = jnp.float32(self.length)
        x_new = r[0] + step[0]
        # Fold back using modular reflection: map into [0, 2L] then mirror
        x_new = jnp.mod(x_new, 2 * L)
        x_new = jnp.where(x_new > L, 2 * L - x_new, x_new)
        y_new = r[1] + step[1]
        z_new = r[2] + step[2]
        return jnp.array([x_new, y_new, z_new], dtype=jnp.float32)

    def reflect_with_log_weight(self, r, step, rho_over_D):
        """Reflect off slab walls and accumulate surface-relaxation log-weight.

        Implements the Brownstein-Tarr weight for a flat wall perpendicular to x:
            dlog_w = -2·ρ/D·d_perp
        where d_perp is the perpendicular overshoot past the wall.

        For a flat wall with normal (1,0,0), cos(α) = |step_x|/|step| and
        d_perp = remaining_step · cos(α) = x_overshoot (the x-component of
        the step past the wall boundary).

        Single-crossing-per-step approximation.  Valid when σ ≪ length.

        Parameters
        ----------
        r          : (3,) float32
        step       : (3,) float32
        rho_over_D : float32, ρ/D
        """
        L = jnp.float32(self.length)
        x_new_raw = r[0] + step[0]

        # Perpendicular overshoot at each wall (zero if not crossed)
        d_perp = (jnp.maximum(x_new_raw - L,               jnp.float32(0.0))
                  + jnp.maximum(-x_new_raw,                 jnp.float32(0.0)))
        any_cross = d_perp > jnp.float32(0.0)

        # Reflect (identical to reflect())
        x_new = jnp.mod(x_new_raw, 2.0 * L)
        x_new = jnp.where(x_new > L, 2.0 * L - x_new, x_new)
        y_new = r[1] + step[1]
        z_new = r[2] + step[2]
        r_out = jnp.array([x_new, y_new, z_new], dtype=jnp.float32)

        dlog_w = jnp.where(any_cross,
                           -jnp.float32(2.0) * rho_over_D * d_perp,
                           jnp.float32(0.0))
        return r_out, dlog_w


class Sphere(Geometry):
    """Reflecting sphere of given radius centred at the origin.

    Parameters
    ----------
    radius : float
        Sphere radius in metres.
    surface_relaxivity_t2 : float, optional
        Surface relaxivity ρ₂ in m/s.  When set, boundary collisions reduce
        the walker magnetisation weight by exp(-2·ρ₂·d_perp/D).
        T2_surface = R / (3·ρ) for a sphere (S/V = 3/R).  Default None.
    permeability : float, optional
        Membrane permeability κ in m/s.  When set, each boundary crossing is
        probabilistic: the walker transmits with p = min(1, 2κ·d_perp/D) and
        reflects otherwise.  Enables bidirectional exchange — walkers may be
        inside or outside the sphere at any time.  Default None (fully
        reflecting wall).  Exchange time τ = R / (3κ).
    """

    def __init__(self, radius: float, surface_relaxivity_t2=None,
                 permeability=None):
        self.radius = float(radius)
        self.surface_relaxivity_t2 = (
            float(surface_relaxivity_t2) if surface_relaxivity_t2 is not None else None
        )
        self.permeability = (
            float(permeability) if permeability is not None else None
        )

    def volume(self) -> float:
        """Volume of the sphere: (4/3)·π·R³ (m³)."""
        return (4.0 / 3.0) * np.pi * self.radius ** 3

    def surface_area(self) -> float:
        """Surface area of the sphere: 4·π·R² (m²)."""
        return 4.0 * np.pi * self.radius ** 2

    def classify_position(self, r: jnp.ndarray) -> jnp.ndarray:
        """Compartment ID: 0=intra (|r| < R), 1=extra (|r| >= R)."""
        R = jnp.float32(self.radius)
        inside = jnp.dot(r, r) < R * R
        return jnp.where(inside, jnp.int32(0), jnp.int32(1))

    def init_positions(self, n_walkers, key):
        """Uniform sampling inside sphere via rejection (CPU numpy)."""
        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**30)))
        # Sphere fill factor ≈ π/6 ≈ 52%; batch of 4× is enough for one pass.
        accepted = []
        while sum(len(a) for a in accepted) < n_walkers:
            pts = rng.uniform(-self.radius, self.radius, (n_walkers * 4, 3))
            accepted.append(pts[np.linalg.norm(pts, axis=1) < self.radius])
        positions = np.concatenate(accepted, axis=0)[:n_walkers]
        return jnp.array(positions, dtype=jnp.float32)

    def reflect(self, r, step):
        """Specular reflection off sphere boundary with multiple reflections.

        Matches disimpy's while-loop convention: decomposes step into a unit
        direction + scalar remaining length, then iterates up to MAX_ITER times.
        At each iteration the distance d to the boundary is computed; if
        d < remaining the walker is moved to the boundary, the direction is
        specularly reflected, and remaining is decremented by d + epsilon
        (epsilon nudges the walker just inside the surface to avoid numerical
        re-intersection). Uses jax.lax.scan for JAX-compilable fixed iteration.
        """
        R   = jnp.float32(self.radius)
        # Two-level epsilon strategy (float32 safe):
        #   EPS_detect: threshold for d > EPS_detect to count as a crossing.
        #               Must be small so we catch near-boundary walkers.
        #   NUDGE:      inward displacement after reflection; must be >> float32
        #               machine_eps at radius scale so the nudge is representable.
        #               After nudging, walker is NUDGE inside boundary, so on the
        #               next timestep d ≈ NUDGE >> EPS_detect → crossing detected.
        # float32 machine_eps ≈ 1.2e-7 (relative), so at radius=1µm the
        # absolute machine eps is ~1.2e-13 m.  NUDGE = 1e-4*radius gives ~1000×
        # margin.  disimpy uses epsilon=1e-13 m (absolute, float64).
        EPS_detect = jnp.float32(1e-7 * self.radius)
        NUDGE      = jnp.float32(1e-4 * self.radius)

        step_l = jnp.linalg.norm(step)
        d_hat  = step / step_l

        def _one_reflection(carry, _):
            r0, d_hat, remaining = carry

            # Distance to sphere surface along d_hat from r0 (r0 inside sphere).
            # Derived from |r0 + t*d_hat|² = R²; positive root (forward exit):
            #   t = -dot(d_hat, r0) + sqrt(dot(d_hat,r0)² - (|r0|² - R²))
            dp    = jnp.dot(d_hat, r0)
            disc  = jnp.maximum(dp * dp - (jnp.dot(r0, r0) - R * R), 0.0)
            d     = -dp + jnp.sqrt(disc)

            intersects = (d > EPS_detect) & (d < remaining)

            # Boundary point and outward normal
            r_hit   = r0 + d * d_hat
            n_out   = r_hit / R                                           # outward unit normal
            # Specular reflection of unit direction
            d_refl  = d_hat - 2.0 * jnp.dot(d_hat, n_out) * n_out
            d_refl  = d_refl / jnp.linalg.norm(d_refl)                   # renormalise
            # Nudge inward so next timestep detects boundary reliably
            r_nudge = r_hit - NUDGE * n_out

            r0_new   = jnp.where(intersects, r_nudge, r0)
            dhat_new = jnp.where(intersects, d_refl,  d_hat)
            rem_new  = jnp.where(intersects, remaining - d - NUDGE, remaining)

            return (r0_new, dhat_new, rem_new), None

        (r_f, d_hat_f, rem_f), _ = jax.lax.scan(
            _one_reflection, (r, d_hat, step_l), None, length=10
        )
        r_out  = r_f + d_hat_f * jnp.maximum(rem_f, 0.0)
        # Safety clamp: if walker escaped (numerical edge case), project back inside
        r_norm = jnp.linalg.norm(r_out)
        r_out  = jnp.where(r_norm >= R, r_out * (R - NUDGE) / r_norm, r_out)
        return r_out

    def reflect_with_log_weight(self, r, step, rho_over_D):
        """Reflect and accumulate per-collision surface-relaxation log-weight.

        Uses the same quadratic intersection as reflect(), but also computes
        the perpendicular penetration depth d_perp = (remaining - d) * cos(α)
        at each collision, where cos(α) = sqrt(disc) / R.

        Returns
        -------
        r_out : jnp.ndarray, shape (3,)
            Final walker position.
        dlog_w : jnp.float32
            Log-weight decrement: -2 * rho_over_D * sum(d_perp).
            For sphere S/V = 3/R → T2_surface = R / (3·ρ).
        """
        R          = jnp.float32(self.radius)
        EPS_detect = jnp.float32(1e-7 * self.radius)
        NUDGE      = jnp.float32(1e-4 * self.radius)

        step_l = jnp.linalg.norm(step)
        d_hat  = step / step_l

        def _one_reflection(carry, _):
            r0, d_hat, remaining = carry

            dp    = jnp.dot(d_hat, r0)
            disc  = jnp.maximum(dp * dp - (jnp.dot(r0, r0) - R * R), 0.0)
            d     = -dp + jnp.sqrt(disc)

            intersects = (d > EPS_detect) & (d < remaining)

            r_hit   = r0 + d * d_hat
            n_out   = r_hit / R
            d_refl  = d_hat - 2.0 * jnp.dot(d_hat, n_out) * n_out
            d_refl  = d_refl / jnp.linalg.norm(d_refl)
            r_nudge = r_hit - NUDGE * n_out

            r0_new   = jnp.where(intersects, r_nudge, r0)
            dhat_new = jnp.where(intersects, d_refl,  d_hat)
            rem_new  = jnp.where(intersects, remaining - d - NUDGE, remaining)

            # Perpendicular penetration depth: cos(α) = sqrt(disc) / R
            cos_alpha = jnp.sqrt(disc) / R
            d_perp = jnp.where(intersects,
                               (remaining - d) * cos_alpha,
                               jnp.float32(0.0))

            return (r0_new, dhat_new, rem_new), d_perp

        (r_f, d_hat_f, rem_f), d_perps = jax.lax.scan(
            _one_reflection, (r, d_hat, step_l), None, length=10
        )
        r_out  = r_f + d_hat_f * jnp.maximum(rem_f, 0.0)
        r_norm = jnp.linalg.norm(r_out)
        r_out  = jnp.where(r_norm >= R, r_out * (R - NUDGE) / r_norm, r_out)

        dlog_w = -2.0 * jnp.float32(rho_over_D) * jnp.sum(d_perps)
        return r_out, dlog_w

    def permeate(self, r, step, kappa_over_D, rho_over_D, perm_key):
        """Probabilistic membrane crossing (Powles 2004) + optional relaxivity.

        At each boundary crossing the walker transmits with probability

            p = min(1,  2 · κ/D · d_perp)

        and reflects otherwise.  When rho_over_D > 0 a Brownstein-Tarr weight
        is applied on reflection only.

        The geometry is bidirectional: walkers may be inside (|r| < R) or
        outside (|r| > R); the appropriate intersection root is selected
        automatically.

        Single-event-per-step approximation.  Requires σ/R < 0.1.
        Exchange time τ = R / (3κ)  (V/κS = R/3).

        Regime/step-size note: the crossing is correct and step-robust in any
        fast/slow regime (validated by the closed-system permeability tests:
        high-κ → free diffusion, monotone-in-κ, etc.).  A SINGLE permeable object
        in an OPEN domain, however, is not a well-mixed reservoir: walkers that
        exit re-enter, so the apparent residence time is biased ABOVE τ=R/(3κ)
        (a step-independent effect that grows with time and is larger for
        lower-dimensional exteriors: 2D cylinder > 3D sphere).  For exchange /
        residence-time work use a periodic packed geometry (a proper reservoir),
        where τ=R/(3κ) holds; do not compare a single open-domain object's
        f_inside(t) to the well-mixed τ.

        Parameters
        ----------
        r          : (3,) float32, current position
        step       : (3,) float32, proposed displacement
        kappa_over_D : float32, κ/D baked in by make_step_fn
        rho_over_D   : float32, ρ/D  (0.0 if no surface relaxivity)
        perm_key   : JAX PRNGKey for the Bernoulli draw

        Returns
        -------
        r_new  : (3,) float32, new position
        dlog_w : float32, log-weight decrement (≤ 0; 0.0 on transmission)
        """
        R    = jnp.float32(self.radius)
        EPS  = jnp.float32(1e-7 * self.radius)
        NUDGE = jnp.float32(1e-4 * self.radius)

        step_l = jnp.linalg.norm(step)
        d_hat  = step / step_l

        # ── Quadratic intersection ────────────────────────────────────────
        dp    = jnp.dot(d_hat, r)
        disc  = dp * dp - (jnp.dot(r, r) - R * R)
        disc_s = jnp.maximum(disc, jnp.float32(0.0))

        # ── Side detection and root selection ────────────────────────────
        inside  = jnp.dot(r, r) < R * R
        t_exit  = -dp + jnp.sqrt(disc_s)   # inside walker exits
        t_entry = -dp - jnp.sqrt(disc_s)   # outside walker enters
        t_hit   = jnp.where(inside, t_exit, t_entry)

        any_hit = (
            (disc > jnp.float32(0.0))
            & (t_hit > EPS)
            & (t_hit < step_l)
            & (step_l > jnp.float32(0.0))
        )
        t_safe = jnp.where(any_hit, t_hit, jnp.float32(0.0))

        # ── Hit geometry ─────────────────────────────────────────────────
        r_hit     = r + t_safe * d_hat
        n_out     = r_hit / R                     # outward normal
        remaining = step_l - t_safe

        # cos(α) = √disc / R  (same as reflect_with_log_weight; works for both
        # inside and outside walkers — see Sphere.reflect_with_log_weight docstring)
        cos_alpha = jnp.sqrt(disc_s) / R
        d_perp    = jnp.where(any_hit, remaining * cos_alpha, jnp.float32(0.0))

        # ── Permeability decision ─────────────────────────────────────────
        p_transmit = jnp.minimum(jnp.float32(1.0),
                                 jnp.float32(2.0) * kappa_over_D * d_perp)
        u        = jax.random.uniform(perm_key, dtype=jnp.float32)
        transmit = any_hit & (u < p_transmit)

        # ── Reflected: specular, nudge back to same side ─────────────────
        d_refl    = d_hat - jnp.float32(2.0) * jnp.dot(d_hat, n_out) * n_out
        d_refl    = d_refl / jnp.linalg.norm(d_refl)
        nudge_dir = jnp.where(inside, -n_out, n_out)   # stay on same side
        r_nudge   = r_hit + NUDGE * nudge_dir
        r_refl    = r_nudge + d_refl * jnp.maximum(remaining - NUDGE,
                                                    jnp.float32(0.0))

        # ── Transmitted: straight through ────────────────────────────────
        r_straight = r + step

        # ── Combine ───────────────────────────────────────────────────────
        r_hit_result = jnp.where(transmit, r_straight, r_refl)
        r_out        = jnp.where(any_hit,  r_hit_result, r + step)

        # ── Relaxivity weight on reflection only ──────────────────────────
        dlog_w = jnp.where(
            any_hit & ~transmit,
            -jnp.float32(2.0) * rho_over_D * d_perp,
            jnp.float32(0.0))

        return r_out, dlog_w


class Cylinder(Geometry):
    """Reflecting infinite cylinder of given radius and orientation.

    Restriction acts in the plane perpendicular to `orientation`.
    Walkers move freely along the cylinder axis.

    Parameters
    ----------
    radius : float
        Cylinder inner radius in metres.
    orientation : array-like of shape (3,)
        Cylinder axis direction (normalised internally).
    surface_relaxivity_t2 : float, optional
        Surface relaxivity ρ₂ in m/s.  When set, each boundary collision
        reduces the walker magnetisation weight by exp(-2·ρ₂·d_out/D),
        where d_out is the step length that would have exited the cylinder.
        This implements the surface-T2 model: 1/T2_eff = 1/T2_bulk + ρ₂·S/V
        with S/V = 2/R for a cylinder.  Default None (no surface relaxation).
    permeability : float, optional
        Membrane permeability κ in m/s.  When set, each boundary crossing
        is probabilistic: the walker transmits through the wall with
        probability p = min(1, 2κ·d_perp/D) and reflects otherwise.
        Enables bidirectional exchange — walkers may be inside or outside
        the cylinder at any time.  Default None (fully reflecting wall).
    """

    def __init__(self, radius: float, orientation, surface_relaxivity_t2=None,
                 permeability=None):
        self.radius = float(radius)
        orientation = np.asarray(orientation, dtype=np.float64)
        self.orientation = (orientation / np.linalg.norm(orientation)).astype(
            np.float32)
        # Rotation matrix: aligns orientation with z-axis
        # R @ orientation = [0, 0, 1]
        _R_np = _rotation_to_z(self.orientation)
        self._R = jnp.array(_R_np, dtype=jnp.float32)
        self._R_inv = jnp.array(_R_np.T, dtype=jnp.float32)
        # GPU batch-matmul bug: when _R == I, XLA's dot_general lowering for
        # vmap(lambda r: _R @ r) produces wrong results on GPU.  Detect the
        # identity case at construction time so permeate/reflect can skip the
        # matmul and use direct indexing instead (pure Python branch, resolved
        # at trace time, so no runtime overhead).
        self._is_identity_rotation = bool(np.allclose(_R_np, np.eye(3)))
        self.surface_relaxivity_t2 = (
            float(surface_relaxivity_t2) if surface_relaxivity_t2 is not None else None
        )
        self.permeability = (
            float(permeability) if permeability is not None else None
        )

    def init_positions(self, n_walkers, key):
        """Uniform sampling in circular cross-section."""
        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**30)))
        # Circle fill factor = π/4 ≈ 78%; batch of 2× is enough for one pass.
        accepted = []
        while sum(len(a) for a in accepted) < n_walkers:
            xy = rng.uniform(-self.radius, self.radius, (n_walkers * 2, 2))
            accepted.append(xy[np.linalg.norm(xy, axis=1) < self.radius])
        xy = np.concatenate(accepted, axis=0)[:n_walkers].astype(np.float32)
        # In cylinder frame: R maps orientation → z (free axis).
        # Restricted cross-section is the x-y plane (indices 0,1); z is free.
        r_cyl = np.stack([xy[:, 0], xy[:, 1], np.zeros(n_walkers)], axis=1)
        R_inv = np.array(self._R_inv)
        r_lab = (R_inv @ r_cyl.T).T
        return jnp.array(r_lab, dtype=jnp.float32)

    def reflect(self, r, step):
        """Specular reflection off cylinder boundary with multiple reflections.

        Works in the cylinder frame (orientation → z-axis via _R). The z-axis
        is free; reflection acts only in the x-y plane (the restricted cross-section).
        Uses the same unit-direction + scalar-remaining convention as disimpy,
        with an epsilon nudge inward after each reflection. jax.lax.scan gives
        a fixed iteration count that is JAX-compilable.
        """
        R          = jnp.float32(self.radius)
        EPS_detect = jnp.float32(1e-7 * self.radius)  # see Sphere.reflect for rationale
        NUDGE      = jnp.float32(1e-4 * self.radius)

        # Transform full step to cylinder frame once.
        # GPU batch-matmul bug: _R @ r with vmap produces wrong r_c when _R==I.
        # Use direct indexing for the identity case (resolved at trace time).
        if self._is_identity_rotation:
            r_c    = r
            step_c = step
        else:
            r_c    = self._R @ r
            step_c = self._R @ step

        # Split into 2D restricted (x-y) and free (z) components
        step_xy = step_c[:2]
        step_z  = step_c[2]

        step_l_xy = jnp.linalg.norm(step_xy)
        # Handle zero lateral step (purely axial movement — no reflection possible)
        d_hat_xy  = jnp.where(step_l_xy > 0, step_xy / step_l_xy,
                               jnp.zeros(2, dtype=jnp.float32))

        def _one_reflection(carry, _):
            r2, d2, remaining = carry  # r2: (2,) xy position, d2: (2,) unit direction

            # Distance to circle boundary along d2 from r2 (r2 inside circle)
            dp   = jnp.dot(d2, r2)
            disc = jnp.maximum(dp * dp - (jnp.dot(r2, r2) - R * R), 0.0)
            d    = -dp + jnp.sqrt(disc)

            intersects = (d > EPS_detect) & (d < remaining) & (step_l_xy > 0)

            r2_hit  = r2 + d * d2
            n2_out  = r2_hit / R
            d2_refl = d2 - 2.0 * jnp.dot(d2, n2_out) * n2_out
            d2_refl = d2_refl / jnp.linalg.norm(d2_refl)
            r2_nudge = r2_hit - NUDGE * n2_out

            r2_new  = jnp.where(intersects, r2_nudge, r2)
            d2_new  = jnp.where(intersects, d2_refl,  d2)
            rem_new = jnp.where(intersects, remaining - d - NUDGE, remaining)

            return (r2_new, d2_new, rem_new), None

        r2_init = r_c[:2]
        (r2_f, d2_f, rem_f), _ = jax.lax.scan(
            _one_reflection, (r2_init, d_hat_xy, step_l_xy), None, length=10
        )
        xy_final  = r2_f + d2_f * jnp.maximum(rem_f, 0.0)
        # Safety clamp for 2D cross-section
        r2_norm   = jnp.linalg.norm(xy_final)
        xy_final  = jnp.where(r2_norm >= R, xy_final * (R - NUDGE) / r2_norm, xy_final)

        # Reconstruct 3D position in cylinder frame: z moves freely
        z_init  = r_c[2]
        z_final = z_init + step_z
        r_c_new = jnp.stack([xy_final[0], xy_final[1], z_final])

        if self._is_identity_rotation:
            return r_c_new
        return self._R_inv @ r_c_new

    def reflect_with_log_weight(self, r, step, rho_over_D):
        """Specular reflection + surface-relaxation log-weight decrement.

        Identical to reflect() but also accumulates the perpendicular outgoing
        step depths across all boundary collisions within one timestep, and
        converts them to a magnetisation log-weight decrement:

            Δlog_w = -2 · ρ_over_D · Σ d_perp_i

        where d_perp_i = (remaining - d) · cos(α) is the perpendicular depth
        of wall penetration at collision i (cos(α) = dot(d_hat, n_out) at hit
        point = sqrt(disc)/R), and rho_over_D = ρ₂/D.  The factor of 2 comes
        from the Brownstein-Tarr boundary condition (ρ · ∂M/∂n = D · M) in the
        partially-absorbing Monte Carlo formulation.  Using d_perp (not the
        full remaining step d_out) is required for the formula to reproduce the
        correct S/V scaling; numerical verification shows that the coefficient
        C = π/2 ≈ 1.571 for d_out and C = 2 for d_perp.

        Parameters
        ----------
        r : (3,) float32, current walker position (lab frame)
        step : (3,) float32, proposed displacement (lab frame)
        rho_over_D : float32, ρ₂/D baked in by make_step_fn

        Returns
        -------
        r_new : (3,) float32, new position (lab frame)
        dlog_w : float32, log-weight decrement (≤ 0)
        """
        R          = jnp.float32(self.radius)
        EPS_detect = jnp.float32(1e-7 * self.radius)
        NUDGE      = jnp.float32(1e-4 * self.radius)

        if self._is_identity_rotation:
            r_c    = r
            step_c = step
        else:
            r_c    = self._R @ r
            step_c = self._R @ step

        step_xy   = step_c[:2]
        step_z    = step_c[2]
        step_l_xy = jnp.linalg.norm(step_xy)
        d_hat_xy  = jnp.where(step_l_xy > 0, step_xy / step_l_xy,
                              jnp.zeros(2, dtype=jnp.float32))

        def _one_reflection(carry, _):
            r2, d2, remaining = carry

            dp   = jnp.dot(d2, r2)
            disc = jnp.maximum(dp * dp - (jnp.dot(r2, r2) - R * R), 0.0)
            d    = -dp + jnp.sqrt(disc)

            intersects = (d > EPS_detect) & (d < remaining) & (step_l_xy > 0)

            r2_hit   = r2 + d * d2
            n2_out   = r2_hit / R
            d2_refl  = d2 - 2.0 * jnp.dot(d2, n2_out) * n2_out
            d2_refl  = d2_refl / jnp.linalg.norm(d2_refl)
            r2_nudge = r2_hit - NUDGE * n2_out

            r2_new  = jnp.where(intersects, r2_nudge, r2)
            d2_new  = jnp.where(intersects, d2_refl,  d2)
            rem_new = jnp.where(intersects, remaining - d - NUDGE, remaining)

            cos_alpha = jnp.sqrt(disc) / R
            d_perp = jnp.where(intersects,
                                (remaining - d) * cos_alpha,
                                jnp.float32(0.0))

            return (r2_new, d2_new, rem_new), d_perp

        r2_init = r_c[:2]
        (r2_f, d2_f, rem_f), d_perps = jax.lax.scan(
            _one_reflection, (r2_init, d_hat_xy, step_l_xy), None, length=10
        )
        xy_final = r2_f + d2_f * jnp.maximum(rem_f, 0.0)
        r2_norm  = jnp.linalg.norm(xy_final)
        xy_final = jnp.where(r2_norm >= R, xy_final * (R - NUDGE) / r2_norm, xy_final)

        z_final = r_c[2] + step_z
        r_c_new = jnp.stack([xy_final[0], xy_final[1], z_final])
        if self._is_identity_rotation:
            r_new = r_c_new
        else:
            r_new = self._R_inv @ r_c_new

        dlog_w = -2.0 * jnp.float32(rho_over_D) * jnp.sum(d_perps)
        return r_new, dlog_w

    def permeate(self, r, step, kappa_over_D, rho_over_D, perm_key):
        """Probabilistic membrane crossing (Powles 2004) + optional relaxivity.

        At each wall crossing the walker transmits through with probability

            p = min(1,  2 · κ/D · d_perp)

        and reflects otherwise (specular, same as reflect()).  When
        rho_over_D > 0 a Brownstein-Tarr weight is applied on reflection:

            Δlog_w = −2 · ρ/D · d_perp

        No weight is applied on transmission.  The geometry is bidirectional:
        walkers may be inside (|r_xy| < R) or outside (|r_xy| > R); the
        appropriate intersection root is selected automatically.

        Single-event-per-step approximation: at most one permeability
        decision per timestep (consistent with the PackedCylinders exterior
        method).  Requires σ/R < 0.1 for accurate results.

        Parameters
        ----------
        r          : (3,) float32, current position (lab frame)
        step       : (3,) float32, proposed displacement (lab frame)
        kappa_over_D : float32, κ/D baked in by make_step_fn
        rho_over_D   : float32, ρ/D  (0.0 if no surface relaxivity)
        perm_key   : JAX PRNGKey for the Bernoulli draw

        Returns
        -------
        r_new   : (3,) float32, new position
        dlog_w  : float32, log-weight decrement (≤ 0; 0.0 on transmission)
        """
        R          = jnp.float32(self.radius)
        EPS        = jnp.float32(1e-7 * self.radius)
        NUDGE      = jnp.float32(1e-4 * self.radius)

        if self._is_identity_rotation:
            r_c    = r
            step_c = step
        else:
            r_c    = self._R @ r
            step_c = self._R @ step
        r2     = r_c[:2]
        step_xy   = step_c[:2]
        step_z    = step_c[2]
        step_l_xy = jnp.linalg.norm(step_xy)
        d_hat_xy  = jnp.where(
            step_l_xy > jnp.float32(0.0),
            step_xy / step_l_xy,
            jnp.zeros(2, dtype=jnp.float32))

        # ── Ray-circle intersection ──────────────────────────────────────
        dp   = jnp.dot(d_hat_xy, r2)
        disc = dp * dp - (jnp.dot(r2, r2) - R * R)
        disc_s = jnp.maximum(disc, jnp.float32(0.0))

        # ── Side detection and root selection ────────────────────────────
        # inside: use exit root  t = −dp + √disc  (forward intersection)
        # outside: use entry root t = −dp − √disc (entry into cylinder)
        inside   = jnp.dot(r2, r2) < R * R
        t_exit   = -dp + jnp.sqrt(disc_s)
        t_entry  = -dp - jnp.sqrt(disc_s)
        t_hit    = jnp.where(inside, t_exit, t_entry)

        any_hit = (
            (disc > jnp.float32(0.0))
            & (t_hit > EPS)
            & (t_hit < step_l_xy)
            & (step_l_xy > jnp.float32(0.0))
        )
        t_safe = jnp.where(any_hit, t_hit, jnp.float32(0.0))

        # ── Hit geometry ─────────────────────────────────────────────────
        # Compute raw hit point and normalize to get outward unit normal.
        # Then SNAP the hit point exactly to the boundary (R * n_out).
        # Float32 arithmetic on r2 + t_safe * d_hat_xy can place the hit
        # point slightly outside the cylinder; if the nudge then starts from
        # outside, NUDGE * (-n_out) may not be large enough to push the walker
        # back inside, leaving it outside on the next step.  Snapping to the
        # boundary makes the nudge direction deterministically correct.
        r2_hit_raw = r2 + t_safe * d_hat_xy
        r2_hit_len = jnp.linalg.norm(r2_hit_raw)
        n_out      = r2_hit_raw / jnp.maximum(r2_hit_len, jnp.float32(1e-30))
        r2_hit     = R * n_out          # snapped to boundary: |r2_hit| = R exactly
        remaining  = step_l_xy - t_safe

        # cos(α) = √disc / R  (same formula as reflect_with_log_weight)
        cos_alpha = jnp.sqrt(disc_s) / R
        d_perp    = jnp.where(any_hit, remaining * cos_alpha, jnp.float32(0.0))

        # ── Permeability decision ─────────────────────────────────────────
        p_transmit = jnp.minimum(jnp.float32(1.0),
                                 jnp.float32(2.0) * kappa_over_D * d_perp)
        u        = jax.random.uniform(perm_key, dtype=jnp.float32)
        transmit = any_hit & (u < p_transmit)

        # ── Transmitted position: straight through ───────────────────────
        r2_straight = r2 + step_xy

        # ── Reflected position: specular, nudge back to same side ────────
        d2_refl  = d_hat_xy - jnp.float32(2.0) * jnp.dot(d_hat_xy, n_out) * n_out
        d2_refl  = d2_refl / jnp.linalg.norm(d2_refl)
        # nudge direction: inward for inside walker, outward for outside walker
        nudge_dir  = jnp.where(inside, -n_out, n_out)
        # r2_hit is exactly on boundary → nudge is guaranteed to land inside/outside
        r2_nudge   = r2_hit + NUDGE * nudge_dir
        r2_refl    = r2_nudge + d2_refl * jnp.maximum(remaining - NUDGE,
                                                        jnp.float32(0.0))

        # ── Safety clamp: keep reflected position on the correct side ─────────
        # Specular reflection off a curved surface can leave |r2_refl| slightly
        # on the wrong side (e.g. tangential steps along the cylinder wall push
        # the walker outside).  GPU float32 rounding makes this 5-6× more
        # likely than CPU.  Mirror the clamp in reflect() / reflect_with_log_weight():
        #   inside walker → |r2_refl| must be < R  → clamp to R-NUDGE
        #   outside walker→ |r2_refl| must be > R  → clamp to R+NUDGE
        r2_refl_norm      = jnp.linalg.norm(r2_refl)
        r2_refl_norm_safe = jnp.maximum(r2_refl_norm, jnp.float32(1e-30))
        target_refl       = jnp.where(inside, R - NUDGE, R + NUDGE)
        wrong_side_refl   = jnp.where(inside, r2_refl_norm >= R, r2_refl_norm <= R)
        r2_refl = jnp.where(wrong_side_refl,
                             r2_refl * target_refl / r2_refl_norm_safe,
                             r2_refl)

        # ── Combine: select transmit / reflect / no-hit ─────────────────
        r2_hit_result = jnp.where(transmit, r2_straight, r2_refl)
        xy_final      = jnp.where(any_hit, r2_hit_result, r2 + step_xy)

        # ── Relaxivity weight on reflection only ─────────────────────────
        dlog_w = jnp.where(
            any_hit & ~transmit,
            -jnp.float32(2.0) * rho_over_D * d_perp,
            jnp.float32(0.0))

        # ── Reconstruct 3-D position ──────────────────────────────────────
        # Build absolute cylinder-frame position then rotate to lab frame.
        # For the identity-rotation case (_is_identity_rotation=True) we skip
        # the _R_inv matmul entirely: the GPU batch-matmul for _R==I gives
        # wrong r_c values (XLA dot_general identity-matrix bug), so both the
        # input transform (_R @ r) and output transform (_R_inv @ r_c_new) are
        # bypassed by Python-level branching resolved at trace time.
        z_final   = r_c[2] + step_z
        r_c_new   = jnp.stack([xy_final[0], xy_final[1], z_final])
        if self._is_identity_rotation:
            return r_c_new, dlog_w
        return self._R_inv @ r_c_new, dlog_w

    def classify_position(self, r: jnp.ndarray) -> jnp.ndarray:
        """Compartment ID: 0=intra (|r_xy| < R), 1=extra (|r_xy| >= R).

        The check is performed in the cylinder frame (r_xy is the component
        perpendicular to the cylinder axis).
        """
        R = jnp.float32(self.radius)
        r_c = r if self._is_identity_rotation else self._R @ r
        r_xy_sq = jnp.dot(r_c[:2], r_c[:2])
        inside = r_xy_sq < R * R
        return jnp.where(inside, jnp.int32(0), jnp.int32(1))

    def volume(self, L: float = 1.0) -> float:
        """Volume of the cylinder: π·R²·L (m³).

        Parameters
        ----------
        L : float, optional
            Cylinder length in metres. Default 1.0 (returns per-unit-length
            volume, i.e. the cross-sectional area π·R²).
        """
        return np.pi * self.radius ** 2 * float(L)

    def surface_area(self, L: float = 1.0, include_caps: bool = False) -> float:
        """Lateral surface area of the cylinder: 2π·R·L (m²).

        Caps are excluded by default because the cylinder is modelled as
        infinite (periodic along its axis) and caps are irrelevant for
        permeability and relaxivity calculations.

        Parameters
        ----------
        L : float, optional
            Cylinder length in metres. Default 1.0 (returns per-unit-length
            lateral area, i.e. the circumference 2·π·R).
        include_caps : bool, optional
            If True, add the two circular end caps 2·π·R². Default False.
        """
        lateral = 2.0 * np.pi * self.radius * float(L)
        if include_caps:
            lateral += 2.0 * np.pi * self.radius ** 2
        return lateral


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
        Cylinder axis direction (normalised internally).
    D_intra : float
        Intra-axonal diffusivity in m^2/s (isotropic).
    D_extra : float
        Extra-axonal diffusivity in m^2/s (isotropic).
    D_myelin : float, optional
        Myelin-water diffusivity in m^2/s (isotropic). Default 0 (stuck pool).
    kappa_inner : float, optional
        Permeability at inner boundary (m/s). Default None (impermeable).
    kappa_outer : float, optional
        Permeability at outer boundary (m/s). Default None (impermeable).
    T2_intra : float, optional
        T2 relaxation time for intra-axonal compartment (s).
    T2_myelin : float, optional
        T2 relaxation time for myelin compartment (s).
    T2_extra : float, optional
        T2 relaxation time for extra-axonal compartment (s).
    water_fractions : tuple of 3 floats, optional
        Relative water content (proton density) per compartment (intra, myelin,
        extra). ``None`` (default) uses the biophysical table: myelin =
        ``myelin_water_proton_density`` (~0.40, the water per unit myelin VOLUME), intra
        and extra = 1.0. Do NOT pass the MWF signal value (~0.15) here -- that is a
        measured signal fraction, not a per-volume weight.
    """

    # Marker: this geometry provides its own step function (not make_step_fn)
    _is_myelinated = True

    def __init__(self, inner_radius, outer_radius, orientation,
                 D_intra, D_extra, D_myelin=0.0,
                 kappa_inner=None, kappa_outer=None,
                 T2_intra=None, T2_myelin=None, T2_extra=None,
                 water_fractions=None):
        if outer_radius <= inner_radius:
            raise ValueError("outer_radius must be > inner_radius")

        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self.D_intra = float(D_intra)
        self.D_myelin = float(D_myelin)
        self.D_extra = float(D_extra)
        self.kappa_inner = float(kappa_inner) if kappa_inner is not None else None
        self.kappa_outer = float(kappa_outer) if kappa_outer is not None else None
        self.T2_intra = float(T2_intra) if T2_intra is not None else None
        self.T2_myelin = float(T2_myelin) if T2_myelin is not None else None
        self.T2_extra = float(T2_extra) if T2_extra is not None else None
        if water_fractions is None:
            from .substrate.biophysical_constants import get_default_value
            water_fractions = (1.0, float(get_default_value('myelin_water_proton_density')), 1.0)
        self.water_fractions = tuple(float(w) for w in water_fractions)

        orientation = np.asarray(orientation, dtype=np.float64)
        self.orientation = (orientation / np.linalg.norm(orientation)).astype(
            np.float32)
        _R_np = _rotation_to_z(self.orientation)
        self._R = jnp.array(_R_np, dtype=jnp.float32)
        self._R_inv = jnp.array(_R_np.T, dtype=jnp.float32)
        self._is_identity_rotation = bool(np.allclose(_R_np, np.eye(3)))

    def init_positions(self, n_walkers, key):
        """Distribute walkers proportional to volume * water_fraction per compartment.

        Extra-axonal volume is set to the annulus between R_outer and 2*R_outer
        for allocation purposes (actual walkers are placed uniformly).
        """
        R_in = self.inner_radius
        R_out = self.outer_radius
        wf = list(self.water_fractions)

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

        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**30)))

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
            compartments[idx:idx + n_intra] = 0
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
            compartments[idx:idx + n_myelin] = 1
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
            compartments[idx:idx + n_extra] = 2
            idx += n_extra

        # Positions are in cylinder frame (xy = cross-section, z = axis).
        # Rotate to lab frame.
        R_inv = np.array(self._R_inv)
        r_lab = (R_inv @ positions.T).T

        self._init_compartments = jnp.array(compartments, dtype=jnp.int32)
        return jnp.array(r_lab, dtype=jnp.float32)

    def reflect(self, r, step):
        """Reflect off boundaries — dispatches based on compartment_id.

        Not used directly for MyelinatedCylinder (handled by custom step_fn).
        Provides a fallback reflecting on the inner boundary only.
        """
        # This method is not used by the custom step_fn but exists
        # to satisfy the abstract method requirement.
        R = jnp.float32(self.inner_radius)
        if self._is_identity_rotation:
            r_c    = r
            step_c = step
        else:
            r_c    = self._R @ r
            step_c = self._R @ step
        r_new_c = r_c + step_c
        # Clamp to inner cylinder as fallback
        r_xy = jnp.linalg.norm(r_new_c[:2])
        NUDGE = jnp.float32(1e-4 * self.inner_radius)
        r_new_c = r_new_c.at[:2].set(
            jnp.where(r_xy >= R, r_new_c[:2] * (R - NUDGE) / r_xy, r_new_c[:2])
        )
        if self._is_identity_rotation:
            return r_new_c
        return self._R_inv @ r_new_c

    def classify_position(self, r: jnp.ndarray) -> jnp.ndarray:
        """Compartment ID from position: 0=intra, 1=myelin, 2=extra.

        Classification is based on the radial distance in the cylinder
        cross-section (r_xy):
          - |r_xy| < R_inner  → 0 (intra-axonal)
          - R_inner <= |r_xy| < R_outer → 1 (myelin)
          - |r_xy| >= R_outer → 2 (extra-axonal)
        """
        R_in  = jnp.float32(self.inner_radius)
        R_out = jnp.float32(self.outer_radius)
        r_c   = r if self._is_identity_rotation else self._R @ r
        r_xy_sq = jnp.dot(r_c[:2], r_c[:2])
        in_intra  = r_xy_sq < R_in  * R_in
        in_myelin = (r_xy_sq >= R_in * R_in) & (r_xy_sq < R_out * R_out)
        comp = jnp.where(in_intra, jnp.int32(0),
               jnp.where(in_myelin, jnp.int32(1), jnp.int32(2)))
        return comp

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


class Ellipsoid(Geometry):
    """Reflecting axis-aligned ellipsoid with semi-axes (a, b, c) along (x, y, z).

    The ellipsoid surface is defined by x²/a² + y²/b² + z²/c² = 1.
    When a = b = c = r the geometry is identical to Sphere(r).
    """

    def __init__(self, semiaxes, surface_relaxivity_t2=None, permeability=None):
        """
        Parameters
        ----------
        semiaxes : array-like of shape (3,)
            Semi-axes [a, b, c] in metres along x, y, z respectively.
        surface_relaxivity_t2 : float, optional
            Surface relaxivity ρ₂ in m/s. When set, boundary collisions reduce
            walker magnetisation by exp(-2·ρ₂·d_perp/D). Default None.
        permeability : float, optional
            Membrane permeability κ in m/s.  When set, each boundary crossing
            is probabilistic: p = min(1, 2κ·d_perp/D).  Bidirectional.
            Default None (fully reflecting wall).
        """
        self.semiaxes = np.asarray(semiaxes, dtype=np.float64)
        self._semi_f32 = jnp.array(self.semiaxes, dtype=jnp.float32)
        self.surface_relaxivity_t2 = (
            float(surface_relaxivity_t2) if surface_relaxivity_t2 is not None else None
        )
        self.permeability = (
            float(permeability) if permeability is not None else None
        )

    def init_positions(self, n_walkers, key):
        """Uniform sampling inside ellipsoid.

        Samples uniformly from the unit ball then scales by semiaxes.  The
        linear map (u_x, u_y, u_z) → (a*u_x, b*u_y, c*u_z) has constant
        Jacobian a*b*c, so the result is uniform inside the ellipsoid.
        """
        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**30)))
        # Unit-ball fill factor ≈ π/6 ≈ 52%; batch of 4× usually suffices.
        accepted = []
        while sum(len(a) for a in accepted) < n_walkers:
            pts = rng.uniform(-1.0, 1.0, (n_walkers * 4, 3))
            accepted.append(pts[np.linalg.norm(pts, axis=1) < 1.0])
        pts = np.concatenate(accepted, axis=0)[:n_walkers]
        positions = pts * self.semiaxes  # scale each axis independently
        return jnp.array(positions, dtype=jnp.float32)

    def reflect(self, r, step):
        """Specular reflection off ellipsoid boundary with multiple reflections.

        Uses the same unit-direction + scalar-remaining convention as Sphere,
        implemented via jax.lax.scan (10 fixed iterations).

        Line-ellipsoid intersection: ray r0 + t*d_hat intersects the ellipsoid
        x²/a² + y²/b² + z²/c² = 1 when:
          A*t² + 2*B*t + C = 0
          A = d·D·d,  B = r0·D·d,  C = r0·D·r0 - 1
          D = diag(1/a², 1/b², 1/c²)
        Forward root: t = (-B + sqrt(B²-A*C)) / A

        Outward normal at r_hit: n ∝ D·r_hit, normalised.
        """
        semi = self._semi_f32                             # (3,)  [a, b, c]
        inv_semi_sq = 1.0 / (semi * semi)                # (3,)  [1/a², 1/b², 1/c²]
        # Two-level epsilon: see Sphere.reflect for rationale.
        # Use smallest semiaxis to scale so all directions are safe.
        _min_semi   = float(np.min(self.semiaxes))
        EPS_detect  = jnp.float32(1e-7 * _min_semi)
        NUDGE       = jnp.float32(1e-4 * _min_semi)

        step_l = jnp.linalg.norm(step)
        d_hat  = step / step_l

        def _one_reflection(carry, _):
            r0, d_hat, remaining = carry

            # Quadratic coefficients for line-ellipsoid intersection
            A    = jnp.dot(d_hat * inv_semi_sq, d_hat)
            B    = jnp.dot(r0   * inv_semi_sq, d_hat)
            C    = jnp.dot(r0   * inv_semi_sq, r0) - 1.0
            disc = jnp.maximum(B * B - A * C, 0.0)
            d    = (-B + jnp.sqrt(disc)) / A             # forward distance to surface

            intersects = (d > EPS_detect) & (d < remaining)

            r_hit   = r0 + d * d_hat
            # Outward normal: gradient of f(x)=x·D·x at r_hit, normalised
            n_raw   = r_hit * inv_semi_sq
            n_out   = n_raw / jnp.linalg.norm(n_raw)
            d_refl  = d_hat - 2.0 * jnp.dot(d_hat, n_out) * n_out
            d_refl  = d_refl / jnp.linalg.norm(d_refl)
            r_nudge = r_hit - NUDGE * n_out

            r0_new   = jnp.where(intersects, r_nudge,  r0)
            dhat_new = jnp.where(intersects, d_refl,   d_hat)
            rem_new  = jnp.where(intersects, remaining - d - NUDGE, remaining)

            return (r0_new, dhat_new, rem_new), None

        (r_f, d_hat_f, rem_f), _ = jax.lax.scan(
            _one_reflection, (r, d_hat, step_l), None, length=10
        )
        r_out = r_f + d_hat_f * jnp.maximum(rem_f, 0.0)
        # Safety clamp: project back inside if escaped (use normalised coords)
        q_norm = jnp.linalg.norm(r_out * jnp.sqrt(inv_semi_sq))  # = sqrt(sum(r²/a²))
        r_out  = jnp.where(q_norm >= 1.0, r_out * (1.0 - NUDGE / _min_semi) / q_norm, r_out)
        return r_out

    def reflect_with_log_weight(self, r, step, rho_over_D):
        """Reflect and accumulate surface-relaxation log-weight for the ellipsoid.

        At each collision, d_perp = (remaining - d) * cos(α) where
        cos(α) = dot(d_hat, n_out) at the hit point, with the ellipsoid outward
        normal n_out ∝ r_hit * D_inv (diagonal scaling matrix).

        Returns (r_new, dlog_w) where dlog_w = -2 * rho_over_D * sum(d_perp).
        For a sphere (a=b=c=R), S/V = 3/R → T2_surface = R/(3ρ).
        """
        semi = self._semi_f32
        inv_semi_sq = 1.0 / (semi * semi)
        _min_semi   = float(np.min(self.semiaxes))
        EPS_detect  = jnp.float32(1e-7 * _min_semi)
        NUDGE       = jnp.float32(1e-4 * _min_semi)

        step_l = jnp.linalg.norm(step)
        d_hat  = step / step_l

        def _one_reflection(carry, _):
            r0, d_hat, remaining = carry

            A    = jnp.dot(d_hat * inv_semi_sq, d_hat)
            B    = jnp.dot(r0   * inv_semi_sq, d_hat)
            C    = jnp.dot(r0   * inv_semi_sq, r0) - 1.0
            disc = jnp.maximum(B * B - A * C, 0.0)
            d    = (-B + jnp.sqrt(disc)) / A

            intersects = (d > EPS_detect) & (d < remaining)

            r_hit   = r0 + d * d_hat
            n_raw   = r_hit * inv_semi_sq
            n_out   = n_raw / jnp.linalg.norm(n_raw)
            d_refl  = d_hat - 2.0 * jnp.dot(d_hat, n_out) * n_out
            d_refl  = d_refl / jnp.linalg.norm(d_refl)
            r_nudge = r_hit - NUDGE * n_out

            r0_new   = jnp.where(intersects, r_nudge,  r0)
            dhat_new = jnp.where(intersects, d_refl,   d_hat)
            rem_new  = jnp.where(intersects, remaining - d - NUDGE, remaining)

            # cos(α) = dot(d_hat, n_out) at hit point; d_perp = (remaining-d)*cos(α)
            cos_alpha = jnp.dot(d_hat, n_out)
            d_perp = jnp.where(intersects,
                               (remaining - d) * cos_alpha,
                               jnp.float32(0.0))

            return (r0_new, dhat_new, rem_new), d_perp

        (r_f, d_hat_f, rem_f), d_perps = jax.lax.scan(
            _one_reflection, (r, d_hat, step_l), None, length=10
        )
        r_out = r_f + d_hat_f * jnp.maximum(rem_f, 0.0)
        q_norm = jnp.linalg.norm(r_out * jnp.sqrt(inv_semi_sq))
        r_out  = jnp.where(q_norm >= 1.0, r_out * (1.0 - NUDGE / _min_semi) / q_norm, r_out)

        dlog_w = -2.0 * jnp.float32(rho_over_D) * jnp.sum(d_perps)
        return r_out, dlog_w

    def permeate(self, r, step, kappa_over_D, rho_over_D, perm_key):
        """Probabilistic membrane crossing (Powles 2004) + optional relaxivity.

        Same protocol as Sphere.permeate but for a general ellipsoid.
        Intersection uses the ellipsoid quadratic A·t² + 2B·t + C = 0 with
        D = diag(1/a², 1/b², 1/c²):

            A = d̂·D·d̂,  B = r·D·d̂,  C = r·D·r − 1

        Inside  (C < 0): forward root  t = (−B + √(B²−A·C)) / A
        Outside (C ≥ 0): backward root t = (−B − √(B²−A·C)) / A

        cos(α) = |d̂·n_out| at the hit point, n_out ∝ r_hit·D (normalised).
        d_perp = remaining · cos(α).

        Single-event-per-step approximation.  Requires σ/min_semi < 0.1.

        Parameters
        ----------
        r          : (3,) float32, current position
        step       : (3,) float32, proposed displacement
        kappa_over_D : float32, κ/D
        rho_over_D   : float32, ρ/D  (0.0 if no surface relaxivity)
        perm_key   : JAX PRNGKey

        Returns
        -------
        r_new  : (3,) float32
        dlog_w : float32
        """
        semi        = self._semi_f32                     # (3,) [a, b, c]
        inv_semi_sq = jnp.float32(1.0) / (semi * semi)  # (3,) [1/a², 1/b², 1/c²]
        _min_semi   = float(np.min(self.semiaxes))
        EPS         = jnp.float32(1e-7 * _min_semi)
        NUDGE       = jnp.float32(1e-4 * _min_semi)

        step_l = jnp.linalg.norm(step)
        d_hat  = step / step_l

        # ── Ellipsoid quadratic ──────────────────────────────────────────
        A      = jnp.dot(d_hat * inv_semi_sq, d_hat)
        B      = jnp.dot(r     * inv_semi_sq, d_hat)
        C      = jnp.dot(r     * inv_semi_sq, r) - jnp.float32(1.0)
        disc_A = jnp.maximum(B * B - A * C, jnp.float32(0.0))

        # ── Side detection and root selection ────────────────────────────
        inside  = C < jnp.float32(0.0)                             # r·D·r < 1
        t_exit  = (-B + jnp.sqrt(disc_A)) / A                     # inside exits
        t_entry = (-B - jnp.sqrt(disc_A)) / A                     # outside enters
        t_hit   = jnp.where(inside, t_exit, t_entry)

        disc_raw = B * B - A * C                                   # unclipped
        any_hit  = (
            (disc_raw > jnp.float32(0.0))
            & (t_hit  > EPS)
            & (t_hit  < step_l)
            & (step_l > jnp.float32(0.0))
        )
        t_safe = jnp.where(any_hit, t_hit, jnp.float32(0.0))

        # ── Hit geometry ─────────────────────────────────────────────────
        r_hit   = r + t_safe * d_hat
        n_raw   = r_hit * inv_semi_sq
        n_out   = n_raw / jnp.linalg.norm(n_raw)           # outward normal
        remaining = step_l - t_safe

        # cos(α) = |d̂·n_out|; always positive for both inside and outside walkers
        cos_alpha = jnp.abs(jnp.dot(d_hat, n_out))
        d_perp    = jnp.where(any_hit, remaining * cos_alpha, jnp.float32(0.0))

        # ── Permeability decision ─────────────────────────────────────────
        p_transmit = jnp.minimum(jnp.float32(1.0),
                                 jnp.float32(2.0) * kappa_over_D * d_perp)
        u        = jax.random.uniform(perm_key, dtype=jnp.float32)
        transmit = any_hit & (u < p_transmit)

        # ── Reflected: specular, nudge back to same side ─────────────────
        d_refl    = d_hat - jnp.float32(2.0) * jnp.dot(d_hat, n_out) * n_out
        d_refl    = d_refl / jnp.linalg.norm(d_refl)
        nudge_dir = jnp.where(inside, -n_out, n_out)        # stay on same side
        r_nudge   = r_hit + NUDGE * nudge_dir
        r_refl    = r_nudge + d_refl * jnp.maximum(remaining - NUDGE,
                                                    jnp.float32(0.0))

        # ── Transmitted: straight through ────────────────────────────────
        r_straight = r + step

        # ── Combine ───────────────────────────────────────────────────────
        r_hit_result = jnp.where(transmit, r_straight, r_refl)
        r_out        = jnp.where(any_hit,  r_hit_result, r + step)

        # ── Relaxivity weight on reflection only ──────────────────────────
        dlog_w = jnp.where(
            any_hit & ~transmit,
            -jnp.float32(2.0) * rho_over_D * d_perp,
            jnp.float32(0.0))

        return r_out, dlog_w

    def classify_position(self, r: jnp.ndarray) -> jnp.ndarray:
        """Compartment ID: 0=intra (inside ellipsoid), 1=extra (outside).

        Uses the ellipsoid equation: x²/a² + y²/b² + z²/c² < 1 → intra.
        """
        inv_semi_sq = jnp.float32(1.0) / (self._semi_f32 * self._semi_f32)
        inside = jnp.dot(r * inv_semi_sq, r) < jnp.float32(1.0)
        return jnp.where(inside, jnp.int32(0), jnp.int32(1))

    def volume(self) -> float:
        """Volume of the ellipsoid: (4/3)·π·a·b·c (m³)."""
        a, b, c = self.semiaxes
        return (4.0 / 3.0) * np.pi * a * b * c

    def surface_area(self) -> float:
        """Surface area of the ellipsoid using the Thomsen approximation (m²).

        Thomsen (2004) approximation:
            S ≈ 4π · ((a^p·b^p + a^p·c^p + b^p·c^p) / 3)^(1/p)
        with p = 1.6075.  Relative error < 1.061% for all ellipsoids.
        """
        p = 1.6075
        a, b, c = self.semiaxes
        ap, bp, cp = a ** p, b ** p, c ** p
        return 4.0 * np.pi * ((ap * bp + ap * cp + bp * cp) / 3.0) ** (1.0 / p)
def _is_inside_batch(pts, vertices, faces, batch_size=2000):
    """Return bool (N,) array: True if each point is inside the closed mesh.

    Shoots a +X ray from each point and counts triangle intersections (Möller-
    Trumbore).  Odd count = inside (Jordan curve theorem generalisation).

    Processes points in batches of `batch_size` to bound peak memory to roughly
    batch_size × N_tri × 3 × 8 bytes ≈ 30 MB for batch_size=2000, N_tri=600.
    """
    tris = vertices[faces]        # (N_tri, 3, 3)
    A    = tris[:, 0, :]          # (N_tri, 3)
    E1   = tris[:, 1, :] - A      # (N_tri, 3)
    E2   = tris[:, 2, :] - A      # (N_tri, 3)

    d  = np.array([1.0, 0.0, 0.0])           # +X ray direction
    P  = np.cross(d[None, :], E2)            # (N_tri, 3)  constant for +X
    det = (P * E1).sum(axis=1)               # (N_tri,)

    inside = np.zeros(len(pts), dtype=bool)
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(0, len(pts), batch_size):
            batch = pts[i : i + batch_size]          # (B, 3)
            T     = batch[:, None, :] - A[None, :, :]   # (B, N_tri, 3)
            u     = (P[None] * T).sum(axis=2) / det     # (B, N_tri)
            Q     = np.cross(T, E1[None])               # (B, N_tri, 3)
            v     = (Q * d[None, None, :]).sum(axis=2) / det  # (B, N_tri)
            t_val = (Q * E2[None]).sum(axis=2) / det          # (B, N_tri)
            # det≈0 (ray ∥ triangle) → u/v/t = ±inf or nan;
            # inf > 1.0 → False, nan comparisons → False — all correctly excluded.
            valid = (
                (t_val > 0.0)
                & (u >= 0.0) & (u <= 1.0)
                & (v >= 0.0) & (u + v <= 1.0)
            )
            inside[i : i + batch_size] = (valid.sum(axis=1) % 2) == 1
    return inside


def _rotation_to_z(v):
    """Compute 3x3 rotation matrix R such that R @ v = [0, 0, 1].

    Uses Rodrigues' formula. Handles parallel and anti-parallel cases.
    """
    v = np.asarray(v, dtype=np.float64)
    v = v / np.linalg.norm(v)
    k = np.array([0.0, 0.0, 1.0])

    dot = np.dot(v, k)
    if abs(dot - 1.0) < 1e-10:
        return np.eye(3)
    if abs(dot + 1.0) < 1e-10:
        # Anti-parallel: rotate 180° about x-axis
        return np.diag([1.0, -1.0, -1.0])

    axis = np.cross(v, k)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))

    # Rodrigues: R = I + sin(θ)K + (1-cos(θ))K²
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], dtype=np.float64)
    R = (np.eye(3)
         + np.sin(angle) * K
         + (1 - np.cos(angle)) * (K @ K))
    return R


# ---------------------------------------------------------------------------
# Packed-cylinder substrate (extra-axonal compartment)
# ---------------------------------------------------------------------------

def pack_cylinders(radii, target_vf=None, L=None, seed=0, max_attempts=100_000):
    """Pack N parallel cylinders in a periodic 2-D square domain using RSA.

    Random Sequential Addition (RSA) places cylinders one by one, rejecting
    positions that would cause overlap with any previously placed cylinder
    (including periodic images).  Large cylinders are placed first to maximise
    the achievable packing fraction.  RSA typically reaches VF ≈ 0.45 for
    monodisperse and ≈ 0.55 for polydisperse populations.

    The returned centres are in the 2-D cross-section plane (the plane
    perpendicular to the cylinder axis).  Pass them to ``PackedCylinders``
    together with the same radii, L, and orientation.

    Parameters
    ----------
    radii : array-like, shape (N,)
        Cylinder radii in metres.  All must be positive.
    target_vf : float, optional
        Target intra-cylindrical volume fraction Σπrᵢ²/L².  Used to derive L.
        Mutually exclusive with ``L``.
    L : float, optional
        Side-length of the periodic square domain in metres.
        Mutually exclusive with ``target_vf``.
    seed : int
        NumPy RNG seed for reproducible packing.
    max_attempts : int
        Maximum random placement trials per cylinder before raising
        ``RuntimeError``.  Increase for high target_vf.

    Returns
    -------
    centers : np.ndarray, shape (N, 2)
        Cylinder centre positions in the 2-D cross-section plane, metres.
    L : float
        Side-length actually used.
    achieved_vf : float
        Achieved intra-cylindrical volume fraction = Σπrᵢ² / L².

    Raises
    ------
    ValueError
        On invalid inputs (conflicting L/target_vf, non-positive radii, radii
        exceeding L/2).
    RuntimeError
        If RSA cannot place a cylinder within ``max_attempts`` trials.  Try
        reducing ``target_vf`` or increasing ``max_attempts``.
    """
    radii = np.asarray(radii, dtype=np.float64).ravel()
    if len(radii) == 0:
        raise ValueError("radii must contain at least one element.")
    if np.any(radii <= 0):
        raise ValueError("All radii must be positive.")
    if (target_vf is None) == (L is None):
        raise ValueError("Provide exactly one of target_vf or L.")

    if target_vf is not None:
        if not 0.0 < float(target_vf) < 1.0:
            raise ValueError(f"target_vf must be in (0, 1), got {target_vf}.")
        L = float(np.sqrt(np.pi * np.sum(radii ** 2) / float(target_vf)))
    else:
        L = float(L)

    if np.any(radii > L / 2.0):
        raise ValueError(
            f"Largest radius ({np.max(radii) * 1e6:.2f} µm) exceeds L/2 "
            f"({L / 2.0 * 1e6:.2f} µm).  Reduce target_vf or supply a larger L.")

    rng = np.random.default_rng(int(seed))
    # Place largest cylinders first — improves RSA packing fraction.
    order = np.argsort(radii)[::-1]
    radii_s   = radii[order]
    centers_s = np.zeros((len(radii), 2))

    for i, r_new in enumerate(radii_s):
        placed = False
        for _ in range(max_attempts):
            c_new = rng.uniform(-L / 2.0, L / 2.0, 2)
            ok = True
            for j in range(i):
                dq = c_new - centers_s[j]
                dq -= L * np.round(dq / L)   # minimum-image distance
                if np.linalg.norm(dq) < r_new + radii_s[j]:
                    ok = False
                    break
            if ok:
                centers_s[i] = c_new
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"RSA failed after {max_attempts} attempts placing cylinder {i} "
                f"(r = {r_new * 1e6:.2f} µm).  "
                f"The target packing fraction may exceed what RSA can achieve; "
                f"try reducing target_vf or increasing max_attempts.")

    # Restore original cylinder ordering
    centers_out = np.empty_like(centers_s)
    centers_out[order] = centers_s
    achieved_vf = float(np.pi * np.sum(radii ** 2) / L ** 2)
    return centers_out, L, achieved_vf


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
        """Specular exterior reflection off the nearest cylinder + periodic wrap.

        Applies one reflection per timestep (valid when step_length ≪ min_gap).
        Finds the nearest intersecting cylinder via a vectorised ray-circle
        intersection test across all N cylinders, then reflects specularly.
        Periodic boundary conditions are enforced via a minimum-image convention
        during intersection testing and a final modular wrap of the position.
        """
        L          = self._L_jax
        centers_2d = self._centers_jax    # (N, 2)
        radii_arr  = self._radii_jax      # (N,)
        EPS        = self._eps_detect
        NUDGE      = self._nudge

        # ── Transform to cylinder frame (orientation → z free axis) ──────────
        if self._is_identity_rotation:
            r_c    = r
            step_c = step
        else:
            r_c    = self._R @ r
            step_c = self._R @ step

        r2      = r_c[:2]      # (2,) current position in cross-section
        step_xy = step_c[:2]   # (2,) proposed xy displacement
        step_z  = step_c[2]    # scalar, free direction

        step_l_xy = jnp.linalg.norm(step_xy)
        d_hat_xy  = jnp.where(
            step_l_xy > jnp.float32(0.0),
            step_xy / step_l_xy,
            jnp.zeros(2, dtype=jnp.float32))

        # ── Vectorised ray-circle entry test across all N cylinders ───────────
        # Minimum-image relative positions q_i = r2 - c_i  (N, 2)
        q_all = r2[None, :] - centers_2d
        q_all = q_all - L * jnp.floor(q_all / L + jnp.float32(0.5))

        # t_entry = -dp - sqrt(dp² - (|q|² - R²))
        # Positive when the ray points toward the cylinder and enters it.
        dp_all   = jnp.sum(d_hat_xy[None, :] * q_all, axis=1)          # (N,)
        disc_all = dp_all ** 2 - (jnp.sum(q_all ** 2, axis=1)
                                  - radii_arr ** 2)                      # (N,)
        disc_s   = jnp.maximum(disc_all, jnp.float32(0.0))
        t_all    = -dp_all - jnp.sqrt(disc_s)                           # (N,)

        valid = (
            (disc_all > jnp.float32(0.0))
            & (t_all  > EPS)
            & (t_all  < step_l_xy)
            & (step_l_xy > jnp.float32(0.0))
        )
        t_valid = jnp.where(valid, t_all, jnp.float32(jnp.inf))

        # Nearest intersecting cylinder
        i_min   = jnp.argmin(t_valid)
        t_min   = t_valid[i_min]
        any_hit = jnp.isfinite(t_min)

        # Guard against t_min=inf causing NaN in the (discarded) hit branch
        t_safe  = jnp.where(any_hit, t_min, jnp.float32(0.0))

        c_hit = centers_2d[i_min]   # (2,)
        R_hit = radii_arr[i_min]    # scalar

        # Outward normal at the hit point (min-image frame)
        q_c   = r2 - c_hit
        q_c   = q_c - L * jnp.floor(q_c / L + jnp.float32(0.5))
        q_hit = q_c + t_safe * d_hat_xy
        n_out = q_hit / R_hit       # unit outward normal (away from cylinder axis)

        # Specular reflection of direction unit vector
        d_refl = d_hat_xy - 2.0 * jnp.dot(d_hat_xy, n_out) * n_out
        # Guard: when step_l_xy==0, d_hat_xy=zeros → d_refl=zeros → norm=0 → NaN.
        # In XLA/vmap, NaN in the false branch of jnp.where can contaminate the
        # selected branch through fused select lowering.  Replace with zeros
        # (safe: remaining==0 when step_l_xy==0, so r2_reflected uses this as
        # d_refl * 0 anyway).
        d_refl_norm = jnp.linalg.norm(d_refl)
        d_refl = jnp.where(
            d_refl_norm > jnp.float32(0.0),
            d_refl / jnp.maximum(d_refl_norm, jnp.float32(1e-30)),
            jnp.zeros(2, dtype=jnp.float32)
        )

        # Position after reflection: hit point nudged outward + remaining travel
        r2_hit    = r2 + t_safe * d_hat_xy
        r2_nudge  = r2_hit + NUDGE * n_out
        remaining = jnp.maximum(step_l_xy - t_safe - NUDGE, jnp.float32(0.0))

        r2_reflected = r2_nudge + d_refl * remaining
        r2_straight  = r2 + step_xy

        xy_final = jnp.where(any_hit, r2_reflected, r2_straight)



        # ── No periodic wrap here ─────────────────────────────────────────────
        # Positions are kept UNFOLDED (true lab-frame coordinates).  All
        # boundary detection uses minimum-image convention and is correct for
        # any unfolded position.  Wrapping the position here would cause the
        # phase integral γ·G·r(t) to use the wrapped coordinate, which
        # aliases displacements > L/2 and drastically underestimates the
        # effective b-value for walkers that cross the box boundary.

        # ── Safety clamp: project walker out if it ended up inside a cylinder ─
        # Fold xy_final into [-L/2, L/2) ONLY for the detection step.  Large
        # unfolded coordinates cause catastrophic cancellation in the raw
        # min-image arithmetic (q = xy_final - c - L*n), which can exceed NUDGE
        # precision.  Folding first keeps the numbers small and the computation
        # accurate.  The actual correction is then applied to the UNFOLDED
        # xy_final as a small additive offset.
        xy_folded = xy_final - L * jnp.floor(xy_final / L + jnp.float32(0.5))
        q_f  = xy_folded[None, :] - centers_2d                        # (N, 2) — bounded
        q_f  = q_f - L * jnp.floor(q_f / L + jnp.float32(0.5))       # min-image
        d2_f = jnp.sum(q_f ** 2, axis=1)                              # (N,)
        pen  = jnp.where(d2_f < radii_arr ** 2,
                         d2_f / (radii_arr ** 2),
                         jnp.float32(1.0))
        k_cl       = jnp.argmin(pen)
        inside_any = pen[k_cl] < jnp.float32(1.0)

        R_cl  = radii_arr[k_cl]
        q_cl  = q_f[k_cl]              # min-image displacement (accurate, small)
        d_cl  = jnp.linalg.norm(q_cl)
        safe_d_cl = jnp.maximum(d_cl, NUDGE)
        # Scalar jnp.where → correction is zero when not inside (no array-branch
        # select, which can mis-fire in vmap with large batches).
        clamp_scale = jnp.where(inside_any,
                                (R_cl + NUDGE) / safe_d_cl,
                                jnp.float32(1.0))
        xy_final = xy_final + q_cl * (clamp_scale - jnp.float32(1.0))

        # ── Reconstruct lab-frame position ───────────────────────────────────
        z_final   = r_c[2] + step_z
        r_c_new   = jnp.stack([xy_final[0], xy_final[1], z_final])
        if self._is_identity_rotation:
            return r_c_new
        return self._R_inv @ r_c_new

    def reflect_with_log_weight(self, r, step, rho_over_D):
        """Specular exterior reflection + surface-relaxation log-weight decrement.

        Identical to reflect() but also computes the perpendicular penetration
        depth at the cylinder wall and returns a magnetisation log-weight
        decrement:

            Δlog_w = -2 · ρ_over_D · d_perp

        where d_perp = (step_l_xy − t_entry) · cos(α) is the length the walker
        would have penetrated into the cylinder wall, and cos(α) = √disc / R at
        the hit point.  This is the same Brownstein-Tarr formula as for the
        interior Cylinder geometry, now applied to the extra-axonal side.  The
        analytical ground truth for a single cylinder of radius R in a periodic
        box of side L is:

            T2_surface = (L² − πR²) / (2πR · ρ₂)   [fast-diffusion limit]

        Parameters
        ----------
        r : (3,) float32, current walker position (lab frame)
        step : (3,) float32, proposed displacement (lab frame)
        rho_over_D : float32, ρ₂/D baked in by make_step_fn

        Returns
        -------
        r_new : (3,) float32, new position (lab frame)
        dlog_w : float32, log-weight decrement (≤ 0)
        """
        L          = self._L_jax
        centers_2d = self._centers_jax
        radii_arr  = self._radii_jax
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

        # ── Vectorised ray-circle entry test ─────────────────────────────────
        q_all    = r2[None, :] - centers_2d
        q_all    = q_all - L * jnp.floor(q_all / L + jnp.float32(0.5))
        dp_all   = jnp.sum(d_hat_xy[None, :] * q_all, axis=1)
        disc_all = dp_all ** 2 - (jnp.sum(q_all ** 2, axis=1) - radii_arr ** 2)
        disc_s   = jnp.maximum(disc_all, jnp.float32(0.0))
        t_all    = -dp_all - jnp.sqrt(disc_s)

        valid    = (
            (disc_all > jnp.float32(0.0))
            & (t_all  > EPS)
            & (t_all  < step_l_xy)
            & (step_l_xy > jnp.float32(0.0))
        )
        t_valid  = jnp.where(valid, t_all, jnp.float32(jnp.inf))

        i_min   = jnp.argmin(t_valid)
        t_min   = t_valid[i_min]
        any_hit = jnp.isfinite(t_min)
        t_safe  = jnp.where(any_hit, t_min, jnp.float32(0.0))

        c_hit = centers_2d[i_min]
        R_hit = radii_arr[i_min]

        q_c   = r2 - c_hit
        q_c   = q_c - L * jnp.floor(q_c / L + jnp.float32(0.5))
        q_hit = q_c + t_safe * d_hat_xy
        n_out = q_hit / R_hit

        d_refl     = d_hat_xy - 2.0 * jnp.dot(d_hat_xy, n_out) * n_out
        d_refl_norm = jnp.linalg.norm(d_refl)
        d_refl     = jnp.where(
            d_refl_norm > jnp.float32(0.0),
            d_refl / jnp.maximum(d_refl_norm, jnp.float32(1e-30)),
            jnp.zeros(2, dtype=jnp.float32)
        )
        r2_hit    = r2 + t_safe * d_hat_xy
        r2_nudge  = r2_hit + NUDGE * n_out
        remaining = step_l_xy - t_safe          # before NUDGE, for d_perp

        r2_reflected = r2_nudge + d_refl * jnp.maximum(remaining - NUDGE, jnp.float32(0.0))
        r2_straight  = r2 + step_xy
        xy_final     = jnp.where(any_hit, r2_reflected, r2_straight)

        # ── No periodic wrap (keep unfolded position for correct phase) ──────
        # See reflect() for rationale.

        # ── Safety clamp (fold-first for float32 stability) ──────────────────
        xy_folded = xy_final - L * jnp.floor(xy_final / L + jnp.float32(0.5))
        q_f  = xy_folded[None, :] - centers_2d
        q_f  = q_f - L * jnp.floor(q_f / L + jnp.float32(0.5))
        d2_f = jnp.sum(q_f ** 2, axis=1)
        pen  = jnp.where(d2_f < radii_arr ** 2,
                         d2_f / (radii_arr ** 2), jnp.float32(1.0))
        k_cl       = jnp.argmin(pen)
        inside_any = pen[k_cl] < jnp.float32(1.0)
        R_cl  = radii_arr[k_cl]
        q_cl  = q_f[k_cl]
        d_cl  = jnp.linalg.norm(q_cl)
        safe_d_cl = jnp.maximum(d_cl, NUDGE)
        clamp_scale = jnp.where(inside_any,
                                (R_cl + NUDGE) / safe_d_cl,
                                jnp.float32(1.0))
        xy_final = xy_final + q_cl * (clamp_scale - jnp.float32(1.0))

        # ── Reconstruct lab-frame position ───────────────────────────────────
        z_final   = r_c[2] + step_z
        r_c_new   = jnp.stack([xy_final[0], xy_final[1], z_final])
        if self._is_identity_rotation:
            r_new = r_c_new
        else:
            r_new = self._R_inv @ r_c_new

        # ── Surface relaxation: d_perp = (step_l - t_entry) · cos(α) ─────────
        # cos(α) = √disc / R at the entry point (same formula as interior Cylinder)
        disc_hit  = disc_all[i_min]
        cos_alpha = jnp.sqrt(jnp.maximum(disc_hit, jnp.float32(0.0))) / R_hit
        d_perp    = jnp.where(any_hit, remaining * cos_alpha, jnp.float32(0.0))
        dlog_w    = -2.0 * jnp.float32(rho_over_D) * d_perp
        return r_new, dlog_w

    def permeate(self, r, step, kappa_over_D, rho_over_D, perm_key):
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
        dp_all   = jnp.sum(d_hat_xy[None, :] * q_all, axis=1)     # (N,)
        disc_all = dp_all ** 2 - (dist2_all - radii_arr ** 2)      # (N,)
        disc_s   = jnp.maximum(disc_all, jnp.float32(0.0))

        t_exit_all  = -dp_all + jnp.sqrt(disc_s)   # (N,) exit roots
        t_entry_all = -dp_all - jnp.sqrt(disc_s)   # (N,) entry roots
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
        p_transmit = jnp.minimum(jnp.float32(1.0),
                                 jnp.float32(2.0) * kappa_over_D * d_perp)
        u        = jax.random.uniform(perm_key, dtype=jnp.float32)
        transmit = any_hit & (u < p_transmit)

        # ── Reflected: specular, nudge to same side ───────────────────────────
        d_refl      = d_hat_xy - jnp.float32(2.0) * jnp.dot(d_hat_xy, n_out) * n_out
        d_refl_norm = jnp.linalg.norm(d_refl)
        d_refl      = jnp.where(
            d_refl_norm > jnp.float32(0.0),
            d_refl / jnp.maximum(d_refl_norm, jnp.float32(1e-30)),
            jnp.zeros(2, dtype=jnp.float32)
        )
        refl_nudge = jnp.where(hit_is_inside, -n_out, n_out)   # same side
        # Work in LOCAL frame (relative to c_hit) using the snapped q_hit.
        # This ensures |q_nudge| = R_hit ± NUDGE (definitively inside/outside).
        q_nudge = q_hit + NUDGE * refl_nudge
        q_refl  = q_nudge + d_refl * jnp.maximum(remaining - NUDGE,
                                                   jnp.float32(0.0))

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
        z_final   = r_c[2] + step_z
        r_c_new   = jnp.stack([xy_final[0], xy_final[1], z_final])
        if self._is_identity_rotation:
            return r_c_new, dlog_w
        return self._R_inv @ r_c_new, dlog_w

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


# ---------------------------------------------------------------------------
# PackedSpheres — extra-axonal diffusion in a periodic 3-D cubic domain
# ---------------------------------------------------------------------------

def pack_spheres(radii, target_vf=None, L=None, seed=0, max_attempts=100_000):
    """Pack N spheres in a periodic 3-D cubic domain using RSA.

    Random Sequential Addition (RSA) places spheres one by one, rejecting
    positions that would cause overlap with any previously placed sphere
    (including periodic images).  Large spheres are placed first.  RSA
    typically reaches VF ≈ 0.38 for monodisperse spheres.

    Parameters
    ----------
    radii : array-like, shape (N,)
        Sphere radii in metres.  All must be positive.
    target_vf : float, optional
        Target intra-sphere volume fraction Σ(4/3)πrᵢ³/L³.  Used to derive L.
        Mutually exclusive with ``L``.
    L : float, optional
        Side-length of the periodic cubic domain in metres.
        Mutually exclusive with ``target_vf``.
    seed : int
        NumPy RNG seed for reproducible packing.
    max_attempts : int
        Maximum random placement trials per sphere before raising RuntimeError.

    Returns
    -------
    centers : np.ndarray, shape (N, 3)
        Sphere centre positions in metres.
    L : float
        Side-length actually used.
    achieved_vf : float
        Achieved volume fraction = Σ(4/3)πrᵢ³ / L³.

    Raises
    ------
    ValueError
        On invalid inputs.
    RuntimeError
        If RSA cannot place a sphere within ``max_attempts`` trials.
    """
    radii = np.asarray(radii, dtype=np.float64).ravel()
    if len(radii) == 0:
        raise ValueError("radii must contain at least one element.")
    if np.any(radii <= 0):
        raise ValueError("All radii must be positive.")
    if (target_vf is None) == (L is None):
        raise ValueError("Provide exactly one of target_vf or L.")

    if target_vf is not None:
        if not 0.0 < float(target_vf) < 1.0:
            raise ValueError(f"target_vf must be in (0, 1), got {target_vf}.")
        L = float(((4.0 / 3.0) * np.pi * np.sum(radii ** 3) / float(target_vf))
                  ** (1.0 / 3.0))
    else:
        L = float(L)

    if np.any(radii > L / 2.0):
        raise ValueError(
            f"Largest radius ({np.max(radii) * 1e6:.2f} µm) exceeds L/2 "
            f"({L / 2.0 * 1e6:.2f} µm).  Reduce target_vf or supply a larger L.")

    rng = np.random.default_rng(int(seed))
    order = np.argsort(radii)[::-1]   # largest first
    radii_s   = radii[order]
    centers_s = np.zeros((len(radii), 3))

    for i, r_new in enumerate(radii_s):
        placed = False
        for _ in range(max_attempts):
            c_new = rng.uniform(-L / 2.0, L / 2.0, 3)
            ok = True
            for j in range(i):
                dq = c_new - centers_s[j]
                dq -= L * np.round(dq / L)   # minimum-image
                if np.linalg.norm(dq) < r_new + radii_s[j]:
                    ok = False
                    break
            if ok:
                centers_s[i] = c_new
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"RSA failed after {max_attempts} attempts placing sphere {i} "
                f"(r = {r_new * 1e6:.2f} µm).  "
                f"The target packing fraction may exceed what RSA can achieve "
                f"(monodisperse RSA limit ≈ 0.38); try reducing target_vf or "
                f"increasing max_attempts.")

    centers_out = np.empty_like(centers_s)
    centers_out[order] = centers_s
    achieved_vf = float((4.0 / 3.0) * np.pi * np.sum(radii ** 3) / L ** 3)
    return centers_out, L, achieved_vf


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
        """Specular exterior reflection off the nearest sphere + periodic wrap.

        Finds the nearest intersecting sphere via vectorised ray-sphere
        intersection (entry root: walker coming from outside).  One reflection
        per timestep (valid when step_length ≪ min_gap).  Positions are kept
        unfolded; minimum-image convention is used for boundary detection.
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

        # ── Vectorised ray-sphere entry test (N spheres) ───────────────────────
        # Minimum-image displacement from each sphere centre to walker
        q_all = r[None, :] - centers                                 # (N, 3)
        q_all = q_all - L * jnp.floor(q_all / L + jnp.float32(0.5))

        dp_all   = jnp.sum(d_hat[None, :] * q_all, axis=1)          # (N,)
        disc_all = dp_all ** 2 - (jnp.sum(q_all ** 2, axis=1)
                                  - radii_arr ** 2)                  # (N,)
        disc_s   = jnp.maximum(disc_all, jnp.float32(0.0))
        t_all    = -dp_all - jnp.sqrt(disc_s)                       # entry root

        valid = (
            (disc_all > jnp.float32(0.0))
            & (t_all  > EPS)
            & (t_all  < step_l)
            & (step_l > jnp.float32(0.0))
        )
        t_valid = jnp.where(valid, t_all, jnp.float32(jnp.inf))

        i_min   = jnp.argmin(t_valid)
        t_min   = t_valid[i_min]
        any_hit = jnp.isfinite(t_min)
        t_safe  = jnp.where(any_hit, t_min, jnp.float32(0.0))

        c_hit = centers[i_min]     # (3,)
        R_hit = radii_arr[i_min]   # scalar

        # Outward normal at hit point (min-image frame)
        q_c   = r - c_hit
        q_c   = q_c - L * jnp.floor(q_c / L + jnp.float32(0.5))
        q_hit = q_c + t_safe * d_hat
        n_out = q_hit / R_hit      # unit outward normal

        # Specular reflection of direction
        d_refl      = d_hat - jnp.float32(2.0) * jnp.dot(d_hat, n_out) * n_out
        d_refl_norm = jnp.linalg.norm(d_refl)
        d_refl      = jnp.where(
            d_refl_norm > jnp.float32(0.0),
            d_refl / jnp.maximum(d_refl_norm, jnp.float32(1e-30)),
            jnp.zeros(3, dtype=jnp.float32)
        )

        # Position after reflection: nudge outward + remaining travel
        r_hit     = r + t_safe * d_hat
        r_nudge   = r_hit + NUDGE * n_out
        remaining = jnp.maximum(step_l - t_safe - NUDGE, jnp.float32(0.0))

        r_reflected = r_nudge + d_refl * remaining
        r_straight  = r + step

        r_out = jnp.where(any_hit, r_reflected, r_straight)

        # ── No periodic wrap (keep unfolded for correct phase) ─────────────────
        # ── Safety clamp: push walker out if it ended up inside any sphere ─────
        r_folded = r_out - L * jnp.floor(r_out / L + jnp.float32(0.5))
        q_f      = r_folded[None, :] - centers                          # (N, 3)
        q_f      = q_f - L * jnp.floor(q_f / L + jnp.float32(0.5))
        d3_f     = jnp.sum(q_f ** 2, axis=1)                            # (N,)
        pen      = jnp.where(d3_f < radii_arr ** 2,
                             d3_f / (radii_arr ** 2),
                             jnp.float32(1.0))
        k_cl       = jnp.argmin(pen)
        inside_any = pen[k_cl] < jnp.float32(1.0)
        R_cl      = radii_arr[k_cl]
        q_cl      = q_f[k_cl]
        d_cl      = jnp.linalg.norm(q_cl)
        safe_d_cl = jnp.maximum(d_cl, NUDGE)
        clamp_scale = jnp.where(inside_any,
                                (R_cl + NUDGE) / safe_d_cl,
                                jnp.float32(1.0))
        r_out = r_out + q_cl * (clamp_scale - jnp.float32(1.0))

        return r_out

    def reflect_with_log_weight(self, r, step, rho_over_D):
        """Specular exterior reflection + surface-relaxation log-weight.

        Same as reflect() but also accumulates the Brownstein-Tarr weight:

            Δlog_w = -2 · rho_over_D · d_perp

        where d_perp = (step_l - t_entry) · cos(α)  and  cos(α) = √disc / R.

        T2_surface = V_extra / (κ · S_total)  [exact, fast-diffusion limit].
        """
        L         = self._L_jax
        centers   = self._centers_jax
        radii_arr = self._radii_jax
        EPS       = self._eps_detect
        NUDGE     = self._nudge

        step_l = jnp.linalg.norm(step)
        d_hat  = jnp.where(
            step_l > jnp.float32(0.0),
            step / step_l,
            jnp.zeros(3, dtype=jnp.float32))

        # ── Vectorised ray-sphere entry test ───────────────────────────────────
        q_all    = r[None, :] - centers
        q_all    = q_all - L * jnp.floor(q_all / L + jnp.float32(0.5))
        dp_all   = jnp.sum(d_hat[None, :] * q_all, axis=1)
        disc_all = dp_all ** 2 - (jnp.sum(q_all ** 2, axis=1) - radii_arr ** 2)
        disc_s   = jnp.maximum(disc_all, jnp.float32(0.0))
        t_all    = -dp_all - jnp.sqrt(disc_s)

        valid    = (
            (disc_all > jnp.float32(0.0))
            & (t_all  > EPS)
            & (t_all  < step_l)
            & (step_l > jnp.float32(0.0))
        )
        t_valid  = jnp.where(valid, t_all, jnp.float32(jnp.inf))

        i_min   = jnp.argmin(t_valid)
        t_min   = t_valid[i_min]
        any_hit = jnp.isfinite(t_min)
        t_safe  = jnp.where(any_hit, t_min, jnp.float32(0.0))

        c_hit = centers[i_min]
        R_hit = radii_arr[i_min]

        q_c   = r - c_hit
        q_c   = q_c - L * jnp.floor(q_c / L + jnp.float32(0.5))
        q_hit = q_c + t_safe * d_hat
        n_out = q_hit / R_hit

        d_refl      = d_hat - jnp.float32(2.0) * jnp.dot(d_hat, n_out) * n_out
        d_refl_norm = jnp.linalg.norm(d_refl)
        d_refl      = jnp.where(
            d_refl_norm > jnp.float32(0.0),
            d_refl / jnp.maximum(d_refl_norm, jnp.float32(1e-30)),
            jnp.zeros(3, dtype=jnp.float32)
        )

        r_hit     = r + t_safe * d_hat
        r_nudge   = r_hit + NUDGE * n_out
        remaining = step_l - t_safe        # before NUDGE, for d_perp

        r_reflected = r_nudge + d_refl * jnp.maximum(remaining - NUDGE,
                                                       jnp.float32(0.0))
        r_straight  = r + step
        r_out       = jnp.where(any_hit, r_reflected, r_straight)

        # ── Safety clamp ───────────────────────────────────────────────────────
        r_folded = r_out - L * jnp.floor(r_out / L + jnp.float32(0.5))
        q_f      = r_folded[None, :] - centers
        q_f      = q_f - L * jnp.floor(q_f / L + jnp.float32(0.5))
        d3_f     = jnp.sum(q_f ** 2, axis=1)
        pen      = jnp.where(d3_f < radii_arr ** 2,
                             d3_f / (radii_arr ** 2), jnp.float32(1.0))
        k_cl       = jnp.argmin(pen)
        inside_any = pen[k_cl] < jnp.float32(1.0)
        R_cl      = radii_arr[k_cl]
        q_cl      = q_f[k_cl]
        d_cl      = jnp.linalg.norm(q_cl)
        safe_d_cl = jnp.maximum(d_cl, NUDGE)
        clamp_scale = jnp.where(inside_any,
                                (R_cl + NUDGE) / safe_d_cl,
                                jnp.float32(1.0))
        r_out = r_out + q_cl * (clamp_scale - jnp.float32(1.0))

        # ── d_perp = remaining · cos(α),  cos(α) = √disc / R ─────────────────
        disc_hit  = disc_all[i_min]
        cos_alpha = jnp.sqrt(jnp.maximum(disc_hit, jnp.float32(0.0))) / R_hit
        d_perp    = jnp.where(any_hit, remaining * cos_alpha, jnp.float32(0.0))
        dlog_w    = -jnp.float32(2.0) * jnp.float32(rho_over_D) * d_perp
        return r_out, dlog_w

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
        dp_all   = jnp.sum(d_hat[None, :] * q_all, axis=1)         # (N,)
        disc_all = dp_all ** 2 - (dist2_all - radii_arr ** 2)       # (N,)
        disc_s   = jnp.maximum(disc_all, jnp.float32(0.0))

        t_exit_all  = -dp_all + jnp.sqrt(disc_s)    # exit root (inside walkers)
        t_entry_all = -dp_all - jnp.sqrt(disc_s)    # entry root (outside walkers)
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
        p_transmit = jnp.minimum(jnp.float32(1.0),
                                 jnp.float32(2.0) * kappa_over_D * d_perp)
        u        = jax.random.uniform(perm_key, dtype=jnp.float32)
        transmit = any_hit & (u < p_transmit)

        # ── Reflected: specular, nudge to same side ────────────────────────────
        d_refl      = d_hat - jnp.float32(2.0) * jnp.dot(d_hat, n_out) * n_out
        d_refl_norm = jnp.linalg.norm(d_refl)
        d_refl      = jnp.where(
            d_refl_norm > jnp.float32(0.0),
            d_refl / jnp.maximum(d_refl_norm, jnp.float32(1e-30)),
            jnp.zeros(3, dtype=jnp.float32)
        )
        refl_nudge = jnp.where(hit_is_inside, -n_out, n_out)
        q_nudge    = q_hit + NUDGE * refl_nudge
        q_refl     = q_nudge + d_refl * jnp.maximum(remaining - NUDGE,
                                                      jnp.float32(0.0))

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
def pack_myelinated_cylinders(inner_radii, g_ratios, target_packing,
                               cell_size=None, seed=0, max_attempts=100_000):
    """Place myelinated cylinders in a periodic square RVE using RSA.

    Each cylinder has an inner (axon) radius and an outer (myelin) radius
    given by outer_radius = inner_radius / g_ratio.  Placement ensures that
    outer boundaries do not overlap each other (including periodic images).

    Parameters
    ----------
    inner_radii : array-like, shape (N,)
        Axon radii in metres.
    g_ratios : array-like, shape (N,) or scalar
        g-ratio per cylinder (outer = inner / g_ratio).  Scalar is broadcast.
    target_packing : float
        Target packing fraction = sum(pi*R_outer^2) / cell_size^2.  Used to
        derive cell_size when ``cell_size`` is None.
    cell_size : float, optional
        Side-length of the periodic square cell in metres.  If provided,
        ``target_packing`` is ignored.
    seed : int
        NumPy RNG seed.
    max_attempts : int
        RSA placement attempts per cylinder.

    Returns
    -------
    inner_radii : np.ndarray, shape (N,)
    g_ratios    : np.ndarray, shape (N,)
    centers     : np.ndarray, shape (N, 2)
        Cylinder centers in metres, ``[-L/2, L/2)`` convention.
    """
    inner_radii = np.asarray(inner_radii, dtype=np.float64).ravel()
    N = len(inner_radii)
    g_ratios_arr = np.broadcast_to(
        np.asarray(g_ratios, dtype=np.float64).ravel(), (N,)).copy()
    outer_radii = inner_radii / g_ratios_arr

    if cell_size is None:
        if target_packing is None:
            raise ValueError("Provide either cell_size or target_packing.")
        cell_size = float(np.sqrt(np.pi * np.sum(outer_radii ** 2) / target_packing))
    L = float(cell_size)

    rng = np.random.default_rng(int(seed))
    order = np.argsort(outer_radii)[::-1]   # place largest first
    outer_s = outer_radii[order]
    centers_s = np.zeros((N, 2))

    for i, r_out in enumerate(outer_s):
        placed = False
        for _ in range(max_attempts):
            c = rng.uniform(-L / 2.0, L / 2.0, 2)
            ok = True
            for j in range(i):
                dq = c - centers_s[j]
                dq -= L * np.round(dq / L)
                if np.linalg.norm(dq) < r_out + outer_s[j]:
                    ok = False
                    break
            if ok:
                centers_s[i] = c
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"RSA failed after {max_attempts} attempts placing cylinder {i}.")

    centers_out = np.empty_like(centers_s)
    centers_out[order] = centers_s
    return inner_radii, g_ratios_arr, centers_out


class PackedMyelinatedCylinders:
    """Periodic RVE with N_actual myelinated cylinders — three-compartment.

    Combines ``PackedCylinders`` (periodic, multi-cylinder substrate) with
    ``MyelinatedCylinder`` (myelin sheath, per-compartment D/T2/permeability)
    into a single JIT-stable geometry.

    Each axon k (k=0..N_actual-1) has:
      - Inner radius  R_inner_k  (axon boundary)
      - Outer radius  R_outer_k = R_inner_k / g_ratio_k  (myelin/extra boundary)
      - Center        (cx_k, cy_k) in the periodic cell [-L/2, L/2)

    Compartment numbering
    ~~~~~~~~~~~~~~~~~~~~~
    - 0              : extra-axonal
    - 1 .. N_max     : intra_k  (axon k, k+1-th slot)
    - N_max+1 .. 2*N_max : myelin_k  (myelin sheath of axon k, k+1-th slot)

    Zero-radius padding
    ~~~~~~~~~~~~~~~~~~~
    Arrays are padded to length ``N_max`` with zeros.  Dummy cylinders (r=0)
    receive zero walkers (area = 0) and have SDF = inf in the step function
    (they never win ``argmin``), so different N_actual with the same N_max
    compile to the same JAX program.

    Parameters
    ----------
    inner_radii : array-like, shape (N_actual,)
        Axon radii in metres.
    g_ratios : array-like, shape (N_actual,) or scalar
        g-ratio per cylinder.
    centers : np.ndarray, shape (N_actual, 2)
        Cylinder centre positions in metres, [-L/2, L/2) convention.
    cell_size : float
        Side-length of the periodic square cell in metres.
    N_max : int, optional
        Fixed JIT padding length (>= N_actual).  Default 128.
    orientation : array-like, shape (3,), optional
        Shared cylinder axis direction. Default [0, 0, 1].
    D_intra, D_myelin, D_extra : float or array-like (N_actual,)
        Diffusivities in m^2/s.  Scalar is broadcast to all cylinders.  ``D_myelin``
        defaults to 0 (stuck myelin water; set > 0 to let it diffuse).
    T2_intra, T2_myelin, T2_extra : float or array-like (N_actual,) or None
        T2 relaxation times in seconds.  None = no T2.
    kappa_inner, kappa_outer : float or array-like (N_actual,) or None
        Inner/outer wall permeabilities in m/s.  None / 0.0 = impermeable.
    rho_inner, rho_outer : float or array-like (N_actual,) or None
        Surface relaxivity at inner/outer myelin walls in m/s.  0.0 = no
        relaxivity.
    """

    _is_packed_myelinated = True

    def __init__(
        self,
        inner_radii,
        g_ratios,
        centers,
        cell_size,
        N_max=128,
        orientation=(0., 0., 1.),
        D_intra=2e-9,
        D_myelin=0.0,
        D_extra=2e-9,
        T2_intra=None,
        T2_myelin=None,
        T2_extra=None,
        kappa_inner=0.0,
        kappa_outer=0.0,
        rho_inner=0.0,
        rho_outer=0.0,
    ):
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

        outer_radii = inner_radii / g_ratios
        self.N_actual = N_actual
        self.N_max = N_max
        self._cell_size = float(cell_size)
        from .substrate.biophysical_constants import get_default_value as _gdv
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
            return _bcast(x, name)

        def _bcast_opt(x, name):
            if x is None:
                return None
            return _bcast(x, name)

        D_intra_arr   = _bcast(D_intra,  'D_intra')
        D_myelin_arr  = _bcast(D_myelin, 'D_myelin')
        D_extra_arr   = _bcast(D_extra,  'D_extra')
        kappa_inner_arr = _bcast_kappa(kappa_inner, 'kappa_inner')
        kappa_outer_arr = _bcast_kappa(kappa_outer, 'kappa_outer')
        rho_inner_arr   = _bcast_kappa(rho_inner,   'rho_inner')
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
        self.orientation = (orientation / np.linalg.norm(orientation)).astype(
            np.float32)
        _R_np = _rotation_to_z(self.orientation)
        self._R     = jnp.array(_R_np, dtype=jnp.float32)
        self._R_inv = jnp.array(_R_np.T, dtype=jnp.float32)
        self._is_identity_rotation = bool(np.allclose(_R_np, np.eye(3)))

        # EPS/NUDGE — scale by smallest non-zero inner radius
        nonzero = inner_p[inner_p > 0]
        ref_r = float(np.min(nonzero)) if len(nonzero) > 0 else 1e-6
        self._eps   = jnp.float32(1e-7 * ref_r)
        self._nudge = jnp.float32(1e-4 * ref_r)

        # Minimum gap (diagnostic, uses outer radii)
        self.min_gap = self._compute_min_gap()

    def _compute_min_gap(self):
        """Minimum clear gap between outer boundaries (actual cylinders only)."""
        N = self.N_actual
        L = self._L_float
        centers = self._centers_np[:N]
        outer   = self._outer_radii_np[:N]
        min_gap = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                dq = centers[i] - centers[j]
                dq -= L * np.round(dq / L)
                gap = np.linalg.norm(dq) - outer[i] - outer[j]
                min_gap = min(min_gap, gap)
            min_gap = min(min_gap, L - 2.0 * outer[i])
        return float(min_gap) if np.isfinite(min_gap) else float('inf')

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

        Dummy cylinders (R=0) automatically receive zero walkers.
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

        total_area = area_extra + np.sum(area_intra) + np.sum(area_myelin)

        n_intra  = np.array([int(round(n_walkers * a / total_area))
                             for a in area_intra], dtype=int)
        n_myelin = np.array([int(round(n_walkers * a / total_area))
                             for a in area_myelin], dtype=int)

        # Extra fills remainder (handles rounding)
        n_extra = n_walkers - int(np.sum(n_intra)) - int(np.sum(n_myelin))
        if n_extra < 0:
            excess = -n_extra
            for k in range(N):
                trim = min(excess, n_intra[k])
                n_intra[k] -= trim
                excess -= trim
                if excess == 0:
                    break
            n_extra = 0

        rng = np.random.default_rng(
            int(jax.random.randint(key, (), 0, 2 ** 30)))

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
            accepted = []
            n_have = 0
            while n_have < n_extra_actual:
                batch = max(n_extra_actual * 4, 1024)
                xy = rng.uniform(-L / 2.0, L / 2.0, (batch, 2))
                outside = np.ones(batch, dtype=bool)
                for k in range(N):
                    dxy = xy - centers[k]
                    dxy -= L * np.round(dxy / L)   # min-image
                    outside &= np.sum(dxy ** 2, axis=1) > outer[k] ** 2
                accepted.append(xy[outside])
                n_have = sum(len(a) for a in accepted)
            xy_ex = np.concatenate(accepted)[:n_extra_actual].astype(np.float32)
            positions[idx:idx + n_extra_actual, 0] = xy_ex[:, 0]
            positions[idx:idx + n_extra_actual, 1] = xy_ex[:, 1]
            compartments[idx:idx + n_extra_actual] = 0

        # Rotate to lab frame (positions in cylinder-frame xy-plane, z=0)
        R_inv = np.array(self._R_inv)
        r_lab = (R_inv @ positions.T).T

        self._init_compartments = jnp.array(compartments, dtype=jnp.int32)
        return jnp.array(r_lab, dtype=jnp.float32)

    def reflect(self, r, step):
        """Fallback reflect — not used; custom step function handles this."""
        return r + step


class PermeableSlab1D(Geometry):
    """Closed 1-D two-compartment slab: a permeable membrane at x=L/2 with reflecting
    outer walls at x=0 and x=L (y, z free).  The cleanest first-principles benchmark for
    membrane permeability -- no curvature and no exterior re-entry (a closed reservoir):

        compartment A = {x < L/2}, B = {x > L/2}; start all walkers in A ->
        f_A(t) = 1/2 + 1/2 exp(-4 kappa t / L)   (closed two-compartment exchange).

    The membrane transmission is the SAME rule as the curved geometries
    (p = min(1, 2 kappa/D * d_perp)), so this isolates the planar prefactor from curvature.

    Parameters
    ----------
    length : float
        Slab length L (m); each compartment has width L/2.
    permeability : float
        Membrane permeability kappa (m/s) at x=L/2.
    surface_relaxivity_t2 : float, optional
        Surface relaxivity applied on membrane reflection (m/s).
    """

    def __init__(self, length, permeability, surface_relaxivity_t2=None):
        self.length = float(length)
        self.permeability = float(permeability)
        self.surface_relaxivity_t2 = (float(surface_relaxivity_t2)
                                      if surface_relaxivity_t2 is not None else None)
        # length scale for permeable_sub_steps (resolve the membrane crossing)
        self.radius = float(length) / 2.0

    def volume(self) -> float:
        return self.length / 2.0          # per compartment (V/S -> tau = L/(2 kappa) one-sided)

    def surface_area(self) -> float:
        return 1.0                         # unit membrane area

    def init_positions(self, n_walkers, key):
        x = jax.random.uniform(key, (n_walkers,), dtype=jnp.float32,
                               minval=0.0, maxval=jnp.float32(self.length / 2.0))  # start in A
        z = jnp.zeros((n_walkers,), dtype=jnp.float32)
        return jnp.stack([x, z, z], axis=1)

    def classify_position(self, r):
        return jnp.int32(jnp.where(r[0] < jnp.float32(self.length / 2.0), 0, 1))

    def _fold(self, x):
        """Reflect x into [0, L] at the outer walls (modular mirror)."""
        L = jnp.float32(self.length)
        xf = jnp.mod(x, 2.0 * L)
        return jnp.where(xf > L, 2.0 * L - xf, xf)

    def reflect(self, r, step):
        # fully-reflecting fallback: bounce at the membrane and fold at outer walls
        L = jnp.float32(self.length); xm = jnp.float32(self.length / 2.0)
        x = r[0]; x_new = x + step[0]
        crossed = (x - xm) * (x_new - xm) < 0.0
        x1 = jnp.where(crossed, 2.0 * xm - x_new, x_new)
        return jnp.array([self._fold(x1), r[1] + step[1], r[2] + step[2]])

    def reflect_with_log_weight(self, r, step, rho_over_D):
        L = jnp.float32(self.length); xm = jnp.float32(self.length / 2.0)
        x = r[0]; x_new = x + step[0]
        crossed = (x - xm) * (x_new - xm) < 0.0
        d_perp = jnp.where(crossed, jnp.abs(x_new - xm), jnp.float32(0.0))
        x1 = jnp.where(crossed, 2.0 * xm - x_new, x_new)
        r_out = jnp.array([self._fold(x1), r[1] + step[1], r[2] + step[2]])
        return r_out, -2.0 * rho_over_D * d_perp

    def permeate(self, r, step, kappa_over_D, rho_over_D, perm_key):
        L = jnp.float32(self.length); xm = jnp.float32(self.length / 2.0)
        x = r[0]; x_new = x + step[0]
        crossed = (x - xm) * (x_new - xm) < 0.0
        d_perp = jnp.where(crossed, jnp.abs(x_new - xm), jnp.float32(0.0))
        p = jnp.minimum(jnp.float32(1.0), jnp.float32(2.0) * kappa_over_D * d_perp)
        u = jax.random.uniform(perm_key, dtype=jnp.float32)
        transmit = crossed & (u < p)
        x1 = jnp.where(crossed & ~transmit, 2.0 * xm - x_new, x_new)   # reflect at membrane
        r_out = jnp.array([self._fold(x1), r[1] + step[1], r[2] + step[2]])
        dlog_w = jnp.where(crossed & ~transmit,
                           -jnp.float32(2.0) * rho_over_D * d_perp, jnp.float32(0.0))
        return r_out, dlog_w


class PermeableShell(Geometry):
    """Closed radial two-compartment shell for first-principles permeability validation in
    2D/3D: a PERMEABLE membrane at r=R_in inside a REFLECTING outer wall at r=R_out.

    ``kind='sphere'`` -> r = |x| (3D);  ``kind='cylinder'`` -> r = |x_perp| to ``orientation``
    (2D radial, free along the axis).  Compartment A = {r < R_in}, B = {R_in < r < R_out}.
    Closed (no exterior re-entry) and finite-diffusion-exact: the exchange time is the lowest
    (spherical-)Bessel eigenvalue, the clean analog of ``PermeableSlab1D`` for curved membranes.
    """

    def __init__(self, r_inner, r_outer, permeability, kind='sphere',
                 orientation=(0.0, 0.0, 1.0), surface_relaxivity_t2=None):
        assert kind in ('sphere', 'cylinder')
        self.r_inner = float(r_inner); self.r_outer = float(r_outer)
        self.permeability = float(permeability)
        self.kind = kind
        self.surface_relaxivity_t2 = (float(surface_relaxivity_t2)
                                      if surface_relaxivity_t2 is not None else None)
        self.radius = float(r_inner)          # sub-step length scale
        o = np.asarray(orientation, dtype=np.float64); self._o = o / np.linalg.norm(o)
        self._axis = jnp.array(self._o, dtype=jnp.float32)

    def _radial(self, r):
        """Radial vector used for the membranes (full r for sphere, perpendicular for cyl)."""
        if self.kind == 'sphere':
            return r
        return r - jnp.dot(r, self._axis) * self._axis     # component perpendicular to axis

    def volume(self):
        if self.kind == 'sphere':
            return (4.0 / 3.0) * np.pi * self.r_inner ** 3          # compartment A
        return np.pi * self.r_inner ** 2

    def surface_area(self):
        if self.kind == 'sphere':
            return 4.0 * np.pi * self.r_inner ** 2
        return 2.0 * np.pi * self.r_inner                          # per unit length

    def init_positions(self, n_walkers, key):
        # uniform inside compartment A (r < R_in)
        k1, k2 = jax.random.split(key)
        v = jax.random.normal(k1, (n_walkers, 3), dtype=jnp.float32)
        if self.kind == 'cylinder':
            v = v - (v @ self._axis)[:, None] * self._axis[None, :]
        vhat = v / jnp.linalg.norm(v, axis=1, keepdims=True)
        dim = 3.0 if self.kind == 'sphere' else 2.0
        u = jax.random.uniform(key, (n_walkers,), dtype=jnp.float32) ** (1.0 / dim)
        pos = vhat * (u * jnp.float32(self.r_inner))[:, None]
        if self.kind == 'cylinder':
            zc = jax.random.uniform(k2, (n_walkers,), dtype=jnp.float32) * jnp.float32(self.r_inner)
            pos = pos + zc[:, None] * self._axis[None, :]
        return pos

    def classify_position(self, r):
        rad = jnp.linalg.norm(self._radial(r))
        return jnp.int32(jnp.where(rad < jnp.float32(self.r_inner), 0, 1))

    def _permeate_impl(self, r, step, kappa_over_D, rho_over_D, perm_key):
        Rin = jnp.float32(self.r_inner); Rout = jnp.float32(self.r_outer)
        EPS = jnp.float32(1e-7 * self.r_inner); BIG = jnp.float32(1e30)
        step_l = jnp.linalg.norm(step)
        d = step / jnp.maximum(step_l, EPS)
        # work in the radial subspace (sphere: full; cylinder: perpendicular)
        rr = self._radial(r); dd = self._radial(d)
        b = jnp.dot(dd, rr); r2 = jnp.dot(rr, rr)
        dd2 = jnp.maximum(jnp.dot(dd, dd), EPS)      # |perp(d)|^2: 1 for sphere, sin^2(theta) for cylinder

        def first_t(Rk):
            # perpendicular trajectory rr + t*dd hits radius Rk: dd2*t^2 + 2b*t + (r2-Rk^2)=0
            c = r2 - Rk * Rk
            disc = b * b - dd2 * c
            s = jnp.sqrt(jnp.maximum(disc, 0.0))
            t1 = (-b - s) / dd2; t2 = (-b + s) / dd2
            t1 = jnp.where((disc > 0) & (t1 > EPS) & (t1 < step_l), t1, BIG)
            t2 = jnp.where((disc > 0) & (t2 > EPS) & (t2 < step_l), t2, BIG)
            return jnp.minimum(t1, t2)

        t_in = first_t(Rin); t_out = first_t(Rout)
        hit_in = t_in < t_out
        t_hit = jnp.minimum(t_in, t_out)
        any_hit = t_hit < jnp.float32(1e29)

        r_hit = r + t_hit * d
        rad_hit = self._radial(r_hit)
        n = rad_hit / jnp.maximum(jnp.linalg.norm(rad_hit), EPS)   # outward radial normal
        remaining = step_l - t_hit
        cos_a = jnp.abs(jnp.dot(d, n))
        d_perp_tangent = remaining * cos_a
        # radial (normal-coordinate) penetration of the endpoint past the curved membrane:
        rad_end = self._radial(r + step)
        Rk = jnp.where(hit_in, Rin, Rout)
        d_perp_radial = jnp.abs(jnp.linalg.norm(rad_end) - Rk)
        d_perp = jnp.where(getattr(self, '_dperp_mode', 'tangent') == 'radial',
                           d_perp_radial, d_perp_tangent)

        p = jnp.minimum(jnp.float32(1.0), jnp.float32(2.0) * kappa_over_D * d_perp)
        u = jax.random.uniform(perm_key, dtype=jnp.float32)
        transmit = hit_in & any_hit & (u < p)                     # R_out never transmits
        reflect_here = any_hit & (~transmit)

        d_refl = d - jnp.float32(2.0) * jnp.dot(d, n) * n
        # nudge the reflected walker OFF the membrane onto its own side, so it never
        # straddles the surface (straddling biases the next step's crossing -> breaks
        # detailed balance).  Side: walkers with radius < the hit radius stay inside it.
        NUDGE = jnp.float32(1e-4 * self.r_inner)
        r_rad_mag = jnp.linalg.norm(self._radial(r))
        Rk = jnp.where(hit_in, Rin, Rout)
        side = jnp.where(r_rad_mag < Rk, jnp.float32(-1.0), jnp.float32(1.0))  # -1 inside Rk
        r_hit_nudged = r_hit + side * NUDGE * n
        r_refl = r_hit_nudged + d_refl * jnp.maximum(remaining - NUDGE, jnp.float32(0.0))
        r_straight = r + step
        r_out = jnp.where(reflect_here, r_refl, r_straight)

        # safety: never let a walker sit outside the reflecting wall R_out
        rad_out = self._radial(r_out); rmag = jnp.linalg.norm(rad_out)
        over = rmag - Rout
        r_out = jnp.where(over > 0.0, r_out - 2.0 * over * (rad_out / jnp.maximum(rmag, EPS)), r_out)

        dlog_w = jnp.where(hit_in & any_hit & (~transmit),
                           -jnp.float32(2.0) * rho_over_D * d_perp, jnp.float32(0.0))
        return r_out, dlog_w

    def permeate(self, r, step, kappa_over_D, rho_over_D, perm_key):
        return self._permeate_impl(r, step, kappa_over_D, rho_over_D, perm_key)

    def reflect(self, r, step):
        r_out, _ = self._permeate_impl(r, step, jnp.float32(0.0), jnp.float32(0.0),
                                       jax.random.PRNGKey(0))
        return r_out
