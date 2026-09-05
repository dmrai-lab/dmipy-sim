"""Analytic single-object geometries: sphere, cylinder, ellipsoid, slab, shell.

One closed surface each, with a closed-form ray intersection. The boundary rules
themselves live in `_boundary`; these supply only the surface parametrisation.
"""
import jax
import jax.numpy as jnp
import numpy as np

from ._boundary import (bounce_loop, keep_side_radial, keep_side_planar, keep_side_quadric,
                        ray_sphere_t, ray_quadric_t, specular,
                        transmit_probability, off_wall, step_off_wall)
from .base import Geometry, LengthScales, _rotation_to_z


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

    _MAX_BOUNCES = 8   #: a step spanning the object needs more than one reflection

    supports_permeability = True   #: has a membrane a walker can cross

    def __init__(self, radius: float, surface_relaxivity_t2=None,
                 permeability=None):
        self.radius = float(radius)
        self.surface_relaxivity_t2 = (
            float(surface_relaxivity_t2) if surface_relaxivity_t2 is not None else None
        )
        self.permeability = (
            float(permeability) if permeability is not None else None
        )

    @property
    def length_scales(self):
        return LengthScales(min_feature=self.radius)

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
        """Impermeable wall interaction -- the kappa = 0 case of :meth:`permeate`.

        Not a separate algorithm. It was one, and the copies drifted: the hand-written
        version CLAMPED exterior walkers to just inside the surface, absorbing rather than
        reflecting them (#88). `permeate` is multi-bounce, so a step spanning the object is
        handled here too. XLA folds kappa = 0 and drops the dead transmit branch.
        """
        return self.permeate(r, step, jnp.float32(0.0), jnp.float32(0.0),
                             jax.random.PRNGKey(0))[0]

    def reflect_with_log_weight(self, r, step, rho_over_D):
        """Impermeable wall interaction that also accrues surface relaxation."""
        return self.permeate(r, step, jnp.float32(0.0), rho_over_D,
                             jax.random.PRNGKey(0))[:2]

    def permeate(self, r, step, kappa_over_D, rho_over_D, perm_key):
        """Wall interaction on the sphere: reflect, or cross if the membrane grants it.

        Multi-bounce -- a step longer than the chord needs more than one reflection, and
        resolving only the first lets the walker exit the far side (#88). The crossing
        decision is still made at most once per step, because permeation is a single-event
        approximation while reflection is not.
        """
        R     = jnp.float32(self.radius)
        EPS   = jnp.float32(1e-7 * self.radius)
        NUDGE = jnp.float32(1e-4 * self.radius)
        step_l = jnp.linalg.norm(step)
        d_hat = jnp.where(step_l > 0, step / jnp.maximum(step_l, EPS),
                          jnp.zeros(3, jnp.float32))
        u = jax.random.uniform(perm_key, dtype=jnp.float32)

        def hit_once(rr, dd, remaining, decided):
            t_entry, t_exit, disc = ray_sphere_t(rr, dd, R)
            disc_s = jnp.maximum(disc, jnp.float32(0.0))
            inside = jnp.dot(rr, rr) < R * R
            t_hit  = jnp.where(inside, t_exit, t_entry)
            any_hit = ((disc > 0) & (t_hit > EPS) & (t_hit < remaining) & (remaining > 0))
            t_safe = jnp.where(any_hit, t_hit, jnp.float32(0.0))
            raw    = rr + t_safe * dd
            n_out  = raw / jnp.maximum(jnp.linalg.norm(raw), jnp.float32(1e-30))
            r_hit  = R * n_out
            rem    = remaining - t_safe
            d_perp = jnp.where(any_hit, rem * jnp.sqrt(disc_s) / R, jnp.float32(0.0))
            first    = any_hit & (~decided)
            transmit = first & (u < transmit_probability(kappa_over_D, d_perp))
            d_refl = specular(dd, n_out)
            d_refl = d_refl / jnp.maximum(jnp.linalg.norm(d_refl), jnp.float32(1e-30))
            r_off  = off_wall(r_hit, n_out, inside, NUDGE)
            r_off, _ = keep_side_radial(r_off, r_off, R, inside, NUDGE, active=~transmit)
            reflecting = any_hit & (~transmit)
            r_new  = jnp.where(reflecting, r_off, rr + remaining * dd)
            d_new  = jnp.where(reflecting, d_refl, dd)
            rem_n  = jnp.where(reflecting, jnp.maximum(rem - NUDGE, jnp.float32(0.0)),
                               jnp.float32(0.0))
            dlw = jnp.where(reflecting & first,
                            -jnp.float32(2.0) * rho_over_D * d_perp, jnp.float32(0.0))
            return r_new, d_new, rem_n, decided | first, dlw, transmit

        r_out, dlog_w, crossed = bounce_loop(hit_once, r, d_hat, step_l, self._MAX_BOUNCES)
        # Final-position sentinel (#86): a step whose exit time marginally exceeds its length
        # fires no collision -- correctly, it never reaches the wall -- so the raw step is
        # kept and float32 can round the endpoint onto |r| = R, where the strict test reads
        # the OTHER compartment. The side that matters is the one at the START of the whole
        # step, not per bounce, and a granted crossing is the only thing that may change it.
        inside0 = jnp.dot(r, r) < R * R
        r_out, _ = keep_side_radial(r_out, r_out, R, inside0, NUDGE, active=~crossed)
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

    _MAX_BOUNCES = 8   #: a step spanning the object needs more than one reflection

    supports_permeability = True   #: has a membrane a walker can cross

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

    @property
    def length_scales(self):
        return LengthScales(min_feature=self.radius)

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
        """Impermeable wall interaction -- the kappa = 0 case of :meth:`permeate`.

        Not a separate algorithm. It was one, and the copies drifted: the hand-written
        version CLAMPED exterior walkers to just inside the surface, absorbing rather than
        reflecting them (#88). `permeate` is multi-bounce, so a step spanning the object is
        handled here too. XLA folds kappa = 0 and drops the dead transmit branch.
        """
        return self.permeate(r, step, jnp.float32(0.0), jnp.float32(0.0),
                             jax.random.PRNGKey(0))[0]

    def reflect_with_log_weight(self, r, step, rho_over_D):
        """Impermeable wall interaction that also accrues surface relaxation."""
        return self.permeate(r, step, jnp.float32(0.0), rho_over_D,
                             jax.random.PRNGKey(0))[:2]

    def permeate(self, r, step, kappa_over_D, rho_over_D, perm_key):
        """Wall interaction on the cylinder: reflect, or cross if the membrane grants it.

        Multi-bounce. A step longer than the chord needs more than one reflection, and
        resolving only the first lets the walker exit the far side -- measured at 58x the
        radius on a 1 um cylinder (#88). The crossing decision is still made at most once
        per step (`decided`), because permeation is a single-event approximation while
        reflection is not.

        `reflect` and `reflect_with_log_weight` are the kappa = 0 and rho > 0 cases of this.
        """
        R     = jnp.float32(self.radius)
        EPS   = jnp.float32(1e-7 * self.radius)
        NUDGE = jnp.float32(1e-4 * self.radius)
        r_c    = r    if self._is_identity_rotation else self._R @ r
        step_c = step if self._is_identity_rotation else self._R @ step
        step_xy, step_z = step_c[:2], step_c[2]
        step_l_xy = jnp.linalg.norm(step_xy)
        d_hat_xy = jnp.where(step_l_xy > 0, step_xy / jnp.maximum(step_l_xy, EPS),
                             jnp.zeros(2, jnp.float32))
        u = jax.random.uniform(perm_key, dtype=jnp.float32)

        def hit_once(r2, d2, remaining, decided):
            t_entry, t_exit, disc = ray_sphere_t(r2, d2, R)
            disc_s = jnp.maximum(disc, jnp.float32(0.0))
            inside = jnp.dot(r2, r2) < R * R
            t_hit  = jnp.where(inside, t_exit, t_entry)
            any_hit = ((disc > 0) & (t_hit > EPS) & (t_hit < remaining) & (remaining > 0))
            t_safe = jnp.where(any_hit, t_hit, jnp.float32(0.0))
            raw    = r2 + t_safe * d2
            n_out  = raw / jnp.maximum(jnp.linalg.norm(raw), jnp.float32(1e-30))
            r2_hit = R * n_out                       # snapped: |r2_hit| = R exactly
            rem    = remaining - t_safe
            d_perp = jnp.where(any_hit, rem * jnp.sqrt(disc_s) / R, jnp.float32(0.0))

            first = any_hit & (~decided)
            transmit = first & (u < transmit_probability(kappa_over_D, d_perp))

            d_refl = specular(d2, n_out)
            d_refl = d_refl / jnp.maximum(jnp.linalg.norm(d_refl), jnp.float32(1e-30))
            # off the wall on the side the walker stays on, then carry on with what is left
            r2_off = off_wall(r2_hit, n_out, inside, NUDGE)
            r2_off, _ = keep_side_radial(r2_off, r2_off, R, inside, NUDGE,
                                         active=~transmit)
            reflecting = any_hit & (~transmit)
            r2_new  = jnp.where(reflecting, r2_off, r2 + remaining * d2)
            d2_new  = jnp.where(reflecting, d_refl, d2)
            rem_new = jnp.where(reflecting, jnp.maximum(rem - NUDGE, jnp.float32(0.0)),
                                jnp.float32(0.0))
            dlw = jnp.where(reflecting & first,
                            -jnp.float32(2.0) * rho_over_D * d_perp, jnp.float32(0.0))
            return r2_new, d2_new, rem_new, decided | first, dlw, transmit

        xy_final, dlog_w, crossed = bounce_loop(hit_once, r_c[:2], d_hat_xy, step_l_xy,
                                                self._MAX_BOUNCES)
        # Final-position sentinel (#86) -- see Sphere.permeate. The side is the one at the
        # START of the whole step; only a granted crossing may change it.
        inside0 = jnp.dot(r_c[:2], r_c[:2]) < R * R
        xy_final, _ = keep_side_radial(xy_final, xy_final, R, inside0, NUDGE,
                                       active=~crossed)
        r_c_new = jnp.stack([xy_final[0], xy_final[1], r_c[2] + step_z])
        r_out = r_c_new if self._is_identity_rotation else self._R_inv @ r_c_new
        return r_out, dlog_w

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


class Ellipsoid(Geometry):
    """Reflecting axis-aligned ellipsoid with semi-axes (a, b, c) along (x, y, z).

    The ellipsoid surface is defined by x²/a² + y²/b² + z²/c² = 1.
    When a = b = c = r the geometry is identical to Sphere(r).
    """

    _MAX_BOUNCES = 8   #: a step spanning the object needs more than one reflection

    supports_permeability = True   #: has a membrane a walker can cross

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

    @property
    def length_scales(self):
        return LengthScales(min_feature=float(np.min(self.semiaxes)))   # the tightest semi-axis

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
            _, d, disc = ray_quadric_t(A, B, C)      # exit root: forward to the surface

            intersects = (d > EPS_detect) & (d < remaining)

            r_hit   = r0 + d * d_hat
            # Outward normal: gradient of f(x)=x·D·x at r_hit, normalised
            n_raw   = r_hit * inv_semi_sq
            n_out   = n_raw / jnp.linalg.norm(n_raw)
            d_refl  = specular(d_hat, n_out)
            d_refl  = d_refl / jnp.linalg.norm(d_refl)
            r_nudge = off_wall(r_hit, n_out, True, NUDGE)

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
            _, d, disc = ray_quadric_t(A, B, C)      # exit root: forward to the surface

            intersects = (d > EPS_detect) & (d < remaining)

            r_hit   = r0 + d * d_hat
            n_raw   = r_hit * inv_semi_sq
            n_out   = n_raw / jnp.linalg.norm(n_raw)
            d_refl  = specular(d_hat, n_out)
            d_refl  = d_refl / jnp.linalg.norm(d_refl)
            r_nudge = off_wall(r_hit, n_out, True, NUDGE)

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
        t_entry, t_exit, disc_raw = ray_quadric_t(A, B, C)
        disc_A = jnp.maximum(disc_raw, jnp.float32(0.0))

        # ── Side detection and root selection ────────────────────────────
        inside  = C < jnp.float32(0.0)                             # r·D·r < 1
        t_hit   = jnp.where(inside, t_exit, t_entry)
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
        p_transmit = transmit_probability(kappa_over_D, d_perp)
        u        = jax.random.uniform(perm_key, dtype=jnp.float32)
        transmit = any_hit & (u < p_transmit)

        # ── Reflected: specular, nudge back to same side ─────────────────
        d_refl    = specular(d_hat, n_out)
        d_refl    = d_refl / jnp.linalg.norm(d_refl)
        r_refl = step_off_wall(r_hit, n_out, inside, d_refl, remaining, NUDGE)

        # ── Transmitted: straight through ────────────────────────────────
        r_straight = r + step

        # ── Combine ───────────────────────────────────────────────────────
        r_hit_result = jnp.where(transmit, r_straight, r_refl)
        r_out        = jnp.where(any_hit,  r_hit_result, r + step)

        # ── Compartment sentinel: no granted crossing => no change of side ────────
        # Same defect as the sphere, on the quadric r.D.r = 1 instead of |r| = R: a step
        # landing exactly on the surface fires no collision and the strict `C < 0` test then
        # reads the other compartment. Measured on a plain random walk at kappa = 0: 0.063%
        # (interior) and 0.248% (exterior) of walkers per 30k steps.
        #
        # Scaling r by sqrt(target/Q) moves it along the ray from the centre onto the level
        # set Q = target, which is the ellipsoid's own radial direction. `inside` is the side
        # at the START of the step and `transmit` the only way to leave it.
        _Q = jnp.dot(r_out * inv_semi_sq, r_out)           # 1.0 exactly on the surface
        r_out, _ = keep_side_quadric(r_out, _Q, inside,
                                     NUDGE / jnp.float32(_min_semi), active=~transmit)

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

    supports_permeability = True   #: has a membrane a walker can cross

    def __init__(self, length, permeability, surface_relaxivity_t2=None):
        self.length = float(length)
        self.permeability = float(permeability)
        self.surface_relaxivity_t2 = (float(surface_relaxivity_t2)
                                      if surface_relaxivity_t2 is not None else None)
        self.radius = float(length) / 2.0     # one compartment's width

    @property
    def length_scales(self):
        return LengthScales(min_feature=self.length / 2.0)     # one compartment's width

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
        p = transmit_probability(kappa_over_D, d_perp)
        u = jax.random.uniform(perm_key, dtype=jnp.float32)
        transmit = crossed & (u < p)
        x1 = jnp.where(crossed & ~transmit, 2.0 * xm - x_new, x_new)   # reflect at membrane
        x1 = self._fold(x1)

        # ── Compartment sentinel: no granted crossing => no change of side ────────
        # `crossed` is a STRICT sign change, so a step landing exactly on the membrane
        # (x_new == xm) gives a zero product and fires no reflection -- while
        # classify_position's strict `x < L/2` reads that same coordinate as the far
        # compartment. The walker changes compartment without crossing. This was the
        # worst of the family on the boundary-landing probe: 80.9% of walkers flipped,
        # with the engine correcting none of them.
        x1, _ = keep_side_planar(x1, xm, x < xm, jnp.float32(1e-4) * xm,
                                 active=~transmit)

        r_out = jnp.array([x1, r[1] + step[1], r[2] + step[2]])
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

    supports_permeability = True   #: has a membrane a walker can cross

    def __init__(self, r_inner, r_outer, permeability, kind='sphere',
                 orientation=(0.0, 0.0, 1.0), surface_relaxivity_t2=None):
        assert kind in ('sphere', 'cylinder')
        self.r_inner = float(r_inner); self.r_outer = float(r_outer)
        self.permeability = float(permeability)
        self.kind = kind
        self.surface_relaxivity_t2 = (float(surface_relaxivity_t2)
                                      if surface_relaxivity_t2 is not None else None)
        self.radius = float(r_inner)
        o = np.asarray(orientation, dtype=np.float64); self._o = o / np.linalg.norm(o)
        self._axis = jnp.array(self._o, dtype=jnp.float32)

    @property
    def length_scales(self):
        return LengthScales(min_feature=self.r_inner)

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
            t1, t2, disc = ray_quadric_t(dd2, b, c)
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

        p = transmit_probability(kappa_over_D, d_perp)
        u = jax.random.uniform(perm_key, dtype=jnp.float32)
        transmit = hit_in & any_hit & (u < p)                     # R_out never transmits
        reflect_here = any_hit & (~transmit)

        d_refl = specular(d, n)
        # nudge the reflected walker OFF the membrane onto its own side, so it never
        # straddles the surface (straddling biases the next step's crossing -> breaks
        # detailed balance).  Side: walkers with radius < the hit radius stay inside it.
        NUDGE = jnp.float32(1e-4 * self.r_inner)
        r_rad_mag = jnp.linalg.norm(self._radial(r))
        Rk = jnp.where(hit_in, Rin, Rout)
        r_refl = step_off_wall(r_hit, n, r_rad_mag < Rk, d_refl, remaining, NUDGE)
        r_straight = r + step
        r_out = jnp.where(reflect_here, r_refl, r_straight)

        # safety: never let a walker sit outside the reflecting wall R_out
        rad_out = self._radial(r_out); rmag = jnp.linalg.norm(rad_out)
        over = rmag - Rout
        r_out = jnp.where(over > 0.0, r_out - 2.0 * over * (rad_out / jnp.maximum(rmag, EPS)), r_out)

        # ── Compartment sentinel: no granted crossing => no change of side ────────
        # Compartments here are split at R_inner (classify_position: rad < R_inner). A step
        # landing exactly on R_inner fires no collision and the strict test then reads the
        # far compartment -- the walker changes compartment without crossing. Scaling only
        # the RADIAL part keeps the axial coordinate untouched, so this is correct for the
        # cylinder kind as well as the sphere. O(1) on values already computed.
        _side0 = jnp.linalg.norm(self._radial(r)) < Rin
        r_out, _ = keep_side_radial(r_out, self._radial(r_out), Rin, _side0, NUDGE,
                                    active=~transmit)

        dlog_w = jnp.where(hit_in & any_hit & (~transmit),
                           -jnp.float32(2.0) * rho_over_D * d_perp, jnp.float32(0.0))
        return r_out, dlog_w

    def permeate(self, r, step, kappa_over_D, rho_over_D, perm_key):
        return self._permeate_impl(r, step, kappa_over_D, rho_over_D, perm_key)

    def reflect(self, r, step):
        r_out, _ = self._permeate_impl(r, step, jnp.float32(0.0), jnp.float32(0.0),
                                       jax.random.PRNGKey(0))
        return r_out
