"""Periodic packings of many objects, with minimum-image geometry.

One wall kernel, :func:`packed_wall_kernel`, serves the cylinder (2-D cross-section) and the
sphere (3-D) packs: it ray-traces the step against the walls within reach, reflects off the one
it meets first, and keeps going until the path is spent.
"""
import jax
import jax.numpy as jnp
import numpy as np

from ._boundary import (keep_side_radial, ray_sphere_t, specular, transmit_probability, off_wall,
                        bounce_loop, bounce_budget)
from .base import Geometry, LengthScales, _rotation_to_z

_TINY = 1e-30


def packed_bounce_budget(R_min, nudge, min_gap, step_max):
    """:func:`dmipy_sim.geometry._boundary.bounce_budget` for a pack."""
    return bounce_budget(R_min, nudge, min_gap, step_max)


def packed_candidate_count(N, R_min, step_max, dim):
    """Objects whose wall can lie within one step of a point: disjoint objects of radius at least
    ``R_min`` each subtend, from that point, an angle of at least ``2 asin(R / (R + step))``, so at
    most ``pi (1 + step / R)`` disks (2-D) or ``2 / (1 - cos theta)`` balls (3-D) fit. Never fewer
    than 8 unless the pack itself is smaller."""
    x = float(R_min) / (float(R_min) + float(step_max))
    if dim == 2:
        n = np.ceil(np.pi * (1.0 + float(step_max) / float(R_min))) + 2
    else:
        n = np.ceil(2.0 / (1.0 - np.sqrt(max(0.0, 1.0 - x * x)))) + 2
    return int(min(N, max(8, n)))


def packed_wall_kernel(centers, radii, L, eps, nudge, step_max, min_gap):
    """The wall interaction of a periodic pack of disjoint objects, in ``centers``' space.

    ``centers`` is ``(N, 2)`` for parallel cylinders (the cross-section plane) or ``(N, 3)`` for
    spheres; ``|q - c_k| = radii_k`` are the walls, ``L`` the periodic cell. Ray-traced hits
    against the walls within reach of the step, ``d_perp = remaining * |cos alpha|`` at the hit,
    specular multi-bounce reflection off the wall the ray actually meets, a strict side sentinel
    at the end (:mod:`._boundary`). A walker's side -- outside every object, or inside one -- is
    carried state that changes only where a crossing is granted.

    Every wall encounter is its own Powles trial, ``p = 2 (kappa/D) d_perp``, with an independent
    uniform. Summed over the encounters of a step this is ``(kappa/D)`` times the boundary local
    time the reflections record, the same estimator the surface-relaxivity channel is validated
    on (Brownstein-Tarr), so the transmission does not depend on how many walls a step happens
    to meet. Deciding only at the first encounter would under-permeate by the number of
    encounters per step wherever a step spans a gap between two walls -- eight-fold on a dense
    pack whose gap is an eighth of the step -- and the only cure would be sub-stepping down to
    the narrowest gap of the pack, at ``(step / gap)^2`` cost, for a passage most walkers never
    visit.

    The bounce budget and the candidate count come from the worst case a step of ``step_max``
    can meet (:func:`packed_bounce_budget`, :func:`packed_candidate_count`).

    Returns ``wall(p, d_hat, step_l, inside0, kappa_over_D, rho_over_D, key)`` giving
    ``(p_new, dlog_w, crossed, illegal)``: the end position, the surface log-weight increment
    ``-2 (rho/D) d_perp`` summed over the reflections, whether the side changed (an odd number
    of crossings), and whether the end sentinel had to correct the side.
    """
    N, dim = int(centers.shape[0]), int(centers.shape[1])
    R_min = float(np.min(np.asarray(radii)))
    max_bounces = packed_bounce_budget(R_min, nudge, min_gap, step_max)
    n_cand = packed_candidate_count(N, R_min, step_max, dim)
    f32 = jnp.float32
    ar = jnp.arange(n_cand)

    def _wrap(q):
        return q - L * jnp.floor(q / L + f32(0.5))

    def wall(p, d_hat, step_l, inside0, kappa_over_D, rho_over_D, key):
        us = jax.random.uniform(key, (max_bounces,), dtype=jnp.float32)      # one draw per encounter
        q_all = _wrap(p[None, :] - centers)                                    # (N, dim)
        dist_wall = jnp.sqrt(jnp.sum(q_all * q_all, -1)) - radii
        if n_cand < N:
            _, idx = jax.lax.top_k(-dist_wall, n_cand)                         # walls within reach
        else:
            idx = ar
        c_c, R_c = centers[idx], radii[idx]

        def hit_once(pp, d, rem, i):
            q = _wrap(pp[None, :] - c_c)                                       # (n_cand, dim)
            d2 = jnp.sum(q * q, -1)
            kc = jnp.argmin(d2 / (R_c * R_c))
            in_own = d2[kc] < R_c[kc] * R_c[kc]                                # inside one object
            t_en, t_ex, disc = ray_sphere_t(q, d, R_c)
            t = jnp.where((ar == kc) & in_own, t_ex, t_en)
            t = jnp.where((disc > 0) & (t > eps) & (t < rem), t, jnp.inf)
            kh = jnp.argmin(t)
            t_hit = t[kh]
            any_hit = jnp.isfinite(t_hit) & (rem > 0)
            t_safe = jnp.where(any_hit, t_hit, f32(0.0))
            R_hit = R_c[kh]
            raw = q[kh] + t_safe * d
            n_out = raw / jnp.maximum(jnp.linalg.norm(raw), _TINY)
            rem_a = rem - t_safe
            d_perp = jnp.where(any_hit, rem_a * jnp.sqrt(jnp.maximum(disc[kh], f32(0.0))) / R_hit,
                               f32(0.0))
            stay_in = in_own & (kh == kc)
            transmit = any_hit & (us[i] < transmit_probability(kappa_over_D, d_perp))
            d_refl = specular(d, n_out)
            d_refl = d_refl / jnp.maximum(jnp.linalg.norm(d_refl), _TINY)
            q_off = off_wall(R_hit * n_out, n_out, stay_in, nudge)
            q_off, _ = keep_side_radial(q_off, q_off, R_hit, stay_in, nudge, active=~transmit)
            reflecting = any_hit & (~transmit)
            p_new = jnp.where(reflecting, pp + (q_off - q[kh]), pp + rem * d)
            d_new = jnp.where(reflecting, d_refl, d)
            rem_new = jnp.where(reflecting, jnp.maximum(rem_a - nudge, f32(0.0)), f32(0.0))
            dlw = jnp.stack([jnp.where(reflecting, -f32(2.0) * rho_over_D * d_perp, f32(0.0)),
                             transmit.astype(f32)])
            return p_new, d_new, rem_new, i + 1, dlw, transmit

        p_f, dlw, _ = bounce_loop(hit_once, p, d_hat, step_l, max_bounces,
                                  dlog_init=jnp.zeros(2, f32), decided_init=jnp.int32(0))
        dlog_w = dlw[0]
        # The side is the one at the START of the step, flipped once per granted crossing.
        crossed = (jnp.round(dlw[1]).astype(jnp.int32) % 2) == 1
        want_in = jnp.logical_xor(inside0, crossed)
        q_f = _wrap(p_f[None, :] - centers)
        k_f = jnp.argmin(jnp.sqrt(jnp.sum(q_f * q_f, -1)) - radii)               # nearest wall
        p_f, illegal = keep_side_radial(p_f, q_f[k_f], radii[k_f], want_in, nudge)
        return p_f, dlog_w, crossed, illegal

    wall.max_bounces = max_bounces
    wall.n_cand = n_cand
    return wall


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
        images), metres. It sizes the bounce budget of the wall kernel.

    Notes
    -----
    Reflection algorithm
    ~~~~~~~~~~~~~~~~~~~~
    :func:`packed_wall_kernel`: the step is ray-traced against the cylinders within reach,
    reflected off the first wall it meets, and continued until its path is spent, so a step
    longer than a gap zig-zags across it rather than ending inside the neighbour.

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
        # the wall kernel, built here (outside any trace) at the worst-case step the sub-step
        # rule allows, R_min / 6
        self._wall = packed_wall_kernel(self._centers_jax, self._radii_jax, self._L_jax,
                                        self._eps_detect, self._nudge, step_max=min_r / 6.0,
                                        min_gap=self.min_gap)

    @property
    def length_scales(self):
        return LengthScales(min_feature=float(np.min(self._radii_np)), min_gap=self.min_gap)

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
        """Wall interaction in the pack: reflect off the cylinders the step meets, or cross one
        where its membrane grants it (a Powles trial at every encounter). Multi-bounce, so a step longer than the gap
        between two cylinders zig-zags across it instead of ending inside the neighbour.

        With ``side`` (int8, ``< 0`` intra, ``>= 0`` extra) the walker's own compartment rules
        the sentinel and the call returns ``(r_new, dlog_w, crossed, illegal)``; without it the
        side is read at the start of the step and the call returns ``(r_new, dlog_w)``.
        """
        if self._is_identity_rotation:
            r_c, step_c = r, step
        else:
            r_c, step_c = self._R @ r, self._R @ step
        r2, step_xy, step_z = r_c[:2], step_c[:2], step_c[2]
        step_l_xy = jnp.linalg.norm(step_xy)
        d_hat_xy = jnp.where(step_l_xy > 0, step_xy / jnp.maximum(step_l_xy, self._eps_detect),
                             jnp.zeros(2, jnp.float32))
        if side is not None:
            inside0 = side < jnp.int8(0)
        else:
            q_all = r2[None, :] - self._centers_jax
            q_all = q_all - self._L_jax * jnp.floor(q_all / self._L_jax + jnp.float32(0.5))
            inside0 = jnp.any(jnp.sum(q_all * q_all, axis=1) < self._radii_jax ** 2)
        xy_final, dlog_w, crossed, illegal = self._wall(
            r2, d_hat_xy, step_l_xy, inside0, jnp.float32(kappa_over_D), jnp.float32(rho_over_D),
            perm_key)
        r_c_new = jnp.stack([xy_final[0], xy_final[1], r_c[2] + step_z])
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
        images), metres. It sizes the bounce budget of the wall kernel.
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
        # the wall kernel, built here (outside any trace) at the worst-case step the sub-step
        # rule allows, R_min / 6
        self._wall = packed_wall_kernel(self._centers_jax, self._radii_jax, self._L_jax,
                                        self._eps_detect, self._nudge, step_max=min_r / 6.0,
                                        min_gap=self.min_gap)

    @property
    def length_scales(self):
        return LengthScales(min_feature=float(np.min(self._radii_np)), min_gap=self.min_gap)

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
        """Wall interaction in the pack: reflect off the spheres the step meets, or cross one where
        its membrane grants it (a Powles trial at every encounter). Multi-bounce, so a step longer than the gap
        between two spheres zig-zags across it instead of ending inside the neighbour. The side at
        the start of the step rules the end sentinel.
        """
        step_l = jnp.linalg.norm(step)
        d_hat = jnp.where(step_l > 0, step / jnp.maximum(step_l, self._eps_detect),
                          jnp.zeros(3, jnp.float32))
        q_all = r[None, :] - self._centers_jax
        q_all = q_all - self._L_jax * jnp.floor(q_all / self._L_jax + jnp.float32(0.5))
        inside0 = jnp.any(jnp.sum(q_all * q_all, axis=1) < self._radii_jax ** 2)
        r_out, dlog_w, _crossed, _illegal = self._wall(
            r, d_hat, step_l, inside0, jnp.float32(kappa_over_D), jnp.float32(rho_over_D), perm_key)
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
