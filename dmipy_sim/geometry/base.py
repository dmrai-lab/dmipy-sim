"""Geometry base class and the trivially-bounded geometries.

Holds the ABC every geometry implements, the two geometries with no curved wall
(free diffusion, a 1-D box), and the frame helpers shared across the package.
"""
from abc import ABC, abstractmethod

from ._boundary import WallHit
import jax
import jax.numpy as jnp
import numpy as np


def initial_positions(geometry, n_walkers, key, r0=None):
    """Seed positions for a walk: ``r0`` when the caller supplied one, else the geometry's own.

    Exists because the default is a trap on meshes. ``Mesh.init_positions`` takes an ``intra`` flag
    that defaults to ``True`` -- INSIDE the surface -- so a driver that seeds itself silently picks a
    compartment on the caller's behalf. For a sphere or cylinder that guess is right. For a fibre
    bundle's extra-axonal pool, whose geometry is the OUTER surface and whose walkers belong outside
    it, the guess re-simulates the intra pool and labels it "extra": measured at 0.54x the extra pool's
    analytic ``(S/V)*D`` before the MT driver was given an ``r0``.

    Every walk driver should route its seeding through here, so that "which pool did this run walk?"
    has one answer and one validation instead of one per driver. It replaced four inline copies, two of
    which checked the shape and two of which did not.
    """
    if r0 is None:
        return geometry.init_positions(n_walkers, key)
    out = jnp.asarray(r0, dtype=jnp.float32)
    if out.shape != (n_walkers, 3):
        raise ValueError(f"r0 must have shape ({n_walkers}, 3), got {tuple(out.shape)}")
    return out


class Geometry(ABC):
    @abstractmethod
    def init_positions(self, n_walkers: int, key: jax.Array) -> jnp.ndarray:
        """Return initial walker positions of shape (n_walkers, 3), float32."""

    #: Does this geometry have a membrane a walker can cross? Absence of a `permeate`
    #: method used to be the signal, which meant "cannot cross" and "not implemented yet"
    #: were indistinguishable and a silent `reflect` fallback was the failure mode.
    supports_permeability = False

    #: Does `permeate` accept a carried `side` (the walker's own compartment)? Only the
    #: geometries where a position alone cannot decide sidedness need it -- see #86.
    carries_side = False

    def interact(self, r, step, *, kappa_over_D=0.0, rho_over_D=0.0,
                 key=None, side=None):
        """One wall interaction, for every geometry: reflect, or cross if granted.

        This is the single entry point the engine should use. `reflect`,
        `reflect_with_log_weight` and `permeate` are the same function at different argument
        values -- and measured, not even an optimisation: `reflect` and `permeate(kappa=0)`
        both cost 0.11 ms / 40k walkers, because XLA folds the constant and drops the dead
        transmit branch. Keeping them apart is what let them drift into four separate bugs
        (#88): packed geometries expelled intra walkers, analytic ones absorbed exterior
        walkers, `Mesh.reflect` silently lost box reflection, and mesh surface local time
        disagreed with itself by 0.07%.

        Returns a :class:`WallHit`, so a caller reads ``.r`` instead of unpacking a tuple
        whose length depended on which arguments were passed.
        """
        kappa_is_zero = isinstance(kappa_over_D, (int, float)) and kappa_over_D == 0.0
        if not kappa_is_zero and not self.supports_permeability:
            raise NotImplementedError(
                f"{type(self).__name__} has no membrane: it cannot be given "
                f"kappa_over_D != 0. Build it with a permeable geometry, or pass "
                f"kappa_over_D=0 for a purely reflecting wall.")

        if self.supports_permeability:
            k = key if key is not None else jax.random.PRNGKey(0)
            args = (r, step, kappa_over_D, rho_over_D, k)
            if side is not None and not self.carries_side:
                raise NotImplementedError(
                    f"{type(self).__name__} does not carry a compartment side; a position "
                    f"exactly on its wall cannot be resolved. Omit `side`.")
            out = self.permeate(*args, side) if side is not None else self.permeate(*args)
            if len(out) == 4:                      # geometry reports crossing itself
                return WallHit(out[0], out[1], out[2], out[3])
            # Otherwise DERIVE it. Reporting `crossed=False` because the geometry does not
            # return the flag is a lie about the physics: measured on PermeableSlab1D,
            # 256/256 walkers ended on the far side of the membrane while `crossed` said
            # none had. A crossing IS a change of compartment, so ask the classifier --
            # which is exact, and the only honest answer available without changing six
            # `permeate` signatures. Costs one `classify_position` per call, on the
            # permeable path only.
            zero = jnp.zeros((), bool)
            if not hasattr(self, "classify_position"):
                return WallHit(out[0], out[1], zero, zero)
            crossed = self.classify_position(out[0]) != self.classify_position(r)
            return WallHit(out[0], out[1], crossed, zero)

        # impermeable: relaxation path if it exists and is asked for, else a plain bounce
        zero_b = jnp.zeros((), bool)
        if hasattr(self, "reflect_with_log_weight"):
            r_new, dlog_w = self.reflect_with_log_weight(r, step, rho_over_D)
            return WallHit(r_new, dlog_w, zero_b, zero_b)
        return WallHit(self.reflect(r, step), jnp.zeros((), jnp.float32), zero_b, zero_b)


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
