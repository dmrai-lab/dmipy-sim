"""The closed-form field of parallel hollow cylinders as a walk reads it: :class:`HollowCylinderField` sums
dmipy-analytic's field basis of one infinite hollow cylinder (:func:`dmipy_analytic.fields.hollow_cylinder_basis`, read through its JAX
twin :func:`dmipy_analytic.jax.fields.hollow_cylinder_basis`) over the cylinders and their periodic images, at a
walker's own position. It is the analytic myelin provider of the forward Bloch engine
(:class:`dmipy_sim.fields.susceptibility.MyelinSusceptibility`) and, per segment, of a strand substrate's field
(:mod:`dmipy_sim.fields.strand_field`).

The susceptibility convention is shared with dmipy-analytic:
``chi = chi_I * I + chi_A * (n n^T - I/3)``, with radial unit director ``n``.
Here ``chi_I = chi_iso = tr(chi)/3`` is the mean susceptibility and
``chi_A = chi_aniso = chi_radial - chi_tangential`` is the eigenvalue difference
(also named ``delta_chi_a``). Both use the same susceptibility reference.
Wharton & Bowtell (2012), Table 1 and Fig. 3C, use
``chi = chi_I * I + diag(chi_A_WB, -chi_A_WB/2, -chi_A_WB/2)`` in the radial frame.
Thus ``chi_A = (3/2) * chi_A_WB``, while ``chi_I`` is identical: W&B's
``chi_A_WB = -0.1 ppm`` corresponds to this package's
``chi_aniso = -0.15e-6`` (``-0.15 ppm``).
Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC3494918/ (Table 1 and Fig. 3C).
"""
from __future__ import annotations

import numpy as np
from dmipy_replay.replay.field import CHANNEL_NAMES

#: the periodic images a cell's closed-form field sums per axis (the image sum is conditionally convergent; six per
#: side is the forward Bloch provider's reference, :class:`~dmipy_sim.fields.susceptibility.MyelinSusceptibility`)
CELL_IMAGES = 6


class HollowCylinderField:
    """The field basis of parallel hollow cylinders along ``z`` in closed form
    (:func:`~dmipy_analytic.fields.hollow_cylinder_basis` summed
    over the cylinders, and over ``n_images`` periodic images per side in a cell of side ``L``): what a walk reads at
    a walker's own position, exactly, where a raster smears the sheath's jump over a voxel. The same
    ``channels(points)`` protocol as :class:`~dmipy_sim.fields.susceptibility_field.FieldGrid` and
    :class:`~dmipy_sim.fields.strand_field.StrandFieldBasis`, and :meth:`device_read` for the walk; the bare
    superposition values (no mean subtracted), the field :class:`~dmipy_sim.fields.susceptibility.MyelinSusceptibility`
    evaluates for the forward engine. ``centers`` ``(N, 2)``, ``inner_radii`` / ``outer_radii`` ``(N,)`` in metres;
    ``L`` the cell's side for a periodic substrate (positions wrapped into ``[-L/2, L/2)``), ``None`` for isolated
    cylinders."""

    channel_names = CHANNEL_NAMES

    def __init__(self, centers, inner_radii, outer_radii, *, L=None, n_images=CELL_IMAGES):
        self.centers = np.asarray(centers, np.float64).reshape(-1, 2)
        self.inner_radii = np.asarray(inner_radii, np.float64).reshape(-1)
        self.outer_radii = np.asarray(outer_radii, np.float64).reshape(-1)
        if not (self.centers.shape[0] == self.inner_radii.shape[0] == self.outer_radii.shape[0]):
            raise ValueError("centers, inner_radii and outer_radii describe the same cylinders")
        if not np.all(self.outer_radii > self.inner_radii):
            raise ValueError("every outer radius exceeds its inner radius")
        self.L = None if L is None else float(L)
        self.n_images = 0 if L is None else int(n_images)
        self.periodic = (self.L is not None, self.L is not None, False)

    @classmethod
    def of(cls, geometry, *, n_images=CELL_IMAGES):
        """The closed-form field of a :class:`~dmipy_sim.geometry.MyelinatedCylinder` (one cylinder at the origin)
        or a :class:`~dmipy_sim.geometry.PackedMyelinatedCylinders` (its periodic cell with ``n_images`` images),
        in the geometry's frame (axis = z)."""
        from ..geometry.myelin import MyelinatedCylinder, PackedMyelinatedCylinders
        if isinstance(geometry, MyelinatedCylinder):
            return cls(np.zeros((1, 2)), [geometry.inner_radius], [geometry.outer_radius])
        if isinstance(geometry, PackedMyelinatedCylinders):
            N = geometry.N_actual
            return cls(np.asarray(geometry._centers_np[:N], float), np.asarray(geometry._inner_radii_np[:N], float),
                       np.asarray(geometry._outer_radii_np[:N], float), L=float(geometry._L_float), n_images=n_images)
        raise TypeError(f"HollowCylinderField.of takes a MyelinatedCylinder or PackedMyelinatedCylinders, got "
                        f"{type(geometry).__name__}")

    @property
    def meta(self):
        """What a walk and a pack record about this field source (JSON-ready); :meth:`from_meta` rebuilds it."""
        return dict(kind="hollow_cylinder", centers=self.centers.tolist(), inner_radii=self.inner_radii.tolist(),
                    outer_radii=self.outer_radii.tolist(), L=self.L, n_images=self.n_images, channels=list(CHANNEL_NAMES),
                    mean_subtracted=False, periodic=list(self.periodic))

    @classmethod
    def from_meta(cls, meta):
        return cls(meta["centers"], meta["inner_radii"], meta["outer_radii"], L=meta.get("L"), n_images=meta.get("n_images", CELL_IMAGES))

    def device_read(self):
        """``(arrays, read, key)``: the device arrays a walk passes as arguments, ``read(arrays, r) -> (12,)`` at one
        position (traceable, float32, elementwise: no matmul touches a position), and the key a compiled walk is
        cached on."""
        import jax.numpy as jnp
        from dmipy_analytic.jax.fields import hollow_cylinder_basis
        if self.L is not None:                                    # the cell's periodic images
            ims = np.arange(-self.n_images, self.n_images + 1) * self.L
            ox, oy = np.meshgrid(ims, ims)
            offs = np.stack([ox.ravel(), oy.ravel()], axis=1)
        else:                                                     # an isolated cylinder: itself alone
            offs = np.zeros((1, 2))
        arrays = (jnp.asarray(self.centers, jnp.float32), jnp.asarray(self.inner_radii, jnp.float32),
                  jnp.asarray(self.outer_radii, jnp.float32), jnp.asarray(offs, jnp.float32),
                  jnp.float32(self.L if self.L is not None else 0.0))
        wrap = self.L is not None

        def read(arrays, r):
            c, a, b, o, L = arrays
            x, y = r[0], r[1]
            if wrap:
                x = ((x + 0.5 * L) % L) - 0.5 * L
                y = ((y + 0.5 * L) % L) - 0.5 * L
            dx = x - (c[:, 0:1] + o[None, :, 0])                                  # (N, M)
            dy = y - (c[:, 1:2] + o[None, :, 1])
            rho_vec = jnp.stack([dx, dy, jnp.zeros_like(dx)], axis=-1)
            u = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0], jnp.float32), rho_vec.shape)
            C = hollow_cylinder_basis(rho_vec, u, jnp.broadcast_to(a[:, None], dx.shape), jnp.broadcast_to(b[:, None], dx.shape))
            return jnp.sum(C, axis=(0, 1))
        key = ("hollow_cylinder", self.centers.shape[0], offs.shape[0], wrap)
        return arrays, read, key

    def channels(self, points):
        """``(n, 12)`` basis channels at ``points`` ``(n, 3)`` (metres), the device read evaluated on the host."""
        import jax
        arrays, read, _ = self.device_read()
        P = np.asarray(points, np.float32).reshape(-1, 3)
        return np.asarray(jax.jit(jax.vmap(lambda r: read(arrays, r)))(P), np.float64)


def __getattr__(name):
    from .._moved import refuse
    raise refuse(__name__, name)
