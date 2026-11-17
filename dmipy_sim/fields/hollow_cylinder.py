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

#: The rows ``j = 1 .. LATTICE_ROWS`` either side of the real axis that :func:`lattice_regular_part` sums. A row pair
#: contributes at most ``8 e^pi e^(-2 pi j)`` times the ``j = 0`` row's scale ``(pi / L)^2`` anywhere in the cell
#: (``|Im x| <= pi / 2``); the first omitted pair, ``j = 7``, is ``1.4e-17``, below float64's unit roundoff ``2^-53``
#: relative to the row's ``1 / 3``, and ``j = 6`` is the smallest count for which that holds.
LATTICE_ROWS = 6

#: The Taylor coefficients of ``1 / sin^2 x - 1 / x^2`` in powers of ``x^2`` (``c_n = (2n - 1) (-1)^(n+1) 2^(2n)
#: B_(2n) / (2n)!`` with the Bernoulli numbers ``B_(2n)``, ``n = 1 .. 6``; the term ``n = 0`` is the ``1 / x^2``
#: subtracted). DLMF 4.19.6 (the Laurent series of ``csc^2`` via ``cot``: ``csc^2 x = -d/dx cot x``).
SERIES = (1.0 / 3.0, 1.0 / 15.0, 2.0 / 189.0, 1.0 / 675.0, 2.0 / 10395.0, 1382.0 / 58046625.0)

#: ``|x|`` below which :func:`lattice_regular_part` evaluates ``1 / sin^2 x - 1 / x^2`` by :data:`SERIES` rather than
#: by the difference, per precision: the largest ``|x|`` (rounded down) at which the first omitted term,
#: ``4 / 1403325 |x|^12``, is at most the format's unit roundoff times the first term ``1 / 3`` (float32 ``0.661``,
#: float64 ``0.124``). The difference loses ``u / |x|^2`` relative to cancellation, so the switch is as far out as the
#: series allows.
SERIES_SWITCH = {"float32": 0.6, "float64": 0.12}


def lattice_regular_part(z, L, *, rows=LATTICE_ROWS):
    """``W(z) = P(z) - 1 / z^2``: the regular part at the origin of the Weierstrass function of the square lattice
    ``{(m + i n) L}``, ``P(z) = 1 / z^2 + sum'_w [1 / (z - w)^2 - 1 / w^2]``, at complex ``z`` (metres) in the cell
    ``|Re z|, |Im z| <= L / 2``; traceable (jax.numpy), in the precision of ``z`` (complex64 or complex128).

    The row-summed form (DLMF 23.8.3, half-periods ``omega_1 = L / 2``, ``omega_3 = i L / 2``)::

        P(z) = (pi / L)^2 sum_(j = -inf..inf) csc^2(pi (z - i j L) / L) - eta_1 / omega_1,   eta_1 / omega_1 = pi / L^2,

    each row ``sum_m 1 / (z - m L - i j L)^2 = (pi / L)^2 csc^2(pi (z - i j L) / L)`` summed in closed form. The
    constant: the square lattice is invariant under ``z -> i z``, so ``zeta(i z) = -i zeta(z)`` and ``eta_3 = -i
    eta_1``; Legendre's relation ``eta_1 omega_3 - eta_3 omega_1 = i pi / 2`` (DLMF 23.2.14) then gives ``eta_1 = pi /
    (4 omega_1) = pi / (2 L)``. Equivalently, the rows summed one after another give Eisenstein's ordered sum, which
    exceeds the symmetric one by ``G_2 = pi / L^2`` (``E_2(i) = 3 / pi``). By the same symmetry ``sum'_w w^-2 = 0`` over
    any square window, so ``P`` is also the limit of ``sum_w 1 / (z - w)^2`` over square windows about the origin.

    Evaluated as ``(pi / L)^2 [g(x) + sum_(j = 1..rows) (csc^2(x - i pi j) + csc^2(x + i pi j))] - pi / L^2`` with
    ``x = pi z / L`` and ``g(x) = csc^2 x - 1 / x^2``, regular at ``x = 0`` (``g(0) = 1 / 3``), by :data:`SERIES` for
    ``|x| <`` :data:`SERIES_SWITCH` and by the difference beyond. The omitted rows are below :data:`LATTICE_ROWS`'s
    bound. ``W`` is analytic in the open disc ``|z| < L`` and doubly periodic up to the pole it removes:
    ``W(z) + 1 / z^2`` is ``L``- and ``i L``-periodic."""
    import jax
    import jax.numpy as jnp
    z = jnp.asarray(z)
    if not jnp.issubdtype(z.dtype, jnp.complexfloating):
        raise TypeError(f"lattice_regular_part takes a complex z (complex64 or complex128), got {z.dtype}")
    real = jnp.real(z).dtype
    L = jnp.asarray(L, real)
    k = jnp.pi / L
    x = k * z
    switch = SERIES_SWITCH[jnp.dtype(real).name]
    small = jnp.abs(x) < switch
    s = x * x
    series = SERIES[-1]
    for c in SERIES[-2::-1]:
        series = c + s * series
    xd = jnp.where(small, jnp.ones_like(x), x)                    # the difference's branch, kept finite where unused
    sd = jnp.sin(xd)
    g = jnp.where(small, series, 1.0 / (sd * sd) - 1.0 / (xd * xd))
    shift = jax.lax.complex(jnp.zeros((rows,), real), jnp.pi * jnp.arange(1, rows + 1, dtype=real))
    xs = x[..., None]
    sm, sp = jnp.sin(xs - shift), jnp.sin(xs + shift)
    rest = jnp.sum(1.0 / (sm * sm) + 1.0 / (sp * sp), axis=-1)
    return k * k * (g + rest) - jnp.pi / (L * L)


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
        or a :class:`~dmipy_sim.geometry.PackedMyelinatedCylinders` (the sheaths of its myelinated cylinders in its
        periodic cell, with ``n_images`` images; a plain cylinder has no sheath and is no source), in the geometry's
        frame (axis = z). A cell without a myelinated cylinder has no field and is refused by name."""
        from ..geometry.myelin import MyelinatedCylinder, PackedMyelinatedCylinders
        if isinstance(geometry, MyelinatedCylinder):
            return cls(np.zeros((1, 2)), [geometry.inner_radius], [geometry.outer_radius])
        if isinstance(geometry, PackedMyelinatedCylinders):
            N, sh = geometry.N_actual, geometry.sheathed
            if not sh.any():
                raise ValueError("HollowCylinderField.of: this PackedMyelinatedCylinders cell has no myelinated cylinder "
                                 "(every g-ratio is 1), so no sheath is a field source")
            return cls(np.asarray(geometry._centers_np[:N], float)[sh], np.asarray(geometry._inner_radii_np[:N], float)[sh],
                       np.asarray(geometry._outer_radii_np[:N], float)[sh], L=float(geometry._L_float), n_images=n_images)
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

    def periodic_read(self, dtype=None):
        """``(arrays, read, key)`` of the exactly periodic field of a cell (``L`` set), in the precision ``dtype``
        (``jnp.float32``, the default, or ``jnp.float64`` under ``jax_enable_x64``): ``read(arrays, r) -> (12,)`` at
        one position ``r`` (metres; traceable, computed in the arrays' precision), and the key a compiled walk is
        cached on. Per source ``k`` and the offset ``d_k`` from its nearest image (``((r - c_k + L / 2) mod L) - L /
        2``), with ``z_k = d_k,x + i d_k,y``::

            F(r) = sum_k [ hollow_cylinder_basis(d_k; a_k, b_k) + T_k(W(z_k)) ],

        ``W`` :func:`lattice_regular_part` and ``T_k`` the tail of every other image: ``alpha_P,k (Re W, -Re W, -Im
        W)`` on ``iso_P_{xx, yy, xy}`` and ``alpha_A,k (Re W, -Re W, -Im W)`` on ``aniso_G_{xx, yy, xy}``, ``alpha_P =
        -(b^2 - a^2) / 2``, ``alpha_A = (b^2 - a^2) / 12`` (outside its sheath one cylinder's field is ``alpha S /
        rho^2`` with ``S_xx / rho^2 = Re(1 / z^2)``, ``S_xy / rho^2 = -Im(1 / z^2)``; no other channel has a tail).
        The closed form at the nearest image carries its own ``alpha / z^2`` outside the sheath and is exact inside
        it, so one expression holds in every region and the walls are the closed form's.

        This is the limit of the closed form summed over square windows of images about the cell: the windows' sum of
        ``1 / (z - w)^2`` tends to the Weierstrass ``P``, with an error ``O(n^-2)`` in the window's half-width ``n``.
        It is ``L``-periodic in ``x`` and ``y`` exactly and harmonic outside the sources. The cell mean is not
        subtracted: it is the mean of the sheaths' own non-tail terms (over the cell, the tails and ``W`` integrate to
        zero by the square's fourfold symmetry), with ``A = sum_k pi (b_k^2 - a_k^2)`` the sheaths' area and ``I_k =
        (b_k^2 - a_k^2) / 4 - (a_k^2 / 2) ln(b_k / a_k)``::

            mean(iso_local) = A / (3 L^2)        mean(iso_P_xx) = mean(iso_P_yy) = A / (2 L^2)
            mean(aniso_G_zz) = -A / (9 L^2)
            mean(aniso_G_xx) = mean(aniso_G_yy) = (1 / L^2) sum_k [ (pi a_k^2 / 2) ln(b_k / a_k) + pi I_k
                                                                    - (5 / 18) pi (b_k^2 - a_k^2) ]

        and zero in every other channel: the field of a long cylindrical sample whose cross-section is the cell (the
        cylindrical Lorentz convention), a uniform offset against the zero-mean convention of a k-space route.

        A sheath whose outer radius reaches ``L / 2`` does not fit its nearest-image cell and is refused by name."""
        import jax
        import jax.numpy as jnp
        from dmipy_analytic.jax.fields import hollow_cylinder_basis
        if self.L is None:
            raise ValueError("periodic_read is the field of a periodic cell: this HollowCylinderField has no cell side L")
        if not np.all(self.outer_radii < 0.5 * self.L):
            raise ValueError(f"HollowCylinderField: an outer radius ({self.outer_radii.max():.6g} m) reaches half the cell's "
                             f"side ({0.5 * self.L:.6g} m): the sheath does not fit its nearest-image cell")
        dtype = jnp.dtype(jnp.float32 if dtype is None else dtype)
        if dtype.name not in SERIES_SWITCH:
            raise ValueError(f"periodic_read reads in float32 or float64, got {dtype.name}")
        if dtype.name == "float64" and not jax.config.read("jax_enable_x64"):
            raise ValueError(f"periodic_read in {dtype.name} needs jax_enable_x64")
        w = self.outer_radii ** 2 - self.inner_radii ** 2
        arrays = (jnp.asarray(self.centers, dtype), jnp.asarray(self.inner_radii, dtype), jnp.asarray(self.outer_radii, dtype),
                  jnp.asarray(-0.5 * w, dtype), jnp.asarray(w / 12.0, dtype), jnp.asarray(self.L, dtype))

        def read(arrays, r):
            c, a, b, tP, tA, L = arrays
            r = jnp.asarray(r).astype(c.dtype)
            d = ((r[:2][None, :] - c + 0.5 * L) % L) - 0.5 * L               # (N, 2): each source's nearest image
            rho_vec = jnp.concatenate([d, jnp.zeros_like(d[:, :1])], axis=-1)
            u = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0], c.dtype), rho_vec.shape)
            F = jnp.sum(hollow_cylinder_basis(rho_vec, u, a, b), axis=0)
            W = lattice_regular_part(jax.lax.complex(d[:, 0], d[:, 1]), L)
            pr, pi_ = jnp.sum(tP * W.real), jnp.sum(tP * W.imag)
            ar, ai = jnp.sum(tA * W.real), jnp.sum(tA * W.imag)
            z0 = jnp.zeros((), c.dtype)
            return F + jnp.stack([z0, pr, -pr, -pi_, z0, z0, ar, -ar, z0, -ai, z0, z0])
        key = ("hollow_cylinder_periodic", self.centers.shape[0], dtype.name)
        return arrays, read, key

    def periodic_channels(self, points, dtype=np.float32):
        """``(n, 12)`` channels of :meth:`periodic_read` at ``points`` ``(n, 3)`` (metres), read in ``dtype`` (the points
        rounded to it) on the host, returned in float64. Non-finite points are refused by name."""
        import jax
        dtype = np.dtype(dtype)
        P = np.asarray(points, np.float64).reshape(-1, 3)
        if not np.all(np.isfinite(P)):
            raise ValueError(f"HollowCylinderField.periodic_channels: {int((~np.isfinite(P).any(1)).sum())} points are not "
                             f"finite")
        arrays, read, _ = self.periodic_read(dtype)
        return np.asarray(jax.jit(jax.vmap(lambda r: read(arrays, r)))(P.astype(dtype)), np.float64)


def __getattr__(name):
    from .._moved import refuse
    raise refuse(__name__, name)
