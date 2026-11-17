"""The closed-form field of parallel hollow cylinders as a walk reads it: :class:`HollowCylinderField` sums
dmipy-analytic's field basis of one infinite hollow cylinder (:func:`dmipy_analytic.fields.hollow_cylinder_basis`, read through its JAX
twin :func:`dmipy_analytic.jax.fields.hollow_cylinder_basis`) over the cylinders at a walker's own position; in a
periodic cell, over every image of the square lattice exactly, through the Weierstrass function of the lattice
(:func:`lattice_regular_part`). The same closed form is the building block of the forward Bloch engine's myelin
provider (:class:`dmipy_sim.fields.susceptibility.MyelinSusceptibility`) and, per segment, of a strand substrate's
field (:mod:`dmipy_sim.fields.strand_field`).

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
    """The field basis of parallel hollow cylinders along ``z`` in closed form: what a walk reads at a walker's own
    position, exactly, where a raster smears the sheath's jump over a voxel. ``centers`` ``(N, 2)``, ``inner_radii`` /
    ``outer_radii`` ``(N,)`` in metres, finite, ``0 < a < b``; ``L`` the side of a periodic square cell (``None``: the
    cylinders alone). The same ``channels(points)`` protocol as
    :class:`~dmipy_sim.fields.susceptibility_field.FieldGrid` and :class:`~dmipy_sim.fields.strand_field.StrandFieldBasis`,
    and :meth:`device_read` for the walk.

    Without a cell the field is the sum of the cylinders' closed forms. In a cell of side ``L`` it is the exactly
    periodic field of the cylinders and all their images: per source ``k`` and the offset ``d_k`` from its nearest
    image (``((r - c_k + L / 2) mod L) - L / 2``), with ``z_k = d_k,x + i d_k,y``::

        F(r) = sum_k [ hollow_cylinder_basis(d_k; a_k, b_k) + T_k(W(z_k)) ],

    ``W`` :func:`lattice_regular_part` and ``T_k`` the tail of every other image: ``alpha_P,k (Re W, -Re W, -Im W)`` on
    ``iso_P_{xx, yy, xy}`` and ``alpha_A,k (Re W, -Re W, -Im W)`` on ``aniso_G_{xx, yy, xy}``, ``alpha_P = -(b^2 - a^2)
    / 2``, ``alpha_A = (b^2 - a^2) / 12`` (outside its sheath one cylinder's field is ``alpha S / rho^2`` with ``S_xx /
    rho^2 = Re(1 / z^2)``, ``S_xy / rho^2 = -Im(1 / z^2)``; no other channel has a tail). The closed form at the
    nearest image carries its own ``alpha / z^2`` outside the sheath and is exact inside it, so one expression holds in
    every region and the walls are the closed form's. A sheath whose outer radius reaches ``L / 2`` does not fit its
    nearest-image cell and is refused by name.

    It is the limit of the closed form summed over square windows of images about the cell (the windows' sum of ``1 /
    (z - w)^2`` tends to the Weierstrass ``P``; a window of ``n`` images per side is off by ``O(n^-2)``), ``L``-periodic
    in ``x`` and ``y`` exactly, and harmonic outside the sources.

    The cell mean is not subtracted: it is the mean of the sheaths' own non-tail terms (over the cell, the tails and
    ``W`` integrate to zero by the square's fourfold symmetry), with ``A = sum_k pi (b_k^2 - a_k^2)`` the sheaths' area
    and ``I_k = (b_k^2 - a_k^2) / 4 - (a_k^2 / 2) ln(b_k / a_k)``::

        mean(iso_local) = A / (3 L^2)        mean(iso_P_xx) = mean(iso_P_yy) = A / (2 L^2)
        mean(aniso_G_zz) = -A / (9 L^2)
        mean(aniso_G_xx) = mean(aniso_G_yy) = (1 / L^2) sum_k [ (pi a_k^2 / 2) ln(b_k / a_k) + pi I_k
                                                                - (5 / 18) pi (b_k^2 - a_k^2) ]

    and zero in every other channel: the field of a long cylindrical sample whose cross-section is the cell (the
    cylindrical Lorentz convention), a uniform offset against the zero-mean convention of a k-space route
    (``meta["mean_subtracted"]`` is ``False``)."""

    channel_names = CHANNEL_NAMES

    def __init__(self, centers, inner_radii, outer_radii, *, L=None):
        self.centers = np.asarray(centers, np.float64).reshape(-1, 2)
        self.inner_radii = np.asarray(inner_radii, np.float64).reshape(-1)
        self.outer_radii = np.asarray(outer_radii, np.float64).reshape(-1)
        if not (self.centers.shape[0] == self.inner_radii.shape[0] == self.outer_radii.shape[0]):
            raise ValueError("centers, inner_radii and outer_radii describe the same cylinders")
        for name, v in (("centers", self.centers), ("inner_radii", self.inner_radii), ("outer_radii", self.outer_radii)):
            if not np.all(np.isfinite(v)):
                raise ValueError(f"HollowCylinderField: {name} are not all finite")
        if not np.all((self.inner_radii > 0) & (self.outer_radii > self.inner_radii)):
            raise ValueError("every outer radius exceeds its inner radius, which is positive")
        if L is not None:
            L = float(L)
            if not (np.isfinite(L) and L > 0):
                raise ValueError(f"HollowCylinderField: the cell's side L is a positive finite length, got {L!r}")
            if not np.all(self.outer_radii < 0.5 * L):
                raise ValueError(f"HollowCylinderField: an outer radius ({self.outer_radii.max():.6g} m) reaches half the "
                                 f"cell's side ({0.5 * L:.6g} m): the sheath does not fit its nearest-image cell")
        self.L = L
        self.periodic = (self.L is not None, self.L is not None, False)

    @classmethod
    def of(cls, geometry):
        """The closed-form field of a :class:`~dmipy_sim.geometry.MyelinatedCylinder` (one cylinder at the origin)
        or a :class:`~dmipy_sim.geometry.PackedMyelinatedCylinders` (the sheaths of its myelinated cylinders in its
        periodic cell; a plain cylinder has no sheath and is no source), in the geometry's frame (axis = z). A cell
        without a myelinated cylinder has no field and is refused by name."""
        from ..geometry.myelin import MyelinatedCylinder, PackedMyelinatedCylinders
        if isinstance(geometry, MyelinatedCylinder):
            return cls(np.zeros((1, 2)), [geometry.inner_radius], [geometry.outer_radius])
        if isinstance(geometry, PackedMyelinatedCylinders):
            N, sh = geometry.N_actual, geometry.sheathed
            if not sh.any():
                raise ValueError("HollowCylinderField.of: this PackedMyelinatedCylinders cell has no myelinated cylinder "
                                 "(every g-ratio is 1), so no sheath is a field source")
            return cls(np.asarray(geometry._centers_np[:N], float)[sh], np.asarray(geometry._inner_radii_np[:N], float)[sh],
                       np.asarray(geometry._outer_radii_np[:N], float)[sh], L=float(geometry._L_float))
        raise TypeError(f"HollowCylinderField.of takes a MyelinatedCylinder or PackedMyelinatedCylinders, got "
                        f"{type(geometry).__name__}")

    @property
    def summation(self):
        """How the field sums the cylinders' images: ``"lattice"`` (a periodic cell: every image, exactly, by
        :func:`lattice_regular_part` with :data:`LATTICE_ROWS` rows) or ``"none"`` (the cylinders alone)."""
        return "none" if self.L is None else "lattice"

    @property
    def meta(self):
        """What a walk and a pack record about this field source (JSON-ready); :meth:`from_meta` rebuilds it."""
        m = dict(kind="hollow_cylinder", centers=self.centers.tolist(), inner_radii=self.inner_radii.tolist(),
                 outer_radii=self.outer_radii.tolist(), L=self.L, summation=self.summation, channels=list(CHANNEL_NAMES),
                 mean_subtracted=False, periodic=list(self.periodic))
        if self.L is not None:
            m["lattice_rows"] = LATTICE_ROWS
        return m

    @classmethod
    def from_meta(cls, meta):
        """The field a record describes. A record of a periodic cell that names ``n_images`` was walked with the
        closed form summed over a window of images, which this field is not (it differs by up to 4.4e-4 of the field
        at six images per side), and is refused by name; a record of cylinders without a cell reads as recorded."""
        L = meta["L"]
        if L is not None:
            if "n_images" in meta:
                raise ValueError(f"this field record was walked with the image-sum field ({meta['n_images']} images per "
                                 f"side); the periodic field differs from it by up to 4.4e-4 of the field at 6 images: "
                                 f"re-walk")
            if meta.get("summation") != "lattice" or meta.get("lattice_rows") != LATTICE_ROWS:
                raise ValueError(f"this field record states summation {meta.get('summation')!r} with "
                                 f"{meta.get('lattice_rows')!r} rows; this revision reads the lattice sum with "
                                 f"{LATTICE_ROWS}: re-walk")
        return cls(meta["centers"], meta["inner_radii"], meta["outer_radii"], L=L)

    def device_read(self, dtype=None):
        """``(arrays, read, key)``: the device arrays a walk passes as arguments, ``read(arrays, r) -> (12,)`` at one
        position ``r`` (metres; traceable, elementwise: no matmul touches a position, computed in the arrays'
        precision), and the key a compiled walk is cached on. ``dtype`` is ``jnp.float32`` (the default: the walk's
        precision) or ``jnp.float64`` (the reference evaluation of the same formula, under ``jax_enable_x64``)."""
        import jax
        import jax.numpy as jnp
        from dmipy_analytic.jax.fields import hollow_cylinder_basis
        dtype = jnp.dtype(jnp.float32 if dtype is None else dtype)
        if dtype.name not in SERIES_SWITCH:
            raise ValueError(f"HollowCylinderField reads in float32 or float64, got {dtype.name}")
        if dtype.name == "float64" and not jax.config.read("jax_enable_x64"):
            raise ValueError("HollowCylinderField.device_read in float64 needs jax_enable_x64")
        w = self.outer_radii ** 2 - self.inner_radii ** 2
        periodic = self.L is not None
        arrays = (jnp.asarray(self.centers, dtype), jnp.asarray(self.inner_radii, dtype), jnp.asarray(self.outer_radii, dtype),
                  jnp.asarray(-0.5 * w, dtype), jnp.asarray(w / 12.0, dtype), jnp.asarray(self.L if periodic else 0.0, dtype))

        def read(arrays, r):
            c, a, b, tP, tA, L = arrays
            r = jnp.asarray(r).astype(c.dtype)
            d = r[:2][None, :] - c
            if periodic:
                d = ((d + 0.5 * L) % L) - 0.5 * L                              # (N, 2): each source's nearest image
            rho_vec = jnp.concatenate([d, jnp.zeros_like(d[:, :1])], axis=-1)
            u = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0], c.dtype), rho_vec.shape)
            F = jnp.sum(hollow_cylinder_basis(rho_vec, u, a, b), axis=0)
            if not periodic:
                return F
            W = lattice_regular_part(jax.lax.complex(d[:, 0], d[:, 1]), L)
            pr, pi_ = jnp.sum(tP * W.real), jnp.sum(tP * W.imag)
            ar, ai = jnp.sum(tA * W.real), jnp.sum(tA * W.imag)
            z0 = jnp.zeros((), c.dtype)
            return F + jnp.stack([z0, pr, -pr, -pi_, z0, z0, ar, -ar, z0, -ai, z0, z0])
        key = ("hollow_cylinder", self.centers.shape[0], self.summation, dtype.name)
        return arrays, read, key

    def channels(self, points, dtype=np.float32):
        """``(n, 12)`` basis channels at ``points`` ``(n, 3)`` (metres): :meth:`device_read` in ``dtype`` (the points
        rounded to it) evaluated on the host, returned in float64. Non-finite points are refused by name."""
        import jax
        dtype = np.dtype(dtype)
        P = np.asarray(points, np.float64).reshape(-1, 3)
        if not np.all(np.isfinite(P)):
            raise ValueError(f"HollowCylinderField.channels: {int((~np.isfinite(P).all(1)).sum())} points are not finite")
        arrays, read, _ = self.device_read(dtype)
        return np.asarray(jax.jit(jax.vmap(lambda r: read(arrays, r)))(P.astype(dtype)), np.float64)


def __getattr__(name):
    from .._moved import refuse
    raise refuse(__name__, name)
