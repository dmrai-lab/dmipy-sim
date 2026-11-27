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

from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
from threading import Lock

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

#: The number of cylinders from which a periodic cell is read near / far by default (``read=None``); below it, and
#: without a cell, the exact read.
NEAR_FAR_MIN_CYLINDERS = 512

#: The default tolerance of the near / far read: the largest remainder bound over the cells, per channel, is at most
#: this fraction of the exact field's rms in its largest channel (over :data:`NEAR_FAR_RMS_POINTS` points).
NEAR_FAR_TOLERANCE = 1e-6

#: The points (uniform in the cell, ``numpy.random.default_rng(0)``) over which the exact field's rms is taken.
NEAR_FAR_RMS_POINTS = 2048

#: The near width the order search starts at and the widest it reaches (cells): a cylinder is near a cell when its
#: outer wall is within this many cell sides of the cell.
NEAR_FAR_WIDTH = 2
NEAR_FAR_MAX_WIDTH = 4

#: The highest series order the order search tries at one near width.
NEAR_FAR_MAX_ORDER = 24

#: The far series' global part (:meth:`HollowCylinderField.near_far`): the images of a cylinder within
#: ``NEAR_FAR_EXPLICIT_RADIUS L`` of the cell's centre are summed as poles per cell, the rest as one function analytic
#: in that disc, sampled at ``NEAR_FAR_SAMPLES`` points on the circle of radius ``NEAR_FAR_SAMPLE_RADIUS L``.
NEAR_FAR_EXPLICIT_RADIUS = 1.2
NEAR_FAR_SAMPLE_RADIUS = 1.0
NEAR_FAR_SAMPLES = 512

#: The lattice rings summed one by one in the bound of the images other than the nearest (beyond them a closed-form
#: tail, :func:`image_tail_bound`).
NEAR_FAR_BOUND_RINGS = 16

#: The fraction of a cell side within which :meth:`HollowCylinderField.remainder_bound` counts a point as on the
#: neighbouring cell too (far above float64 rounding of a cell index, far below a cell).
NEAR_FAR_EDGE = 1e-7

#: The read names a :class:`HollowCylinderField` takes (``read=``) and the ``summation`` each records.
READS = {"exact": "lattice", "near_far": "lattice_near_far"}


_NEAR_FAR_CACHE_SIZE = 4
_near_far_cache = OrderedDict()
_near_far_lock = Lock()


def clear_near_far_cache():
    """Discard the process's cached hollow-cylinder near / far host tables."""
    with _near_far_lock:
        _near_far_cache.clear()


def _near_far_key(field):
    """Fingerprint the geometry and numerical settings of the float64 host tables."""
    arrays = (field.centers, field.inner_radii, field.outer_radii)
    settings = (
        NEAR_FAR_MAX_ORDER, NEAR_FAR_WIDTH, NEAR_FAR_MAX_WIDTH, NEAR_FAR_EXPLICIT_RADIUS,
        NEAR_FAR_SAMPLE_RADIUS, NEAR_FAR_SAMPLES, NEAR_FAR_BOUND_RINGS, NEAR_FAR_RMS_POINTS,
        LATTICE_ROWS, SERIES, tuple(sorted(SERIES_SWITCH.items())),
    )
    header = (tuple(a.shape for a in arrays), field.L.hex(), field.tolerance.hex(), field._forced, settings)
    digest = sha256(repr(header).encode("ascii"))
    for a in arrays:
        digest.update(a.tobytes())
    return digest.digest()


def series_tail(x, order):
    """``sum_(n > order) (n + 1) x^n = x^(order + 1) ((order + 2) - (order + 1) x) / (1 - x)^2`` for ``0 <= x < 1``
    (numpy, elementwise): the Taylor remainder of ``1 / (1 + u)^2`` at ``|u| <= x``, term by term in modulus."""
    x = np.asarray(x, np.float64)
    return x ** (order + 1) * ((order + 2) - (order + 1) * x) / (1.0 - x) ** 2


def image_tail_bound(rho, L, order, rings=NEAR_FAR_BOUND_RINGS):
    """``E = sum_(w != 0) series_tail(rho / d_w, order) / d_w^2`` with ``d_w = L hypot((|m| - 1/2)_+, (|n| - 1/2)_+)``
    the distance from the lattice point ``w = (m + i n) L`` to the square ``[-L/2, L/2]^2`` (m^-2): a bound, for any
    ``delta`` in that square, of ``sum_(w != 0) series_tail(rho / |delta - w|, order) / |delta - w|^2``.

    The rings ``max(|m|, |n|) <= rings`` are summed one by one; beyond, ring ``s`` holds ``8 s <= 16 (s - 1/2)``
    points at ``d >= (s - 1/2) L``, and ``h(d) = series_tail(rho / d) / d^2`` falls at least as ``d^-(order + 3)``
    (``series_tail(x) / x^(order + 1)`` increases with ``x``), so the rest is at most ``16 h(d0) [(S + 1/2) + (S +
    1/2)^2 / (order + 1)]``, ``d0 = (S + 1/2) L``. Needs ``rho < L / 2``."""
    if not rho < 0.5 * L:
        raise ValueError(f"image_tail_bound: rho {rho:.6g} m is not below half the cell side {0.5 * L:.6g} m")
    m = np.arange(-rings, rings + 1)
    M, Nn = np.meshgrid(m, m, indexing="ij")
    keep = (M != 0) | (Nn != 0)
    d = L * np.hypot(np.maximum(np.abs(M[keep]) - 0.5, 0.0), np.maximum(np.abs(Nn[keep]) - 0.5, 0.0))
    inner = float(np.sum(series_tail(rho / d, order) / d ** 2))
    S = rings + 0.5
    d0 = S * L
    h0 = float(series_tail(rho / d0, order)) / d0 ** 2
    return inner + 16.0 * h0 * (S + S * S / (order + 1))


def _host():
    """The context of the host build: float64 JAX on the CPU device."""
    import contextlib
    import jax
    stack = contextlib.ExitStack()
    stack.enter_context(jax.enable_x64(True))
    stack.enter_context(jax.default_device(jax.devices("cpu")[0]))
    return stack


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
    (``meta["mean_subtracted"]`` is ``False``).

    **Two reads of the same field** (``read=``). ``"exact"`` sums the formula above over every cylinder at every
    position. ``"near_far"`` (a periodic cell only) is the same field read as near cylinders plus a series:

    - *Analyticity.* Outside its outer wall a cylinder's channels are ``(0, tP Re f, -tP Re f, -tP Im f, 0, 0, tA Re
      f, -tA Re f, 0, -tA Im f, 0, 0)`` with ``f = 1 / z^2`` (the closed form's outside region), ``tP = -w / 2``,
      ``tA = w / 12``, ``w = b^2 - a^2``; with every image, ``f`` becomes ``P(z - c_k)`` (``P = 1 / z^2 + W`` the
      Weierstrass function), and the tail of a cylinder read at its nearest image is ``W``. So in a cell the whole
      non-local part is ``-S / 2`` on ``iso_P_{xx, yy}`` (``+`` on ``yy``), ``S / 2`` on ``iso_P_xy`` as ``Im``, and
      ``S / 12`` on ``aniso_G`` likewise, of ONE complex function ``S(z) = sum_far w_k P(z - c_k) + sum_near w_k W(z -
      c*_k)`` (``c*_k`` a near cylinder's nearest image), analytic in the grid cell: its only singularities are the
      double poles of every image of a far cylinder and every image but the nearest of a near one, all outside the
      cell. The two weights of a cylinder are one weight times ``-1/2`` and ``1/12``, so one series serves both.
    - *Grid and near set.* ``G = round(L / mean(2 b))`` cells per side of ``cs = L / G``, centres ``z_0``, half-diagonal
      ``rho = cs / sqrt 2``. A cylinder is near a cell when its outer wall comes within ``near_cells`` cell sides of
      the cell (distance from the square to its nearest image, less ``b``); the list is the cell's, ascending, read
      through the shared grid (:func:`~dmipy_sim.geometry._grid.gather` with the cell itself as the one offset). A
      near cylinder is the exact read's closed form at its nearest image (interior included, so walls are the closed
      form's); a far one is outside its wall everywhere in the cell, at distance ``d > rho`` from ``z_0``.
    - *Series.* ``S(z_0 + zeta) = sum_(n <= order) s_n (zeta / cs)^n``, Horner's rule; the coefficients are summed on
      the host in float64 over every cylinder and every image: the images within ``1.2 L`` of the cell's centre as
      double poles per cell (``w (n + 1) (-cs / delta)^n / delta^2``, ``delta = z_0 - pole``; a near cylinder's nearest
      image omitted), the rest as one function analytic in ``|z| < 1.2 L``, Taylor-expanded about the origin from
      :data:`NEAR_FAR_SAMPLES` samples on ``|z| = L`` (its coefficients' aliasing is below ``(1 / 1.2)^512``) and shifted
      to each centre.
    - *Remainder.* Each double pole at distance ``d`` contributes ``w (n + 1) zeta^n / d^(n + 2)`` in modulus to
      order ``n``, so the truncation of ``S`` at ``|zeta| <= rho`` is at most::

          B(z_0) = sum_far |w_k| series_tail(rho / d_k, order) / d_k^2 + sum_all |w_k| E(order),

      ``d_k`` the distance from ``z_0`` to far cylinder ``k``'s nearest image, :func:`series_tail` the geometric tail
      ``sum_(n > order) (n + 1) x^n``, and ``E`` (:func:`image_tail_bound`) the same over every other image, bounded
      by its distance to the square ``[-L/2, L/2]^2``. A channel's error is at most ``B / 2`` (``iso_P``) or ``B /
      12`` (``aniso_G``) and zero in the six local channels (:meth:`remainder_bound`).
    - *Order rule.* ``order`` is the smallest whose largest ``B / 2`` over the cells is at most ``tolerance`` times
      the exact field's rms in its largest channel (:data:`NEAR_FAR_RMS_POINTS` uniform points); if no order up to
      :data:`NEAR_FAR_MAX_ORDER` meets it at :data:`NEAR_FAR_WIDTH` cells the near width grows by one, and beyond
      :data:`NEAR_FAR_MAX_WIDTH` the build is refused by name with the bounds it reached. ``order`` and ``near_cells``
      force either (both: no search; the record states the bound whatever it is).

    ``read=None`` is ``"near_far"`` in a periodic cell of at least :data:`NEAR_FAR_MIN_CYLINDERS` cylinders and
    ``"exact"`` otherwise. A read costs the near list (about 30 closed forms) plus ``order`` complex products,
    whatever the number of cylinders."""

    channel_names = CHANNEL_NAMES

    def __init__(self, centers, inner_radii, outer_radii, *, L=None, read=None, order=None, near_cells=None,
                 tolerance=NEAR_FAR_TOLERANCE):
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
        if read is None:
            read = "exact" if (L is None or self.centers.shape[0] < NEAR_FAR_MIN_CYLINDERS) else "near_far"
        if read not in READS:
            raise ValueError(f"HollowCylinderField: read is None (the stated rule), 'exact' or 'near_far'; got {read!r}")
        if read == "near_far" and L is None:
            raise ValueError("HollowCylinderField(read='near_far') is refused: the near / far read is the series of a "
                             "periodic cell, and these cylinders have no cell (L is None)")
        if read == "exact" and (order is not None or near_cells is not None):
            raise ValueError("HollowCylinderField: order and near_cells set the near / far read; the exact read takes "
                             "neither")
        if order is not None and not (isinstance(order, (int, np.integer)) and 0 <= order):
            raise ValueError(f"HollowCylinderField: order is a non-negative integer, got {order!r}")
        if near_cells is not None and not (isinstance(near_cells, (int, np.integer)) and 1 <= near_cells):
            raise ValueError(f"HollowCylinderField: near_cells is a positive integer, got {near_cells!r}")
        tolerance = float(tolerance)
        if not (np.isfinite(tolerance) and tolerance > 0):
            raise ValueError(f"HollowCylinderField: tolerance is a positive fraction, got {tolerance!r}")
        self.read = read
        self._forced = (None if order is None else int(order), None if near_cells is None else int(near_cells))
        self.tolerance = tolerance

    @classmethod
    def of(cls, geometry, **read_options):
        """The closed-form field of a :class:`~dmipy_sim.geometry.MyelinatedCylinder` (one cylinder at the origin)
        or a :class:`~dmipy_sim.geometry.PackedMyelinatedCylinders` (the sheaths of its myelinated cylinders in its
        periodic cell; a plain cylinder has no sheath and is no source), in the geometry's frame (axis = z). A cell
        without a myelinated cylinder has no field and is refused by name. ``read_options`` (``read``, ``order``,
        ``near_cells``, ``tolerance``) are the constructor's."""
        from ..geometry.myelin import MyelinatedCylinder, PackedMyelinatedCylinders
        if isinstance(geometry, MyelinatedCylinder):
            return cls(np.zeros((1, 2)), [geometry.inner_radius], [geometry.outer_radius], **read_options)
        if isinstance(geometry, PackedMyelinatedCylinders):
            N, sh = geometry.N_actual, geometry.sheathed
            if not sh.any():
                raise ValueError("HollowCylinderField.of: this PackedMyelinatedCylinders cell has no myelinated cylinder "
                                 "(every g-ratio is 1), so no sheath is a field source")
            return cls(np.asarray(geometry._centers_np[:N], float)[sh], np.asarray(geometry._inner_radii_np[:N], float)[sh],
                       np.asarray(geometry._outer_radii_np[:N], float)[sh], L=float(geometry._L_float), **read_options)
        raise TypeError(f"HollowCylinderField.of takes a MyelinatedCylinder or PackedMyelinatedCylinders, got "
                        f"{type(geometry).__name__}")

    @property
    def summation(self):
        """How the field sums the cylinders' images: ``"lattice"`` (a periodic cell, the exact read: every image,
        exactly, by :func:`lattice_regular_part` with :data:`LATTICE_ROWS` rows), ``"lattice_near_far"`` (a periodic
        cell, the near / far read: the near cylinders' closed form plus the per-cell series of the rest,
        :meth:`near_far`) or ``"none"`` (the cylinders alone)."""
        return "none" if self.L is None else READS[self.read]

    @property
    def meta(self):
        """What a walk and a pack record about this field source (JSON-ready); :meth:`from_meta` rebuilds it. The near
        / far read adds ``series_order``, ``near_cells``, ``series_cells_per_side``, ``series_list_length``,
        ``series_tolerance`` (relative), ``series_tolerance_abs`` and ``series_bound_max`` (the largest remainder bound
        over the cells in any channel, field units per unit susceptibility) and ``series_table_bytes`` (the float32
        read's near table and coefficients)."""
        m = dict(kind="hollow_cylinder", centers=self.centers.tolist(), inner_radii=self.inner_radii.tolist(),
                 outer_radii=self.outer_radii.tolist(), L=self.L, summation=self.summation, channels=list(CHANNEL_NAMES),
                 mean_subtracted=False, periodic=list(self.periodic))
        if self.L is not None:
            m["lattice_rows"] = LATTICE_ROWS
        if self.read == "near_far":
            nf = self.near_far()
            m.update(series_order=nf["order"], near_cells=nf["near_cells"], series_cells_per_side=nf["cells_per_side"],
                     series_list_length=nf["list_length"], series_tolerance=self.tolerance,
                     series_tolerance_abs=nf["tolerance_abs"], series_bound_max=nf["bound_max"],
                     series_table_bytes=nf["table_bytes"])
        return m

    @classmethod
    def from_meta(cls, meta, *, read=None):
        """The field a record describes, read as it was recorded. A record of a periodic cell that names ``n_images``
        was walked with the closed form summed over a window of images, which this field is not (it differs by up to
        4.4e-4 of the field at six images per side), and is refused by name; so is a record whose ``summation`` or
        ``lattice_rows`` this revision does not read. ``read`` (``"exact"`` / ``"near_far"``) states the read the caller
        expects: a record written with the other read is refused by name. A near / far record is rebuilt at its
        recorded order, near width and tolerance, and refused by name unless the rebuild states the recorded cells per
        side, list length and table bytes exactly and the recorded largest bound to a relative 1e-9 (the last bits of
        a float64 sum can differ between hosts). A record of cylinders without a cell reads as recorded."""
        L = meta["L"]
        if L is None:
            if read == "near_far":
                raise ValueError("this field record has no periodic cell (L is None): it was written with the cylinders "
                                 "alone, and read='near_far' is the series of a periodic cell")
            return cls(meta["centers"], meta["inner_radii"], meta["outer_radii"], L=None)
        if "n_images" in meta:
            raise ValueError(f"this field record was walked with the image-sum field ({meta['n_images']} images per "
                             f"side); the periodic field differs from it by up to 4.4e-4 of the field at 6 images: "
                             f"re-walk")
        recorded = {v: k for k, v in READS.items()}.get(meta.get("summation"))
        if recorded is None or meta.get("lattice_rows") != LATTICE_ROWS:
            raise ValueError(f"this field record states summation {meta.get('summation')!r} with "
                             f"{meta.get('lattice_rows')!r} rows; this revision reads the lattice sum with "
                             f"{LATTICE_ROWS} rows as {sorted(READS.values())}: re-walk")
        if read is not None and read != recorded:
            raise ValueError(f"this field record was written with the {recorded!r} read (summation "
                             f"{meta['summation']!r}); read={read!r} is the other read, which differs from it: re-walk "
                             f"or read it as recorded")
        if recorded == "exact":
            return cls(meta["centers"], meta["inner_radii"], meta["outer_radii"], L=L, read="exact")
        f = cls(meta["centers"], meta["inner_radii"], meta["outer_radii"], L=L, read="near_far",
                order=int(meta["series_order"]), near_cells=int(meta["near_cells"]),
                tolerance=float(meta["series_tolerance"]))
        got = f.meta
        same = all(got[k] == meta[k] for k in ("series_cells_per_side", "series_list_length", "series_table_bytes"))
        if not (same and np.isclose(got["series_bound_max"], meta["series_bound_max"], rtol=1e-9, atol=0.0)):
            keys = ("series_cells_per_side", "series_list_length", "series_table_bytes", "series_bound_max")
            raise ValueError(f"this near / far field record states {({k: meta[k] for k in keys})}; this revision builds "
                             f"{({k: got[k] for k in keys})} at the same order and near width: re-walk")
        return f

    def near_far(self):
        """The near / far read's host tables, reused across fields with the same inputs in this process.

        Returns a fresh dict with read-only array views and independent copies of the width search record. It contains
        ``order``, ``near_cells``, ``cells_per_side`` ``G``, ``cell_size`` ``cs`` (m), ``half_diagonal`` ``rho`` (m),
        ``list_length`` ``M``,
        ``near`` ``(G * G, M)`` int32 (each cell's near cylinders, ascending, ``-1`` padding), ``coefficients`` ``(G *
        G, order + 1)`` complex128 (the series of ``S`` in ``zeta / cs``), ``bound`` ``(G * G,)`` float64 (the
        remainder bound of ``S`` per cell), ``rms`` (the exact field's rms per channel), ``tolerance_abs``,
        ``bound_max`` (the largest channel bound, ``max(bound) / 2``), ``table_bytes``, ``widths`` (the record of every
        near width the order search tried: its list length and the largest channel bound at every order) and the
        global part's ``explicit_poles``. Refused by name for the exact read; see the class docstring for what it is."""
        if self.read != "near_far":
            raise ValueError(f"HollowCylinderField.near_far: this field's read is {self.read!r}")
        with _near_far_lock:
            key = _near_far_key(self)
            if key not in _near_far_cache:
                _near_far_cache[key] = self._near_far_uncached()
                if len(_near_far_cache) > _NEAR_FAR_CACHE_SIZE:
                    _near_far_cache.popitem(last=False)
            _near_far_cache.move_to_end(key)
            tables = _near_far_cache[key]
        return {k: v.view() if isinstance(v, np.ndarray) else deepcopy(v) for k, v in tables.items()}

    def _near_far_uncached(self):
        """Build independent read-only host tables for a near / far field."""
        tables = _build_near_far(self)
        for key, value in tables.items():
            if isinstance(value, np.ndarray):
                # Immutable backing also prevents callers from re-enabling writes or editing a base array.
                tables[key] = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
        return tables

    def remainder_bound(self, points):
        """``(n, 12)`` float64: the near / far read's remainder bound per channel at ``points`` ``(n, 2 or 3)``
        (metres): ``bound / 2`` on ``iso_P_{xx, yy, xy}``, ``bound / 12`` on ``aniso_G_{xx, yy, xy}``, 0 on the six
        channels the near cylinders' closed form carries alone, with ``bound`` the largest over the cells whose square,
        widened by :data:`NEAR_FAR_EDGE` of a cell side, holds the point (wrapped into the cell): a point on a cell's
        edge or corner is read in whichever cell the read's rounding of ``floor((p - gmin) / cs)`` picks, and the
        compiled read may round differently from any host arithmetic. The bound holds for the float64 read up to its
        rounding."""
        nf = self.near_far()
        P = np.asarray(points, np.float64)
        P = P.reshape(-1, P.shape[-1])[:, :2]
        L, G, cs = self.L, nf["cells_per_side"], nf["cell_size"]
        u = np.mod(P + 0.5 * L, L) / cs
        B = np.zeros(P.shape[0])
        for sx in (-NEAR_FAR_EDGE, NEAR_FAR_EDGE):
            for sy in (-NEAR_FAR_EDGE, NEAR_FAR_EDGE):
                ij = np.mod(np.floor(u + np.array([sx, sy])).astype(np.int64), G)
                B = np.maximum(B, nf["bound"][ij[:, 0] * G + ij[:, 1]])
        out = np.zeros((P.shape[0], 12))
        out[:, [1, 2, 3]] = (B / 2.0)[:, None]
        out[:, [6, 7, 9]] = (B / 12.0)[:, None]
        return out

    def device_read(self, dtype=None):
        """``(arrays, read, key)``: the device arrays a walk passes as arguments, ``read(arrays, r) -> (12,)`` at one
        position ``r`` (metres; traceable, elementwise: no matmul touches a position, computed in the arrays'
        precision), and the key a compiled walk is cached on. ``dtype`` is ``jnp.float32`` (the default: the walk's
        precision) or ``jnp.float64`` (the reference evaluation of the same formula, under ``jax_enable_x64``). The read
        is :attr:`read`'s: the exact sum over every cylinder, or the near cylinders plus the cell's series."""
        import jax
        import jax.numpy as jnp
        dtype = jnp.dtype(jnp.float32 if dtype is None else dtype)
        if dtype.name not in SERIES_SWITCH:
            raise ValueError(f"HollowCylinderField reads in float32 or float64, got {dtype.name}")
        if dtype.name == "float64" and not jax.config.read("jax_enable_x64"):
            raise ValueError("HollowCylinderField.device_read in float64 needs jax_enable_x64")
        if self.read == "near_far":
            return _near_far_read(self, dtype)
        return _exact_read(self, dtype)

    def device_arrays(self, dtype=None):
        """The six arrays :meth:`device_read` passes to the walk for the exact read, as read-only host arrays in
        ``dtype`` (``float32`` by default, the walk's): ``centers`` ``(N, 2)``, ``inner_radii`` and ``outer_radii``
        ``(N,)`` (m), the tail weights ``-(b^2 - a^2) / 2`` and ``(b^2 - a^2) / 12`` ``(N,)`` (m^2) of ``iso_P`` and
        ``aniso_G``, and the cell's side ``L`` ``()`` (m; 0 for the cylinders alone, :attr:`L` None). The field at a
        position is :meth:`device_read`'s ``read`` of these; the channels are :attr:`channel_names`, in that order. A
        field whose read is the near / far one is refused by name: these arrays describe the exact read, which is not
        the read its record states."""
        if self.read != "exact":
            raise ValueError("HollowCylinderField.device_arrays is the exact read's description; this field's read is "
                             "'near_far' (summation 'lattice_near_far'): build it with read='exact' for a reader of "
                             "these arrays")
        arrays, _, _ = self.device_read(dtype)
        out = []
        for a in arrays:
            a = np.array(a)
            a.flags.writeable = False
            out.append(a)
        return tuple(out)

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


def _exact_read(field, dtype):
    """:meth:`HollowCylinderField.device_read` of the exact read: every cylinder at its nearest image, plus every
    other image's tail through :func:`lattice_regular_part` in a cell."""
    import jax
    import jax.numpy as jnp
    from dmipy_analytic.jax.fields import hollow_cylinder_basis
    w = field.outer_radii ** 2 - field.inner_radii ** 2
    periodic = field.L is not None
    arrays = (jnp.asarray(field.centers, dtype), jnp.asarray(field.inner_radii, dtype),
              jnp.asarray(field.outer_radii, dtype), jnp.asarray(-0.5 * w, dtype), jnp.asarray(w / 12.0, dtype),
              jnp.asarray(field.L if periodic else 0.0, dtype))

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
    key = ("hollow_cylinder", field.centers.shape[0], field.summation, jnp.dtype(dtype).name)
    return arrays, read, key


def _near_far_read(field, dtype):
    """:meth:`HollowCylinderField.device_read` of the near / far read: the position wrapped into the cell, its grid
    cell found (:func:`~dmipy_sim.geometry._grid.cell_index`), the cell's near cylinders gathered
    (:func:`~dmipy_sim.geometry._grid.gather` with the cell itself as the one offset) and summed in closed form at
    their nearest image as the exact read sums them, then the cell's series of ``S`` by Horner's rule in ``zeta / cs``
    added as ``-S / 2`` on ``iso_P`` and ``S / 12`` on ``aniso_G``."""
    import jax
    import jax.numpy as jnp
    from dmipy_analytic.jax.fields import hollow_cylinder_basis
    from ..geometry._grid import cell_index, gather, wrap_periodic
    nf = field.near_far()
    G, order, M = nf["cells_per_side"], nf["order"], nf["list_length"]
    cdt = jnp.complex64 if jnp.dtype(dtype).name == "float32" else jnp.complex128
    gmin = -0.5 * field.L
    arrays = (jnp.asarray(field.centers, dtype), jnp.asarray(field.inner_radii, dtype),
              jnp.asarray(field.outer_radii, dtype), jnp.asarray(field.L, dtype), jnp.asarray(nf["near"], jnp.int32),
              jnp.asarray(nf["coefficients"], cdt), jnp.asarray(gmin, dtype), jnp.asarray(nf["cell_size"], dtype),
              jnp.asarray(1.0 / nf["cell_size"], dtype))
    OFF = jnp.zeros((1, 3), jnp.int32)
    DIMS = jnp.asarray([G, G, 1], jnp.int32)
    PER = jnp.ones(2, jnp.int32)

    def read(arrays, r):
        c, a, b, L, NEAR, COEF, g0, cs, inv = arrays
        r = jnp.asarray(r).astype(c.dtype)
        xy = r[:2]
        p = wrap_periodic(jnp.stack([g0, g0]), L, PER, xy)
        p3 = jnp.concatenate([p, jnp.zeros(1, c.dtype)])
        G3 = jnp.stack([g0, g0, jnp.zeros((), c.dtype)])
        ids, valid = gather(NEAR, OFF, G3, cs, DIMS, p3)
        cell = cell_index(G3, cs, DIMS, p3)
        d = xy[None, :] - c[ids]
        d = ((d + 0.5 * L) % L) - 0.5 * L                                  # (M, 2): each near source's nearest image
        rho_vec = jnp.concatenate([d, jnp.zeros_like(d[:, :1])], axis=-1)
        u = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0], c.dtype), rho_vec.shape)
        F = jnp.sum(jnp.where(valid[:, None], hollow_cylinder_basis(rho_vec, u, a[ids], b[ids]), 0.0), axis=0)
        ctr = g0 + (cell[:2].astype(c.dtype) + 0.5) * cs
        zeta = jax.lax.complex((p[0] - ctr[0]) * inv, (p[1] - ctr[1]) * inv)
        co = COEF[cell[0] * G + cell[1]]
        S = co[order]
        for n in range(order - 1, -1, -1):
            S = S * zeta + co[n]
        sr, si = S.real, S.imag
        z0 = jnp.zeros((), c.dtype)
        return F + jnp.stack([z0, -0.5 * sr, 0.5 * sr, 0.5 * si, z0, z0, sr / 12.0, -sr / 12.0, z0, -si / 12.0, z0, z0])
    key = ("hollow_cylinder", field.centers.shape[0], field.summation, jnp.dtype(dtype).name, order,
           nf["near_cells"], G, M)
    return arrays, read, key


def _exact_rms(field, n_points=NEAR_FAR_RMS_POINTS, chunk=64):
    """The exact field's rms per channel over ``n_points`` uniform points of the cell (``default_rng(0)``), float64."""
    import jax
    import jax.numpy as jnp
    P = np.random.default_rng(0).uniform(-0.5 * field.L, 0.5 * field.L, (n_points, 2))
    P = np.concatenate([P, np.zeros((n_points, 1))], 1)
    with _host():
        arrays, read, _ = _exact_read(field, jnp.float64)
        f = jax.jit(jax.vmap(lambda r: read(arrays, r)))
        F = np.concatenate([np.asarray(f(jnp.asarray(P[i:i + chunk]))) for i in range(0, n_points, chunk)])
    return np.sqrt(np.mean(F * F, axis=0))


def _wrap(v, L):
    return v - L * np.round(v / L)


def _build_near_far(field):
    """The host build of :meth:`HollowCylinderField.near_far`, in float64 (JAX on the CPU device)."""
    import jax
    import jax.numpy as jnp
    from ..geometry._grid import bucket_periodic_plane
    L = field.L
    c = _wrap(field.centers, L)                                      # every source in [-L/2, L/2]^2
    cz = c[:, 0] + 1j * c[:, 1]
    b = field.outer_radii
    w = b * b - field.inner_radii ** 2
    n = c.shape[0]
    G = max(1, int(round(L / (2.0 * float(np.mean(b))))))
    cs = L / G
    rho = cs / np.sqrt(2.0)
    x0 = -0.5 * L + (np.arange(G) + 0.5) * cs
    Z0 = (x0[:, None] + 1j * x0[None, :]).ravel()                   # cell id i * G + j, i along x
    rms = _exact_rms(field)
    tol_abs = field.tolerance * float(np.max(rms))
    max_order = NEAR_FAR_MAX_ORDER if field._forced[0] is None else max(NEAR_FAR_MAX_ORDER, field._forced[0])
    E0 = np.array([image_tail_bound(rho, L, N) for N in range(max_order + 1)])
    wabs = float(np.sum(np.abs(w)))

    # the explicit poles: every image c_k + (m + i n) L, |m|, |n| <= 1, within NEAR_FAR_EXPLICIT_RADIUS L of the centre
    shifts = np.array([[m, k] for m in (-1, 0, 1) for k in (-1, 0, 1)])
    img = cz[:, None] + (shifts[None, :, 0] + 1j * shifts[None, :, 1]) * L          # (n, 9)
    inS = np.abs(img) < NEAR_FAR_EXPLICIT_RADIUS * L
    pole_index = np.full((n, 9), -1, np.int64)
    pole_index[inS] = np.arange(int(inS.sum()))
    poles, pole_w = img[inS], np.broadcast_to(w[:, None], img.shape)[inS]

    def near_table(width):
        half = b + width * cs + 1e-9 * L
        cand = bucket_periodic_plane(c, half, L, G)
        valid = cand >= 0
        k = np.where(valid, cand, 0)
        dx = _wrap(Z0.real[:, None] - c[k, 0], L)
        dy = _wrap(Z0.imag[:, None] - c[k, 1], L)
        wall = np.hypot(np.maximum(np.abs(dx) - 0.5 * cs, 0.0), np.maximum(np.abs(dy) - 0.5 * cs, 0.0)) - b[k]
        near = valid & (wall <= width * cs)
        if np.any(near & (np.maximum(np.abs(dx), np.abs(dy)) + 0.5 * cs >= 0.5 * L)):
            raise ValueError(f"HollowCylinderField(read='near_far'): at a near width of {width} cells ({G} cells per "
                             f"side) a near cylinder's nearest image is not the same at every point of a cell; the cell "
                             f"of side {L:.6g} m is too small for this near set")
        M = max(1, int(near.sum(1).max()))
        pos = np.argsort(~near, axis=1, kind="stable")[:, :M]
        rows = np.arange(G * G)[:, None]
        table = np.where(near[rows, pos], cand[rows, pos], -1).astype(np.int32)
        own = np.full(table.shape, -1, np.int64)                      # the pole of each near cylinder's nearest image
        ok = table >= 0
        kk = np.where(ok, table, 0)
        sx = np.round((Z0.real[:, None] - c[kk, 0]) / L).astype(np.int64)
        sy = np.round((Z0.imag[:, None] - c[kk, 1]) / L).astype(np.int64)
        own = np.where(ok, pole_index[kk, (sx + 1) * 3 + (sy + 1)], -1)
        if np.any(ok & (own < 0)):
            raise ValueError(f"HollowCylinderField(read='near_far'): at a near width of {width} cells a near cylinder's "
                             f"nearest image lies beyond {NEAR_FAR_EXPLICIT_RADIUS} L of the cell's centre, outside the "
                             f"explicit poles")
        return table, M, np.where(ok, own, -1)

    chunk = max(1, int(2 ** 22 // max(n, len(poles))))

    @jax.jit
    def _tail_sums(z0, far):
        dl = z0[:, None] - cz_j[None, :]
        dx = dl.real - L * jnp.round(dl.real / L)
        dy = dl.imag - L * jnp.round(dl.imag / L)
        d = jnp.hypot(dx, dy)
        x = jnp.where(far, rho / jnp.where(far, d, 1.0), 0.0)
        q = jnp.where(far, wabs_j / ((1.0 - x) ** 2 * jnp.where(far, d, 1.0) ** 2), 0.0)
        T = []
        for _ in range(max_order + 2):
            q = q * x
            T.append(jnp.sum(q, axis=1))
        return jnp.stack(T, 1)                                         # T_m, m = 1 .. max_order + 2

    @jax.jit
    def _pole_sums(z0, keep, order_arr):
        dl = z0[:, None] - poles_j[None, :]
        dl = jnp.where(keep, dl, 1.0)
        q = jnp.where(keep, pw_j / (dl * dl), 0.0)
        r = -cs / dl
        out = []
        for k in range(order_arr.shape[0]):
            out.append(jnp.sum((k + 1) * q, axis=1))
            q = q * r
        return jnp.stack(out, 1)

    def bounds(table):
        """(G * G, max_order + 1): the remainder bound of S per cell at every order."""
        out = []
        for s0 in range(0, G * G, chunk):
            t = table[s0:s0 + chunk]
            far = np.ones((t.shape[0], n), bool)
            rr, cc = np.nonzero(t >= 0)
            far[rr, t[rr, cc]] = False
            out.append(np.asarray(_tail_sums(jnp.asarray(Z0[s0:s0 + chunk]), jnp.asarray(far))))
        T = np.concatenate(out)
        N = np.arange(max_order + 1)
        return (N + 2) * T[:, N] - (N + 1) * T[:, N + 1] + wabs * E0[None, :]

    with _host():
        cz_j, wabs_j = jnp.asarray(cz), jnp.asarray(np.abs(w))
        poles_j, pw_j = jnp.asarray(poles), jnp.asarray(pole_w)
        order_f, width_f = field._forced
        widths = [width_f] if width_f is not None else (
            [NEAR_FAR_WIDTH] if order_f is not None else list(range(NEAR_FAR_WIDTH, NEAR_FAR_MAX_WIDTH + 1)))
        tried, chosen = [], None
        for width in widths:
            table, M, own = near_table(width)
            B = bounds(table)
            chan = B.max(0) / 2.0                                     # the largest channel bound at every order
            tried.append(dict(near_cells=width, list_length=M, bound_max=chan.tolist()))
            if order_f is not None:
                chosen = (order_f, width, table, M, own, B[:, order_f])
                break
            meets = np.flatnonzero(chan <= tol_abs)
            if meets.size:
                chosen = (int(meets[0]), width, table, M, own, B[:, int(meets[0])])
                break
        if chosen is None:
            raise ValueError(f"HollowCylinderField(read='near_far') is refused: no order up to {NEAR_FAR_MAX_ORDER} "
                             f"brings the largest remainder bound to the tolerance {tol_abs:.3e} ({field.tolerance:g} of "
                             f"the exact field's rms {float(np.max(rms)):.3e}) at near widths "
                             f"{[t['near_cells'] for t in tried]}; the bounds reached at the highest order: "
                             f"{[t['bound_max'][-1] for t in tried]}")
        order, width, table, M, own, bound = chosen

        # the global part: V(z) = sum_k w_k [P(z - c_k) - (its explicit poles)], analytic in |z| < R_EX L
        Rg = NEAR_FAR_SAMPLE_RADIUS * L
        zs = Rg * np.exp(2j * np.pi * np.arange(NEAR_FAR_SAMPLES) / NEAR_FAR_SAMPLES)

        @jax.jit
        def _global(zs, czk, wk, pik):
            u = zs[:, None] - czk[None, :]
            sx = jnp.round(u.real / L)
            sy = jnp.round(u.imag / L)
            us = u - (sx + 1j * sy) * L                               # the nearest image
            idx = ((sx + 1) * 3 + (sy + 1)).astype(jnp.int32)
            nearest_in = jnp.take_along_axis(jnp.broadcast_to(pik[None], (zs.shape[0],) + pik.shape),
                                             idx[..., None], axis=-1)[..., 0] >= 0
            val = lattice_regular_part(us, L) + jnp.where(nearest_in, 0.0, 1.0 / (us * us))
            for j, (m, k) in enumerate(shifts):
                e = u - (m + 1j * k) * L
                drop = (pik[None, :, j] >= 0) & (idx != j)
                val = val - jnp.where(drop, 1.0 / jnp.where(drop, e * e, 1.0), 0.0)
            return jnp.sum(wk[None, :] * val, axis=1)
        f = np.zeros(NEAR_FAR_SAMPLES, complex)
        step = max(1, int(2 ** 21 // NEAR_FAR_SAMPLES))
        for s0 in range(0, n, step):
            f += np.asarray(_global(jnp.asarray(zs), jnp.asarray(cz[s0:s0 + step]), jnp.asarray(w[s0:s0 + step]),
                                    jnp.asarray(pole_index[s0:s0 + step])))
        alpha = np.fft.fft(f) / NEAR_FAR_SAMPLES                       # V(Rg t) = sum_p alpha_p t^p
        tau = Z0 / Rg
        a = np.broadcast_to(alpha, (G * G, NEAR_FAR_SAMPLES)).copy()
        deg = NEAR_FAR_SAMPLES - 1
        for i in range(order + 1):                                     # Taylor shift to tau (Knuth 4.6.4)
            for j in range(deg - 1, i - 1, -1):
                a[:, j] += tau * a[:, j + 1]
        coef = a[:, :order + 1] * (cs / Rg) ** np.arange(order + 1)

        # the explicit poles per cell, but each near cylinder's nearest image
        for s0 in range(0, G * G, chunk):
            o = own[s0:s0 + chunk]
            keep = np.ones((o.shape[0], len(poles)), bool)
            rr, cc = np.nonzero(o >= 0)
            keep[rr, o[rr, cc]] = False
            coef[s0:s0 + chunk] += np.asarray(_pole_sums(jnp.asarray(Z0[s0:s0 + chunk]), jnp.asarray(keep),
                                                         jnp.zeros(order + 1)))
    table_bytes = int(table.size * 4 + coef.size * 8)
    return dict(order=int(order), near_cells=int(width), cells_per_side=G, cell_size=cs, half_diagonal=rho,
                list_length=int(M), near=table, coefficients=coef, bound=bound, rms=rms, tolerance_abs=tol_abs,
                bound_max=float(bound.max() / 2.0), table_bytes=table_bytes, widths=tried,
                explicit_poles=int(len(poles)))


def __getattr__(name):
    from .._moved import refuse
    raise refuse(__name__, name)
