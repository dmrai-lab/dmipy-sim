"""The exactly periodic field of a cell of parallel hollow cylinders (:class:`HollowCylinderField` with a cell side ``L``).

Exact oracles, each against a bound derived in this file (no tolerance is chosen):

1. the closed form summed over square windows of ``n`` images per side (a test-local reference) converges to it, with
   the window's remainder bounded by the Laurent expansion of the lattice sum outside the window (``O(n^-2)``);
2. it is periodic: no jump at the cell's faces (the image sum's jump is the control that fails), ``F(r + s) = F(r)``
   for lattice shifts, and the lattice sum's rows are converged;
3. it is harmonic across the cell's face (``harmonic_residual``; the image sum fails it);
4. a lone sheath in a large cell tends to the isolated closed form as ``L^-4``, with the coefficient ``3 G_4`` of the
   square lattice's Eisenstein series;
5. its cell mean is the closed form stated in the docstring (Gauss-Legendre quadratures of the three pieces);
6. the float32 read is within a first-order rounding bound of the float64 read of the same function, on a grid that
   covers the cell and the sheaths' neighbourhoods;
7. it costs no more per point than the image sum of six images per side;
8. a numpy coding through a different identity (the Fourier series of DLMF 23.8.1) agrees with the float64 read.

The rounding model (used by 2, 4, 6 and the series switch): per elementary operation a relative error of at most ``u``
(the format's unit roundoff), first order. ``x = (pi / L) z`` carries ``3 u |x|`` (``pi / L``: two roundings; the
product: one); a complex ``sin`` is within ``4 u`` relative, a complex product ``3 u`` and a quotient ``4 u``, so
``csc^2 y`` is evaluated within ``(2 x 4 + 3 + 4) u = 15 u`` relative at its argument and ``1 / x^2`` within ``7 u``;
the argument's own error propagates through the derivative; a sum of ``n`` terms adds ``(n - 1) u`` times the sum of
their magnitudes. The closed form ``hollow_cylinder_basis`` evaluates each channel as a few products of ``e_1`` (from
``rho``, ``2 u``, and a quotient, ``3 u``), ``r^2`` and one ``ln``: within ``20 u`` times the sum of the magnitudes of
its region's coefficients. The wrapped offset ``((r - c + L / 2) mod L) - L / 2`` costs four roundings of values below
``|r| + |c| + L`` per coordinate; the field's gradient is at most ``2 / a`` per sheath (the sheath's
``a^2 / (2 rho^2) S``: ``a^2 / rho^3`` radially and ``a^2 / rho^3`` from ``S``'s turn) plus ``|alpha| sup|W'|``.
"""
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_analytic.fields import hollow_cylinder_basis as basis_np
from dmipy_sim.compartments import Compartments, Pool
from dmipy_sim.fields.hollow_cylinder import (HollowCylinderField, LATTICE_ROWS, SERIES, SERIES_SWITCH,
                                              lattice_regular_part)
from dmipy_sim.geometry import PackedMyelinatedCylinders

U32, U64 = 2.0 ** -24, 2.0 ** -53
R = 2e-6
L = 4.4 * R
#: the field sources of the mixed cell (g-ratios 0.7, 1, 0.7, 1): cylinders 0 and 2 are sheathed
OUTER = np.array([R, 0.8 * R, R, 0.9 * R])
CENTRES = np.array([[0.0, 0.0], [2.2, 0.0], [0.0, 2.2], [2.2, 2.2]]) * R
G4 = math.gamma(0.25) ** 8 / (960.0 * math.pi ** 2)       # sum' w^-4 of the unit square lattice (DLMF 23.5(iii))


def _smoke_cell():
    D = 2e-9
    g = np.array([0.7, 1.0, 0.7, 1.0])
    return PackedMyelinatedCylinders(OUTER * g, g, CENTRES, L, N_max=4,
                                     compartments=Compartments(intra=Pool(D=D), myelin=Pool(D=0.0), extra=Pool(D=D)))


def _smoke_field():
    return HollowCylinderField.of(_smoke_cell())


def _one_sheath(Lc):
    """One sheathed cylinder (a = 0.7 R, b = R) off-centre in a cell of side ``Lc``."""
    return HollowCylinderField([[0.13 * Lc, -0.21 * Lc]], [0.7 * R], [R], L=Lc)


def _read(f, P, dtype):
    """The field's periodic read at ``P`` (n, 2) in ``dtype``; float64 under ``jax_enable_x64``."""
    P3 = np.concatenate([np.asarray(P, np.float64), np.zeros((len(P), 1))], 1)
    if np.dtype(dtype) == np.float64:
        was = bool(jax.config.read("jax_enable_x64"))
        jax.config.update("jax_enable_x64", True)
        try:
            return f.channels(P3, np.float64)
        finally:
            jax.config.update("jax_enable_x64", was)
    return f.channels(P3, np.float32)


# ── test-local references ───────────────────────────────────────────────────────────────────────────────────────────

def _image_sum(f, P, n):
    """The closed form summed over the ``(2n + 1)^2`` images about the cell of every source, at ``P`` wrapped into the
    cell (float64 numpy): the truncated sum this file holds the periodic read to."""
    out = np.zeros((len(P), 12))
    p = ((np.asarray(P, np.float64) + 0.5 * f.L) % f.L) - 0.5 * f.L
    ims = np.arange(-n, n + 1) * f.L
    off = np.stack(np.meshgrid(ims, ims), -1).reshape(-1, 2)
    chunk = max(1, int(1e6 // len(off)))
    for k in range(len(f.centers)):
        for i in range(0, len(p), chunk):
            d = p[i:i + chunk, None, :] - (f.centers[k] + off)[None]
            rv = np.concatenate([d, np.zeros(d.shape[:2] + (1,))], -1)
            out[i:i + chunk] += basis_np(rv, np.broadcast_to([0.0, 0.0, 1.0], rv.shape), f.inner_radii[k],
                                         f.outer_radii[k]).sum(1)
    return out


def _g_series(x, terms=12):
    """``csc^2 x - 1 / x^2`` by its Taylor series to ``terms`` terms (exact rationals from the Bernoulli numbers)."""
    from fractions import Fraction
    B = [Fraction(1)]
    for m in range(1, 2 * terms + 3):
        B.append(-sum(math.comb(m + 1, k) * B[k] for k in range(m)) / Fraction(m + 1))
    c = [float(Fraction((2 * n - 1) * (-1) ** (n + 1) * 2 ** (2 * n)) * B[2 * n] / math.factorial(2 * n))
         for n in range(1, terms + 1)]
    s = x * x
    return sum(cn * s ** i for i, cn in enumerate(c))


def _W_fourier(z, Lc):
    """``P(z) - 1 / z^2`` by DLMF 23.8.1 (``omega_1 = L / 2``, ``tau = i``, ``q = e^-pi``, ``eta_1 / omega_1 = pi /
    L^2``): ``-pi / L^2 + (pi / L)^2 csc^2(pi z / L) - 8 (pi / L)^2 sum_n n q^2n / (1 - q^2n) cos(2 pi n z / L)``,
    valid for ``|Im z| < L``; its terms fall as ``e^(-pi n)`` in the cell."""
    x = np.pi * z / Lc
    with np.errstate(all="ignore"):
        g = np.where(np.abs(x) < 0.5, _g_series(x), 1.0 / np.sin(x) ** 2 - 1.0 / x ** 2)
    n = np.arange(1, 80)
    q2 = np.exp(-2.0 * np.pi * n)
    four = (n * q2 / (1.0 - q2) * np.cos(2.0 * np.pi * n * z[..., None] / Lc)).sum(-1)
    return (np.pi / Lc) ** 2 * (g - 8.0 * four) - np.pi / Lc ** 2


def _periodic_np(f, P):
    """An independent float64 numpy coding of the periodic field: the numpy closed form plus the tails through
    :func:`_W_fourier`."""
    out = np.zeros((len(P), 12))
    for k in range(len(f.centers)):
        d = ((np.asarray(P) - f.centers[k] + 0.5 * f.L) % f.L) - 0.5 * f.L
        rv = np.concatenate([d, np.zeros((len(d), 1))], -1)
        out += basis_np(rv, np.broadcast_to([0.0, 0.0, 1.0], rv.shape), f.inner_radii[k], f.outer_radii[k])
        W = _W_fourier(d[:, 0] + 1j * d[:, 1], f.L)
        w = f.outer_radii[k] ** 2 - f.inner_radii[k] ** 2
        out[:, [1, 2, 3]] += -0.5 * w * np.stack([W.real, -W.real, -W.imag], 1)
        out[:, [6, 7, 9]] += w / 12.0 * np.stack([W.real, -W.real, -W.imag], 1)
    return out


# ── the deterministic samples ───────────────────────────────────────────────────────────────────────────────────────

def _margin(f):
    return 1e-4 * f.inner_radii.min()          # the walk's nudge off a wall: 1e-4 x the smallest inner radius


def _off_walls(f, P):
    ok = np.ones(len(P), bool)
    for k in range(len(f.centers)):
        d = ((P - f.centers[k] + 0.5 * f.L) % f.L) - 0.5 * f.L
        r = np.hypot(d[:, 0], d[:, 1])
        ok &= (np.abs(r - f.inner_radii[k]) > _margin(f)) & (np.abs(r - f.outer_radii[k]) > _margin(f))
    return ok


def _lattice(f, n):
    """The ``n x n`` cell midpoints, less the points within the margin of a face."""
    g = (np.arange(n) + 0.5) / n * f.L - 0.5 * f.L
    P = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    return P[_off_walls(f, P)]


def _rings(f):
    """Per sheath, 64 angles on the radii ``(1 -+ d) x face`` (``d`` 2e-4 .. 1e-1) and ``0.01, 0.1, 0.5 a``."""
    ph = 2 * np.pi * (np.arange(64) + 0.5) / 64
    pts = []
    for k in range(len(f.centers)):
        a, b = f.inner_radii[k], f.outer_radii[k]
        radii = [fc * (1 + s * d) for fc in (a, b) for d in (2e-4, 1e-3, 1e-2, 1e-1) for s in (-1, 1)]
        for r in radii + [0.01 * a, 0.1 * a, 0.5 * a]:
            pts.append(f.centers[k] + r * np.stack([np.cos(ph), np.sin(ph)], 1))
    P = ((np.concatenate(pts) + 0.5 * f.L) % f.L) - 0.5 * f.L
    return P[_off_walls(f, P)]


# ── the bounds ──────────────────────────────────────────────────────────────────────────────────────────────────────

def _S8():
    """``sum' |w|^-8`` over the unit square lattice, from above: the sum to shell 60 plus the shells beyond (shell
    ``s`` has ``8 s`` points at ``|w| >= s``: at most ``8 / (6 * 60^6)``)."""
    m = np.arange(-60, 61)
    X, Y = np.meshgrid(m, m)
    r2 = (X * X + Y * Y).astype(float)
    r2[60, 60] = np.inf
    return float(np.sum(r2 ** -4.0)) + 8.0 / (6.0 * 60.0 ** 6)


def _W_bound(s):
    """``L^2 |W(z)|`` at ``|z| = s L``, ``s < 1``: ``W(z) = sum_(m >= 1) (4m - 1) G_4m z^(4m - 2)`` (the lattice sums
    ``G_2k`` vanish unless ``4 | 2k`` on the square lattice), ``|G_4m| <= L^-4m sum'|w|^-8`` for ``m >= 2``."""
    return 3.0 * G4 * s * s + _S8() * sum((4 * m - 1) * s ** (4 * m - 2) for m in range(2, 200))


def _Wp_bound(s):
    """``L^3 |W'(z)|`` at ``|z| = s L``, from the same series."""
    return 6.0 * G4 * s + _S8() * sum((4 * m - 1) * (4 * m - 2) * s ** (4 * m - 3) for m in range(2, 200))


def _gradient_bound(f):
    """``sup |grad F|`` per channel over the cell (1/m): ``2 / a`` per sheath plus ``|alpha| sup|W'|`` at the cell's
    corner ``|z| = L / sqrt 2``."""
    w = f.outer_radii ** 2 - f.inner_radii ** 2
    return float(np.sum(2.0 / f.inner_radii) + np.sum(0.5 * w) * _Wp_bound(2 ** -0.5) / f.L ** 3)


def _A_W(x, u, rows=LATTICE_ROWS, switch=None, x_roundings=3):
    """The first-order rounding bound of :func:`lattice_regular_part` at ``x = pi z / L``, in units of ``u (pi /
    L)^2`` (the rounding model of the module docstring); ``x`` carries ``x_roundings`` relative roundings."""
    x = np.asarray(x, complex)
    ax = np.abs(x)
    dx = float(x_roundings) * ax
    with np.errstate(all="ignore"):
        csc2 = 1.0 / np.sin(x) ** 2
        cot = 1.0 / np.tan(x)
        g = csc2 - 1.0 / x ** 2
        gp = np.abs(-2.0 * csc2 * cot + 2.0 / x ** 3)
        direct = 15.0 * np.abs(csc2) + 7.0 / ax ** 2 + np.abs(g) + gp * dx
    s = np.abs(x * x)
    series = 20.0 * sum(c * s ** i for i, c in enumerate(SERIES)) + np.nan_to_num(gp, nan=0.0) * dx
    sw = SERIES_SWITCH["float32" if u == U32 else "float64"] if switch is None else switch
    t0 = np.where(ax < sw, series, direct)
    terms = [np.where(ax < sw, np.abs(_g_series(x)), np.abs(np.nan_to_num(g)))]
    rows_err = 0.0
    for j in range(1, rows + 1):
        for y in (x - 1j * np.pi * j, x + 1j * np.pi * j):
            t = np.abs(1.0 / np.sin(y) ** 2)
            dy = dx + 2.0 * (np.pi * j + 0.5 * np.pi)
            rows_err = rows_err + 15.0 * t + 2.0 * np.abs(1.0 / np.tan(y)) * t * dy
            terms.append(t)
    tot = sum(terms)
    return t0 + rows_err + 2 * rows * tot + 5.0 * tot + 3.0 / np.pi + 1.0 / np.pi


def _B_mag(f, P):
    """Per point, ``sum_k`` of the magnitudes of the closed form's coefficients in the region of ``P`` (the scale of its
    evaluation error), the larger of the isotropic and anisotropic groups."""
    out = np.zeros(len(P))
    for k in range(len(f.centers)):
        a, b = f.inner_radii[k], f.outer_radii[k]
        d = ((P - f.centers[k] + 0.5 * f.L) % f.L) - 0.5 * f.L
        rho = np.hypot(d[:, 0], d[:, 1])
        r2 = np.maximum(rho, 1e-30) ** 2
        lum, sh = rho < a, (rho >= a) & (rho < b)
        iso = np.where(sh, 0.5 + a * a / (2 * r2), np.where(lum, 0.0, (b * b - a * a) / (2 * r2))) + 1.0 / 3.0
        ani = np.where(lum, 0.5 * np.log(b / a),
                       np.where(sh, 0.5 * np.log(b / rho) + 5.0 / 18.0 + 1.0 / 9.0 + (1 + a * a / r2) / 12.0,
                                (b * b - a * a) / (12 * r2)))
        out += np.maximum(iso, ani)
    return out


def _rounding_bound(f, P, u, *, M=None):
    """Per point, the first-order bound of a periodic read's rounding at ``P`` in a format of unit roundoff ``u``: the
    closed form (``20 u`` of its coefficients), each source's lattice term (``|alpha| (pi / L)^2 u A_W`` plus the
    rounding of ``alpha``), the final sums, and the wrapped offset's four roundings of values below ``M + max|c| + L``
    (``M = max |r|`` by default) through the gradient."""
    P = np.asarray(P, np.float64)
    M = float(np.abs(P).max()) if M is None else M
    lat = np.zeros(len(P))
    for k in range(len(f.centers)):
        d = ((P - f.centers[k] + 0.5 * f.L) % f.L) - 0.5 * f.L
        x = np.pi * (d[:, 0] + 1j * d[:, 1]) / f.L
        alpha = 0.5 * (f.outer_radii[k] ** 2 - f.inner_radii[k] ** 2)
        W = np.abs(_W_fourier(d[:, 0] + 1j * d[:, 1], f.L))
        lat += alpha * ((np.pi / f.L) ** 2 * u * _A_W(x, u) + 2.0 * u * W)
    B = _B_mag(f, P)
    wrap = 4.0 * u * (M + np.abs(f.centers).max() + f.L) * math.sqrt(2.0) * _gradient_bound(f)
    return 20.0 * u * B + lat + 3.0 * u * (B + lat) + wrap


# ── 1. the image sum converges to it ────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("which", ["smoke_cell", "one_sheath"])
def test_the_image_sum_converges_to_the_periodic_field(which):
    """``|F_n - F| <= sum_k alpha_k |R_n(z_k)|`` per point and channel, at ``n = 6 .. 64``: the window's remainder
    ``R_n(z) = sum_(w outside) [(z - w)^-2 - w^-2] = sum_(m >= 2, 4 | m + 2) (m + 1) z^m T_(m+2)`` with ``|T_p| <=
    sum_(s > n) 8 s (s L)^-p <= 8 L^-p n^(2 - p) / (p - 2)`` (``z`` the offset from the source, the window the
    image sum's own, about the cell), plus the float64 rounding of the image sum's ``(2n + 1)^2`` terms."""
    f = _smoke_field() if which == "smoke_cell" else _one_sheath(L)
    P = _lattice(f, 32)
    F = _read(f, P, np.float64)
    Fmax = np.abs(F).max()
    p = ((P + 0.5 * f.L) % f.L) - 0.5 * f.L
    errs = []
    for n in (6, 12, 24, 48, 64):
        Fn = _image_sum(f, P, n)
        bound = np.zeros(len(P))
        for k in range(len(f.centers)):
            t = np.hypot(*(p - f.centers[k]).T) / (n * f.L)
            assert t.max() < 1.0
            rem = sum(8.0 * (m + 1) / m * t ** m for m in range(2, 400, 4))
            bound += 0.5 * (f.outer_radii[k] ** 2 - f.inner_radii[k] ** 2) * rem / f.L ** 2
        bound += 2 * len(f.centers) * (2 * n + 1) ** 2 * U64 * Fmax
        err = np.abs(Fn - F)
        assert np.all(err <= bound[:, None]), (n, float((err - bound[:, None]).max()))
        errs.append(err.max())
    # the remainder falls as n^-2: each halving of 1 / n divides the largest error by close to four
    assert errs[0] > errs[1] > errs[2] > errs[3] > errs[4]
    if which == "smoke_cell":
        # the specification's limit on the smoke cell: the remainder's constant, max |F_n - F| n^2 <= 0.02 (the
        # pointwise bound above is the proof that the limit is the periodic field; this holds its constant)
        assert all(e * n * n <= 0.02 for e, n in zip(errs, (6, 12, 24, 48, 64))), [e * n * n for e, n in zip(errs, (6, 12, 24, 48, 64))]


def test_the_float64_read_is_an_independent_coding_of_the_same_field():
    """The float64 read against the numpy closed form plus DLMF 23.8.1's Fourier series of ``P`` (a different identity
    for the lattice sum, its constant included): within the float64 rounding bound of the read plus that of the
    numpy coding (the same model)."""
    f = _smoke_field()
    P = np.concatenate([_lattice(f, 64), _rings(f)])
    err = np.abs(_read(f, P, np.float64) - _periodic_np(f, P))
    bound = 2.0 * _rounding_bound(f, P, U64)
    assert np.all(err <= bound[:, None]), float((err - bound[:, None]).max())


# ── 2. periodic ─────────────────────────────────────────────────────────────────────────────────────────────────────

def _edge_jump(read, f, eps):
    t = np.linspace(-0.45, 0.45, 40) * f.L
    lo, hi = -0.5 * f.L + eps * f.L, 0.5 * f.L - eps * f.L
    jumps = []
    for A, B in ((np.stack([np.full(40, hi), t], 1), np.stack([np.full(40, lo), t], 1)),
                 (np.stack([t, np.full(40, hi)], 1), np.stack([t, np.full(40, lo)], 1))):
        keep = _off_walls(f, A) & _off_walls(f, B)
        jumps.append(np.abs(read(A[keep]) - read(B[keep])).max())
    return max(jumps)


def test_the_periodic_field_does_not_jump_at_the_cells_faces():
    """At ``eps = 1e-9 L`` either side of each face: ``|F(L/2 - eps L) - F(-L/2 + eps L)| <= sup|grad F| 2 eps L +``
    the rounding of both reads; the image sum of six images per side jumps by more (the control)."""
    f = _smoke_field()
    eps = 1e-9
    t = np.linspace(-0.45, 0.45, 40) * f.L
    edge = np.concatenate([np.stack([np.full(40, s * 0.5 * f.L), t], 1) for s in (-1, 1)] +
                          [np.stack([t, np.full(40, s * 0.5 * f.L)], 1) for s in (-1, 1)])
    bound = _gradient_bound(f) * 2 * eps * f.L + 2.0 * float(_rounding_bound(f, edge, U64).max())
    jump = _edge_jump(lambda P: _read(f, P, np.float64), f, eps)
    assert jump <= bound, (jump, bound)
    assert _edge_jump(lambda P: _image_sum(f, P, 6), f, eps) > bound


def test_the_periodic_field_is_periodic_under_lattice_shifts():
    """``|F(r + s) - F(r)|`` for ``s = (L, 0), (0, L), (L, L), (-3L, 5L), (40L, -17L)``: within the rounding of both
    reads, whose wrapped offsets are of values up to ``|r + s|``."""
    f = _smoke_field()
    P = _lattice(f, 32)
    F = _read(f, P, np.float64)
    for s in ((1, 0), (0, 1), (1, 1), (-3, 5), (40, -17)):
        Q = P + np.asarray(s, float) * f.L
        err = np.abs(_read(f, Q, np.float64) - F)
        bound = _rounding_bound(f, P, U64) + _rounding_bound(f, Q, U64)
        assert np.all(err <= bound[:, None]), (s, float((err - bound[:, None]).max()))


def test_the_lattice_sum_is_periodic_and_its_rows_are_converged(x64):
    """``Sigma = W + 1 / z^2``: ``Sigma(z + s) = Sigma(z)`` for ``s = L, i L, L + i L, -L + i L`` within the rounding
    of both evaluations; ``W`` with ``LATTICE_ROWS`` rows against twice as many within the omitted rows' bound
    ``8 e^pi sum_(j > J) e^(-2 pi j)`` (in units of ``(pi / L)^2``) plus both roundings."""
    f = _smoke_field()
    P = _lattice(f, 32)
    z = P[:, 0] + 1j * P[:, 1]
    k2 = (np.pi / f.L) ** 2
    Wj = lambda zz, rows=LATTICE_ROWS: np.asarray(lattice_regular_part(jnp.asarray(zz, jnp.complex128), f.L, rows=rows))
    S0 = Wj(z) + 1.0 / z ** 2
    for s in (1, 1j, 1 + 1j, -1 + 1j):
        zs = z + s * f.L
        S1 = Wj(zs) + 1.0 / zs ** 2
        x0, x1 = np.pi * z / f.L, np.pi * zs / f.L
        # the shifted argument carries one more rounding (z + s L); 1 / z^2 is within 7 u of its value
        bound = k2 * U64 * (_A_W(x0, U64) + _A_W(x1, U64, x_roundings=4)) + 8.0 * U64 * (np.abs(z) ** -2 + np.abs(zs) ** -2)
        assert np.all(np.abs(S1 - S0) <= bound), s
    J = LATTICE_ROWS
    omitted = 8.0 * np.exp(np.pi) * sum(np.exp(-2 * np.pi * j) for j in range(J + 1, 2 * J + 1))
    x = np.pi * z / f.L
    bound = k2 * (omitted + U64 * (_A_W(x, U64) + _A_W(x, U64, rows=2 * J)))
    assert np.all(np.abs(Wj(z) - Wj(z, 2 * J)) <= bound)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_series_and_the_difference_meet_at_the_switch(dtype, x64):
    """At ``|x| =`` the switch, on 64 directions: the series and the difference agree within the series' first omitted
    term (``<= u / 3`` by the switch's definition) plus both roundings."""
    u = U32 if dtype == "float32" else U64
    sw = SERIES_SWITCH[dtype]
    ph = 2 * np.pi * (np.arange(64) + 0.5) / 64
    ct = jnp.complex64 if dtype == "float32" else jnp.complex128
    Lc = 1.0
    for r in (sw * (1 - 4 * u), sw * (1 + 4 * u)):
        x = r * np.exp(1j * ph)
        z = x * Lc / np.pi
        got = np.asarray(lattice_regular_part(jnp.asarray(z, ct), Lc), np.complex128)
        ref = _W_fourier(z.astype(np.complex128), Lc)
        bound = (np.pi / Lc) ** 2 * (u * _A_W(x, u) + u / 3.0 * 1.1) + 2 * U64 * (np.pi / Lc) ** 2 * _A_W(x, U64)
        assert np.all(np.abs(got - ref) <= bound), (dtype, r, float((np.abs(got - ref) / bound).max()))


# ── 3. harmonic across the face ─────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("angle", [0.3, 1.1])
def test_the_periodic_field_is_harmonic_across_the_cells_face(angle):
    """``H . M_P . H`` for an in-plane ``H``, on 40 points straddling the face ``x = L / 2`` (more than 0.4 um from every
    sheath): ``harmonic_residual <= HARMONIC_TOL`` for the periodic field, and above it for the image sum (its jump
    is a delta in the Laplacian), so the check is not vacuous."""
    from dmipy_replay.acquisition.maxwell import HARMONIC_TOL, harmonic_residual
    f = _smoke_field()
    H = np.array([np.cos(angle), np.sin(angle)])
    y = np.linspace(-0.45 * f.L, 0.45 * f.L, 40)
    P = np.stack([np.full(40, 0.5 * f.L - 1e-9 * f.L), y], 1)
    far = np.ones(40, bool)
    for k in range(len(f.centers)):
        d = ((P - f.centers[k] + 0.5 * f.L) % f.L) - 0.5 * f.L
        far &= np.hypot(*d.T) > f.outer_radii[k] + 0.4e-6
    pts = np.concatenate([P[far], np.zeros((far.sum(), 1))], 1)

    def value(read):
        def field(X):
            C = read(np.asarray(X)[:, :2])
            return H[0] ** 2 * C[:, 1] + H[1] ** 2 * C[:, 2] + 2 * H[0] * H[1] * C[:, 3]
        return field
    assert harmonic_residual(value(lambda X: _read(f, X, np.float64)), pts, h=2e-8) <= HARMONIC_TOL
    assert not harmonic_residual(value(lambda X: _image_sum(f, X, 6)), pts, h=2e-8) <= HARMONIC_TOL


# ── 4. the isolated limit ───────────────────────────────────────────────────────────────────────────────────────────

def test_a_lone_sheath_tends_to_the_isolated_closed_form_as_L_to_the_minus_four():
    """One sheath at ``(0.13 L, -0.21 L)`` in cells of side ``L = 100, 300, 1000 b``; 640 points ``rho / b = 0.05 +
    0.5 k`` at 16 angles. Per point and channel ``F - F_isolated = alpha T(W(z))``: (i) within ``alpha |W|`` bound
    (:func:`_W_bound`); (ii) minus its leading term ``alpha T(3 G_4 z^2 / L^4)`` within the bound of the rest of the
    series (``|W - 3 G_4 z^2 / L^4| <= L^-2 sum'|w|^-8 sum_(m >= 2) (4m - 1) s^(4m - 2)``), so the law is ``L^-4``
    with the square lattice's ``G_4 = Gamma(1/4)^8 / (960 pi^2)``; both plus the rounding of the two reads. ``G_4``
    itself against the lattice sum."""
    m = np.arange(-400, 401)
    X, Y = np.meshgrid(m, m)
    w = (X + 1j * Y).ravel()
    w = w[w != 0]
    tail = 8.0 / (2.0 * 400.0 ** 2)                                   # shells beyond 400: sum 8 s s^-4
    assert abs(np.sum(w ** -4.0).real - G4) <= tail + len(w) * U64 * np.sum(np.abs(w) ** -4.0)
    ph = 2 * np.pi * (np.arange(16) + 0.5) / 16
    rr = (0.05 + 0.5 * np.arange(40)) * R
    for Lc in (100 * R, 300 * R, 1000 * R):
        f = _one_sheath(Lc)
        z = (rr[:, None] * np.exp(1j * ph)[None]).ravel()
        P = f.centers[0] + np.stack([z.real, z.imag], 1)
        P = P[_off_walls(f, P)]
        z = (P[:, 0] - f.centers[0, 0]) + 1j * (P[:, 1] - f.centers[0, 1])
        Fp = _read(f, P, np.float64)
        Fi = basis_np(np.concatenate([P - f.centers[0], np.zeros((len(P), 1))], 1), np.broadcast_to([0, 0, 1.0], (len(P), 3)),
                      f.inner_radii[0], f.outer_radii[0])
        s = np.abs(z) / Lc
        wdt = f.outer_radii[0] ** 2 - f.inner_radii[0] ** 2
        alphas = np.zeros(12)
        alphas[[1, 2, 3]] = 0.5 * wdt
        alphas[[6, 7, 9]] = wdt / 12.0
        rnd = _rounding_bound(f, P, U64) + 20.0 * U64 * _B_mag(f, P)
        err = np.abs(Fp - Fi)
        assert np.all(err <= alphas[None] * np.vectorize(_W_bound)(s)[:, None] / Lc ** 2 + rnd[:, None])
        lead = 3.0 * G4 * z ** 2 / Lc ** 4
        T = np.zeros((len(P), 12))
        T[:, [1, 2, 3]] = -0.5 * wdt * np.stack([lead.real, -lead.real, -lead.imag], 1)
        T[:, [6, 7, 9]] = wdt / 12.0 * np.stack([lead.real, -lead.real, -lead.imag], 1)
        rest = _S8() * sum((4 * k - 1) * s ** (4 * k - 2) for k in range(2, 50)) / Lc ** 2
        assert np.all(np.abs(Fp - Fi - T) <= alphas[None] * rest[:, None] + rnd[:, None])
        assert np.abs(T).max() > (alphas[None] * rest[:, None] + rnd[:, None]).max()      # the law exceeds its remainder


# ── 5. the cell mean ────────────────────────────────────────────────────────────────────────────────────────────────

def _docstring_mean(f):
    """The cell mean stated in :class:`HollowCylinderField`'s docstring."""
    a, b, Lc = f.inner_radii, f.outer_radii, f.L
    A = np.sum(np.pi * (b ** 2 - a ** 2))
    I = (b ** 2 - a ** 2) / 4 - (a ** 2 / 2) * np.log(b / a)
    out = np.zeros(12)
    out[0] = A / (3 * Lc ** 2)
    out[1] = out[2] = A / (2 * Lc ** 2)
    out[8] = -A / (9 * Lc ** 2)
    out[6] = out[7] = np.sum(np.pi * a ** 2 / 2 * np.log(b / a) + np.pi * I - 5 / 18 * np.pi * (b ** 2 - a ** 2)) / Lc ** 2
    return out


def test_the_cell_mean_is_the_docstrings_closed_form():
    """The cell integral of ``F = sum_k [B_k + alpha_k T(W(z_k))]`` is ``sum_k`` of (i) ``B_k`` over its disc ``rho <
    b_k`` (polar Gauss-Legendre per annulus ``[0, a]``, ``[a, b]``, 16 radial nodes and 16 angles: the angular
    trapezoid is exact below trigonometric degree 16 and the radial integrands are polynomials and ``rho ln(b /
    rho)``, analytic on the annulus' interval), (ii) ``alpha_k`` times ``int W`` over the square (tensor
    Gauss-Legendre, 32 nodes: ``W`` is analytic on the closed square) and (iii) ``alpha_k`` times the tail ``1 / z^2``
    over the square less the disc (per octant ``int cos 2 phi ln(rho_s(phi) / b) dphi``, 32 nodes); (ii) and (iii)
    vanish by the square's fourfold symmetry. Each against the docstring's means within the summation bound of its
    quadrature (``n u`` times the sum of the magnitudes of its terms)."""
    f = _smoke_field()
    total = np.zeros(12)
    scale = np.zeros(12)
    xg, wg = np.polynomial.legendre.leggauss(16)
    th = 2 * np.pi * np.arange(16) / 16
    for k in range(len(f.centers)):
        a, b = f.inner_radii[k], f.outer_radii[k]
        for lo, hi in ((0.0, a), (a, b)):
            rho = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
            wr = 0.5 * (hi - lo) * wg * rho
            pts = (rho[:, None, None] * np.stack([np.cos(th), np.sin(th)], -1)[None]).reshape(-1, 2)
            rv = np.concatenate([pts, np.zeros((len(pts), 1))], 1)
            vals = basis_np(rv, np.broadcast_to([0, 0, 1.0], rv.shape), a, b).reshape(16, 16, 12)
            terms = vals * wr[:, None, None] * (2 * np.pi / 16)
            total += terms.sum((0, 1))
            scale += np.abs(terms).sum((0, 1))
    x32, w32 = np.polynomial.legendre.leggauss(32)
    g = 0.5 * f.L * x32
    Z = g[:, None] + 1j * g[None, :]
    was = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    try:
        W = np.asarray(lattice_regular_part(jnp.asarray(Z, jnp.complex128), f.L))
    finally:
        jax.config.update("jax_enable_x64", was)
    wts = (0.5 * f.L) ** 2 * w32[:, None] * w32[None, :]
    intW = np.sum(wts * W)
    intW_scale = np.sum(np.abs(wts * W))
    for k in range(len(f.centers)):
        bk = f.outer_radii[k]
        oct_ = oct_im = 0.0
        oct_scale = 0.0
        for o in range(8):
            phi = (o + 0.5 + 0.5 * x32) * np.pi / 4
            rho_s = 0.5 * f.L / np.maximum(np.abs(np.cos(phi)), np.abs(np.sin(phi)))
            t = np.cos(2 * phi) * np.log(rho_s / bk) * w32 * np.pi / 8
            ti = -np.sin(2 * phi) * np.log(rho_s / bk) * w32 * np.pi / 8      # Im(1 / z^2) = -sin 2 phi / rho^2
            oct_ += t.sum()
            oct_im += ti.sum()
            oct_scale += np.abs(t).sum() + np.abs(ti).sum()
        wdt = f.outer_radii[k] ** 2 - f.inner_radii[k] ** 2
        for ch, sg, part, ps in ((1, -0.5, intW.real + oct_, intW_scale + oct_scale),
                                 (2, 0.5, intW.real + oct_, intW_scale + oct_scale),
                                 (3, 0.5, intW.imag + oct_im, intW_scale + oct_scale),
                                 (6, 1 / 12, intW.real + oct_, intW_scale + oct_scale),
                                 (7, -1 / 12, intW.real + oct_, intW_scale + oct_scale),
                                 (9, -1 / 12, intW.imag + oct_im, intW_scale + oct_scale)):
            total[ch] += sg * wdt * part
            scale[ch] += abs(sg) * wdt * ps
    assert abs(intW) <= 32 * 32 * U64 * intW_scale
    mean = total / f.L ** 2
    bound = 2 * 256 * U64 * scale / f.L ** 2
    assert np.all(np.abs(mean - _docstring_mean(f)) <= bound), (mean, _docstring_mean(f), bound)
    assert _docstring_mean(f)[1] > 0.0 and np.all(_docstring_mean(f)[[3, 4, 5, 9, 10, 11]] == 0.0)


# ── 6. float32 against float64 ──────────────────────────────────────────────────────────────────────────────────────

def test_the_float32_read_is_within_its_rounding_bound_of_the_float64_read():
    """On the 64 x 64 cell lattice and the sheaths' rings: the float32 read against the float64 read of the same
    function (i) at the same float32-rounded inputs, within the float32 rounding bound; (ii) at the true positions,
    adding the inputs' rounding (``u |r|`` per coordinate) through the gradient."""
    f = _smoke_field()
    P = np.concatenate([_lattice(f, 64), _rings(f)])
    P32 = P.astype(np.float32).astype(np.float64)
    F32 = _read(f, P, np.float32)
    bound = _rounding_bound(f, P32, U32) + _rounding_bound(f, P32, U64)
    err = np.abs(F32 - _read(f, P32, np.float64))
    assert np.all(err <= bound[:, None]), float((err / bound[:, None]).max())
    err_true = np.abs(F32 - _read(f, P, np.float64))
    bound_true = bound + U32 * np.abs(P).max() * math.sqrt(2.0) * _gradient_bound(f)
    assert np.all(err_true <= bound_true[:, None])
    # the specification's limits on this grid: 1.3e-6 at the same float32-rounded inputs (the formula's conditioning)
    # and 7.7e-6 at the true positions (with the inputs' rounding through the gradient); the per-point bounds above
    # charge every wrapped offset's rounding at its worst and are looser
    assert err.max() <= 1.3e-6 and err_true.max() <= 7.7e-6, (err.max(), err_true.max())


# ── 7. cost ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def _image_read_jax(f, n, dtype):
    """The closed form summed over ``(2n + 1)^2`` images per source about the cell, in JAX (``dtype``): the read the
    periodic one replaces, as a cost reference."""
    from dmipy_analytic.jax.fields import hollow_cylinder_basis
    ims = np.arange(-n, n + 1) * f.L
    offs = jnp.asarray(np.stack(np.meshgrid(ims, ims), -1).reshape(-1, 2), dtype)
    c, a, b, Lc = (jnp.asarray(f.centers, dtype), jnp.asarray(f.inner_radii, dtype), jnp.asarray(f.outer_radii, dtype),
                   jnp.asarray(f.L, dtype))

    def read(r):
        p = ((r[:2] + 0.5 * Lc) % Lc) - 0.5 * Lc
        d = p[None, None, :] - (c[:, None, :] + offs[None])
        rv = jnp.concatenate([d, jnp.zeros_like(d[..., :1])], -1)
        u = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0], dtype), rv.shape)
        return jnp.sum(hollow_cylinder_basis(rv, u, jnp.broadcast_to(a[:, None], d.shape[:2]),
                                             jnp.broadcast_to(b[:, None], d.shape[:2])), axis=(0, 1))
    return jax.jit(jax.vmap(read))


def _best(fn, P, reps=3):
    jax.block_until_ready(fn(P))
    best = np.inf
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn(P))
        best = min(best, time.perf_counter() - t)
    return best


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_periodic_read_costs_no_more_than_six_images_per_side(dtype, x64):
    f = _smoke_field()
    dt = jnp.float32 if dtype == "float32" else jnp.float64
    rng = np.random.default_rng(0)
    P = jnp.asarray(rng.uniform(-0.5 * f.L, 0.5 * f.L, (8192, 3)), dt)
    arrays, read, _ = f.device_read(dt)
    periodic = jax.jit(jax.vmap(lambda r: read(arrays, r)))
    assert _best(periodic, P) <= _best(_image_read_jax(f, 6, dt), P)


# ── refusals ────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_a_sheath_reaching_half_the_cell_is_refused_by_name():
    with pytest.raises(ValueError, match="reaches half the cell's side"):
        HollowCylinderField([[0.0, 0.0]], [0.7 * R], [R], L=2 * R)
    _one_sheath(2.0001 * R).device_read()                       # just inside: read


@pytest.mark.parametrize("bad", ["centers", "inner_radii", "outer_radii", "L"])
def test_non_finite_construction_is_refused_by_name(bad):
    kw = dict(centers=[[0.0, 0.0]], inner_radii=[0.7 * R], outer_radii=[R], L=L)
    kw[bad] = np.nan if bad == "L" else np.full(np.shape(kw[bad]), np.nan)
    with pytest.raises(ValueError, match=("the cell's side L" if bad == "L" else f"{bad} are not all finite")):
        HollowCylinderField(kw.pop("centers"), kw.pop("inner_radii"), kw.pop("outer_radii"), **kw)


def test_the_image_count_is_not_a_parameter():
    """One field: the periodic lattice sum. ``n_images`` is no argument and ``CELL_IMAGES`` no name of the module."""
    import dmipy_sim.fields.hollow_cylinder as hc
    with pytest.raises(TypeError, match="n_images"):
        HollowCylinderField([[0.0, 0.0]], [0.7 * R], [R], L=L, n_images=6)
    with pytest.raises(TypeError, match="n_images"):
        HollowCylinderField.of(_smoke_cell(), n_images=6)
    assert "CELL_IMAGES" not in vars(hc)


def test_a_record_of_the_image_sum_field_is_refused_and_an_isolated_one_reads():
    """A field record that names ``n_images`` for a cell is refused by name; a record of an isolated cylinder (no
    cell, no images) reads as recorded; a cell's record round-trips."""
    f = _smoke_field()
    m = dict(f.meta)
    assert m["summation"] == "lattice" and m["lattice_rows"] == LATTICE_ROWS and "n_images" not in m
    assert HollowCylinderField.from_meta(m).meta == m
    old = {k: v for k, v in m.items() if k not in ("summation", "lattice_rows")}
    old["n_images"] = 6
    with pytest.raises(ValueError, match="walked with the image-sum field .* re-walk"):
        HollowCylinderField.from_meta(old)
    iso = dict(kind="hollow_cylinder", centers=[[0.0, 0.0]], inner_radii=[0.7 * R], outer_radii=[R], L=None, n_images=0,
               channels=list(HollowCylinderField.channel_names), mean_subtracted=False, periodic=[False, False, False])
    back = HollowCylinderField.from_meta(iso)
    assert back.L is None and back.summation == "none"


def test_non_finite_points_are_refused_by_name():
    f = _smoke_field()
    with pytest.raises(ValueError, match="points are not finite"):
        f.channels(np.array([[0.0, np.nan, 0.0], [1e-6, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="points are not finite"):
        f.channels(np.array([[np.inf, 0.0, 0.0]]))


def test_a_float64_read_without_x64_is_refused_by_name():
    if jax.config.read("jax_enable_x64"):
        pytest.skip("this process runs with jax_enable_x64: the refusal is of a float64 read without it")
    with pytest.raises(ValueError, match="needs jax_enable_x64"):
        _smoke_field().device_read(jnp.float64)


# ── 8. the walk reads it ────────────────────────────────────────────────────────────────────────────────────────────

def _walk(geo, f, n=32, T_max=1e-3):
    from dmipy_sim.engine.core import _walk_geometry
    return _walk_geometry(n, 2e-9, geo, T_max, 5e-5, seed=3, require_gpu=False, K=4, field_basis=f, K_c3=4,
                          field_read_interval=1.25e-5, raw_subset=n)


def _block_reads(w):
    """The raw subset's field reads at the block ends (a read instant every ``reads_per_block``-th read)."""
    nb, nr = w.raw_positions.shape[1] - 1, w.raw_field.shape[1] - 1
    assert nr % nb == 0
    return w.raw_field[:, ::nr // nb]


def _assert_an_image_sum_record_is_refused_on_load(path, old_path):
    """The same walk file with its field record as the image-sum field wrote it (``n_images``, no summation): its
    load is refused by name."""
    import json
    from safetensors import safe_open
    from safetensors.numpy import save_file
    from dmipy_sim.coefficient_walk import CoefficientWalk
    with safe_open(str(path), framework="numpy") as fh:
        md = dict(fh.metadata())
        arrays = {k: fh.get_tensor(k) for k in fh.keys()}
    h = json.loads(md["walk"])
    fm = h["field"]["meta"]
    for k in ("summation", "lattice_rows"):
        fm.pop(k)
    fm["n_images"] = 6
    md["walk"] = json.dumps(h)
    save_file(arrays, str(old_path), metadata=md)
    with pytest.raises(ValueError, match="walked with the image-sum field .* re-walk"):
        CoefficientWalk.load(old_path)


def test_a_walk_of_a_cell_reads_the_periodic_field(tmp_path):
    """A walk of the cell with the field and its raw subset: the field modes are finite, the record states the lattice
    sum and no image count (and a walk file of it loads), and every raw walker's read at every block end is the
    float32 read of the same function at its recorded position, within a derived bound. The walk reads the field at
    the walker's position in the cell (the kernel's ``r_incell``) and records the continuous position (``r_unwrapped``,
    advanced by the minimum-image step), which it does not read: the two differ by a lattice vector and by the
    rounding of the continuous position's sums, at most ``u (M + 3 L)`` per coordinate per step (the step's minimum
    image ``r_new - r - L round(.)`` and its addition to a position below ``M``), through the gradient bound, plus both
    reads' float32 rounding."""
    from dmipy_sim.coefficient_walk import CoefficientWalk
    cell = _smoke_cell()
    f = HollowCylinderField.of(cell)
    w = _walk(cell, f)
    assert np.all(np.isfinite(w.field_coeffs))
    meta = w.field_basis.meta
    assert meta["summation"] == "lattice" and "n_images" not in meta and meta["mean_subtracted"] is False
    w.save(tmp_path / "w.safetensors")
    assert CoefficientWalk.load(tmp_path / "w.safetensors").field_basis.meta == meta
    _assert_an_image_sum_record_is_refused_on_load(tmp_path / "w.safetensors", tmp_path / "old.safetensors")
    F = _block_reads(w)
    pos = w.raw_positions.astype(np.float64)
    direct = f.channels(pos.reshape(-1, 3), np.float32).reshape(F.shape)
    steps = np.arange(pos.shape[1]) * round(w.block["h"] / w.dt_sim)
    M = float(np.abs(pos).max())
    drift = steps * U32 * (M + 3 * f.L) * math.sqrt(2.0) * _gradient_bound(f)
    rnd = 2.0 * _rounding_bound(f, pos.reshape(-1, 3)[:, :2], U32, M=M).reshape(pos.shape[:2])
    err = np.abs(F.astype(np.float64) - direct)
    assert np.all(err <= (drift[None, :] + rnd)[..., None]), float((err - (drift[None, :] + rnd)[..., None]).max())


def test_a_walk_of_an_isolated_cylinder_reads_the_closed_form_at_its_recorded_position():
    """An isolated myelinated cylinder's walk reads the field at the position it records: every raw walker's read at
    every block end is the float32 read at its recorded position, bit for bit."""
    import dmipy_sim as d
    g = d.MyelinatedCylinder(0.7e-6, 1e-6, (0, 0, 1), compartments=Compartments(intra=Pool(D=2e-9), extra=Pool(D=2e-9),
                                                                                myelin=Pool(D=0.0)))
    f = HollowCylinderField.of(g)
    w = _walk(g, f)
    F = _block_reads(w)
    direct = f.channels(w.raw_positions.reshape(-1, 3), np.float32).reshape(F.shape).astype(np.float32)
    np.testing.assert_array_equal(F, direct)
