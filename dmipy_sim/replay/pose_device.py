"""The pose expansion's device kernels: what the closed form (:meth:`ReplayPack._pose_coeffs_closed_many`) computes over
every walker, formed on the device it can use, in walker chunks (dmrai-lab/dmipy-sim#449).

Every kernel has the host route as its oracle: the numpy computation it replaces is what the tests compare it to,
at float32 rounding (the products run at ``Precision.HIGHEST``; a float32 ``@`` on a CUDA device is TF32 otherwise).
"""
from __future__ import annotations

import functools

import numpy as np

from .compression import resolve_device


@functools.lru_cache(maxsize=8)
def _field_factor_kernel(n_q, n_cols):
    """``(a (c,), A (c, 3, 3), dirs (n_q, 3), Yw (n_q, n_cols)) -> F (c, n_cols)`` complex64: the harmonics of
    ``exp(i (a_w + u^T A_w u))`` over the sphere by the product quadrature, one chunk of walkers."""
    import jax
    import jax.numpy as jnp
    hi = jax.lax.Precision.HIGHEST

    @jax.jit
    def kernel(a, A, dirs, Yw):
        q = jnp.einsum("qa,wab,qb->wq", dirs, A, dirs, precision=hi)          # (c, n_q): the quadratic form per point
        ph = a[:, None] + q
        c, s = jnp.cos(ph), jnp.sin(ph)
        return jnp.matmul(c, Yw, precision=hi) + 1j * jnp.matmul(s, Yw, precision=hi)

    return kernel


def field_factor(a, A, dirs, Yw, *, device="auto", chunk_bytes=1 << 28):
    """The field factor's harmonics ``(n_w, n_cols)`` complex128 for every walker: ``F[w] = sum_q w_q Y(u_q) exp(i (a_w +
    u_q^T A_w u_q))`` with ``Yw = Y * w_q[:, None]`` the weighted harmonics of the quadrature points ``dirs``. On the
    device in chunks of walkers sized to ``chunk_bytes`` of phase; the numpy route when there is none, which holds
    one set of chunk buffers for the whole pass (the phase, its cosine and sine, the two real products) and writes
    every chunk into them: a fresh gigabyte per chunk is a million page faults, and on a host that charges for
    them the pass is paid to the kernel, not to the arithmetic (measured on an 8-core box at 224,000 walkers and
    924 quadrature points: 29 s with 13 s of system time and 1.6 million faults, against 6.8 s, 0.3 s and 5,000)."""
    a = np.asarray(a, np.float64); A = np.asarray(A, np.float64)
    dirs = np.asarray(dirs, np.float64); Yw = np.asarray(Yw, np.float64)
    n_w, n_q = a.shape[0], dirs.shape[0]
    n_c = Yw.shape[1]
    out = np.empty((n_w, n_c), np.complex128)
    kind, dev = route(device)
    if kind == "torch":
        return _field_factor_torch(a, A, dirs, Yw, dev, chunk_bytes)
    if kind == "numpy":
        step = max(1, min(n_w, int(chunk_bytes // (16 * n_q))))
        Q6 = _field_quadratic(dirs, np)                                                 # (n_q, 6): u^T A u = A6 . Q6
        A6 = np.stack([A[:, 0, 0], A[:, 1, 1], A[:, 2, 2], A[:, 0, 1], A[:, 0, 2], A[:, 1, 2]], 1)   # (n_w, 6)
        ph = np.empty((step, n_q)); c = np.empty((step, n_q)); s = np.empty((step, n_q))
        re = np.empty((step, n_c)); im = np.empty((step, n_c))
        for lo in range(0, n_w, step):
            sl = slice(lo, min(lo + step, n_w)); n = sl.stop - sl.start
            np.matmul(A6[sl], Q6.T, out=ph[:n]); ph[:n] += a[sl, None]
            np.cos(ph[:n], out=c[:n]); np.sin(ph[:n], out=s[:n])
            np.matmul(c[:n], Yw, out=re[:n]); np.matmul(s[:n], Yw, out=im[:n])
            out[sl].real = re[:n]; out[sl].imag = im[:n]
        return out
    import jax.numpy as jnp
    kernel = _field_factor_kernel(int(n_q), int(Yw.shape[1]))
    d_dirs = jnp.asarray(dirs, jnp.float32); d_Yw = jnp.asarray(Yw, jnp.float32)
    step = max(1, int(chunk_bytes // (8 * n_q)))                                   # float32 phase + its cos and sin
    for lo in range(0, n_w, step):
        sl = slice(lo, min(lo + step, n_w))
        out[sl] = np.asarray(kernel(jnp.asarray(a[sl], jnp.float32), jnp.asarray(A[sl], jnp.float32), d_dirs, d_Yw),
                             np.complex128)
    return out


def _field_quadratic(bs, xp):
    """``(c, 6)``: the quadratic form of the field directions ``bs`` ``(c, 3)`` in the substrate frame -- ``[x^2, y^2,
    z^2, 2xy, 2xz, 2yz]`` -- against which a path channel's six field moments contract; ``xp`` is numpy or jax.numpy."""
    return xp.stack([bs[:, 0] ** 2, bs[:, 1] ** 2, bs[:, 2] ** 2, 2 * bs[:, 0] * bs[:, 1],
                     2 * bs[:, 0] * bs[:, 2], 2 * bs[:, 1] * bs[:, 2]], axis=1)


@functools.lru_cache(maxsize=8)
def _samples_kernel(n_meas, with_field):
    """``(Rc (c, 3, 3), Q (w, n_meas, 9), ew (w), [b (3), s_loc (w), P6 (w, 6), A6 (w, 6) or zeros, k_iso, k_aniso])
    -> (E_re, E_im) (c, n_meas)`` float32: the ensemble signal of every measurement at every rotation of the chunk,
    the walkers' phases ``<R, M_w>`` as one contraction and the field's quadratic form in the rotated field
    direction when the pack carries one."""
    import jax
    import jax.numpy as jnp
    hi = jax.lax.Precision.HIGHEST

    @jax.jit
    def kernel(Rc, Q, ew, b, s_loc, P6, A6, k_iso, k_aniso):
        c = Rc.shape[0]
        ph = jnp.einsum("ck,wmk->cwm", Rc.reshape(c, 9), Q, precision=hi)          # (c, w, n_meas) radians
        if with_field:
            bs = jnp.einsum("cji,j->ci", Rc, b, precision=hi)                       # the field in the substrate frame
            Qf = _field_quadratic(bs, jnp)                                          # (c, 6)
            phi = k_iso * (s_loc[None, :] - jnp.matmul(Qf, P6.T, precision=hi)) + k_aniso * jnp.matmul(Qf, A6.T, precision=hi)
            ph = ph + phi[:, :, None]
        wc, ws = ew[None, :, None] * jnp.cos(ph), ew[None, :, None] * jnp.sin(ph)
        return wc.sum(1), ws.sum(1)

    return kernel


def pose_samples(R, Q, ew, norm, *, field=None, device="auto", chunk_bytes=1 << 30):
    """The ensemble signal ``(n_R, n_meas)`` complex128 of every measurement at every rotation of ``R``
    ``(n_R, 3, 3)``: ``E[r, i] = sum_w ew_w exp(i (<R_r, M_w,i> + phi_w(R_r))) / norm`` with ``Q`` the walkers'
    moment tensors ``(n_w, n_meas, 3, 3)`` and, when ``field`` is given, ``phi`` the susceptibility phase
    ``(k_iso, k_aniso, b, s_loc (n_w,), P6 (n_w, 6), A6 (n_w, 6) or None)`` in the rotated field direction.
    The quadrature route's sampling (:meth:`ReplayPack._pose_coeffs`): on the device in chunks of rotations and
    walkers sized to ``chunk_bytes`` of phase, every chunk's partial sum accumulated on the host in float64; the
    numpy route when there is none."""
    R = np.asarray(R, np.float64); Q = np.asarray(Q, np.float64); ew = np.asarray(ew, np.float64)
    n_R, n_w, n_meas = R.shape[0], Q.shape[0], Q.shape[1]
    E = np.zeros((n_R, n_meas), np.complex128)
    if field is not None:
        k_iso, k_aniso, b, s_loc, P6, A6 = field
        b = np.asarray(b, np.float64); b = b / np.linalg.norm(b)
        s_loc = np.asarray(s_loc, np.float64); P6 = np.asarray(P6, np.float64)
        A6 = np.zeros_like(P6) if A6 is None else np.asarray(A6, np.float64)
    if resolve_device(device) == "numpy":
        step = max(1, int(chunk_bytes // (16 * n_w * n_meas)))
        for lo in range(0, n_R, step):
            sl = slice(lo, min(lo + step, n_R)); Rc = R[sl]
            if field is None:
                Ew = np.broadcast_to(ew[None, :].astype(np.complex128), (Rc.shape[0], n_w))
            else:
                bs = np.einsum("nji,j->ni", Rc, b)
                Qf = _field_quadratic(bs, np)
                phi = float(k_iso) * (s_loc[None, :] - Qf @ P6.T) + float(k_aniso) * (Qf @ A6.T)
                Ew = np.exp(1j * phi) * ew[None, :]
            for i in range(n_meas):
                E[sl, i] = (Ew * np.exp(1j * np.einsum("nab,wab->nw", Rc, Q[:, i]))).sum(1) / norm
        return E
    import jax.numpy as jnp
    kernel = _samples_kernel(int(n_meas), field is not None)
    Q9 = Q.reshape(n_w, n_meas, 9)
    w_step = max(1, int(chunk_bytes // (8 * 64 * n_meas)))                           # walkers per 64-rotation chunk
    w_step = min(w_step, n_w); r_step = max(1, int(chunk_bytes // (8 * w_step * n_meas)))
    zeros3 = jnp.zeros(3, jnp.float32)
    for wlo in range(0, n_w, w_step):
        ws = slice(wlo, min(wlo + w_step, n_w))
        d_Q = jnp.asarray(Q9[ws], jnp.float32); d_ew = jnp.asarray(ew[ws], jnp.float32)
        if field is None:
            d_b, d_s, d_P, d_A, ki, ka = zeros3, jnp.zeros(ws.stop - ws.start, jnp.float32), jnp.zeros((ws.stop - ws.start, 6), jnp.float32), jnp.zeros((ws.stop - ws.start, 6), jnp.float32), 0.0, 0.0
        else:
            d_b = jnp.asarray(b, jnp.float32); d_s = jnp.asarray(s_loc[ws], jnp.float32)
            d_P = jnp.asarray(P6[ws], jnp.float32); d_A = jnp.asarray(A6[ws], jnp.float32); ki, ka = float(k_iso), float(k_aniso)
        for lo in range(0, n_R, r_step):
            sl = slice(lo, min(lo + r_step, n_R))
            re, im = kernel(jnp.asarray(R[sl], jnp.float32), d_Q, d_ew, d_b, d_s, d_P, d_A, jnp.float32(ki), jnp.float32(ka))
            E[sl] += (np.asarray(re, np.float64) + 1j * np.asarray(im, np.float64)) / norm
    return E


@functools.lru_cache(maxsize=16)
def _real_sh_kernel(L):
    """``dirs (n, 3) -> (n, (L+1)^2)`` float32: :func:`so3.real_sh` (full layout) as one jitted pass of the same
    three-term recurrences on the fully normalised associated Legendre functions."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def kernel(dirs):
        x = jnp.clip(dirs[:, 2], -1.0, 1.0)
        phi = jnp.arctan2(dirs[:, 1], dirs[:, 0])
        s = jnp.sqrt(jnp.clip(1.0 - x * x, 0.0, 1.0))
        P = {}
        P[(0, 0)] = jnp.full(dirs.shape[0], 1.0 / np.sqrt(4.0 * np.pi), jnp.float32)
        for m in range(1, L + 1):
            P[(m, m)] = -np.float32(np.sqrt((2.0 * m + 1.0) / (2.0 * m))) * s * P[(m - 1, m - 1)]
        for m in range(0, L):
            P[(m + 1, m)] = np.float32(np.sqrt(2.0 * m + 3.0)) * x * P[(m, m)]
        for m in range(0, L + 1):
            for l in range(m + 2, L + 1):
                a = np.float32(np.sqrt((4.0 * l * l - 1.0) / (l * l - m * m)))
                b = np.float32(np.sqrt(((l - 1.0) ** 2 - m * m) / (4.0 * (l - 1.0) ** 2 - 1.0)))
                P[(l, m)] = a * x * P[(l - 1, m)] - a * b * P[(l - 2, m)]
        cols = []
        r2 = np.float32(np.sqrt(2.0))
        for l in range(L + 1):
            block = [None] * (2 * l + 1)
            block[l] = P[(l, 0)]
            for m in range(1, l + 1):
                cm, sm = jnp.cos(m * phi), jnp.sin(m * phi)
                block[l + m] = r2 * P[(l, m)] * cm
                block[l - m] = r2 * P[(l, m)] * sm
            cols.extend(block)
        return jnp.stack(cols, axis=1)

    return kernel


def real_sh(L, dirs, *, device="auto", chunk_bytes=1 << 30):
    """Orthonormal real spherical harmonics ``(n, (L+1)^2)`` of unit directions ``dirs`` ``(n, 3)`` in the full layout of
    :func:`so3.real_sh`: on the device in chunks of directions, float64 on the host; numpy's when there is none."""
    from . import so3
    dirs = np.asarray(dirs, np.float64).reshape(-1, 3)
    if resolve_device(device) == "numpy":
        return so3.real_sh(int(L), dirs, full=True)
    import jax.numpy as jnp
    kernel = _real_sh_kernel(int(L))
    n_cols = (int(L) + 1) ** 2
    out = np.empty((dirs.shape[0], n_cols), np.float64)
    step = max(1, int(chunk_bytes // (4 * (n_cols + 8))))
    for lo in range(0, dirs.shape[0], step):
        sl = slice(lo, min(lo + step, dirs.shape[0]))
        out[sl] = np.asarray(kernel(jnp.asarray(dirs[sl], jnp.float32)), np.float64)
    return out


@functools.lru_cache(maxsize=16)
def _spherical_jn_kernel(L, N):
    """``x (n,) -> (L+1, n)`` float32: ``j_0..j_L(x)`` by the downward (Miller) recurrence from order ``N``, normalised
    as the host's (to ``j_0``, or to ``j_1`` near a zero of ``j_0``), as one jitted scan with the same rescaling
    against overflow as the host's."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def kernel(x):
        small = jnp.abs(x) < 1e-6
        xs = jnp.where(small, 1.0, x)
        hi, lo = jnp.zeros_like(xs), jnp.full_like(xs, 1e-30)

        def body(carry, l):
            hi, lo, out = carry
            cur = (2 * l + 3) / xs * lo - hi
            big = jnp.abs(cur) > 1e18
            scale = jnp.where(big, 1e-18, 1.0)
            out = out * scale[None, :]
            out = jnp.where((jnp.arange(out.shape[0]) == l)[:, None], cur[None, :] * scale[None, :], out)
            return (lo * scale, cur * scale, out), None

        out0 = jnp.zeros((N + 1, x.shape[0]), jnp.float32)
        (hi, lo, out), _ = jax.lax.scan(body, (hi, lo, out0), jnp.arange(N, -1, -1))
        j0 = jnp.where(small, 1.0, jnp.sin(xs) / xs)                   # j_0, or j_1 near a zero of j_0 (#604)
        j1 = jnp.sin(xs) / (xs * xs) - jnp.cos(xs) / xs
        use1 = (jnp.abs(j1) > jnp.abs(j0)) & ~small
        den = jnp.where(use1, out[1], out[0])
        scale = jnp.where(use1, j1, j0) / jnp.where(den == 0, 1.0, den)
        out = out[:L + 1] * scale[None, :]
        out = jnp.where(small[None, :], jnp.zeros_like(out).at[0].set(1.0), out)
        return out

    return kernel


def spherical_jn_all(L, x, *, device="auto", extra=24, chunk_bytes=1 << 30):
    """``j_0(x) .. j_L(x)`` for every entry of ``x``, ``(L+1,) + x.shape`` float64: :func:`so3.spherical_jn_all`
    on the device in chunks, numpy's when there is none."""
    from . import so3
    x = np.asarray(x, np.float64)
    if resolve_device(device) == "numpy":
        return so3.spherical_jn_all(L, x, extra=extra)
    import jax.numpy as jnp
    L = int(L); N = L + int(extra) + int(np.ceil(np.abs(x).max())) if x.size else L + int(extra)
    kernel = _spherical_jn_kernel(L, int(N))
    flat = x.reshape(-1)
    out = np.empty((L + 1, flat.size), np.float64)
    step = max(1, int(chunk_bytes // (4 * (N + 4))))
    for lo in range(0, flat.size, step):
        sl = slice(lo, min(lo + step, flat.size))
        out[:, sl] = np.asarray(kernel(jnp.asarray(flat[sl], jnp.float32)), np.float64)
    return out.reshape((L + 1,) + x.shape)


@functools.lru_cache(maxsize=8)
def _field_bodies_kernel(nc, L, l_used, N_bessel, n_f):
    """``(kappa (c, nc), m_hat (c, nc, 3), w (c,), F_re (c, n_f), F_im (c, n_f)) -> (B_re, B_im) (rows, n_f)`` float32 with
    ``rows = nc * sum(2l+1 for l in l_used)``, rows ordered ``(l, g, n)``: the stacked bodies of every gradient order
    and group against the field factor for one chunk of walkers, the Bessel values and the harmonics formed on the
    device and never returned."""
    import jax
    import jax.numpy as jnp
    hi = jax.lax.Precision.HIGHEST
    sh = _real_sh_kernel(L)
    jn = _spherical_jn_kernel(max(l_used), N_bessel)

    @jax.jit
    def kernel(kappa, m_hat, w, F_re, F_im):
        c = kappa.shape[0]
        Y = sh(m_hat.reshape(-1, 3)).reshape(c, nc, (L + 1) ** 2)                  # (c, nc, (L+1)^2)
        J = jn(kappa.reshape(-1)).reshape(-1, c, nc)                               # (lmax+1, c, nc)
        cols = []
        for l in l_used:
            Xl = (w[:, None] * J[l])[:, :, None] * Y[:, :, l * l:l * l + 2 * l + 1]  # (c, nc, 2l+1)
            cols.append(Xl.reshape(c, -1))
        X = jnp.concatenate(cols, axis=1)                                           # (c, rows)
        return jnp.matmul(X.T, F_re, precision=hi), jnp.matmul(X.T, F_im, precision=hi)

    return kernel


def host_bodies(kappa, m_hat, w, L, l_used, keep_n=None, J=None):
    """``{l: (n_w, nc, 2k+1)}``: the bodies ``w_w j_l(kappa_wg) Y_ln(m^_wg)`` of every walker and group for the orders
    ``l_used``, ``n`` within ``keep_n`` of zero (every ``2l+1`` column when ``None``), on the host. The gradient-only
    expansion sums them over the walkers; the field expansion's host route contracts them against the field factor.
    ``J`` ``(L+1, n_w, nc)`` supplies the Bessel values when a device already formed them."""
    from . import so3
    kappa = np.asarray(kappa, np.float64); m_hat = np.asarray(m_hat, np.float64); w = np.asarray(w, np.float64)
    n_w, nc = kappa.shape
    L = int(L)
    Y = so3.real_sh(L, m_hat.reshape(-1, 3), full=True).reshape(n_w, nc, (L + 1) ** 2)
    J = so3.spherical_jn_all(max(l_used), kappa) if J is None else np.asarray(J)
    out = {}
    for l in l_used:
        k = so3._n_cols(l, keep_n) // 2
        out[l] = (w[:, None] * J[l])[:, :, None] * Y[:, :, l * l + l - k:l * l + l + k + 1]
    return out


def paired_products(X, row_n, F_re, F_im, col_n, keep_n):
    """``X^T F`` restricted to the products a retained azimuthal band couples, ``(rows, n_f)`` complex: entry
    ``(r, j)`` formed only where ``|row_n[r] - col_n[j]| <= keep_n`` and zero elsewhere. Two real harmonics of
    azimuthal orders ``|n|`` and ``|N|`` multiply into orders ``|n| + |N|`` and ``||n| - |N||`` only, so a coupling
    into ``|N_c| <= keep_n`` reads no other pair: an ODF composition (``keep_n = 0``) needs the diagonal
    ``|N| = |n|`` alone. ``X`` is ``(n_w, rows)`` real with ``row_n`` each row's ``|n|``; ``col_n`` is each factor
    column's ``|N|`` in ascending order (the caller sorts the factor's columns once), so every product reads
    contiguous columns: one product per ``|n|``, against the factor columns it meets."""
    col_n = np.asarray(col_n); row_n = np.asarray(row_n)
    if np.any(np.diff(col_n) < 0):
        raise ValueError("paired_products reads the factor's columns sorted by |N|")
    order = np.argsort(row_n, kind="stable")
    Xs = np.ascontiguousarray(X[:, order]); row_s = row_n[order]
    out = np.zeros((X.shape[1], F_re.shape[1]), np.complex128)
    for a in np.unique(row_s):
        r0, r1 = np.searchsorted(row_s, [a, a + 1])
        c0, c1 = np.searchsorted(col_n, [a - keep_n, a + keep_n + 1])
        if c1 > c0:
            Xa = Xs[:, r0:r1]
            out[r0:r1, c0:c1] = (Xa.T @ F_re[:, c0:c1]) + 1j * (Xa.T @ F_im[:, c0:c1])
    back = np.empty_like(order); back[order] = np.arange(order.size)
    return out[back]


def paired_bodies(kappa, m_hat, w, F_re, F_im, L, l_used, col_n, keep_n):
    """:func:`field_bodies`' host route restricted to the products a retained azimuthal band couples
    (:func:`paired_products`): the same ``(rows, n_f)`` layout, zero where the band couples nothing."""
    bodies = host_bodies(kappa, m_hat, w, L, l_used)
    n_w, nc = np.asarray(kappa).shape
    X = np.concatenate([bodies[l].reshape(n_w, -1) for l in l_used], axis=1)          # (n_w, rows), field_bodies' order
    row_n = np.concatenate([np.tile(np.abs(np.arange(-l, l + 1)), nc) for l in l_used])
    return paired_products(X, row_n, F_re, F_im, col_n, keep_n)


def field_bodies(kappa, m_hat, w, F_re, F_im, L, l_used, *, n_bessel, device="auto", chunk_bytes=1 << 30):
    """``B[(l, g, n), (l', m')] = sum_w w_w j_l(kappa_wg) Y_ln(m^_wg) F_w,l'm'`` for the groups of one chunk, ``(rows, n_f)``
    complex128, ``rows = nc * sum(2l+1 for l in l_used)``: the closed form's bodies against the field factor. On the
    device per walker chunk (:func:`_field_bodies_kernel`), the partial sums accumulated on the host in float64; the
    host route (numpy harmonics, Bessel values and products) when there is none."""
    kappa = np.asarray(kappa, np.float64); m_hat = np.asarray(m_hat, np.float64); w = np.asarray(w, np.float64)
    F_re = np.asarray(F_re, np.float64); F_im = np.asarray(F_im, np.float64)
    n_w, nc = kappa.shape
    l_used = tuple(int(l) for l in l_used)
    rows = nc * sum(2 * l + 1 for l in l_used)
    kind, dev = route(device)
    if kind == "torch":
        B = field_bodies_torch(kappa, m_hat, w, F_re, F_im, L, l_used, n_bessel=n_bessel, device=dev,
                               chunk_bytes=chunk_bytes).cpu().numpy()                      # (nc, R, n_f)
        offs = np.cumsum([0] + [2 * l + 1 for l in l_used])
        return np.concatenate([B[:, offs[i]:offs[i + 1]].reshape(-1, B.shape[2]) for i in range(len(l_used))])
    if kind == "numpy":
        bodies = host_bodies(kappa, m_hat, w, L, l_used)
        X = np.concatenate([bodies[l].reshape(n_w, -1) for l in l_used], axis=1)
        return (X.T @ F_re) + 1j * (X.T @ F_im)
    import jax.numpy as jnp
    kernel = _field_bodies_kernel(int(nc), int(L), l_used, int(n_bessel), int(F_re.shape[1]))
    out = np.zeros((rows, F_re.shape[1]), np.complex128)
    per_walker = 4 * (nc * ((int(L) + 1) ** 2 + int(n_bessel) + 4) + rows + 2 * F_re.shape[1])
    step = max(1, int(chunk_bytes // per_walker))
    for lo in range(0, n_w, step):
        sl = slice(lo, min(lo + step, n_w))
        re, im = kernel(jnp.asarray(kappa[sl], jnp.float32), jnp.asarray(m_hat[sl], jnp.float32), jnp.asarray(w[sl], jnp.float32),
                        jnp.asarray(F_re[sl], jnp.float32), jnp.asarray(F_im[sl], jnp.float32))
        out += np.asarray(re, np.float64) + 1j * np.asarray(im, np.float64)
    return out


@functools.lru_cache(maxsize=8)
def _bessel_tails_kernel(L_hi, N):
    """``(kappa (c, nc), w (c,)) -> (L_hi+1, nc)`` float32: ``sum_w |w_w| |j_l(kappa_wg)|`` for every order and group of
    one chunk of walkers -- what the band and the tail bound of the closed form read; the values never return."""
    import jax
    import jax.numpy as jnp
    jn = _spherical_jn_kernel(L_hi, N)

    @jax.jit
    def kernel(kappa, w):
        c, nc = kappa.shape
        J = jn(kappa.reshape(-1)).reshape(-1, c, nc)                               # (L_hi+1, c, nc)
        return jnp.einsum("lwg,w->lg", jnp.abs(J), jnp.abs(w))

    return kernel


def bessel_tails(kappa, w, L_hi, *, device="auto", chunk_bytes=1 << 30):
    """``(L_hi+1, n_grp)`` float64: ``sum_w |w_w| |j_l(kappa_wg)|`` per order and group, the weighted Bessel magnitudes the
    closed form's band and tail bound read. On the device per walker chunk, summed on the host in float64; numpy's
    recurrence when there is none."""
    from . import so3
    kind, dev = route(device)
    if kind == "torch":
        return _bessel_tails_torch(kappa, w, int(L_hi), dev, chunk_bytes)
    kappa = np.asarray(kappa, np.float64); w = np.asarray(w, np.float64)
    if kind == "numpy":
        J = so3.spherical_jn_all(int(L_hi), kappa)
        return np.einsum("lwg,w->lg", np.abs(J), np.abs(w))
    import jax.numpy as jnp
    n_w, nc = kappa.shape
    N = int(L_hi) + 24 + int(np.ceil(np.abs(kappa).max())) if kappa.size else int(L_hi) + 24
    kernel = _bessel_tails_kernel(int(L_hi), N)
    out = np.zeros((int(L_hi) + 1, nc), np.float64)
    step = max(1, int(chunk_bytes // (4 * nc * (N + 4))))
    for lo in range(0, n_w, step):
        sl = slice(lo, min(lo + step, n_w))
        out += np.asarray(kernel(jnp.asarray(kappa[sl], jnp.float32), jnp.asarray(w[sl], jnp.float32)), np.float64)
    return out


# ---- torch ---------------------------------------------------------------------------------------------------------
# The same kernels on torch (dmrai-lab/dmipy-sim#603): Hugging Face's shared GPU pool runs PyTorch only. A device word
# "torch" (CUDA when torch sees it, else its CPU) or "torch:<device>" selects them; every float32 product runs at
# IEEE precision (a float32 matmul on CUDA may otherwise be TF32), and every sum over walkers is accumulated in
# float64.

def route(device):
    """The kernels' route for a device word: ``("numpy", None)``, ``("jax", None)`` or ``("torch", torch.device)``.
    ``"auto"``, ``"numpy"`` and ``"jax"`` are :func:`~dmipy_sim.replay.compression.resolve_device`'s; ``"torch"`` is
    torch on CUDA when it is available, else on its CPU, and ``"torch:<device>"`` names the torch device."""
    if isinstance(device, str) and (device == "torch" or device.startswith("torch:")):
        import torch
        name = device[6:] or ("cuda" if torch.cuda.is_available() else "cpu")
        return "torch", torch.device(name)
    return resolve_device(device), None


TORCH_CHUNK_BYTES = 1 << 32
"""Bytes of float32 temporaries a torch kernel holds per walker chunk (the harmonics, the Bessel values and the
stacked bodies): the chunk count, not the arithmetic, is what a torch route pays the host for -- each chunk launches
the recurrences' few thousand small kernels -- so the chunks are as large as a shared GPU's slice allows."""


TORCH_SUM_WALKERS = 1024
"""Walkers a float32 product sums before its sum joins the float64 accumulator: a float32 sum's rounding grows with
its length, and a chunk of the walkers is thousands long."""


class ieee:
    """A context in which torch's float32 matmuls run at IEEE float32 (``set_float32_matmul_precision("highest")``),
    the caller's setting restored on exit."""

    def __enter__(self):
        import torch
        self._was = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        return self

    def __exit__(self, *exc):
        import torch
        torch.set_float32_matmul_precision(self._was)
        return False


@functools.lru_cache(maxsize=16)
def _legendre_steps(L):
    """``(2, L+1, L+1)``: the coefficients of the step in ``l`` at every ``m``, ``P_lm = A_lm x P_(l-1)m - B_lm
    P_(l-2)m`` -- the standard recurrence for ``m <= l - 2``, ``A = sqrt(2l + 1)``, ``B = 0`` one step off the diagonal
    (``m = l - 1``), zero beyond."""
    out = np.zeros((2, L + 1, L + 1))
    for l in range(1, L + 1):
        lo = np.arange(max(l - 1, 0), dtype=np.float64)
        out[0, l, :l - 1] = np.sqrt((4.0 * l * l - 1.0) / (l * l - lo * lo))
        out[1, l, :l - 1] = out[0, l, :l - 1] * np.sqrt(((l - 1.0) ** 2 - lo * lo) / (4.0 * (l - 1.0) ** 2 - 1.0))
        out[0, l, l - 1] = np.sqrt(2.0 * l + 1.0)
    return out


@functools.lru_cache(maxsize=16)
def _sh_columns(L):
    """``(2, (L+1)^2)``: per column of the full layout, its entry ``l (L+1) + |m|`` of the flattened ``P[l, m]`` and
    its azimuthal factor's place ``m + L``."""
    l_of = np.concatenate([np.full(2 * l + 1, l) for l in range(L + 1)])
    m_of = np.concatenate([np.arange(-l, l + 1) for l in range(L + 1)])
    return np.stack([l_of * (L + 1) + np.abs(m_of), m_of + L])


def real_sh_torch(L, dirs):
    """:func:`so3.real_sh` (full layout) of unit directions ``dirs`` ``(n, 3)``, a torch tensor in its own dtype and
    device: the same recurrences on the fully normalised associated Legendre functions, each order's step taken for
    every ``m`` at once (``(n, L+1)`` per order) so the recurrence is ``L`` steps of a few kernels each, then every
    column gathered from them and its azimuthal factor in one pass."""
    import torch
    L = int(L)
    n = dirs.shape[0]
    x = dirs[:, 2].clamp(-1.0, 1.0)
    phi = torch.atan2(dirs[:, 1], dirs[:, 0])
    if x.dtype == torch.float32:                                             # 1 - x^2 cancels near a pole in float32
        s = torch.hypot(dirs[:, 0], dirs[:, 1]).clamp(0.0, 1.0)
    else:
        s = (1.0 - x * x).clamp(0.0, 1.0).sqrt()
    diag = [torch.full_like(x, 1.0 / np.sqrt(4.0 * np.pi))]                  # P_mm, by the sectoral recurrence
    for m in range(1, L + 1):
        diag.append(-float(np.sqrt((2.0 * m + 1.0) / (2.0 * m))) * s * diag[m - 1])
    diag = torch.stack(diag, dim=1)                                          # (n, L+1)
    P = torch.zeros((n, L + 1, L + 1), dtype=x.dtype, device=x.device)       # P[:, l, m]
    P[:, 0, 0] = diag[:, 0]
    AB = torch.as_tensor(_legendre_steps(L), dtype=x.dtype, device=x.device)  # (2, L+1, L+1), one transfer
    for l in range(1, L + 1):
        P[:, l] = (AB[0, l][None, :] * x[:, None]) * P[:, l - 1] - (AB[1, l][None, :] * P[:, l - 2] if l >= 2 else 0.0)
        P[:, l, l] = diag[:, l]
    ms = torch.arange(1, L + 1, dtype=x.dtype, device=x.device)
    trig = torch.cat([float(np.sqrt(2.0)) * torch.sin(phi[:, None] * ms.flip(0)), torch.ones_like(phi)[:, None],
                      float(np.sqrt(2.0)) * torch.cos(phi[:, None] * ms)], dim=1)           # (n, 2L+1): m = -L..L
    idx = torch.as_tensor(_sh_columns(L), device=x.device)                   # (2, (L+1)^2), one transfer
    return P.reshape(n, -1)[:, idx[0]] * trig[:, idx[1]]


def spherical_jn_torch(L, x, N):
    """``j_0(x) .. j_L(x)`` ``(L+1,) + x.shape`` for a torch tensor ``x`` in its own dtype and device: the downward
    (Miller) recurrence of :func:`so3.spherical_jn_all` from order ``N``, normalised to ``j_0 = sin x / x``. A value
    about to overflow is rescaled where it stands and the rescalings are counted per entry, so every stored order
    is brought to the final scale once at the end, with no host synchronisation inside the recurrence."""
    import torch
    L, N = int(L), int(N)
    f32 = x.dtype == torch.float32
    big, s = (1e18, 1e-18) if f32 else (1e200, 1e-200)
    small = x.abs() < (1e-6 if f32 else 1e-12)
    xs = torch.where(small, torch.ones_like(x), x)
    hi, lo = torch.zeros_like(xs), torch.full_like(xs, 1e-30)
    out = torch.empty((L + 1,) + tuple(x.shape), dtype=x.dtype, device=x.device)
    n_at = torch.empty((L + 1,) + tuple(x.shape), dtype=torch.int32, device=x.device)
    n = torch.zeros(x.shape, dtype=torch.int32, device=x.device)        # rescalings so far, per entry
    for l in range(N, -1, -1):
        cur = (2 * l + 3) / xs * lo - hi
        hi, lo = lo, cur
        if l <= L:
            out[l] = cur; n_at[l] = n                                  # this order's own rescaling counts below
        m = cur.abs() > big
        hi = torch.where(m, hi * s, hi); lo = torch.where(m, lo * s, lo)
        n = n + m.to(torch.int32)
    out = out * torch.pow(torch.full_like(out, s), (n[None] - n_at).to(out.dtype))   # each order to the final scale
    # normalised on j_0 = sin x / x, or on j_1 = sin x / x^2 - cos x / x where j_0 is the smaller: near a zero of
    # j_0 the ratio j_0 / out_0 loses the digits a float32 recurrence has
    j0 = torch.where(small, torch.ones_like(xs), torch.sin(xs) / xs)
    j1 = torch.sin(xs) / (xs * xs) - torch.cos(xs) / xs
    r1 = out[1] if L >= 1 else hi                                       # the recurrence's j_1, at the final scale
    use1 = (j1.abs() > j0.abs()) & ~small
    num = torch.where(use1, j1, j0); den = torch.where(use1, r1, out[0])
    scale = num / torch.where(den == 0, torch.ones_like(den), den)
    out = out * scale[None]
    if L >= 1:
        out[1:] = torch.where(small[None], torch.zeros_like(out[1:]), out[1:])
    out[0] = torch.where(small, torch.ones_like(out[0]), out[0])
    return out


def as_torch(a, device, dtype=None):
    """``a`` (an array or a tensor) as a torch tensor on ``device``, in ``dtype`` when given."""
    import torch
    return torch.as_tensor(a, device=device, dtype=dtype)


def _bessel_tails_torch(kappa, w, L_hi, dev, chunk_bytes):
    """:func:`bessel_tails` on torch: float32 values per walker chunk, the sums accumulated in float64."""
    import torch
    n_w, nc = kappa.shape
    k_max = float(kappa.max()) if n_w * nc else 0.0
    N = L_hi + 24 + int(np.ceil(abs(k_max)))
    out = torch.zeros((L_hi + 1, nc), dtype=torch.float64, device=dev)
    w_d = as_torch(w, dev, torch.float64).abs()
    step = max(1, int(chunk_bytes // (4 * nc * (2 * L_hi + 8))))
    with torch.no_grad():
        for lo in range(0, n_w, step):
            sl = slice(lo, min(lo + step, n_w))
            J = spherical_jn_torch(L_hi, as_torch(kappa[sl], dev, torch.float32), N)          # (L_hi+1, c, nc)
            out += torch.einsum("lwg,w->lg", J.abs().double(), w_d[sl])
    return out.cpu().numpy()


def _field_factor_torch(a, A, dirs, Yw, dev, chunk_bytes):
    """:func:`field_factor` on torch: the phase, its cosine and sine and the two products in float32 at IEEE
    precision, per walker chunk."""
    import torch
    n_w, n_q = a.shape[0], dirs.shape[0]
    out = np.empty((n_w, Yw.shape[1]), np.complex128)
    Q6 = as_torch(_field_quadratic(dirs, np), dev, torch.float32)                         # (n_q, 6)
    A6 = np.stack([A[:, 0, 0], A[:, 1, 1], A[:, 2, 2], A[:, 0, 1], A[:, 0, 2], A[:, 1, 2]], 1)
    Yd = as_torch(Yw, dev, torch.float32)
    step = max(1, int(chunk_bytes // (12 * n_q)))
    with torch.no_grad(), ieee():
        for lo in range(0, n_w, step):
            sl = slice(lo, min(lo + step, n_w))
            ph = as_torch(A6[sl], dev, torch.float32) @ Q6.T + as_torch(a[sl], dev, torch.float32)[:, None]
            re = torch.cos(ph) @ Yd; im = torch.sin(ph) @ Yd
            out[sl].real = re.double().cpu().numpy(); out[sl].imag = im.double().cpu().numpy()
    return out


def field_bodies_torch(kappa, m_hat, w, F_re, F_im, L, l_used, *, n_bessel, device, chunk_bytes=TORCH_CHUNK_BYTES):
    """:func:`field_bodies` on torch, kept on ``device``: ``(nc, R, n_f)`` complex128 with ``R = sum(2l+1 for l in
    l_used)`` (the orders side by side, each order's ``2l+1`` columns): the Bessel values, the harmonics and the
    products in float32 at IEEE precision per walker chunk, the sums over the walkers accumulated in float64.
    ``kappa`` ``(n_w, nc)``, ``m_hat`` ``(n_w, nc, 3)``, ``w`` ``(n_w,)``, ``F_re`` / ``F_im`` ``(n_w, n_f)``: arrays or
    tensors (on ``device`` already, they are read in place)."""
    import torch
    n_w, nc = kappa.shape
    l_used = tuple(int(l) for l in l_used)
    L = int(L); lmax = max(l_used)
    R = sum(2 * l + 1 for l in l_used)
    cols = torch.as_tensor(np.concatenate([np.arange(l * l, l * l + 2 * l + 1) for l in l_used]), device=device)
    ord_ = torch.as_tensor(np.concatenate([np.full(2 * l + 1, l) for l in l_used]), device=device)
    n_f = F_re.shape[1]
    out = torch.zeros((nc * R, n_f), dtype=torch.float64, device=device)
    out_i = torch.zeros((nc * R, n_f), dtype=torch.float64, device=device)
    per_walker = 4 * (nc * ((L + 1) ** 2 + int(n_bessel) + 2 * R + 4) + 2 * n_f)
    step = max(1, int(chunk_bytes // per_walker))
    with torch.no_grad(), ieee():
        for lo in range(0, n_w, step):
            sl = slice(lo, min(lo + step, n_w)); c = sl.stop - sl.start
            Y = real_sh_torch(L, as_torch(m_hat[sl], device, torch.float32).reshape(-1, 3)).reshape(c, nc, -1)
            J = spherical_jn_torch(lmax, as_torch(kappa[sl], device, torch.float32), n_bessel)     # (lmax+1, c, nc)
            X = (as_torch(w[sl], device, torch.float32)[:, None, None] * J.permute(1, 2, 0)[:, :, ord_]) * Y[:, :, cols]
            X = X.reshape(c, nc * R)                                                             # (c, nc R): (g, (l, n))
            Fr, Fi = as_torch(F_re[sl], device, torch.float32), as_torch(F_im[sl], device, torch.float32)
            for b in range(0, c, TORCH_SUM_WALKERS):                                             # float32 sums of a
                bs = slice(b, min(b + TORCH_SUM_WALKERS, c))                                     # block, float64 across
                out += (X[bs].T @ Fr[bs]).double()
                out_i += (X[bs].T @ Fi[bs]).double()
    return torch.complex(out, out_i).reshape(nc, R, n_f)
