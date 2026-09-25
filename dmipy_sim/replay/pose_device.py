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


def field_factor(a, A, dirs, Yw, *, device="auto", chunk_bytes=1 << 30):
    """The field factor's harmonics ``(n_w, n_cols)`` complex128 for every walker: ``F[w] = sum_q w_q Y(u_q) exp(i (a_w +
    u_q^T A_w u_q))`` with ``Yw = Y * w_q[:, None]`` the weighted harmonics of the quadrature points ``dirs``. On the
    device in chunks of walkers sized to ``chunk_bytes`` of phase; the numpy route when there is none."""
    a = np.asarray(a, np.float64); A = np.asarray(A, np.float64)
    dirs = np.asarray(dirs, np.float64); Yw = np.asarray(Yw, np.float64)
    n_w, n_q = a.shape[0], dirs.shape[0]
    out = np.empty((n_w, Yw.shape[1]), np.complex128)
    if resolve_device(device) == "numpy":
        step = max(1, int(chunk_bytes // (16 * n_q)))
        for lo in range(0, n_w, step):
            sl = slice(lo, min(lo + step, n_w))
            q = np.einsum("qa,wab,qb->wq", dirs, A[sl], dirs)
            f = np.exp(1j * (a[sl, None] + q))
            out[sl] = (f.real @ Yw) + 1j * (f.imag @ Yw)
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
            Qf = jnp.stack([bs[:, 0] ** 2, bs[:, 1] ** 2, bs[:, 2] ** 2, 2 * bs[:, 0] * bs[:, 1],
                            2 * bs[:, 0] * bs[:, 2], 2 * bs[:, 1] * bs[:, 2]], axis=1)      # (c, 6)
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
                Qf = np.stack([bs[:, 0] ** 2, bs[:, 1] ** 2, bs[:, 2] ** 2, 2 * bs[:, 0] * bs[:, 1],
                               2 * bs[:, 0] * bs[:, 2], 2 * bs[:, 1] * bs[:, 2]], axis=1)
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
    to ``j_0 = sin x / x``, as one jitted scan with the same rescaling against overflow as the host's."""
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
        j0 = jnp.where(small, 1.0, jnp.sin(xs) / xs)
        scale = j0 / jnp.where(out[0] == 0, 1.0, out[0])
        out = out[:L + 1] * scale[None, :]
        out = jnp.where(small[None, :], jnp.zeros_like(out).at[0].set(1.0), out)
        return out

    return kernel


def spherical_jn_all(L, x, *, device="auto", extra=24, chunk_bytes=1 << 30):
    """``j_0(x) .. j_L(x)`` for every entry of ``x``, ``(L+1,) + x.shape`` float64: :func:`replay._spherical_jn_all`
    on the device in chunks, numpy's when there is none."""
    from .replay import _spherical_jn_all
    x = np.asarray(x, np.float64)
    if resolve_device(device) == "numpy":
        return _spherical_jn_all(L, x, extra=extra)
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
    if resolve_device(device) == "numpy":
        from . import so3
        from .replay import _spherical_jn_all
        Ym = so3.real_sh(int(L), m_hat.reshape(-1, 3), full=True).reshape(n_w, nc, (int(L) + 1) ** 2)
        J = _spherical_jn_all(max(l_used), kappa)
        X = np.concatenate([((w[:, None] * J[l])[:, :, None] * Ym[:, :, l * l:l * l + 2 * l + 1]).reshape(n_w, -1)
                            for l in l_used], axis=1)
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
    from .replay import _spherical_jn_all
    kappa = np.asarray(kappa, np.float64); w = np.asarray(w, np.float64)
    if resolve_device(device) == "numpy":
        J = _spherical_jn_all(int(L_hi), kappa)
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




@functools.lru_cache(maxsize=8)
def _prefix_decode_kernel(n_t, n_cut, K):
    """``C (c, K+2, 3) -> positions (c, n_cut, 3)`` float32: the first ``n_cut`` saves of a bridge-coded walk of
    ``n_t`` saves, the sine bands evaluated only where they are read."""
    import jax
    import jax.numpy as jnp
    hi = jax.lax.Precision.HIGHEST
    N = n_t - 2
    n = np.arange(1, N + 1)[None, :]; k = np.arange(1, K + 1)[:, None]
    Sk = np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * n * k / (N + 1))              # (K, N): DST-I, ortho, as scipy's idst reads it
    inner = min(n_cut, n_t - 1) - 1                                             # interior saves 1 .. n_cut-1 (the last save is a band end)
    S_part = jnp.asarray(Sk[:, :max(inner, 0)], jnp.float32)                    # (K, inner)
    tau = jnp.asarray(np.arange(n_cut) / (n_t - 1.0), jnp.float32)

    @jax.jit
    def kernel(C):
        a, v, B = C[:, 0, :], C[:, 1, :], C[:, 2:, :]
        u = jnp.einsum("wkd,kn->wnd", B, S_part, precision=hi)                  # (c, inner, 3)
        pad = jnp.zeros((C.shape[0], 1, 3), jnp.float32)
        u = jnp.concatenate([pad, u] + ([pad] if n_cut == n_t else []), axis=1)[:, :n_cut, :]
        return a[:, None, :] + v[:, None, :] * tau[None, :, None] + u

    return kernel


def decode_prefix(C, n_t, n_cut, *, device="auto", chunk_bytes=1 << 30):
    """The first ``n_cut`` saves ``(n_w, n_cut, 3)`` float32 of a bridge-coded walk of ``n_t`` saves from its coefficients
    ``C`` ``(n_w, K+2, 3)``: on the device per walker chunk; numpy's whole decode cut to ``n_cut`` when there is none."""
    from .compression import _bridge_positions
    C = np.asarray(C, np.float64)
    n_w, K = C.shape[0], C.shape[1] - 2
    if resolve_device(device) == "numpy":
        return _bridge_positions(C, int(n_t))[:, :int(n_cut), :].astype(np.float32)
    import jax.numpy as jnp
    kernel = _prefix_decode_kernel(int(n_t), int(n_cut), int(K))
    out = np.empty((n_w, int(n_cut), 3), np.float32)
    step = max(1, int(chunk_bytes // (4 * 3 * (int(n_cut) + K + 2))))
    for lo in range(0, n_w, step):
        sl = slice(lo, min(lo + step, n_w))
        out[sl] = np.asarray(kernel(jnp.asarray(C[sl], jnp.float32)), np.float32)
    return out
