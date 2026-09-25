"""The pose expansion's device kernels: what the closed form (:meth:`ReplayPack._pose_coeffs_closed`) computes over
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
def _products_kernel(n_rows, n_cols):
    """``(X (c, n_rows), F_re (c, n_cols), F_im (c, n_cols)) -> (X^T F_re, X^T F_im)`` float32 for one chunk of walkers."""
    import jax
    import jax.numpy as jnp
    hi = jax.lax.Precision.HIGHEST

    @jax.jit
    def kernel(X, F_re, F_im):
        Xt = X.T
        return jnp.matmul(Xt, F_re, precision=hi), jnp.matmul(Xt, F_im, precision=hi)

    return kernel


def field_products(X, F_re, F_im, *, device="auto", chunk_bytes=1 << 30):
    """``X^T (F_re + i F_im)`` over the walkers, ``(n_rows, n_cols)`` complex128: every gradient order's body against
    the field factor as one product. On the device per walker chunk (float32 at ``HIGHEST``), each chunk's partial
    sum accumulated on the host in float64 so that the sum over the walkers never rounds in float32; the numpy
    route when there is none."""
    X = np.asarray(X, np.float64); F_re = np.asarray(F_re, np.float64); F_im = np.asarray(F_im, np.float64)
    n_w = X.shape[0]
    if resolve_device(device) == "numpy":
        return (X.T @ F_re) + 1j * (X.T @ F_im)
    import jax.numpy as jnp
    kernel = _products_kernel(int(X.shape[1]), int(F_re.shape[1]))
    out = np.zeros((X.shape[1], F_re.shape[1]), np.complex128)
    step = max(1, int(chunk_bytes // (4 * (X.shape[1] + 2 * F_re.shape[1]))))
    for lo in range(0, n_w, step):
        sl = slice(lo, min(lo + step, n_w))
        re, im = kernel(jnp.asarray(X[sl], jnp.float32), jnp.asarray(F_re[sl], jnp.float32), jnp.asarray(F_im[sl], jnp.float32))
        out += np.asarray(re, np.float64) + 1j * np.asarray(im, np.float64)
    return out
