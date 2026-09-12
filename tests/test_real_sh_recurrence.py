"""``so3.real_sh`` by recurrence equals the per-(l, m) ``lpmv`` evaluation it replaced, to 1e-12, at every order
and layout -- the basis is normative (RPH.md 4.1), so a faster evaluation must be the same numbers."""
import numpy as np
from scipy.special import gammaln, lpmv

from dmipy_sim.replay.so3 import n_sh_coeffs, real_sh, sh_block


def _real_sh_lpmv(lmax, dirs, full=False):
    d = np.asarray(dirs, np.float64).reshape(-1, 3)
    x = np.clip(d[:, 2], -1.0, 1.0); phi = np.arctan2(d[:, 1], d[:, 0])
    out = np.empty((d.shape[0], n_sh_coeffs(lmax, full)), np.float64)
    for l in range(0, lmax + 1, 1 if full else 2):
        col = out[:, sh_block(l, full)]
        col[:, l] = np.sqrt((2 * l + 1) / (4 * np.pi)) * lpmv(0, l, x)
        for m in range(1, l + 1):
            K = np.sqrt((2 * l + 1) / (4 * np.pi) * np.exp(gammaln(l - m + 1) - gammaln(l + m + 1)))
            P = lpmv(m, l, x)
            col[:, l + m] = np.sqrt(2.0) * K * P * np.cos(m * phi)
            col[:, l - m] = np.sqrt(2.0) * K * P * np.sin(m * phi)
    return out


def test_the_recurrence_is_the_lpmv_basis_to_1e12():
    rng = np.random.default_rng(0)
    d = rng.normal(size=(2000, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = np.concatenate([d, [[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0]]])          # the poles included
    for lmax in (0, 1, 4, 8, 13, 20):
        for full in (False, True):
            np.testing.assert_allclose(real_sh(lmax, d, full=full), _real_sh_lpmv(lmax, d, full=full), atol=1e-12, rtol=1e-12)


def test_the_basis_is_orthonormal_by_quadrature():
    from dmipy_sim.replay.so3 import sphere_quadrature
    dirs, w = sphere_quadrature(24, 48)
    Y = real_sh(12, dirs, full=True)
    np.testing.assert_allclose((Y * w[:, None]).T @ Y, np.eye(Y.shape[1]), atol=1e-10)
