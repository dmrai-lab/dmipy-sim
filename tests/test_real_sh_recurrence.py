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


def test_two_real_wigner_blocks_multiply_through_the_coupling_tables():
    """``M^l1[m,n] M^l2[m',n'] = sum_L K_L[(m,m'),M] M^L[M,N] conj(K_L[(n,n'),N])`` with ``K_L`` from the complex
    Clebsch-Gordan coefficients and the real-from-complex unitaries: the product of two expansions on SO(3) is a
    contraction with these tables, the whole basis of composing a field with a gradient (#197 step 3)."""
    from dmipy_sim.replay import so3
    for (l1, l2) in [(1, 1), (2, 3), (4, 2), (6, 5), (9, 5)]:
        C = np.concatenate([so3.clebsch_gordan(l1, l2, L) for L in range(abs(l1 - l2), l1 + l2 + 1)], axis=1)
        np.testing.assert_allclose(C.T @ C, np.eye(C.shape[1]), atol=1e-10)          # CG: orthogonal
        np.testing.assert_allclose(C @ C.T, np.eye(C.shape[0]), atol=1e-10)
    Rs = so3.haar_rotations(3, 11)
    for (l1, l2) in [(3, 1), (4, 6), (9, 5)]:
        B1, B2 = so3.wigner_blocks(l1, Rs)[l1], so3.wigner_blocks(l2, Rs)[l2]
        BL = so3.wigner_blocks(l1 + l2, Rs)
        K = so3.coupling(l1, l2)
        for r in range(Rs.shape[0]):
            prod = sum(K[L] @ BL[L][r] @ K[L].conj().T for L in K)
            np.testing.assert_allclose(prod, np.kron(B1[r], B2[r]), atol=1e-12)
