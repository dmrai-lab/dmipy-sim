"""The SO(3) representation of a pose response: the basis, and the distributions composed against it (#157)."""
import numpy as np
import pytest

from dmipy_sim.replay import so3
from dmipy_sim.replay.fod import FOD


def test_the_rotation_matrices_are_the_representation_they_claim_to_be():
    """``M^l(R)`` is read off the action of ``R`` on the harmonics, so it must be orthogonal, multiplicative,
    and reproduce that action -- the three properties every later step leans on."""
    R = so3.haar_rotations(5, seed=1)
    lmax = 6
    blocks = so3.wigner_blocks(lmax, R)
    d = so3.haar_rotations(7, seed=2)[:, :, 2]                       # some directions
    for l, M in enumerate(blocks):
        assert M.shape == (5, 2 * l + 1, 2 * l + 1)
        for k in range(5):
            np.testing.assert_allclose(M[k] @ M[k].T, np.eye(2 * l + 1), atol=1e-9)
            # the defining property: Y_l(R d) = M^l(R) Y_l(d)
            Yl = lambda x: so3._sh_l(l, x)
            np.testing.assert_allclose(Yl(d @ R[k].T), Yl(d) @ M[k].T, atol=1e-9)
    # a homomorphism, which is what makes a composition of poses composable
    A, B = R[0], R[1]
    for l, (Ma, Mb, Mab) in enumerate(zip(so3.wigner_blocks(lmax, A[None]),
                                          so3.wigner_blocks(lmax, B[None]),
                                          so3.wigner_blocks(lmax, (A @ B)[None]))):
        np.testing.assert_allclose(Mab[0], Ma[0] @ Mb[0], atol=1e-9)


def test_the_basis_is_orthonormal_under_the_quadrature():
    """Peter-Weyl: ``sqrt(2l+1) D^l_{mn}`` are orthonormal in the normalised Haar measure, and the quadrature
    is exact for the band it is built for. Without this the composition is not an inner product."""
    lmax = 4
    R, w, _d, _r = so3.so3_quadrature(lmax)
    A = so3.so3_design(lmax, R)
    G = (A * w[:, None]).T @ A
    assert A.shape[1] == so3.n_so3_coeffs(lmax)
    np.testing.assert_allclose(G, np.eye(A.shape[1]), atol=1e-9)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)


def test_a_band_limited_response_is_recovered_exactly():
    """A function in the space is recovered by the quadrature projection with no solve, and reproduced at
    rotations off the grid: the projection is exact, not an interpolation."""
    lmax, nmax = 4, 2
    truth = np.random.default_rng(0).normal(size=so3.n_so3_coeffs(lmax, nmax))
    R, w, A = so3.quadrature_design(lmax, nmax)
    c = so3.project(A, w, A @ truth)
    np.testing.assert_allclose(c, truth, atol=1e-10)
    Q = so3.haar_rotations(20, seed=4)
    np.testing.assert_allclose(so3.evaluate(c, lmax, Q, nmax),
                               so3.so3_design(lmax, Q, nmax) @ truth, atol=1e-9)
    # a whole acquisition projects in one product, real or complex
    Y = A @ np.random.default_rng(1).normal(size=(truth.size, 3)) * (1 + 2j)
    np.testing.assert_allclose(so3.project(A, w, Y).shape, (truth.size, 3))


def test_one_pose_composes_to_the_response_at_that_pose():
    """The delta distribution is the consistency check of the whole scheme: composing it must return the
    response evaluated there, so a phantom slot holding a single pose needs no separate code path."""
    lmax, nmax = 6, 3
    c = np.random.default_rng(1).normal(size=so3.n_so3_coeffs(lmax, nmax))
    R0 = so3.haar_rotations(1, seed=5)[0]
    f = so3.delta_coeffs(R0, lmax, nmax)
    np.testing.assert_allclose(f @ c, float(so3.evaluate(c, lmax, R0[None], nmax)[0]), rtol=1e-9)


def test_a_distribution_over_directions_has_no_azimuthal_coefficients():
    """An ODF says where the axis points and nothing about the substrate's spin about it. Its coefficients must
    therefore vanish for every ``n != 0``, which is what makes the roll integral exact and free -- no sampling
    of rolls anywhere."""
    lmax, nmax = 6, 3
    f = so3.axis_density_coeffs(FOD.watson(8.0, mu=(0.3, 0.5, 0.81), lmax=6).coeffs, lmax, nmax)
    idx = so3.so3_index(lmax, nmax)
    off = np.array([abs(v) for (l, m, n), v in zip(idx, f) if n != 0])
    on = np.array([abs(v) for (l, m, n), v in zip(idx, f) if n == 0])
    assert off.max() < 1e-8 < on.max()
    assert abs(f[0] - 1.0) < 1e-9                                    # a density integrates to one
    # and the contract that matters: composing through SO(3) equals the sphere integral of the same ODF
    # against the same response, for any response the band can hold
    lmax_r = 4
    c = np.random.default_rng(2).normal(size=so3.n_so3_coeffs(lmax_r, nmax))
    fod = FOD.watson(2.0, mu=(0.3, 0.5, 0.81), lmax=6)               # broad enough to be positive in band
    f_r = so3.axis_density_coeffs(fod.coeffs, lmax_r, nmax)
    from dmipy_sim.replay.so3 import sphere_quadrature
    dirs, w = sphere_quadrature(24, 48)
    rho = fod.evaluate(dirs)
    assert rho.min() > 0                                             # no clipping, so the two sides agree exactly
    rolls = np.arange(16) * (2 * np.pi / 16)
    R = np.stack([so3.rotation_of(d, roll=r) for d in dirs for r in rolls])
    E = so3.evaluate(c, lmax_r, R, nmax).reshape(len(dirs), len(rolls))
    direct = float((w * rho * E.mean(axis=1)).sum() / (w * rho).sum())   # roll-averaged, ODF-weighted
    np.testing.assert_allclose(f_r @ c, direct, rtol=1e-6, atol=1e-9)


def test_watson_and_an_isotropic_bingham_are_the_same_distribution():
    """A Bingham with equal dispersions is a Watson: the anisotropic form must contain the isotropic one, or
    the two dispersion parameters do not mean what they claim."""
    lmax, nmax = 6, 2
    frame = so3.rotation_of((0.0, 0.0, 1.0))
    w8 = so3.watson_coeffs(8.0, mu=(0.0, 0.0, 1.0), lmax=lmax, nmax=nmax)
    b8 = so3.bingham_coeffs(frame, (8.0, 8.0), lmax=lmax, nmax=nmax)   # equal concentrations: the same cone
    np.testing.assert_allclose(b8, w8, atol=1e-8)
    # and an anisotropic one is genuinely different: a fan in one plane, not a cone
    fan = so3.bingham_coeffs(frame, (1.0, 40.0), lmax=lmax, nmax=nmax)
    assert np.abs(fan - w8).max() > 1e-2
    assert abs(fan[0] - 1.0) < 1e-9


def test_the_azimuthal_band_is_measured_not_asserted():
    """``energy`` is what chooses ``nmax``: a response with structure only in the pose direction puts all its
    energy at ``n = 0``, and one with azimuthal structure does not."""
    lmax, nmax = 4, 2
    idx = so3.so3_index(lmax, nmax)
    axial = np.array([1.0 if n == 0 else 0.0 for (_l, _m, n) in idx])
    per_l, per_n = so3.energy(axial, lmax, nmax)
    assert per_n[0] > 0 and np.allclose(per_n[1:], 0.0)
    assert per_l.sum() == pytest.approx(per_n.sum())
    skew = np.array([1.0 if n == 1 else 0.0 for (_l, _m, n) in idx])
    _pl, pn = so3.energy(skew, lmax, nmax)
    assert pn[1] > 0 and pn[0] == 0.0
