"""dmipy-fit as an oracle for the shared mathematics (#159).

The ecosystem's dependency runs one way -- fit and design import sim, sim imports neither -- so the
base mathematics of the orientation distributions and of the harmonic rotation blocks lives in sim
and is written once. While fit still carries its own copies of those forms, these tests assert that
the two agree to machine precision, which is what makes removing fit's copies a provable no-op
rather than a hope. They skip where dmipy-fit is absent: it is an optional extra here, and nothing
shipped in this package imports it.
"""
import numpy as np
import numpy.testing as npt
import pytest

from dmipy_sim.math.sh_analytical import bingham_sh, watson_sh
from dmipy_sim.replay import so3

LMAX = 8


@pytest.fixture(scope="module")
def fit_sh():
    return pytest.importorskip("dmipy_fit.utils.sh_analytical",
                               reason="dmipy-fit (optional extra) is the oracle for these forms")


@pytest.mark.parametrize("kappa", [0.5, 2.0, 8.0, 16.0, 60.0])
def test_watson_agrees_with_fit(fit_sh, kappa):
    mu = np.array([0.3, 0.5, 0.81]); mu /= np.linalg.norm(mu)
    npt.assert_allclose(watson_sh(mu, kappa, l_max=LMAX),
                        fit_sh.watson_sh(mu, kappa, l_max=LMAX), atol=1e-14)


@pytest.mark.parametrize("k1,k2", [(8.0, 8.0), (16.0, 6.0), (20.0, 1.0)])
def test_bingham_agrees_with_fit(fit_sh, k1, k2):
    """The same distribution, reached from two parameterisations. Sim states a frame and one
    concentration about each of its first two axes; fit states ``(mu, psi, kappa, beta)`` with the
    peak along ``mu``. They map as ``kappa = k1``, ``beta = k1 - k2``, with sim's pose axis as
    ``mu`` and its second column as ``mu_beta``."""
    from dmipy_fit.utils.utils import rotation_matrix_100_to_theta_phi_psi
    R = rotation_matrix_100_to_theta_phi_psi(0.0, 0.0, 0.0)          # fit's frame for mu = z, psi = 0
    mu_beta = R @ np.array([0.0, 1.0, 0.0])
    mu = np.array([0.0, 0.0, 1.0])
    frame = np.stack([np.cross(mu_beta, mu), mu_beta, mu], axis=1)
    npt.assert_allclose(bingham_sh(frame, (k1, k2), l_max=LMAX),
                        fit_sh.bingham_sh(mu, 0.0, k1, k1 - k2, l_max=LMAX), atol=1e-12)


def test_the_rotation_blocks_agree_with_fit(fit_sh):
    """One implementation of the harmonic rotation blocks in the ecosystem. Sim's is batched over
    rotations and needs no dipy; fit's takes one rotation at a time through dipy's basis. They are
    the same matrices."""
    R = so3.haar_rotations(3, seed=3)
    mine = so3.wigner_blocks(LMAX, R)
    for k in range(R.shape[0]):
        theirs = fit_sh._sh_rotation_matrix_blocks(R[k], LMAX)
        for i, l in enumerate(range(0, LMAX + 1, 2)):
            npt.assert_allclose(mine[l][k], theirs[i], atol=1e-13)


@pytest.mark.parametrize("kappa", [2.0, 8.0])
def test_the_coefficients_are_the_projection_of_the_fitting_model_density(fit_sh, kappa):
    """Closing the loop on the definition rather than on an implementation: fit's fitting model
    evaluates the Watson density, and sim's coefficients must be that density's projection onto the
    basis. Compared as coefficients rather than pointwise, so the shared SH truncation cancels
    instead of setting the tolerance."""
    pytest.importorskip("dmipy_fit.distributions")
    from dmipy_fit.distributions.distributions import SD1Watson, kappa2odi
    model = SD1Watson(mu=np.array([0.0, 0.0]), odi=float(kappa2odi(kappa)))   # theta, phi: the z axis
    dirs, w = so3.sphere_quadrature(48, 96)
    Y = so3.real_sh(LMAX, dirs)
    c_model = (Y * (w * model(dirs))[:, None]).sum(axis=0)            # project the model's density
    npt.assert_allclose(watson_sh(np.array([0.0, 0.0, 1.0]), kappa, l_max=LMAX), c_model, atol=1e-9)
