"""dmipy-fit as an oracle for the shared mathematics (#159).

The ecosystem's dependency runs one way -- fit and design import sim, sim imports neither -- so the base mathematics
of the orientation distributions and of the harmonic rotation blocks lives in sim and is written once; dmipy-fit
imports it and holds it to its own tests (``dmipy_fit/utils/tests/test_analytical_sh.py``: the coefficients, and the
Bingham density's projection onto them). What stays here is the loop closed on the definition: fit's fitting model
evaluates the Watson density, and sim's coefficients must be that density's projection. The test skips where
dmipy-fit is absent: it is an optional extra here, and nothing shipped in this package imports it.
"""
import numpy as np
import numpy.testing as npt
import pytest

from dmipy_sim.math.sh_analytical import watson_sh
from dmipy_sim.replay import so3

LMAX = 8


@pytest.mark.parametrize("kappa", [2.0, 8.0])
def test_the_coefficients_are_the_projection_of_the_fitting_model_density(kappa):
    """Compared as coefficients rather than pointwise, so the shared SH truncation cancels instead of setting the
    tolerance."""
    pytest.importorskip("dmipy_fit.distributions")
    from dmipy_fit.distributions.distributions import SD1Watson, kappa2odi
    model = SD1Watson(mu=np.array([0.0, 0.0]), odi=float(kappa2odi(kappa)))   # theta, phi: the z axis
    dirs, w = so3.sphere_quadrature(48, 96)
    Y = so3.real_sh(LMAX, dirs)
    c_model = (Y * (w * model(dirs))[:, None]).sum(axis=0)            # project the model's density
    npt.assert_allclose(watson_sh(np.array([0.0, 0.0, 1.0]), kappa, l_max=LMAX), c_model, atol=1e-9)
