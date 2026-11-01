"""Physics tests: surface relaxivity, the rho = 0 baseline.

With no surface relaxivity a b ~ 0 walk loses no signal: E(t) = 1 in every primitive geometry, at any echo time. The
Brownstein-Tarr recovery with relaxivity is held, against the exact finite-diffusion eigenvalue rather than the
fast-diffusion limit, by tests/validation/test_surface_relaxivity_ladder.py.
"""
import numpy as np
import numpy.testing as npt
import pytest

from dmipy_sim import simulate, Box1D, Cylinder, Sphere, set_b
from dmipy_sim.sequences import pgse

D_DEFAULT = 2e-9     # m^2/s
R_DEFAULT = 10e-6    # m
SEED = 42
N_BASELINE = 1_000   # every walker carries weight exactly 1 here, so any count shows a lost weight


def _b0_waveform(TE_s, n_t=500):
    """PGSE waveform with negligible b (b=1 s/m² → exp(-bD) ≈ 1); its lobes are ideal (square), the unit
    amplitude being a placeholder for the b-value that ``set_b`` fixes."""
    delta = max(TE_s * 0.05, 5e-6)
    DELTA = TE_s - delta
    bvecs = np.array([[1.0, 0.0, 0.0]])
    wf = set_b(
        pgse(bvecs, delta, DELTA, gradient_strengths=1.0, n_t=n_t, slew_rate=np.inf),
        np.array([1.0]))   # b=1 s/m² → attenuation = exp(-2e-9) ≈ 1
    actual_TE = wf.echo_idx * wf.dt
    return wf, actual_TE



@pytest.mark.parametrize("geometry", [Box1D(length=R_DEFAULT), Cylinder(radius=R_DEFAULT, orientation=[0, 0, 1.]),
                                      Sphere(radius=R_DEFAULT)], ids=["box1d", "cylinder", "sphere"])
def test_no_relaxivity_loses_no_signal(geometry):
    """A geometry without surface relaxivity: the signal at b ~ 0 is 1.0 at a 100 ms echo."""
    wf, _ = _b0_waveform(0.1)
    s = simulate(N_BASELINE, D_DEFAULT, wf, geometry, seed=SEED)
    npt.assert_allclose(float(s[0]), 1.0, atol=0.005)
