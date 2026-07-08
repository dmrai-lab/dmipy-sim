"""Free Gaussian diffusion, 1D PGSE.

Validates the scan loop, phase accumulation, and signal extraction against
the analytical result E = exp(-b*D).
"""

import numpy as np
import numpy.testing as npt

from dmipy_sim import simulate, pgse, FreeDiffusion, set_b, calc_b
from .conftest import D, N_WALKERS, SEED


def test_free_diffusion_pgse_1d():
    delta = 8e-3     # s
    DELTA = 50e-3    # s
    n_t = 1000
    n_b = 100

    b_values = np.linspace(1, 2e9, n_b)

    # Build PGSE waveform along x with unit amplitude, then scale to b-values
    bvecs = np.tile([1.0, 0.0, 0.0], (n_b, 1))
    wf_unit = pgse(delta=delta, DELTA=DELTA, G_magnitude=1.0, bvecs=bvecs, n_t=n_t)
    wf = set_b(wf_unit, b_values)

    # Verify b-values were set correctly
    b_check = calc_b(wf)
    npt.assert_allclose(b_check, b_values, rtol=1e-3)

    signals = simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED)

    E_ref = np.exp(-b_values * D)
    npt.assert_allclose(signals, E_ref, atol=0.01,
                        err_msg="Free diffusion 1D PGSE deviates from exp(-bD)")
