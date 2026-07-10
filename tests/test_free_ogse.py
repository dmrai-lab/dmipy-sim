"""Free Gaussian diffusion, OGSE waveform.

For free diffusion E = exp(-b*D) regardless of waveform shape (D_eff = D).
Validates general waveform phase accumulation.
"""

import numpy as np
import numpy.testing as npt

from dmipy_sim import simulate, ogse, FreeDiffusion, set_b
from .conftest import D, N_WALKERS, SEED


def test_free_diffusion_ogse():
    T_total = 80e-3   # s
    frequency = 50.0  # Hz — 4 full cycles in 80 ms
    n_t = 1000
    n_b = 50

    b_values = np.linspace(1, 2e9, n_b)
    bvecs = np.tile([1.0, 0.0, 0.0], (n_b, 1))

    wf = set_b(ogse(frequency=frequency, T_total=T_total, G_magnitude=1.0,
                    bvecs=bvecs, n_t=n_t), b_values)

    signals = simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED)

    E_ref = np.exp(-b_values * D)
    npt.assert_allclose(signals, E_ref, atol=0.01,
                        err_msg="Free diffusion OGSE deviates from exp(-bD)")
