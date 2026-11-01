"""Free Gaussian diffusion, 3D PGSE, multiple gradient directions.

Validates isotropic diffusion: E = exp(-b*D) for all gradient directions.
"""

import numpy as np
import numpy.testing as npt

from dmipy_sim import simulate, pgse, FreeDiffusion, set_b
from .conftest import D, N_CPU, SEED, mc_atol


def test_free_diffusion_pgse_3d():
    delta = 8e-3
    DELTA = 50e-3
    n_t = 1000
    b_values_1d = np.linspace(1, 2e9, 4)
    # 6 gradient directions: axes + diagonals
    directions = np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 1., 0.],
        [1., 0., 1.],
        [0., 1., 1.],
    ])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    # one walk carries every (direction, b) as its own measurement
    bvecs = np.repeat(directions, len(b_values_1d), axis=0)
    b_all = np.tile(b_values_1d, len(directions))
    wf = set_b(pgse(bvecs, delta, DELTA, gradient_strengths=1.0, n_t=n_t), b_all)
    signals = simulate(N_CPU, D, wf, FreeDiffusion(), seed=SEED)
    npt.assert_allclose(signals, np.exp(-b_all * D), atol=mc_atol(N_CPU),
                        err_msg="Free diffusion 3D failed for a gradient direction")
