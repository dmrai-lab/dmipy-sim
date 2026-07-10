"""Free Gaussian diffusion, 3D PGSE, multiple gradient directions.

Validates isotropic diffusion: E = exp(-b*D) for all gradient directions.
"""

import numpy as np
import numpy.testing as npt

from dmipy_sim import simulate, pgse, FreeDiffusion, set_b
from .conftest import D, N_WALKERS, SEED


def test_free_diffusion_pgse_3d():
    delta = 8e-3
    DELTA = 50e-3
    n_t = 1000
    b_values_1d = np.linspace(1, 2e9, 10)

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

    for bvec in directions:
        bvecs = np.tile(bvec, (len(b_values_1d), 1))
        wf = set_b(pgse(delta=delta, DELTA=DELTA, G_magnitude=1.0,
                        bvecs=bvecs, n_t=n_t), b_values_1d)
        signals = simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED)
        E_ref = np.exp(-b_values_1d * D)
        npt.assert_allclose(
            signals, E_ref, atol=0.01,
            err_msg=f"Free diffusion 3D failed for direction {bvec}")
