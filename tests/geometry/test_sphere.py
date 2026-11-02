"""Restricted diffusion in a sphere: the signal is above free diffusion at the same b-values. The sphere's eigenmode
(matrix-method) agreement is held by test_coefficient_walk.py::test_sphere_and_cylinder_agree_with_the_matrix_method."""

import numpy as np
import numpy.testing as npt

from dmipy_sim import simulate, Sphere
from dmipy_replay import set_b
from tests.conftest import D, SEED, mc_atol

N_SPH = 8_000   # the sphere's wall rule sub-steps the walk; restricted and free signals differ by 0.4+, 10+ SE of two runs


# NOTE: `test_sphere_walkers_contained*` was removed (#88). It asserted two things:
# containment (max |r| < R) and that walkers reach the wall. Containment is now covered
# deterministically and exhaustively by `tests/test_wall_impacts.py`, which sweeps offset x
# direction x step length -- including the step-spans-the-object case this walk could not
# reach -- in milliseconds rather than minutes. "Walkers reach the wall" is a far weaker
# statement than the signal comparisons kept in this file: a restricted signal is
# itself proof that walkers are hitting walls.


def test_sphere_signal_above_free():
    """Restricted sphere signal must be >= free diffusion at same b-values."""
    from dmipy_sim import FreeDiffusion
    from dmipy_replay.sequences import pgse
    b_values = np.linspace(1e8, 3e9, 6)
    bvecs = np.tile([1., 0., 0.], (6, 1))
    wf = set_b(pgse(bvecs, 0.5e-3, 40e-3, gradient_strengths=1.0, n_t=1000, slew_rate=np.inf), b_values)

    S_sphere = simulate(N_SPH, D, wf, Sphere(5e-6), seed=SEED)
    S_free   = simulate(N_SPH, D, wf, FreeDiffusion(), seed=SEED + 1)

    assert np.all(S_sphere >= S_free - mc_atol(N_SPH, runs=2)), (
        f"Sphere signal should be >= free diffusion. "
        f"Max violation: {np.max(S_free - S_sphere):.4f}")
    assert np.max(S_sphere - S_free) > mc_atol(N_SPH, runs=2)      # and the restriction is there
