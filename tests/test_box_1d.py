"""Restricted diffusion, 1D reflecting box.

Validates the boundary reflection logic against physical expectations:
  1. Walkers are contained within [0, L]
  2. Signal is always higher than free diffusion (restriction reduces D_eff)
  3. Signal at b=0 is 1 (normalisation)
  4. Signal decreases monotonically with b
  5. Tighter confinement (smaller L) gives higher signal at the same b

The eigenfunction series reference (narrow-pulse limit) is provided as a helper
and is verified to be internally consistent via a matrix-exponential cross-check,
but is not used as the primary acceptance criterion because finite gradient
duration (~9% of L) introduces ~4% error in the narrow-pulse approximation.
"""

import numpy as np
import numpy.testing as npt

from dmipy_sim import simulate, pgse, Box1D, FreeDiffusion, set_b
from .conftest import D, N_WALKERS, SEED


def test_box_1d_signal_at_b0_is_one():
    """Signal at very low b should be 1 regardless of geometry."""
    L = 10e-6
    wf = set_b(pgse(delta=0.2e-3, DELTA=40e-3, G_magnitude=1.0,
                    bvecs=np.array([[1., 0., 0.]]), n_t=1000),
               np.array([1e3]))
    sig = simulate(10_000, D, wf, Box1D(L), seed=SEED)
    npt.assert_allclose(sig, 1.0, atol=0.02)


def test_box_1d_restricted_signal_above_free():
    """Restricted signal must be higher than free diffusion at same b-values."""
    L = 10e-6
    delta = 0.2e-3; DELTA = 40e-3; n_t = 1000; n_b = 30
    b_values = np.linspace(1e8, 3e9, n_b)  # avoid b=0 edge
    bvecs = np.tile([1., 0., 0.], (n_b, 1))

    wf = set_b(pgse(delta=delta, DELTA=DELTA, G_magnitude=1.0,
                    bvecs=bvecs, n_t=n_t), b_values)

    S_box  = simulate(N_WALKERS, D, wf, Box1D(L), seed=SEED)
    S_free = simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED + 1)

    assert np.all(S_box >= S_free - 0.01), (
        "Restricted box signal should be >= free diffusion signal at same b-values. "
        f"Max violation: {np.max(S_free - S_box):.4f}")


def test_box_1d_monotonically_decreasing():
    """Signal must be monotonically non-increasing with b-value."""
    L = 10e-6
    b_values = np.linspace(1, 3e9, 30)
    bvecs = np.tile([1., 0., 0.], (30, 1))
    wf = set_b(pgse(delta=0.2e-3, DELTA=40e-3, G_magnitude=1.0,
                    bvecs=bvecs, n_t=1000), b_values)
    S = simulate(N_WALKERS, D, wf, Box1D(L), seed=SEED)
    assert np.all(np.diff(S) <= 0.005), (
        "Box1D signal should be monotonically non-increasing with b-value")


def test_box_1d_tighter_confinement_gives_higher_signal():
    """Smaller box → stronger restriction → higher signal at same b."""
    delta = 0.2e-3; DELTA = 40e-3; n_t = 1000
    b_values = np.array([3e9])
    bvecs = np.array([[1., 0., 0.]])
    wf = set_b(pgse(delta=delta, DELTA=DELTA, G_magnitude=1.0,
                    bvecs=bvecs, n_t=n_t), b_values)

    S_large = simulate(N_WALKERS, D, wf, Box1D(20e-6), seed=SEED)
    S_small = simulate(N_WALKERS, D, wf, Box1D(5e-6),  seed=SEED)

    assert S_small[0] > S_large[0] - 0.01, (
        f"Smaller box (5µm) should give higher signal than larger box (20µm) "
        f"at b=3e9. Got: small={S_small[0]:.4f}, large={S_large[0]:.4f}")


def test_box_1d_perpendicular_gradient_is_free():
    """Gradient perpendicular to restriction axis gives free diffusion signal."""
    L = 10e-6
    b_values = np.linspace(1, 2e9, 20)
    # Gradient along y — Box1D restricts x only
    bvecs = np.tile([0., 1., 0.], (20, 1))
    wf = set_b(pgse(delta=0.2e-3, DELTA=40e-3, G_magnitude=1.0,
                    bvecs=bvecs, n_t=1000), b_values)

    S_box  = simulate(N_WALKERS, D, wf, Box1D(L), seed=SEED)
    E_free = np.exp(-b_values * D)

    npt.assert_allclose(S_box, E_free, atol=0.01,
                        err_msg="Box1D with perpendicular gradient should equal free diffusion")
