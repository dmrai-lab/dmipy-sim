"""Restricted diffusion, ellipsoid geometry.

Matches disimpy's test_ellipsoid_diffusion:
  - Walker containment: max |r| < radius (strict) and ≈ radius (reflection fires)
  - Isotropic ellipsoid (semiaxes = [r, r, r]) must match Sphere(r) signal

Both tests use the same PGSE gradient setup as disimpy's example_gradient():
  raw gradient along x, delta≈8ms, DELTA≈50ms, then linearly interpolated.
"""

import numpy as np
import numpy.testing as npt
import jax.numpy as jnp

from dmipy_sim import simulate, Ellipsoid, Sphere, set_b
from dmipy_sim.waveforms import Waveform, pgse
from .conftest import D, N_WALKERS, SEED


def _disimpy_example_waveform(n_t=1000):
    """Replicate disimpy's example_gradient() used in test_ellipsoid_diffusion.

    disimpy builds:
      gradient = np.zeros((1, int(1e3), 3))
      dt = 1e-4  (so T = 0.1 s)
      gradient[0, 1:80, 0] = 1    # delta ≈ 79 * 1e-4 = 7.9 ms
      gradient[0, -80:-1, 0] = -1 # DELTA ≈ T - delta ≈ 92.1 ms
    then interpolates to n_t=1000.

    Since the raw gradient already has n_t_raw=1000, interpolation is identity.
    """
    n_t_raw = 1000
    dt_raw  = 1e-4   # 0.1 ms
    T_total = dt_raw * (n_t_raw - 1)

    grad_raw = np.zeros((1, n_t_raw, 3), dtype=np.float64)
    grad_raw[0, 1:80, 0]   = 1.0
    grad_raw[0, -80:-1, 0] = -1.0

    # Interpolate to n_t (identity here since n_t_raw == n_t == 1000)
    dt_new = T_total / (n_t - 1)
    t_old  = np.linspace(0, T_total, n_t_raw)
    t_new  = np.linspace(0, T_total, n_t)
    G_interp = np.zeros((1, n_t, 3), dtype=np.float32)
    for j in range(3):
        G_interp[0, :, j] = np.interp(t_new, t_old, grad_raw[0, :, j])

    return Waveform(G=jnp.array(G_interp), dt=float(dt_new), echo_idx=n_t - 1)


def test_ellipsoid_walkers_contained():
    """Final walker positions must be strictly inside the ellipsoid and at
    least one walker must have reached the boundary (verifying reflection).

    Matches disimpy's test_ellipsoid_diffusion containment check:
      max_pos = max(||trajectories||)
      assert max_pos < radius
      assert_almost_equal(max_pos, radius)

    Uses an isotropic ellipsoid (semiaxes=[r,r,r]) so containment reduces
    to ||r|| < radius. Checked on final positions only.
    """
    radius = 5e-6
    semiaxes = np.ones(3) * radius
    wf = set_b(_disimpy_example_waveform(), np.array([1e9]))
    _, pos = simulate(10_000, D, wf, Ellipsoid(semiaxes), seed=SEED,
                      return_positions=True)
    norms = np.linalg.norm(pos, axis=1)
    max_pos = float(np.max(norms))
    assert max_pos < radius, (
        f"Walker escaped ellipsoid: max |r|={max_pos:.3e} > radius={radius:.3e}")
    assert max_pos > 0.99 * radius, (
        f"No walker reached boundary (max |r|={max_pos:.3e}); reflection may not be working")


def test_ellipsoid_isotropic_matches_sphere():
    """Isotropic ellipsoid (semiaxes = [r,r,r]) must give the same signal as Sphere(r).

    Matches disimpy's test_ellipsoid_diffusion signal comparison:
      substrate = substrates.ellipsoid(np.ones(3) * radius)
      signals_sphere = simulation(..., substrates.sphere(radius))
      assert_almost_equal(signals, signals_sphere)

    Uses identical seeds so the only difference is the geometry class.
    Tolerance atol=0.02 accounts for MC noise at N=100 000 walkers.
    """
    radius = 5e-6
    semiaxes = np.ones(3) * radius
    wf = set_b(_disimpy_example_waveform(), np.array([1e9]))

    S_ellipsoid = simulate(N_WALKERS, D, wf, Ellipsoid(semiaxes), seed=SEED)
    S_sphere    = simulate(N_WALKERS, D, wf, Sphere(radius),       seed=SEED)

    npt.assert_allclose(S_ellipsoid, S_sphere, atol=0.02,
                        err_msg="Isotropic ellipsoid should match sphere signal")


def test_ellipsoid_signal_above_free():
    """Prolate ellipsoid with gradient along short axis must exceed free diffusion."""
    from dmipy_sim import FreeDiffusion
    b_values = np.linspace(1e8, 3e9, 20)
    bvecs = np.tile([1., 0., 0.], (20, 1))   # gradient along x (short semi-axis)
    # Prolate ellipsoid: short axes a=b=2µm, long axis c=10µm
    semiaxes = np.array([2e-6, 2e-6, 10e-6])
    wf = set_b(pgse(delta=0.2e-3, DELTA=40e-3, G_magnitude=1.0,
                    bvecs=bvecs, n_t=1000), b_values)

    S_ellipsoid = simulate(N_WALKERS, D, wf, Ellipsoid(semiaxes), seed=SEED)
    S_free      = simulate(N_WALKERS, D, wf, FreeDiffusion(),     seed=SEED + 1)

    assert np.all(S_ellipsoid >= S_free - 0.01), (
        f"Ellipsoid signal should exceed free diffusion along restricted axis. "
        f"Max violation: {np.max(S_free - S_ellipsoid):.4f}")


def test_ellipsoid_anisotropic_axis_dependence():
    """Prolate ellipsoid: gradient along short axis must show more restriction
    than gradient along long axis.

    Physical expectation: displacement along the short axis (a=2µm) is more
    restricted than along the long axis (c=10µm) → shorter-axis signal > longer-axis signal.
    """
    semiaxes = np.array([2e-6, 2e-6, 10e-6])   # short x,y; long z
    b = 2e9  # single high b-value to maximise contrast

    wf_x = set_b(pgse(delta=0.2e-3, DELTA=40e-3, G_magnitude=1.0,
                      bvecs=np.array([[1., 0., 0.]]), n_t=1000),
                 np.array([b]))
    wf_z = set_b(pgse(delta=0.2e-3, DELTA=40e-3, G_magnitude=1.0,
                      bvecs=np.array([[0., 0., 1.]]), n_t=1000),
                 np.array([b]))

    S_x = simulate(N_WALKERS, D, wf_x, Ellipsoid(semiaxes), seed=SEED)
    S_z = simulate(N_WALKERS, D, wf_z, Ellipsoid(semiaxes), seed=SEED + 1)

    # Short-axis gradient (x, 2µm) → more restriction → higher signal
    assert float(S_x[0]) > float(S_z[0]) - 0.01, (
        f"Short-axis signal ({S_x[0]:.4f}) should exceed "
        f"long-axis signal ({S_z[0]:.4f}) for prolate ellipsoid")
