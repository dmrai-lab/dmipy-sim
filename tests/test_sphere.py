"""Restricted diffusion, sphere, MISST validation.

Validates the Sphere geometry against MISST reference signals from Kerkelae's
disimpy, which itself is validated against the MISST matrix-operator toolbox.

Two gradient configurations match disimpy's test_sphere_diffusion exactly:
  Config 1: delta ≈ 30 ms  (broad pulse)
  Config 2: delta ≈  1 ms  (narrow pulse)

Gradient is built in disimpy's raw convention (indices, not time masks) then
linearly interpolated to n_t=1000 time points — matching disimpy bit-for-bit.
"""

import numpy as np
import numpy.testing as npt
import jax.numpy as jnp

from dmipy_sim import simulate, Sphere, set_b
from dmipy_sim.waveforms import Waveform, calc_b
from .conftest import D, N_WALKERS, SEED, load_fixture


def _build_disimpy_waveform(n_t_raw, pulse_slice_start, pulse_slice_end, n_t=1000):
    """Replicate disimpy's raw-then-interpolate gradient construction.

    Parameters
    ----------
    n_t_raw : int
        Number of raw time points (e.g. 700).
    pulse_slice_start : int
        Start index of first pulse (disimpy uses 1).
    pulse_slice_end : int
        End index (exclusive) of first pulse; second pulse mirrors at -end.
    n_t : int
        Number of time points after interpolation.

    Returns
    -------
    Waveform with G-amplitude 1.0 along x, shape (1, n_t, 3).
    """
    T = 70e-3  # same for both configs
    dt_raw = T / (n_t_raw - 1)

    grad_raw = np.zeros((1, n_t_raw, 3), dtype=np.float64)
    # Pulse 1
    grad_raw[0, pulse_slice_start:pulse_slice_end, 0] = 1.0
    # Pulse 2 (mirrored: -300:-1 means from index n_t_raw-300 to n_t_raw-2)
    n_pulse = pulse_slice_end - pulse_slice_start
    grad_raw[0, -(n_pulse + 1):-1, 0] = -1.0

    # Interpolate to n_t points (same as disimpy's gradients.interpolate_gradient)
    T_total = dt_raw * (n_t_raw - 1)
    dt = T_total / (n_t - 1)
    t_old = np.linspace(0, T_total, n_t_raw)
    t_new = np.linspace(0, T_total, n_t)

    G_interp = np.zeros((1, n_t, 3), dtype=np.float32)
    for j in range(3):
        G_interp[0, :, j] = np.interp(t_new, t_old, grad_raw[0, :, j])

    return Waveform(G=jnp.array(G_interp), dt=float(dt), echo_idx=n_t - 1)


def _tile_and_set_b(wf_single, b_values):
    """Tile a single-measurement waveform to n_b measurements and set b-values."""
    n_b = len(b_values)
    G_tiled = jnp.tile(wf_single.G, (n_b, 1, 1))  # (n_b, n_t, 3)
    wf = Waveform(G=G_tiled, dt=wf_single.dt, echo_idx=wf_single.echo_idx)
    return set_b(wf, b_values)


def test_sphere_misst_config1():
    """Config 1: delta ≈ 30 ms, Delta ≈ 40 ms, r=5 µm vs MISST fixture."""
    misst = load_fixture("misst_sphere_delta30ms_Delta40ms_r5um.npy")
    b_values = np.linspace(1, 3e9, 100)

    # disimpy: gradient[0, 1:300, 0]=1  (299 steps = pulse1)
    #          gradient[0, -300:-1, 0]=-1 (pulse2 mirrors at same width)
    wf_single = _build_disimpy_waveform(n_t_raw=700,
                                         pulse_slice_start=1,
                                         pulse_slice_end=300)
    wf = _tile_and_set_b(wf_single, b_values)

    signals = simulate(N_WALKERS, D, wf, Sphere(5e-6), seed=SEED)

    npt.assert_allclose(signals, misst, atol=0.02,
                        err_msg="Sphere Config1 (delta≈30ms) vs MISST")


def test_sphere_misst_config2():
    """Config 2: delta ≈ 1 ms, Delta ≈ 40 ms, r=5 µm vs MISST fixture."""
    misst = load_fixture("misst_sphere_delta1ms_Delta40ms_r5um.npy")
    b_values = np.linspace(1, 3e9, 100)

    # disimpy: T=41ms, n_t_raw=410
    #          gradient[0, 1:10, 0]=1  (9 steps = pulse1)
    #          gradient[0, -10:-1, 0]=-1
    T_config2 = 41e-3
    n_t_raw = 410
    dt_raw = T_config2 / (n_t_raw - 1)
    n_t = 1000

    grad_raw = np.zeros((1, n_t_raw, 3), dtype=np.float64)
    grad_raw[0, 1:10, 0] = 1.0
    grad_raw[0, -10:-1, 0] = -1.0

    T_total = dt_raw * (n_t_raw - 1)
    dt = T_total / (n_t - 1)
    t_old = np.linspace(0, T_total, n_t_raw)
    t_new = np.linspace(0, T_total, n_t)

    G_interp = np.zeros((1, n_t, 3), dtype=np.float32)
    for j in range(3):
        G_interp[0, :, j] = np.interp(t_new, t_old, grad_raw[0, :, j])

    wf_single = Waveform(G=jnp.array(G_interp), dt=float(dt), echo_idx=n_t - 1)
    wf = _tile_and_set_b(wf_single, b_values)

    signals = simulate(N_WALKERS, D, wf, Sphere(5e-6), seed=SEED)

    npt.assert_allclose(signals, misst, atol=0.02,
                        err_msg="Sphere Config2 (delta≈1ms) vs MISST")


def test_sphere_walkers_contained():
    """Final walker positions must be strictly inside the sphere, and at least
    one walker must have reached the boundary (verifying reflection fires).

    Matches disimpy's test_sphere_diffusion containment check:
      max_pos = np.max(np.linalg.norm(trajectories, axis=2))
      assert max_pos < radius
      assert_almost_equal(max_pos, radius)
    We check final positions only (no trajectory output), so use a generous
    n_walkers (10 000) to ensure boundary proximity is observed.
    """
    from dmipy_sim.waveforms import pgse
    radius = 5e-6
    wf = set_b(pgse(delta=8e-3, DELTA=50e-3, G_magnitude=1.0,
                    bvecs=np.array([[1., 0., 0.]]), n_t=1000),
               np.array([1e9]))
    _, pos = simulate(10_000, D, wf, Sphere(radius), seed=SEED,
                      return_positions=True)
    norms = np.linalg.norm(pos, axis=1)
    max_pos = float(np.max(norms))
    assert max_pos < radius, f"Walker escaped sphere: max |r|={max_pos:.3e} > radius={radius:.3e}"
    # At least one walker should have come close to the boundary (within 1%)
    assert max_pos > 0.99 * radius, (
        f"No walker reached boundary (max |r|={max_pos:.3e}); reflection may not be working"
    )


def test_sphere_signal_above_free():
    """Restricted sphere signal must be >= free diffusion at same b-values."""
    from dmipy_sim import FreeDiffusion
    from dmipy_sim.waveforms import pgse
    b_values = np.linspace(1e8, 3e9, 20)
    bvecs = np.tile([1., 0., 0.], (20, 1))
    wf = set_b(pgse(delta=0.2e-3, DELTA=40e-3, G_magnitude=1.0,
                    bvecs=bvecs, n_t=1000), b_values)

    S_sphere = simulate(N_WALKERS, D, wf, Sphere(5e-6), seed=SEED)
    S_free   = simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED + 1)

    assert np.all(S_sphere >= S_free - 0.01), (
        f"Sphere signal should be >= free diffusion. "
        f"Max violation: {np.max(S_free - S_sphere):.4f}")
