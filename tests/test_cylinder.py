"""Restricted diffusion, cylinder, MISST validation.

Matches disimpy's test_cylinder_diffusion exactly:
  - Gradient along x, cylinder axis along z (gradient ⊥ axis → maximum restriction)
  - Config 1: delta ≈ 30 ms, Delta ≈ 40 ms, r = 5 µm
  - Config 2: delta ≈  1 ms, Delta ≈ 40 ms, r = 5 µm
"""

import numpy as np
import numpy.testing as npt
import jax.numpy as jnp

from dmipy_sim import simulate, Cylinder, FreeDiffusion, set_b
from dmipy_sim.waveforms import Waveform
from .conftest import D, N_WALKERS, SEED, load_fixture


def _build_disimpy_waveform(T, n_t_raw, pulse_start, pulse_end, n_t=1000):
    """Replicate disimpy's raw-then-interpolate gradient construction."""
    dt_raw = T / (n_t_raw - 1)

    grad_raw = np.zeros((1, n_t_raw, 3), dtype=np.float64)
    n_pulse = pulse_end - pulse_start
    grad_raw[0, pulse_start:pulse_end, 0] = 1.0
    grad_raw[0, -(n_pulse + 1):-1, 0] = -1.0

    T_total = dt_raw * (n_t_raw - 1)
    dt = T_total / (n_t - 1)
    t_old = np.linspace(0, T_total, n_t_raw)
    t_new = np.linspace(0, T_total, n_t)

    G_interp = np.zeros((1, n_t, 3), dtype=np.float32)
    for j in range(3):
        G_interp[0, :, j] = np.interp(t_new, t_old, grad_raw[0, :, j])

    return Waveform(G=jnp.array(G_interp), dt=float(dt), echo_idx=n_t - 1)


def _tile_and_set_b(wf_single, b_values):
    n_b = len(b_values)
    G_tiled = jnp.tile(wf_single.G, (n_b, 1, 1))
    wf = Waveform(G=G_tiled, dt=wf_single.dt, echo_idx=wf_single.echo_idx)
    return set_b(wf, b_values)


def test_cylinder_misst_config1():
    """Config 1: delta ≈ 30 ms, Delta ≈ 40 ms, r=5 µm vs MISST fixture."""
    misst = load_fixture("misst_cylinder_delta30ms_Delta40ms_r5um.npy")
    b_values = np.linspace(1, 3e9, 100)

    wf_single = _build_disimpy_waveform(T=70e-3, n_t_raw=700,
                                         pulse_start=1, pulse_end=300)
    wf = _tile_and_set_b(wf_single, b_values)

    signals = simulate(N_WALKERS, D, wf,
                       Cylinder(radius=5e-6, orientation=[0, 0, 1.0]),
                       seed=SEED)

    npt.assert_allclose(signals, misst, atol=0.02,
                        err_msg="Cylinder Config1 (delta≈30ms) vs MISST")


def test_cylinder_misst_config2():
    """Config 2: delta ≈ 1 ms, Delta ≈ 40 ms, r=5 µm vs MISST fixture."""
    misst = load_fixture("misst_cylinder_delta1ms_Delta40ms_r5um.npy")
    b_values = np.linspace(1, 3e9, 100)

    wf_single = _build_disimpy_waveform(T=41e-3, n_t_raw=410,
                                         pulse_start=1, pulse_end=10)
    wf = _tile_and_set_b(wf_single, b_values)

    signals = simulate(N_WALKERS, D, wf,
                       Cylinder(radius=5e-6, orientation=[0, 0, 1.0]),
                       seed=SEED)

    npt.assert_allclose(signals, misst, atol=0.02,
                        err_msg="Cylinder Config2 (delta≈1ms) vs MISST")


def test_cylinder_parallel_gradient_is_free():
    """Gradient along cylinder axis → walkers diffuse freely → exp(-bD)."""
    from dmipy_sim.waveforms import pgse
    b_values = np.linspace(1e8, 3e9, 20)
    bvecs = np.tile([0., 0., 1.], (20, 1))  # gradient along z = cylinder axis
    wf = set_b(pgse(delta=0.2e-3, DELTA=40e-3, G_magnitude=1.0,
                    bvecs=bvecs, n_t=1000), b_values)

    signals = simulate(N_WALKERS, D, wf,
                       Cylinder(radius=5e-6, orientation=[0, 0, 1.0]),
                       seed=SEED)
    expected = np.exp(-b_values * D)

    npt.assert_allclose(signals, expected, atol=0.02,
                        err_msg="Parallel gradient should give free diffusion")


def test_cylinder_signal_above_free_perp():
    """Perpendicular gradient → restricted; signal must exceed free diffusion."""
    from dmipy_sim.waveforms import pgse
    b_values = np.linspace(1e8, 3e9, 20)
    bvecs = np.tile([1., 0., 0.], (20, 1))  # gradient ⊥ cylinder axis
    wf = set_b(pgse(delta=0.2e-3, DELTA=40e-3, G_magnitude=1.0,
                    bvecs=bvecs, n_t=1000), b_values)

    S_cyl  = simulate(N_WALKERS, D, wf,
                      Cylinder(radius=5e-6, orientation=[0, 0, 1.0]), seed=SEED)
    S_free = simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED + 1)

    assert np.all(S_cyl >= S_free - 0.01), (
        f"Cylinder signal should be >= free. Max violation: {np.max(S_free - S_cyl):.4f}")


# NOTE: `test_cylinder_walkers_contained*` was removed (#88). It asserted two things:
# containment (max |r| < R) and that walkers reach the wall. Containment is now covered
# deterministically and exhaustively by `tests/test_wall_impacts.py`, which sweeps offset x
# direction x step length -- including the step-spans-the-object case this walk could not
# reach -- in milliseconds rather than minutes. "Walkers reach the wall" is a far weaker
# statement than the MISST signal comparisons kept in this file: a restricted signal is
# itself proof that walkers are hitting walls.


def test_cylinder_orientation_sign_invariance():
    """Flipping the orientation sign must give identical signals.

    Matches disimpy's rotation sub-test:
      signals_1 = simulate(..., orientation=[1,0,1])
      signals_2 = simulate(..., orientation=-[1,0,1])
      assert_almost_equal(signals_1 / n_s, signals_2 / n_s)

    Uses the same gradient setup as disimpy (x-direction, 100 b-values,
    PGSE Config 1 raw-then-interpolate construction).
    """
    b_values = np.linspace(1, 3e9, 100)
    wf_single = _build_disimpy_waveform(T=70e-3, n_t_raw=700,
                                         pulse_start=1, pulse_end=300)
    wf = _tile_and_set_b(wf_single, b_values)

    S1 = simulate(N_WALKERS, D, wf,
                  Cylinder(radius=5e-6, orientation=[1.0, 0.0, 1.0]),  seed=SEED)
    S2 = simulate(N_WALKERS, D, wf,
                  Cylinder(radius=5e-6, orientation=[-1.0, 0.0, -1.0]), seed=SEED)

    npt.assert_allclose(S1, S2, atol=0.01,
                        err_msg="Orientation sign flip must give identical signals")


def test_cylinder_general_orientation_parallel_gradient_is_free():
    """Gradient parallel to a non-z cylinder axis → free diffusion exp(-bD).

    Matches disimpy's rotation sub-test:
      signals_3 = simulate(..., orientation=-[1,0,0])  # axis along x
      assert_almost_equal(signals_3 / n_s, exp(-bs*D), 2)

    With orientation=[1,0,0] (or -[1,0,0]) the cylinder axis is x.
    A gradient along x is parallel to the axis → no restriction → exp(-bD).
    Uses the same raw-then-interpolate gradient as disimpy (x-direction).
    """
    b_values = np.linspace(1, 3e9, 100)
    wf_single = _build_disimpy_waveform(T=70e-3, n_t_raw=700,
                                         pulse_start=1, pulse_end=300)
    wf = _tile_and_set_b(wf_single, b_values)

    # orientation=-[1,0,0]: axis along x, gradient along x → parallel
    S = simulate(N_WALKERS, D, wf,
                 Cylinder(radius=5e-6, orientation=[-1.0, 0.0, 0.0]),
                 seed=SEED)
    expected = np.exp(-b_values * D)

    npt.assert_allclose(S, expected, atol=0.02,
                        err_msg="Gradient ∥ cylinder axis (x) should give free diffusion")
