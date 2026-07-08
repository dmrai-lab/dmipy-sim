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


def test_cylinder_walkers_contained_multiple_radii():
    """Final walker cross-section norms must lie inside the cylinder for
    radii 1 µm, 5 µm, 1 mm.

    Matches disimpy's test_cylinder_diffusion containment loop:
      for radius in [1e-6, 5e-6, 1e-3]:
          substrate = substrates.cylinder(radius=radius, orientation=[1,0,0])
          ...
          max_pos = max(||trajectories[..., 1:]||)  # y,z components only
          assert max_pos < radius
          assert_almost_equal(max_pos, radius)

    With orientation=[1,0,0] (cylinder axis along x), restriction acts in
    the y-z plane; we check max(sqrt(y²+z²)) < radius and > 0.99*radius.
    """
    from dmipy_sim.waveforms import pgse
    bvecs = np.array([[1., 0., 0.]])   # gradient along x (⊥ to axis=[1,0,0])
    for radius in [1e-6, 5e-6, 1e-3]:
        n_t = 1000
        # Use a moderate diffusion time so walkers explore the geometry
        wf = set_b(pgse(delta=8e-3, DELTA=50e-3, G_magnitude=1.0,
                        bvecs=bvecs, n_t=n_t), np.array([1e9]))
        _, pos = simulate(5_000, D, wf,
                          Cylinder(radius=radius, orientation=[1.0, 0.0, 0.0]),
                          seed=SEED, return_positions=True)
        # Cross-section norm: y-z plane (axis is x)
        cross_norms = np.linalg.norm(pos[:, 1:], axis=1)
        max_cross = float(np.max(cross_norms))
        # Allow 3×NUDGE (= 3e-4 × radius) above the boundary: the safety clamp
        # projects to R−NUDGE in the 2-D cylinder frame, but the R_inv matrix
        # multiply back to lab frame introduces float32 rounding that can add up
        # to ~2×NUDGE at small radii (< 5 µm) on GPU.
        assert max_cross < radius * (1 + 3e-4), (
            f"r={radius:.0e}: walker escaped cylinder: "
            f"max cross-section norm={max_cross:.3e}")
        assert max_cross > 0.99 * radius, (
            f"r={radius:.0e}: no walker reached boundary "
            f"(max={max_cross:.3e}); reflection may not be working")


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
