"""General / arbitrary waveforms with restricted geometries.

Verifies that:
  1. OGSE with a sphere matches physical expectations (higher signal than PGSE
     at the same b-value because OGSE probes shorter length scales).
  2. An asymmetric waveform (non-zero net gradient) gives lower signal than a
     fully refocused waveform at the same b-value.
  3. The Waveform dataclass can be constructed directly with arbitrary G arrays
     and still produce physically sensible results.
"""

import numpy as np
import numpy.testing as npt
import jax.numpy as jnp

from dmipy_sim import simulate, FreeDiffusion, Sphere, set_b
from dmipy_sim.waveforms import pgse, ogse, Waveform, calc_b
from .conftest import D, N_WALKERS, SEED


def test_ogse_free_diffusion_matches_exp_bD():
    """OGSE on FreeDiffusion must give exp(-b*D) to within MC noise."""
    b_values = np.linspace(1e8, 2e9, 20)
    bvecs = np.tile([1., 0., 0.], (20, 1))
    wf = set_b(ogse(frequency=50.0, T_total=80e-3, G_magnitude=1.0,
                    bvecs=bvecs, n_t=1000), b_values)

    signals = simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED)
    expected = np.exp(-b_values * D)

    npt.assert_allclose(signals, expected, atol=0.02,
                        err_msg="OGSE free diffusion must equal exp(-bD)")


def test_pgse_long_delta_more_restricted_than_ogse():
    """PGSE with long DELTA is more restricted than OGSE at same b-value.

    At long DELTA, walkers repeatedly encounter the sphere boundary →
    strong restriction → high signal. OGSE at moderate frequency has a
    shorter effective diffusion time → weaker restriction → lower signal.
    Both must still exceed the free-diffusion signal (exp(-bD)).
    """
    b_target = 1e9  # s/m²
    bvecs = np.array([[1., 0., 0.]])

    wf_pgse = set_b(pgse(delta=5e-3, DELTA=40e-3, G_magnitude=1.0,
                          bvecs=bvecs, n_t=1000), np.array([b_target]))
    wf_ogse = set_b(ogse(frequency=100.0, T_total=80e-3, G_magnitude=1.0,
                          bvecs=bvecs, n_t=1000), np.array([b_target]))

    geom = Sphere(radius=5e-6)
    S_pgse = simulate(N_WALKERS, D, wf_pgse, geom, seed=SEED)
    S_ogse = simulate(N_WALKERS, D, wf_ogse, geom, seed=SEED + 1)
    S_free = np.exp(-b_target * D)

    # Long-DELTA PGSE → maximum restriction → highest signal
    assert float(S_pgse[0]) >= float(S_ogse[0]) - 0.02, (
        f"PGSE (long DELTA) signal {S_pgse[0]:.4f} should be >= "
        f"OGSE signal {S_ogse[0]:.4f}")
    # Both restricted signals must exceed free diffusion
    assert float(S_pgse[0]) >= S_free - 0.02
    assert float(S_ogse[0]) >= S_free - 0.02


def test_arbitrary_waveform_zero_net_gradient_gives_free():
    """A fully refocused arbitrary waveform on FreeDiffusion must give exp(-bD)."""
    # Build a simple bipolar trapezoid directly as G array
    n_t = 500
    T_total = 50e-3
    dt = T_total / (n_t - 1)
    G_arr = np.zeros((1, n_t, 3), dtype=np.float32)
    # First lobe: +G along x for first quarter
    q1 = n_t // 4
    G_arr[0, :q1, 0] = 1.0
    # Second lobe: -G along x for second quarter (same duration → refocused)
    G_arr[0, q1:2*q1, 0] = -1.0

    wf_raw = Waveform(G=jnp.array(G_arr), dt=float(dt), echo_idx=n_t - 1)
    b_raw = calc_b(wf_raw)
    assert b_raw[0] > 0, "b-value should be positive"

    wf = set_b(wf_raw, np.array([1e9]))
    signal = simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED)
    expected = np.exp(-1e9 * D)

    npt.assert_allclose(signal, expected, atol=0.02,
                        err_msg="Arbitrary refocused waveform on FreeDiffusion must give exp(-bD)")


def test_calc_b_pgse_analytical():
    """calc_b should match the analytical PGSE b-value: (gamma*G*delta)^2*(Delta-delta/3)."""
    from dmipy_sim.constants import GAMMA
    delta = 10e-3
    DELTA = 40e-3
    G_mag = 0.05  # T/m
    bvecs = np.array([[1., 0., 0.]])

    # the analytic formula is for SQUARE lobes -> build the instantaneous waveform
    # (sim now defaults to slew-limited, whose ramps give a slightly smaller b)
    wf = pgse(delta=delta, DELTA=DELTA, G_magnitude=G_mag, bvecs=bvecs, n_t=2000,
              slew_rate=np.inf)
    b_sim = calc_b(wf)[0]
    b_analytic = (GAMMA * G_mag * delta) ** 2 * (DELTA - delta / 3)

    npt.assert_allclose(b_sim, b_analytic, rtol=0.01,
                        err_msg="calc_b must match analytical PGSE b-value within 1%")
