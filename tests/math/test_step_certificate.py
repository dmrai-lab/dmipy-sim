"""The step certificate (dmrai-lab/dmipy-sim#715 part 2b): a walk's diffusion spectrum read off its saved increments,
and two pilot walks of one substrate at two steps measuring the first-order error of the fine step over the band."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine.physics import tolerance_sub_steps, STEP_TOLERANCE
from dmipy_sim.math.step_law import diffusion_spectrum, step_certificate, PILOT_STEP_RATIO

D0 = 2e-9


def test_a_free_walk_s_spectrum_is_flat_and_a_pore_s_rises_to_the_free_value():
    """Free increments are white on any save grid, so the estimate is ``D`` at every frequency within its standard
    error; a 5 um sphere's spectrum is under a fifth of ``D`` at the lowest bin and at ``D`` above its pore rate."""
    w = d.simulate_trajectories(3000, D0, d.FreeDiffusion(), 0.02, 1e-4, seed=0, require_gpu=False)
    f, D, se = diffusion_spectrum(w.positions, w.dt)
    assert f[0] == 0.0 and f[-1] == pytest.approx(0.5 / w.dt)
    z = (D[1:] - D0) / se[1:]
    assert np.abs(z).max() < 4.5 and np.abs(z.mean()) < 0.3                       # flat, and unbiased over the band
    w = d.simulate_trajectories(3000, D0, d.Sphere(5e-6), 0.1, 1e-4, seed=0, require_gpu=False)
    f, D, se = diffusion_spectrum(w.positions, w.dt)
    f1 = D0 * (2.0816 / 5e-6) ** 2 / (2 * np.pi)                                  # the pore rate, 55 Hz
    assert D[1] < 0.3 * D0 and f[1] < f1 / 4
    high = (f > 1500.0) & (f < 2500.0)                                             # thirty pore rates up: near D
    assert 0.85 < D[high].mean() / D0 < 1.05


def test_two_pilot_walks_measure_the_fine_step_s_error():
    """On a 2 um sphere read to 1 kHz, a walk sized for ``eps = 4e-3`` (a quarter of the rule's count at 1e-3)
    against a walk stepping four times longer, both in the first-order regime: the band's rms error is under the
    tolerance the rule sized the step for and carries most of it, the pilot resolves it, and the per-bin errors
    change sign across the pore's rate (345 Hz)."""
    from dmipy_sim.engine.physics import regime_sub_steps
    g = d.Sphere(2e-6)
    dt, T, f_band = 1e-4, 0.04, 1000.0
    n_rule = tolerance_sub_steps(g, D0, dt, f_band=f_band)
    n_fine = max(2, n_rule // 4)
    assert n_fine // PILOT_STEP_RATIO >= regime_sub_steps(g, D0, dt)
    eps = STEP_TOLERANCE * n_rule / n_fine                                         # first order: eps scales as the step
    n_coarse = max(1, n_fine // PILOT_STEP_RATIO)
    fine = d.simulate_trajectories(16000, D0, g, T, dt, seed=1, require_gpu=False, sub_steps=n_fine)
    coarse = d.simulate_trajectories(16000, D0, g, T, dt, seed=2, require_gpu=False, sub_steps=n_coarse)
    cert = step_certificate(fine.positions, coarse.positions, dt, n_fine / n_coarse, f_band=f_band, eps_declared=eps)
    assert cert["n_bins"] == 40 and cert["ratio"] == pytest.approx(n_fine / n_coarse) and cert["resolves"]
    assert cert["floor"] < 0.5 * eps
    assert 0.4 * eps < cert["eps_measured"] < 1.1 * eps + 3 * cert["eps_se"]
    with pytest.raises(ValueError, match="no frequency bin"):
        step_certificate(fine.positions, coarse.positions, dt, 4.0, f_band=1.0, eps_declared=eps)
