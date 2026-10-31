"""The step certificate (dmrai-lab/dmipy-sim#715 part 2b, on the stored bands since #740): a walk's diffusion spectrum
read off its bands, and two pilot walks of one substrate at two steps measuring the first-order error of the fine
step over the band."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.engine.accumulate import block_length
from dmipy_sim.engine.physics import tolerance_sub_steps, STEP_TOLERANCE
from dmipy_sim.math.step_law import diffusion_spectrum, step_certificate, PILOT_STEP_RATIO

D0 = 2e-9


def test_a_free_walk_s_spectrum_is_flat_and_a_pore_s_rises_to_the_free_value():
    """A free walk's bands are its Karhunen-Loeve coefficients, so the estimate is ``D`` at every band within its
    standard error; a 5 um sphere's spectrum is under a third of ``D`` at the lowest band and near ``D`` thirty pore
    rates up. The bands sit at ``k / (2 T)`` and there is no zero-frequency one."""
    T, K = 0.02, 80
    w = _walk_geometry(3000, D0, d.FreeDiffusion(), T, K=K, seed=0, require_gpu=False, window_T=0.1, tiers=())
    f, D, se = diffusion_spectrum(w.position_coeffs, w.T_window)
    assert f.size == K and f[0] == pytest.approx(0.5 / T) and f[-1] == pytest.approx(K / (2 * T))
    z = (D - D0) / se
    assert np.abs(z).max() < 4.5 and abs(z.mean()) < 0.3                         # flat, and unbiased over the band
    T, K = 0.1, 400
    w = _walk_geometry(3000, D0, d.Sphere(5e-6), T, K=K, seed=0, require_gpu=False, window_T=0.1, tiers=())
    f, D, se = diffusion_spectrum(w.position_coeffs, T)
    f1 = D0 * (2.0816 / 5e-6) ** 2 / (2 * np.pi)                                  # the pore rate, 55 Hz
    assert D[0] < 0.3 * D0 and f[0] < f1 / 4
    high = (f > 1500.0) & (f < 2000.0)                                             # thirty pore rates up: near D
    assert 0.85 < D[high].mean() / D0 < 1.05


def test_two_pilot_walks_measure_the_fine_step_s_error():
    """On a 2 um sphere read to 1 kHz over 40 ms (K = 80, the band's own block), a walk sized for ``eps`` four times
    the rule's against a walk stepping four times longer, both in the first-order regime: the band's rms error is
    under the tolerance the rule sized the step for and carries most of it, and the pilot resolves it. Pilots whose
    bands stop below the band certified, or of different band counts, are refused by name."""
    from dmipy_sim.engine.physics import regime_sub_steps
    g = d.Sphere(2e-6)
    T, K, f_band = 0.04, 80, 1000.0
    h = block_length(T, K)
    n_rule = tolerance_sub_steps(g, D0, h, f_band=f_band)
    n_fine = max(2, n_rule // 4)
    assert n_fine // PILOT_STEP_RATIO >= regime_sub_steps(g, D0, h)
    eps = STEP_TOLERANCE * n_rule / n_fine                                         # first order: eps scales as the step
    n_coarse = max(1, n_fine // PILOT_STEP_RATIO)
    kw = dict(K=K, require_gpu=False, window_T=T, tiers=())
    fine = _walk_geometry(16000, D0, g, T, seed=1, sub_steps=n_fine, **kw)
    coarse = _walk_geometry(16000, D0, g, T, seed=2, sub_steps=n_coarse, **kw)
    assert fine.h == coarse.h == pytest.approx(h)
    cert = step_certificate(fine.position_coeffs, coarse.position_coeffs, T, n_fine / n_coarse, f_band=f_band,
                            eps_declared=eps)
    assert cert["n_bins"] == 80 and cert["ratio"] == pytest.approx(n_fine / n_coarse) and cert["resolves"]
    assert cert["floor"] < eps
    assert 0.4 * eps < cert["eps_measured"] < 1.1 * eps + 3 * cert["eps_se"]
    with pytest.raises(ValueError, match="K >= 2 T f_band"):
        step_certificate(fine.position_coeffs, coarse.position_coeffs, T, 4.0, f_band=2000.0, eps_declared=eps)
    with pytest.raises(ValueError, match="no band of a window"):
        step_certificate(fine.position_coeffs, coarse.position_coeffs, T, 4.0, f_band=1.0, eps_declared=eps)
    with pytest.raises(ValueError, match="one window, K and block"):
        step_certificate(fine.position_coeffs, coarse.position_coeffs[:, :, :-1], T, 4.0, f_band=f_band,
                         eps_declared=eps)
