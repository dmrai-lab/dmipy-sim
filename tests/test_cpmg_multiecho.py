"""Multi-echo CPMG signal from a SINGLE walk (simulate_cpmg): T2 decay + diffusion.

No EPG, no trajectory replay — one walk through the ideal-180 train, signal sampled at
each echo. At b=0 the per-echo signal is the pure T2 decay exp(-k*TE/T2).
"""
import numpy as np
import numpy.testing as npt

from dmipy_sim import cpmg, simulate_cpmg, simulate_bloch, Sphere, FreeDiffusion


def test_cpmg_pure_t2_decay():
    """G=0 CPMG: echo k signal == exp(-k*TE/T2) (single walk, all echoes)."""
    TE, T2, n_ech = 12e-3, 60e-3, 8
    wf = cpmg(n_ech, TE, gradient_strengths=0.0, gradient_directions=[[1, 0, 0]], n_t_per_echo=60)
    S = np.asarray(simulate_cpmg(20000, 2e-9, wf, Sphere(radius=5e-6), T2=T2,
                                 seed=1, require_gpu=False)).ravel()
    expected = np.exp(-np.arange(1, n_ech + 1) * TE / T2)
    npt.assert_allclose(S, expected, atol=0.01)
    assert S.shape == (n_ech,)


def test_cpmg_returns_all_echoes_monotone():
    """Diffusion-weighted CPMG returns one signal per echo, monotonically decaying."""
    wf = cpmg(6, 10e-3, gradient_strengths=0.02, gradient_directions=[[1, 0, 0]], n_t_per_echo=50)
    S = np.asarray(simulate_cpmg(20000, 2e-9, wf, FreeDiffusion(), T2=80e-3,
                                 seed=2, require_gpu=False)).ravel()
    assert S.shape == (6,)
    assert np.all(np.diff(S) <= 1e-3)      # non-increasing across echoes


def test_meiboom_gill_is_the_default_and_carr_purcell_the_option():
    """The refocusing pulses turn about y (a quarter turn from the 90 about x): a flip-angle error then cancels
    every second echo (Meiboom-Gill) instead of accumulating (Carr-Purcell, refocus_axis_deg=0). Both trains
    are the same object to the transverse-only engine; the vector-Bloch engine plays the axes."""
    mg = cpmg(8, 10e-3, beta_deg=150.0, n_t_per_echo=20)
    cp = cpmg(8, 10e-3, beta_deg=150.0, refocus_axis_deg=0.0, n_t_per_echo=20)
    assert [e.axis_deg for e in mg.rf] == [0.0] + [90.0] * 8 and [e.axis_deg for e in cp.rf] == [0.0] * 9
    assert mg.encoding.cpmg_refocus_axis_deg == 90.0 and cp.encoding.cpmg_refocus_axis_deg == 0.0
    S_mg = np.abs(np.asarray(simulate_bloch(400, 2e-9, mg, FreeDiffusion(), seed=0, require_gpu=False))).ravel()
    S_cp = np.abs(np.asarray(simulate_bloch(400, 2e-9, cp, FreeDiffusion(), seed=0, require_gpu=False))).ravel()
    assert S_mg.shape == (8,) and S_mg[1] > 0.95 and S_mg[7] > 0.9            # the even echoes recover
    assert S_cp[7] < 0.6 < S_mg[7]                                              # Carr-Purcell decays with the error
