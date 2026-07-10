"""Multi-echo CPMG signal from a SINGLE walk (simulate_cpmg): T2 decay + diffusion.

No EPG, no trajectory replay — one walk through the ideal-180 train, signal sampled at
each echo. At b=0 the per-echo signal is the pure T2 decay exp(-k*TE/T2).
"""
import numpy as np
import numpy.testing as npt

from dmipy_sim import cpmg, simulate_cpmg, Sphere, FreeDiffusion


def test_cpmg_pure_t2_decay():
    """G=0 CPMG: echo k signal == exp(-k*TE/T2) (single walk, all echoes)."""
    TE, T2, n_ech = 12e-3, 60e-3, 8
    wf = cpmg(n_echoes=n_ech, TE=TE, G_magnitude=0.0, bvecs=[[1, 0, 0]], n_t_per_echo=60)
    S = np.asarray(simulate_cpmg(20000, 2e-9, wf, Sphere(radius=5e-6), T2=T2,
                                 seed=1, require_gpu=False)).ravel()
    expected = np.exp(-np.arange(1, n_ech + 1) * TE / T2)
    npt.assert_allclose(S, expected, atol=0.01)
    assert S.shape == (n_ech,)


def test_cpmg_returns_all_echoes_monotone():
    """Diffusion-weighted CPMG returns one signal per echo, monotonically decaying."""
    wf = cpmg(n_echoes=6, TE=10e-3, G_magnitude=0.02, bvecs=[[1, 0, 0]], n_t_per_echo=50)
    S = np.asarray(simulate_cpmg(20000, 2e-9, wf, FreeDiffusion(), T2=80e-3,
                                 seed=2, require_gpu=False)).ravel()
    assert S.shape == (6,)
    assert np.all(np.diff(S) <= 1e-3)      # non-increasing across echoes
