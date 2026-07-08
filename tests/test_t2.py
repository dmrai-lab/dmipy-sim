"""T2 relaxation as a post-hoc multiplicative factor.

T2 decay is applied as exp(-TE/T2) multiplied onto the diffusion signal.
At b=0 (no diffusion weighting), signal equals exp(-TE/T2) exactly.
"""

import numpy as np
import numpy.testing as npt

from dmipy_sim import simulate, FreeDiffusion, set_b
from dmipy_sim.waveforms import pgse
from .conftest import D, N_WALKERS, SEED


def _waveform_b0(n_b=1):
    """PGSE waveform with b≈0 (tiny gradient)."""
    bvecs = np.tile([1., 0., 0.], (n_b, 1))
    # Very small b so diffusion attenuation is negligible
    wf = set_b(pgse(delta=1e-3, DELTA=40e-3, G_magnitude=1.0,
                    bvecs=bvecs, n_t=500), np.full(n_b, 1.0))
    return wf  # b = 1 s/m² → exp(-bD) ≈ 1.0


def test_t2_at_b0_equals_exp_te_t2():
    """At b≈0, T2-weighted signal equals exp(-TE/T2) regardless of geometry."""
    T2 = 80e-3  # 80 ms
    wf = _waveform_b0()
    TE = wf.echo_idx * wf.dt

    signal = simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED, T2=T2)
    expected = np.exp(-TE / T2)

    npt.assert_allclose(signal, expected, atol=0.01,
                        err_msg=f"T2 signal at b≈0 must equal exp(-TE/T2)={expected:.4f}")


def test_t2_scales_diffusion_signal():
    """T2 signal equals (diffusion-only signal) × exp(-TE/T2).

    T2 is now accumulated per-walker inside the scan body, so the two
    simulations run different JAX carry shapes and diverge at the level of
    MC noise.  The tolerance is set to the MC noise floor (1/sqrt(N_walkers)).
    """
    T2 = 60e-3
    b_values = np.linspace(1e8, 2e9, 20)
    bvecs = np.tile([1., 0., 0.], (20, 1))
    wf = set_b(pgse(delta=1e-3, DELTA=40e-3, G_magnitude=1.0,
                    bvecs=bvecs, n_t=500), b_values)
    # Magnetisation is fully transverse throughout, so T2 accumulates over the
    # whole scan: tau_perp = n_t * dt.
    tau_perp = wf.G.shape[1] * wf.dt

    S_no_T2 = simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED)
    S_T2    = simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED, T2=T2)

    # MC noise floor: 1/sqrt(N_walkers)
    atol_mc = 1.0 / np.sqrt(N_WALKERS)
    npt.assert_allclose(S_T2, S_no_T2 * np.exp(-tau_perp / T2), atol=3 * atol_mc,
                        err_msg="T2 signal must equal diffusion signal × exp(-tau_perp/T2)")


def test_shorter_t2_gives_lower_signal():
    """Shorter T2 → stronger T2 attenuation → lower signal."""
    b_values = np.linspace(1e8, 1e9, 10)
    bvecs = np.tile([1., 0., 0.], (10, 1))
    wf = set_b(pgse(delta=1e-3, DELTA=40e-3, G_magnitude=1.0,
                    bvecs=bvecs, n_t=500), b_values)

    S_long_T2  = simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED, T2=200e-3)
    S_short_T2 = simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED, T2=40e-3)

    assert np.all(S_short_T2 <= S_long_T2 + 1e-5), (
        "Shorter T2 must give lower or equal signal at all b-values")


def test_no_t2_unaffected():
    """Omitting T2 must give same result as simulate() without T2 kwarg."""
    b_values = np.array([1e9])
    bvecs = np.array([[1., 0., 0.]])
    wf = set_b(pgse(delta=1e-3, DELTA=40e-3, G_magnitude=1.0,
                    bvecs=bvecs, n_t=500), b_values)

    S1 = simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED)
    S2 = simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED, T2=None)

    npt.assert_array_equal(S1, S2)
