"""Two-compartment mixture (no exchange).

The mixture signal is the volume-fraction-weighted sum of each compartment's
signal. We test this against separately computed single-compartment signals.
"""

import numpy as np
import numpy.testing as npt

from dmipy_sim import simulate, simulate_mixture, FreeDiffusion, Sphere, set_b
from dmipy_sim.waveforms import pgse
from .conftest import D, N_WALKERS, SEED


def _waveform(n_b=20):
    b_values = np.linspace(1e8, 2e9, n_b)
    bvecs = np.tile([1., 0., 0.], (n_b, 1))
    return set_b(pgse(delta=0.2e-3, DELTA=40e-3, G_magnitude=1.0,
                      bvecs=bvecs, n_t=1000), b_values), b_values


def test_mixture_is_weighted_sum():
    """simulate_mixture must equal f1*S1 + f2*S2 up to MC noise."""
    wf, _ = _waveform()
    f1, f2 = 0.7, 0.3

    S1 = simulate(N_WALKERS, D,        wf, FreeDiffusion(), seed=SEED)
    S2 = simulate(N_WALKERS, D * 0.5,  wf, Sphere(5e-6),   seed=SEED + 1)

    S_mix = simulate_mixture([
        {'fraction': f1, 'n_walkers': N_WALKERS, 'diffusivity': D,
         'geometry': FreeDiffusion()},
        {'fraction': f2, 'n_walkers': N_WALKERS, 'diffusivity': D * 0.5,
         'geometry': Sphere(5e-6)},
    ], wf, seed=SEED)

    expected = f1 * S1 + f2 * S2
    npt.assert_allclose(S_mix, expected, atol=1e-6,
                        err_msg="Mixture signal must equal weighted sum of compartments")


def test_mixture_between_compartments():
    """Mixture signal must lie between the two compartment signals at each b."""
    wf, _ = _waveform()
    f1, f2 = 0.5, 0.5

    S_free     = simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED)
    S_sphere   = simulate(N_WALKERS, D, wf, Sphere(5e-6),   seed=SEED + 1)
    S_mix      = simulate_mixture([
        {'fraction': f1, 'n_walkers': N_WALKERS, 'diffusivity': D,
         'geometry': FreeDiffusion()},
        {'fraction': f2, 'n_walkers': N_WALKERS, 'diffusivity': D,
         'geometry': Sphere(5e-6)},
    ], wf, seed=SEED)

    lo = np.minimum(S_free, S_sphere)
    hi = np.maximum(S_free, S_sphere)
    assert np.all(S_mix >= lo - 0.01) and np.all(S_mix <= hi + 0.01), (
        "Mixture signal must lie between the two component signals")


def test_mixture_fraction_limit():
    """f=1.0 for a single compartment must reproduce simulate() exactly."""
    wf, _ = _waveform()

    S_direct = simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED)
    S_mix    = simulate_mixture([
        {'fraction': 1.0, 'n_walkers': N_WALKERS, 'diffusivity': D,
         'geometry': FreeDiffusion()},
    ], wf, seed=SEED)

    npt.assert_allclose(S_mix, S_direct, atol=1e-6,
                        err_msg="f=1 mixture must equal direct simulate()")


def test_mixture_fractions_must_sum_to_one():
    """Invalid fractions raise ValueError."""
    import pytest
    wf, _ = _waveform(n_b=1)
    with pytest.raises(ValueError, match="sum to 1"):
        simulate_mixture([
            {'fraction': 0.6, 'n_walkers': 100, 'diffusivity': D,
             'geometry': FreeDiffusion()},
            {'fraction': 0.6, 'n_walkers': 100, 'diffusivity': D,
             'geometry': FreeDiffusion()},
        ], wf)
