"""The b of an unbalanced waveform is the attenuation the walk produces, the moment anchored at the readout
(dmipy-sim#392): a free walker from a fixed start accrues ``sum_s dr_s . (q(T) - q(s))``, so the Gaussian
attenuation is ``exp(-D int |q - q(T)|^2 dt)``, which is the textbook ``int |q|^2 dt`` only when ``q(T) = 0``."""
import numpy as np
import pytest

from dmipy_sim.acquisition.waveforms import b_from_gradient, btensor_from_gradient
from dmipy_sim.constants import GAMMA
from dmipy_sim import sequences


def _spoiled_pgse(turns_per_mm):
    """A PGSE at b = 1000 s/mm^2 with a played spoiler after its second lobe: ``turns_per_mm`` turns per mm."""
    seq = sequences.pgse([[1.0, 0.0, 0.0]], 5e-3, 15e-3, bvalues=[1.0e9], TE=40e-3, n_t=400, slew_rate=np.inf)
    G = np.array(seq.G_eff, dtype=np.float64).copy()
    n = 20                                                            # a 2 ms lobe at the end of the grid
    G[0, -n - 1:-1, 0] = 2.0 * np.pi * turns_per_mm * 1e3 / (GAMMA * n * seq.dt)   # winding along the encoding axis
    return G, float(seq.dt)


def test_a_balanced_waveform_is_unchanged():
    seq = sequences.pgse([[1.0, 0.0, 0.0]], 5e-3, 15e-3, bvalues=[1.0e9], TE=40e-3, n_t=400, slew_rate=np.inf)
    G = np.asarray(seq.G_eff, dtype=np.float64)
    q = np.cumsum(G * seq.dt, axis=1) * GAMMA
    assert np.allclose(q[:, -1, :], 0.0, atol=1e-6)
    assert b_from_gradient(G, seq.dt) == pytest.approx(np.trapezoid(np.sum(q ** 2, axis=2), dx=seq.dt, axis=1), rel=1e-12)
    assert np.trace(btensor_from_gradient(G, seq.dt)[0]) == pytest.approx(b_from_gradient(G, seq.dt)[0], rel=1e-12)


def test_an_unbalanced_waveform_attenuates_free_walkers_at_the_readout_anchored_b():
    """200 000 free walkers from the origin under a spoiled PGSE: ``-ln E / D`` is the readout-anchored b, and
    not the ``t = 0``-anchored integral, which is the same number only for a balanced waveform."""
    D = 2.0e-9
    G, dt = _spoiled_pgse(turns_per_mm=8.0)
    n_t = G.shape[1]
    q = np.cumsum(G * dt, axis=1) * GAMMA
    b_zero_anchored = np.trapezoid(np.sum(q ** 2, axis=2), dx=dt, axis=1)[0]
    b_readout = b_from_gradient(G, dt)[0]
    # the spoiler along the encoding axis cancels part of the plateau's moment, so the weighting DROPS: a
    # displacement matters by the moment still to come after it, not by the moment accrued before it
    assert abs(b_readout - b_zero_anchored) > 0.05 * b_zero_anchored
    rng = np.random.default_rng(0)
    n_w = 200_000
    phi = np.zeros(n_w)
    r = np.zeros((n_w, 3))
    for t in range(n_t):                                              # the walk's own left-point rule
        r += rng.normal(0.0, np.sqrt(2.0 * D * dt), (n_w, 3))
        phi += GAMMA * dt * (r @ G[0, t])
    E = np.abs(np.mean(np.exp(1j * phi)))
    b_walk = -np.log(E) / D
    se = np.sqrt((1.0 - E ** 2) / (2.0 * n_w)) / (E * D)              # the standard error of -ln E / D
    assert b_walk == pytest.approx(b_readout, abs=3.0 * se + 0.01 * b_readout)
    assert abs(b_walk - b_zero_anchored) > 3.0 * se + 0.01 * b_readout
