"""OGSE waveforms are valid, frequency-selective gradients.

A real OGSE must (i) null its gradient moment at the echo (q(TE)=0 -> diffusion-only, no net imaging
gradient), and the cosine form must (ii) be DC-free (zero-mean q) with (iii) a gradient power spectrum peaked
at the oscillation frequency. Both shapes play the same block on either side of the 180 and hold a whole
number of periods (cosine) or half-periods (trapezoid lobes); a block that would not is refused, not snapped.
"""
import numpy as np
import pytest

from dmipy_sim.sequences import ogse
from dmipy_sim.constants import GAMMA

BVECS = np.array([[1.0, 0.0, 0.0]])


def _q_and_dt(wf):
    """q as the walk accumulates it: ``G[k]`` acts over the step ``[k dt, (k + 1) dt)``."""
    G = np.asarray(wf.G_eff)[0, :, 0]          # the effective gradient: what q integrates
    dt = float(wf.dt)
    q = GAMMA * np.cumsum(G) * dt
    return q, dt, G


@pytest.mark.parametrize("freq", [25.0, 50.0, 100.0])
def test_cosine_ogse_refocused_and_dc_free(freq):
    wf = ogse(BVECS, freq, 0.040, gradient_strengths=0.05, shape="cosine", slew_rate=np.inf, n_t=1601)
    q, dt, G = _q_and_dt(wf)
    qmax = np.max(np.abs(q)) + 1e-30
    # (i) gradient moment nulled at the echo
    assert abs(q[wf.echo_idx - 1]) / qmax < 1e-6, "cosine-OGSE not refocused (q(TE)!=0)"
    # (ii) DC-free: q is zero-mean (the defining OGSE property)
    assert abs(np.mean(q)) / qmax < 0.05, "cosine-OGSE q has a DC component"


@pytest.mark.parametrize("freq", [50.0, 100.0])
def test_cosine_ogse_spectral_peak_at_the_frequency(freq):
    wf = ogse(BVECS, freq, 0.040, gradient_strengths=0.05, shape="cosine", slew_rate=np.inf, n_t=2001)
    q, dt, G = _q_and_dt(wf)
    spec = np.abs(np.fft.rfft(q)) ** 2
    fr = np.fft.rfftfreq(len(q), dt)
    f_peak = fr[1 + np.argmax(spec[1:])]            # skip DC bin
    assert abs(f_peak - freq) <= 1.5 / wf.T + 1.0, f"spectral peak {f_peak:.0f} far from {freq:.0f}"


def test_trapezoidal_ogse_refocused_at_every_slew():
    for slew in (200e3, 200.0, np.inf):
        wf = ogse(BVECS, 2 / (2 * 0.030), 0.030, gradient_strengths=0.05, shape="trapezoid", Delta=0.040, n_t=801,
                  slew_rate=slew)
        q, dt, G = _q_and_dt(wf)
        assert abs(q[wf.echo_idx - 1]) / (np.max(np.abs(q)) + 1e-30) < 1e-6, "trapezoidal OGSE not refocused"
        assert wf.rf.refocus_time == pytest.approx(0.035)              # the 180 at TE/2, the blocks against it
    one = ogse(BVECS, 1 / (2 * 0.030), 0.030, gradient_strengths=0.05, shape="trapezoid", n_t=801)   # one lobe
    G = np.asarray(one.G)[0, :, 0]
    assert np.all(G >= 0.0) and one.refocusing_residual < 1e-6            # a PGSE: two same-sign lobes


def test_a_block_that_does_not_hold_whole_periods_is_refused():
    # 37 Hz over a 40 ms block -> 1.48 periods: refused rather than snapped
    with pytest.raises(ValueError, match="must be whole"):
        ogse(BVECS, 37.0, 0.040, gradient_strengths=0.05, shape="cosine", slew_rate=np.inf, n_t=401)
    with pytest.raises(ValueError, match="must be whole"):
        ogse(BVECS, 37.0, 0.040, gradient_strengths=0.05, shape="trapezoid", n_t=401)
    with pytest.raises(ValueError, match="shape must be"):
        ogse(BVECS, 25.0, 0.040, gradient_strengths=0.05, shape="sine", n_t=401)


def test_slew_is_a_limit_not_a_fork():
    """A finite slew ramps the edges of the same block; a cosine whose own slope exceeds it is refused."""
    ideal = ogse(BVECS, 50.0, 0.040, gradient_strengths=0.03, shape="cosine", slew_rate=np.inf, n_t=801)
    ramped = ogse(BVECS, 50.0, 0.040, gradient_strengths=0.03, shape="cosine", slew_rate=200.0, n_t=801)
    Gi, Gr = np.asarray(ideal.G)[0, :, 0], np.asarray(ramped.G)[0, :, 0]
    assert ideal.n_t == ramped.n_t and ideal.T == ramped.T
    inner = slice(int(0.005 / ideal.dt), int(0.035 / ideal.dt))
    np.testing.assert_allclose(Gr[inner], Gi[inner], atol=1e-12)          # the same block inside
    assert abs(Gr[0]) < abs(Gi[0])                                        # ramped into at the edge
    with pytest.raises(ValueError, match="slews at"):
        ogse(BVECS, 500.0, 0.010, gradient_strengths=0.1, shape="cosine", slew_rate=200.0, n_t=801)
