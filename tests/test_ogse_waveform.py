"""OGSE waveforms are valid, frequency-selective gradients.

A real OGSE must (i) null its gradient moment at the echo (q(TE)=0 -> diffusion-
only, no net imaging gradient), and the cosine form must (ii) be DC-free (zero-mean
q) with (iii) a gradient power spectrum peaked at the oscillation frequency. These
properties failed before the fix (the cosine sign-flip at T/2 was not aligned to an
integer number of periods, leaving a DC component and a frequency-shifted peak).
"""
import numpy as np
import pytest

from dmipy_sim.waveforms import ogse, trapezoidal_ogse
from dmipy_sim.constants import GAMMA

BVECS = np.array([[1.0, 0.0, 0.0]])


def _q_and_dt(wf):
    G = np.asarray(wf.G)[0, :, 0]
    dt = float(wf.dt)
    # trapezoidal zeroth moment (left-Riemann leaves an O(dt) residual that is a
    # sampling artifact, not a waveform property)
    q = GAMMA * np.concatenate([[0.0], np.cumsum(0.5 * (G[1:] + G[:-1])) * dt])
    return q, dt, G


@pytest.mark.parametrize("freq", [25.0, 50.0, 100.0])
def test_cosine_ogse_refocused_and_dc_free(freq):
    wf = ogse(frequency=freq, T_total=0.080, G_magnitude=0.05, bvecs=BVECS,
              n_t=1601, kind='cosine')
    q, dt, G = _q_and_dt(wf)
    qmax = np.max(np.abs(q)) + 1e-30
    # (i) gradient moment nulled at the echo
    assert abs(q[-1]) / qmax < 1e-2, "cosine-OGSE not refocused (q(TE)!=0)"
    # (ii) DC-free: q is zero-mean (the defining OGSE property)
    assert abs(np.mean(q)) / qmax < 0.05, "cosine-OGSE q has a DC component"


@pytest.mark.parametrize("freq", [25.0, 50.0, 100.0])
def test_sine_ogse_also_refocused(freq):
    """The sine SE-OGSE is also a valid refocused waveform (the 180 symmetrises q)."""
    wf = ogse(frequency=freq, T_total=0.080, G_magnitude=0.05, bvecs=BVECS,
              n_t=1601, kind='sine')
    q, _, _ = _q_and_dt(wf)
    assert abs(q[-1]) / (np.max(np.abs(q)) + 1e-30) < 1e-2, "sine-OGSE not refocused"


@pytest.mark.parametrize("freq", [50.0, 100.0])
def test_cosine_ogse_spectral_peak_near_frequency(freq):
    T = 0.080
    wf = ogse(frequency=freq, T_total=T, G_magnitude=0.05, bvecs=BVECS,
              n_t=2001, kind='cosine')
    q, dt, G = _q_and_dt(wf)
    spec = np.abs(np.fft.rfft(q)) ** 2
    fr = np.fft.rfftfreq(len(q), dt)
    f_peak = fr[1 + np.argmax(spec[1:])]            # skip DC bin
    # snapped frequency = nearest integer periods per half-echo
    N = max(1, round(freq * T / 2)); f_eff = N / (T / 2)
    assert abs(f_peak - f_eff) <= 1.5 / T + 1.0, \
        f"spectral peak {f_peak:.0f} far from f_eff {f_eff:.0f}"


def test_trapezoidal_ogse_refocused():
    wf = trapezoidal_ogse(N=2, delta=0.030, DELTA=0.040, G_magnitude=0.05,
                          bvecs=BVECS, n_t=801)
    q, dt, G = _q_and_dt(wf)
    assert abs(q[-1]) / (np.max(np.abs(q)) + 1e-30) < 0.05, \
        "trapezoidal OGSE not refocused"


def test_ogse_frequency_snapping_warns():
    # 37 Hz over an 80 ms echo -> 37*0.04=1.48 periods/half -> snaps to 1 (25 Hz)
    with pytest.warns(UserWarning, match="snapped"):
        ogse(frequency=37.0, T_total=0.080, G_magnitude=0.05, bvecs=BVECS, n_t=401)
