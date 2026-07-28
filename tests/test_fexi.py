"""FEXI (filter-exchange) stimulated-echo sequence constructor.

Fast mechanics checks on free diffusion (no substrate, so no exchange): the detection block
must read the true diffusivity, the two encodings are self-refocused, and the measured ADC is
independent of the mixing time (there are no pools to exchange). The AXR exchange recovery itself
needs a permeable packed substrate and is a heavier GPU demo, not a unit test.
"""
import numpy as np
import pytest

from dmipy_sim import FreeDiffusion
from dmipy_sim.pulse_sequence import fexi, run_bloch_sequence


def _adc(seq, D=2e-9, n=6000):
    S = run_bloch_sequence(seq, n, D, FreeDiffusion(), seed=1, require_gpu=False)  # (n_meas,)
    S = np.abs(np.asarray(S))
    return -np.log(S[1] / S[0]) / seq.b_detect[1]


def test_fexi_builds_and_carries_b_detect():
    seq = fexi(delta=5e-3, t_mix=30e-3, dt=2e-4, g_filter=0.15, g_detect=[0.0, 0.4])
    assert seq.family == "fexi" and seq.complex_signal
    assert len(seq.rf_events) == 3 and all(e['flip_deg'] == 90 for e in seq.rf_events)
    assert seq.crusher is not None and seq.b_detect.shape == (2,)
    assert seq.b_detect[0] == 0.0 and seq.b_detect[1] > 1e8       # a real detection weighting


def test_detection_reads_true_diffusivity():
    seq = fexi(delta=5e-3, t_mix=30e-3, dt=2e-4, g_filter=0.0, g_detect=[0.0, 0.45])
    adc = _adc(seq)
    assert adc == pytest.approx(2e-9, rel=0.1)                    # free ADC ≈ D


def test_adc_independent_of_mixing_time_in_free_diffusion():
    """No pools to exchange ⇒ the filtered ADC must not drift with t_mix (the correct null)."""
    a_short = _adc(fexi(delta=5e-3, t_mix=20e-3, dt=2e-4, g_filter=0.15, g_detect=[0.0, 0.45]))
    a_long = _adc(fexi(delta=5e-3, t_mix=120e-3, dt=2e-4, g_filter=0.15, g_detect=[0.0, 0.45]))
    assert a_short == pytest.approx(2e-9, rel=0.12)
    assert a_long == pytest.approx(a_short, rel=0.12)


def test_filter_attenuates_stored_signal():
    """Turning the diffusion filter on must reduce the stored b=0 signal."""
    off = fexi(delta=6e-3, t_mix=30e-3, dt=2e-4, g_filter=0.0, g_detect=[0.0])
    on = fexi(delta=6e-3, t_mix=30e-3, dt=2e-4, g_filter=0.5, g_detect=[0.0])
    S_off = abs(complex(run_bloch_sequence(off, 6000, 2e-9, FreeDiffusion(), seed=1, require_gpu=False)[0]))
    S_on = abs(complex(run_bloch_sequence(on, 6000, 2e-9, FreeDiffusion(), seed=1, require_gpu=False)[0]))
    assert S_on < 0.7 * S_off
