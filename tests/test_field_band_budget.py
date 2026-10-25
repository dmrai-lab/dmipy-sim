"""An operational request size must not choose field capability."""
import numpy as np
import pytest
from dmipy_sim.fields.strand_field import CHANNEL_NAMES
from dmipy_sim.replay.bank import derive_field_band


def series():
    t = (np.arange(32)+.5)/32
    values = np.zeros((12,32,len(CHANNEL_NAMES)))
    values[:,:,1] = np.linspace(.8,1.2,12)[:,None] * 1e-4 * np.sin(2*np.pi*t)
    values[:,:,2] = -values[:,:,1]
    return values


def test_explicit_field_budget_does_not_depend_on_pilot_noise_or_request_count():
    values = series()
    kw = dict(K_max=16, B0=3, chi=1e-6, target_floor=1e-4,
              env=dict(B0_list=[3],theta_deg=[90],max_refocus_pulses=1))
    k, record = derive_field_band(values, CHANNEL_NAMES, .001, **kw)
    doubled, other = derive_field_band(np.repeat(values,2,axis=0), CHANNEL_NAMES, .001, **kw)
    assert doubled == k and record["floor"] == other["floor"] == 1e-4
    assert record["within_floor"] and "explicit" in record["floor_is"]


def test_an_unachievable_budget_is_refused_instead_of_selecting_a_failing_rung():
    with pytest.raises(ValueError, match="no field band"):
        derive_field_band(series(), CHANNEL_NAMES, .001, K_max=16, B0=3, chi=1e-6, target_floor=1e-30,
                          env=dict(B0_list=[3],theta_deg=[90],max_refocus_pulses=1))
