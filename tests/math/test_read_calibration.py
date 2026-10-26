"""Actual-rung calibration and explicit refusal of aliased field references."""
import numpy as np
import pytest
from dmipy_sim.engine.accumulate import calibrate_field_reads, READ_CONSTANT


def calibrate(values, **kwargs):
    return calibrate_field_reads(values, labels=np.zeros(len(values),int), pool_names={0:"pool"},
        step_s=.01, rounds_per_block=4, blocks_per_window=2, B0=1, chi=1e-9,
        T_e=.08, floor=1e-3, **kwargs)[0]


def test_linear_field_uses_its_actual_selected_rung_with_explicit_uncertainty():
    t = np.arange(9)*.01
    values = np.broadcast_to(t[None,:,None],(12,9,1))
    record = calibrate(values, given=.03)
    assert record["n_read"] == 3 and record["constant_evidence"]["selected_rung"] == 3
    assert record["constants"] == pytest.approx([READ_CONSTANT])
    assert record["constant_evidence"]["reference"] == "every-round trapezoids"
    assert record["budget_met"] and record["calibration_resolved"]


def test_aliasing_cannot_hide_nonzero_integral_error_behind_zero_read_variation():
    # Every fourth read is zero, while the reference sees a positive integral in each block.
    values = np.broadcast_to(np.array([0,1,1,1,0,1,1,1,0])[None,:,None],(12,9,1))
    with pytest.raises(ValueError, match="aliased"):
        calibrate(values, given=.04)


def test_finest_read_does_not_claim_its_constant_was_measured_at_that_read():
    values = np.ones((12,9,1))
    record = calibrate(values, given=.001)
    assert record["n_read"] == 1 and not record["constants_measured_at_read"]
    assert record["at_step"] and record["term_pilot"] == 0
