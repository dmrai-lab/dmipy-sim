"""Calibration reuse mechanics with explicit synthetic evidence; these fixtures do not certify physics."""
import numpy as np
import pytest
from dmipy_sim.engine import adaptive
from dmipy_sim.engine.backends import parity_cases
from dmipy_sim.fields.strand_field import StrandFieldBasis
from dmipy_sim.engine.calibration import FieldCalibration, REQUIRED_TERMS, SIGNAL_METRIC
from dmipy_sim.math.accuracy import ErrorBound, assess_accuracy
from dmipy_sim.math.step_law import assess_step_certificate


def synthetic_evidence():
    term = lambda:ErrorBound(.001, SIGNAL_METRIC, 1.0, "synthetic workflow fixture, not physical evidence")
    accuracy = assess_accuracy(target=.01, metric=SIGNAL_METRIC, confidence=.99, statistical=term(),
                               numerical={name:term() for name in REQUIRED_TERMS}, required=REQUIRED_TERMS)
    step = assess_step_certificate(dict(eps_measured=.0001, eps_se=.00001, floor=.0001,
                                       eps_declared=.001, resolves=True))
    return accuracy, step


def test_a_valid_scoped_record_skips_the_hidden_pilot_and_does_not_depend_on_count(monkeypatch):
    geometry = parity_cases()["strands_extra"]()
    basis = StrandFieldBasis(geometry.centerlines, .7*geometry.radii, geometry.radii,
                             cutoff_m=12e-6, domain=geometry.box)
    kw = dict(T_max=2e-4, block_h=1e-4, K=4, K_c3=2, field_basis=basis,
              query_scope=dict(identity="synthetic-query", metric=SIGNAL_METRIC, target=.01, confidence=.99), require_gpu=False, walker_batch_size=32)
    original = adaptive.simulate_trajectories_adaptive(16, 2e-9, geometry, seed=3, **kw)
    accuracy, step = synthetic_evidence()
    calibration = FieldCalibration.from_walk(original, accuracy=accuracy, step=step)
    from dmipy_sim.engine import accumulate
    def no_pilot(*args, **kwargs):
        pytest.fail("a reused calibration must not run its pilot again")
    monkeypatch.setattr(accumulate, "calibrate_field_reads", no_pilot)
    reused = adaptive.simulate_trajectories_adaptive(24, 2e-9, geometry, seed=19,
                                                   field_calibration=calibration,
                                                   field_read_interval=original.field_read_interval + .1*original.stepping["round_s"], **kw)
    assert reused.n_walkers == 24 and reused.K_c3 == original.K_c3
    assert reused.block["field_read"]["reused"] and reused.block["field_read"]["pilot_seconds"] == 0
    assert reused.stepping["field_pilot_backend"] is None
    with pytest.raises(ValueError, match="physics"):
        adaptive.simulate_trajectories_adaptive(24, 3e-9, geometry, field_calibration=calibration, **kw)

    # Actual protocol/field budgets are part of the scope, not descriptive labels.
    for override, group in [
        (dict(envelope=dict(B0_list=[3.])), "query"),
        (dict(query_scope=dict(kw["query_scope"], identity="different-RF-and-tissue")), "query"),
        (dict(calibration_floor=.002), "numerics"),
    ]:
        with pytest.raises(ValueError, match=group):
            adaptive.simulate_trajectories_adaptive(24, 2e-9, geometry,
                field_calibration=calibration, **dict(kw, **override))
