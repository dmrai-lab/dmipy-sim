"""Validated calibration records cannot drift across physical/query or numerical domains."""
import json
import pytest
from dmipy_sim.engine.calibration import FieldCalibration, REQUIRED_TERMS, SIGNAL_METRIC
from dmipy_sim.math.accuracy import ErrorBound, assess_accuracy
from dmipy_sim.math.step_law import assess_step_certificate, STEP_METRIC


def arguments():
    scope = dict(physics="physical-input-hash", numerics=dict(blocks_per_window=64, floor=.001),
                 query=dict(request=dict(identity="actual-protocol-and-tissue-hash", metric=SIGNAL_METRIC, target=.01, confidence=.99)), backend="backend-source-and-device", code="public-source-hash")
    terms = {name:ErrorBound(.0001, SIGNAL_METRIC, 1.0, "scoped analytic or refinement evidence")
             for name in REQUIRED_TERMS}
    accuracy = assess_accuracy(target=.01, metric=SIGNAL_METRIC, confidence=.99,
                              statistical=ErrorBound(.001, SIGNAL_METRIC, 1.0, "bounded signal estimator"),
                              numerical=terms, required=REQUIRED_TERMS)
    step = assess_step_certificate(dict(eps_measured=.0001, eps_se=.00001, floor=.0001,
                                       eps_declared=.001, resolves=True, metric=STEP_METRIC,
                                       octaves=[dict(f_lo_hz=5., f_hi_hz=1300., n_bins=260, eps=.0001,
                                                    eps_se=.00001, floor=.0001)]))
    return dict(scope=scope, accuracy=accuracy, step=step,
                read=dict(dt_f=.0001, n_read=2, constants=[1/12]*12),
                band=dict(K=16))


def test_snapshot_and_roundtrip_do_not_share_mutable_state(tmp_path):
    args = arguments()
    record = FieldCalibration.create(**args)
    args["read"]["constants"][0] = 99
    assert record.record["read"]["constants"][0] == pytest.approx(1/12)
    path = tmp_path / "calibration.json"
    record.save(path)
    assert FieldCalibration.load(path).record == record.record

    record.for_scope(args["scope"])
    data = json.loads(path.read_text())
    data["band"]["K"] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        FieldCalibration.load(path)


@pytest.mark.parametrize("group", ["physics", "numerics", "query", "backend", "code"])
def test_reuse_refuses_each_kind_of_scope_drift(group):
    args = arguments()
    record = FieldCalibration.create(**args)
    args["scope"][group] = "different"
    with pytest.raises(ValueError, match=group):
        record.for_scope(args["scope"])


@pytest.mark.parametrize("missing", REQUIRED_TERMS)
def test_missing_physical_error_evidence_is_not_hidden_by_a_pass_flag(missing):
    args = arguments()
    del args["accuracy"]["terms"][missing]
    with pytest.raises(ValueError, match="accuracy is unresolved"):
        FieldCalibration.create(**args)


def test_unresolved_step_is_not_reusable_even_with_a_passing_signal_budget():
    args = arguments()
    args["step"]["status"] = "unresolved"
    with pytest.raises(ValueError, match="resolved"):
        FieldCalibration.create(**args)


def test_joined_walk_requires_explicit_pool_and_preserves_pool_scope():
    from types import SimpleNamespace
    args = arguments()
    record = FieldCalibration.create(**args)
    block = dict(calibration_scope=args["scope"], field_read=args["read"], field_band=args["band"])
    walk = SimpleNamespace(block=dict(calibration_pools={"extra":block}))
    with pytest.raises(ValueError, match="pool name"):
        FieldCalibration.from_walk(walk, accuracy=args["accuracy"], step=args["step"])
    with pytest.raises(ValueError, match="no calibration provenance"):
        FieldCalibration.from_walk(walk, pool="intra", accuracy=args["accuracy"], step=args["step"])
    assert FieldCalibration.from_walk(walk, pool="extra", accuracy=args["accuracy"], step=args["step"]).record == record.record


def test_measured_read_charge_cannot_be_hidden_by_smaller_supplied_bound():
    args = arguments()
    args["read"]["term_pilot"] = .0002
    with pytest.raises(ValueError, match="below the measured"):
        FieldCalibration.create(**args)
