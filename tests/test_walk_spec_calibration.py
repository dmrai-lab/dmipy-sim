"""The public producer requires scoped records for every diffusing pool."""
import pytest
from dmipy_sim.spec import walk_spec
from dmipy_sim.engine.calibration import FieldCalibration
from tests.test_calibration_record import arguments
from tests.test_field_read_interval import spec


def test_walk_spec_requires_a_complete_pool_mapping(spec, monkeypatch):
    import importlib
    module = importlib.import_module("dmipy_sim.spec.walk")
    record = FieldCalibration.create(**arguments())
    kwargs = dict(n_walkers=12, T_max=.001, adaptive_steps=True)
    with pytest.raises(TypeError, match="maps pool names"):
        walk_spec(spec, field_calibration=record, **kwargs)
    with pytest.raises(ValueError, match="exactly"):
        walk_spec(spec, field_calibration={"extra":record}, **kwargs)
    captured = {}
    marker = object()
    def bundle(*args, **kw):
        captured.update(kw)
        return marker
    monkeypatch.setattr(module, "_walk_bundle", bundle)
    records = {"extra":record, "intra":record}
    query = dict(identity="protocol-domain", metric="max_abs_complex_signal_error", target=.01, confidence=.99)
    assert walk_spec(spec, field_calibration=records, calibration_floor=.001, query_scope=query, **kwargs) is marker
    assert captured["field_calibration"] == records
    assert captured["query_scope"] == query and captured["calibration_floor"] == .001


def test_pool_bands_cannot_be_silently_changed_to_match_the_first_pool(spec):
    args = arguments()
    first = FieldCalibration.create(**args)
    args["band"]["K"] = 8
    other = FieldCalibration.create(**args)
    with pytest.raises(ValueError, match="common field band"):
        walk_spec(spec, n_walkers=12, T_max=.001, adaptive_steps=True,
                  field_calibration={"intra":first, "extra":other})
