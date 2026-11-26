"""A fixed-step field calibration records evidence and is reusable on the measured scope."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

from dmipy_sim import measure_field_read_calibration, walk
from dmipy_sim.coefficient_walk import join_walks
from dmipy_sim.engine import core
from dmipy_sim.engine.accumulate import READ_CONSTANT, field_read_term
from dmipy_sim.engine.calibration import FieldReadCalibration, canonical, fingerprint, fixed_step_constants
from dmipy_sim.replay.bank import build_replay_pack
from tests.conftest import spec_without_source
from tests.test_spec_field_closed_form import _cell
from tests._envelope import walk_envelope


def _settings(spec):
    return dict(T_max=5e-4, block_h=1.25e-4, envelope=walk_envelope(of=spec), seed=3,
                require_gpu=False, K=4, K_c3=4, field_read_interval=1.25e-4)


def _walk_bytes(w, path):
    """Every saved array, including event columns, with dtype and shape."""
    w.save(path)
    return {k: (a.dtype.str, a.shape, a.tobytes()) for k, a in load_file(str(path)).items()}


def test_recorded_bits_fixed_step_default(tmp_path):
    from tests.recorded_bits import (
        FIXED_STEP_BASELINE_ENV, FIXED_STEP_DEFAULT, fixed_step_default_hashes, in_recorded_bits_job, recorded_canary,
    )

    canary = recorded_canary()
    if canary in FIXED_STEP_DEFAULT:
        expected = FIXED_STEP_DEFAULT[canary]
    else:
        reason = f"CPU family {canary} lacks fixed_step_default; tools/record_recorded_bits.py prints its entry"
        if not in_recorded_bits_job():
            pytest.skip(reason)
        baseline = os.environ.get(FIXED_STEP_BASELINE_ENV)
        assert baseline and Path(baseline).is_file(), f"{reason}; missing the same-runner base revision reference"
        entries = json.loads(Path(baseline).read_text())
        assert canary in entries, f"the base revision reference has no entry for CPU family {canary}"
        expected = entries[canary]
    actual = fixed_step_default_hashes(tmp_path / "default.walk")
    assert actual.keys() == expected.keys(), f"fixed_step_default has different entries: {sorted(actual.keys() ^ expected.keys())}"
    for name, digest in expected.items():
        assert actual[name] == digest, f"fixed_step_default {name} differs for CPU family {canary}"


def test_handwritten_read_calibration_names_a_missing_entry(tmp_path):
    record = dict(kind="field-read-calibration", scope={"channels": ["test"], "checks": [], "n_read": 1},
                  rung=None, constants=[READ_CONSTANT], evidence={"standard_error": [None], "pilot_walkers": 2})
    record["digest"] = fingerprint(record)
    path = tmp_path / "calibration.json"
    path.write_text(canonical(record))
    with pytest.raises(ValueError, match=r"field-read calibration missing required entry 'seed'$"):
        FieldReadCalibration.load(path)


def test_measure_reuse_join_and_charge(x64, monkeypatch, tmp_path):
    spec = _cell().spec
    kw = dict(_settings(spec), accumulation_dtype="float64")
    calibration = measure_field_read_calibration(spec, pilot_walkers=24, **kw)
    default = walk(spec, 24, **kw)
    assert calibration.record["evidence"]["pilot_walkers"] == 24
    assert calibration.record["evidence"]["seed"] == kw["seed"]
    assert default.block["field_read"]["pilot_walkers"] == 24
    assert np.maximum(calibration.record["constants"], READ_CONSTANT).tolist() == default.block["field_read"]["constants"]
    calibration.save(tmp_path / "calibration.json")
    loaded = FieldReadCalibration.load(tmp_path / "calibration.json")
    assert loaded.record == calibration.record

    def unexpected(*args, **kwargs):
        pytest.fail("a walk with a stated calibration entered its field pilot")

    monkeypatch.setattr(core, "_jax_field_pilot", unexpected)
    monkeypatch.setattr(core, "field_pilot_on_backend", unexpected)
    kw["field_calibration"] = loaded
    whole = walk(spec, 24, **kw)
    first = walk(spec, 12, **kw)
    rest = walk(spec, 12, first_walker=12, **kw)
    repeat = walk(spec, 24, **kw)
    assert "field pilot" not in whole.run.summary["phases_s"]
    read = whole.block["field_read"]
    assert read["constants"] == calibration.record["constants"]
    assert read["calibration_digest"] == calibration.record["digest"]
    assert read["reused"] is True
    assert read.keys() == {
        "rule", "dt_f_rule", "dt_f", "n_read", "reads_per_block", "at_step", "B0", "chi", "floor", "T_e",
        "step_s", "steps_per_block", "constant", "constants", "constants_measured", "channels",
        "reused", "calibration_digest",
    }
    joined = join_walks([first, rest])
    assert canonical(joined.block) == canonical(whole.block) == canonical(repeat.block)
    arrays = _walk_bytes(whole, tmp_path / "whole.walk")
    assert _walk_bytes(joined, tmp_path / "joined.walk") == arrays
    assert _walk_bytes(default, tmp_path / "default.walk") == arrays

    pack = build_replay_pack(whole, id="fixed-step-field-read", license="x", citation="x", K=4, device="numpy",
                             position_container=None, blt_container=None, field_container=None)
    assert pack.meta["compression"]["channels"]["susc_field_basis"]["read"] == read
    at = pack.meta["replay_envelope"]["field_read"]["term_at"]
    assert at["floor"] == read["floor"]
    qv = np.asarray(whole.qv_field_increments, np.float64)[:, 0].mean(axis=0)
    charged = np.maximum(calibration.record["constants"], READ_CONSTANT)
    assert np.any((charged > READ_CONSTANT) & (qv > 0))
    expect = field_read_term(whole.field_read_interval, qv, at["B0"], at["chi"], charged)
    assert expect > field_read_term(whole.field_read_interval, qv, at["B0"], at["chi"])
    assert pack.fidelity["terms"]["field_read"] == expect


@pytest.fixture(scope="module")
def measured():
    spec = _cell().spec
    kw = _settings(spec)
    return spec, kw, measure_field_read_calibration(spec, pilot_walkers=8, **kw)


_SCOPE_CHANGES = {
    "spec": "different spec",
    "diffusivity": 1e-9,
    "dt_sim": 1e-6,
    "sub_steps": 200,
    "block_h": 2e-4,
    "n_blocks": 5,
    "K": 3,
    "block_order": 2,
    "window_T": 1e-3,
    "n_windows": 2,
    "K_c3": 3,
    "channels": ["other"] * 12,
    "field_basis": "different basis",
    "field_read_interval": 1e-5,
    "n_read": 178,
    "dt_f": 1e-5,
    "checks": [179],
    "B0": 3.0,
    "chi": 2e-6,
    "floor": 1e-4,
    "start_rule": {"rule": "different placement"},
    "record": False,
    "bounce_budget": 10,
    "backend": {"name": "other"},
    "code": "different code",
}


@pytest.mark.parametrize("entry", _SCOPE_CHANGES)
def test_walk_refuses_each_scope_entry_before_seeding(measured, entry, monkeypatch):
    spec, kw, calibration = measured
    data = calibration.record
    assert data["scope"].keys() == _SCOPE_CHANGES.keys() | {"permeability_factor", "object_nodes", "walker_rule"}
    assert data["scope"][entry] != _SCOPE_CHANGES[entry]
    data["scope"][entry] = _SCOPE_CHANGES[entry]
    altered = FieldReadCalibration.create(scope=data["scope"], rung=data["rung"], constants=data["constants"],
                                         **data["evidence"])

    def unexpected(*args, **kwargs):
        pytest.fail("an incompatible calibration reached walker seeding")

    monkeypatch.setattr(core, "seed_walkers", unexpected)
    with pytest.raises(ValueError, match=rf"incompatible field calibration scope: {entry}$"):
        walk(spec, 8, field_calibration=altered, **kw)


def test_walk_refuses_another_precision(measured, x64):
    spec, kw, calibration = measured
    with pytest.raises(ValueError, match="incompatible field calibration scope: backend"):
        walk(spec, 8, field_calibration=calibration, accumulation_dtype="float64", **kw)


@pytest.mark.parametrize("input_name", ["permeability_factor", "object_nodes"])
def test_per_walker_calibration_requires_matching_rule(input_name, monkeypatch):
    from dmipy_sim.engine.physics import ObjectNodes
    from tests.test_permeability_factor import _mixed

    spec = _mixed().spec
    kw = _settings(spec)
    per_walker = {
        "permeability_factor": np.full(8, 0.25, np.float32),
        "object_nodes": ObjectNodes(np.array([0.25, 1.0], np.float32), np.array([0.5, 0.5]), 7, np.ones(8, bool)),
    }
    inputs = {input_name: per_walker[input_name]}
    rule = "assignment-v1"
    calibration = measure_field_read_calibration(spec, pilot_walkers=8, walker_rule=rule, **inputs, **kw)

    def unexpected(*args, **kwargs):
        pytest.fail("an incompatible calibration reached walker seeding")

    with monkeypatch.context() as guard:
        guard.setattr(core, "seed_walkers", unexpected)
        with pytest.raises(ValueError, match="field_calibration requires walker_rule"):
            measure_field_read_calibration(spec, pilot_walkers=8, **inputs, **kw)
        with pytest.raises(ValueError, match="field_calibration requires walker_rule"):
            walk(spec, 8, field_calibration=calibration, **inputs, **kw)
        with pytest.raises(ValueError, match="incompatible field calibration scope: walker_rule$"):
            walk(spec, 8, field_calibration=calibration, walker_rule="assignment-v2", **inputs, **kw)
        with pytest.raises(ValueError, match=rf"incompatible field calibration scope: {input_name}$"):
            walk(spec, 8, field_calibration=calibration, walker_rule=rule, **kw)

    if input_name == "object_nodes":
        nodes = inputs[input_name]._replace(seed=9, walkers=np.arange(8) % 2 == 0)
        other = walk(spec, 8, field_calibration=calibration, walker_rule=rule, object_nodes=nodes, **kw)
        assert other.block["field_read"]["calibration_digest"] == calibration.record["digest"]


def test_permeability_rule_reuses_calibration_across_distinct_walker_ranges():
    from tests.test_permeability_factor import _mixed

    spec = _mixed().spec
    kw = _settings(spec)
    rule = {"assignment": "alternating ranges of eight", "factors": [0.25, 1.0]}
    first = np.full(8, 0.25, np.float32)
    calibration = measure_field_read_calibration(spec, pilot_walkers=8, walker_rule=rule, permeability_factor=first, **kw)
    same_rule = {"factors": [0.25, 1.0], "assignment": "alternating ranges of eight"}
    for start, factors in [(0, first), (8, np.ones(8, np.float32))]:
        piece = walk(spec, 8, first_walker=start, field_calibration=calibration, walker_rule=same_rule,
                     permeability_factor=factors, **kw)
        assert piece.block["field_read"]["calibration_digest"] == calibration.record["digest"]
        np.testing.assert_array_equal(piece.permeability_factor, factors)


@pytest.mark.parametrize("absent", ["K_c3", "field_read_interval", "field"])
def test_read_calibration_requires_a_field_band_and_interval(measured, absent):
    spec, settings, calibration = measured
    kw = settings.copy()
    if absent == "field":
        spec = spec_without_source(spec)
    else:
        kw.pop(absent)
    with pytest.raises(ValueError, match=f"field_calibration requires .*{absent}"):
        walk(spec, 8, field_calibration=calibration, **kw)
    with pytest.raises(ValueError, match=f"field_calibration requires .*{absent}"):
        measure_field_read_calibration(spec, pilot_walkers=8, **kw)


def test_standard_error_of_ratio_uses_the_walker_spread():
    # Two windows per walker, dt_f = 2 * 0.5 = 1 s. Walker means of squared error are 1, 9
    # and of QV are 2, 6. The ratio is 5/4. Residuals are -1.5, +1.5, so its SE is 1.5/4.
    # Channel 1 has zero variation and uses the trapezoid bound with no estimate.
    check = np.array([[[[1, 0], [1, 0]], [[1, 0], [3, 0]]],
                      [[[3, 0], [4, 0]], [[3, 0], [8, 0]]]])
    constants, se = fixed_step_constants(check, 0.5, 2, ["varying", "constant"])
    assert constants == [1.25, READ_CONSTANT]
    assert se[0] == pytest.approx(0.375, rel=2 * np.finfo(float).eps)
    assert se[1] is None
    check[0, 0, 1, 0] = -1
    with pytest.raises(ValueError, match="nonnegative quadratic variation"):
        fixed_step_constants(check, 0.5, 2, ["varying", "constant"])


def test_measurement_refuses_an_excursion_hidden_between_coarse_reads(monkeypatch):
    from dmipy_sim.fields.hollow_cylinder import HollowCylinderField

    cell = _cell()
    spec = cell.spec
    channels = HollowCylinderField.of(cell).channel_names
    # The fine trapezoid of (0, 1, 0) at unit spacing is 1. The coarse trapezoid and QV are 0.
    samples = np.array([0.0, 1.0, 0.0])
    fine_integral = (samples[:-1] + samples[1:]).sum() / 2
    coarse_integral = samples[0] + samples[-1]
    coarse_qv = (samples[-1] - samples[0]) ** 2
    with pytest.raises(ValueError, match="channel 'excursion' at rung 2 .*zero coarse quadratic variation"):
        check = np.tile([coarse_integral - fine_integral, coarse_qv], (2, 1)).reshape(2, 1, 2, 1)
        fixed_step_constants(check, 1.0, 2, ["excursion"])

    def pilot(call, indices, checks, dt_sim, keep_means):
        check = np.zeros((len(indices), 1, 2, len(checks), len(channels)))
        check[:, :, 0, :, 0] = (coarse_integral - fine_integral) * dt_sim
        return dict(chk=check)

    monkeypatch.setattr(core, "_jax_field_pilot", pilot)
    with pytest.raises(ValueError, match=rf"channel {channels[0]!r} at rung \d+ .*zero coarse quadratic variation"):
        measure_field_read_calibration(spec, pilot_walkers=2, **_settings(spec))


def test_backend_declares_precision_and_pilot_override_names_the_actual_engine(measured, monkeypatch):
    from tests.backends.test_backend_capabilities import _CellField

    spec, kw, jax_calibration = measured
    backend = _CellField(precision="float32")
    with pytest.raises(ValueError, match="backend 'cellfield' must declare its precision"):
        measure_field_read_calibration(spec, pilot_walkers=8, backend=backend, **kw)
    with pytest.raises(ValueError, match="backend 'cellfield' must declare its precision"):
        walk(spec, 8, backend=backend, field_calibration=jax_calibration, **kw)
    assert not backend.requests

    backend.precision = "float32"
    native = measure_field_read_calibration(spec, pilot_walkers=8, backend=backend, **kw)
    assert native.record["scope"]["backend"] == {"name": "cellfield", "precision": "float32", "build": None}
    assert len(backend.requests) == 1
    pilot, = backend.requests
    assert pilot.sub_steps == 1 and pilot.raw_subset == 8

    with pytest.raises(ValueError, match="incompatible field calibration scope: backend"):
        walk(spec, 8, field_calibration=native, **kw)

    got = walk(spec, 8, backend=backend, field_calibration=native, **kw)
    assert got.block["field_read"]["constants"] == native.record["constants"]
    assert len(backend.requests) == 2
    assert backend.requests[-1].K_c3 == kw["K_c3"]

    def jax_pilot(call, indices, checks, dt_sim, keep_means):
        return dict(chk=np.zeros((len(indices), 1, 2, len(checks), 12)))

    monkeypatch.setattr(core, "_jax_field_pilot", jax_pilot)
    override = measure_field_read_calibration(spec, pilot_walkers=8, backend=backend, field_pilot="jax", **kw)
    assert len(backend.requests) == 2
    assert override.record["scope"]["backend"] == jax_calibration.record["scope"]["backend"]
