"""A fixed-step field calibration records evidence and is reusable on the measured scope."""
import hashlib
import time

import numpy as np
import pytest
from safetensors.numpy import load_file

from dmipy_sim import measure_field_read_calibration, walk
from dmipy_sim.coefficient_walk import join_walks
from dmipy_sim.engine import core
from dmipy_sim.engine.accumulate import READ_CONSTANT, field_read_term
from dmipy_sim.engine.calibration import FieldReadCalibration, canonical, fixed_step_constants
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


def _hashes(w, path):
    out = {k: hashlib.sha256(data).hexdigest() for k, (_, _, data) in _walk_bytes(w, path).items()}
    out["block"] = hashlib.sha256(canonical(w.block).encode()).hexdigest()
    return out


# Recorded from the default walk on the base revision with time.time fixed at 1000 s.
# The JAX/JAXLIB build is selected by the recorded_bits fixture (0.10.2 / 0.10.2).
_BASELINE_FAMILY = "e74c68c7486e5194fea424efc88122e001a185a20b6925b53b1b9f79a1cef439"
_BASELINE = {
    "boundary_bands": "c2b4e7f01cdad7545689f0327b9db2f20d0371a42e42b71f014ea13023f0bb5b",
    "boundary_endpoint": "4d964515c4cafd4572b3707244d4766651c80a071c4838260a195bec4cdeaef7",
    "field_coeffs": "c9595d30efbc1885c15f5e82fbbe33777a5f1428e7ff2f3cc993d9c84611df13",
    "position_coeffs": "d21a361471fd25ff07a33fca8c9be65361f122835d642f7989c569398d1fdce1",
    "qv_boundary": "aaabb75945ca9f3206caefd7225c05e1e901bb27b609b7b8f1e6744ecc24852b",
    "qv_field": "f72fdd3a50cb92f6d46f608db6216303049808821eb25da37bf0b517023b45ed",
    "qv_field_increments": "67ed3b30aa8bb3087e8381e73104f284ad2890bf45fcb248c84ec28f967035b7",
    "qv_positions": "266cdebff7a95b7fe0adcc84ba9ff13c2d564614ba2c560015112d8aa7942b57",
    "compartment": "4a59c1136395d4124da5604dfd60e7d6739bf17eb1d80a6d40f84fc839f5f00e",
    "block": "ce1443b33ae6e40e0fce297b8b6187891ca6b19a8ebfc0cf3a8fa87a20cf69fa",
}


def test_recorded_bits_fixed_step_default(recorded_bits, monkeypatch, tmp_path):
    from tests.recorded_bits import FAMILIES

    if recorded_bits is not FAMILIES[_BASELINE_FAMILY]:
        pytest.skip(f"fixed-step field read baseline was recorded on CPU arithmetic family {_BASELINE_FAMILY}")
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    spec = _cell().spec
    assert _hashes(walk(spec, 24, **_settings(spec)), tmp_path / "default.walk") == _BASELINE


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
    "permeability_factor": "per walker",
    "object_nodes": {"rule": "per object"},
    "backend": {"name": "other"},
    "code": "different code",
}


@pytest.mark.parametrize("entry", _SCOPE_CHANGES)
def test_walk_refuses_each_scope_entry_before_seeding(measured, entry, monkeypatch):
    spec, kw, calibration = measured
    data = calibration.record
    assert data["scope"].keys() == _SCOPE_CHANGES.keys()
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
    constants, se = fixed_step_constants(check, 0.5, 2)
    assert constants == [1.25, READ_CONSTANT]
    assert se[0] == pytest.approx(0.375, rel=2 * np.finfo(float).eps)
    assert se[1] is None
    check[0, 0, 1, 0] = -1
    with pytest.raises(ValueError, match="nonnegative quadratic variation"):
        fixed_step_constants(check, 0.5, 2)


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
