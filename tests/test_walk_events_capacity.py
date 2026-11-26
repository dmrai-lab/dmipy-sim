"""The public walk's events buffer and its record on a walk made in pieces."""

from dataclasses import fields
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from dmipy_replay import envelope_of
from dmipy_replay.replay.events import EventList
from dmipy_replay.spec.substrate import SpecError, SubstrateSpec, load_spec
from dmipy_sim import PackedCylinders, walk
from dmipy_sim.coefficient_walk import join_walks


@pytest.fixture
def cell_walk():
    def run(n_walkers=32, *, permeability=4e-3, **kwargs):
        spec = PackedCylinders([1e-6], [[2e-6, 2e-6]], 4e-6, permeability=permeability).spec
        return walk(spec, n_walkers, 4e-3, 2.5e-4, envelope=envelope_of("prisma"),
                    diffusivity=2e-9, seed=3, K=8, require_gpu=False, **kwargs)

    return run


def _assert_array_bytes(actual, expected, name):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, name
    assert actual.tobytes() == expected.tobytes(), name


def test_public_walk_pieces_join_to_the_bytes_of_the_whole_walk(cell_walk, x64):
    # Exercise a NumPy integer on the whole walk and a Python integer on the pieces.
    options = dict(accumulation_dtype="float64")
    whole = cell_walk(events_capacity=np.int64(64), raw_subset=4, **options)
    assert whole.comp_events.n_events > 0
    assert np.diff(whole.comp_events.offsets).max() > 1
    assert whole.block["events"] == {"rule": "given", "capacity": 64}
    assert type(whole.block["events"]["capacity"]) is int

    first = cell_walk(16, events_capacity=64, raw_subset=4, **options)
    second = cell_walk(16, events_capacity=64, first_walker=16, **options)
    joined = join_walks([first, second])
    for field in fields(whole):
        expected = getattr(whole, field.name)
        actual = getattr(joined, field.name)
        if isinstance(expected, EventList):
            for name in ("offsets", "t", "label"):
                _assert_array_bytes(getattr(actual, name), getattr(expected, name), f"{field.name}.{name}")
        elif isinstance(expected, np.ndarray):
            _assert_array_bytes(actual, expected, field.name)
        elif expected is None:
            assert actual is None, field.name
    assert joined.block == whole.block


def test_public_walk_refuses_an_events_buffer_overflow(cell_walk, x64):
    with pytest.raises(ValueError, match=r"a walker recorded (?:[2-9]|[1-9][0-9]+) pool crossings where the events "
                                         r"buffer holds 1: the walk is refused rather than an event dropped.*"
                                         r"Walk again with events_capacity="):
        cell_walk(events_capacity=1, accumulation_dtype="float64")


@pytest.mark.parametrize("capacity", [0, -1, True, np.bool_(True), 1.0, "1", np.timedelta64(1, "s")],
                         ids=["zero", "negative", "bool", "numpy_bool", "float", "string", "timedelta"])
def test_public_walk_refuses_invalid_events_capacity(cell_walk, capacity):
    with pytest.raises(ValueError, match="events_capacity must be a positive integer"):
        cell_walk(events_capacity=capacity)


def test_omitting_events_capacity_leaves_the_engine_default(cell_walk, monkeypatch):
    engine = Mock(side_effect=RuntimeError("engine reached"))
    monkeypatch.setattr("dmipy_sim.engine.core._walk_geometry", engine)
    with pytest.raises(RuntimeError, match="engine reached"):
        cell_walk()
    assert engine.call_args.kwargs.get("events_capacity") is None


@pytest.mark.parametrize("permeability, tiers", [(None, "all"), (4e-3, ())], ids=["sealed", "positions_only"])
def test_jax_walk_without_event_recording_omits_events_record(cell_walk, permeability, tiers):
    result = cell_walk(2, permeability=permeability, tiers=tiers, events_capacity=1)
    assert result.comp_events is None and result.bound_events is None
    assert "events" not in result.block


def test_bundle_refuses_capacity_and_accepts_explicit_none(monkeypatch):
    spec = load_spec(Path(__file__).parent / "fixtures" / "substrates" / "mesh_bundle.sub.json")
    envelope = envelope_of("prisma", chi_max=2e-6, B0_list=(3.0,))
    bundle = Mock(return_value=object())
    monkeypatch.setattr("dmipy_sim.spec.walk._walk_bundle", bundle)

    with pytest.raises(SpecError, match="events_capacity.*walked pool by pool"):
        walk(spec, 2, 4e-3, envelope=envelope, events_capacity=64)
    bundle.assert_not_called()
    assert walk(spec, 2, 4e-3, envelope=envelope, events_capacity=None) is bundle.return_value
    assert "events_capacity" not in bundle.call_args.kwargs["walk_kw"]

    bundle.reset_mock()
    permeable = spec.to_dict()
    permeable["walls"][0]["permeability"]["in_to_out"] = 1e-5
    permeable["validity"]["tiers"].append("exchange")
    with pytest.raises(SpecError, match="permeable multi-surface walk is not implemented"):
        walk(SubstrateSpec.from_dict(permeable), 2, 4e-3, envelope=envelope, events_capacity=64)
    bundle.assert_not_called()
