"""The field band is held to the numerical budget the walk is given (``field_target_floor``), not to the statistical
floor of its pilot scaled to the walk's walker count: the band is the first whose codec error is within the budget,
the walk records the budget, and a budget no band reaches is refused by name."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec import SpecError
from dmipy_sim import walk
from tests.test_bank import _env7, _susc_walk
from tests.test_grid_field_in_walk import D0, UM

BUDGET = 1e-4                                      # below the pack error of the band the pilot's statistical floor picks


def _pack_fidelity(w, **kw):
    return build_replay_pack(w, id="test/field-target", license="x", citation="x", envelope=_env7(), K=16, **kw).fidelity


@pytest.fixture(scope="module")
def default_walk():
    return _susc_walk(300, seed=2)


@pytest.fixture(scope="module")
def capped_walk():
    return _susc_walk(300, seed=2, field_target_floor=BUDGET)


def test_without_a_budget_the_band_is_held_to_the_scaled_split_half_floor(default_walk):
    band = default_walk.block["field_band"]
    assert default_walk.K_c3 == band["K"] == 16
    assert band["target_floor"] is None and "split-half" in band["floor_is"]
    assert band["floor"] > 100 * BUDGET and band["floor_scale"] == pytest.approx(np.sqrt(band["pilot_walkers"] / band["production_walkers"]))
    assert _pack_fidelity(default_walk)["err_susc_path"] > BUDGET                      # the band the budget is for


def test_the_band_meets_the_budget_whatever_the_walker_count(default_walk, capped_walk):
    band = capped_walk.block["field_band"]
    assert capped_walk.K_c3 == band["K"] > default_walk.K_c3
    assert band["target_floor"] == BUDGET and band["floor"] == BUDGET and "explicit" in band["floor_is"]
    assert band["within_floor"] and band["err"] <= BUDGET
    fid = _pack_fidelity(capped_walk, sigma_star=BUDGET, planned_floor=BUDGET)
    assert fid["err_susc_path"] <= BUDGET and fid["floor_susc_path"] == BUDGET


def test_a_budget_no_band_reaches_is_refused_by_name():
    with pytest.raises(ValueError, match="no field band/container meets the requested error budget"):
        _susc_walk(300, seed=2, field_target_floor=1e-30)


def test_the_budget_goes_with_a_derived_field_band():
    from tests.test_bank import _slab_grid
    kw = dict(require_gpu=False)
    geom = d.Box1D(6e-6)
    with pytest.raises(ValueError, match="goes with field_basis= and without K_c3"):
        _walk_geometry(8, D0, geom, 2e-3, 5e-4, field_target_floor=BUDGET, **kw)
    with pytest.raises(ValueError, match="goes with field_basis= and without K_c3"):
        _walk_geometry(8, D0, geom, 2e-3, 5e-4, field_basis=_slab_grid(), K_c3=4, field_target_floor=BUDGET, **kw)
    with pytest.raises(ValueError, match="positive error budget"):
        _walk_geometry(8, D0, geom, 2e-3, 5e-4, field_basis=_slab_grid(), field_target_floor=0.0, **kw)


def test_walk_spec_passes_the_budget_to_a_gridded_walk_and_refuses_it_elsewhere():
    geom = d.MyelinatedCylinder(1.0 * UM, 1.5 * UM, (0, 0, 1), compartments=d.Compartments(intra=d.Pool(D=D0), extra=d.Pool(D=D0)))
    w = walk(geom.spec, 300, 5e-3, seed=2, require_gpu=False, field_target_floor=1e-3)
    assert w.block["field_band"]["target_floor"] == 1e-3 and w.block["field_band"]["floor"] == 1e-3
    with pytest.raises(SpecError, match="K_c3 was given"):
        walk(geom.spec, 8, 5e-3, seed=2, require_gpu=False, K_c3=4, field_target_floor=1e-3)
    with pytest.raises(SpecError, match="gridded field"):
        walk(geom.spec, 8, 5e-3, seed=2, require_gpu=False, adaptive_steps=True, field_target_floor=1e-3)
