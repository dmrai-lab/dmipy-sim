"""Slow tests of dmipy_sim.spectrum.strands at 250 um, 1 mm and 5 mm -- dmipy-sim#697's own test requirement
("each generated edge is walkable by walk_spec", "two tiles of the 5 mm case differ"), run in the
nightly/offline `slow` lane (`tests/conftest.py::_SLOW_MC_MODULES`), never in a per-PR CI pass. Only the 250 um
case builds a real far grid here (a few minutes); 1 mm and 5 mm skip it (``T_max_s=()``) since the far-grid
build's own cost is already measured for real on the L40S and reported in #697's PR body, not something this
suite needs to re-pay on every nightly run."""
import json
import os

import numpy as np
import pytest

from dmipy_sim.spectrum import strands
from dmipy_sim.spectrum.common import verify_index


@pytest.fixture(scope="module")
def entry_250um(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("spectrum_250um"))
    return strands(250e-6, 1, out_dir=out)


def test_250um_both_walk_lengths_pass_the_halo_rule(entry_250um):
    assert sorted(entry_250um["refused"]) == []
    assert sorted(entry_250um["variants"]) == [0.02, 0.1]
    assert entry_250um["far_grid"] is not None


def test_250um_manifests_and_plans_exist(entry_250um):
    out = entry_250um["out_dir"]
    for sub in ("T020ms", "T100ms"):
        man = json.load(open(os.path.join(out, sub, "manifest.json")))
        assert man["substrate"]["kind"] == "disco"
        assert man["substrate"]["far_grid"] is not None
        assert os.path.exists(os.path.join(out, sub, "..", "substrate", "far.npy"))
        assert man["pack"]["K"] >= 2


def test_250um_index_verifies(entry_250um):
    assert verify_index(entry_250um["index"]) > 0


def test_250um_is_walkable_by_walk_spec(entry_250um):
    from dmipy_sim.spec import disco_spec, walk_spec
    out = entry_250um["out_dir"]
    spec = disco_spec(os.path.join(out, "substrate/strands.tck"), os.path.join(out, "substrate/strands_diameters.txt"),
                      coordinate_unit_m=25e-6, diameter_unit_m=1e-3, side_m=250e-6, field=False)
    w = walk_spec(spec, 60, 5e-4, 5e-5, seed=0, require_gpu=False, n_probe=10_000)
    assert np.isfinite(w.positions).all()


@pytest.mark.parametrize("edge_m", [1e-3])
def test_1mm_is_walkable_and_has_a_plan(edge_m, tmp_path):
    """``T_max_s=()`` skips the far-grid build (minutes; measured for real on the L40S in #697's PR body) --
    the counts/blocks/walkability checks below do not need it, since the per-voxel plan is finished before the
    far grid is ever touched and the walkability check here uses a field-off spec it builds itself."""
    out = str(tmp_path)
    r = strands(edge_m, 1, out_dir=out, T_max_s=())
    assert r["refused"] == []
    assert r["far_grid"] is None
    counts = np.load(os.path.join(r["out_dir"], "substrate/counts.npz"))
    blocks = json.load(open(os.path.join(r["out_dir"], "substrate/blocks.json")))["blocks"]
    for pool in ("intra", "extra"):
        assert int(counts[f"count_{pool}"].sum()) == sum(b["walkers"][pool] for b in blocks)
    from dmipy_sim.spec import disco_spec, walk_spec
    spec = disco_spec(os.path.join(r["out_dir"], "substrate/strands.tck"),
                      os.path.join(r["out_dir"], "substrate/strands_diameters.txt"),
                      coordinate_unit_m=25e-6, diameter_unit_m=1e-3, side_m=edge_m, field=False)
    w = walk_spec(spec, 60, 5e-4, 5e-5, seed=0, require_gpu=False, n_probe=10_000)
    assert np.isfinite(w.positions).all()


def test_5mm_two_tiles_differ():
    """dmipy-sim#697's own test: two tiles of the 5 mm (tiled) case differ. Checked directly on the tiling
    primitive at the spectrum's own statistics (the fast test_strands.py::
    test_generate_column_two_columns_differ_and_are_reproducible checks the same property at synthetic
    statistics); the full 5 mm generation itself is exercised for real -- sizes, times, peak RSS -- on the
    L40S and reported in #697's PR body, not re-run here (dozens of minutes even with #697's strand_frame
    fix)."""
    from dmipy_sim.spectrum.strands import _generate_column, DISCO_INNER_RADIUS_STATS_M, DISCO_AREAL_DENSITY_PER_M2
    from dmipy_sim.spectrum.disco_stats import DISCO_POINTS_PER_MM
    kw = dict(radii_stats=DISCO_INNER_RADIUS_STATS_M, density=DISCO_AREAL_DENSITY_PER_M2, points_per_mm=DISCO_POINTS_PER_MM)
    col0 = list(_generate_column(1, 0, 0, 1e-3, 5e-3, **kw))
    col1 = list(_generate_column(1, 0, 1, 1e-3, 5e-3, **kw))
    assert len(col0) != len(col1) or any(
        c0.shape != c1.shape or not np.array_equal(c0, c1) for (c0, _), (c1, _) in zip(col0, col1))
