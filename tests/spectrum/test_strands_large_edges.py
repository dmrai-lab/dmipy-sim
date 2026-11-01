"""Tests of dmipy_sim.spectrum.strands at the 1 mm and 5 mm edges -- dmipy-sim#697's own test requirement
("each generated edge is walkable by walk", "two tiles of the 5 mm case differ"). ``T_max_s=()`` skips the far-grid
build, whose cost is measured for real on a GPU host and reported in #697's PR body; the 25 um case in test_strands.py
covers the plan, the manifest and the index writer."""
import json
import os

import numpy as np
import pytest

from dmipy_sim.spectrum import strands


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
    from dmipy_sim.spec import disco_spec
    from dmipy_sim import walk
    spec = disco_spec(os.path.join(r["out_dir"], "substrate/strands.tck"),
                      os.path.join(r["out_dir"], "substrate/strands_diameters.txt"),
                      coordinate_unit_m=25e-6, diameter_unit_m=1e-3, side_m=edge_m, field=False)
    w = walk(spec, 60, 5e-4, 5e-5, seed=0, require_gpu=False, n_probe=10_000)
    assert np.isfinite(w.position_coeffs).all()


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
