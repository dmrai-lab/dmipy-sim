"""Slow tests of dmipy_sim.spectrum.packs at 250 um, 1 mm and 5 mm -- dmipy-sim#697's own test requirement
("two tiles of the 5 mm case differ"), run in the nightly/offline `slow` lane
(`tests/conftest.py::_SLOW_MC_MODULES`)."""
import json
import os

import numpy as np
import pytest

from dmipy_sim.spectrum.common import verify_index
from dmipy_sim.spectrum.packs import cylinders, spheres


@pytest.mark.parametrize("edge_m", [250e-6, 1e-3])
@pytest.mark.parametrize("fn", [cylinders, spheres])
def test_edge_passes_both_walk_lengths_and_has_a_plan(fn, edge_m, tmp_path):
    out = str(tmp_path / f"{fn.__name__}_{edge_m}")
    r = fn(edge_m, 1, out_dir=out)
    assert r["refused"] == []
    assert sorted(r["variants"]) == [0.02, 0.1]
    counts = np.load(os.path.join(r["out_dir"], "substrate/counts.npz"))
    blocks = json.load(open(os.path.join(r["out_dir"], "substrate/blocks.json")))["blocks"]
    for pool in ("intra", "extra"):
        assert int(counts[f"count_{pool}"].sum()) == sum(b["walkers"][pool] for b in blocks)
    assert verify_index(r["index"]) > 0


def test_5mm_cylinders_two_tiles_differ(tmp_path):
    """dmipy-sim#697's own test: two tiles of the 5 mm (tiled) case differ. Cylinders only here (fast, 2-D
    tiling); spheres' own tiling is covered by the fast tier's test_place_tile_two_columns_differ... and
    (separately, on the L40S) a real 5 mm run measured for #697's PR body."""
    from dmipy_sim.spectrum.packs import _place_tile
    kw = dict(n_objects=400, target_fraction=0.5, shape=4.0)
    r0, c0, _ = _place_tile("cylinder", 5e-3, 1e-3, (0, 0), 1, **kw)
    r1, c1, _ = _place_tile("cylinder", 5e-3, 1e-3, (0, 1), 1, **kw)
    assert not np.array_equal(c0, c1)
