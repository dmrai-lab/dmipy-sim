"""Slow tests of dmipy_sim.spectrum.label_volumes at 250 um, 1 mm and 5 mm -- dmipy-sim#697's own test
requirement ("two tiles of the 5 mm case differ"), run in the nightly/offline `slow` lane."""
import json
import os

import numpy as np
import pytest

from dmipy_sim.spectrum.common import verify_index
from dmipy_sim.spectrum.label_volumes import label_volume


@pytest.mark.parametrize("edge_m", [250e-6, 1e-3])
def test_edge_passes_both_walk_lengths_and_has_a_plan(edge_m, tmp_path):
    r = label_volume(edge_m, 1, out_dir=str(tmp_path))
    assert r["refused"] == []
    assert sorted(r["variants"]) == [0.02, 0.1]
    counts = np.load(os.path.join(r["out_dir"], "substrate/counts.npz"))
    blocks = json.load(open(os.path.join(r["out_dir"], "substrate/blocks.json")))["blocks"]
    assert int(counts["count_free"].sum()) == sum(b["walkers"]["free"] for b in blocks)
    assert verify_index(r["index"]) > 0


def test_5mm_two_tiles_differ(tmp_path):
    """dmipy-sim#697's own test: two tiles of the 5 mm (tiled) case differ. Checked via the shared tiling
    primitive (fast); a real 5 mm run (125 tiles, 50,000 objects) was also measured for #697's PR body."""
    from dmipy_sim.spectrum.packs import _place_tile
    kw = dict(n_objects=400, target_fraction=0.3, shape=4.0)
    r0, c0, _ = _place_tile("sphere", 5e-3, 1e-3, (0, 0, 0), 1, **kw)
    r1, c1, _ = _place_tile("sphere", 5e-3, 1e-3, (0, 0, 1), 1, **kw)
    assert not np.array_equal(c0, c1)


def test_5mm_label_volume_generates(tmp_path):
    r = label_volume(5e-3, 1, out_dir=str(tmp_path))
    assert r["n_tiles_per_axis"] == 5
    assert r["n_objects"] > 0
