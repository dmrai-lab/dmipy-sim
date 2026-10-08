"""Fast tests of dmipy_sim.spectrum.strands: the 25 um edge only (one voxel, both policy walk lengths refused by
the halo rule -- dmipy-sim#697). The 250 um / 1 mm / 5 mm cases (a real far-grid build, minutes each) are
exercised in test_strands_slow.py."""
import json
import os

import numpy as np
import pytest

from dmipy_sim.spectrum import strands
from dmipy_sim.spectrum._tck_stream import StreamingTckWriter
from dmipy_sim.spectrum.common import verify_index
from dmipy_sim.spectrum.strands import PILOT_TARGET_FLOOR, _generate_column


@pytest.fixture(scope="module")
def entry_25um(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("spectrum_25um"))
    return strands(25e-6, 1, out_dir=out)


def test_25um_refuses_both_policy_walk_lengths(entry_25um):
    assert sorted(entry_25um["refused"]) == [0.02, 0.1]
    assert entry_25um["variants"] == []
    assert entry_25um["far_grid"] is None


def test_25um_pilot_hits_its_stated_floor(entry_25um):
    p = entry_25um["pilot"]
    assert p["n_pilot"] == int(np.ceil(1.0 / PILOT_TARGET_FLOOR ** 2))
    assert p["floor_pilot"] == pytest.approx(1.0 / np.sqrt(p["n_pilot"]))
    assert p["floor_pilot"] <= PILOT_TARGET_FLOOR + 1e-12


def test_25um_writes_the_expected_files(entry_25um):
    out = entry_25um["out_dir"]
    for rel in ("substrate/strands.tck", "substrate/strands_diameters.txt", "substrate/spec.json",
               "substrate/counts.npz", "substrate/blocks.json", "pilot/strands.tck", "pilot/strands_diameters.txt",
               "spectrum.json"):
        assert os.path.exists(os.path.join(out, rel)), rel
    # no manifest/plan dir: both walk lengths refused at this edge
    assert not os.path.exists(os.path.join(out, "T020ms"))
    assert not os.path.exists(os.path.join(out, "T100ms"))


def test_25um_index_hashes_verify(entry_25um):
    n = verify_index(entry_25um["index"])
    assert n >= 7


def test_25um_is_walkable_by_walk_spec(entry_25um):
    """The generated 25 um geometry itself (not the pilot) is a walkable dmipy_sim spec on the JAX engine, CPU:
    dmipy-sim#697's own test requirement."""
    from dmipy_sim.spec import disco_spec, walk_spec
    out = entry_25um["out_dir"]
    spec = disco_spec(os.path.join(out, "substrate/strands.tck"), os.path.join(out, "substrate/strands_diameters.txt"),
                      coordinate_unit_m=25e-6, diameter_unit_m=1e-3, side_m=25e-6, field=False)
    w = walk_spec(spec, 40, 2e-4, 2e-5, seed=0, require_gpu=False, n_probe=5_000)
    assert w.positions.shape[0] == 40
    assert np.isfinite(w.positions).all()


def test_25um_counts_sum_matches_the_blocks_table(entry_25um):
    """The per-voxel counts.npz sums (per pool) to what blocks.json's own per-block walker tally adds up to --
    the plan's own target, not a number recomputed independently."""
    out = entry_25um["out_dir"]
    counts = np.load(os.path.join(out, "substrate/counts.npz"))
    blocks = json.load(open(os.path.join(out, "substrate/blocks.json")))["blocks"]
    for pool in ("intra", "extra"):
        total = int(counts[f"count_{pool}"].sum())
        from_blocks = sum(b["walkers"][pool] for b in blocks)
        assert total == from_blocks


def test_generate_column_two_columns_differ_and_are_reproducible():
    """The tiling primitive two entries of a 5 mm (or any multi-tile) pack rely on: tile (0, 0) and tile (0, 1)
    draw unrelated, non-equal strand sets, and regenerating tile (0, 0) reproduces it exactly -- the property
    the slow end-to-end 5 mm test also checks, here at no cost."""
    stats = dict(mean=1.0e-6, std=0.3e-6, min=0.7e-6, max=2.0e-6)
    kw = dict(radii_stats=stats, density=12196e6, points_per_mm=53.2)
    col0a = list(_generate_column(1, 0, 0, 1e-3, 1e-3, **kw))
    col0b = list(_generate_column(1, 0, 0, 1e-3, 1e-3, **kw))
    col1 = list(_generate_column(1, 0, 1, 1e-3, 1e-3, **kw))
    assert len(col0a) == len(col0b)
    for (c0, r0), (c0b, r0b) in zip(col0a, col0b):
        assert np.array_equal(c0, c0b) and r0 == r0b
    assert len(col0a) != len(col1) or any(
        c0.shape != c1.shape or not np.array_equal(c0, c1) for (c0, _), (c1, _) in zip(col0a, col1))


def test_strands_refuses_a_bad_tiling_edge(tmp_path):
    with pytest.raises(ValueError):
        strands(1.5e-3, 1, out_dir=str(tmp_path))
