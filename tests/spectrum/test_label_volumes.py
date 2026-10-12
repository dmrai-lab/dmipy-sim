"""Fast tests of dmipy_sim.spectrum.label_volumes: the 25 um edge (both policy walk lengths refused). 250 um /
1 mm / 5 mm are exercised in test_label_volumes_slow.py."""
import json
import os

import numpy as np
import pytest

from dmipy_sim.spectrum.common import verify_index
from dmipy_sim.spectrum.label_volumes import TARGET_FRACTION, label_volume


@pytest.fixture(scope="module")
def entry_25um(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("spectrum_lv25"))
    return label_volume(25e-6, 1, out_dir=out)


def test_25um_refuses_both_policy_walk_lengths(entry_25um):
    assert sorted(entry_25um["refused"]) == [0.02, 0.1]
    assert entry_25um["variants"] == []


def test_25um_achieves_close_to_the_target_fraction(entry_25um):
    assert entry_25um["achieved_fraction_mean"] == pytest.approx(TARGET_FRACTION, abs=0.1)


def test_25um_writes_the_expected_files_and_index_verifies(entry_25um):
    out = entry_25um["out_dir"]
    for rel in ("substrate/spec.json", "substrate/labels.nrrd", "substrate/counts.npz", "substrate/blocks.json",
               "spectrum.json"):
        assert os.path.exists(os.path.join(out, rel)), rel
    assert verify_index(entry_25um["index"]) >= 4


def test_25um_counts_sum_matches_the_blocks_table(entry_25um):
    out = entry_25um["out_dir"]
    counts = np.load(os.path.join(out, "substrate/counts.npz"))
    blocks = json.load(open(os.path.join(out, "substrate/blocks.json")))["blocks"]
    assert int(counts["count_free"].sum()) == sum(b["walkers"]["free"] for b in blocks)


def test_25um_is_walkable(entry_25um):
    """dmipy-sim#697's own test requirement: the generated geometry is walkable on the JAX engine, CPU."""
    from dmipy_sim.spec import geometry_from_spec, load_spec
    from dmipy_sim.engine.core import simulate_trajectories
    spec = load_spec(os.path.join(entry_25um["out_dir"], "substrate/spec.json"))
    geom = geometry_from_spec(spec)
    w = simulate_trajectories(20, 2.0e-9, geom, T_max=5e-5, dt_save=5e-6, seed=0)
    assert np.isfinite(w.positions).all()
