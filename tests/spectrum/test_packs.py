"""Fast tests of dmipy_sim.spectrum.packs: the 25 um edge only (both policy walk lengths refused by the halo
rule, as for every class in the spectrum). 250 um / 1 mm / 5 mm are exercised in test_packs_slow.py."""
import json
import os

import numpy as np
import pytest

from dmipy_sim.spectrum.common import verify_index
from dmipy_sim.spectrum.packs import TARGET_FRACTION, _gamma_scale_for, _place_tile, cylinders, spheres


@pytest.fixture(scope="module")
def entry_cyl_25um(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("spectrum_cyl25"))
    return cylinders(25e-6, 1, out_dir=out)


@pytest.fixture(scope="module")
def entry_sph_25um(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("spectrum_sph25"))
    return spheres(25e-6, 1, out_dir=out)


@pytest.mark.parametrize("entry_name", ["entry_cyl_25um", "entry_sph_25um"])
def test_25um_refuses_both_policy_walk_lengths(entry_name, request):
    entry = request.getfixturevalue(entry_name)
    assert sorted(entry["refused"]) == [0.02, 0.1]
    assert entry["variants"] == []


@pytest.mark.parametrize("entry_name", ["entry_cyl_25um", "entry_sph_25um"])
def test_25um_achieves_close_to_the_target_fraction(entry_name, request):
    entry = request.getfixturevalue(entry_name)
    kind = "cylinder" if "cyl" in entry_name else "sphere"
    assert entry["achieved_fraction_mean"] == pytest.approx(TARGET_FRACTION[kind], abs=0.1)


@pytest.mark.parametrize("entry_name", ["entry_cyl_25um", "entry_sph_25um"])
def test_25um_writes_the_expected_files_and_index_verifies(entry_name, request):
    entry = request.getfixturevalue(entry_name)
    out = entry["out_dir"]
    for rel in ("substrate/spec.json", "substrate/counts.npz", "substrate/blocks.json", "spectrum.json"):
        assert os.path.exists(os.path.join(out, rel)), rel
    assert not os.path.exists(os.path.join(out, "T020ms"))
    assert not os.path.exists(os.path.join(out, "T100ms"))
    assert verify_index(entry["index"]) >= 3


def test_25um_counts_sum_matches_the_blocks_table(entry_cyl_25um):
    out = entry_cyl_25um["out_dir"]
    counts = np.load(os.path.join(out, "substrate/counts.npz"))
    blocks = json.load(open(os.path.join(out, "substrate/blocks.json")))["blocks"]
    for pool in ("intra", "extra"):
        total = int(counts[f"count_{pool}"].sum())
        from_blocks = sum(b["walkers"][pool] for b in blocks)
        assert total == from_blocks


def test_25um_is_walkable(entry_cyl_25um):
    """dmipy-sim#697's own test requirement: the generated geometry is walkable on the JAX engine, CPU."""
    from dmipy_sim.spec import geometry_from_spec, load_spec
    from dmipy_sim.engine.core import simulate_trajectories
    spec = load_spec(os.path.join(entry_cyl_25um["out_dir"], "substrate/spec.json"))
    geom = geometry_from_spec(spec)
    w = simulate_trajectories(20, 2.0e-9, geom, T_max=5e-5, dt_save=5e-6, seed=0)
    assert np.isfinite(w.positions).all()


def test_place_tile_two_columns_differ_and_are_reproducible():
    """The tiling primitive two entries of a 5 mm (multi-tile) pack rely on: tile (0, 0) and tile (0, 1) draw
    unrelated, non-equal object sets, and regenerating tile (0, 0) reproduces it exactly."""
    kw = dict(n_objects=50, target_fraction=0.5, shape=4.0)
    r0a, c0a, _ = _place_tile("cylinder", 1e-3, 1e-3, (0, 0), 1, **kw)
    r0b, c0b, _ = _place_tile("cylinder", 1e-3, 1e-3, (0, 0), 1, **kw)
    r1, c1, _ = _place_tile("cylinder", 1e-3, 1e-3, (0, 1), 1, **kw)
    assert np.array_equal(r0a, r0b) and np.array_equal(c0a, c0b)
    assert not np.array_equal(c0a, c1)


def test_gamma_scale_for_gives_the_right_fraction_in_expectation():
    """The solved scale, at many objects (law of large numbers), reproduces the stated target fraction to a
    few percent -- both kinds."""
    rng = np.random.default_rng(0)
    for kind, dim in (("cylinder", 2), ("sphere", 3)):
        n = 20_000
        scale = _gamma_scale_for(kind, 1e-3, n, 0.3, shape=4.0)
        radii = rng.gamma(4.0, scale, n)
        measure = (np.pi * radii ** 2).sum() if dim == 2 else ((4.0 / 3.0) * np.pi * radii ** 3).sum()
        frac = measure / (1e-3 ** dim)
        assert frac == pytest.approx(0.3, rel=0.1)
