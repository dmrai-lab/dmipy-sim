"""Fast unit tests of the substrate spectrum's shared machinery (dmipy_sim.spectrum.common)."""
import json
import os

import numpy as np
import pytest

from dmipy_sim.spectrum.common import (EDGES_M, HALO_FREE_DIFFUSIVITY_M2_S, SpectrumHaloError, WALK_LENGTHS_S,
                                       check_halo, derive_seed, halo_m, n_tiles_per_axis, sha256_of, tile_seed,
                                       verify_index, write_index)


def test_halo_matches_the_issue_text_numbers():
    # dmipy-sim#697 / tessera#43: halos of 72 and 160 um at 20 and 100 ms, at the common 2.0e-9 m^2/s policy D.
    assert HALO_FREE_DIFFUSIVITY_M2_S == 2.0e-9
    assert halo_m(0.020) == pytest.approx(71.55e-6, rel=5e-3)
    assert halo_m(0.100) == pytest.approx(160.0e-6, rel=5e-3)


def test_check_halo_refuses_25um_at_both_walk_lengths():
    for T in WALK_LENGTHS_S:
        with pytest.raises(SpectrumHaloError, match="25.0 um"):
            check_halo(25e-6, T)


def test_check_halo_passes_the_three_larger_edges():
    for edge in EDGES_M[1:]:
        for T in WALK_LENGTHS_S:
            h = check_halo(edge, T)
            assert 0 < h <= edge


def test_n_tiles_per_axis():
    assert n_tiles_per_axis(25e-6) == 1
    assert n_tiles_per_axis(250e-6) == 1
    assert n_tiles_per_axis(1e-3) == 1
    assert n_tiles_per_axis(5e-3) == 5
    with pytest.raises(ValueError):
        n_tiles_per_axis(1.5e-3)


def test_derive_seed_and_tile_seed_are_deterministic_and_distinct():
    a = tile_seed(1, 0, 0)
    b = tile_seed(1, 0, 1)
    c = tile_seed(1, 0, 0)
    assert a == c
    assert a != b
    assert derive_seed(1, "pilot") != derive_seed(2, "pilot")
    assert derive_seed(1, "pilot") == derive_seed(1, "pilot")


def test_write_index_and_verify_round_trip(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    sub = tmp_path / "sub"; sub.mkdir()
    (sub / "b.bin").write_bytes(bytes(range(256)))
    idx = write_index(str(tmp_path), extra=dict(id="t"))
    assert os.path.basename(idx) == "spectrum.json"
    d = json.load(open(idx))
    assert set(d["files"]) == {"a.txt", "sub/b.bin"}
    assert d["files"]["a.txt"]["sha256"] == sha256_of(str(tmp_path / "a.txt"))
    assert verify_index(idx) == 2


def test_verify_index_catches_a_tampered_file(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    idx = write_index(str(tmp_path))
    with open(tmp_path / "a.txt", "w") as fh:
        fh.write("tampered")
    with pytest.raises(ValueError, match="hashes to"):
        verify_index(idx)
