"""Replay phantom (.rph) reader/writer: the invariants RPH.md states as MUSTs."""
import json
import numpy as np
import numpy.testing as npt
import pytest

from dmipy_sim.phantom import read_rph, write_rph, ReplayPhantom, SUBSTRATE_KINDS


def _minimal(tmp_path, fracs, embed=None, rpk=None):
    subs = [{"id": "wm/pack", "kind": "pack", "m0": 1.0},
            {"id": "csf/free-water", "kind": "analytic", "m0": 1.0,
             "model": "free_water", "params": {"diffusivity": 3e-9}},
            {"id": "background/inert", "kind": "inert", "m0": 0.0}]
    n = len(fracs)
    p = tmp_path / "t.rph"
    write_rph(p, voxel_index=np.zeros((n, 3), np.int32),
              substrate_id=np.tile([0, 1, 2], (n, 1)),
              geometric_fraction=np.asarray(fracs, np.float32),
              odf_sh=np.zeros((n, 3, 15), np.float32),
              substrates=subs, grid={"shape": [n, 1, 1], "voxel_size_m": [1e-3] * 3},
              lmax=4, id="test/phantom", license="CC0-1.0", citation="test",
              embed_packs=({0: rpk} if embed else None))
    return p


def test_fractions_must_sum_to_one(tmp_path):
    """A voxel is always full: an unmodelled remainder is a substrate, not an absence.
    The writer checks rather than trusting, because a phantom that quietly sums to 0.9 makes
    every signal 10% low with nothing to point at."""
    _minimal(tmp_path, [[0.5, 0.3, 0.2]])                      # fine
    with pytest.raises(ValueError, match="summing to"):
        _minimal(tmp_path, [[0.5, 0.3, 0.1]])                  # 0.9 -- refused
    with pytest.raises(ValueError, match="summing to"):
        _minimal(tmp_path, [[0.6, 0.3, 0.2]])                  # 1.1 -- refused


def test_roundtrip_and_structure(tmp_path):
    p = _minimal(tmp_path, [[0.5, 0.3, 0.2], [1.0, 0.0, 0.0]])
    ph = read_rph(p)
    assert isinstance(ph, ReplayPhantom) and ph.n_voxels == 2 and ph.lmax == 4
    assert [s["kind"] for s in ph.substrates] == ["pack", "analytic", "inert"]
    npt.assert_allclose(ph.geometric_fraction.sum(1), 1.0, atol=1e-6)
    assert "ReplayPhantom" in repr(ph) and "test/phantom" in repr(ph)


def test_pack_accessor_refuses_what_it_cannot_give(tmp_path):
    """Asking for walkers that are not there must say which of the three reasons applies."""
    ph = read_rph(_minimal(tmp_path, [[0.5, 0.3, 0.2]]))
    with pytest.raises(ValueError, match="not embedded|not a pack|uri"):
        ph.pack(0)                       # a pack, but referenced not embedded
    with pytest.raises(ValueError, match="not a pack"):
        ph.pack(1)                       # analytic: a closed form, no walkers
    with pytest.raises(ValueError, match="not a pack"):
        ph.pack(2)                       # inert: emits nothing


def test_unknown_substrate_kind_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not in"):
        write_rph(tmp_path / "x.rph", voxel_index=np.zeros((1, 3), np.int32),
                  substrate_id=np.zeros((1, 1), np.int16),
                  geometric_fraction=np.ones((1, 1), np.float32),
                  odf_sh=np.zeros((1, 1, 15), np.float32),
                  substrates=[{"id": "?", "kind": "air", "m0": 1.0}],
                  grid={"shape": [1, 1, 1]}, lmax=4, id="t", license="CC0-1.0", citation="t")


def test_tiers_are_the_intersection_over_packs(tmp_path):
    """A phantom adds no capability: a channel missing from any cited pack is missing from the
    phantom, so `tiers` intersects and `require` refuses rather than silently dropping it."""
    p = _minimal(tmp_path, [[0.5, 0.3, 0.2]])
    ph = read_rph(p)
    ph.substrates[0].update(embedded=True, pack_meta={
        "replay_envelope": {"gradient": True, "bulk_relaxation": True, "field": False}})
    assert ph.tiers() == {"gradient", "bulk_relaxation"}
    ph.require("gradient", "bulk_relaxation")
    with pytest.raises(ValueError, match="cannot serve"):
        ph.require("field")
