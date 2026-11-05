"""``build_replay_pack(codec="bitshuffle-zlib", chunk_walkers=...)`` (dmipy-sim#904): a pack written with the codec reads back
through dmipy-replay as the fixed-width pack of the same walk, integer for integer and replay for replay; the codec and the
chunk size are in the metadata; the fixed-width container stays the default; the codec's parameters are stated or refused."""
import json

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_replay import read_rpk, sequences
from dmipy_replay.replay import codec, container
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.replay.bank import build_replay_pack
from tests._envelope import battery_reach, stated


@pytest.fixture(scope="module")
def walk():
    g = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6)
    return _walk_geometry(600, 2e-9, g, 20e-3, 2.5e-4, seed=7, require_gpu=False)


def _build(walk, path, **kw):
    return build_replay_pack(walk, id="test/codec", license="x", citation="x", K=16, out_path=str(path),
                             **stated(of=walk, G_max=battery_reach(walk.T_window)), **kw)


@pytest.fixture(scope="module")
def packs(walk, tmp_path_factory):
    d_ = tmp_path_factory.mktemp("codec")
    _build(walk, d_ / "fixed.rpk")
    _build(walk, d_ / "coded.rpk", codec="bitshuffle-zlib", chunk_walkers=256)
    return d_


def test_the_default_is_the_fixed_width_container(packs):
    _, hdr = container.read_safetensors(packs / "fixed.rpk")
    meta = json.loads(hdr["rpk"])
    assert not any("codec" in r or "chunk_walkers" in r for r in meta["compression"]["container"])


def test_the_codec_and_the_chunk_size_are_in_the_metadata_and_the_file_stores_the_chunks(packs):
    arrays, hdr = container.read_safetensors(packs / "coded.rpk")
    meta = json.loads(hdr["rpk"])
    ranges = meta["compression"]["container"]
    assert ranges and all(r["codec"] == "bitshuffle-zlib" and r["chunk_walkers"] == 256 for r in ranges)
    streams = [k for k in arrays if codec._STREAM.match(k)]
    assert streams and all(f"{k}_chunks" in arrays and arrays[f"{k}_chunks"].dtype == np.int64 for k in streams)
    assert not any(k.startswith("pos_x_b") and not codec.is_stored_range(k) for k in arrays)


def test_a_written_pack_reads_back_bit_identically_through_dmipy_replay(packs):
    fixed, coded = read_rpk(str(packs / "fixed.rpk")), read_rpk(str(packs / "coded.rpk"))
    seq = sequences.pgse([[1, 0, 0], [0, 0, 1]], 2e-3, 5e-3, gradient_strengths=[0.3, 0.3], TE=10e-3)
    np.testing.assert_array_equal(fixed.position_coeffs, coded.position_coeffs)
    np.testing.assert_array_equal(np.asarray(fixed.replay(seq)), np.asarray(coded.replay(seq)))
    np.testing.assert_array_equal(fixed.spin_weights, coded.spin_weights)
    assert fixed.meta["fidelity"] == coded.meta["fidelity"]
    for k, v in fixed.arrays.items():
        got = coded.arrays[k]
        if isinstance(got, codec.CodecRange):
            assert np.array_equal(got.fixed_width(), np.asarray(v)), k          # the container's own bytes, re-packed
        else:
            assert np.array_equal(np.asarray(got), np.asarray(v)), k


def test_the_returned_pack_is_the_one_on_disk(walk, tmp_path):
    pk = _build(walk, tmp_path / "p.rpk", codec="bitshuffle-zlib", chunk_walkers=128)
    back = read_rpk(str(tmp_path / "p.rpk"))
    assert any(isinstance(v, codec.CodecRange) for v in pk.arrays.values())
    np.testing.assert_array_equal(pk.position_coeffs, back.position_coeffs)


def test_the_codec_parameters_are_stated_or_refused(walk, tmp_path):
    with pytest.raises(ValueError, match="needs chunk_walkers"):
        _build(walk, tmp_path / "a.rpk", codec="bitshuffle-zlib")
    with pytest.raises(ValueError, match="belongs to a codec"):
        _build(walk, tmp_path / "b.rpk", chunk_walkers=256)
    with pytest.raises(ValueError, match="power of two"):
        _build(walk, tmp_path / "c.rpk", codec="bitshuffle-zlib", chunk_walkers=1000)
    with pytest.raises(ValueError, match="zstd"):
        _build(walk, tmp_path / "d.rpk", codec="zstd", chunk_walkers=256)
    with pytest.raises(ValueError, match="no integer band range"):
        _build(walk, tmp_path / "e.rpk", codec="bitshuffle-zlib", chunk_walkers=256, position_container=None,
               blt_container=None, field_container=None)
