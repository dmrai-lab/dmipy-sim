"""A pack read a walker range at a time (dmipy-sim#785): a range's tensors decode to the whole pack's rows of those
walkers in either precision, a pack's shape and starts are read without decoding a band, a written pack is mapped
copy-on-write and replaced rather than rewritten, and a merge streamed to its file is the merge held in memory."""
from __future__ import annotations

import numpy as np
import pytest

import dmipy_sim.replay.compression as cx
from dmipy_sim.replay.replay import ReplayPack, read_rpk, write_rpk

CONTAINER = [{"bands": [0, 2], "bits": 12}, {"bands": [2, 5], "bits": 8}, {"bands": [5, 9], "bits": 7}, {"bands": [9, 12], "bits": 4}]


def _coefficients(n_w, K, seed):
    rng = np.random.default_rng(seed)
    C = np.empty((n_w, K + 2, 3))
    C[:, 0, :] = 5e-6 + rng.normal(0, 2e-6, (n_w, 3))
    C[:, 1, :] = rng.normal(0, 3e-7, (n_w, 3))
    C[:, 2:, :] = rng.normal(0, 1.0, (n_w, K, 3)) * 1e-7 / np.arange(1, K + 1)[None, :, None]
    return C


def _merged_arrays(K=12):
    """The position tensors of two shards of 37 and 26 walkers in the banded container, merged: their scale tables
    stacked on the block axis and every walker's ``band_block``, plus a static label and an events stream."""
    shards = [cx.pack_position_arrays(_coefficients(n, K, s), np.float32, container=CONTAINER) for n, s in ((37, 1), (26, 2))]
    arrays = {k: np.concatenate([a[k] for a in shards]) for k in shards[0] if k != "pos_band_scale"}
    arrays["pos_band_scale"] = np.concatenate([a["pos_band_scale"] for a in shards])
    arrays["band_block"] = np.r_[np.zeros(37), np.ones(26)].astype(np.uint16)
    rng = np.random.default_rng(3)
    counts = rng.integers(1, 4, 63)
    arrays["comp_events_offsets"] = np.r_[0, np.cumsum(counts)].astype(np.int64)
    arrays["comp_events_t"] = rng.uniform(0, 1e-3, int(counts.sum()))
    arrays["comp_events_label"] = rng.integers(0, 3, int(counts.sum())).astype(np.int8)
    arrays["comp_static"] = rng.integers(0, 2, 63).astype(np.int8)
    return arrays


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_walker_ranges_decode_to_the_whole_packs_rows(dtype):
    """Every walker range of a merged banded pack decodes, in float64 (the reference) and float32 (production), to
    exactly the whole pack's rows of those walkers; its shape and starts are the stored ones; its events stream is
    those walkers' events with the offsets rebased; the scale tables stay whole."""
    arrays = _merged_arrays()
    whole = cx.read_position_coeffs(arrays, dtype=dtype)
    assert whole.dtype == dtype and cx.position_shape(arrays) == whole.shape == (63, 14, 3)
    np.testing.assert_array_equal(cx.position_starts(arrays), cx.read_position_coeffs(arrays, dtype=np.float64)[:, 0, :])
    col = {"name": "comp", "kind": "events"}
    start_w, ev_w = cx.occupancy_events(arrays, col)
    for step in (1, 5, 37, 40, 63, 100):
        parts = [cx.walker_rows(arrays, a, a + step) for a in range(0, 63, step)]
        np.testing.assert_array_equal(np.concatenate([cx.read_position_coeffs(p, dtype=dtype) for p in parts]), whole)
        assert all(p["pos_band_scale"] is arrays["pos_band_scale"] for p in parts)
        evs = [cx.occupancy_events(p, col) for p in parts]
        np.testing.assert_array_equal(np.concatenate([s for s, _ in evs]), start_w)
        np.testing.assert_array_equal(np.concatenate([e.t for _, e in evs]), ev_w.t)
        np.testing.assert_array_equal(np.concatenate([np.diff(e.offsets) for _, e in evs]), np.diff(ev_w.offsets))
        np.testing.assert_array_equal(np.concatenate([p["comp_static"] for p in parts]), arrays["comp_static"])
    with pytest.raises(ValueError, match="neither walker-leading"):
        cx.walker_rows(dict(arrays, stray=np.zeros(5)), 0, 3)


def test_a_packs_shape_and_starts_are_read_without_decoding(monkeypatch, tmp_path):
    """``n_walkers``, ``n_coeffs``, ``K``, ``r0`` and ``spin_weights`` of a pack read the stored shapes and endpoints:
    with the band decoder disabled they still answer, and a walker range of the pack is a pack of those walkers."""
    from dmipy_sim.replay.bank import build_replay_pack
    import dmipy_sim as d
    w = d.simulate_trajectories(40, 2e-9, d.Cylinder(3e-6, (0, 0, 1)), 2e-3, seed=3, K=8, require_gpu=False)
    pk = build_replay_pack(w, id="t/shape", license="x", citation="x", K=8, out_path=str(tmp_path / "p.rpk"))
    pk = read_rpk(str(tmp_path / "p.rpk"))
    r0 = cx.read_position_coeffs(pk.arrays)[:, 0, :]

    def refuse(*a, **k):
        raise AssertionError("a shape was read by decoding the bands")
    monkeypatch.setattr(cx, "read_position_coeffs", refuse)
    assert (pk.n_walkers, pk.n_coeffs, pk.K) == (40, 10, 8)
    np.testing.assert_array_equal(pk.r0, r0)
    assert len(pk.spin_weights) == 40
    sub = pk.walkers(10, 25)
    assert sub.n_walkers == 15 == sub.meta["walk_params"]["n_walkers"]
    np.testing.assert_array_equal(sub.r0, r0[10:25])


def test_a_pack_is_mapped_copy_on_write_and_replaced_not_rewritten(tmp_path):
    """``write_rpk`` writes a tensor given as pieces as their concatenation and replaces the file whole; ``read_rpk``
    maps it copy-on-write: writing into a read tensor leaves the file as it was, and writing the path again while it
    is mapped leaves the mapped pack reading the file it was mapped from."""
    from safetensors import safe_open
    a = np.arange(12, dtype=np.float32).reshape(4, 3); b = np.arange(6, dtype=np.int8)
    meta = dict(id="t/map")
    path = str(tmp_path / "m.rpk")
    write_rpk(path, {"a": [a[:1], a[1:]], "b": b, "s": np.float64(2.5)}, meta)
    with safe_open(path, framework="numpy") as f:                      # a safetensors file any reader opens
        np.testing.assert_array_equal(f.get_tensor("a"), a); np.testing.assert_array_equal(f.get_tensor("b"), b)
        assert f.get_tensor("s").tolist() == [2.5]                     # a 0-d array is stored as one element
    import json, struct
    with safe_open(path, framework="numpy") as f:
        assert json.loads(f.metadata()["rpk"]) == meta
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
    assert n % 8 == 0                                                   # the widest tensor first, every one aligned
    pk = ReplayPack(*_read_plain(path))
    np.testing.assert_array_equal(pk.arrays["a"], a)
    pk.arrays["a"][0, 0] = 99.0                                       # the reader's copy, not the file
    np.testing.assert_array_equal(_read_plain(path)[0]["a"], a)
    write_rpk(path, {"a": a * 2, "b": b, "s": np.float64(2.5)}, meta)  # the path replaced while it is mapped
    np.testing.assert_array_equal(pk.arrays["a"][1:], a[1:])
    np.testing.assert_array_equal(_read_plain(path)[0]["a"], a * 2)


def _read_plain(path):
    """``(arrays, meta)`` of an ``.rpk`` file as :func:`read_rpk` maps them, without the pack's spec checks."""
    import json, struct
    from dmipy_sim.replay.replay import _NP_DTYPES
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n))
    meta = json.loads(hdr.pop("__metadata__")["rpk"])
    mm = np.memmap(path, np.uint8, mode="c")
    return {k: mm[8 + n + d["data_offsets"][0]:8 + n + d["data_offsets"][1]].view(_NP_DTYPES[d["dtype"]]).reshape(d["shape"])
            for k, d in hdr.items()}, meta


def test_a_merge_streamed_to_its_file_is_the_merge_in_memory(spec_grid, monkeypatch):
    """Two passes of one block merged with ``overlap="recertify"``: streamed to ``out_path`` (and mapped back) the
    merged pack is the one held in memory -- every tensor and the metadata, the union's weights and the floors re-read
    from the coefficients -- when the floors are read a few walkers at a time too: the float32 certificate rows the
    same, the float64 summary to the rounding of a sum over other walker chunks."""
    from dmipy_sim.replay.bank import build_replay_pack, merge_packs
    from dmipy_sim.spec import walk_spec, StratifiedByVoxel
    from tests.conftest import spec_without_source
    spec, grid, tmp = spec_grid
    spec = spec_without_source(spec)
    packs = []
    for p_, per in ((1, 4), (2, 9)):
        want = np.zeros(grid.shape, np.int64); want[0] = per
        w = walk_spec(spec, T_max=8e-4, block_h=2e-4, seed=300 + p_, require_gpu=False, K=3,
                      seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}))
        packs.append(build_replay_pack(w, id=f"t/stream-{p_}", license="x", citation="x", K=3, voxel_grid=grid,
                                       out_path=str(tmp / f"stream-{p_}.rpk")))
    packs = [read_rpk(str(tmp / f"stream-{p_}.rpk")) for p_ in (1, 2)]
    mem = merge_packs(packs, id="t/stream", overlap="recertify")
    monkeypatch.setattr(cx, "SLICE_BYTES", 8 * 3 * 5 * 7)              # seven walkers' coefficients at a time
    disk = merge_packs(packs, id="t/stream", overlap="recertify", out_path=str(tmp / "stream.rpk"))
    assert disk.source == str(tmp / "stream.rpk") and sorted(disk.arrays) == sorted(mem.arrays)
    for k in mem.arrays:
        np.testing.assert_array_equal(np.asarray(disk.arrays[k]), np.asarray(mem.arrays[k]), err_msg=k)
    pv_d, pv_m = disk.meta["fidelity"]["per_voxel"], mem.meta["fidelity"]["per_voxel"]
    assert pv_m["recertified"] and pv_d.keys() == pv_m.keys()
    for k in pv_m:                                   # a floor summed over other walker chunks: the same to rounding
        assert pv_d[k] == pytest.approx(pv_m[k], rel=1e-12) if isinstance(pv_m[k], float) else pv_d[k] == pv_m[k], k
    strip = lambda m: dict(m, fidelity={k: v for k, v in m["fidelity"].items() if k != "per_voxel"})
    assert strip(disk.meta) == strip(mem.meta)


def test_a_pack_file_is_the_reference_writers_byte_for_byte(tmp_path):
    """``write_rpk`` writes the bytes ``safetensors.numpy.save_file`` writes for the same contiguous arrays and the
    ``"rpk"`` metadata -- every dtype a pack stores, ties in dtype ordered by name, an empty tensor, a 0-d one (one
    element, as ``np.ascontiguousarray`` makes it), a big-endian one, a non-ASCII metadata value -- and a tensor given
    as pieces is the file of their concatenation."""
    import json
    from safetensors.numpy import save_file
    rng = np.random.default_rng(0)
    arrays = {"pos_x_b0u8": rng.integers(-100, 100, (100, 37), dtype=np.int8), "x_ends": rng.normal(size=(100, 2)),
              "w": rng.random(100).astype(np.float32), "a": np.zeros(3, np.float32), "empty": np.zeros((0, 5), np.float32),
              "scalar": np.array(3.5), "flags": rng.random(17) > 0.5, "ids": np.arange(10, dtype=np.int64),
              "be": np.arange(6, dtype=">f4").reshape(2, 3), "u16": np.arange(9, dtype=np.uint16),
              "i16": np.arange(4, dtype=np.int16), "u8": np.arange(5, dtype=np.uint8), "h": np.ones(2, np.float16),
              "u32": np.arange(3, dtype=np.uint32), "i32": np.arange(3, dtype=np.int32), "u64": np.arange(2, dtype=np.uint64),
              "strided": np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2]}
    meta = {"id": "t/ref", "nested": {"a": [1, 2.5, None]}, "name": "Façade"}
    ref = str(tmp_path / "ref.rpk")
    save_file({k: np.ascontiguousarray(v) for k, v in arrays.items()}, ref, metadata={"rpk": json.dumps(meta)})
    write_rpk(str(tmp_path / "ours.rpk"), arrays, meta)
    assert (tmp_path / "ours.rpk").read_bytes() == (tmp_path / "ref.rpk").read_bytes()
    pieces = dict(arrays, x_ends=[arrays["x_ends"][:30], arrays["x_ends"][30:31], arrays["x_ends"][31:]],
                  ids=[arrays["ids"][:0], arrays["ids"]])
    write_rpk(str(tmp_path / "pieces.rpk"), pieces, meta)
    assert (tmp_path / "pieces.rpk").read_bytes() == (tmp_path / "ref.rpk").read_bytes()
