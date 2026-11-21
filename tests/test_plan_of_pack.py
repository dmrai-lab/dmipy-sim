"""``plan_of_pack``: the exact bytes of a built pack per tier -- raw, bit-packed, stored -- and the faults
``plan_problems`` reports.

The bit-packed count is the closed form of the header's containers through ``compression.packed_row_bytes``, the
function ``pack_bits`` itself writes with; the stored count is the sum of the file's tensor byte ranges; the raw
count is the float container's coefficient bytes. The two committed fixtures pin every stage; packs built inside
the tests cover the coded container, a second window and the float container; planted faults check that each
problem sentence is produced.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_replay.replay import container
from dmipy_replay.replay.compression import read_position_coeffs
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.replay import compression, plan
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.compression import pack_bits, packed_row_bytes
from dmipy_sim.replay.container import write_rpk
from dmipy_sim.replay.plan import plan_of_pack, plan_problems
from tests._envelope import battery_reach, stated

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
CYLINDER = "cylinder_k12.rpk"
STRANDS = "strands_field_k4.rpk"

#: the sources a plan's figures carry
SOURCES = ("header:data_offsets", "header:containers", "derived:float32_count", "stored:not_banded",
           "none:not_banded", "none:float_container")

#: the stages of the two committed packs: (file bytes, header bytes) and per tier (raw, bit-packed, stored)
STAGES = {
    CYLINDER: {"file": (48000, 7408),
               "tiers": {"positions": (67200, 34944, 34944), "contact": (22400, 5248, 5248),
                         "occupancy": (400, None, 400)}},
    STRANDS: {"file": (26220, 18944),
              "tiers": {"positions": (6480, 3558, 3558), "contact": (2160, 916, 916), "field": (17280, 2352, 2352),
                        "occupancy": (90, None, 90), "weights": (360, None, 360)}},
}


@pytest.fixture(scope="module")
def walk():
    g = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6)
    return _walk_geometry(600, 2e-9, g, 20e-3, 2.5e-4, seed=7, require_gpu=False)


def _build(walk, path, **kw):
    return build_replay_pack(walk, id="test/codec", license="x", citation="x", K=16, out_path=str(path),
                             **stated(of=walk, G_max=battery_reach(walk.T_window)), **kw)


@pytest.fixture(scope="module")
def packs(walk, tmp_path_factory):
    d_ = tmp_path_factory.mktemp("plan")
    _build(walk, d_ / "fixed.rpk")
    _build(walk, d_ / "coded.rpk", codec="bitshuffle-zlib", chunk_walkers=256)
    return d_


def test_packed_row_bytes_is_the_ceil_of_the_bits_of_each_range():
    assert packed_row_bytes([{"bands": [0, 4], "bits": 9}]) == 5
    assert packed_row_bytes([{"bands": [0, 5], "bits": 3}]) == 2
    assert packed_row_bytes([{"bands": [0, 8], "bits": 3}, {"modes": [8, 10], "bits": 16}]) == 7
    assert packed_row_bytes([{"bands": [0, 3], "bits": 8}]) == 3
    with pytest.raises(ValueError, match="neither 'bands' nor 'modes'"):
        packed_row_bytes([{"bits": 9}])
    with pytest.raises(ValueError, match="ends before it starts"):
        packed_row_bytes([{"bands": [4, 2], "bits": 9}])
    with pytest.raises(ValueError, match="fewer than 1 bit"):
        packed_row_bytes([{"bands": [0, 4], "bits": 0}])
    for bits in range(3, 17):
        for m in range(1, 21):
            assert pack_bits(np.zeros((2, m), np.uint16), bits).shape[1] \
                == packed_row_bytes([{"bands": [0, m], "bits": bits}]) == -(-m * bits // 8)


def test_the_committed_fixtures_have_the_stated_stages():
    for name, expected in STAGES.items():
        p = plan_of_pack(DATA / name)
        assert (p["file"]["bytes"], p["file"]["header_bytes"]) == expected["file"]
        assert p["file"]["unaccounted_bytes"] == 0
        assert list(p["tiers"]) == list(expected["tiers"])
        for tier, (raw, bit_packed, stored) in expected["tiers"].items():
            t = p["tiers"][tier]
            assert (t["raw"]["value"], t["bit_packed"]["value"], t["stored"]["value"]) == (raw, bit_packed, stored)
            assert t["stored"]["source"] == "header:data_offsets"
            for figure in (t["raw"], t["bit_packed"], t["stored"]):
                assert figure["source"] in SOURCES
        assert plan_problems(p) == []
        assert json.loads(json.dumps(p)) == p
    assert plan_of_pack(DATA / CYLINDER)["tiers"]["positions"]["bands"] == 12
    assert plan_of_pack(DATA / STRANDS)["tiers"]["positions"]["bands"] == 4
    assert plan_of_pack(DATA / STRANDS)["tiers"]["field"]["channels"] == 12


def test_a_built_pack_with_and_without_the_codec(packs):
    fixed = plan_of_pack(packs / "fixed.rpk")
    coded = plan_of_pack(packs / "coded.rpk")
    assert plan_problems(fixed) == [] and plan_problems(coded) == []
    for p in (fixed, coded):
        assert p["file"]["unaccounted_bytes"] == 0
    for t in fixed["tiers"].values():
        if t["banded"]:
            assert t["bit_packed"]["value"] == t["stored"]["value"]
    for tier in ("positions", "contact"):
        assert coded["tiers"][tier]["codec"] == "bitshuffle-zlib"
        assert coded["tiers"][tier]["bit_packed"]["value"] == fixed["tiers"][tier]["stored"]["value"]
        assert coded["tiers"][tier]["raw"] == fixed["tiers"][tier]["raw"]
    for tier, t in fixed["tiers"].items():
        if tier not in ("positions", "contact"):
            assert coded["tiers"][tier]["stored"] == t["stored"]


def test_a_second_window_is_counted_and_a_float_container_is_raw(packs, tmp_path):
    arrays, hdr = container.read_safetensors(packs / "fixed.rpk")
    meta = json.loads(hdr["rpk"])
    one = plan_of_pack(packs / "fixed.rpk")

    two_arrays = dict(arrays)
    two_arrays.update({f"s1/{k}": v for k, v in arrays.items()})
    two_meta = json.loads(json.dumps(meta))
    two_meta["walk_params"]["windows"] = {"n": 2, "T": meta["walk_params"]["windows"]["T"], "walks": []}
    two = tmp_path / "two_windows.rpk"
    write_rpk(two, two_arrays, two_meta)
    p2 = plan_of_pack(two)
    assert p2["pack"]["windows"] == 2
    assert plan_problems(p2) == []
    for tier in ("positions", "contact"):
        for figure in ("raw", "bit_packed", "stored"):
            assert p2["tiers"][tier][figure]["value"] == 2 * one["tiers"][tier][figure]["value"]

    float_arrays = {k: v for k, v in arrays.items() if not k.startswith("pos_")}
    coeffs = read_position_coeffs(arrays, dtype=np.float32)
    for i, axis in enumerate(("x", "y", "z")):
        float_arrays[f"pos_{axis}"] = np.ascontiguousarray(coeffs[:, :, i])
    float_meta = json.loads(json.dumps(meta))
    float_meta["compression"].pop("container")
    flt = tmp_path / "float.rpk"
    write_rpk(flt, float_arrays, float_meta)
    pf = plan_of_pack(flt)
    pos = pf["tiers"]["positions"]
    n, K = pf["pack"]["n_walkers"], pos["bands"]
    assert pos["bit_packed"] == {"value": None, "source": "none:float_container"}
    assert pos["raw"]["value"] == pos["stored"]["value"] == 4 * 3 * n * (K + 2)


def test_each_planted_fault_is_found(tmp_path, monkeypatch):
    strands, cylinder = DATA / STRANDS, DATA / CYLINDER
    assert plan_problems(plan_of_pack(strands)) == []
    assert plan_problems(plan_of_pack(cylinder)) == []

    def rewrite(src, dst, mutate=None, extra=b""):
        arrays, hdr = container.read_safetensors(src)
        meta = json.loads(hdr["rpk"])
        if mutate is not None:
            mutate(meta)
        write_rpk(dst, arrays, meta)
        if extra:
            with open(dst, "ab") as f:
                f.write(extra)

    # A, a range widened by one bit: the cylinder fixture's first range is one band at 16 bits, so one more bit
    # widens its row from 2 to 3 bytes against the stored tensors (on the strands fixture 4 bands at 10 bits still
    # occupy 5 bytes, so the same plant there changes no byte count)
    widened = tmp_path / "widened.rpk"
    rewrite(cylinder, widened,
            lambda m: m["compression"]["container"][0].__setitem__("bits", m["compression"]["container"][0]["bits"] + 1))
    problems = plan_problems(plan_of_pack(widened))
    assert any("tier positions: bit-packed bytes" in s for s in problems), problems

    # B, a tier omitted from the writer's table
    with monkeypatch.context() as m:
        m.setattr(plan, "_TIER_PREFIXES", tuple(r for r in plan._TIER_PREFIXES if r != ("blt_", "contact")))
        problems = plan_problems(plan_of_pack(strands))
    assert any("tier 'other' holds tensors" in s and "blt_start" in s for s in problems), problems

    # C, floor for ceil
    def floor_row_bytes(ranges):
        total = 0
        for r in ranges:
            k0, k1 = r["bands"] if "bands" in r else r["modes"]
            total += (int(k1) - int(k0)) * int(r["bits"]) // 8
        return total

    with monkeypatch.context() as m:
        m.setattr(compression, "packed_row_bytes", floor_row_bytes)
        problems = plan_problems(plan_of_pack(strands))
    assert any("tier positions: bit-packed bytes" in s for s in problems), problems

    # D, a stale remainder
    trailing = tmp_path / "trailing.rpk"
    rewrite(strands, trailing, extra=b"12345678")
    p = plan_of_pack(trailing)
    assert p["file"]["unaccounted_bytes"] == 8
    assert any("unaccounted" in s for s in plan_problems(p))
