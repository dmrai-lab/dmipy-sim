"""The codec-ratio record's tool: ``measure_pack``, ``build_record``, ``summarise``, ``check_record``, ``main``.

A pack built with the codec and the same pack without it pin the re-encode to the writer's own bytes, to the byte;
the committed record checks clean and skips by name; planted faults (a stale ratio, a stale byte count, a changed
pack hash) make ``--check`` exit 1.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

import dmipy_sim as d
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.plan import plan_of_pack
from tests._envelope import battery_reach, stated

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
RECORD = ROOT / "dmipy_sim" / "replay" / "codec_ratios.json"
FIXTURES = [DATA / "cylinder_k12.rpk", DATA / "strands_field_k4.rpk"]

PACK_KEYS = {"id", "file", "sha256", "n_walkers", "windows", "K", "chunk_walkers", "tiers"}
TIER_KEYS = {"bands", "bit_packed_bytes", "coded_bytes", "ratio", "via", "range_bit_packed_bytes",
             "range_coded_bytes", "range_ratio", "chunks", "full_chunks", "chunk_ratio_min", "chunk_ratio_max"}


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("codec_ratio_record", ROOT / "tools" / "codec_ratio_record.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def walk():
    g = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6)
    return _walk_geometry(600, 2e-9, g, 20e-3, 2.5e-4, seed=7, require_gpu=False)


def _build(walk, path, **kw):
    return build_replay_pack(walk, id="test/codec", license="x", citation="x", K=16, out_path=str(path),
                             **stated(of=walk, G_max=battery_reach(walk.T_window)), **kw)


@pytest.fixture(scope="module")
def packs(walk, tmp_path_factory):
    d_ = tmp_path_factory.mktemp("ratio")
    _build(walk, d_ / "fixed.rpk")
    _build(walk, d_ / "coded.rpk", codec="bitshuffle-zlib", chunk_walkers=256)
    return d_


def test_the_record_of_a_built_pack_has_the_schema(tool, packs, tmp_path):
    record = tool.build_record([str(packs / "fixed.rpk")], chunk_walkers=256)
    assert set(record) == {"schema", "codec", "level", "tolerance", "packs", "summary"}
    assert record["schema"] == "codec_ratio_record/1"
    assert record["codec"] == "bitshuffle-zlib" and record["level"] == 6
    assert record["tolerance"] == {"coded_relative": 0.005}
    assert len(record["packs"]) == 1
    entry = record["packs"][0]
    assert set(entry) == PACK_KEYS
    assert entry["sha256"] == hashlib.sha256((packs / "fixed.rpk").read_bytes()).hexdigest()
    assert entry["file"] == "fixed.rpk"
    text = json.dumps(record, indent=1, sort_keys=True)
    assert str(tmp_path) not in text
    assert set(entry["tiers"]) == {"positions", "contact"}
    for t in entry["tiers"].values():
        assert set(t) == TIER_KEYS
        assert t["full_chunks"] == 2                      # 600 walkers in chunks of 256
        assert t["chunk_ratio_min"] is not None and t["chunk_ratio_max"] is not None
    assert json.loads(json.dumps(record, indent=1, sort_keys=True)) == record


def test_the_re_encode_is_the_writers_own_to_the_byte(tool, packs):
    fixed_entry = tool.build_record([str(packs / "fixed.rpk")], chunk_walkers=256)["packs"][0]
    coded_plan = plan_of_pack(packs / "coded.rpk")
    for tier, t in fixed_entry["tiers"].items():
        assert t["coded_bytes"] == coded_plan["tiers"][tier]["stored"]["value"]
        assert t["ratio"] == round(t["coded_bytes"] / t["bit_packed_bytes"], 6)
    coded_entry = tool.measure_pack(packs / "coded.rpk")
    for tier, t in fixed_entry["tiers"].items():
        assert coded_entry["tiers"][tier]["via"] == "file"
        assert coded_entry["tiers"][tier]["ratio"] == t["ratio"]
        assert coded_entry["tiers"][tier]["coded_bytes"] == t["coded_bytes"]


def test_the_summary_is_the_min_and_max_over_full_chunk_packs(tool):
    def pack(n_walkers, chunk_walkers, ratio):
        return {"id": f"p{n_walkers}", "n_walkers": n_walkers, "chunk_walkers": chunk_walkers,
                "tiers": {"positions": {"ratio": ratio}}}

    packs = [pack(8192, 4096, 0.5), pack(4096, 4096, 0.7), pack(150, 4096, 0.1)]
    assert tool.summarise(packs) == {"positions": {"min": 0.5, "max": 0.7, "packs": 2}}


def test_the_committed_record_checks_clean_and_skips_what_is_absent(tool, capsys, tmp_path):
    assert tool.main(["--check"]) == 0
    out = capsys.readouterr().out
    assert "OK fixture/cylinder_k12" in out and "OK fixture/strands_field_k4" in out
    assert "checked 2, skipped 0" in out and "SKIP" not in out
    record = json.loads(RECORD.read_text())
    absent = json.loads(json.dumps(record))
    absent["packs"].append({**absent["packs"][0], "id": "fixture/absent", "file": "absent.rpk"})
    path = tmp_path / "absent.json"
    path.write_text(json.dumps(absent, indent=1, sort_keys=True) + "\n")
    assert tool.main(["--check", "--record", str(path)]) == 0
    out = capsys.readouterr().out
    assert "SKIP fixture/absent: absent.rpk not found" in out
    assert "checked 2, skipped 1" in out


def test_each_planted_fault_makes_check_exit_one(tool, capsys, tmp_path):
    out_path = tmp_path / "rec.json"
    assert tool.main([*[str(p) for p in FIXTURES], "--out", str(out_path)]) == 0
    capsys.readouterr()
    assert tool.main(["--check", "--record", str(out_path)]) == 0
    capsys.readouterr()
    base = json.loads(out_path.read_text())

    def planted(mutate):
        rec = json.loads(json.dumps(base))
        mutate(rec)
        path = tmp_path / "planted.json"
        path.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
        code = tool.main(["--check", "--record", str(path)])
        return code, capsys.readouterr().out

    def stale_ratio(rec):
        t = rec["packs"][0]["tiers"]["positions"]
        t["ratio"] = round(t["ratio"] * 1.05, 6)

    code, out = planted(stale_ratio)
    assert code == 1 and "ratio" in out

    def stale_byte_count(rec):
        rec["packs"][0]["tiers"]["positions"]["bit_packed_bytes"] += 1

    code, out = planted(stale_byte_count)
    assert code == 1 and "bit_packed_bytes" in out

    def changed_hash(rec):
        p = rec["packs"][0]
        p["sha256"] = ("0" if p["sha256"][0] != "0" else "1") + p["sha256"][1:]

    code, out = planted(changed_hash)
    assert code == 1 and "pack file changed" in out

    def ratio_within_tolerance(rec):
        t = rec["packs"][0]["tiers"]["positions"]
        t["ratio"] = round(t["ratio"] * 1.001, 6)

    code, out = planted(ratio_within_tolerance)
    assert code == 0
