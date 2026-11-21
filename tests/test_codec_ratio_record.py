"""The codec-ratio record's tool: ``measure_pack``, ``build_record``, ``summarise``, ``check_record``, ``main``.

A pack built with the codec and the same pack without it pin the re-encode to the writer's own bytes, to the byte;
the committed record checks clean and skips by name whatever pack file is absent; planted faults (a stale ratio, a
stale byte count, a changed pack hash) make ``--check`` exit 1. A pack the writer built over two windows pins the
per-chunk ratios to the chunk's coded bytes over all windows, not one window's.
"""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_replay.replay import codec, container
from dmipy_replay.replay.compression import _load_range, range_name  # _load_range: private name of the reader's codec
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.replay import plan as plan_mod
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.compression import packed_row_bytes
from dmipy_sim.replay.plan import plan_of_pack
from tests._envelope import battery_reach, stated

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
RECORD = ROOT / "dmipy_sim" / "replay" / "codec_ratios.json"
FIXTURES = [DATA / "cylinder_k12.rpk", DATA / "strands_field_k4.rpk"]

PACK_KEYS = {"id", "file", "sha256", "n_walkers", "windows", "K", "chunk_walkers", "tiers"}
TIER_KEYS = {"bands", "bit_packed_bytes", "coded_bytes", "ratio", "via", "range_bit_packed_bytes",
             "range_coded_bytes", "range_ratio", "chunks", "full_chunks", "chunk_ratio_min", "chunk_ratio_max"}

#: the band range tensors of each banded tier, per spatial axis / channel base
_TIER_BASES = {"positions": ("pos_x", "pos_y", "pos_z"), "contact": ("blt",), "field": ("susc_path_dct",)}


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


@pytest.fixture(scope="module")
def two_windows(tmp_path_factory):
    """A pack the writer built over two windows (``window_T`` half of ``T_max``): window ``i``'s tensors under
    ``s<i>/`` and ``walk_params.windows.n == 2``."""
    g = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6)
    w = _walk_geometry(600, 2e-9, g, 10e-3, 2.5e-4, seed=7, require_gpu=False, window_T=5e-3)
    path = tmp_path_factory.mktemp("two_windows") / "two.rpk"
    build_replay_pack(w, id="test/two-windows", license="x", citation="x", out_path=str(path),
                      **stated(of=w, G_max=battery_reach(w.T_window)))
    return path


def _expected_chunk_ratios(path, tier, chunk_walkers):
    """The chunk ratios of ``tier`` measured independently of the tool: per full chunk index, the chunk's coded
    bytes summed over the tier's ranges, axes and windows, over the same chunk's bit-packed bytes of all windows,
    re-encoded with the codec's own ``encode_chunks``. Returns ``(full_chunks, min, max)``."""
    arrays, hdr = container.read_safetensors(path)
    meta = json.loads(hdr["rpk"])
    plan = plan_of_pack(path)
    tv = plan["tiers"][tier]
    bases = _TIER_BASES[tier]
    ranges = plan_mod.tier_ranges(meta, tier)
    row = packed_row_bytes(ranges)
    cw, windows, n = int(chunk_walkers), plan["pack"]["windows"], plan["pack"]["n_walkers"]
    axes, channels = len(bases), tv["channels"] or 1
    chunk_coded = {}
    for w in range(windows):
        prefix = "" if w == 0 else f"s{w}/"
        for i, r in enumerate(ranges):
            bits = int(r["bits"])
            k0, k1 = r["bands"] if "bands" in r else r["modes"]
            width = int(k1) - int(k0)
            for base in bases:
                t = arrays[f"{prefix}{range_name(base, i, bits)}"]
                q = _load_range(t, bits, width)
                _stream, chunks = codec.encode_chunks(np.asarray(q), bits, cw, level=6)
                for c in range(len(chunks) - 1):
                    chunk_coded[c] = chunk_coded.get(c, 0) + int(chunks[c + 1] - chunks[c])
    full = n // cw
    denom = windows * cw * axes * channels * row
    ratios = [round(chunk_coded[c] / denom, 6) for c in range(full)]
    return full, (min(ratios) if ratios else None), (max(ratios) if ratios else None)


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


def test_a_windowed_packs_chunk_ratios_are_per_chunk_over_all_windows(tool, two_windows):
    plan = plan_of_pack(two_windows)
    assert plan["pack"]["windows"] == 2
    entry = tool.measure_pack(two_windows, chunk_walkers=256)
    for tier in ("positions", "contact"):
        t = entry["tiers"][tier]
        assert t["chunks"] == 3 and t["full_chunks"] == 2          # 600 walkers in chunks of 256
        full, lo, hi = _expected_chunk_ratios(two_windows, tier, 256)
        assert full == 2
        assert t["chunk_ratio_min"] == lo and t["chunk_ratio_max"] == hi


def test_the_tool_reads_a_tiers_ranges_through_the_plans_lookup(tool, monkeypatch):
    assert not hasattr(tool, "_tier_ranges")                      # no second copy of the lookup
    real = plan_mod.tier_ranges

    def widened(meta, tier):
        ranges = real(meta, tier)
        if tier == "positions" and ranges:
            out = [dict(r) for r in ranges]
            out[0] = {**out[0], "bits": int(out[0]["bits"]) + 1}
            return out
        return ranges

    with monkeypatch.context() as m:
        m.setattr(plan_mod, "tier_ranges", widened)
        with pytest.raises(ValueError, match="pos_x_b0u17"):
            tool.measure_pack(DATA / "cylinder_k12.rpk")


def test_the_summary_is_the_min_and_max_over_full_chunk_packs(tool):
    def pack(n_walkers, chunk_walkers, ratio):
        return {"id": f"p{n_walkers}", "n_walkers": n_walkers, "chunk_walkers": chunk_walkers,
                "tiers": {"positions": {"ratio": ratio}}}

    packs = [pack(8192, 4096, 0.5), pack(4096, 4096, 0.7), pack(150, 4096, 0.1)]
    assert tool.summarise(packs) == {"positions": {"min": 0.5, "max": 0.7, "packs": 2}}


def test_the_committed_record_checks_clean_and_skips_what_is_absent(tool, capsys, tmp_path):
    record = json.loads(RECORD.read_text())
    assert tool.main(["--check"]) == 0
    out = capsys.readouterr().out
    for entry in record["packs"]:                     # every entry is checked or printed as SKIP, never silently dropped
        if (DATA / entry["file"]).is_file():
            assert f"OK {entry['id']}" in out
        else:
            assert f"SKIP {entry['id']}: {entry['file']} not found" in out
    checked, skipped = (int(x) for x in re.search(r"checked (\d+), skipped (\d+)", out).groups())
    assert checked >= 2 and checked + skipped == len(record["packs"])

    # a record with more packs than are present (nine entries, seven of them absent): the absent ones are
    # SKIPped by name and do not fail the check
    bigger = json.loads(json.dumps(record))
    for i in range(7):
        bigger["packs"].append({**record["packs"][0], "id": f"fixture/absent{i}", "file": f"absent{i}.rpk"})
    path = tmp_path / "bigger.json"
    path.write_text(json.dumps(bigger, indent=1, sort_keys=True) + "\n")
    assert tool.main(["--check", "--record", str(path)]) == 0
    out = capsys.readouterr().out
    for i in range(7):
        assert f"SKIP fixture/absent{i}: absent{i}.rpk not found" in out
    assert "checked 2, skipped 7" in out

    # the committed record carries the in-repo ratios of the record's table to 6 decimals
    tiers = {p["id"]: p["tiers"] for p in record["packs"]}
    expected = {("fixture/cylinder_k12", "positions"): (1.009444, 1.013095),
                ("fixture/cylinder_k12", "contact"): (0.872713, 0.666),
                ("fixture/strands_field_k4", "positions"): (1.062114, 1.163704),
                ("fixture/strands_field_k4", "contact"): (0.924672, 0.616667),
                ("fixture/strands_field_k4", "field"): (0.423895, 0.372685)}
    for (pid, tier), (ratio, range_ratio) in expected.items():
        assert tiers[pid][tier]["ratio"] == ratio, (pid, tier)
        assert tiers[pid][tier]["range_ratio"] == range_ratio, (pid, tier)


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
