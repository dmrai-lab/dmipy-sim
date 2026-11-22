"""``pack_plan``'s coded stage: the ratio spread measured on the record's packs of the planned pack's chunk fill.

With a codec, ``pack_plan`` states a banded tier's ``coded`` and ``stored`` as a range from the committed ratio record
``dmipy_sim/replay/codec_ratios.json``. The basis is the planned pack's own chunk fill at the requested
``chunk_walkers``: ``full_chunks`` (the record's packs with at least one full chunk) for a plan of at least one chunk,
``sub_chunk_packs`` (the record's packs with less than one chunk) below it, named in the figure's source. The range is
the spread measured over the packs of that basis and holds for them; the lane bound is the only bound, the range's
upper end below one chunk, and it stands alone (a proxy) where the record holds no pack of the basis; the ``ratios``
keyword hands the plan another record of the same schema. The range contains the stored bytes of every pack of the
record in ``tests/data`` re-encoded with the codec's own :func:`~dmipy_replay.replay.codec.encode_ranges`; the lane
bound holds for incompressible data; a record that does not fit is refused by name; and the planted faults (doubled
ratios, no lane bound, a ratio taken from packs of another chunk size) are each found.
"""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from dmipy_replay.replay import codec, container
from dmipy_sim.replay import plan
from dmipy_sim.replay.container import write_rpk
from dmipy_sim.replay.plan import compare_plans, pack_plan, pack_plan_problems, plan_of_pack
from tests.test_pack_plan import (CYLINDER, DATA, STRANDS, built_pack, coded_pack, inputs_of, packs,  # noqa: F401
                                  walk)

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "dmipy_sim" / "replay" / "codec_ratios.json"

#: the packs of the packaged record, ``{id: n_walkers}``, all at 4096 walkers per chunk
PACKAGED = {"cactus/bundle00000-100ms": 111011, "fixture/plane-linear-field": 8000, "fixture/slab-exchange": 8000,
            "fixture/cylinder": 2000, "fixture/cylinder_k12": 400, "fixture/field": 150, "fixture/permeable": 2000,
            "fixture/strands_field_k4": 90, "fixture/windows": 600}


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("codec_ratio_record", ROOT / "tools" / "codec_ratio_record.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _encode(path, tmp_path, cw):
    """Re-encode the pack at ``path``'s band ranges with the codec's own ``encode_ranges`` in chunks of ``cw``
    walkers, write the result with ``write_rpk`` and return its ``plan_of_pack`` (the file's stored bytes)."""
    arrays, hdr = container.read_safetensors(path)
    meta = json.loads(hdr["rpk"])
    arrays, meta = codec.encode_ranges(arrays, meta, codec="bitshuffle-zlib", chunk_walkers=cw)
    out = tmp_path / f"{Path(path).stem}_coded{cw}.rpk"
    write_rpk(out, arrays, meta)
    return plan_of_pack(out)


def _hand_record(entries):
    """A record of the ``codec_ratio_record/1`` schema whose pack entries are ``(n_walkers, chunk_walkers, ratio)``
    triples, each carrying the ``positions`` tier alone."""
    return {"schema": "codec_ratio_record/1", "codec": "bitshuffle-zlib", "level": 6,
            "tolerance": {"coded_relative": 0.005},
            "packs": [{"id": f"hand/{i}", "file": f"hand{i}.rpk", "sha256": "0" * 64, "n_walkers": n, "windows": 1,
                       "K": 12, "chunk_walkers": cw, "tiers": {"positions": {"ratio": r}}}
                      for i, (n, cw, r) in enumerate(entries)],
            "summary": {}}


def basis_problems(pl, record):
    """One sentence per figure whose source names the ratio record while the record's packs of that tier were
    measured at another chunk size than the plan's ``inputs["chunk_walkers"]``."""
    problems = []
    cw = pl["inputs"]["chunk_walkers"]
    for path, fig in plan._figures_of(pl["tiers"]):
        src = fig.get("source") or ""
        if "codec_ratios.json" in src:
            tier = src.rsplit("/", 1)[-1]
            cws = {p["chunk_walkers"] for p in record["packs"] if tier in p["tiers"]}
            if cws and cws != {cw}:
                problems.append(f"{path}: ratio taken from packs of another chunk size")
    return problems


def test_the_coded_range_contains_the_stored_bytes_of_every_pack_of_the_record(tool, packs, coded_pack, tmp_path):
    record = json.loads(RECORD.read_text())
    tol = record["tolerance"]["coded_relative"]

    # every pack of the committed record whose file is in tests/data, re-encoded at 4096: the plan is sound and
    # each banded tier's coded range contains the file's stored bytes, measured within the record's tolerance
    present = [e for e in record["packs"] if (DATA / e["file"]).is_file()]
    assert present, "no pack of the record is in tests/data: the loop below would check nothing"
    for entry in present:
        path = DATA / entry["file"]
        built = _encode(path, tmp_path, 4096)
        pre = pack_plan(**{**inputs_of(path), "codec": "bitshuffle-zlib", "chunk_walkers": 4096})
        assert compare_plans(pre, built) == []
        assert pack_plan_problems(pre) == []
        measured = tool.measure_pack(path)["tiers"]
        for tier, bt in built["tiers"].items():
            if not (bt["banded"] and bt["codec"]):
                continue
            fig = pre["tiers"][tier]["coded"]
            stored = bt["stored"]["value"]
            assert fig["provenance"] == "measured", (entry["id"], tier)
            assert "codec_ratios.json#sub_chunk_packs/" in fig["source"], (entry["id"], tier, fig["source"])
            assert fig["lo"] <= stored <= fig["hi"], (entry["id"], tier, fig, stored)
            assert abs(measured[tier]["coded_bytes"] - stored) <= tol * abs(stored), (entry["id"], tier)

    # the built 600-walker coded pack at chunk_walkers=256: no record pack has that chunk size, so the coded range
    # is the lane bound alone (a proxy) and it contains the file's stored bytes
    kw = inputs_of(coded_pack)
    assert kw["codec"] == "bitshuffle-zlib" and kw["chunk_walkers"] == 256
    built = plan_of_pack(coded_pack)
    pre = pack_plan(**kw)
    assert compare_plans(pre, built) == []
    assert pack_plan_problems(pre) == []
    for tier in ("positions", "contact"):
        fig = pre["tiers"][tier]["coded"]
        assert fig["value"] is None and fig["provenance"] == "proxy"
        assert fig["source"] == "bound:lane_widened_block_padded"
        assert fig["lo"] <= built["tiers"][tier]["stored"]["value"] <= fig["hi"]

    # every plan of the B1 pack set re-run with the default codec is itself sound
    for path in packs:
        pl = pack_plan(**{**inputs_of(path), "codec": "bitshuffle-zlib", "chunk_walkers": 4096})
        assert pack_plan_problems(pl) == []


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("cw", [256, 4096])
@pytest.mark.parametrize("axes", [1, 3])
@pytest.mark.parametrize("Cx", [1, 12])
@pytest.mark.parametrize("K", [1, 7, 40])
@pytest.mark.parametrize("N", [5, 300, 4097])
def test_the_lane_bound_holds_for_incompressible_data(N, K, Cx, bits, axes, cw):
    rng = np.random.default_rng(0)
    lo = -(1 << (codec.lane_bits(bits) - 1))
    q = rng.integers(lo, -lo, size=(N, Cx, K)).astype(codec.lane_dtype(bits))
    stream, chunks = codec.encode_chunks(q, bits, cw)
    stored = axes * (int(stream.nbytes) + 8 * len(chunks))
    bound = plan._lane_bound("positions", N, 1, K, Cx, axes, cw, 0)
    assert stored <= bound, (N, K, Cx, bits, axes, cw, stored, bound)


def test_the_ratio_basis_follows_the_planned_packs_chunk_fill():
    kw = {**inputs_of(CYLINDER), "codec": "bitshuffle-zlib", "chunk_walkers": 4096}
    packaged = pack_plan(**kw)
    assert packaged["inputs"]["ratios"] == {"schema": "codec_ratio_record/1", "codec": "bitshuffle-zlib", "packs": len(PACKAGED)}
    assert packaged["tiers"]["positions"]["coded"]["source"] == "codec_ratios.json#sub_chunk_packs/positions"

    # full-chunk packs present (0.5 and 0.7 at 8192 and 4096 walkers; 0.1 at 150 is sub-chunk and excluded):
    # the range [0.5, 0.7] over the bit-packed stage, source full_chunks, and no widening at n_walkers >= cw
    rec_full = _hand_record([(8192, 4096, 0.5), (4096, 4096, 0.7), (150, 4096, 0.1)])
    big = pack_plan(**{**kw, "n_walkers": 8192}, ratios=rec_full)
    fig, bp = big["tiers"]["positions"]["coded"], big["tiers"]["positions"]["bit_packed"]
    assert fig["provenance"] == "measured" and fig["source"] == "codec_ratios.json#full_chunks/positions"
    assert fig["lo"] == math.floor(0.5 * bp["lo"]) and fig["hi"] == math.ceil(0.7 * bp["hi"])

    # the same record at n_walkers < cw: the basis is the sub-chunk pack alone (0.1 at 150 walkers), hi is widened to
    # the lane bound
    small = pack_plan(**kw, ratios=rec_full)
    fig_s, bp_s = small["tiers"]["positions"]["coded"], small["tiers"]["positions"]["bit_packed"]
    assert fig_s["provenance"] == "measured" and fig_s["source"] == "codec_ratios.json#sub_chunk_packs/positions"
    assert fig_s["lo"] == math.floor(0.1 * bp_s["lo"])
    proxy = pack_plan(**kw, ratios=_hand_record([(8192, 1024, 0.5)]))    # another chunk size: case 3
    lane_hi = proxy["tiers"]["positions"]["coded"]["hi"]
    assert proxy["tiers"]["positions"]["coded"]["provenance"] == "proxy"
    assert proxy["tiers"]["positions"]["coded"]["source"] == "bound:lane_widened_block_padded"
    assert fig_s["hi"] == max(math.ceil(0.1 * bp_s["hi"]), lane_hi) == lane_hi

    # a tier the record lacks is case 3 for that tier only
    assert small["tiers"]["contact"]["coded"]["provenance"] == "proxy"
    assert small["tiers"]["contact"]["coded"]["source"] == "bound:lane_widened_block_padded"

    # only sub-chunk packs: the range from them, source sub_chunk_packs, hi widened by the lane bound (case 2)
    rec_sub = _hand_record([(150, 4096, 0.3), (400, 4096, 0.6)])
    sub = pack_plan(**kw, ratios=rec_sub)
    fig, bp = sub["tiers"]["positions"]["coded"], sub["tiers"]["positions"]["bit_packed"]
    assert fig["provenance"] == "measured" and fig["source"] == "codec_ratios.json#sub_chunk_packs/positions"
    assert fig["lo"] == math.floor(0.3 * bp["lo"])
    assert fig["hi"] == max(math.ceil(0.6 * bp["hi"]), lane_hi)

    # packs at another chunk size only: case 3, the lane bound alone as a proxy
    assert proxy["tiers"]["positions"]["coded"]["lo"] < lane_hi


def test_a_record_that_does_not_fit_is_refused_by_name():
    kw = {**inputs_of(CYLINDER), "codec": "bitshuffle-zlib", "chunk_walkers": 4096}
    record = json.loads(RECORD.read_text())

    bad_schema = json.loads(json.dumps(record))
    bad_schema["schema"] = "x"
    with pytest.raises(ValueError, match="ratios"):
        pack_plan(**kw, ratios=bad_schema)

    bad_codec = json.loads(json.dumps(record))
    bad_codec["codec"] = "other"
    with pytest.raises(ValueError, match="codec"):
        pack_plan(**kw, ratios=bad_codec)

    with pytest.raises(ValueError, match="ratios"):
        pack_plan(**kw, ratios=["not", "a", "record"])

    # with codec=None the record is not read and inputs["ratios"] is None
    pl = pack_plan(**{**inputs_of(CYLINDER), "codec": None}, ratios=bad_schema)
    assert pl["inputs"]["ratios"] is None


def test_each_planted_fault_is_found(tmp_path, coded_pack, monkeypatch):
    record = json.loads(RECORD.read_text())

    # the sound comparison first: every pack of the record re-encoded at 4096 is inside the planned range
    present = [e for e in record["packs"] if (DATA / e["file"]).is_file()]
    assert present, "no pack of the record is in tests/data: the loop below would check nothing"
    for entry in present:
        path = DATA / entry["file"]
        built = _encode(path, tmp_path, 4096)
        pre = pack_plan(**{**inputs_of(path), "codec": "bitshuffle-zlib", "chunk_walkers": 4096})
        assert compare_plans(pre, built) == []

    # A, the ratios doubled: the planned lo rises above the pack's stored bytes
    doubled = json.loads(json.dumps(record))
    for p in doubled["packs"]:
        for t in p["tiers"].values():
            t["ratio"] = t["ratio"] * 2
    built = _encode(STRANDS, tmp_path, 4096)
    problems = compare_plans(
        pack_plan(**{**inputs_of(STRANDS), "codec": "bitshuffle-zlib", "chunk_walkers": 4096}, ratios=doubled),
        built)
    assert any("tier positions: stored" in s and "is outside the planned" in s for s in problems), problems

    # B, no lane bound: the 600-walker coded pack at chunk_walkers=256 is case 3, its coded hi is the bound
    kw = inputs_of(coded_pack)
    built = plan_of_pack(coded_pack)
    assert compare_plans(pack_plan(**kw), built) == []
    with monkeypatch.context() as m:
        m.setattr(plan, "_lane_bound", lambda *a, **k: 0)
        problems = compare_plans(pack_plan(**kw), built)
    assert any("tier positions: stored" in s for s in problems), problems

    # C, a ratio taken from packs of another chunk size: the sound plan is case 3, the planted one is measured
    rec_other = _hand_record([(8192, 1024, 0.5), (4096, 1024, 0.7), (300, 1024, 0.9)])
    kw4096 = {**inputs_of(CYLINDER), "codec": "bitshuffle-zlib", "chunk_walkers": 4096}
    sound = pack_plan(**kw4096, ratios=rec_other)
    assert sound["tiers"]["positions"]["coded"]["provenance"] == "proxy"
    assert basis_problems(sound, rec_other) == []
    real = plan._ratio_packs
    with monkeypatch.context() as m:
        m.setattr(plan, "_ratio_packs", lambda record, tier, cw, n: real(record, tier, 1024, n))
        planted = pack_plan(**kw4096, ratios=rec_other)
    assert planted["tiers"]["positions"]["coded"]["provenance"] == "measured"
    problems = basis_problems(planted, rec_other)
    assert any("ratio taken from packs of another chunk size" in s for s in problems), problems


@pytest.mark.parametrize("n", [1, 4095, 4096, 4097, 8192, 111011])
def test_the_basis_is_the_planned_packs_own_chunk_fill(n):
    kw = {**inputs_of(CYLINDER), "codec": "bitshuffle-zlib", "chunk_walkers": 4096, "n_walkers": n}
    record = _hand_record([(8192, 4096, 0.5), (4096, 4096, 0.7), (4095, 4096, 0.9), (90, 4096, 1.1)])
    full = n >= 4096
    basis, lo_ratio, hi_ratio = ("full_chunks", 0.5, 0.7) if full else ("sub_chunk_packs", 0.9, 1.1)
    lane = pack_plan(**kw, ratios=_hand_record([(8192, 1024, 0.5)]))["tiers"]["positions"]
    lane_hi = lane["coded"]["hi"]
    assert lane["coded"]["provenance"] == "proxy" and lane["coded"]["lo"] < lane_hi

    tier = pack_plan(**kw, ratios=record)["tiers"]["positions"]
    fig, bp = tier["coded"], tier["bit_packed"]
    assert fig["provenance"] == "measured" and fig["source"] == f"codec_ratios.json#{basis}/positions"
    assert fig["lo"] == math.floor(lo_ratio * bp["lo"])
    if full:
        assert fig["hi"] == math.ceil(hi_ratio * bp["hi"]) < lane_hi
    else:
        assert fig["hi"] == max(math.ceil(hi_ratio * bp["hi"]), lane_hi)

    # a record that holds only the packs of the other basis gives the lane bound alone
    other = _hand_record([(90, 4096, 1.1), (4095, 4096, 0.9)] if full else [(8192, 4096, 0.5), (4096, 4096, 0.7)])
    fig = pack_plan(**kw, ratios=other)["tiers"]["positions"]["coded"]
    assert fig["provenance"] == "proxy" and fig["source"] == "bound:lane_widened_block_padded"
    assert fig["hi"] == lane_hi


def test_a_ragged_pack_and_a_sub_chunk_pack_of_one_record_take_their_own_basis(tool, built_pack, tmp_path):
    paths = [CYLINDER, built_pack, STRANDS]
    record = tool.build_record(paths, chunk_walkers=256)
    by_file = {Path(p["file"]).name: p for p in record["packs"]}
    counts = [by_file[Path(p).name]["n_walkers"] for p in paths]
    assert counts == [400, 600, 90]
    assert [by_file[Path(p).name]["tiers"]["positions"]["full_chunks"] for p in paths] == [1, 2, 0]
    for path, basis in zip(paths, ["full_chunks", "full_chunks", "sub_chunk_packs"]):
        built = _encode(path, tmp_path, 256)
        pre = pack_plan(**{**inputs_of(path), "codec": "bitshuffle-zlib", "chunk_walkers": 256}, ratios=record)
        assert compare_plans(pre, built) == []
        checked = 0
        for tier, bt in built["tiers"].items():
            if not (bt["banded"] and bt["codec"]):
                continue
            fig = pre["tiers"][tier]["coded"]
            assert fig["provenance"] == "measured", (path, tier)
            assert fig["source"] == f"codec_ratios.json#{basis}/{tier}", (path, tier, fig["source"])
            assert fig["lo"] <= bt["stored"]["value"] <= fig["hi"], (path, tier, fig)
            checked += 1
        assert checked


def test_the_packaged_record_is_the_nine_public_packs_and_holds_both_bases(tool):
    record = json.loads(RECORD.read_text())
    assert {p["id"]: p["n_walkers"] for p in record["packs"]} == PACKAGED
    assert {p["chunk_walkers"] for p in record["packs"]} == {4096}
    assert record["summary"] == tool.summarise(record["packs"])
    for tier in ("positions", "contact", "field"):
        have = [p for p in record["packs"] if tier in p["tiers"]]
        full = [p for p in have if p["n_walkers"] >= p["chunk_walkers"]]
        assert full and len(full) < len(have), tier
        assert record["summary"][tier]["packs"] == len(full), tier
