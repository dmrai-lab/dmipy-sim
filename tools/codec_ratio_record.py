"""The codec-ratio record: the measured stored-over-bit-packed ratio of the lossless codec, per tier per pack.

``measure_pack`` measures one pack: a tier stored uncoded is re-encoded with the codec's own ``encode_chunks`` (level
6, the chunk size given), so the ratio is the codec's and not a formula's; a tier stored coded takes its ratio from
the file. ``build_record`` assembles the packs' entries, ``summarise`` the per-tier spread over the packs with at
least one full chunk, and ``check_record`` regenerates every entry in memory and compares it with the record. The
committed record ``dmipy_sim/replay/codec_ratios.json`` is written by this tool and checked by it:

    python tools/codec_ratio_record.py PACK [PACK ...] --out RECORD [--chunk-walkers 4096]
    python tools/codec_ratio_record.py --check [--record RECORD] [--packs-dir DIR ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dmipy_replay.replay import codec as _codec  # private names of the reader's codec: _range_entry, encode_chunks
from dmipy_replay.replay.compression import _load_range, range_name  # _load_range: private name of the reader's codec
from dmipy_replay.replay.container import read_safetensors
from dmipy_replay.replay.hub import sha256_of

from dmipy_sim.replay.compression import packed_row_bytes
from dmipy_sim.replay.plan import plan_of_pack

SCHEMA = "codec_ratio_record/1"
CODEC = "bitshuffle-zlib"
LEVEL = 6
DEFAULT_CHUNK_WALKERS = 4096          #: the codec's own block size, the registry's recommendation
TOLERANCE = {"coded_relative": 0.005}  #: the compressed size depends on the zlib build, which is not pinned
RECORD_PATH = ROOT / "dmipy_sim" / "replay" / "codec_ratios.json"
DEFAULT_PACKS_DIR = ROOT / "tests" / "data"

#: the band range tensors of each banded tier, per spatial axis / channel base
_TIER_BASES = {"positions": ("pos_x", "pos_y", "pos_z"), "contact": ("blt",), "field": ("susc_path_dct",)}

#: the record's fields compared exactly, and those compared within the record's tolerance (their value is a
#: compressed size or a ratio derived from one)
_EXACT_PACK_FIELDS = ("id", "file", "sha256", "n_walkers", "windows", "K", "chunk_walkers")
_EXACT_TIER_FIELDS = ("bands", "bit_packed_bytes", "range_bit_packed_bytes", "chunks", "full_chunks", "via")
_CODED_TIER_FIELDS = ("coded_bytes", "range_coded_bytes", "ratio", "range_ratio", "chunk_ratio_min", "chunk_ratio_max")


def _tier_ranges(meta, tier):
    """The band ranges the pack's metadata declares for ``tier`` (positions, contact, field)."""
    cx = meta.get("compression") or {}
    ch = cx.get("channels") or {}
    if tier == "positions":
        return cx.get("container")
    if tier == "contact":
        return (ch.get("boundary_local_time") or {}).get("container")
    if tier == "field":
        return (ch.get("susc_field_basis") or {}).get("container")
    return None


def measure_pack(path, chunk_walkers=4096):
    """One record entry for the pack at ``path``: ``id``, ``file`` (a base name), ``sha256``, ``n_walkers``,
    ``windows``, ``K``, ``chunk_walkers`` and, per banded tier, ``bands``, ``bit_packed_bytes`` (the plan's closed
    form), ``coded_bytes``, ``ratio`` and ``via``, plus the range-level and per-chunk numbers of a re-encode. An
    uncoded tier (``via`` ``"re-encoded"``) is re-encoded range by range, axis by axis and window by window with
    :func:`dmipy_replay.replay.codec.encode_chunks` at level 6: ``range_bit_packed_bytes`` is the range tensors'
    stored bytes, ``range_coded_bytes`` the streams' and chunk tables' bytes, ``coded_bytes`` the tier's bit-packed
    bytes with the range tensors replaced by their coded form, ``chunks`` the chunks per range, ``full_chunks`` the
    full ones, and ``chunk_ratio_min`` / ``chunk_ratio_max`` the per-chunk coded-over-bit-packed ratios over the full
    chunks only (``null`` when there is none). A coded tier (``via`` ``"file"``) takes ``coded_bytes`` from the
    file's stored bytes, its own ``chunk_walkers``, and ``null`` range and chunk numbers."""
    path = os.fspath(path)
    plan = plan_of_pack(path)
    arrays, hdr = read_safetensors(path)
    meta = json.loads(hdr["rpk"])
    n_walkers, windows = plan["pack"]["n_walkers"], plan["pack"]["windows"]
    cx = meta.get("compression") or {}
    if "K" not in cx:
        raise ValueError("the pack's metadata has no 'compression.K'")
    cw = int(chunk_walkers)
    entry = {"id": plan["pack"]["id"], "file": os.path.basename(path), "sha256": sha256_of(path),
             "n_walkers": n_walkers, "windows": windows, "K": int(cx["K"]), "chunk_walkers": cw, "tiers": {}}
    for tier, tv in plan["tiers"].items():
        if not tv["banded"]:
            continue
        bases = _TIER_BASES[tier]
        axes = len(bases)
        channels = tv["channels"] or 1
        ranges = _tier_ranges(meta, tier)
        row_bytes = packed_row_bytes(ranges)
        bit_packed = tv["bit_packed"]["value"]
        if tv["codec"] is None:
            range_bit_packed = 0
            range_coded = 0
            chunk_coded = {}
            for w in range(windows):
                prefix = "" if w == 0 else f"s{w}/"
                for i in range(len(ranges)):
                    r, _n_ch, width = _codec._range_entry(meta, bases[0], i)
                    bits = int(r["bits"])
                    for base in bases:
                        name = f"{prefix}{range_name(base, i, bits)}"
                        if name not in arrays:
                            raise ValueError(f"the pack has no tensor {name!r} for its declared range")
                        t = arrays[name]
                        range_bit_packed += int(t.nbytes)
                        if tier == "field":  # the field's axes swapped as encode_ranges swaps them
                            q = np.swapaxes(_load_range(np.swapaxes(t, 1, 2), bits, width), 1, 2)
                        else:
                            q = _load_range(t, bits, width)
                        stream, chunks = _codec.encode_chunks(np.asarray(q), bits, cw, level=LEVEL)
                        range_coded += int(stream.nbytes) + int(chunks.nbytes)
                        for c in range(len(chunks) - 1):
                            chunk_coded[c] = chunk_coded.get(c, 0) + int(chunks[c + 1] - chunks[c])
            coded = bit_packed - range_bit_packed + range_coded
            chunks_n = -(-n_walkers // cw)
            full = n_walkers // cw
            denom = cw * axes * channels * row_bytes
            ratios = [round(chunk_coded[c] / denom, 6) for c in range(full)]
            entry["tiers"][tier] = {"bands": tv["bands"], "bit_packed_bytes": bit_packed, "coded_bytes": coded,
                                    "ratio": round(coded / bit_packed, 6), "via": "re-encoded",
                                    "range_bit_packed_bytes": range_bit_packed, "range_coded_bytes": range_coded,
                                    "range_ratio": round(range_coded / range_bit_packed, 6),
                                    "chunks": chunks_n, "full_chunks": full,
                                    "chunk_ratio_min": min(ratios) if ratios else None,
                                    "chunk_ratio_max": max(ratios) if ratios else None}
        else:
            entry["chunk_walkers"] = int(ranges[0]["chunk_walkers"])
            stored = tv["stored"]["value"]
            entry["tiers"][tier] = {"bands": tv["bands"], "bit_packed_bytes": bit_packed, "coded_bytes": stored,
                                    "ratio": round(stored / bit_packed, 6), "via": "file",
                                    "range_bit_packed_bytes": None, "range_coded_bytes": None,
                                    "range_ratio": None, "chunks": None, "full_chunks": None,
                                    "chunk_ratio_min": None, "chunk_ratio_max": None}
    return entry


def summarise(packs):
    """Per tier, over the packs that have the tier and at least one full chunk of their own ``chunk_walkers``:
    ``{"min", "max", "packs"}`` of the tier's ``ratio``; tiers with no such pack are absent."""
    ratios = {}
    for p in packs:
        if p["n_walkers"] < p["chunk_walkers"]:
            continue
        for tier, t in p["tiers"].items():
            ratios.setdefault(tier, []).append(t["ratio"])
    return {tier: {"min": min(v), "max": max(v), "packs": len(v)} for tier, v in sorted(ratios.items())}


def build_record(paths, chunk_walkers=4096):
    """The record ``codec_ratio_record/1`` of the packs at ``paths``: every pack measured at ``chunk_walkers``,
    listed once per sha256, sorted by ``(id, sha256)``, with the ``summary``."""
    packs, seen = [], set()
    for p in paths:
        entry = measure_pack(p, chunk_walkers=chunk_walkers)
        if entry["sha256"] in seen:
            continue
        seen.add(entry["sha256"])
        packs.append(entry)
    packs.sort(key=lambda e: (e["id"] or "", e["sha256"]))
    return {"schema": SCHEMA, "codec": CODEC, "level": LEVEL, "tolerance": dict(TOLERANCE),
            "packs": packs, "summary": summarise(packs)}


def _compare_entry(entry, measured, tol):
    """The problem sentences for a record entry against its re-measurement: every field but the coded ones equal
    exactly, the coded ones (a compressed size or a ratio derived from one) within ``tol`` relative."""
    problems = []
    for f in _EXACT_PACK_FIELDS:
        if entry.get(f) != measured.get(f):
            problems.append(f"{entry['id']}: {f} {entry.get(f)!r} != {measured.get(f)!r}")
    if set(entry["tiers"]) != set(measured["tiers"]):
        problems.append(f"{entry['id']}: tiers {sorted(entry['tiers'])} != {sorted(measured['tiers'])}")
    for tier in sorted(set(entry["tiers"]) & set(measured["tiers"])):
        a, b = entry["tiers"][tier], measured["tiers"][tier]
        for f in _EXACT_TIER_FIELDS:
            if a.get(f) != b.get(f):
                problems.append(f"{entry['id']}: tier {tier}: {f} {a.get(f)!r} != {b.get(f)!r}")
        for f in _CODED_TIER_FIELDS:
            x, y = a.get(f), b.get(f)
            if x is None or y is None:
                if x is not y:
                    problems.append(f"{entry['id']}: tier {tier}: {f} {x!r} != {y!r}")
            elif abs(x - y) > tol * abs(y):
                problems.append(f"{entry['id']}: tier {tier}: {f} {x!r} != {y!r}")
    return problems


def check_record(record, search_dirs):
    """Regenerate every entry of ``record`` in memory -- each entry's file looked up in the directories of
    ``search_dirs`` in order -- and compare: ``(problems, checked, skipped)``. A present, matching pack prints
    ``OK <id>``; a present pack with another sha256 is a problem; an absent one prints
    ``SKIP <id>: <file> not found`` and does not fail."""
    problems, checked, skipped = [], 0, 0
    tol = float((record.get("tolerance") or {}).get("coded_relative", TOLERANCE["coded_relative"]))
    for entry in record["packs"]:
        found = None
        for d in search_dirs:
            candidate = Path(d) / entry["file"]
            if candidate.is_file():
                found = candidate
                break
        if found is None:
            print(f"SKIP {entry['id']}: {entry['file']} not found")
            skipped += 1
            continue
        digest = sha256_of(found)
        if digest != entry["sha256"]:
            sentence = f"pack file changed: {entry['id']} ({entry['file']})"
            problems.append(sentence)
            print(sentence)
            continue
        measured = measure_pack(found, entry.get("chunk_walkers", DEFAULT_CHUNK_WALKERS))
        entry_problems = _compare_entry(entry, measured, tol)
        if entry_problems:
            problems.extend(entry_problems)
            for s in entry_problems:
                print(s)
        else:
            checked += 1
            print(f"OK {entry['id']}")
    print(f"checked {checked}, skipped {skipped}")
    return problems, checked, skipped


def main(argv=None):
    """The command line: write a record of the packs given, or check the record. Returns the exit code."""
    ap = argparse.ArgumentParser(description="measure, write and check the codec-ratio record")
    ap.add_argument("packs", nargs="*", metavar="PACK", help="a pack file to measure (write mode)")
    ap.add_argument("--out", metavar="RECORD", help="write the record to this path (write mode only; no default)")
    ap.add_argument("--chunk-walkers", type=int, default=DEFAULT_CHUNK_WALKERS,
                    help=f"the chunk size of the re-encode (default {DEFAULT_CHUNK_WALKERS})")
    ap.add_argument("--check", action="store_true", help="check the record instead of writing one")
    ap.add_argument("--record", default=str(RECORD_PATH), help=f"the record to check (default: {RECORD_PATH})")
    ap.add_argument("--packs-dir", action="append", default=None, metavar="DIR",
                    help="a directory to look a record entry's file up in, repeatable; given at all, it replaces "
                         f"the default {DEFAULT_PACKS_DIR}")
    args = ap.parse_args(argv)
    if args.check:
        if args.packs or args.out:
            ap.error("--check takes no PACK and no --out")
        with open(args.record) as f:
            record = json.load(f)
        dirs = [Path(d) for d in args.packs_dir] if args.packs_dir else [DEFAULT_PACKS_DIR]
        problems, _checked, _skipped = check_record(record, dirs)
        return 1 if problems else 0
    if not args.packs:
        ap.error("give PACK files with --out, or --check")
    if not args.out:
        ap.error("--out is required to write a record")
    record = build_record(args.packs, chunk_walkers=args.chunk_walkers)
    with open(args.out, "w") as f:
        f.write(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}: {len(record['packs'])} pack(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
