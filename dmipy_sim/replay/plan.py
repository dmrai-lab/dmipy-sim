"""The plan of a built pack: per tier its tensors and the three byte counts of its stages.

``plan_of_pack(path)`` reads a pack file's header only -- no tensor is read -- and returns, per tier of the writer's
tier table, the tensors, the band count and the raw, bit-packed and stored byte counts, each with its source, so the
stages of a pack are read from one place. The bit-packed count is the closed form of the header's containers through
:func:`~dmipy_sim.replay.compression.packed_row_bytes`, the same function ``pack_bits`` writes with; the stored
count is the sum of the file's own tensor byte ranges; the raw count is the float container's coefficient bytes.
``plan_problems(plan)`` lists, one sentence per fault, what a plan cannot account for.
"""
from __future__ import annotations

import json
import os
import re

from dmipy_replay.replay import codec as _codec  # private names of the reader's codec: _FIXED, _STREAM, _field_channels
from dmipy_replay.replay.container import parse_header

from . import compression as _cx

#: the writer's tier table: a tensor's base name (its name without a leading ``s<i>/``) starts with one of these
#: prefixes, in this order, and belongs to that tier; a base name matching no prefix is in the tier ``other``
_TIER_PREFIXES = (("pos_", "positions"), ("blt_", "contact"), ("susc_path", "field"), ("comp_", "occupancy"),
                  ("bound_", "mt"), ("spin_weights", "weights"), ("voxel_", "voxel"), ("band_block", "band_block"))

#: the tiers whose band container the metadata names (the banded kinds)
_RANGE_TIERS = ("positions", "contact", "field")

_WINDOW_PREFIX = re.compile(r"^s\d+/")

#: the sources a plan's figures carry
SOURCES = ("header:data_offsets", "header:containers", "derived:float32_count", "stored:not_banded",
           "none:not_banded", "none:float_container")


def _is_range_tensor(name):
    """Whether ``name`` is a band range tensor of the fixed-width or the coded container, or a coded range's chunk
    table: the name patterns of the reader's codec."""
    return bool(_codec._FIXED.match(name) or _codec._STREAM.match(name)
                or (name.endswith("_chunks") and _codec._STREAM.match(name[:-len("_chunks")])))


def _meta_key(meta, dotted):
    """The value at ``dotted`` (``"walk_params.n_walkers"``) of a pack's metadata; a missing key is refused naming it."""
    node = meta
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"the pack's metadata has no {dotted!r}")
        node = node[part]
    return node


def _tier_ranges(meta, tier):
    """The band ranges the metadata declares for ``tier``: ``compression.container`` (positions),
    ``compression.channels.boundary_local_time.container`` (contact),
    ``compression.channels.susc_field_basis.container`` (field); ``None`` for the other tiers."""
    cx = meta.get("compression") or {}
    ch = cx.get("channels") or {}
    if tier == "positions":
        return cx.get("container")
    if tier == "contact":
        return (ch.get("boundary_local_time") or {}).get("container")
    if tier == "field":
        return (ch.get("susc_field_basis") or {}).get("container")
    return None


def plan_of_pack(path):
    """The plan of the pack file at ``path`` (a ``str`` or ``os.PathLike``), schema ``plan_of_pack/1``: a plain dict

    ``{"schema", "file": {"name", "bytes", "header_bytes", "tensor_bytes", "unaccounted_bytes"},
      "pack": {"id", "n_walkers", "windows"},
      "tiers": {<tier>: {"tensors": [{"name", "dtype", "shape", "bytes"}], "banded", "bands", "channels", "codec",
                        "raw": {"value", "source"}, "bit_packed": {"value", "source"},
                        "stored": {"value", "source"}}}}``

    with only the tiers whose tensors are in the file, in the order of the writer's tier table then ``other``. The
    file's header is read (the first 8 bytes, then the header's length, through
    :func:`~dmipy_replay.replay.container.parse_header`) and the pack's metadata under the ``"rpk"`` key, plus
    :func:`os.path.getsize`; no tensor is read. ``stored`` is the sum of the tier's tensors' ``data_offsets`` ranges
    (source ``header:data_offsets``); ``bit_packed`` of a banded tier is ``W * N * axes * C`` times
    :func:`~dmipy_sim.replay.compression.packed_row_bytes` of its ranges plus the stored bytes of its other tensors
    (source ``header:containers``), ``None`` with source ``none:float_container`` or ``none:not_banded`` otherwise;
    ``raw`` is the float container's coefficient bytes (source ``derived:float32_count``) for the banded kinds and
    the stored bytes (source ``stored:not_banded``) for every other tier. Metadata the plan needs and does not find
    (``walk_params.n_walkers``, ``compression.K``, a channel's ``K``) is refused naming the key; a range tensor whose
    banded-kind tier declares no container is refused naming the tier. A window ``i >= 1`` is read with window 0's
    containers."""
    path = os.fspath(path)
    with open(path, "rb") as f:
        first = f.read(8)
        n = int.from_bytes(first, "little")
        head = first + f.read(n)
    n, header = parse_header(head)
    md = header.get("__metadata__") or {}
    if "rpk" not in md:
        raise ValueError(f"{os.path.basename(path)}: the header's __metadata__ has no 'rpk' key")
    meta = json.loads(md["rpk"])
    size = os.path.getsize(path)

    wp = meta.get("walk_params")
    if not isinstance(wp, dict) or "n_walkers" not in wp:
        raise ValueError("the pack's metadata has no 'walk_params.n_walkers'")
    n_walkers = int(wp["n_walkers"])
    win = wp.get("windows")
    windows = int(win["n"]) if isinstance(win, dict) and "n" in win else 1

    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    groups = {}
    for name in sorted(tensors):
        base = _WINDOW_PREFIX.sub("", name)
        for prefix, tier in _TIER_PREFIXES:
            if base.startswith(prefix):
                groups.setdefault(tier, []).append(name)
                break
        else:
            groups.setdefault("other", []).append(name)
    ordered = [tier for _, tier in _TIER_PREFIXES if tier in groups]
    if "other" in groups and "other" not in ordered:
        ordered.append("other")

    tiers = {}
    for tier in ordered:
        entries = []
        for name in groups[tier]:
            a, b = (int(x) for x in tensors[name]["data_offsets"])
            entries.append({"name": name, "dtype": tensors[name]["dtype"],
                            "shape": [int(s) for s in tensors[name]["shape"]], "bytes": b - a,
                            "_range": _is_range_tensor(name)})
        entries.sort(key=lambda e: e["name"])
        ranges = _tier_ranges(meta, tier)
        has_range_tensor = any(e["_range"] for e in entries)
        if has_range_tensor and tier in _RANGE_TIERS and not ranges:
            raise ValueError(f"tier {tier!r} holds range tensors but the metadata declares no container for it")
        banded = bool(ranges) and has_range_tensor
        if tier == "positions":
            K = int(_meta_key(meta, "compression.K"))
            channels, axes = None, 3
        elif tier == "contact":
            K = int(_meta_key(meta, "compression.channels.boundary_local_time.K"))
            channels, axes = None, 1
        elif tier == "field":
            K = int(_meta_key(meta, "compression.channels.susc_field_basis.K"))
            channels, axes = int(_codec._field_channels(_meta_key(meta, "compression.channels.susc_field_basis"))), 1
        else:
            K = channels = None
            axes = 1
        C = channels if channels is not None else 1
        stored = sum(e["bytes"] for e in entries)
        other = sum(e["bytes"] for e in entries if not e["_range"])
        if banded:
            bit_packed = windows * n_walkers * axes * C * _cx.packed_row_bytes(ranges) + other
            bp_source, codec_name = "header:containers", ranges[0].get("codec")
        else:
            bit_packed = None
            bp_source = "none:float_container" if tier in _RANGE_TIERS else "none:not_banded"
            codec_name = None
        if tier == "positions":
            raw, raw_source = 4 * windows * n_walkers * 3 * (K + 2), "derived:float32_count"
        elif tier == "contact":
            raw, raw_source = 4 * windows * n_walkers * (K + 2), "derived:float32_count"
        elif tier == "field":
            raw, raw_source = 4 * windows * n_walkers * C * K, "derived:float32_count"
        else:
            raw, raw_source = stored, "stored:not_banded"
        tiers[tier] = {
            "tensors": [{k: e[k] for k in ("name", "dtype", "shape", "bytes")} for e in entries],
            "banded": banded, "bands": K, "channels": channels, "codec": codec_name,
            "raw": {"value": raw, "source": raw_source},
            "bit_packed": {"value": bit_packed, "source": bp_source},
            "stored": {"value": stored, "source": "header:data_offsets"},
        }

    tensor_bytes = sum(int(tensors[name]["data_offsets"][1]) - int(tensors[name]["data_offsets"][0])
                       for name in tensors)
    return {
        "schema": "plan_of_pack/1",
        "file": {"name": os.path.basename(path), "bytes": size, "header_bytes": 8 + n,
                 "tensor_bytes": tensor_bytes, "unaccounted_bytes": size - (8 + n) - tensor_bytes},
        "pack": {"id": meta.get("id"), "n_walkers": n_walkers, "windows": windows},
        "tiers": tiers,
    }


def plan_problems(plan):
    """One sentence per fault found in ``plan`` (empty: sound): the file's unaccounted bytes, a tensor in the tier
    ``other`` the writer's tier table does not name, and a banded tier stored uncoded whose bit-packed bytes differ
    from its stored bytes."""
    problems = []
    unaccounted = plan["file"]["unaccounted_bytes"]
    if unaccounted != 0:
        problems.append(f"{unaccounted} bytes of the file are unaccounted for by the tiers and the header")
    tiers = plan["tiers"]
    if "other" in tiers:
        names = ", ".join(t["name"] for t in tiers["other"]["tensors"])
        problems.append(f"tier 'other' holds tensors the writer's tier table does not name: {names}")
    for name, tier in tiers.items():
        if name != "other" and tier["banded"] and tier["codec"] is None \
                and tier["bit_packed"]["value"] != tier["stored"]["value"]:
            problems.append(f"tier {name}: bit-packed bytes {tier['bit_packed']['value']} "
                            f"differ from stored bytes {tier['stored']['value']}")
    return problems
