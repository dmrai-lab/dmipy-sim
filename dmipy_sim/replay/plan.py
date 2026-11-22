"""The plan of a built pack: per tier its tensors and the three byte counts of its stages.

``plan_of_pack(path)`` reads a pack file's header only -- no tensor is read -- and returns, per tier of the writer's
tier table, the tensors, the band count and the raw, bit-packed and stored byte counts, each with its source, so the
stages of a pack are read from one place. The bit-packed count is the closed form of the header's containers through
:func:`~dmipy_sim.replay.compression.packed_row_bytes`, the same function ``pack_bits`` writes with; the stored
count is the sum of the file's own tensor byte ranges; the raw count is the float container's coefficient bytes.
``plan_problems(plan)`` lists, one sentence per fault, what a plan cannot account for.

``pack_plan(spec, envelope, tolerance, n_walkers, ...)`` is the pre-walk plan of a pack, schema ``pack_plan/1``: before
any walk it returns, per tier that the stated rules switch on, the band count and the bytes at the raw, bit-packed,
coded and stored stages, every figure built by :func:`_fig` and labelled ``derived``, ``measured`` or ``proxy``
with its source and a range where it is a proxy. ``compare_plans`` compares a plan with a built pack's
:func:`plan_of_pack` key by key; ``pack_plan_problems`` lists the faults of a plan itself.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass

import numpy as np
from dmipy_replay.replay import codec as _codec  # private names of the reader's codec: _FIXED, _STREAM, _field_channels
from dmipy_replay.replay.container import parse_header
from dmipy_replay.replay.storage import mode_count

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


def tier_ranges(meta, tier):
    """The band ranges the pack's metadata declares for ``tier``: ``compression.container`` (positions),
    ``compression.channels.boundary_local_time.container`` (contact),
    ``compression.channels.susc_field_basis.container`` (field); ``None`` for the other tiers. The one lookup of a
    tier's container, used by :func:`plan_of_pack` and by the tools that read a pack's ranges."""
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
        ranges = tier_ranges(meta, tier)
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


# ---------------------------------------------------------------- the pre-walk plan of a pack (pack_plan/1)

#: the bits per band a banded tier's container is planned at, as a range: the clip :func:`~dmipy_sim.replay.compression
#: .derive_container` applies (``b_min`` to ``b_max = 16``); a test replaces the pair
BITS_RANGE = (_cx.MIN_BITS, 16)

#: the stand-in upper bound on the events per walker of an occupancy column: four times the engine's event-capacity
#: floor (:func:`~dmipy_sim.engine.backends.event_capacity`, ``floor=16``)
EVENTS_PER_WALKER_HI = 64

#: the labels a figure's provenance carries
PROVENANCE = ("derived", "measured", "proxy")

#: the keys of a figure
_FIG_KEYS = frozenset(("value", "lo", "hi", "provenance", "source"))

#: the keys of a tier's dictionary of a pre-walk plan
_TIER_KEYS = ("why", "banded", "bands", "channels", "codec", "bits", "eps", "events",
              "raw", "bit_packed", "coded", "stored")

#: the stages of a plan
_STAGES = ("raw", "bit_packed", "coded", "stored")


@dataclass(frozen=True)
class PlanEnvelope:
    """The maxima a pack is planned for; ``None`` states that no such term is planned.

    ``band_hz`` (Hz) is the temporal band, ``window_s`` (s) the length of one window and ``n_windows`` (>= 1) the
    number of windows of the pack, so the band count is ``mode_count(window_s, band_hz)``. ``G_max`` (T/m) and
    ``slew_max`` (T/m/s) are recorded and read by no rule of version 1. ``B0_max`` (T) is the one maximum field
    strength; the field tier is planned when a pool of the spec has a susceptibility and it is not 0. ``chi_max`` is
    the largest susceptibility scale; its absence refuses a planned field tier. ``rho2_max`` (m/s) is the largest
    surface relaxivity; positive switches the contact tier on. ``max_refocusing_pulses`` bounds the field band's
    stand-in range from below.
    """
    band_hz: float
    window_s: float
    n_windows: int
    G_max: float | None = None
    slew_max: float | None = None
    B0_max: float | None = None
    chi_max: float | None = None
    rho2_max: float | None = None
    max_refocusing_pulses: int | None = None

    def __post_init__(self):
        for key, positive in (("band_hz", True), ("window_s", True)):
            v = getattr(self, key)
            if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) \
                    or not math.isfinite(float(v)) or (positive and float(v) <= 0.0):
                _refuse(key, f"a finite {'positive ' if positive else ''}number; got {v!r}")
            object.__setattr__(self, key, float(v))
        n = self.n_windows
        if n is None or isinstance(n, bool) or not isinstance(n, int) or n < 1:
            _refuse("n_windows", f"an integer >= 1; got {n!r}")
        object.__setattr__(self, "n_windows", int(n))
        for key in ("G_max", "slew_max", "B0_max", "chi_max", "rho2_max"):
            v = getattr(self, key)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v))):
                _refuse(key, f"a finite number or None; got {v!r}")
            object.__setattr__(self, key, None if v is None else float(v))
        p = self.max_refocusing_pulses
        if p is not None and (isinstance(p, bool) or not isinstance(p, int) or p < 0):
            _refuse("max_refocusing_pulses", f"a non-negative integer or None; got {p!r}")
        object.__setattr__(self, "max_refocusing_pulses", None if p is None else int(p))

    @classmethod
    def from_dict(cls, d):
        """The :class:`PlanEnvelope` of the dict ``d`` with these fields' names as keys. A missing required key
        (``band_hz``, ``window_s``, ``n_windows``) or an unknown key is refused by name through :func:`_refuse`."""
        if not isinstance(d, dict):
            _refuse("envelope", f"a dict; got {type(d).__name__}")
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        for key in d:
            if key not in fields:
                _refuse(key, "an unknown key of the envelope dict")
        for key in ("band_hz", "window_s", "n_windows"):
            if key not in d:
                _refuse(key, "a required key the envelope dict lacks")
        return cls(**d)

    def to_dict(self):
        """The plain dict of this envelope's fields, JSON-ready."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _plain(x):
    """``x`` as a Python ``int``, ``float``, ``bool`` or ``str`` (NumPy scalars converted), or ``None``."""
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def _fig(value=None, lo=None, hi=None, *, provenance, source):
    """One figure of a plan: ``{"value", "lo", "hi", "provenance", "source"}`` with Python ``int``/``float``/``None``
    entries. A derived figure has ``value == lo == hi``; a proxy or measured figure has ``value`` ``None`` and a
    range ``lo <= hi``; a not-applicable figure has all three ``None`` with provenance ``derived`` and a source
    beginning ``none:``."""
    return {"value": _plain(value), "lo": _plain(lo), "hi": _plain(hi), "provenance": provenance, "source": source}


def _refuse(name, why):
    """The one function every refusal of :func:`pack_plan` calls: a ``ValueError`` naming the input."""
    raise ValueError(f"pack_plan: {name}: {why}")


def _not_banded():
    """The not-applicable figure of a tier the bit-packed container does not carry."""
    return _fig(None, None, None, provenance="derived", source="none:not_banded")


def _banded_tail(codec, bit_packed):
    """The ``(coded, stored)`` figures of a banded tier: with no codec the coded stage is not applicable and the
    stored stage is the bit-packed one; with a codec the coded size is not planned in this version, so both figures
    are unknown before the walk (``value``, ``lo`` and ``hi`` all ``None``) and the stored stage is the coded one."""
    if codec is None:
        return _fig(None, None, None, provenance="derived", source="none:no_codec"), bit_packed
    coded = _fig(None, None, None, provenance="proxy",
                 source="the coded size is not planned in this version: unknown before the walk")
    return coded, coded


def _bits_fig():
    """The bits-per-band figure every banded tier carries: the clip of :func:`~dmipy_sim.replay.compression
    .derive_container`, as a proxy range."""
    return _fig(None, BITS_RANGE[0], BITS_RANGE[1], provenance="proxy",
                source=f"compression.derive_container clips the bits per band to [{BITS_RANGE[0]}, {BITS_RANGE[1]}]")


def _stage_figure(tier, stage):
    """The figure a tier contributes to the total at ``stage``: its own for ``raw`` and ``stored``, and for
    ``bit_packed`` and ``coded`` its own when it is banded and its ``raw`` otherwise."""
    if stage in ("raw", "stored"):
        return tier.get(stage)
    return tier.get(stage) if tier.get("banded") else tier.get("raw")


def _validate_provider_tier(provider, tier_name, tier, taken):
    """Whether a provider's tier ``tier_name`` (``tier``) may enter the plan beside the tiers already ``taken``:
    a ``ValueError`` naming the provider and the tier when the name is already used, the tier lacks ``why``, ``raw``,
    ``bit_packed``, ``coded`` or ``stored``, a figure is not the five-key shape, holds a NumPy scalar, a provenance is
    outside :data:`PROVENANCE`, or the tier is not JSON."""
    def bad(why):
        raise ValueError(f"pack_plan: provider {provider}: tier {tier_name!r}: {why}")

    if tier_name in taken:
        bad("the tier name is already used")
    if not isinstance(tier, dict):
        bad(f"a tier is a dict; got {type(tier).__name__}")
    for key in ("why", "raw", "bit_packed", "coded", "stored"):
        if key not in tier:
            bad(f"lacks the key {key!r}")
    if not isinstance(tier["why"], str) or not tier["why"]:
        bad("'why' is a non-empty string")
    for key in ("raw", "bit_packed", "coded", "stored"):
        fig = tier[key]
        if not isinstance(fig, dict) or set(fig) != _FIG_KEYS:
            bad(f"{key!r} is not a five-key figure")
        if fig["provenance"] not in PROVENANCE:
            bad(f"{key!r} carries provenance {fig['provenance']!r}, not one of {', '.join(PROVENANCE)}")
        for slot in ("value", "lo", "hi"):
            if isinstance(fig[slot], np.generic):
                bad(f"{key!r} holds a NumPy scalar in {slot!r}: {type(fig[slot]).__name__}")
    try:
        json.dumps(tier, allow_nan=False)
    except (TypeError, ValueError) as e:
        bad(f"is not JSON: {e}")


def _wall_rate(wall, group, key):
    """The rate ``key`` of a wall's ``group`` (``permeability``, ``mt_reactivity``), 0 when it states none."""
    rates = wall.get(group) if isinstance(wall, dict) else None
    v = rates.get(key) if isinstance(rates, dict) else None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def _events_figures(column, pools, N, W, bank_shared):
    """The ``(events, raw)`` figures of one events column (``comp``, ``bound``) of ``N`` walkers over ``W`` windows:
    the entries per window as a stand-in proxy range, and the raw bytes of the column's ``<c>_events_*`` tensors
    over that range. ``bank_shared`` asks whether the column's tensors are shared across windows."""
    n = 1 if bank_shared(f"{column}_events_offsets") else W
    label = 1 if len(pools) <= 128 else 2
    e_lo, e_hi = N, N * (1 + EVENTS_PER_WALKER_HI)
    events = _fig(None, e_lo, e_hi, provenance="proxy",
                  source=f"stand-in: the crossings per walker of column {column!r} are not bounded before the walk; "
                         f"[{e_lo}, {e_hi}] entries per window")
    raw_lo = n * (8 * (N + 1) + e_lo * (8 + label))
    raw_hi = n * (8 * (N + 1) + e_hi * (8 + label))
    raw = _fig(None, raw_lo, raw_hi, provenance="proxy",
               source=f"compression.encode_events_occupancy: {column}_events_offsets int64 (N+1,), "
                      f"{column}_events_t float64 (E,), {column}_events_label int{8 * label} (E,) per window")
    return events, raw


def pack_plan(spec, envelope, tolerance, n_walkers, *, codec="bitshuffle-zlib", chunk_walkers=4096, weights=False,
              n_voxels=None, field_bands=None, providers=()):
    """The pre-walk plan of a pack, schema ``pack_plan/1``: a plain dict

    ``{"schema", "inputs": {"spec": {"id", "sha256"}, "envelope", "tolerance", "n_walkers", "codec", "chunk_walkers",
      "weights", "n_voxels", "field_bands", "providers", "ratios", "unread"}, "pack": {"n_walkers", "windows"},
      "tiers": {<tier>: {"why", "banded", "bands", "channels", "codec", "bits", "eps", "events", "raw", "bit_packed",
      "coded", "stored"}}}, "total": {"raw", "bit_packed", "coded", "stored"}}``

    returned before any walk. ``spec`` is a dict (a pack's ``substrate`` metadata) or an object with ``to_dict()``;
    only its ``pools`` (each with a ``susceptibility`` that is ``None`` or a dict of susceptibility scales) and its
    ``walls`` (each with ``permeability`` and ``mt_reactivity``) are read. ``envelope`` is a :class:`PlanEnvelope` or a
    dict for :meth:`PlanEnvelope.from_dict`: maxima only. ``tolerance`` is one absolute error (finite, > 0), recorded
    and read by no rule of version 1. ``n_walkers`` is an integer >= 1. ``codec`` is ``None`` or ``"bitshuffle-zlib"``;
    ``chunk_walkers`` a power of two. ``weights`` says the pack carries ``spin_weights``; ``n_voxels`` (``None`` or an
    integer >= 1) that it carries a voxel certificate; ``field_bands`` (``None`` or an integer >= 1) a stated field
    band count. ``providers`` is a sequence of callables, each called with a deep copy of ``inputs`` and returning
    ``{tier: tier}`` added after the simulator's own tiers.

    The tiers appear only when the stated rules switch them on, in the order positions, contact, field, occupancy,
    mt, weights, voxel, then the providers'. Every figure is built by :func:`_fig` and carries one of
    :data:`PROVENANCE` with a non-empty source; the band count of positions and of the contact is
    ``mode_count(window_s, band_hz)``, the bit-packed rows are
    :func:`~dmipy_sim.replay.compression.packed_row_bytes` at the ends of :data:`BITS_RANGE`, and whether a tensor is
    shared across windows is asked of ``bank._is_shared_key`` per tensor name. Only a missing input is refused, by
    name, through :func:`_refuse`. The plan is JSON and equal to its own round trip."""
    from . import bank as _bank

    spec_dict = spec.to_dict() if hasattr(spec, "to_dict") else spec
    if not isinstance(spec_dict, dict):
        _refuse("spec", f"a dict of the spec's to_dict() or an object with to_dict(); got {type(spec).__name__}")
    if isinstance(envelope, PlanEnvelope):
        env = envelope
    elif isinstance(envelope, dict):
        env = PlanEnvelope.from_dict(envelope)
    else:
        _refuse("envelope", f"a PlanEnvelope or a dict for PlanEnvelope.from_dict; got {type(envelope).__name__}")
    if tolerance is None or isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) \
            or not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        _refuse("tolerance", f"one absolute error, finite and > 0; got {tolerance!r}")
    tol = float(tolerance)
    if isinstance(n_walkers, bool) or not isinstance(n_walkers, (int, float)) \
            or not math.isfinite(n_walkers) or n_walkers < 1 or int(n_walkers) != n_walkers:
        _refuse("n_walkers", f"an integer >= 1; got {n_walkers!r}")
    N = int(n_walkers)
    if codec is not None and codec != "bitshuffle-zlib":
        _refuse("codec", f"None or 'bitshuffle-zlib'; got {codec!r}")
    if isinstance(chunk_walkers, bool) or not isinstance(chunk_walkers, int) \
            or chunk_walkers < 1 or chunk_walkers & (chunk_walkers - 1):
        _refuse("chunk_walkers", f"a power of two; got {chunk_walkers!r}")
    cw = int(chunk_walkers)
    if not isinstance(weights, bool):
        _refuse("weights", f"a bool; got {weights!r}")
    if n_voxels is not None and (isinstance(n_voxels, bool) or not isinstance(n_voxels, int) or n_voxels < 1):
        _refuse("n_voxels", f"None or an integer >= 1; got {n_voxels!r}")
    if field_bands is not None and (isinstance(field_bands, bool) or not isinstance(field_bands, int) or field_bands < 1):
        _refuse("field_bands", f"None or an integer >= 1; got {field_bands!r}")
    if "pools" not in spec_dict:
        _refuse("pools", "the spec states no pools")
    if "walls" not in spec_dict:
        _refuse("walls", "the spec states no walls")
    pools = spec_dict["pools"]
    walls = spec_dict["walls"]

    W = env.n_windows
    tiers = {}

    # positions: always written; K = mode_count(window_s, band_hz)
    K = mode_count(env.window_s, env.band_hz)
    n_pos = 1 if _bank._is_shared_key("pos_band_scale") else W
    row_lo = _cx.packed_row_bytes([{"bands": [0, K], "bits": BITS_RANGE[0]}])
    row_hi = _cx.packed_row_bytes([{"bands": [0, K], "bits": BITS_RANGE[1]}])
    pos_raw = 4 * n_pos * N * 3 * (K + 2)
    pos_bp = _fig(None, n_pos * (N * 3 * row_lo + 24 * N + 12 * K), n_pos * (N * 3 * row_hi + 24 * N + 12 * K),
                  provenance="proxy",
                  source="compression.packed_row_bytes of one walker's row at the ends of BITS_RANGE, plus the three "
                         "pos_*_ends float32 (N, 2) and the pos_band_scale float32 (1, 3, K) per window")
    pos_coded, pos_stored = _banded_tail(codec, pos_bp)
    tiers["positions"] = {
        "why": "always written",
        "banded": True,
        "bands": _fig(K, K, K, provenance="derived",
                      source="dmipy_replay.replay.storage.mode_count(window_s, band_hz)"),
        "channels": None,
        "codec": codec,
        "bits": _bits_fig(),
        "eps": None,
        "events": None,
        "raw": _fig(pos_raw, pos_raw, pos_raw, provenance="derived",
                    source="plan_of_pack: positions raw 4*W*N*3*(K+2) float32"),
        "bit_packed": pos_bp,
        "coded": pos_coded,
        "stored": pos_stored,
    }

    # contact: the envelope certifies a contact term
    if env.rho2_max is not None and env.rho2_max > 0.0:
        Kb = K
        n_c = 1 if _bank._is_shared_key("blt_band_scale") else W
        row_lo = _cx.packed_row_bytes([{"bands": [0, Kb], "bits": BITS_RANGE[0]}])
        row_hi = _cx.packed_row_bytes([{"bands": [0, Kb], "bits": BITS_RANGE[1]}])
        c_raw = 4 * n_c * N * (Kb + 2)
        c_bp = _fig(None, n_c * (N * row_lo + 8 * N + 4 * Kb), n_c * (N * row_hi + 8 * N + 4 * Kb),
                    provenance="proxy",
                    source="compression.packed_row_bytes of one walker's row at the ends of BITS_RANGE, plus "
                           "blt_start and blt_endpoint float32 (N,) and the blt_band_scale float32 (1, K_b) per window")
        c_coded, c_stored = _banded_tail(codec, c_bp)
        tiers["contact"] = {
            "why": "envelope.rho2_max > 0",
            "banded": True,
            "bands": _fig(Kb, Kb, Kb, provenance="derived", source="the contact's band count Kb equals K"),
            "channels": None,
            "codec": codec,
            "bits": _bits_fig(),
            "eps": None,
            "events": None,
            "raw": _fig(c_raw, c_raw, c_raw, provenance="derived", source="plan_of_pack: contact raw 4*W*N*(K_b+2) float32"),
            "bit_packed": c_bp,
            "coded": c_coded,
            "stored": c_stored,
        }

    # field: a pool has a susceptibility and the envelope states a field strength
    suscs = [p.get("susceptibility") if isinstance(p, dict) else None for p in pools]
    has_susc = any(s is not None for s in suscs)
    if has_susc:
        if env.B0_max is None:
            _refuse("B0_max", "a pool of the spec has a susceptibility and the envelope states no maximum field strength")
        if env.B0_max != 0:
            if env.chi_max is None:
                _refuse("chi_max", "a pool of the spec has a susceptibility and the envelope states no susceptibility scale")
            C = 12 if any(isinstance(s, dict) and s.get("chi_aniso") is not None and float(s["chi_aniso"]) != 0.0
                          for s in suscs) else 6
            pulses = env.max_refocusing_pulses or 0
            if field_bands is not None:
                kf_lo = kf_hi = int(field_bands)
                bands_fig = _fig(kf_lo, kf_lo, kf_lo, provenance="derived", source="input:field_bands")
                kf_range = False
            else:
                kf_lo = max(1, 2 * int(pulses))
                kf_hi = int(_bank.SUSC_PATH_LADDER[-1])
                bands_fig = _fig(None, kf_lo, kf_hi, provenance="proxy",
                                 source=f"stand-in: the field band count is derived on the walk (bank.derive_field_band "
                                        f"climbs SUSC_PATH_LADDER from min_K = 2*max_refocusing_pulses); stated range "
                                        f"[{kf_lo}, {kf_hi}]")
                kf_range = True
            n_f = 1 if _bank._is_shared_key("susc_path_scale") else W
            f_raw_lo = 4 * n_f * N * C * kf_lo
            f_raw_hi = 4 * n_f * N * C * kf_hi
            if kf_range:
                f_raw = _fig(None, f_raw_lo, f_raw_hi, provenance="proxy",
                             source="plan_of_pack: field raw 4*W*N*C*K_f while K_f is a range")
            else:
                f_raw = _fig(f_raw_lo, f_raw_lo, f_raw_lo, provenance="derived",
                             source="plan_of_pack: field raw 4*W*N*C*K_f")
            rm_lo = _cx.packed_row_bytes([{"modes": [0, kf_lo], "bits": BITS_RANGE[0]}])
            rm_hi = _cx.packed_row_bytes([{"modes": [0, kf_hi], "bits": BITS_RANGE[1]}])
            f_bp = _fig(None, n_f * (N * C * rm_lo + 4 * C * kf_lo), n_f * (N * C * rm_hi + 4 * C * kf_hi),
                        provenance="proxy",
                        source="compression.packed_row_bytes of one walker's row of modes at the ends of BITS_RANGE "
                               "over the ends of K_f's range, plus the susc_path_scale float32 (C, K_f) per window")
            f_coded, f_stored = _banded_tail(codec, f_bp)
            tiers["field"] = {
                "why": "a pool of the spec has a susceptibility and envelope.B0_max is not 0",
                "banded": True,
                "bands": bands_fig,
                "channels": _fig(C, C, C, provenance="derived",
                                 source="dmipy_replay.replay.field.CHANNEL_NAMES (12 with an anisotropic channel, "
                                        "ISO_CHANNEL_NAMES 6 without) chosen by a non-zero chi_aniso"),
                "codec": codec,
                "bits": _bits_fig(),
                "eps": None,
                "events": None,
                "raw": f_raw,
                "bit_packed": f_bp,
                "coded": f_coded,
                "stored": f_stored,
            }

    # occupancy: the walk has a compartment (two pools or more); events iff a wall is permeable
    if len(pools) >= 2:
        permeable = any(_wall_rate(w, "permeability", "in_to_out") > 0.0
                        or _wall_rate(w, "permeability", "out_to_in") > 0.0 for w in walls)
        not_banded = _not_banded()
        if permeable:
            events, raw = _events_figures("comp", pools, N, W, _bank._is_shared_key)
            tiers["occupancy"] = {
                "why": "len(spec['pools']) >= 2, kind events: a wall is permeable",
                "banded": False, "bands": None, "channels": None, "codec": None, "bits": None, "eps": None,
                "events": events, "raw": raw, "bit_packed": not_banded, "coded": _not_banded(), "stored": raw,
            }
        else:
            n_o = 1 if _bank._is_shared_key("comp_static") else W
            raw = _fig(n_o * N, n_o * N, n_o * N, provenance="derived",
                       source="compression.encode_static_occupancy: comp_static int8 (N,), shared across windows "
                              "(bank._is_shared_key('comp_static'))")
            tiers["occupancy"] = {
                "why": "len(spec['pools']) >= 2, kind static: no wall is permeable",
                "banded": False, "bands": None, "channels": None, "codec": None, "bits": None, "eps": None,
                "events": None, "raw": raw, "bit_packed": not_banded, "coded": _not_banded(), "stored": raw,
            }

    # mt: a wall binds; always events, and it rides on the occupancy channel
    if any(_wall_rate(w, "mt_reactivity", "inside") > 0.0 or _wall_rate(w, "mt_reactivity", "outside") > 0.0
           for w in walls):
        if "occupancy" not in tiers:
            _refuse("mt", "the mt tier rides on the occupancy channel, which this spec's pool count does not switch on")
        events, raw = _events_figures("bound", pools, N, W, _bank._is_shared_key)
        tiers["mt"] = {
            "why": "a wall has mt_reactivity > 0",
            "banded": False, "bands": None, "channels": None, "codec": None, "bits": None, "eps": None,
            "events": events, "raw": raw, "bit_packed": _not_banded(), "coded": _not_banded(), "stored": raw,
        }

    # weights: the pack carries spin_weights
    if weights:
        if "occupancy" not in tiers:
            _refuse("weights", "the weights tier rides on the occupancy channel, which this spec's pool count does not "
                               "switch on")
        n_w = 1 if _bank._is_shared_key("spin_weights") else W
        raw = _fig(4 * n_w * N, 4 * n_w * N, 4 * n_w * N, provenance="derived",
                   source="spin_weights float32 (N,), shared across windows (bank._is_shared_key('spin_weights'))")
        tiers["weights"] = {
            "why": "weights=True",
            "banded": False, "bands": None, "channels": None, "codec": None, "bits": None, "eps": None,
            "events": None, "raw": raw, "bit_packed": _not_banded(), "coded": _not_banded(), "stored": raw,
        }

    # voxel: the pack carries a voxel certificate
    if n_voxels is not None:
        V = int(n_voxels)
        n_v = 1 if _bank._is_shared_key("voxel_ijk") else W
        p_lo, p_hi = 1, len(pools)
        raw = _fig(None, n_v * (12 * V + 12 * V * p_lo), n_v * (12 * V + 12 * V * p_hi), provenance="proxy",
                   source=f"voxel_ijk int32 (V, 3) and voxel_certificate float32 (V, P, 3) shared across windows, "
                          f"P in [{p_lo}, {p_hi}]")
        tiers["voxel"] = {
            "why": "n_voxels is not None",
            "banded": False, "bands": None, "channels": None, "codec": None, "bits": None, "eps": None,
            "events": None, "raw": raw, "bit_packed": _not_banded(), "coded": _not_banded(), "stored": raw,
        }

    inputs = {
        "spec": {"id": spec_dict.get("id") or None,
                 "sha256": hashlib.sha256(json.dumps(spec_dict, sort_keys=True, default=str).encode("utf-8")).hexdigest()},
        "envelope": env.to_dict(),
        "tolerance": tol,
        "n_walkers": N,
        "codec": codec,
        "chunk_walkers": cw,
        "weights": bool(weights),
        "n_voxels": None if n_voxels is None else int(n_voxels),
        "field_bands": None if field_bands is None else int(field_bands),
        "providers": [getattr(p, "__name__", type(p).__name__) for p in providers],
        "ratios": None,
        "unread": ["tolerance", "G_max", "slew_max"],
    }

    for prov in providers:
        pname = getattr(prov, "__name__", type(prov).__name__)
        result = prov(copy.deepcopy(inputs))
        if not isinstance(result, dict):
            raise ValueError(f"pack_plan: provider {pname}: returns {{tier: tier}}; got {type(result).__name__}")
        for tname, tier in result.items():
            _validate_provider_tier(pname, tname, tier, set(tiers))
            tiers[tname] = tier

    rank = {"derived": 0, "measured": 1, "proxy": 2}
    total = {}
    for stage in _STAGES:
        figs = [f for f in (_stage_figure(t, stage) for t in tiers.values()) if isinstance(f, dict)]
        vals = [f["value"] for f in figs]
        los = [f["lo"] for f in figs]
        his = [f["hi"] for f in figs]
        value = sum(vals) if figs and all(v is not None for v in vals) else None
        lo = sum(los) if figs and all(v is not None for v in los) else None
        hi = sum(his) if figs and all(v is not None for v in his) else None
        prov = max((f["provenance"] for f in figs), key=lambda p: rank.get(p, len(rank))) if figs else "derived"
        total[stage] = _fig(value, lo, hi, provenance=prov, source="sum_of_tiers")

    return {"schema": "pack_plan/1", "inputs": inputs, "pack": {"n_walkers": N, "windows": W},
            "tiers": tiers, "total": total}


def compare_plans(plan, built):
    """One sentence per fault of comparing the pre-walk ``plan`` with the :func:`plan_of_pack` result ``built`` of a
    built pack (empty: sound): a tier only in one of the two, a key of ``bands``, ``channels``, ``raw``,
    ``bit_packed``, ``stored`` that a planned derived figure pins and the pack does not hold, a planned range the
    pack's value falls outside (an end that is ``None`` is not checked, so a figure whose ``value``, ``lo`` and
    ``hi`` are all ``None`` -- an unknown size -- is not compared and adds no sentence), a value ``None`` in the
    pack against a planned one, a banded tier whose codec differs (its ``stored`` is then not compared), and
    ``pack.n_walkers`` or ``pack.windows`` that differ."""
    problems = []
    pt = plan.get("tiers") or {}
    bt = built.get("tiers") or {}
    for t in list(pt) + [k for k in bt if k not in pt]:
        if t not in bt:
            problems.append(f"tier {t} is in the plan and not in the pack")
            continue
        if t not in pt:
            problems.append(f"tier {t} is in the pack and not in the plan")
            continue
        p, b = pt[t], bt[t]
        codec_differs = bool(p.get("banded")) and p.get("codec") != b.get("codec")
        if codec_differs:
            problems.append(f"tier {t}: codec {b.get('codec')} in the pack, {p.get('codec')} in the plan")
        for key in ("bands", "channels", "raw", "bit_packed", "stored"):
            if key == "stored" and codec_differs:
                continue
            if key not in p or key not in b:
                continue
            fig = p[key]
            x = b[key]
            if isinstance(x, dict):
                x = x.get("value")
            if not isinstance(fig, dict) or "lo" not in fig:
                continue
            v, lo, hi = fig.get("value"), fig.get("lo"), fig.get("hi")
            if v is None and lo is None and hi is None:
                continue
            if x is None:
                planned = v if v is not None else f"[{lo}, {hi}]"
                problems.append(f"tier {t}: {key} is None in the pack and {planned} in the plan")
                continue
            if fig.get("provenance") == "derived" and v is not None:
                if x != v:
                    problems.append(f"tier {t}: {key} {x} differs from the planned {v}")
            elif (lo is not None and x < lo) or (hi is not None and x > hi):
                problems.append(f"tier {t}: {key} {x} is outside the planned [{lo}, {hi}]")
    pp, bp = plan.get("pack") or {}, built.get("pack") or {}
    for key in ("n_walkers", "windows"):
        if pp.get(key) != bp.get(key):
            problems.append(f"pack: {key} {bp.get(key)} differs from the planned {pp.get(key)}")
    return problems


def _figures_of(node, path=""):
    """Every figure (a dict with a ``lo`` key) under ``node``, as ``(path, figure)`` pairs."""
    out = []
    if isinstance(node, dict):
        if "lo" in node:
            out.append((path, node))
        for k, v in node.items():
            out.extend(_figures_of(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.extend(_figures_of(v, f"{path}[{i}]"))
    return out


def pack_plan_problems(plan):
    """One sentence per fault of the pre-walk ``plan`` itself (empty: sound): a figure without exactly the five keys,
    a provenance outside :data:`PROVENANCE`, a derived figure that is not exact, a proxy or measured figure that
    states a central value, ``lo`` above ``hi``, an empty source, a ``stand-in:`` source on a non-proxy figure, a
    ``total`` whose ``lo``/``hi`` is not the sum of its tiers', and a plan that is not JSON or not equal to its own
    JSON round trip."""
    problems = []
    try:
        s = json.dumps(plan, allow_nan=False)
        if json.loads(s) != plan:
            problems.append("the plan differs from its JSON round trip")
    except (TypeError, ValueError) as e:
        problems.append(f"the plan is not JSON: {e}")

    for path, fig in _figures_of({"tiers": plan.get("tiers"), "total": plan.get("total")}):
        for key in _FIG_KEYS:
            if key not in fig:
                problems.append(f"figure {path} lacks {key}")
        for key in sorted(set(fig) - _FIG_KEYS):
            problems.append(f"figure {path} carries {key!r}")
        p, v, lo, hi, src = (fig.get(k) for k in ("provenance", "value", "lo", "hi", "source"))
        if p not in PROVENANCE:
            problems.append(f"figure {path} carries provenance '{p}', not one of derived, measured, proxy")
        if isinstance(src, str) and src.startswith("stand-in:") and p != "proxy":
            problems.append(f"figure {path} carries a stand-in source with provenance '{p}'")
        if p == "derived" and v is not None and (lo != hi or v != lo):
            problems.append(f"figure {path} is derived and is not exact")
        if p in ("proxy", "measured") and v is not None:
            problems.append(f"figure {path} is {p} and states a central value")
        if lo is not None and hi is not None and lo > hi:
            problems.append(f"figure {path} has lo above hi")
        if not (isinstance(src, str) and src):
            problems.append(f"figure {path} has an empty source")

    tiers = [t for t in (plan.get("tiers") or {}).values() if isinstance(t, dict)]
    total = plan.get("total") or {}
    for stage in _STAGES:
        if stage not in total or not isinstance(total[stage], dict):
            continue
        figs = [f for f in (_stage_figure(t, stage) for t in tiers) if isinstance(f, dict) and "lo" in f]
        for which in ("lo", "hi"):
            vals = [f.get(which) for f in figs]
            if all(v is not None for v in vals):
                if total[stage].get(which) != sum(vals):
                    problems.append(f"total {stage} {which} differs from the sum of the tiers")
            elif total[stage].get(which) is not None:
                problems.append(f"total {stage} {which} differs from the sum of the tiers")
    return problems
