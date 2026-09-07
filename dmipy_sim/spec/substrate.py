"""Substrate specification 0.1: dataclasses, JSON round trip and the validator.

The document is normative (replay-pack-spec/SUBSTRATE.md); ``substrate.schema.json`` beside this
module is the type schema; :func:`validate` checks the schema's constraints (without needing the
``jsonschema`` package) and the invariants the schema cannot express: dense pool ids with 0 the free
pool, walls between existing pools, void outside a wall implying an impermeable wall, seeded pools
that exist, a box with positive extent, tiers consistent with the content.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Optional

SPEC_VERSION = "0.1"
SCHEMA_PATH = Path(__file__).with_name("substrate.schema.json")
BOUNDARIES = ("periodic", "reflect", "open")
SURFACE_KINDS = ("sphere", "cylinder", "ellipsoid", "plane", "swept_polyline", "sphere_union", "mesh")
DIRECTORS = ("none", "radial", "file")
SEEDING_RULES = ("uniform_by_volume", "explicit")
WEIGHT_RULES = ("water_fraction", "thin")
TIERS = ("gradient", "relaxation", "surface", "field", "exchange")


class SpecError(ValueError):
    """A substrate spec that does not describe a walkable situation; the message names the field."""


@dataclass(frozen=True)
class Directional:
    """A per-crossing-direction wall property (m/s): crossing from inside to outside and back."""
    in_to_out: float = 0.0
    out_to_in: float = 0.0


@dataclass(frozen=True)
class Sided:
    """A per-side wall property (m/s), as seen by a spin hitting the wall from that side."""
    inside: float = 0.0
    outside: float = 0.0


@dataclass(frozen=True)
class Susceptibility:
    """A pool's susceptibility: the director rule that shapes the field basis, plus nominal values -- ``None``
    when the producer knows the pool is a field source but not its chi (a replay knob either way)."""
    chi_iso: Optional[float] = None
    chi_aniso: Optional[float] = None
    director: str = "none"
    file: Optional[str] = None


@dataclass(frozen=True)
class Pool:
    id: int
    name: str
    D: Optional[float]           # None: the free diffusivity the walk is driven with
    water_fraction: float = 1.0
    T2: Optional[float] = None
    T1: Optional[float] = None
    susceptibility: Optional[Susceptibility] = None


@dataclass(frozen=True)
class Surface:
    """An analytic surface, a sphere union or a mesh file; ``instances`` holds per-instance parameter arrays.
    A ``sphere_union`` is inline (``instances.centers`` / ``radii``) or a CATERPillar table (``file``,
    ``format: caterpillar``, ``column`` = which radius, ``cell_type`` = which rows)."""
    kind: str
    center: Optional[list] = None
    radius: Optional[float] = None
    axis: Optional[list] = None
    length: Optional[float] = None
    semiaxes: Optional[list] = None
    rotation: Optional[list] = None
    point: Optional[list] = None
    column: Optional[str] = None        # sphere_union from a table: which radius column
    cell_type: Optional[str] = None     # sphere_union from a table: which rows (axon | glial_cell | blood_vessel)
    normal: Optional[list] = None
    centerline: Optional[list] = None
    file: Optional[str] = None
    format: Optional[str] = None
    scale: Optional[float] = None
    sha256: Optional[str] = None
    instances: Optional[dict] = None


@dataclass(frozen=True)
class Wall:
    name: str
    surface: Surface
    inside_pool: int
    outside_pool: Optional[int]
    permeability: Directional = Directional()
    surface_relaxivity: Sided = Sided()
    mt_reactivity: Sided = Sided()


@dataclass(frozen=True)
class Domain:
    box_min: list
    box_max: list
    boundary: list
    surface_relaxivity: float = 0.0


@dataclass(frozen=True)
class Frame:
    axis: list = field(default_factory=lambda: [0.0, 0.0, 1.0])
    in_plane: Optional[list] = None


@dataclass(frozen=True)
class Seeding:
    pools: list
    rule: str = "uniform_by_volume"
    weights: str = "water_fraction"


@dataclass(frozen=True)
class Validity:
    smallest_feature: float
    tiers: list
    min_gap: Optional[float] = None
    mesh_edge_feature_ratio: Optional[float] = None


@dataclass(frozen=True)
class SubstrateSpec:
    id: str
    domain: Domain
    pools: list
    walls: list
    seeding: Seeding
    validity: Validity
    frame: Frame = field(default_factory=Frame)
    description: str = ""
    request: Optional[dict] = None
    realisation: Optional[dict] = None
    provenance: Optional[dict] = None
    substrate_spec_version: str = SPEC_VERSION

    # ---- pools and walls by name ----
    def pool(self, name_or_id):
        for p in self.pools:
            if p.id == name_or_id or p.name == name_or_id:
                return p
        raise KeyError(f"no pool {name_or_id!r}; pools are {[p.name for p in self.pools]}")

    def wall(self, name):
        for w in self.walls:
            if w.name == name:
                return w
        raise KeyError(f"no wall {name!r}; walls are {[w.name for w in self.walls]}")

    @property
    def field_source_pools(self):
        """The pools that generate a susceptibility field."""
        return [p for p in self.pools if p.susceptibility is not None]

    # ---- JSON ----
    def to_dict(self):
        return _strip(asdict(self))

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d):
        return _build(cls, dict(d))

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    def save(self, path):
        Path(path).write_text(self.to_json())
        return path

    def validate(self):
        validate(self.to_dict())
        return self


def load_spec(path):
    """Read and validate a ``.sub.json`` file."""
    spec = SubstrateSpec.from_json(Path(path).read_text())
    spec.validate()
    return spec


_NESTED = {"domain": Domain, "frame": Frame, "seeding": Seeding, "validity": Validity,
           "surface": Surface, "permeability": Directional, "surface_relaxivity": Sided,
           "mt_reactivity": Sided, "susceptibility": Susceptibility}


def _build(cls, d):
    kw = {}
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise SpecError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        if cls is SubstrateSpec and f.name == "pools":
            v = [_build(Pool, dict(x)) if isinstance(x, dict) else x for x in v]
        elif cls is SubstrateSpec and f.name == "walls":
            v = [_build(Wall, dict(x)) if isinstance(x, dict) else x for x in v]
        elif f.name in _NESTED and isinstance(v, dict) and not (cls is Domain and f.name == "surface_relaxivity"):
            v = _build(_NESTED[f.name], dict(v))
        kw[f.name] = v
    return cls(**kw)


def _strip(x):
    """Drop None-valued optional keys of nested objects so the JSON stays as the schema shows it."""
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items()
                if not (v is None and k in ("center", "radius", "axis", "length", "semiaxes", "rotation", "point",
                                            "normal", "centerline", "file", "format", "scale", "sha256", "instances"))}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


# ---------------------------------------------------------------------------------- validation
def _req(d, key, where):
    if key not in d:
        raise SpecError(f"{where}: missing required key {key!r}")
    return d[key]


def _vec3(v, where):
    if not (isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v)):
        raise SpecError(f"{where}: expected three numbers, got {v!r}")


def _nonneg(v, where):
    if not isinstance(v, (int, float)) or v < 0:
        raise SpecError(f"{where}: expected a number >= 0, got {v!r}")


def validate(d):
    """Validate a spec dict; raises :class:`SpecError` naming the offending field, returns the dict."""
    if is_dataclass(d):
        d = d.to_dict()
    for k in ("substrate_spec_version", "id", "domain", "frame", "pools", "walls", "seeding", "validity"):
        _req(d, k, "spec")
    if not str(d["substrate_spec_version"]).startswith("0."):
        raise SpecError(f"substrate_spec_version {d['substrate_spec_version']!r}: this validator reads 0.x")
    if not isinstance(d["id"], str) or not d["id"]:
        raise SpecError("id must be a non-empty string")
    dom = d["domain"]
    lo, hi = _req(dom, "box_min", "domain"), _req(dom, "box_max", "domain")
    _vec3(lo, "domain.box_min"); _vec3(hi, "domain.box_max")
    if not all(a < b for a, b in zip(lo, hi)):
        raise SpecError(f"domain: box_min {lo} must be below box_max {hi} on every axis")
    bc = _req(dom, "boundary", "domain")
    if not (isinstance(bc, list) and len(bc) == 3 and all(b in BOUNDARIES for b in bc)):
        raise SpecError(f"domain.boundary must be three of {BOUNDARIES}, got {bc!r}")
    _vec3(_req(d["frame"], "axis", "frame"), "frame.axis")
    pools = d["pools"]
    if not pools:
        raise SpecError("pools: at least the free pool (id 0) is required")
    ids = [p.get("id") for p in pools]
    if ids != list(range(len(pools))):
        raise SpecError(f"pools: ids must be dense 0..{len(pools) - 1} in order, got {ids}")
    if pools[0].get("name") not in ("extra", "free"):
        raise SpecError(f"pools[0] is the free / extra-cellular pool and must be named 'extra' or 'free', got {pools[0].get('name')!r}")
    names = [p.get("name") for p in pools]
    if len(set(names)) != len(names):
        raise SpecError(f"pools: names must be unique, got {names}")
    for p in pools:
        w = f"pool {p.get('name')!r}"
        if p.get("D") is not None:
            _nonneg(p["D"], f"{w}.D")
        wf = _req(p, "water_fraction", w)
        if not (isinstance(wf, (int, float)) and 0.0 <= wf <= 1.0):
            raise SpecError(f"{w}.water_fraction must be in [0, 1], got {wf!r}")
        for k in ("T2", "T1"):
            if p.get(k) is not None and not (isinstance(p[k], (int, float)) and p[k] > 0):
                raise SpecError(f"{w}.{k} must be a positive time or null, got {p[k]!r}")
        su = p.get("susceptibility")
        if su is not None:
            for k in ("chi_iso", "chi_aniso", "director"):
                _req(su, k, f"{w}.susceptibility")
            for k in ("chi_iso", "chi_aniso"):
                if su[k] is not None and not isinstance(su[k], (int, float)):
                    raise SpecError(f"{w}.susceptibility.{k} must be a number or null, got {su[k]!r}")
            if su["director"] not in DIRECTORS:
                raise SpecError(f"{w}.susceptibility.director must be one of {DIRECTORS}, got {su['director']!r}")
            if su["director"] == "file" and not su.get("file"):
                raise SpecError(f"{w}.susceptibility: director 'file' needs 'file'")
    n_pools = len(pools)
    walls = d["walls"]
    wnames = [w.get("name") for w in walls]
    if len(set(wnames)) != len(wnames):
        raise SpecError(f"walls: names must be unique, got {wnames}")
    for w in walls:
        where = f"wall {w.get('name')!r}"
        s = _req(w, "surface", where)
        if s.get("kind") not in SURFACE_KINDS:
            raise SpecError(f"{where}.surface.kind must be one of {SURFACE_KINDS}, got {s.get('kind')!r}")
        if s["kind"] == "mesh" and not s.get("file"):
            raise SpecError(f"{where}.surface: a mesh surface needs 'file'")
        if s["kind"] == "sphere_union":
            inst = s.get("instances") or {}
            if s.get("file"):
                if s.get("format") != "caterpillar" or s.get("column") not in ("inner_radius", "outer_radius"):
                    raise SpecError(f"{where}.surface: a sphere_union file needs format 'caterpillar' and column "
                                    f"'inner_radius' or 'outer_radius'")
            elif not (inst.get("centers") and inst.get("radii")):
                raise SpecError(f"{where}.surface: a sphere_union needs 'file' or instances.centers and instances.radii")
        if s["kind"] in ("sphere", "cylinder") and s.get("radius") is None and not (s.get("instances") or {}).get("radii"):
            raise SpecError(f"{where}.surface: a {s['kind']} needs 'radius' or instances.radii")
        inst = s.get("instances")
        if inst:
            lens = {k: len(v) for k, v in inst.items()}
            if len(set(lens.values())) != 1:
                raise SpecError(f"{where}.surface.instances arrays must have one length, got {lens}")
        ip = _req(w, "inside_pool", where)
        if not (isinstance(ip, int) and 0 <= ip < n_pools):
            raise SpecError(f"{where}.inside_pool {ip!r} is not a pool id (0..{n_pools - 1})")
        op = _req(w, "outside_pool", where)
        if op is not None and not (isinstance(op, int) and 0 <= op < n_pools):
            raise SpecError(f"{where}.outside_pool {op!r} is not a pool id or null")
        if op == ip:
            raise SpecError(f"{where}: inside_pool and outside_pool are both {ip}; a wall separates two pools")
        perm = _req(w, "permeability", where)
        for k in ("in_to_out", "out_to_in"):
            _nonneg(_req(perm, k, f"{where}.permeability"), f"{where}.permeability.{k}")
        if op is None and (perm["in_to_out"] > 0 or perm["out_to_in"] > 0):
            raise SpecError(f"{where}: outside_pool is void, so the wall must be impermeable (permeability 0)")
        for k in ("surface_relaxivity", "mt_reactivity"):
            sd = _req(w, k, where)
            for side in ("inside", "outside"):
                _nonneg(_req(sd, side, f"{where}.{k}"), f"{where}.{k}.{side}")
    seed = d["seeding"]
    sp = _req(seed, "pools", "seeding")
    if not sp or any(not (isinstance(i, int) and 0 <= i < n_pools) for i in sp):
        raise SpecError(f"seeding.pools {sp!r} must be non-empty pool ids (0..{n_pools - 1})")
    if _req(seed, "rule", "seeding") not in SEEDING_RULES:
        raise SpecError(f"seeding.rule must be one of {SEEDING_RULES}")
    if _req(seed, "weights", "seeding") not in WEIGHT_RULES:
        raise SpecError(f"seeding.weights must be one of {WEIGHT_RULES}")
    for i in sp:
        if pools[i]["water_fraction"] == 0.0:
            raise SpecError(f"seeding: pool {pools[i]['name']!r} is seeded but has water_fraction 0")
    val = d["validity"]
    sf = _req(val, "smallest_feature", "validity")
    if not (isinstance(sf, (int, float)) and sf > 0):
        raise SpecError(f"validity.smallest_feature must be > 0, got {sf!r}")
    tiers = _req(val, "tiers", "validity")
    if any(t not in TIERS for t in tiers) or len(set(tiers)) != len(tiers):
        raise SpecError(f"validity.tiers must be distinct entries of {TIERS}, got {tiers!r}")
    has_wall = bool(walls) or (float(dom.get("surface_relaxivity", 0.0) or 0.0) > 0 and "reflect" in bc)
    if "surface" in tiers and not has_wall:
        raise SpecError("validity.tiers lists 'surface' but the substrate has no wall (and no relaxing reflect face)")
    if "field" in tiers and not any(p.get("susceptibility") for p in pools):
        raise SpecError("validity.tiers lists 'field' but no pool is a field source (susceptibility)")
    if "relaxation" in tiers and not (n_pools > 1 or any(p.get("T2") for p in pools)):
        raise SpecError("validity.tiers lists 'relaxation' but there is one pool and no T2")
    if "exchange" in tiers and not any(w["permeability"]["in_to_out"] > 0 or w["permeability"]["out_to_in"] > 0
                                       or w["mt_reactivity"]["inside"] > 0 or w["mt_reactivity"]["outside"] > 0
                                       for w in walls):
        raise SpecError("validity.tiers lists 'exchange' but no wall is permeable or MT-reactive")
    return d
