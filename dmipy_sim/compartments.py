"""One spelling for per-compartment tissue properties.

A substrate has up to three water pools -- ``extra`` (id 0), ``intra`` (id 1), ``myelin``
(id 2) -- the ids every compartment channel, ``return_compartments`` array and ``.rpk`` pack
use. `Compartments` maps pool names to a `Pool` of bulk and wall properties; geometries and the
`Substrate` accept one, and ``pool_id`` is the only place a name becomes an index.
"""
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Optional

POOL_IDS = {"extra": 0, "intra": 1, "myelin": 2}
POOL_NAMES = tuple(POOL_IDS)                     # index = pool id


def pool_id(name):
    """Compartment id of a pool name (``extra`` 0, ``intra`` 1, ``myelin`` 2); an int passes through."""
    if isinstance(name, str):
        try:
            return POOL_IDS[name]
        except KeyError:
            raise KeyError(f"unknown pool {name!r}; pools are {POOL_NAMES}") from None
    i = int(name)
    if i not in POOL_IDS.values():
        raise KeyError(f"unknown pool id {i}; ids are {sorted(POOL_IDS.values())}")
    return i


@dataclass(frozen=True)
class Pool:
    """Tissue properties of one water pool. Bulk: ``D`` (m^2/s), ``T2``, ``T1`` (s),
    ``water_fraction`` (proton-density weight). Wall, seen from this pool's side of its membrane:
    ``surface_relaxivity_t2`` (rho, m/s) and ``permeability`` (kappa, m/s, crossing OUT of this
    pool). ``None`` leaves a property to the geometry / driver default."""
    D: Optional[float] = None
    T2: Optional[float] = None
    T1: Optional[float] = None
    water_fraction: Optional[float] = None
    surface_relaxivity_t2: Optional[float] = None
    permeability: Optional[float] = None

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None:
                v = float(v)
                if v < 0.0 or (f.name in ("T2", "T1") and v == 0.0):
                    raise ValueError(f"Pool.{f.name} must be {'positive' if f.name in ('T2', 'T1') else 'non-negative'}, got {v}")
                object.__setattr__(self, f.name, v)

    @classmethod
    def coerce(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            bad = set(value) - {f.name for f in fields(cls)}
            if bad:
                raise KeyError(f"unknown Pool properties {sorted(bad)}; known: {[f.name for f in fields(cls)]}")
            return cls(**value)
        raise TypeError(f"a pool is a Pool or a mapping of its properties, got {type(value).__name__}")


class Compartments(Mapping):
    """Pools by name: ``Compartments(extra=Pool(T2=0.08), intra={"T2": 0.05, "D": 1.7e-9})``.

    A read-only mapping ordered by pool id. ``by_id(prop)`` returns the property for every pool
    the substrate has, in id order, and requires it on all of them or none: a per-compartment
    property given for one pool and left to default on another is the silent-misrepresentation
    case, so it raises.
    """

    def __init__(self, pools=None, /, **by_name):
        given = dict(pools or {})
        given.update(by_name)
        self._pools = {}
        for name in POOL_NAMES:                       # id order
            if name in given:
                self._pools[name] = Pool.coerce(given.pop(name))
        if given:
            raise KeyError(f"unknown pools {sorted(given)}; pools are {POOL_NAMES}")

    def __getitem__(self, name):
        return self._pools[POOL_NAMES[pool_id(name)]]

    def __iter__(self):
        return iter(self._pools)

    def __len__(self):
        return len(self._pools)

    def __repr__(self):
        return "Compartments(" + ", ".join(f"{k}={v}" for k, v in self._pools.items()) + ")"

    def __eq__(self, other):
        return isinstance(other, Compartments) and self._pools == other._pools

    @property
    def ids(self):
        """Pool ids present, ascending."""
        return tuple(POOL_IDS[n] for n in self._pools)

    def by_id(self, prop, default=None):
        """``prop`` of every pool in id order, or ``None`` when no pool sets it.

        With ``default`` given, pools that leave the property unset take it; without one, a
        property set on some pools and not others raises.
        """
        vals = [getattr(p, prop) for p in self._pools.values()]
        if all(v is None for v in vals):
            return None
        if any(v is None for v in vals):
            if default is None:
                unset = [n for n, v in zip(self._pools, vals) if v is None]
                raise ValueError(f"per-compartment {prop!r} is set for some pools but not {unset}; "
                                 f"give it for every pool or none")
            vals = [float(default) if v is None else v for v in vals]
        return tuple(vals)

    def replace(self, **by_name):
        """A copy with the named pools replaced or added."""
        merged = dict(self._pools)
        merged.update(by_name)
        return Compartments(merged)

    @classmethod
    def coerce(cls, value):
        """A Compartments from a Compartments, a ``{name: Pool | mapping}`` mapping, or None (empty)."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(value)
        raise TypeError(f"compartments must be a Compartments or a mapping of pool name -> Pool, "
                        f"got {type(value).__name__}")
