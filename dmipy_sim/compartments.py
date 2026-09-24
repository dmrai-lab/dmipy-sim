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
    ``surface_relaxivity_t2`` (rho, m/s). ``None`` leaves a property to the geometry / driver
    default. A membrane's permeability is the geometry's (``permeability=``), not a pool's."""
    D: Optional[float] = None
    T2: Optional[float] = None
    T1: Optional[float] = None
    water_fraction: Optional[float] = None
    surface_relaxivity_t2: Optional[float] = None

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None:
                v = float(v)
                if v < 0.0 or (f.name in ("T2", "T1") and v == 0.0):
                    raise ValueError(f"Pool.{f.name} must be {'positive' if f.name in ('T2', 'T1') else 'non-negative'}, got {v}")
                object.__setattr__(self, f.name, v)


class Compartments(Mapping):
    """Pools by name: ``Compartments(extra=Pool(T2=0.08), intra=Pool(T2=0.05, D=1.7e-9))``.

    A read-only mapping ordered by pool id. ``by_id(prop)`` returns the property for every pool
    the substrate has, in id order, and requires it on all of them or none: a per-compartment
    property given for one pool and left to default on another is the silent-misrepresentation
    case, so it raises.
    """

    def __init__(self, **given):
        self._pools = {}
        unknown = sorted(set(given) - set(POOL_NAMES))
        if unknown:
            raise KeyError(f"unknown pools {unknown}; pools are {POOL_NAMES}")
        for name in POOL_NAMES:                       # id order
            if name in given:
                pool = given[name]
                if not isinstance(pool, Pool):
                    raise TypeError(f"the {name} pool is a Pool ({name}=Pool(T2=..., D=...)), got {type(pool).__name__}")
                self._pools[name] = pool

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
        return Compartments(**merged)

    @classmethod
    def coerce(cls, value):
        """The ``compartments=`` argument of a geometry: a Compartments, or None (no pool declared)."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        raise TypeError(f"compartments is a Compartments (Compartments(intra=Pool(T2=...), extra=Pool(T2=...))), "
                        f"got {type(value).__name__}")
