"""`Tissue`: the material a replay applies to a pack -- T2 and T1 per pool, the walls' surface relaxivity, the
bulk diffusivity, the field source's susceptibility. A pack carries channels and its substrate spec, never these
numbers; `ReplayPack.nominal` (:meth:`Tissue.from_spec`) reads the spec's NOMINAL values when the caller asks
for them, explicitly. The scanner's field is not tissue and is the replay's ``scanner=``; the field's direction
is the pose's (``orientation=``).
"""
import dataclasses
from dataclasses import dataclass
from typing import Optional

import numpy as np


PER_POOL = ("T2", "T1")
KNOBS = ("T2", "T1", "rho", "D", "kappa", "chi_iso", "chi_aniso")


def _check_time(what, v):
    """``v`` as a relaxation time in seconds: positive, ``inf`` for no decay; ``0`` is refused since the kernels
    read a rate and a zero time is not "no decay in this pool"."""
    if isinstance(v, bool) or np.ndim(v) != 0:
        raise TypeError(f"{what} is a time in seconds; got {v!r}")
    v = float(v)
    if not v > 0.0:
        raise ValueError(f"{what} is a relaxation time in seconds and must be positive; got {v!r}. "
                         f"No decay in a pool is float('inf'), never 0")
    return v


def _check_per_pool(what, v):
    """``T2`` / ``T1`` as given: ``None``, one positive number (a closed form's one pool) or a ``{pool name:
    seconds}`` mapping (a pack's pools). A list by pool id is the ``.rph`` file form, resolved through the pack's
    spec by :class:`~dmipy_sim.phantom.PackSubstrate`, and is refused here."""
    if v is None:
        return None
    if isinstance(v, dict):
        out = {}
        for k, t in v.items():
            if not isinstance(k, str):
                raise TypeError(f"{what} maps pool NAMES to seconds; got the key {k!r}")
            out[k] = _check_time(f"{what}[{k!r}]", t)
        return out
    if isinstance(v, (list, tuple)) or np.ndim(v) != 0:
        raise TypeError(f"{what} is {{pool name: seconds}} on a pack, or one number on a closed form; a list by "
                        f"pool id is the .rph file form and is read through PackSubstrate, not given here (got {v!r})")
    return _check_time(what, v)


@dataclass(frozen=True)
class Tissue:
    """The physical values of a material. Every field is optional: a value that is ``None`` switches its tier off.

    * ``T2`` / ``T1`` (s): on a pack, a ``{pool name: seconds}`` mapping over EVERY pool of the pack's embedded
      spec (the replay refuses a missing pool, an unknown name, a scalar or a list); ``float("inf")`` is no decay
      in that pool, and ``0`` is refused. On a closed form (one unnamed pool) one number.
    * ``rho`` (m/s): the walls' surface relaxivity (C2), scaled by ``D``.
    * ``D`` (m^2/s): the bulk diffusivity -- the walk's recorded value unless given, for a pack; a closed form's
      diffusion coefficient. Given on a pack, the pack is READ at that diffusivity
      (:meth:`~dmipy_sim.replay.ReplayPack.at_diffusivity`): the save grid divided by ``D / D_walk``, every
      channel following in its own space, which is what a change of temperature does. Faster than walked only.
    * ``kappa`` (m/s): the walls' permeability the pack is READ at (:meth:`~dmipy_sim.replay.ReplayPack.at_permeability`).
      A walk realised one ratio ``kappa / D``, and the same path is the walk at ``(a D, a kappa)`` for any
      ``a >= 1``: stating ``kappa`` picks ``a``, stating ``D`` too must agree, and a pair off the walk's line is
      refused. One walk at the slowest, longest setting of a study serves every faster one on its line.
    * ``chi_iso`` / ``chi_aniso``: the field source's susceptibility (C3), evaluated at the scanner's field.
    """
    T2: Optional[object] = None
    T1: Optional[object] = None
    rho: Optional[float] = None
    D: Optional[float] = None
    kappa: Optional[float] = None
    chi_iso: Optional[float] = None
    chi_aniso: float = 0.0

    def __post_init__(self):
        for k in PER_POOL:
            object.__setattr__(self, k, _check_per_pool(k, getattr(self, k)))

    @classmethod
    def from_spec(cls, spec, **overrides):
        """The spec's nominal values: pool T2 / T1 by pool name (the pools that declare one; ``None`` when none
        does, so a spec that declares a T2 in some pools gives a mapping the replay refuses by naming the rest),
        the walls' common relaxivity, the field-source pool's susceptibility. Walls with different relaxivities
        leave ``rho`` None: give it. Any keyword overrides."""
        pools = sorted(spec.pools, key=lambda p: p.id)
        T2 = {p.name: p.T2 for p in pools if p.T2 is not None} or None
        T1 = {p.name: p.T1 for p in pools if p.T1 is not None} or None
        rhos = {r for w in spec.walls for r in (w.surface_relaxivity.inside, w.surface_relaxivity.outside) if r > 0}
        rhos |= ({spec.domain.surface_relaxivity} if spec.domain.surface_relaxivity else set())
        src = spec.field_source_pools
        kw = dict(T2=T2, T1=T1, rho=(rhos.pop() if len(rhos) == 1 else None),
                  chi_iso=(src[0].susceptibility.chi_iso if src else None),
                  chi_aniso=((src[0].susceptibility.chi_aniso or 0.0) if src else 0.0))
        kw.update(overrides)
        return cls(**kw)

    def replace(self, **changes):
        """This tissue with some values changed. A per-pool mapping merges pool by pool onto the one held, so
        ``pack.nominal.replace(T2={"intra": 0.08})`` changes one pool and keeps the others."""
        for k in PER_POOL:
            new, old = changes.get(k), getattr(self, k)
            if isinstance(new, dict) and isinstance(old, dict):
                changes[k] = {**old, **new}
        return dataclasses.replace(self, **changes)

    @property
    def relaxes(self):
        """Whether a bulk relaxation time is declared (the tier that a readout at an echo time applies)."""
        return self.T2 is not None or self.T1 is not None

    def to_meta(self, spec=None):
        """The declared values as a JSON-ready dict, only what is set. With the pack's ``spec`` (a
        :class:`~dmipy_sim.phantom.PackSubstrate` writing a ``.rph``), a per-pool mapping is written in the file
        form: a list by pool id, one entry per spec pool, ``null`` for no decay (RPH.md 3.2); a missing or unknown
        pool is refused. Without a spec (a study's record) the mapping is written by name, ``null`` for ``inf``."""
        out = {}
        for k in KNOBS:
            v = getattr(self, k)
            if v is None or (k == "chi_aniso" and v == 0.0):
                continue
            if k in PER_POOL and isinstance(v, dict):
                if spec is not None:
                    v = [None if np.isinf(t) else float(t) for t in _by_pool_id(spec, v, k)]
                else:
                    v = {name: (None if np.isinf(t) else float(t)) for name, t in v.items()}
            elif k in PER_POOL and spec is not None:
                raise ValueError(f"{k} on a pack is {{pool name: seconds}} over the pack's pools "
                                 f"{[p.name for p in spec.pools]}; got the one number {v!r}")
            else:
                v = float(v)
            out[k] = v
        return out

    @classmethod
    def from_meta(cls, meta, spec=None):
        """A tissue back from :meth:`to_meta`; ``None`` for an empty or absent entry. With the pack's ``spec`` the
        per-pool values MUST be the file form, a list by pool id of the spec's length (``null`` for no decay),
        and come back as the mapping by name; a dict, a scalar or a list of another length is refused. Without
        a spec a list is refused (it needs the pack to be read) and a mapping or a number is taken as given."""
        m = dict(meta or {})
        unknown = set(m) - set(KNOBS)
        if unknown:
            raise ValueError(f"a tissue entry declares {sorted(unknown)}; it takes T2, T1, rho, D, kappa, chi_iso, chi_aniso")
        for k in PER_POOL:
            v = m.get(k)
            if v is None:
                continue
            if spec is not None:
                if not isinstance(v, list):
                    raise ValueError(f"a .rph substrate's {k} is a list by pool id, one entry per pool of the pack's "
                                     f"spec ({len(spec.pools)}), null for no decay (RPH.md 3.2); got {v!r}")
                pools = sorted(spec.pools, key=lambda p: p.id)
                if len(v) != len(pools):
                    raise ValueError(f"a .rph substrate's {k} lists {len(v)} value(s) and the pack's spec has "
                                     f"{len(pools)} pools {[p.name for p in pools]}: one entry per pool, by id")
                m[k] = {p.name: (np.inf if t is None else t) for p, t in zip(pools, v)}
            elif isinstance(v, list):
                raise ValueError(f"{k} is a list by pool id, the .rph file form, which only the pack's spec can read; "
                                 f"read it through PackSubstrate, or give {{pool name: seconds}}")
            elif isinstance(v, dict):
                m[k] = {name: (np.inf if t is None else t) for name, t in v.items()}
        return cls(**m) if m else None


def _by_pool_id(spec, values, what):
    """A ``{pool name: value}`` mapping as the list by pool id over EVERY pool of ``spec``: an unknown name or a
    missing pool is refused, naming it. The one resolver of a per-pool value on a pack."""
    pools = sorted(spec.pools, key=lambda p: p.id)
    names = [p.name for p in pools]
    unknown = sorted(set(values) - set(names))
    if unknown:
        raise ValueError(f"{what} names the pool(s) {unknown}, which the pack's spec does not have; its pools are {names}")
    missing = [n for n in names if n not in values]
    if missing:
        raise ValueError(f"{what} gives no value for the pool(s) {missing}; on a pack it covers every pool of the "
                         f"embedded spec {names} (float('inf') for no decay in a pool). A pool the walk never "
                         f"seeds still takes one, since the mapping is judged on the spec, not on the walkers")
    return [values[n] for n in names]
