"""`Tissue`: the material a replay applies to a pack -- T2 and T1 per pool, the walls' surface relaxivity, the
bulk diffusivity, the field source's susceptibility. A pack carries channels and its substrate spec, never these
numbers; `ReplayPack.nominal` (:meth:`Tissue.from_spec`) reads the spec's NOMINAL values when the caller asks
for them, explicitly. The scanner's field is not tissue and is the replay's ``scanner=``; the field's direction
is the pose's (``orientation=``).
"""
import dataclasses
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Tissue:
    """The physical values of a material. Every field is optional: a value that is ``None`` switches its tier off.

    * ``T2`` / ``T1`` (s): by pool -- a ``{pool name: value}`` dict resolved through the pack's spec, a list by pool
      id, or one value for every pool.
    * ``rho`` (m/s): the walls' surface relaxivity (C2), scaled by ``D``.
    * ``D`` (m^2/s): the bulk diffusivity -- the walk's recorded value unless given, for a pack; a closed form's
      diffusion coefficient.
    * ``chi_iso`` / ``chi_aniso``: the field source's susceptibility (C3), evaluated at the scanner's field.
    """
    T2: Optional[object] = None
    T1: Optional[object] = None
    rho: Optional[float] = None
    D: Optional[float] = None
    chi_iso: Optional[float] = None
    chi_aniso: float = 0.0

    @classmethod
    def from_spec(cls, spec, **overrides):
        """The spec's nominal values: pool T2 / T1 (by id), the walls' common relaxivity, the field-source pool's
        susceptibility. Walls with different relaxivities leave ``rho`` None: give it. Any keyword overrides."""
        pools = sorted(spec.pools, key=lambda p: p.id)
        T2 = [p.T2 for p in pools]; T1 = [p.T1 for p in pools]
        rhos = {r for w in spec.walls for r in (w.surface_relaxivity.inside, w.surface_relaxivity.outside) if r > 0}
        rhos |= ({spec.domain.surface_relaxivity} if spec.domain.surface_relaxivity else set())
        src = spec.field_source_pools
        kw = dict(T2=(None if any(t is None for t in T2) else T2), T1=(None if any(t is None for t in T1) else T1),
                  rho=(rhos.pop() if len(rhos) == 1 else None),
                  chi_iso=(src[0].susceptibility.chi_iso if src else None),
                  chi_aniso=((src[0].susceptibility.chi_aniso or 0.0) if src else 0.0))
        kw.update(overrides)
        return cls(**kw)

    def replace(self, **changes):
        """This tissue with some values changed: ``pack.nominal.replace(T2={"intra": 0.08})``."""
        return dataclasses.replace(self, **changes)

    @property
    def relaxes(self):
        """Whether a bulk relaxation time is declared (the tier that a readout at an echo time applies)."""
        return self.T2 is not None or self.T1 is not None

    def to_meta(self):
        """The declared values as a JSON-ready dict (a ``.rph`` substrate's ``tissue`` entry): only what is set."""
        import numpy as np
        out = {}
        for k in ("T2", "T1", "rho", "D", "chi_iso", "chi_aniso"):
            v = getattr(self, k)
            if v is None or (k == "chi_aniso" and v == 0.0):
                continue
            out[k] = (dict(v) if isinstance(v, dict) else
                      [float(x) for x in np.asarray(v, float).reshape(-1)] if np.ndim(v) else float(v))
        return out

    @classmethod
    def from_meta(cls, meta):
        """A tissue back from :meth:`to_meta`; ``None`` for an empty or absent entry."""
        m = dict(meta or {})
        unknown = set(m) - {"T2", "T1", "rho", "D", "chi_iso", "chi_aniso"}
        if unknown:
            raise ValueError(f"a tissue entry declares {sorted(unknown)}; it takes T2, T1, rho, D, chi_iso, chi_aniso")
        return cls(**m) if m else None
