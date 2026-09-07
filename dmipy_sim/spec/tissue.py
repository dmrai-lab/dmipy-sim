"""`Tissue`: the physical values a replay applies to a pack -- T2 and T1 per pool, the wall's surface
relaxivity, the field (B0, its direction, the susceptibilities). A pack carries channels and its
substrate spec, never these numbers; `Tissue.from_spec` reads the spec's NOMINAL values when the caller
says so, explicitly.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Tissue:
    T2: Optional[list] = None            # by pool id (s)
    T1: Optional[list] = None            # by pool id (s)
    rho: Optional[float] = None          # surface relaxivity at the walls (m/s)
    B0: Optional[float] = None           # T; None = no field
    b0_dir: tuple = (0.0, 0.0, 1.0)      # in the substrate frame
    chi_iso: Optional[float] = None
    chi_aniso: float = 0.0

    @classmethod
    def from_spec(cls, spec, *, B0=None, b0_dir=(0.0, 0.0, 1.0), **overrides):
        """The spec's nominal pool T2 / T1, the walls' common relaxivity and the field-source pool's
        susceptibility, with ``B0`` supplied here (a spec has no field strength). Any keyword overrides.
        Walls with different relaxivities leave ``rho`` None: give it."""
        pools = sorted(spec.pools, key=lambda p: p.id)
        T2 = [p.T2 for p in pools]; T1 = [p.T1 for p in pools]
        rhos = {r for w in spec.walls for r in (w.surface_relaxivity.inside, w.surface_relaxivity.outside) if r > 0}
        rhos |= ({spec.domain.surface_relaxivity} if spec.domain.surface_relaxivity else set())
        src = spec.field_source_pools
        kw = dict(T2=(None if any(t is None for t in T2) else T2), T1=(None if any(t is None for t in T1) else T1),
                  rho=(rhos.pop() if len(rhos) == 1 else None), B0=B0, b0_dir=tuple(b0_dir),
                  chi_iso=(src[0].susceptibility.chi_iso if src else None),
                  chi_aniso=((src[0].susceptibility.chi_aniso or 0.0) if src else 0.0))
        kw.update(overrides)
        return cls(**kw)

    def knobs(self):
        """Keyword arguments for :meth:`ReplayPack.replay`."""
        return dict(T2=self.T2, T1=self.T1, rho=self.rho, B0=self.B0, b0_dir=self.b0_dir,
                    chi_iso=self.chi_iso, chi_aniso=self.chi_aniso)
