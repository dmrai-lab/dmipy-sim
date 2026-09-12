"""What a voxel may hold: a solved pack, a closed form, or nothing (RPH.md 3.1).

Each declaration is an object that carries its own id, its proton density and, for a pack, the tissue knobs
it replays at. A phantom is keyed by these objects, so an id is typed once. The closed forms implement
:class:`AnalyticSubstrate`; :class:`FreeWater` is the one this package ships, and the file names a closed
form by ``model`` so a reader that does not know one refuses it rather than guessing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["AnalyticSubstrate", "PackSubstrate", "FreeWater", "Inert", "substrate_from_meta"]

_KNOBS = {"T2_s": "T2", "T1_s": "T1", "rho_m_s": "rho", "chi_iso": "chi_iso", "chi_aniso": "chi_aniso"}


@runtime_checkable
class AnalyticSubstrate(Protocol):
    """A closed form standing in for walkers. ``response`` is the complex signal of the form **at one pose**,
    one value per measurement of ``seq``; the phantom composes poses, the form never disperses itself.

    ``oriented`` says whether the form has a pose at all: ``False`` for an isotropic form (free water), whose
    ``response`` ignores ``pose``; ``True`` for a form with an axis (a stick, a cylinder), whose ``response`` at
    ``pose`` -- a 3x3 rotation of the canonical frame, the axis along its third column -- is what the phantom
    expands over SO(3) and contracts with a voxel's orientation distribution, exactly as a pack's (RPH.md 6).
    An oriented form takes an orientation field in :meth:`Phantom.compose` like a pack and is refused without
    one. ``model`` names the form for the file: sim's own (``free_water``) or a namespaced one,
    ``"<package>:<Name>"``, read back by ``<package>.phantom.analytic_substrate(meta)`` (RPH.md 3.1)."""
    oriented: bool

    def response(self, seq, pose=None) -> np.ndarray: ...

    def to_meta(self) -> dict: ...


class _Declared:
    """An id, a kind and a proton density: what every substrate declaration has (RPH.md 3, 5)."""

    kind = None

    def __init__(self, name, m0):
        self.name = str(name)
        self.m0 = float(m0)
        if self.m0 < 0:
            raise ValueError(f"m0 is a proton density and cannot be negative: {m0}")

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r}, m0={self.m0:g})"

    def _base_meta(self):
        return {"id": self.name, "kind": self.kind, "m0": self.m0}


class PackSubstrate(_Declared):
    """A solved replay pack as a substrate.

    ``pack`` is a ``.rpk`` path or an in-memory :class:`~dmipy_sim.replay.replay.ReplayPack`. ``m0`` is required:
    proton density only means anything relative to the other substrates of the same phantom, so there is no
    default, the same way there is no default pack. The tissue knobs (``T2_s``, ``T1_s`` by the pack's own pool
    names or by pool id, the wall relaxivity ``rho_m_s``, the field source's ``chi_iso`` / ``chi_aniso``) are the
    values this substrate replays at (RPH.md 3.2); anything not given takes the pack's nominal value from its
    embedded specification.
    """

    kind = "pack"

    def __init__(self, pack, *, m0, name=None, T2_s=None, T1_s=None, rho_m_s=None, chi_iso=None, chi_aniso=None):
        from ..replay.replay import ReplayPack
        if isinstance(pack, (str, Path)):
            self.uri, self._pack = str(pack), None
        elif isinstance(pack, ReplayPack):
            self.uri, self._pack = None, pack
        else:
            raise TypeError(f"pack is a .rpk path or a ReplayPack; got {type(pack).__name__}")
        if name is None:
            name = self._pack.meta.get("id") if self._pack is not None else Path(self.uri).stem
        super().__init__(name, m0)
        given = dict(T2_s=T2_s, T1_s=T1_s, rho_m_s=rho_m_s, chi_iso=chi_iso, chi_aniso=chi_aniso)
        self.tissue = {_KNOBS[k]: (dict(v) if isinstance(v, dict) else v) for k, v in given.items() if v is not None}

    @property
    def pack(self):
        """The pack: loaded from ``uri`` on first use, and a missing file is reported by its path."""
        if self._pack is None:
            from ..replay.replay import read_rpk
            if not Path(self.uri).exists():
                raise FileNotFoundError(f"substrate {self.name!r} cites the pack {self.uri!r}, which does not exist here; "
                                        f"pass it in memory with packs={{...}} or restore the file")
            self._pack = read_rpk(self.uri)
        return self._pack

    def to_meta(self):
        m = self._base_meta()
        if self.tissue:
            m["tissue"] = {k: (list(np.asarray(v, float).reshape(-1)) if not isinstance(v, dict) and np.ndim(v) else v)
                          for k, v in self.tissue.items()}
        return m

    @classmethod
    def from_meta(cls, meta, *, pack):
        t = dict(meta.get("tissue") or {})
        inv = {v: k for k, v in _KNOBS.items()}
        return cls(pack, m0=meta["m0"], name=meta["id"], **{inv[k]: v for k, v in t.items() if k in inv})


class FreeWater(_Declared):
    """Isotropic Gaussian diffusion in closed form, ``E = exp(-b D)``: the one analytic substrate the format
    defines (RPH.md 3.1).

    A pack cannot stand in for it: the signal decays exponentially in b while a Monte-Carlo floor decays only as
    ``1 / sqrt(N)``, so at b = 3000 s/mm² a few-thousand-walker free-water pack carries orders of magnitude more
    noise than signal. The closed form is exact, has no walkers and no pose. ``m0`` is required, as on a pack.
    """

    kind = "analytic"
    model = "free_water"
    oriented = False

    def __init__(self, *, D_m2_s, m0, name="csf/free-water"):
        super().__init__(name, m0)
        self.D_m2_s = float(D_m2_s)
        if self.D_m2_s <= 0:
            raise ValueError(f"D_m2_s is a diffusivity in m^2/s and must be positive: {D_m2_s}")

    def response(self, seq, pose=None):
        """``exp(-b D)`` per measurement: the sequence's declared b, else the integral of its effective gradient.
        The pose is ignored: an isotropic form has none (RPH.md 6)."""
        return np.exp(-_b_values(seq) * self.D_m2_s).astype(np.complex128)

    def to_meta(self):
        return {**self._base_meta(), "model": self.model, "params": {"diffusivity": self.D_m2_s}}

    @classmethod
    def from_meta(cls, meta):
        return cls(D_m2_s=meta["params"]["diffusivity"], m0=meta["m0"], name=meta["id"])

    def __repr__(self):
        return f"FreeWater(D_m2_s={self.D_m2_s:g}, m0={self.m0:g}, name={self.name!r})"


class Inert(_Declared):
    """Volume that fills a voxel and emits nothing, so its proton density is zero by definition and not a
    parameter. Inert is not air: an air interface dominates the field of neighbouring voxels, which a
    voxel-by-voxel phantom has no mechanism for (RPH.md 3.1)."""

    kind = "inert"

    def __init__(self, *, name="background/inert"):
        super().__init__(name, 0.0)

    def to_meta(self):
        return self._base_meta()

    @classmethod
    def from_meta(cls, meta):
        return cls(name=meta["id"])

    def __repr__(self):
        return f"Inert({self.name!r})"


_ANALYTIC = {FreeWater.model: FreeWater}


def substrate_from_meta(meta, *, pack=None):
    """The declaration object of one ``substrates[i]`` entry of a ``.rph``; ``pack`` supplies a pack substrate's
    pack (an in-memory pack, a path, or None to leave the recorded ``uri`` to resolve on first use)."""
    kind = meta.get("kind")
    if kind == "pack":
        return PackSubstrate.from_meta(meta, pack=pack if pack is not None else meta.get("uri"))
    if kind == "analytic":
        model = meta.get("model")
        cls = _ANALYTIC.get(model)
        if cls is not None:
            return cls.from_meta(meta)
        if isinstance(model, str) and ":" in model:                 # "<package>:<Name>": that package reads it
            pkg = model.split(":", 1)[0]
            import importlib
            try:
                mod = importlib.import_module(f"{pkg}.phantom")
            except ImportError as e:
                raise ValueError(f"substrate {meta.get('id')!r} names the closed form {model!r}, which the package "
                                 f"{pkg!r} defines; it is not installed here ({e}). Install it or replace the "
                                 f"substrate (RPH.md 3.1: refuse, never guess)") from e
            reader = getattr(mod, "analytic_substrate", None)
            if reader is None:
                raise ValueError(f"{pkg}.phantom defines no analytic_substrate(meta): it cannot read {model!r}")
            return reader(meta)
        raise ValueError(f"substrate {meta.get('id')!r} names the closed form {model!r}, which this "
                         f"replayer does not implement; it knows {sorted(_ANALYTIC)} and namespaced models "
                         f"'<package>:<Name>' (RPH.md 3.1: refuse, never guess)")
    if kind == "inert":
        return Inert.from_meta(meta)
    raise ValueError(f"substrate kind {kind!r} is not one of ('pack', 'analytic', 'inert')")


def _b_values(seq):
    enc = getattr(seq, "encoding", None)
    if enc is not None and getattr(enc, "bvalues", None) is not None:
        return np.asarray(enc.bvalues, np.float64).reshape(-1)
    return np.atleast_1d(np.asarray(seq.b(), np.float64))
