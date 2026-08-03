"""Shared prewalked substrates for the test suite (Monte-Carlo replay).

A walker trajectory depends ONLY on ``(geometry, diffusivity, seed, n_walkers,
T_max, dt_save)`` — the *walk signature*.  The gradient waveform, b-value, T2, T1,
surface relaxivity ρ, etc. are *replay knobs* applied post-hoc.  So many tests that
today re-walk the same substrate for different acquisitions can instead walk ONCE and
replay each acquisition, a large wall-clock win.

This module provides:

- :class:`PrewalkSpec` — the walk signature (geometry factory + walk params).
- :func:`get_prewalk` — walk once, memoised per session AND cached on disk keyed by a
  hash of the signature *and the walk-relevant engine code* (so a change to a replay
  operator reuses cached walks, while a change to the walk kernels invalidates them —
  the walk/replay dependency split).
- :class:`Prewalk` — the saved walk with a :meth:`Prewalk.replay` helper wrapping
  ``replay``.
- :func:`spec_for_waveform` — derive a spec whose grid matches a waveform exactly (so
  the replay introduces no resampling error).

Only use this for PHYSICS-vs-analytic tests where the engine is a means.  Do NOT use it
for tests whose system-under-test is the engine/kernel itself (engine dispatch, forward
Bloch, replay parity, permeability-crossing, myelin kernel).

Cache dir: ``$DMIPY_TEST_WALK_CACHE`` or ``tests/_walkcache/`` (git-ignored). Delete it
to force a clean re-walk.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from dmipy_sim import simulate_trajectories
from dmipy_sim.trajectories import replay

# Walk-relevant engine modules: a change to any of these invalidates cached walks.
# Replay-only modules (trajectories.py, waveforms.py) are deliberately EXCLUDED — editing
# a replay operator must NOT force a re-walk (that is the whole point of the split).
_WALK_CODE_MODULES = ("geometries.py", "physics.py", "mesh.py", "core.py", "constants.py")


def _walk_code_hash() -> str:
    """Hash the walk-relevant engine source so cached walks invalidate when the walk
    physics changes (but survive edits to replay operators)."""
    import dmipy_sim
    root = Path(dmipy_sim.__file__).parent
    h = hashlib.sha256()
    for name in _WALK_CODE_MODULES:
        p = root / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _cache_dir() -> Path:
    d = os.environ.get("DMIPY_TEST_WALK_CACHE")
    p = Path(d) if d else Path(__file__).parent / "_walkcache"
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class PrewalkSpec:
    """A walk signature. ``geom_key`` MUST fully identify the geometry that
    ``geom_factory()`` builds (it is what goes into the cache key — geometry objects are
    not hashed)."""
    geom_key: str
    geom_factory: Callable = field(compare=False, repr=False)
    D: float = 2e-9
    seed: int = 123
    n_walkers: int = 100_000
    T_max: float = 60e-3
    dt_save: float = 1e-4
    save_relaxation_data: bool = True

    def signature(self) -> str:
        raw = (f"{self.geom_key}|D={self.D:.6e}|seed={self.seed}|N={self.n_walkers}|"
               f"Tmax={self.T_max:.9e}|dt={self.dt_save:.9e}|"
               f"relax={self.save_relaxation_data}|walkcode={_walk_code_hash()}")
        return hashlib.sha256(raw.encode()).hexdigest()[:24]


class Prewalk:
    """A saved walk + a replay helper. ``traj`` is float16 positions ``(N, n_t, 3)``."""

    def __init__(self, spec: PrewalkSpec, traj, dt, sub_steps, dt_sim,
                 dlog=None, comp=None):
        self.spec = spec
        self.traj = traj
        self.dt = float(dt)
        self.sub_steps = int(sub_steps)
        self.dt_sim = float(dt_sim)
        self.dlog = dlog          # boundary local time (ρ/D=1) or None
        self.comp = comp          # per-step compartment / occupancy or None
        self.D = float(spec.D)

    @property
    def n_walkers(self):
        return self.traj.shape[0]

    def replay(self, waveform, *, T2=None, T1=None, rho=None,
               T2_per_comp=None, T1_per_comp=None, chi_perp=None,
               stimulated_echo=None, n_walkers=None):
        """Replay ``waveform`` on the stored walk. Mirrors the physics of
        ``simulate(..., T2=, T1=)`` with surface relaxivity ρ (applied off the stored
        boundary channel) and per-compartment relaxation.

        ``n_walkers`` (< the walk's N) subsamples the leading walkers — useful when a
        test wants a smaller ensemble than the shared canonical walk.
        """
        G = np.asarray(waveform.G)
        dt_wf = float(waveform.dt)
        traj = self.traj if n_walkers is None else self.traj[:n_walkers]
        if chi_perp is None:
            chi_perp = getattr(waveform, "chi_perp", None)
        if chi_perp is None:
            chi_perp = np.ones(G.shape[1], dtype=np.float64)
        if stimulated_echo is None:
            stimulated_echo = bool(getattr(waveform, "stimulated_echo", False))
        kw = {}
        if rho is not None:
            if self.dlog is None:
                raise ValueError("rho replay needs a walk with save_relaxation_data=True")
            dlog = self.dlog if n_walkers is None else self.dlog[:n_walkers]
            kw.update(surface_relaxivity=rho, D=self.D, dlog_boundary_unit=dlog)
        if T2_per_comp is not None or T1_per_comp is not None:
            if self.comp is None:
                raise ValueError("per-compartment replay needs save_relaxation_data=True")
            comp = self.comp if n_walkers is None else self.comp[:n_walkers]
            kw.update(comp_traj=comp)
            if T2_per_comp is not None:
                kw["T2_per_comp"] = T2_per_comp
            if T1_per_comp is not None:
                kw["T1_per_comp"] = T1_per_comp
        return replay(
            traj, self.dt, G, dt_wf, chi_perp=chi_perp, T2=T2, T1=T1,
            stimulated_echo=stimulated_echo, **kw)

    def surface_b0_signal(self, TE, rho, *, n_walkers=None):
        """b≈0 spin-echo signal ``E(TE) = <exp((ρ/D)·Σ_{t≤TE} dlog_boundary_unit)>``,
        the quantity a surface-relaxivity T2 fit measures (diffusion weighting is
        negligible at b≈0, and constant across TE so it never affects the T2 slope).

        The walk is truncated at ``TE`` — so ONE long prewalk serves an entire TE sweep,
        and (ρ being a replay knob off the boundary channel) all ρ share that walk.
        Returns ``(E, actual_TE)``. Requires ``save_relaxation_data=True``.
        """
        if self.dlog is None:
            raise ValueError("surface_b0_signal needs a walk with save_relaxation_data=True")
        n_t = min(self.traj.shape[1], int(round(float(TE) / self.dt)) + 1)
        dl = self.dlog[:, :n_t] if n_walkers is None else self.dlog[:n_walkers, :n_t]
        logw = (float(rho) / self.D) * np.asarray(dl, dtype=np.float64).sum(axis=1)
        return float(np.mean(np.exp(logw))), (n_t - 1) * self.dt


_MEM: dict[str, Prewalk] = {}


def get_prewalk(spec: PrewalkSpec) -> Prewalk:
    """Walk once for ``spec``; reuse a session (memory) or on-disk cached walk when the
    signature — including the walk-code hash — matches."""
    sig = spec.signature()
    if sig in _MEM:
        return _MEM[sig]

    path = _cache_dir() / f"{spec.geom_key}.{sig}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        pw = Prewalk(spec, z["traj"], float(z["dt"]), int(z["sub_steps"]),
                     float(z["dt_sim"]),
                     dlog=z["dlog"] if "dlog" in z.files else None,
                     comp=z["comp"] if "comp" in z.files else None)
        _MEM[sig] = pw
        return pw

    res = simulate_trajectories(
        spec.n_walkers, spec.D, spec.geom_factory(), T_max=spec.T_max,
        dt_save=spec.dt_save, seed=spec.seed,
        save_relaxation_data=spec.save_relaxation_data, require_gpu=False)
    traj, dt, sub_steps, dt_sim = res[0], res[1], res[2], res[3]
    dlog = res[4] if len(res) >= 6 else None
    comp = res[5] if len(res) >= 6 else None
    pw = Prewalk(spec, np.asarray(traj), dt, sub_steps, dt_sim, dlog=dlog, comp=comp)

    save = {"traj": pw.traj, "dt": pw.dt, "sub_steps": pw.sub_steps, "dt_sim": pw.dt_sim}
    if dlog is not None:
        save["dlog"] = np.asarray(dlog)
    if comp is not None:
        save["comp"] = np.asarray(comp)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **save)
    os.replace(tmp, path)     # atomic publish (safe under xdist)
    _MEM[sig] = pw
    return pw


def spec_for_waveform(geom_key, geom_factory, waveform, *, D=2e-9, seed=123,
                      n_walkers=100_000, save_relaxation_data=True) -> PrewalkSpec:
    """A spec whose grid matches ``waveform`` exactly (``dt_save == waveform.dt`` and
    ``n_t`` equal), so the replay does no G-resampling and is bit-faithful to a fused
    walk on the same grid."""
    n_t = int(np.asarray(waveform.G).shape[1])
    dt = float(waveform.dt)
    return PrewalkSpec(geom_key=geom_key, geom_factory=geom_factory, D=D, seed=seed,
                       n_walkers=n_walkers, T_max=(n_t - 1) * dt, dt_save=dt,
                       save_relaxation_data=save_relaxation_data)
