"""Lean forward-native pulse-sequence construction for the vector-Bloch engine.

Builds the engine-ready representation ``dmipy_sim.bloch.simulate_bloch`` consumes:
a PHYSICAL gradient ``G(t)`` on a ``dt`` grid, a finite-pulse RF event list, and an
optional emergent voxel-scale crusher.  Scoped to what the magnetization-transfer /
three-observables work needs -- a spin-echo or gradient-echo readout, optionally
preceded by an off-resonance MT-prep saturation block.  (No replay, no
susceptibility; the private repo's pgste / b-tensor / ogse families are out of scope
here -- the readout does the refocusing emergently via its 180, no sign folding.)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from .bloch import simulate_bloch

__all__ = ["BlochSequence", "gradient_echo", "spin_echo", "prepend_mt_prep",
           "run_bloch_sequence"]


@dataclass
class BlochSequence:
    """Engine-ready emergent representation of one acquisition (forward, no replay)."""
    G: np.ndarray                       # (n_meas, n_t, 3) PHYSICAL gradient, dt grid
    dt: float                           # time step (s)
    rf_events: list                     # [{t_s, flip_deg, axis_deg, duration_s, offset_hz}]
    complex_signal: bool = True         # True -> read (Mx+iMy); False -> Re only
    echo_steps: list = None             # multi-echo sample steps; None -> read the last step
    crusher: dict = None                # {'windows_s':[(t0,t1)], 'n_cycles':...}
    family: str = "gre"
    notes: str = ""

    @property
    def n_t(self) -> int:
        return self.G.shape[1]

    @property
    def n_meas(self) -> int:
        return self.G.shape[0]


# ── readout builders ────────────────────────────────────────────────────────────
def gradient_echo(TE, dt, *, n_meas=1, exc_axis_deg=90.0):
    """90 excitation then free evolution to ``TE``; read the transverse (complex).

    No 180, so a static off-resonance is NOT refocused -- the readout reports the
    (relaxed) transverse of whatever longitudinal magnetisation the excitation tips,
    which is exactly the MT-reduced ``Mz`` after an MT-prep block.
    """
    n_t = int(round(TE / dt)) + 1
    G = np.zeros((n_meas, n_t, 3), dtype=np.float64)
    rf = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': exc_axis_deg,
           'duration_s': 0.0, 'offset_hz': 0.0}]
    return BlochSequence(G=G, dt=dt, rf_events=rf, complex_signal=True,
                         family="gre", notes="90 + free evolution to TE (complex)")


def spin_echo(TE, dt, *, n_meas=1, exc_axis_deg=90.0):
    """90 at 0, 180 at TE/2, echo at TE (real).  Gradient is zero unless set on ``G``.

    The 180 refocuses static dephasing EMERGENTLY (it conjugates the accumulated
    phase) -- no ``eps_P`` sign, no folded/effective gradient convention.
    """
    n_t = int(round(TE / dt)) + 1
    G = np.zeros((n_meas, n_t, 3), dtype=np.float64)
    rf = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': exc_axis_deg,
           'duration_s': 0.0, 'offset_hz': 0.0},
          {'t_s': TE / 2.0, 'flip_deg': 180.0, 'axis_deg': 0.0,
           'duration_s': 0.0, 'offset_hz': 0.0}]
    return BlochSequence(G=G, dt=dt, rf_events=rf, complex_signal=False,
                         family="se", notes="90 - 180@TE/2 - echo (emergent refocusing)")


# ── MT-prep saturation block ────────────────────────────────────────────────────
def prepend_mt_prep(seq, mt_prep):
    """Prepend an off-resonance MT-prep saturation block to a readout ``seq``.

    The saturation pulse is *just another RF event* -- long, off-resonance
    (``offset_hz``) -- placed before the readout, FOLLOWED BY a SEPARATE voxel-scale
    crusher window (RF off) that dephases the residual transverse so only the
    (MT-reduced) longitudinal magnetisation is excited by the readout.  A crusher
    concurrent with the pulse would be continuously refilled by the RF, so it must be
    its own window afterwards.  The crusher is modelled at the mm voxel scale it
    physically acts on (a um cell-scale gradient cannot wind >> 2 pi across the cell).

    ``mt_prep`` keys: ``offset_hz``, ``duration_s`` (sat pulse), ``b1_hz``
    (= gamma*B1/2pi -> ``flip_deg = 360*b1_hz*duration``) or ``flip_deg``; optional
    ``axis_deg`` (0), ``spoiler_s`` (crusher window, default 0.5 ms), ``n_cycles``
    (crusher strength, default 32).
    """
    dt = seq.dt
    off = float(mt_prep['offset_hz'])
    dur = float(mt_prep['duration_s'])
    axis = float(mt_prep.get('axis_deg', 0.0))
    flip = (float(mt_prep['flip_deg']) if mt_prep.get('flip_deg') is not None
            else 360.0 * float(mt_prep['b1_hz']) * dur)
    n_sat = max(1, int(round(dur / dt)))
    n_spoil = int(round(float(mt_prep.get('spoiler_s', 0.5e-3)) / dt))
    shift = n_sat + n_spoil                              # sat pulse THEN a crusher

    Gpre = np.zeros((seq.n_meas, shift, 3), dtype=seq.G.dtype)   # no gradient in the prep
    G_new = np.concatenate([Gpre, seq.G], axis=1)
    t_shift = shift * dt
    sat = {'t_s': dur / 2.0, 'flip_deg': flip, 'axis_deg': axis,
           'duration_s': dur, 'offset_hz': off}          # RF only over [0, dur]
    rf_new = [sat] + [{**e, 't_s': float(e['t_s']) + t_shift} for e in seq.rf_events]
    echo_new = (None if seq.echo_steps is None
                else [int(s) + shift for s in seq.echo_steps])
    spoil_win = (n_sat * dt, (n_sat + n_spoil) * dt)
    crush = (dict(windows_s=[spoil_win], n_cycles=float(mt_prep.get('n_cycles', 32.0)))
             if n_spoil > 0 else seq.crusher)
    return replace(seq, G=G_new, rf_events=rf_new, echo_steps=echo_new, crusher=crush,
                   notes=seq.notes + f"; MT-prep {off:.0f} Hz / {dur*1e3:.0f} ms")


# ── run a BlochSequence through the forward engine ──────────────────────────────
def run_bloch_sequence(seq, n_walkers, diffusivity, geometry, *, seed=0, **kw):
    """Run a :class:`BlochSequence` through ``simulate_bloch`` and return the signal.

    Extra keywords (``T2``, ``T1``, ``M0``, ``off_resonance_hz``, ``kappa_MT``,
    ``dwell_time``, ``T2_bound``, ``T1_bound``, ``off_resonance_bound``,
    ``return_mz``, ``require_gpu``) pass straight to ``simulate_bloch``.  With
    ``seq.echo_steps`` None the last step (the readout echo) is returned.
    """
    wf = SimpleNamespace(G=np.asarray(seq.G), dt=float(seq.dt))
    return simulate_bloch(n_walkers, diffusivity, wf, geometry, seq.rf_events,
                          seed=seed, echo_steps=seq.echo_steps, crusher=seq.crusher, **kw)
