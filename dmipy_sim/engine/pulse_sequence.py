"""Bloch-side sequence builders: bare readouts for the vector-Bloch engine and the MT preparation block.

Everything here is a :class:`~dmipy_replay.acquisition.scanner_sequence.ScannerSequence` -- the physical gradient,
the schedule, the readout, and where the engine models one, the emergent voxel-scale ``crusher`` (windings over
windows; a um cell-scale gradient cannot wind >> 2 pi across a cell, so the crusher acts at the mm scale it
physically has). The vector-Bloch oracle (``engine.bloch._simulate_bloch``) runs one, applying its pulses
itself and so reads ``G``; a spin echo refocuses EMERGENTLY there, with no sign folding.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from dmipy_replay.acquisition.rf import RFEvent, RFSchedule
from dmipy_replay.acquisition.scanner_sequence import ScannerSequence
from .bloch import _simulate_bloch

__all__ = ["bare_gradient_echo", "bare_spin_echo", "saturation_pulse", "prepend_mt_prep",
           "emergent_z_spectrum"]


# ── bare readouts (no gradient) ─────────────────────────────────────────────────
def bare_gradient_echo(TE, dt, *, n_meas=1, exc_axis_deg=90.0):
    """A 90 then free evolution to ``TE``, no gradient: read the transverse. No 180, so a static off-resonance is
    NOT refocused -- the readout reports the (relaxed) transverse of whatever longitudinal magnetisation the
    excitation tips, which is exactly the MT-reduced ``Mz`` after an MT-prep block."""
    n_t = int(round(TE / dt)) + 1
    return ScannerSequence(G=np.zeros((n_meas, n_t, 3)), dt=dt, rf=(RFEvent(0.0, 90.0, 'Mz→Mxy', axis_deg=exc_axis_deg),),
                           family="gre", notes="90 + free evolution to TE")


def bare_spin_echo(TE, dt, *, n_meas=1, exc_axis_deg=90.0):
    """A 90 at 0, a 180 at TE/2, the echo at TE, no gradient. The 180 refocuses static dephasing EMERGENTLY in the
    vector-Bloch engine (it conjugates the accumulated phase)."""
    n_t = int(round(TE / dt)) + 1
    return ScannerSequence(G=np.zeros((n_meas, n_t, 3)), dt=dt,
                           rf=(RFEvent(0.0, 90.0, 'Mz→Mxy', axis_deg=exc_axis_deg), RFEvent(TE / 2.0, 180.0, 'refocus')),
                           family="se", notes="90 - 180@TE/2 - echo")


# ── MT-prep saturation block ────────────────────────────────────────────────────
def saturation_pulse(offset_hz, duration_s, *, b1_hz=None, flip_deg=None, axis_deg=0.0):
    """The off-resonance MT saturation pulse as an :class:`RFEvent` (label ``'saturate'``), centred on its own
    duration. Its flip is ``flip_deg``, or from the continuous-wave amplitude ``b1_hz = gamma B1 / 2 pi`` as
    ``360 * b1_hz * duration_s``; give exactly one."""
    if (b1_hz is None) == (flip_deg is None):
        raise ValueError("give exactly one of b1_hz and flip_deg")
    flip = float(flip_deg) if flip_deg is not None else 360.0 * float(b1_hz) * float(duration_s)
    return RFEvent(float(duration_s) / 2.0, flip, 'saturate', axis_deg=axis_deg, duration_s=duration_s,
                   offset_hz=offset_hz)


def prepend_mt_prep(seq, sat, *, spoiler_s=0.5e-3, n_cycles=32.0):
    """Prepend an off-resonance MT-prep block -- the saturation pulse ``sat`` then a crusher -- to a readout.

    ``sat`` is the saturation pulse as an :class:`RFEvent` (:func:`saturation_pulse` builds it): long,
    off-resonance (``offset_hz``), *just another RF event*, placed over ``[0, sat.duration_s)`` and FOLLOWED
    BY a SEPARATE voxel-scale crusher window of ``spoiler_s`` (RF off) that dephases the residual transverse
    so only the (MT-reduced) longitudinal magnetisation is excited by the readout. A crusher concurrent with
    the pulse would be continuously refilled by the RF, so it must be its own window afterwards.
    """
    if not isinstance(sat, RFEvent):
        raise TypeError(f"sat is the saturation pulse as an RFEvent (saturation_pulse(...) builds it), got {type(sat).__name__}")
    dt = seq.dt
    dur = float(sat.duration_s)
    if dur <= 0.0:
        raise ValueError("the saturation pulse needs a duration_s > 0")
    n_sat = max(1, int(round(dur / dt)))
    n_spoil = int(round(float(spoiler_s) / dt))
    shift = n_sat + n_spoil                              # sat pulse THEN a crusher

    Gpre = np.zeros((seq.n_meas, shift, 3), dtype=seq.G.dtype)   # no gradient in the prep
    G_new = np.concatenate([Gpre, seq.G], axis=1)
    t_shift = shift * dt
    sat = replace(sat, t_s=dur / 2.0)                    # RF only over [0, dur]
    rf_new = RFSchedule((sat,) + seq.rf.shifted(t_shift))
    readout_new = tuple(int(s) + shift for s in seq.readout)
    spoil_win = (n_sat * dt, (n_sat + n_spoil) * dt)
    crush = (dict(windows_s=[spoil_win], n_cycles=float(n_cycles)) if n_spoil > 0 else seq.crusher)
    return replace(seq, G=G_new, rf=rf_new, readout=readout_new, crusher=crush,
                   notes=seq.notes + f"; MT-prep {sat.offset_hz:.0f} Hz / {dur*1e3:.0f} ms")


# ── run through the forward engine ──────────────────────────────────────────────
# ── turnkey emergent Z-spectrum sweep ─────────────────────────────────────────────
def emergent_z_spectrum(offsets_hz, geometry, *, n_walkers, diffusivity, w1_hz, t_sat, dt,
                        T2, kappa_MT, dwell_time, T1=1.0, T2_bound=1e-5, T1_bound=1.0,
                        equilibrate_binding="burnin", seed=0):
    """Emergent CW-saturation Z-spectrum from the forward vector-Bloch engine.

    For each off-resonance ``offset`` (Hz), apply a continuous-wave saturation pulse of
    nutation ``w1_hz`` (= gamma*B1/2pi) and duration ``t_sat`` (s) to the MT-binding
    substrate, then read the walker-mean longitudinal magnetization ``Mz`` (normalised to
    ``M0=1``).  The broad, short-``T2_bound`` bound pool saturates over a wide offset range
    while the narrow free-water line is spared -> the emergent MT dip.  No super-Lorentzian
    lineshape is imposed; the dip is produced by real short-``T2_bound`` spins.

    This is the *emergent* (Monte-Carlo) counterpart of the analytic two-pool oracle
    :func:`dmipy_analytic.relaxation.mt_z_spectrum`; the two agree to the MC noise floor once the bound
    pool is burned in (``equilibrate_binding`` other than ``'off'``).  Fine ``dt`` is
    required so the carrier ``2*pi*offset*dt`` does not alias.

    Returns
    -------
    numpy.ndarray
        ``Mz`` of the free pool at each offset, shape ``(len(offsets_hz),)``.
    """
    offsets = np.atleast_1d(np.asarray(offsets_hz, dtype=float))
    n_t = int(round(float(t_sat) / float(dt))) + 1
    mz = np.empty(offsets.shape, dtype=float)
    for i, off in enumerate(offsets):
        sat = saturation_pulse(float(off), float(t_sat), b1_hz=float(w1_hz))          # the CW flip over the window
        seq = ScannerSequence(G=np.zeros((1, n_t, 3)), dt=float(dt), rf=(sat,), family="mt-sat")
        _, m = _simulate_bloch(n_walkers, diffusivity, seq, geometry,
                              T2=T2, T1=T1, kappa_MT=kappa_MT, dwell_time=dwell_time,
                              T2_bound=T2_bound, T1_bound=T1_bound, return_mz=True,
                              equilibrate_binding=equilibrate_binding, seed=seed)
        mz[i] = float(m[0])
    return mz
