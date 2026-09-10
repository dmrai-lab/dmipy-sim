"""Bloch-side sequence builders: bare readouts for the vector-Bloch engine, FEXI, and the MT preparation block.

Everything here is a :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence` -- the physical gradient,
the schedule, the readout, and where the engine models one, the emergent voxel-scale ``crusher`` (windings over
windows; a um cell-scale gradient cannot wind >> 2 pi across a cell, so the crusher acts at the mm scale it
physically has). :func:`run_bloch_sequence` drives one through ``simulate_bloch``, which applies the pulses
itself and so reads ``G``; a spin echo refocuses EMERGENTLY there, with no sign folding.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..acquisition.rf import RFEvent, RFSchedule
from ..acquisition.scanner_sequence import ScannerSequence
from .bloch import simulate_bloch
from ..constants import GAMMA

__all__ = ["bare_gradient_echo", "bare_spin_echo", "fexi", "saturation_pulse", "prepend_mt_prep",
           "run_bloch_sequence", "emergent_z_spectrum"]


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


def fexi(delta, t_mix, dt, *, g_filter, g_detect, Delta=None, delta_detect=None,
         Delta_detect=None, direction=(1., 0., 0.), crush_cycles=32.0, exc_axis_deg=0.0):
    """FEXI (filter-exchange) stimulated-echo diffusion sequence.

    A double-diffusion-encoding stimulated echo for measuring water exchange (Lasič et al.
    2011): a self-refocused **diffusion filter** dephases fast-diffusing water, a 90° stores
    the survivor longitudinally over a **mixing time** ``t_mix`` (during which nothing encodes
    but walkers keep diffusing and *exchanging* across membranes), a 90° recalls it, and a
    second self-refocused **detection** block measures the apparent diffusivity. As ``t_mix``
    grows, the filtered ADC recovers toward equilibrium at the exchange rate (AXR).

    Structure — each encoding block is a **bipolar pair** (two inverted lobes, ``+g … −g``), the
    "PGSTE with 2 lobes each side" (Kiselev & Li 2026):

        90 ─[+g_f … −g_f]─ 90(store) ─·crusher·─ t_mix ─ 90(recall) ─[+g_d … −g_d]─ echo
             └── filter ──┘            └── longitudinal storage; EXCHANGE ──┘  └── detection ──┘

    Each bipolar block self-refocuses ``q→0`` (no 180 needed), so — unlike PGSTE — the mixing time
    carries no diffusion encoding (pure exchange weighting). The lobe separation ``Delta`` sets the
    diffusion time; a contiguous pair (``Delta=delta``) has too short a time to separate a
    restricted pool (raise ``Delta`` / the gradient). Runs through :func:`simulate_bloch` (the
    crusher + stimulated-echo storage select the filtered pathway — a scalar ``chi_perp`` walk
    cannot); exchange needs a **permeable** substrate. Returns a ``ScannerSequence`` whose
    ``notes`` name it; the per-measurement detection b-value is :func:`fexi_b_detect`.

    Parameters
    ----------
    delta : float
        Duration of each filter (and, unless ``delta_detect``, detection) gradient lobe (s).
    t_mix : float
        Mixing time (s) — the longitudinal-storage / exchange period.
    dt : float
        Time step (s).
    g_filter : float
        Filter gradient amplitude (T/m) — the fixed diffusion filter (suppresses fast water).
    g_detect : float or array
        Detection gradient amplitude(s) (T/m); an array gives one measurement per value (e.g.
        ``[0, g]`` to fit an ADC).
    Delta : float, optional
        Filter lobe **separation** (leading edge to leading edge, s); the diffusion time. A gap
        ``Delta − delta`` is inserted between the two lobes — needed for restriction/ADC contrast
        (a contiguous bipolar pair, the default ``Delta = delta``, has too short a diffusion time
        to separate a restricted pool). Typical FEXI ``δ/Δ ≈ 4/15 ms``.
    delta_detect, Delta_detect : float, optional
        Detection lobe duration / separation (s); default to ``delta`` / ``Delta``.
    direction : (3,) array
        Gradient direction (filter and detection share it).
    crush_cycles : float
        Voxel-scale crusher strength over the mixing window (dephases the non-stored pathway).
    exc_axis_deg : float
        B1 phase of the three 90° pulses (they share an axis so the store keeps ``cos φ``).
    """
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    g_detect = np.atleast_1d(np.asarray(g_detect, dtype=np.float64))
    n_meas = g_detect.shape[0]
    Delta = delta if Delta is None else Delta
    dd = delta if delta_detect is None else delta_detect
    Dd = Delta if Delta_detect is None else Delta_detect
    ndf = int(round(delta / dt)); ngf = max(1, int(round((Delta - delta) / dt)))
    ndd = int(round(dd / dt)); ngd = max(1, int(round((Dd - dd) / dt)))
    nmix = int(round(t_mix / dt))
    # Bipolar blocks (Kiselev & Li 2026): each encoding is two INVERTED lobes (+g … −g) that
    # self-refocus q → 0 without a 180 — the standard FEXI filter. The gap ``Delta − delta`` sets
    # the diffusion time (a contiguous pair has too short a time to feel restriction). No RF
    # inside the blocks, so the only pulses are the three stimulated-echo 90s.
    i_store = 2 * ndf + ngf
    i_recall = i_store + nmix
    n_t = i_recall + 2 * ndd + ngd + 1

    G = np.zeros((n_meas, n_t, 3), dtype=np.float64)
    for m in range(n_meas):
        G[m, 0:ndf] = g_filter * d                                  # filter lobe +
        G[m, ndf + ngf:2 * ndf + ngf] = -g_filter * d               # filter lobe − (inverted)
        G[m, i_recall:i_recall + ndd] = g_detect[m] * d             # detection lobe +
        G[m, i_recall + ndd + ngd:i_recall + 2 * ndd + ngd] = -g_detect[m] * d

    rf = tuple(RFEvent(i * dt, 90.0, lab, axis_deg=exc_axis_deg)
               for i, lab in zip((0, i_store, i_recall), ('Mz→Mxy', 'store', 'recall')))
    crusher = {'windows_s': [((i_store + 1) * dt, (i_recall - 1) * dt)],
               'n_cycles': float(crush_cycles)}
    return ScannerSequence(G=G, dt=dt, rf=rf, crusher=crusher, family="fexi",
                           notes="PGSE filter - store - t_mix (exchange) - recall - PGSE detect")


def fexi_b_detect(seq):
    """The detection b-value per measurement (s/m^2) of a :func:`fexi` sequence: ``q = gamma int G dt`` over the
    self-refocused detection block after the recall."""
    i_recall = int(round(seq.rf[2].t_s / seq.dt))
    b = np.empty(seq.n_meas)
    for m in range(seq.n_meas):
        qd = GAMMA * np.cumsum(np.asarray(seq.G[m, i_recall:, :], np.float64), axis=0) * seq.dt
        b[m] = float(np.sum(qd ** 2) * seq.dt)
    return b


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
def run_bloch_sequence(seq, n_walkers, diffusivity, geometry, *, seed=0, **kw):
    """Run a :class:`ScannerSequence` through ``simulate_bloch`` and return the signal.

    Extra keywords (``T2``, ``T1``, ``M0``, ``off_resonance_hz``, ``kappa_MT``, ``dwell_time``, ``T2_bound``,
    ``T1_bound``, ``off_resonance_bound``, ``return_mz``, ``require_gpu``) pass straight to ``simulate_bloch``.
    A readout at the last sample returns that echo; a multi-echo readout returns every echo.
    """
    echo_steps = None if seq.readout == (seq.n_t - 1,) else list(seq.readout)
    return simulate_bloch(n_walkers, diffusivity, seq, geometry, seq.rf,
                          seed=seed, echo_steps=echo_steps, crusher=seq.crusher, **kw)


# ── turnkey emergent Z-spectrum sweep ─────────────────────────────────────────────
def emergent_z_spectrum(offsets_hz, geometry, *, n_walkers, diffusivity, w1_hz, t_sat, dt,
                        T2, kappa_MT, dwell_time, T1=1.0, T2_bound=1e-5, T1_bound=1.0,
                        equilibrate_binding="auto", seed=0):
    """Emergent CW-saturation Z-spectrum from the forward vector-Bloch engine.

    For each off-resonance ``offset`` (Hz), apply a continuous-wave saturation pulse of
    nutation ``w1_hz`` (= gamma*B1/2pi) and duration ``t_sat`` (s) to the MT-binding
    substrate, then read the walker-mean longitudinal magnetization ``Mz`` (normalised to
    ``M0=1``).  The broad, short-``T2_bound`` bound pool saturates over a wide offset range
    while the narrow free-water line is spared -> the emergent MT dip.  No super-Lorentzian
    lineshape is imposed; the dip is produced by real short-``T2_bound`` spins.

    This is the *emergent* (Monte-Carlo) counterpart of the analytic two-pool oracle
    :func:`dmipy_sim.engine.mt.mt_z_spectrum`; the two agree to the MC noise floor once the bound
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
        _, m = simulate_bloch(n_walkers, diffusivity, seq, geometry, seq.rf,
                              T2=T2, T1=T1, kappa_MT=kappa_MT, dwell_time=dwell_time,
                              T2_bound=T2_bound, T1_bound=T1_bound, return_mz=True,
                              equilibrate_binding=equilibrate_binding, seed=seed)
        mz[i] = float(m[0])
    return mz
