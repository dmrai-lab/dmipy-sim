"""The sequence builders: each family of acquisition as one :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence`.

A builder declares the pulses it plays, places each measurement's gradient relative to them (lobes symmetric
about a 180, whatever the row's own timing), scales so the declared ``bvalues`` are the numeric b of ``G_eff``
exactly, attaches the per-measurement :class:`~dmipy_sim.acquisition.scanner_sequence.Encoding` an analytical
layer reads, and ``validate()``\\ s. ``G`` is the PHYSICAL gradient throughout; the effective one is derived.

Spin echoes: :func:`pgse`, :func:`ogse` (two trains about a 180; its ``slew_rate=np.inf`` limit is one
continuous cosine with no pulse), :func:`cpmg`, :func:`from_btensor_waveform`. Stimulated echoes (three
pulses, no 180; the static field refocuses only when the two transverse periods match): :func:`pgste`,
:func:`from_pgste_waveform`. Gradient echoes (self-refocused encodings, an excitation only): :func:`gre`,
:func:`ste`, :func:`pte`, :func:`from_waveform`. :func:`instantaneous` rebuilds any of them at infinite slew.
"""
from __future__ import annotations

import numpy as np

from ..acquisition.rf import RFEvent, RFSchedule
from ..acquisition.scanner_sequence import Encoding, ScannerSequence
from ..math.gradient_conversions import g_from_b, q_from_b
from ..constants import GAMMA, DEFAULT_SLEW_RATE, resolve_slew as _resolve_slew
from ._helpers import (
    _trap_profile, _trap_cosine_profile, _calc_b_from_waveform, _resolve_te, _scale_to_b,
    unify_length_reference_delta_Delta, check_acquisition_scheme,
)

__all__ = ["pgse", "pgste", "gre", "cpmg", "ogse", "ste", "pte", "from_waveform", "from_btensor_waveform",
           "from_pgste_waveform", "instantaneous", "to_gradient_array"]


def pgse(bvalues, gradient_directions, delta, Delta, TE=None, n_t=1000, slew_rate=DEFAULT_SLEW_RATE, timing=None):
    """PGSE: two same-sign lobes separated by Delta, the 180 midway between them.

    Every measurement's lobe pair is centred on the one 180 at ``T_total / 2`` (``T_total`` is the longest
    row's ``Delta + delta + ramp``), so a row with a shorter ``Delta`` sits inside the same echo with its 180 in
    its own gap rather than starting at t = 0 and having the pulse land inside its second lobe. Slew-limited
    (realizable) by default (``slew_rate`` in T/m/s); ``slew_rate=np.inf`` is the instantaneous (square) limit
    -- vertical ramps, same structure.

    ``timing`` (a :class:`~dmipy_sim.acquisition.timing.SequenceTiming`) builds to a budget: the pulses take
    their durations, the pair sits inside the two encoding windows, ``TE`` is the budget's (or the smallest that
    fits, ``Delta + delta + ramp + 2 max(lead-in, readout tail)``), the coherence mask is fractional across the
    pulses, and a gap ``Delta - delta - ramp`` narrower than the refocusing window is refused. Without it the
    pulses are instantaneous and the echo forms at ``Delta + delta + ramp``.
    """
    bvalues = np.asarray(bvalues, dtype=np.float64)
    gradient_directions = np.asarray(gradient_directions, dtype=np.float64)
    delta_, Delta_, TE_in = unify_length_reference_delta_Delta(bvalues, delta, Delta, TE)
    check_acquisition_scheme(bvalues, gradient_directions, delta_, Delta_, TE_in)

    gradient_strengths = g_from_b(bvalues, delta_, Delta_)
    qvalues = q_from_b(bvalues, delta_, Delta_)

    n_m = len(bvalues)
    slew_rate, square = _resolve_slew(slew_rate)
    if square:
        eps_ = np.zeros(n_m)
    else:
        eps_ = np.minimum(gradient_strengths / float(slew_rate), delta_)
    span = Delta_ + delta_ + eps_                                     # each row's lobe pair, end to end
    if timing is None:
        # ideal instantaneous 90/180: the echo forms at the END of the grid (the longest row's span), so the
        # 180 sits at T_total / 2 and every row's pair is centred on it -- a shorter row is shifted right by
        # half its slack instead of starting at t = 0 with the pulse inside its second lobe
        T_total = float(np.max(span))
        TE_, te_auto = _resolve_te(TE, T_total, n_m)
        schedule = RFSchedule([RFEvent(0.0, 90, 'Mz→Mxy'), RFEvent(T_total / 2.0, 180, 'refocus')])
    else:
        # to a budget: the pulses take their durations, the pair sits inside the two encoding windows
        gap = Delta_ - delta_ - eps_
        if np.any(gap < timing.t_refocus - 1e-12):
            raise ValueError(f"the gap between the lobes, Delta - delta - ramp = {float(np.min(gap))*1e3:.3f} ms, is "
                             f"narrower than the refocusing window {timing.t_refocus*1e3:.3f} ms: the 180 does not fit")
        te_min = float(np.max(span)) + 2.0 * max(timing.t_lead, timing.t_readout_pre_echo)
        T_total = timing.resolve_TE(TE if TE is not None else (timing.TE if timing.TE is not None else te_min))
        if T_total < te_min - 1e-12:
            raise ValueError(f"TE = {T_total*1e3:.3f} ms is below the {te_min*1e3:.3f} ms this encoding needs inside "
                             f"the budget's windows")
        TE_, te_auto = np.full(n_m, T_total), TE is None and timing.TE is None
        schedule = timing.rf_events(T_total)
    dt = T_total / (n_t - 1)
    t_grid = np.arange(n_t) * dt
    shift = (T_total - span) / 2.0                                    # per row; 0 for the longest
    G_arr = np.zeros((n_m, n_t, 3), dtype=np.float64)                  # the PHYSICAL gradient
    if square:
        for m in range(n_m):
            n_pulse = max(1, round(float(delta_[m]) / dt))
            n_Delta = round(float(Delta_[m]) / dt)
            n0 = round(float(shift[m]) / dt)
            g_vec = gradient_strengths[m] * gradient_directions[m]
            G_arr[m, n0:n0 + n_pulse, :] = g_vec
            G_arr[m, n0 + n_Delta:n0 + n_Delta + n_pulse, :] = g_vec
    else:
        for m in range(n_m):
            tm = t_grid - shift[m]
            prof = (_trap_profile(tm, 0.0, delta_[m], eps_[m]) +
                    _trap_profile(tm, Delta_[m], delta_[m], eps_[m]))
            G_arr[m] = (gradient_strengths[m] * prof)[:, None] * gradient_directions[m]
    # the declared b IS the numeric b of the effective gradient the walk integrates
    sign = schedule.sign(t_grid)[None, :, None]
    G_arr = _scale_to_b(G_arr * sign, dt, bvalues) * sign

    return ScannerSequence(
        G=G_arr, dt=dt, rf=schedule, timing=timing, family='pgse',
        encoding=Encoding(bvalues=bvalues, gradient_directions=gradient_directions, TE=TE_, qvalues=qvalues,
                          gradient_strengths=gradient_strengths, delta=delta_, Delta=Delta_,
                          minimum_te=T_total, te_auto=te_auto, ramp_time=eps_),
        build_spec=('pgse', dict(bvalues=bvalues, gradient_directions=gradient_directions, delta=delta, Delta=Delta,
                                 TE=TE, n_t=n_t, slew_rate=slew_rate, timing=timing))).validate()


def pgste(bvalues, gradient_directions, delta, TM, TE=None, n_t=1000, slew_rate=DEFAULT_SLEW_RATE,
          ste_flip_angles=(90.0, 90.0, 90.0)):
    """PGSTE (stimulated echo): a dephasing lobe, longitudinal storage over ``TM``, a rephasing lobe.

    Three pulses and no 180 -- an excitation, a store that tips the encoded magnetisation onto z (only the
    stored half returns: the idealised 0.5 the engine applies), a recall after ``TM`` -- so the diffusion time
    ``Delta = delta + TM`` runs on T1, not T2. The two lobes are the same sign; the recall's sign flip folds the
    second one into ``G_eff``. The store follows the first lobe's longest ramp, the recall comes exactly ``TM``
    later, and the second lobe starts at the recall, so the schedule's mixing time is the declared one for every
    slew. ``ste_flip_angles`` are the three flips (deg), read by the analytical layer for the STE amplitude.
    """
    bvalues = np.asarray(bvalues, dtype=np.float64)
    gradient_directions = np.asarray(gradient_directions, dtype=np.float64)
    n_m = len(bvalues)
    delta_ = np.full(n_m, float(delta))
    TM_ = np.full(n_m, float(TM))
    Delta_ = delta_ + TM_
    gradient_strengths = g_from_b(bvalues, delta_, Delta_)
    qvalues = q_from_b(bvalues, delta_, Delta_)
    slew_rate, square = _resolve_slew(slew_rate)
    eps_ = np.zeros(n_m) if square else np.minimum(gradient_strengths / float(slew_rate), delta_)
    eps_max = float(np.max(eps_))
    t_store = float(delta) + eps_max
    t_recall = t_store + float(TM)
    T_total = t_recall + float(delta) + eps_max
    dt = T_total / (n_t - 1)
    a1, a2, a3 = (float(a) for a in ste_flip_angles)
    schedule = RFSchedule([RFEvent(0.0, a1, 'Mz→Mxy'), RFEvent(t_store, a2, 'store'), RFEvent(t_recall, a3, 'recall')])
    t_grid = np.arange(n_t) * dt
    G_arr = np.zeros((n_m, n_t, 3), dtype=np.float64)                  # the PHYSICAL gradient: same-sign lobes
    if square:
        n_pulse = max(1, round(float(delta) / dt))
        n_recall = round(t_recall / dt)
        for m in range(n_m):
            g_vec = gradient_strengths[m] * gradient_directions[m]
            G_arr[m, :n_pulse, :] = g_vec
            G_arr[m, n_recall:n_recall + n_pulse, :] = g_vec
    else:
        for m in range(n_m):
            prof = (_trap_profile(t_grid, 0.0, float(delta), eps_[m]) +
                    _trap_profile(t_grid, t_recall, float(delta), eps_[m]))
            G_arr[m] = (gradient_strengths[m] * prof)[:, None] * gradient_directions[m]
    sign = schedule.sign(t_grid)[None, :, None]
    G_arr = _scale_to_b(G_arr * sign, dt, bvalues) * sign
    TE_, te_auto = _resolve_te(TE, T_total, n_m)
    return ScannerSequence(
        G=G_arr, dt=dt, rf=schedule, family='pgste',
        encoding=Encoding(bvalues=bvalues, gradient_directions=gradient_directions, TE=TE_, qvalues=qvalues,
                          gradient_strengths=gradient_strengths, delta=delta_, Delta=Delta_, minimum_te=T_total,
                          te_auto=te_auto, ramp_time=eps_, tau_perp_SE=np.full(n_m, 2.0 * float(delta)),
                          ste_flip_angles=(a1, a2, a3)),
        build_spec=('pgste', dict(bvalues=bvalues, gradient_directions=gradient_directions, delta=delta, TM=TM, TE=TE,
                                  n_t=n_t, slew_rate=slew_rate, ste_flip_angles=ste_flip_angles))).validate()


def gre(TE, gradient_directions=None, bvalues=None, delta=None, Delta=None, n_t=1000):
    """Gradient echo (no 180): a self-refocusing bipolar pair per measurement, or a pure FID.

    The grid spans the longest echo time; diffusion lobes (if any) sit at the start, the rest is zero-gradient
    precession. With no refocusing pulse the physical and effective gradients coincide and a static field is
    not refocused -- the readout is complex.
    """
    TE_ = np.atleast_1d(np.asarray(TE, dtype=np.float64))
    n_m = len(TE_)
    if bvalues is None:
        bvalues = np.zeros(n_m, dtype=np.float64)
    bvalues = np.broadcast_to(np.asarray(bvalues, dtype=np.float64), (n_m,)).copy()
    if gradient_directions is None:
        gradient_directions = np.tile([0.0, 0.0, 1.0], (n_m, 1))
    gradient_directions = np.asarray(gradient_directions, dtype=np.float64)
    has_diff = bool(np.any(bvalues > 0))
    if has_diff:
        if delta is None or Delta is None:
            raise ValueError("delta and Delta are required for a diffusion-weighted GRE (bvalues > 0).")
        delta_ = np.broadcast_to(np.asarray(delta, float), (n_m,)).copy()
        Delta_ = np.broadcast_to(np.asarray(Delta, float), (n_m,)).copy()
        gradient_strengths = g_from_b(bvalues, delta_, Delta_)
        qvalues = q_from_b(bvalues, delta_, Delta_)
    else:
        delta_ = np.zeros(n_m); Delta_ = np.zeros(n_m)
        gradient_strengths = np.zeros(n_m)
        qvalues = np.zeros(n_m)
    T_total = float(np.max(TE_))
    dt = T_total / (n_t - 1)
    G_arr = np.zeros((n_m, n_t, 3), dtype=np.float64)
    if has_diff:
        for m in range(n_m):
            n_pulse = max(1, round(float(delta_[m]) / dt))
            n_Delta = round(float(Delta_[m]) / dt)
            g_vec = gradient_strengths[m] * gradient_directions[m]
            G_arr[m, :n_pulse, :] = g_vec
            G_arr[m, n_Delta:n_Delta + n_pulse, :] = -g_vec
        G_arr = _scale_to_b(G_arr, dt, bvalues)
    return ScannerSequence(
        G=G_arr, dt=dt, rf=RFSchedule([RFEvent(0.0, 90, 'Mz→Mxy')]), family='gre',
        encoding=Encoding(bvalues=bvalues, gradient_directions=gradient_directions, TE=TE_, qvalues=qvalues,
                          gradient_strengths=gradient_strengths, delta=delta_, Delta=Delta_, refocused=False),
        build_spec=('gre', dict(TE=TE, gradient_directions=gradient_directions, bvalues=bvalues, delta=delta,
                                Delta=Delta, n_t=n_t))).validate()


def cpmg(n_echoes, TE, bvalues=None, gradient_directions=None, beta_deg=180.0, n_t_per_echo=100):
    """CPMG multi-echo spin echo with an optional diffusion lobe per echo interval.

    This family is defined by its EFFECTIVE gradient -- a bipolar lobe pair per echo interval, so each echo
    self-refocuses -- and its physical gradient is that un-folded through the declared 180 train: a constant
    lobe per interval whose polarity alternates each echo. Echo ``k`` forms at the end of its interval; the 180s
    sit at ``(k + 1/2) TE``; the readout is every echo.
    """
    if n_t_per_echo % 2 != 0:
        raise ValueError(f"n_t_per_echo must be even, got {n_t_per_echo}.")
    n_echoes = int(n_echoes)
    TE_echo = float(TE)
    TE_ = (np.arange(n_echoes, dtype=np.float64) + 1.0) * TE_echo
    if bvalues is None:
        bvalues = np.zeros(n_echoes, dtype=np.float64)
    bvalues = np.broadcast_to(np.asarray(bvalues, float), (n_echoes,)).copy()
    if gradient_directions is None:
        gradient_directions = np.tile([0.0, 0.0, 1.0], (n_echoes, 1))
    gradient_directions = np.asarray(gradient_directions, dtype=np.float64)

    n_half = n_t_per_echo // 2
    n_t_total = n_echoes * n_t_per_echo
    dt = TE_echo / n_t_per_echo
    Delta_lobe = np.full(n_echoes, n_half * dt)
    delta_lobe = np.full(n_echoes, n_half * dt)
    gstr = np.where(bvalues > 0, g_from_b(np.maximum(bvalues, 1.0), delta_lobe, Delta_lobe), 0.0)
    qvals = np.where(bvalues > 0, q_from_b(np.maximum(bvalues, 1.0), delta_lobe, Delta_lobe), 0.0)
    schedule = RFSchedule([RFEvent(0.0, 90, 'Mz→Mxy')] +
                          [RFEvent((k + 0.5) * TE_echo, 180, 'refocus') for k in range(n_echoes)])
    G_eff = np.zeros((n_echoes, n_t_total, 3), dtype=np.float64)
    for m in range(n_echoes):
        lobe = gstr[m] * gradient_directions[m]
        for k in range(n_echoes):
            base = k * n_t_per_echo
            G_eff[m, base:base + n_half, :] = lobe
            G_eff[m, base + n_half:base + n_t_per_echo, :] = -lobe
    sign = schedule.sign(np.arange(n_t_total) * dt)[None, :, None]
    G_arr = _scale_to_b(G_eff, dt, bvalues) * sign
    return ScannerSequence(
        G=G_arr, dt=dt, rf=schedule, family='cpmg',
        encoding=Encoding(bvalues=bvalues, gradient_directions=gradient_directions, TE=TE_, qvalues=qvals,
                          gradient_strengths=gstr, delta=delta_lobe, Delta=Delta_lobe, refocused=True,
                          cpmg_n_echoes=n_echoes, cpmg_TE=TE_echo, cpmg_beta_deg=float(beta_deg),
                          n_t_per_echo=int(n_t_per_echo)),
        build_spec=('cpmg', dict(n_echoes=n_echoes, TE=TE, bvalues=bvalues, gradient_directions=gradient_directions,
                                 beta_deg=beta_deg, n_t_per_echo=n_t_per_echo))).validate()


def from_waveform(G, dt, gradient_directions, delta=None, Delta=None, TE=None, allow_unrefocused=False):
    """An arbitrary gradient waveform as an acquisition; b numerically from ``G``. No pulses are declared, so
    the gradient must refocus on its own (``allow_unrefocused=True`` turns that refusal into a warning)."""
    from warnings import warn
    G = np.asarray(G, dtype=np.float32)
    gradient_directions = np.asarray(gradient_directions, dtype=np.float64)
    bvalues = _calc_b_from_waveform(G, dt)
    delta_, Delta_, TE_ = unify_length_reference_delta_Delta(bvalues, delta, Delta, TE)
    qvalues = gradient_strengths = None
    if delta_ is not None and Delta_ is not None:
        gradient_strengths = g_from_b(bvalues, delta_, Delta_)
        qvalues = q_from_b(bvalues, delta_, Delta_)
    seq = ScannerSequence(G=G, dt=dt, family='waveform',
                          encoding=Encoding(bvalues=bvalues, gradient_directions=gradient_directions, TE=TE_,
                                            qvalues=qvalues, gradient_strengths=gradient_strengths, delta=delta_,
                                            Delta=Delta_))
    try:
        seq.validate()
    except ValueError as e:
        msg = (f"from_waveform: {e}; stationary spins will not rephase at TE. Ensure the waveform "
               f"refocuses, or pass allow_unrefocused=True if intentional.")
        if allow_unrefocused:
            warn(msg)
        else:
            raise ValueError(msg) from None
    return seq


def ogse(bvalues, gradient_directions, oscillation_frequency, gradient_duration, n_cycles=1, gradient_rise_time=0.,
         TE=None, n_t=1000, slew_rate=DEFAULT_SLEW_RATE, refocus_duration=0.0):
    """Cosine OGSE: two slew-limited trains straddling a declared 180 at ``T_total / 2`` by default;
    ``slew_rate=np.inf`` gives the idealized single continuous cosine (no 180 modelled). Each measurement's
    train pair is centred on the 180, whatever its own ``gradient_duration``."""
    bvalues = np.asarray(bvalues, dtype=np.float64)
    gradient_directions = np.asarray(gradient_directions, dtype=np.float64)
    n_m = len(bvalues)
    gamma = GAMMA

    osc_freq = np.broadcast_to(np.asarray(oscillation_frequency, float), (n_m,)).copy()
    sigma = np.broadcast_to(np.asarray(gradient_duration, float), (n_m,)).copy()
    n_cyc = np.broadcast_to(np.asarray(n_cycles, float), (n_m,)).copy()
    t_r = np.broadcast_to(np.asarray(gradient_rise_time, float), (n_m,)).copy()

    safe_sigma = np.where(sigma > 0, sigma, np.ones_like(sigma))
    safe_freq = np.where(osc_freq > 0, osc_freq, np.ones_like(osc_freq))
    G_mag = np.sqrt(bvalues * (8.0 * np.pi ** 2 * safe_freq ** 2) / (gamma ** 2 * safe_sigma))
    G_mag = np.where(bvalues > 0, G_mag, 0.0)

    gap = float(refocus_duration)
    slew_rate, square = _resolve_slew(slew_rate)
    if square:
        T_total = float(np.max(sigma))
    else:
        T_total = float(np.max(2.0 * sigma + gap))
    dt = T_total / (n_t - 1)
    t_full = np.arange(n_t) * dt
    schedule = (RFSchedule() if square else
                RFSchedule([RFEvent(0.0, 90, 'Mz→Mxy'), RFEvent(T_total / 2.0, 180, 'refocus')]))
    sign = schedule.sign(t_full)[None, :, None]
    G_arr = np.zeros((n_m, n_t, 3), dtype=np.float64)                  # the PHYSICAL gradient
    for m in range(n_m):
        if bvalues[m] <= 0 or G_mag[m] == 0:
            continue
        if square:
            n_sig = max(1, round(sigma[m] / dt))
            t = np.arange(n_sig) * dt
            g_t = G_mag[m] * np.cos(2.0 * np.pi * osc_freq[m] * t)
            G_arr[m, :n_sig, :] = g_t[:, None] * gradient_directions[m]
        else:
            sg, fm, sr = float(sigma[m]), osc_freq[m], float(slew_rate)
            shift = (T_total - (2.0 * sg + gap)) / 2.0                 # centre this row's pair on the 180
            tm = t_full - shift
            # the slew-limited trapezoidal ramps depend on the amplitude, so iterate the
            # profile to the requested b before the exact scaling below
            g_amp = G_mag[m]
            for _ in range(6):
                pre = _trap_cosine_profile(tm, sg, fm, sr, g_amp)
                post = _trap_cosine_profile(tm - (sg + gap), sg, fm, sr, g_amp)
                Gm = (pre + post)[:, None] * gradient_directions[m]
                b_m = _calc_b_from_waveform((Gm * sign[0])[None], dt)[0]
                if b_m <= 0 or abs(b_m - bvalues[m]) <= 1e-6 * bvalues[m]:
                    break
                g_amp *= np.sqrt(bvalues[m] / b_m)
            G_arr[m] = Gm
    # the declared b IS the numeric b of the effective gradient the walk integrates
    G_arr = _scale_to_b(G_arr * sign, dt, bvalues) * sign

    TE_, te_auto = _resolve_te(TE, T_total, n_m)
    qvalues = G_mag * gamma * sigma / (2.0 * np.pi)
    return ScannerSequence(
        G=G_arr, dt=dt, rf=schedule, family='ogse',
        encoding=Encoding(bvalues=bvalues, gradient_directions=gradient_directions, TE=TE_, qvalues=qvalues,
                          gradient_strengths=G_mag, minimum_te=T_total, te_auto=te_auto,
                          oscillation_frequency=osc_freq, gradient_rise_time=t_r, n_oscillation_cycles=n_cyc,
                          gradient_duration=sigma),
        build_spec=('ogse', dict(bvalues=bvalues, gradient_directions=gradient_directions,
                                 oscillation_frequency=oscillation_frequency, gradient_duration=gradient_duration,
                                 n_cycles=n_cycles, gradient_rise_time=gradient_rise_time, TE=TE, n_t=n_t,
                                 slew_rate=slew_rate, refocus_duration=refocus_duration))).validate()


def ste(bvalues, delta, Delta, TE=None, n_t=1000):
    """Spherical tensor encoding (b_delta = 0): three sequential self-refocused bipolar pairs, one per axis -- a
    gradient-echo encoding (an excitation only; no 180 is folded). Whole pairs on the grid, so each refocuses
    exactly."""
    bvalues = np.atleast_1d(np.asarray(bvalues, dtype=np.float64))
    n_m = len(bvalues)
    T_total = float(delta) + float(Delta)
    dt = T_total / (n_t - 1)
    n_seg = max(1, n_t // 6)
    G_template = np.zeros((n_t, 3), dtype=np.float32)
    for i in range(3):
        enc_start = 2 * i * n_seg
        enc_end = min((2 * i + 1) * n_seg, n_t)
        dec_end = min((2 * i + 2) * n_seg, n_t)
        G_template[enc_start:enc_end, i] = 1.0
        G_template[enc_end:dec_end, i] = -1.0
    b_unit = _calc_b_from_waveform(G_template[None], dt)[0]
    G_arr = np.zeros((n_m, n_t, 3), dtype=np.float32)
    for m in range(n_m):
        scale = float(np.sqrt(bvalues[m] / b_unit)) if b_unit > 0 else 0.0
        G_arr[m] = G_template * scale
    gradient_directions = np.tile([0., 0., 1.], (n_m, 1))
    bvalues_num = _calc_b_from_waveform(G_arr, dt)
    TE_, te_auto = _resolve_te(TE, T_total, n_m)
    return ScannerSequence(
        G=G_arr, dt=dt, rf=RFSchedule([RFEvent(0.0, 90, 'Mz→Mxy')]), family='ste',
        encoding=Encoding(bvalues=bvalues_num, gradient_directions=gradient_directions, TE=TE_,
                          minimum_te=T_total, te_auto=te_auto, refocused=False),
        build_spec=('ste', dict(bvalues=bvalues, delta=delta, Delta=Delta, TE=TE, n_t=n_t))).validate()


def pte(bvalues, plane_normal, delta, Delta, TE=None, n_t=1000):
    """Planar tensor encoding (b_delta = -0.5): two sequential self-refocused bipolar pairs in the plane -- a
    gradient-echo encoding (an excitation only; no 180 is folded)."""
    bvalues = np.atleast_1d(np.asarray(bvalues, dtype=np.float64))
    n_m = len(bvalues)
    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    ref = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    u = ref - np.dot(ref, n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    T_total = float(delta) + float(Delta)
    dt = T_total / (n_t - 1)
    n_seg = max(1, n_t // 4)
    G_template = np.zeros((n_t, 3), dtype=np.float32)
    for i, axis in enumerate([u, v]):
        enc_start = 2 * i * n_seg
        enc_end = min((2 * i + 1) * n_seg, n_t)
        dec_end = min((2 * i + 2) * n_seg, n_t)
        G_template[enc_start:enc_end] = axis.astype(np.float32)
        G_template[enc_end:dec_end] = -axis.astype(np.float32)
    b_unit = _calc_b_from_waveform(G_template[None], dt)[0]
    G_arr = np.zeros((n_m, n_t, 3), dtype=np.float32)
    for m in range(n_m):
        scale = float(np.sqrt(bvalues[m] / b_unit)) if b_unit > 0 else 0.0
        G_arr[m] = G_template * scale
    gradient_directions = np.tile(u, (n_m, 1))
    bvalues_num = _calc_b_from_waveform(G_arr, dt)
    TE_, te_auto = _resolve_te(TE, T_total, n_m)
    return ScannerSequence(
        G=G_arr, dt=dt, rf=RFSchedule([RFEvent(0.0, 90, 'Mz→Mxy')]), family='pte',
        encoding=Encoding(bvalues=bvalues_num, gradient_directions=gradient_directions, TE=TE_,
                          minimum_te=T_total, te_auto=te_auto, refocused=False),
        build_spec=('pte', dict(bvalues=bvalues, plane_normal=plane_normal, delta=delta, Delta=Delta, TE=TE,
                                n_t=n_t))).validate()


def from_btensor_waveform(G, dt, *, echo_idx=None, TE=None):
    """A precomputed b-tensor gradient waveform -- the PHYSICAL gradient, as played -- as a spin echo.

    For an externally designed b-tensor encoding (e.g. a dmipy-design ``design_waveform`` output) rather than
    the canonical square pairs of :func:`ste` / :func:`pte`. The b-tensor SHAPE (spherical / planar / linear,
    i.e. b_delta) is whatever the numbers produce: it is computed from ``G_eff``, not declared, so there is no
    shape argument; ``seq.btensor()`` reads the realised shape. The 180 is declared at ``echo_idx`` (default
    TE/2 -- the only position at which the static field refocuses at the echo, so a non-TE/2 ``echo_idx``
    raises) and folds into ``G_eff`` through the schedule's sign, as for every family.
    """
    G = np.asarray(G, dtype=np.float32)
    if G.ndim == 2:
        G = G[None]
    n_m, n_t, _ = G.shape
    dt = float(dt)
    T_total = (n_t - 1) * dt
    gradient_directions = np.tile([0., 0., 1.], (n_m, 1))
    TE_, te_auto = _resolve_te(TE, T_total, n_m)
    te2 = int(round((float(TE_[0]) / 2.0) / dt))      # the spin-echo 180 position
    if echo_idx is None:
        echo_idx = te2
    echo_idx = int(np.clip(echo_idx, 0, n_t - 1))
    if abs(echo_idx - te2) > 1:                       # beyond rounding
        raise ValueError(
            f"from_btensor_waveform: echo_idx={echo_idx} places the 180 at {echo_idx * dt * 1e3:.2f} ms, not "
            f"TE/2={te2 * dt * 1e3:.2f} ms. A spin echo refocuses the static field at 2*t_180="
            f"{2 * echo_idx * dt * 1e3:.2f} ms, not at the echo TE={float(TE_[0]) * 1e3:.2f} ms.")
    schedule = RFSchedule([RFEvent(0.0, 90, 'Mz→Mxy'), RFEvent(echo_idx * dt, 180, 'refocus')])
    sign = schedule.sign(np.arange(n_t) * dt)[None, :, None]
    bvalues_num = _calc_b_from_waveform(G * sign, dt)
    return ScannerSequence(
        G=G, dt=dt, rf=schedule, family='btensor',
        encoding=Encoding(bvalues=bvalues_num, gradient_directions=gradient_directions, TE=TE_, minimum_te=T_total,
                          te_auto=te_auto),
        build_spec=('from_btensor_waveform', dict(G=G, dt=dt, echo_idx=echo_idx, TE=TE))).validate()


def from_pgste_waveform(G, dt, *, store_idx, recall_idx, gradient_directions=None, ste_flip_angles=(90., 90., 90.),
                        TE=None):
    """A precomputed PGSTE (stimulated-echo) gradient waveform -- the PHYSICAL gradient, as played -- as a
    stimulated echo.

    The stimulated-echo analogue of :func:`from_btensor_waveform`, for an externally designed PGSTE encoding
    (e.g. a dmipy-design ``design_stimulated_echo`` output) rather than the canonical lobes of :func:`pgste`.
    Three pulses (excite / store / recall) and NO 180, so the static field is refocused only when the two
    transverse encoding periods match (``tau1 = tau3``, the PGSTE analogue of a 180 at TE/2); unequal periods
    leave a residual static dephasing at the echo and are refused. The two same-sign lobes straddle a
    gradient-off mixing time ``TM`` on z.

    ``store_idx`` / ``recall_idx`` are the samples of the store and recall pulses; ``delta = store_idx * dt`` and
    ``TM = (recall_idx - store_idx) * dt`` follow.
    """
    G = np.asarray(G, dtype=np.float32)
    if G.ndim == 2:
        G = G[None]
    n_m, n_t, _ = G.shape
    dt = float(dt)
    store_idx = int(store_idx); recall_idx = int(recall_idx)
    if not (0 < store_idx < recall_idx < n_t):
        raise ValueError(f"from_pgste_waveform: need 0 < store_idx ({store_idx}) < recall_idx ({recall_idx}) < n_t ({n_t}).")
    T_total = (n_t - 1) * dt
    enc = np.any(np.abs(G[0]) > 1e-9, axis=1)             # (n_t,) encoded samples
    tau1 = int(np.sum(enc[:store_idx])) * dt
    tau3 = int(np.sum(enc[recall_idx:])) * dt
    if abs(tau1 - tau3) > 2.0 * dt:
        raise ValueError(
            f"from_pgste_waveform: tau1 = {tau1*1e3:.2f} ms != tau3 = {tau3*1e3:.2f} ms. A stimulated echo refocuses the "
            f"static field only when the two transverse encoding periods match; unequal periods leave a residual "
            f"static dephasing at the echo.")
    a1, a2, a3 = (float(a) for a in ste_flip_angles)
    schedule = RFSchedule([RFEvent(0.0, a1, 'Mz→Mxy'), RFEvent(store_idx * dt, a2, 'store'),
                           RFEvent(recall_idx * dt, a3, 'recall')])
    sign = schedule.sign(np.arange(n_t) * dt)[None, :, None]
    delta = float(store_idx * dt)
    TM = float((recall_idx - store_idx) * dt)
    bvalues_num = _calc_b_from_waveform(G * sign, dt)
    gradient_directions = (np.tile([0., 0., 1.], (n_m, 1)) if gradient_directions is None
                           else np.asarray(gradient_directions, dtype=np.float64))
    TE_, te_auto = _resolve_te(TE, T_total, n_m)
    delta_ = np.full(n_m, delta)
    return ScannerSequence(
        G=G, dt=dt, rf=schedule, family='pgste',
        encoding=Encoding(bvalues=bvalues_num, gradient_directions=gradient_directions, TE=TE_, delta=delta_,
                          Delta=delta_ + TM, minimum_te=T_total, te_auto=te_auto, ste_flip_angles=(a1, a2, a3)),
        build_spec=('from_pgste_waveform', dict(G=G, dt=dt, store_idx=store_idx, recall_idx=recall_idx,
                                                gradient_directions=gradient_directions,
                                                ste_flip_angles=ste_flip_angles, TE=TE))).validate()


def instantaneous(seq):
    """The idealised instantaneous (infinite-slew / square) limit of a built sequence: the same builder with
    ``slew_rate=np.inf`` for the families that slew-limit (pgse, pgste, ogse); the others rebuild as they are.
    This is the idealised view an analytical model reads."""
    if seq.build_spec is None:
        return seq
    name, kwargs = seq.build_spec
    kwargs = dict(kwargs)
    if 'slew_rate' in kwargs:
        kwargs['slew_rate'] = np.inf
    return globals()[name](**kwargs)


def to_gradient_array(seq, n_t=1000):
    """``(G_eff, dt)`` -- the EFFECTIVE gradient -- of the square PGSE with this sequence's b-values, directions,
    delta and Delta on an ``n_t`` grid: :func:`pgse` at infinite slew, as the analytical layer integrates it."""
    e = seq.encoding
    if e is None or e.delta is None or e.Delta is None or e.gradient_strengths is None:
        raise ValueError("to_gradient_array() requires an encoding with delta, Delta, and gradient_strengths.")
    delta_tol = np.float32(1e-6)
    if (np.max(e.delta) - np.min(e.delta)) > delta_tol:
        raise ValueError("to_gradient_array() requires uniform delta.")
    if (np.max(e.Delta) - np.min(e.Delta)) > delta_tol:
        raise ValueError("to_gradient_array() requires uniform Delta.")
    sq = pgse(e.bvalues, e.gradient_directions, float(e.delta[0]), float(e.Delta[0]), n_t=n_t, slew_rate=np.inf)
    return sq.G_eff, sq.dt
