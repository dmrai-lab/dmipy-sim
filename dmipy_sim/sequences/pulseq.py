"""Pulseq (.seq) interoperability for dmipy-sim.

Pulseq (Layton et al., MRM 2017) is the de-facto open, vendor-neutral pulse-
sequence format.  This module bridges it to dmipy-sim's base representation
(``Waveform``: G(t) in T/m + dt + RF event schedule), so that:

  * ``from_pulseq`` rasterises ANY ``.seq`` onto our uniform grid and returns a
    Monte-Carlo-simulable ``Waveform`` -- i.e. dmipy-sim can simulate the field's
    sequences directly, no manual parameter transfer;
  * ``to_pulseq`` exports a ``Waveform`` back to a ``.seq`` (the round-trip is the
    consistency/safety check on the bridge);
  * ``PULSEQ_SYSTEMS`` is a small curated catalogue of scanner hardware limits in
    Pulseq's own ``Opts`` schema (max_grad/max_slew/raster/dead-times), so our
    slew-limited constructors and the exported files speak the same language.

Units: pypulseq works in Hz/m with gamma in Hz/T; we work in T/m with
``dmipy_sim.constants.GAMMA`` in rad/s/T.  The boundary conversion uses
``gamma_Hz = GAMMA / (2*pi)`` consistently in both directions, so round-trips do
not pick up a gyromagnetic mismatch.

Requires ``pypulseq`` (the reference implementation; installed --no-deps so it
cannot perturb the numpy/jax/GPU stack).  ``from_pulseq``/``to_pulseq`` raise a
clear ImportError if it is absent.

Scope (v1): the diffusion-relevant subset -- gradients (the physics), the
excitation/refocusing RF schedule, and the ADC/echo time.  Unsupported Pulseq
features (frequency/phase offsets, rotations, trigger/extension events) are not
interpreted; ``from_pulseq`` warns rather than silently dropping them.
"""
from __future__ import annotations

import json

import warnings

import numpy as np

from ..constants import GAMMA

GAMMA_HZ = GAMMA / (2.0 * np.pi)   # Hz/T (proton); pypulseq's gamma convention


def _require_pypulseq():
    try:
        import pypulseq as pp
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pypulseq is required for Pulseq interop. Install it (isolated from the "
            "numpy/jax stack) with:  pip install --no-deps pypulseq"
        ) from e
    return pp


# -- scanner catalogue (Pulseq Opts schema) ----------------------------------
# Representative hardware limits.  Gmax in mT/m, slew in T/m/s (= mT/m/ms).
# These are first-order {Gmax, slew} models -- the binding in-vivo limit is often
# peripheral-nerve-stimulation (IEC 60601-2-33), which is vendor-specific (SAFE
# model) and NOT captured here.  Values are widely-published nominal maxima.
PULSEQ_SYSTEMS = {
    'siemens_prisma':     dict(max_grad=80.,   max_slew=200.,   grad_unit='mT/m', slew_unit='T/m/s'),
    'siemens_connectom':  dict(max_grad=300.,  max_slew=200.,   grad_unit='mT/m', slew_unit='T/m/s'),
    'ge_premier':         dict(max_grad=70.,   max_slew=200.,   grad_unit='mT/m', slew_unit='T/m/s'),
    'philips_ingenia':    dict(max_grad=80.,   max_slew=200.,   grad_unit='mT/m', slew_unit='T/m/s'),
    'clinical_typical':   dict(max_grad=45.,   max_slew=150.,   grad_unit='mT/m', slew_unit='T/m/s'),
    'preclinical_bruker': dict(max_grad=1000., max_slew=10000., grad_unit='mT/m', slew_unit='T/m/s'),
}


def make_system(scanner=None, *, grad_raster_time=None, **overrides):
    """Build a pypulseq ``Opts`` from a named scanner (or overrides).

    ``scanner`` keys :data:`PULSEQ_SYSTEMS`; ``overrides`` set/replace any Opts
    field (e.g. ``max_slew=300``).  ``gamma`` defaults to dmipy-sim's value so
    Hz/m <-> T/m conversions are self-consistent.
    """
    pp = _require_pypulseq()
    kw = dict(PULSEQ_SYSTEMS.get(scanner, {})) if scanner else {}
    kw.setdefault('gamma', GAMMA_HZ)
    if grad_raster_time is not None:
        kw['grad_raster_time'] = float(grad_raster_time)
    kw.update(overrides)
    return pp.Opts(**kw)


def _permissive_system(dt):
    """A limit-free Opts on the waveform's own raster -- for exact round-trips
    (no resampling, no slew/Gmax clipping of an already-built waveform)."""
    pp = _require_pypulseq()
    return pp.Opts(max_grad=1e9, grad_unit='Hz/m', max_slew=1e12, slew_unit='Hz/m/s',
                   grad_raster_time=float(dt), rf_raster_time=float(dt),
                   block_duration_raster=float(dt),
                   rf_dead_time=0.0, rf_ringdown_time=0.0, adc_dead_time=0.0,
                   gamma=GAMMA_HZ)


def _encode_rf_events(rf_events):
    if not rf_events:
        return ''
    slim = [{'t_s': float(e['t_s']), 'flip_deg': float(e.get('flip_deg', 0.0)),
             'label': str(e.get('label', ''))} for e in rf_events]
    return json.dumps(slim, separators=(',', ':'))


def _decode_rf_events(s):
    if not s:
        return None
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return None


def _event_times(arr):
    """Extract event times (s) from a pypulseq waveforms_and_times field.

    Excitation/refocusing come as (3, n) [row 0 = time, rows 1-2 = freq/phase
    offset]; ADC comes as (n, 2) [col 0 = time]; tolerate 1-D too.
    """
    if arr is None:
        return np.array([])
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return np.array([])
    if a.ndim == 2:
        return a[0] if a.shape[0] == 3 else a[:, 0]
    return a.ravel()


# -- export: Waveform -> .seq -------------------------------------------------

def _ste_from_rf_schedule(rf_events):
    """``(TM, stimulated_echo)`` derived from the RF schedule itself.

    A stimulated echo is defined by its pulses, not by a label: magnetisation is tipped down, STORED along
    z by a second 90 degrees, and RECALLED by a third, so the mixing time is the gap between the storage and
    recall pulses. That is already in the schedule the bridge carries, so deriving it beats writing the
    value into a private definition -- one less thing that can disagree with the waveform it describes, and
    it works for any sequence whose RF is described, not only for files we wrote.

    Falls back to the flip-angle pattern when the pulses are unlabelled: three 90-degree pulses, TM between
    the second and third. A 90/180 spin echo yields ``(None, False)``.
    """
    if not rf_events:
        return None, False
    ev = sorted(rf_events, key=lambda e: float(e.get('t_s', 0.0)))
    by_label = {str(e.get('label', '')).lower(): e for e in ev}
    if 'store' in by_label and 'recall' in by_label:
        return float(by_label['recall']['t_s']) - float(by_label['store']['t_s']), True
    nineties = [e for e in ev if abs(float(e.get('flip_deg', 0.0)) - 90.0) < 1e-3]
    if len(nineties) >= 3:
        return float(nineties[2]['t_s']) - float(nineties[1]['t_s']), True
    return None, False


def to_pulseq(waveform, m=0, *, system=None, filename=None,
              excitation_flip_deg=90.0, native_rf=True):
    """Export measurement ``m`` of a :class:`dmipy_sim.waveforms.Waveform` to a
    pypulseq ``Sequence`` (written to ``filename`` if given).

    The full gradient G(t) is emitted as one arbitrary-gradient block (exact, on
    the waveform's own raster), bracketed by an excitation RF and an ADC; the RF
    schedule, dt and echo index travel in the ``[DEFINITIONS]`` so ``from_pulseq``
    reconstructs the Waveform faithfully (the round-trip safety net).  v1 carries
    the refocusing RF as metadata rather than splitting the gradient into native
    180-blocks -- enough for round-trip + simulation, not yet a scanner-runnable
    spin echo (that is the v2 native-RF-splitting follow-up).
    """
    pp = _require_pypulseq()
    dt = float(waveform.dt)
    # The PHYSICAL gradient is what a scanner plays. waveform.G is the EFFECTIVE gradient, whose sign is
    # already folded through the refocusing pulses -- exporting that would describe a sequence no scanner
    # can run, and would double-count the inversion for any reader that applies the RF itself.
    if getattr(waveform, 'G_display', None) is not None:
        G = np.asarray(waveform.G_display)[m].astype(float)
    else:
        # No physical copy stored: waveform.G is the effective gradient, so recover the physical one by
        # un-folding the same sign schedule the importer will re-apply. s = +-1, so multiplying inverts.
        Geff = np.asarray(waveform.G)[m].astype(float)
        tg = np.arange(Geff.shape[0]) * dt
        G = Geff * _effective_sign(waveform.rf_events, tg)[:, None]
    sys = system or _permissive_system(dt)
    gamma_hz = float(getattr(sys, 'gamma', GAMMA_HZ))
    seq = pp.Sequence(system=sys)

    # A finite pulse needs a slot with no gradient on it. Where the constructor leaves one -- pgse
    # slew-limited, pgste -- the pulse costs nothing and the sequence keeps its duration and sample count.
    # Where it does not, the pulse must INTERRUPT the gradient, which lengthens the sequence by its
    # duration. That is not an artefact of the export: a scanner has to make the same room, and a design
    # with no dead time genuinely cannot play its RF for free. Measured on the constructors here: ogse and
    # cpmg place every pulse on live gradient, pgse-square its excitation.
    Ghz = G * gamma_hz                                # Hz/m
    ev = sorted(waveform.rf_events or [], key=lambda e: float(e['t_s']))
    ks = [int(np.clip(round(float(e['t_s']) / dt), 0, max(len(Ghz) - 1, 0))) for e in ev]
    inserted = [k for k in ks if k < len(Ghz) and np.any(np.abs(Ghz[k]) > 0)]
    if inserted and native_rf:
        warnings.warn(
            f"to_pulseq: {len(inserted)} RF pulse(s) fall on live gradient, so each interrupts it and adds "
            f"{dt*1e6:.1f} us; the exported sequence is {len(inserted)*dt*1e3:.3f} ms longer than the "
            f"waveform and its b-value/TE shift accordingly. This is what the sequence costs on a scanner. "
            f"Use native_rf=False for an exact round trip that keeps the RF as metadata instead.",
            RuntimeWarning, stacklevel=2)

    def _grad_block(a, b):
        """Gradient samples [a, b) as one arbitrary block per active channel."""
        if b <= a:
            return
        # Pulseq needs an arbitrary gradient to start and end at zero. Splitting at RF makes that true by
        # construction wherever the pulse sits in a gap, so pad only when a segment really is cut on a live
        # edge -- padding unconditionally would add two samples PER SEGMENT and silently stretch the
        # sequence, which is the same lengthening the insertion warning is about, but hidden.
        blocks = []
        for ci, ch in enumerate(('x', 'y', 'z')):
            col = Ghz[a:b, ci]
            if np.any(col):
                pre = [] if col[0] == 0.0 else [0.0]
                post = [] if col[-1] == 0.0 else [0.0]
                wf = np.concatenate([pre, col, post]) if (pre or post) else col
                blocks.append(pp.make_arbitrary_grad(channel=ch, waveform=wf, system=sys))
        if blocks:
            seq.add_block(*blocks)
        else:
            seq.add_block(pp.make_delay((b - a) * dt))

    if not native_rf:
        # v1 semantics unchanged: one excitation block, the schedule as metadata, and the EFFECTIVE
        # gradient -- a reader that applies no RF still integrates the right thing.
        Ghz = np.asarray(waveform.G)[m].astype(float) * gamma_hz
        seq.add_block(pp.make_block_pulse(flip_angle=np.deg2rad(excitation_flip_deg),
                                          duration=dt, system=sys))
        _grad_block(0, len(Ghz))
        seq.add_block(pp.make_adc(num_samples=1, duration=dt, system=sys))
        _write_defs(seq, waveform, dt, len(Ghz))
        if filename:
            seq.write(filename)
        return seq

    prev = 0
    for e, k in zip(ev, ks):
        k = int(np.clip(k, 0, len(Ghz) - 1))
        _grad_block(prev, k)
        flip = float(e.get('flip_deg', 90.0))
        # 'use' is what lets a reader classify the pulse without our labels: pypulseq reports excitation
        # and refocusing events separately, which is exactly the distinction the effective gradient needs.
        use = 'refocusing' if abs(flip - 180.0) < 1.0 else 'excitation'
        seq.add_block(pp.make_block_pulse(flip_angle=np.deg2rad(flip), duration=dt,
                                          system=sys, use=use))
        prev = k + 1
    _grad_block(prev, len(Ghz))
    seq.add_block(pp.make_adc(num_samples=1, duration=dt, system=sys))

    _write_defs(seq, waveform, dt, len(Ghz))

    if filename:
        seq.write(filename)
    return seq


def _write_defs(seq, waveform, dt, n_t):
    """dt / n_t / echo index / RF schedule in [DEFINITIONS].

    Conveniences for our own round trip, not the physics: TM, the storage window and the effective-gradient
    sign are all derived from the RF blocks on import, so a reader that ignores these still gets the
    sequence right.
    """
    seq.set_definition('dmipy_dt', dt)
    seq.set_definition('dmipy_echo_idx', int(waveform.echo_idx))
    seq.set_definition('dmipy_n_t', int(n_t))
    seq.set_definition('dmipy_rf_events', _encode_rf_events(waveform.rf_events))



def _rf_from_pulseq(seq):
    """RF schedule from Pulseq's own event classification: ``[{t_s, flip_deg, label}, ...]``.

    ``Sequence.waveforms_and_times()`` already separates excitation from refocusing events and reports
    their times, so the classification that matters for the effective gradient comes from the library
    rather than from re-deriving flip angles out of the pulse shapes here. Labels follow the pattern: the
    first excitation tips down, a later pair stores and recalls, and refocusing events invert -- which is
    what lets a file written by any tool be read.
    """
    wav = seq.waveforms_and_times()
    t_exc = _event_times(wav[1]) if len(wav) > 1 else np.array([])
    t_ref = _event_times(wav[2]) if len(wav) > 2 else np.array([])
    ev = ([{'t_s': float(t), 'flip_deg': 90.0, 'label': ''} for t in np.atleast_1d(t_exc)] +
          [{'t_s': float(t), 'flip_deg': 180.0, 'label': 'refocus'} for t in np.atleast_1d(t_ref)])
    ev.sort(key=lambda e: e['t_s'])
    n90 = 0
    for e in ev:
        if e['label'] != 'refocus':
            e['label'] = ('Mz\u2192Mxy', 'store', 'recall')[min(n90, 2)]
            n90 += 1
    return ev or None


def _effective_sign(rf_events, t_grid):
    """Sign of the EFFECTIVE gradient over time, from the pulses alone.

    A refocusing pulse inverts the accumulated phase, so the effective gradient changes sign after it. A
    stimulated echo does the same across its storage/recall pair: phase is parked along z at the storage
    pulse and the recalled pathway rephases like a spin echo, so the sign flips at RECALL. Verified against
    the constructors: pgse flips after its 180 (first non-zero sample 20.05 ms, pulse at 12.47 ms) and
    pgste after its recall (25.04 ms, pulse at 24.96 ms).
    """
    s = np.ones_like(t_grid, dtype=np.float32)
    if not rf_events:
        return s
    for e in rf_events:
        lab = str(e.get('label', ''))
        flips = abs(float(e.get('flip_deg', 0.0)) - 180.0) < 20.0 or lab == 'refocus' or lab == 'recall'
        if flips:
            s[t_grid >= float(e['t_s'])] *= -1.0
    return s


def _longitudinal_mask(rf_events, t_grid):
    """``chi_perp``: 0 where magnetisation is stored along z, 1 where it is transverse.

    Between a storage pulse and its recall the spins carry no phase, which is precisely the T1-weighted
    period a stimulated echo exists to create. Returns ``None`` when there is no storage interval, so a
    spin echo keeps the default.
    """
    if not rf_events:
        return None
    st = next((e for e in rf_events if e.get('label') == 'store'), None)
    rc = next((e for e in rf_events if e.get('label') == 'recall'), None)
    if st is None or rc is None:
        return None
    chi = np.ones_like(t_grid, dtype=np.float32)
    chi[(t_grid >= float(st['t_s'])) & (t_grid < float(rc['t_s']))] = 0.0
    return chi


# -- import: .seq -> Waveform -------------------------------------------------
def from_pulseq(src, *, dt=None):
    """Read a Pulseq ``.seq`` (path or ``pypulseq.Sequence``) and rasterise it to
    a Monte-Carlo-simulable :class:`dmipy_sim.waveforms.Waveform` (single
    measurement, shape (1, n_t, 3) in T/m).

    Gradients are rasterised exactly (piecewise-linear interpolation onto the
    uniform grid).  The RF schedule and echo index come from dmipy ``[DEFINITIONS]``
    when present (our own files), otherwise from Pulseq's native excitation/
    refocusing event times (external files); a 90/180 flip is assumed for
    excitation/refocusing when only times are available.
    """
    pp = _require_pypulseq()
    from ..waveforms import Waveform
    import jax.numpy as jnp

    if isinstance(src, pp.Sequence):
        seq = src
    else:
        seq = pp.Sequence()
        seq.read(str(src))

    defs = getattr(seq, 'definitions', {}) or {}
    gamma_hz = float(getattr(getattr(seq, 'system', None), 'gamma', GAMMA_HZ) or GAMMA_HZ)

    wav = seq.waveforms_and_times()
    gw = wav[0]                       # list of 3 channels, each (2, N): [t_s; amp Hz/m]
    # out[1]/out[2] are (3, n_event): row 0 = times, rows 1-2 = freq/phase offsets.
    t_exc = _event_times(wav[1]) if len(wav) > 1 else np.array([])
    t_ref = _event_times(wav[2]) if len(wav) > 2 else np.array([])
    t_adc = wav[-1] if len(wav) >= 4 else None

    dt = float(dt if dt is not None else defs.get('dmipy_dt', seq.grad_raster_time))

    # Anchor t=0 of the Waveform at the excitation (our convention: rf/echo times
    # are relative to excitation).  Fall back to the first gradient sample, else 0.
    t0 = float(t_exc[0]) if t_exc.size else np.inf
    if not np.isfinite(t0):
        for ci in range(min(3, len(gw))):
            arr = np.asarray(gw[ci], dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 1:
                t0 = min(t0, float(arr[0, 0]))
        t0 = t0 if np.isfinite(t0) else 0.0

    T = float(seq.duration()[0])
    n_t = (int(defs['dmipy_n_t']) if 'dmipy_n_t' in defs
           else max(2, int(round((T - t0) / dt)) + 1))
    t_grid = np.arange(n_t) * dt

    G = np.zeros((n_t, 3), dtype=np.float32)
    raster = float(getattr(seq, 'grad_raster_time', dt) or dt)
    for ci in range(min(3, len(gw))):
        arr = np.asarray(gw[ci], dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            tt, aa = arr[0], arr[1]
            # waveforms_and_times() lists samples only WITHIN gradient events; between events the gradient
            # is zero by definition. Interpolating the bare list therefore draws a straight line across
            # every gap -- which silently fills a diffusion gap or a stimulated echo's whole storage period
            # with gradient that is not there. Make the zeros explicit at each gap edge before sampling.
            gaps = np.where(np.diff(tt) > 1.5 * raster)[0]
            if gaps.size:
                eps = 0.5 * raster
                tt = np.concatenate([tt, tt[gaps] + eps, tt[gaps + 1] - eps])
                aa = np.concatenate([aa, np.zeros(gaps.size), np.zeros(gaps.size)])
                order = np.argsort(tt)
                tt, aa = tt[order], aa[order]
            G[:, ci] = np.interp(t_grid + t0, tt, aa, left=0.0, right=0.0) / gamma_hz

    # RF schedule, read from the blocks themselves so a file that describes its RF is understood whoever
    # wrote it. The stored dmipy_rf_events is a fallback for older files that carry only the excitation.
    blk_events = _rf_from_pulseq(seq)
    meta_events = _decode_rf_events(defs.get('dmipy_rf_events'))
    if blk_events:
        for e in blk_events:
            e['t_s'] = float(e['t_s']) - t0
    # Which convention is this file written in? A sequence whose blocks carry the WHOLE schedule states its
    # RF natively, so its gradient is physical and the pulses must be folded in. One that describes more
    # pulses in metadata than it plays as blocks is the older form, whose gradient is already effective --
    # folding again would invert it twice. The file says which it is; no flag is needed.
    native = bool(blk_events) and not (meta_events and len(meta_events) > len(blk_events))
    rf_events = blk_events if native else (meta_events or blk_events)
    if rf_events is None:
        rf_events = ([{'t_s': float(t) - t0, 'flip_deg': 90.0, 'label': 'excitation'}
                      for t in t_exc] +
                     [{'t_s': float(t) - t0, 'flip_deg': 180.0, 'label': 'refocusing'}
                      for t in t_ref]) or None

    # A .seq carries the PHYSICAL gradient; the simulator integrates the EFFECTIVE one. Fold the pulses in
    # rather than trusting a stored copy -- this is what makes the round trip physics rather than metadata.
    if native:
        G = G * _effective_sign(rf_events, t_grid)[:, None]
    chi_perp = _longitudinal_mask(rf_events, t_grid)

    if 'dmipy_echo_idx' in defs:
        echo_idx = int(defs['dmipy_echo_idx'])
    else:
        ta = _event_times(t_adc)
        echo_idx = (int(round((float(ta[-1]) - t0) / dt)) if ta.size else n_t - 1)
    echo_idx = int(np.clip(echo_idx, 0, n_t - 1))

    # TM / stimulated-echo state come from the RF schedule rather than a stored value: the pulses ARE the
    # definition, so a file that describes its RF describes its mixing time, and there is no second copy to
    # fall out of sync with the first.
    TM, stimulated_echo = _ste_from_rf_schedule(rf_events)

    return Waveform(G=jnp.asarray(G[None]), dt=dt, echo_idx=echo_idx,
                    rf_events=rf_events, TM=TM, stimulated_echo=stimulated_echo,
                    chi_perp=None if chi_perp is None else jnp.asarray(chi_perp))


def pulseq_timing(src):
    """Extract a diffusion spin-echo timing budget from a Pulseq ``.seq``.

    Reads the real event schedule so the diffusion-encoding windows (and any
    pre-/post-180 asymmetry) can be *derived* from the sequence rather than
    guessed.  Assumes a spin echo: the first RF block is the 90 excitation, the
    second is the 180 refocusing, and there is one ADC (readout) block.

    Parameters
    ----------
    src : str | Path | pypulseq.Sequence

    Returns
    -------
    dict with keys (all seconds):
        ``t_excite``           90 RF duration (encoding starts after it),
        ``t_refocus``          180 RF duration (gradient off across it),
        ``TE``                 echo time = 2·(t_180_centre − t_90_centre),
        ``t_readout_pre_echo`` readout-start → echo (post-180 encoding must end
                               by ``TE − t_readout_pre_echo``),
        ``readout_duration``   ADC window length.

    These feed ``dmipy_design.optimizers.SequenceTiming`` (via
    ``SequenceTiming.from_pulseq``), which turns them into the encoding-window
    masks for the waveform optimizer.
    """
    pp = _require_pypulseq()
    if isinstance(src, pp.Sequence):
        seq = src
    else:
        seq = pp.Sequence()
        seq.read(str(src))

    def _rf_duration(rf):
        d = float(getattr(rf, 'shape_dur', 0.0) or 0.0)
        if d <= 0.0 and getattr(rf, 't', None) is not None and len(rf.t):
            d = float(rf.t[-1])
        return d

    rf_blocks = []          # (centre_time_s, duration_s)
    adc_block = None        # (start_time_s, duration_s)
    t = 0.0
    n_blocks = len(seq.block_events)
    for i in range(1, n_blocks + 1):
        blk = seq.get_block(i)
        dur = float(seq.block_durations[i])
        rf = getattr(blk, 'rf', None)
        if rf is not None:
            rdur = _rf_duration(rf)
            rf_blocks.append((t + float(getattr(rf, 'delay', 0.0)) + rdur / 2.0, rdur))
        adc = getattr(blk, 'adc', None)
        if adc is not None and adc_block is None:
            a_dur = float(adc.num_samples) * float(adc.dwell)
            adc_block = (t + float(getattr(adc, 'delay', 0.0)), a_dur)
        t += dur

    if len(rf_blocks) < 2:
        raise ValueError(
            f"pulseq_timing expects >=2 RF blocks (90 excitation + 180 refocus); "
            f"found {len(rf_blocks)}.")
    if adc_block is None:
        raise ValueError("pulseq_timing found no ADC (readout) block.")

    (t90c, t90d), (t180c, t180d) = rf_blocks[0], rf_blocks[1]
    TE = 2.0 * (t180c - t90c)
    echo_time = t90c + TE                       # == 2·t180c − t90c
    adc_start, readout_duration = adc_block
    return {
        't_excite': t90d,
        't_refocus': t180d,
        'TE': TE,
        't_readout_pre_echo': echo_time - adc_start,
        'readout_duration': readout_duration,
    }
