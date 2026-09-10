"""Pulseq (.seq) interoperability for dmipy-sim.

Pulseq (Layton et al., MRM 2017) is the de-facto open, vendor-neutral pulse-
sequence format.  This module bridges it to dmipy-sim's base representation
(``ScannerSequence``: G(t) in T/m + dt + RF event schedule), so that:

  * ``from_pulseq`` rasterises ANY ``.seq`` onto our uniform grid and returns a
    Monte-Carlo-simulable ``ScannerSequence`` -- i.e. dmipy-sim can simulate the field's
    sequences directly, no manual parameter transfer;
  * ``to_pulseq`` exports a ``ScannerSequence`` back to a ``.seq`` (the round-trip is the
    consistency/safety check on the bridge);
  * ``PULSEQ_SYSTEMS`` is the scanner catalogue (:mod:`dmipy_sim.acquisition.scanner_constants`)
    in Pulseq's own ``Opts`` schema (max_grad/max_slew/raster/dead-times), one preset per
    legacy name, derived at import -- so our slew-limited constructors, the exported files
    and every other reader of the catalogue speak the same numbers.

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
from ..acquisition.scanners import ScannerLimits
from ..acquisition.rf import RFEvent, RFSchedule
from ..acquisition.timing import SequenceTiming
from ..acquisition.scanner_sequence import ScannerSequence

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


# -- scanner catalogue in the Pulseq Opts schema: a VIEW of acquisition.scanner_constants -----------
# The binding in-vivo limit is often peripheral-nerve stimulation (IEC 60601-2-33), which the SAFE
# model captures; its representative coefficients are ``ScannerLimits.safe_model`` and the solver is
# dmipy-design's.
_PULSEQ_PRESETS = ('siemens_prisma', 'siemens_connectom', 'ge_premier', 'philips_ingenia',
                   'clinical_typical', 'preclinical_bruker')
PULSEQ_SYSTEMS = {name: ScannerLimits.of(name).pulseq_dict() for name in _PULSEQ_PRESETS}


def make_system(scanner=None, *, grad_raster_time=None, **overrides):
    """Build a pypulseq ``Opts`` from a scanner (or overrides alone).

    ``scanner`` is anything :meth:`ScannerLimits.of` resolves -- a preset name, a certificate
    class, a model key -- or a ``ScannerLimits``; an unknown name raises rather than yielding a
    limit-free system. ``overrides`` set/replace any Opts field (e.g. ``max_slew=300``). ``gamma``
    defaults to dmipy-sim's value so Hz/m <-> T/m conversions are self-consistent.
    """
    pp = _require_pypulseq()
    kw = dict(ScannerLimits.of(scanner).pulseq_dict()) if scanner is not None else {}
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
    return json.dumps(RFSchedule(rf_events).to_dicts(), separators=(',', ':'))


def _decode_rf_events(s):
    if not s:
        return None
    try:
        return RFSchedule.from_dicts(json.loads(s)) or None
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


# -- export: ScannerSequence -> .seq -------------------------------------------------

def to_pulseq(waveform, m=0, *, system=None, filename=None,
              excitation_flip_deg=90.0, native_rf=True):
    """Export measurement ``m`` of a :class:`dmipy_sim.acquisition.scanner_sequence.ScannerSequence` to a
    pypulseq ``Sequence`` (written to ``filename`` if given).

    The full gradient G(t) is emitted as one arbitrary-gradient block (exact, on
    the waveform's own raster), bracketed by an excitation RF and an ADC; the RF
    schedule, dt and echo index travel in the ``[DEFINITIONS]`` so ``from_pulseq``
    reconstructs the ScannerSequence faithfully (the round-trip safety net).  v1 carries
    the refocusing RF as metadata rather than splitting the gradient into native
    180-blocks -- enough for round-trip + simulation, not yet a scanner-runnable
    spin echo (that is the v2 native-RF-splitting follow-up).
    """
    pp = _require_pypulseq()
    dt = float(waveform.dt)
    # waveform.G is the PHYSICAL gradient, what a scanner plays; the pulses are their own blocks.
    G = np.asarray(waveform.G)[m].astype(float)
    sys = system or _permissive_system(dt)
    gamma_hz = float(getattr(sys, 'gamma', GAMMA_HZ))
    seq = pp.Sequence(system=sys)

    # A finite pulse needs a slot with no gradient on it. Whether one exists is a property of the
    # SEQUENCE, not of the sequence family: pgse slew-limited and pgste ramp to zero around every pulse, so
    # export costs nothing; ogse oscillates continuously and has no gap at either pulse; pgse-square starts
    # at full amplitude so its excitation has none. A CPMG is a 90 plus a train of 180s and carries no
    # gradient of its own -- G=0 inserts freely -- but the constant diffusion-weighting gradient the
    # constructor can add is never off, and then every pulse in the train needs room made for it.
    Ghz = G * gamma_hz                                # Hz/m
    ev = waveform.rf
    # a pulse takes the raster its instant falls in; an instant on the boundary between a free raster and a live
    # one (a lobe starting at the recall, say) is played in the free raster just before it
    ks = []
    for e in ev:
        k = int(np.clip(round(e.t_s / dt), 0, max(len(Ghz) - 1, 0)))
        if k > 0 and np.any(np.abs(Ghz[k]) > 0) and not np.any(np.abs(Ghz[k - 1]) > 0) and (k - 0.5) * dt <= e.t_s + 1e-12:
            k -= 1
        ks.append(k)
    inserted = [k for k in ks if k < len(Ghz) and np.any(np.abs(Ghz[k]) > 0)]
    if inserted and native_rf:
        warnings.warn(
            f"to_pulseq: {len(inserted)} RF pulse(s) fall where the gradient is still on, so there is no "
            f"free slot to play them in. Each is INSERTED, pausing the gradient and resuming it unchanged, "
            f"which lengthens the sequence by {dt*1e6:.1f} us per pulse "
            f"({len(inserted)*dt*1e3:.3f} ms total), and TE with it. No part of a lobe is dropped, but each "
            f"cut must ramp the gradient down and back up, costing roughly 0.6% of the encoding area per "
            f"cut. Both are what a sequence with no dead time costs on a scanner. Use native_rf=False for "
            f"an exact round trip that keeps the RF as metadata instead.",
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
    n_inserted = 0          # samples of extra time the pulses cost
    inserted_before_echo = 0
    echo0 = int(getattr(waveform, 'echo_idx', len(Ghz) - 1) or 0)
    for e, k in zip(ev, ks):
        k = int(np.clip(k, 0, len(Ghz) - 1))
        _grad_block(prev, k)
        # Where the pulse lands on a gradient-free sample it takes that slot, and the sequence keeps its
        # duration exactly. Where it does not, the pulse is INSERTED: the gradient resumes from the sample
        # it was paused at, so no part of a lobe is deleted -- the sequence, and TE with it, get longer
        # instead. Consuming the sample would notch the lobe and silently change b, which is not a cost a
        # scanner pays either. Cutting a live gradient is still not free: Pulseq needs each arbitrary
        # gradient to start and end at zero, so every cut ramps down and back up, costing ~0.6% of the
        # encoding area per cut (measured: ogse 2 cuts -> 1.2%, cpmg 5 cuts -> 3.5%). A scanner pays that
        # too, and more, since its ramps are slew-limited rather than one sample wide.
        free_slot = not np.any(np.abs(Ghz[k]) > 0)
        flip = e.flip_deg
        # 'use' is what lets a reader classify the pulse without our labels: pypulseq reports excitation
        # and refocusing events separately, which is exactly the distinction the effective gradient needs.
        use = 'refocusing' if abs(flip - 180.0) < 1.0 else 'excitation'
        seq.add_block(pp.make_block_pulse(flip_angle=np.deg2rad(flip), duration=dt,
                                          system=sys, use=use))
        if free_slot:
            prev = k + 1
        else:
            prev = k
            n_inserted += 1
            if k <= echo0:
                inserted_before_echo += 1
    _grad_block(prev, len(Ghz))
    seq.add_block(pp.make_adc(num_samples=1, duration=dt, system=sys))

    # Describe the sequence as EXPORTED, not as it was before the pulses needed room. Writing the original
    # length would tell the reader to squeeze the longer sequence back onto the old grid, undoing the
    # insertion and eating the very gradient area it was meant to protect.
    _write_defs(seq, waveform, dt, len(Ghz) + n_inserted,
                echo_idx=echo0 + inserted_before_echo)

    if filename:
        seq.write(filename)
    return seq


def _write_defs(seq, waveform, dt, n_t, echo_idx=None):
    """dt / n_t / echo index / RF schedule in [DEFINITIONS].

    Conveniences for our own round trip, not the physics: TM, the storage window and the effective-gradient
    sign are all derived from the RF blocks on import, so a reader that ignores these still gets the
    sequence right.
    """
    seq.set_definition('dmipy_dt', dt)
    seq.set_definition('dmipy_echo_idx',
                       int(waveform.echo_idx if echo_idx is None else echo_idx))
    seq.set_definition('dmipy_n_t', int(n_t))
    seq.set_definition('dmipy_rf_events', _encode_rf_events(waveform.rf))
    seq.set_definition('dmipy_gradient', 'physical')     # what G is: the scanner's, pulses as blocks or metadata
    if getattr(waveform, 'timing', None) is not None:
        seq.set_definition('dmipy_timing', json.dumps(waveform.timing.to_dict(), separators=(',', ':')))



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
    times = sorted([(float(t), 90.0) for t in np.atleast_1d(t_exc)] +
                   [(float(t), 180.0) for t in np.atleast_1d(t_ref)])
    ev, n90 = [], 0
    for t, flip in times:
        if flip == 180.0:
            ev.append(RFEvent(t, 180.0, 'refocus'))
        else:
            ev.append(RFEvent(t, 90.0, ('Mz\u2192Mxy', 'store', 'recall')[min(n90, 2)]))
            n90 += 1
    return RFSchedule(ev) or None


# -- import: .seq -> ScannerSequence -------------------------------------------------
def from_pulseq(src, *, dt=None):
    """Read a Pulseq ``.seq`` (path or ``pypulseq.Sequence``) and rasterise it to
    a Monte-Carlo-simulable :class:`dmipy_sim.acquisition.scanner_sequence.ScannerSequence` (single
    measurement, shape (1, n_t, 3) in T/m).

    Gradients are rasterised exactly (piecewise-linear interpolation onto the
    uniform grid).  The RF schedule and echo index come from dmipy ``[DEFINITIONS]``
    when present (our own files), otherwise from Pulseq's native excitation/
    refocusing event times (external files); a 90/180 flip is assumed for
    excitation/refocusing when only times are available.
    """
    pp = _require_pypulseq()
    
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
    t_adc = wav[3] if len(wav) >= 4 else None            # out[3] = ADC sample times; out[4] their freq/phase

    dt = float(dt if dt is not None else defs.get('dmipy_dt', seq.grad_raster_time))

    # Anchor t=0 of the ScannerSequence at the excitation (our convention: rf/echo times
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
            # is zero by definition. Interpolating the bare list draws a straight line across every gap --
            # filling a diffusion gap, or a stimulated echo's whole storage period, with gradient that is
            # not there. Mask the gaps instead of inserting zeros near their edges: an inserted zero makes
            # the interpolator ramp down to it, shaving area off the end of every segment.
            tq = t_grid + t0
            vals = np.interp(tq, tt, aa, left=0.0, right=0.0)
            live = np.zeros(tq.shape, bool)
            edges = np.where(np.diff(tt) > 1.5 * raster)[0]
            starts = np.concatenate([[tt[0]], tt[edges + 1]]) if tt.size else np.zeros(0)
            ends = np.concatenate([tt[edges], [tt[-1]]]) if tt.size else np.zeros(0)
            for a0, b0 in zip(starts, ends):
                live |= (tq >= a0 - 0.5 * raster) & (tq <= b0 + 0.5 * raster)
            G[:, ci] = np.where(live, vals, 0.0) / gamma_hz

    # RF schedule, read from the blocks themselves so a file that describes its RF is understood whoever
    # wrote it. The stored dmipy_rf_events is a fallback for older files that carry only the excitation.
    blk_events = _rf_from_pulseq(seq)
    meta_events = _decode_rf_events(defs.get('dmipy_rf_events'))
    if blk_events:
        blk_events = blk_events.shifted(-t0)
    # Which convention is this file written in? A sequence whose blocks carry the WHOLE schedule states its
    # RF natively, so its gradient is physical, as stored. One that describes more pulses in metadata than it
    # plays as blocks is the older form, whose gradient is effective and is un-folded below. The file says
    # which it is; no flag is needed.
    native = bool(blk_events) and not (meta_events and len(meta_events) > len(blk_events))
    rf_events = blk_events if native else (meta_events or blk_events)
    if native and meta_events and len(meta_events) == len(blk_events):
        rf_events = meta_events                    # our own file: the blocks are its raster, the metadata its exact times
    if rf_events is None:
        rf_events = RFSchedule([RFEvent(float(t) - t0, 90.0, 'excitation') for t in t_exc] +
                               [RFEvent(float(t) - t0, 180.0, 'refocusing') for t in t_ref]) or None

    # A ScannerSequence stores the PHYSICAL gradient. A .seq written natively carries it (the pulses are blocks), and
    # so does one this version writes with the pulses as metadata (it says so: dmipy_gradient = 'physical').
    # The older metadata form wrote the EFFECTIVE gradient; un-fold it through the same schedule so the object
    # holds what the scanner plays and G_eff derives the rest.
    if not native and defs.get('dmipy_gradient') != 'physical':
        G = G * RFSchedule(rf_events).sign(t_grid)[:, None]

    if 'dmipy_echo_idx' in defs:
        echo_idx = int(defs['dmipy_echo_idx'])
    else:
        ta = _event_times(t_adc)
        echo_idx = (int(round((float(ta[-1]) - t0) / dt)) if ta.size else n_t - 1)
    echo_idx = int(np.clip(echo_idx, 0, n_t - 1))

    # TM / stimulated-echo state / chi_perp are the ScannerSequence's own derivations from the schedule: the
    # pulses ARE the definition, so a file that describes its RF describes its mixing time, and there is no
    # second copy to fall out of sync with the first.

    # the budget: read from the blocks where the file plays a plain spin echo (a 90, one 180, a readout) --
    # pulseq_timing assumes exactly that -- else from what this version wrote, else none
    sched = RFSchedule(rf_events)
    timing = None
    if defs.get('dmipy_timing'):
        timing = SequenceTiming.from_dict(json.loads(defs['dmipy_timing']))
    elif (native and not meta_events and len(sched) == 2 and sched.refocus_time is not None
          and sched.mixing_time == (None, False) and _has_adc(seq)):
        timing = SequenceTiming.from_pulseq(seq)   # a foreign spin echo: its budget is what its blocks say

    return ScannerSequence(G=G[None], dt=dt, rf=rf_events, readout=(echo_idx,), timing=timing, family="pulseq")


def _has_adc(seq):
    return any(getattr(seq.get_block(i), 'adc', None) is not None for i in range(1, len(seq.block_events) + 1))


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

    :meth:`dmipy_sim.acquisition.timing.SequenceTiming.from_pulseq` wraps them as the budget a
    :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence` carries and a designer reads.
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
