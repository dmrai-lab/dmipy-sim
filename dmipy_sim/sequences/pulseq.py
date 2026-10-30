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

from dmipy_replay.constants import GAMMA, GAMMA_BAR
from dmipy_replay.acquisition.scanners import ScannerLimits
from dmipy_replay.acquisition.rf import RFEvent, RFSchedule
from dmipy_replay.acquisition.timing import SequenceTiming
from dmipy_replay.acquisition.scanner_sequence import ScannerSequence



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
    kw.setdefault('gamma', GAMMA_BAR)
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
                   gamma=GAMMA_BAR)


def _encode_rf_events(rf_events):
    if not rf_events:
        return ''
    return json.dumps(RFSchedule(rf_events).to_dicts(), separators=(',', ':'))








# -- export: ScannerSequence -> .seq -------------------------------------------------

def to_pulseq(waveform, m=0, *, system=None, filename=None,
              excitation_flip_deg=90.0, native_rf=True):
    """Export measurement ``m`` of a :class:`dmipy_sim.acquisition.scanner_sequence.ScannerSequence` to a
    pypulseq ``Sequence`` (written to ``filename`` if given).

    The gradient is emitted as arbitrary-gradient blocks split at the pulses, the pulses as blocks, an ADC at
    the readout; the RF schedule, dt, echo index and budget travel in the ``[DEFINITIONS]`` so ``from_pulseq``
    reconstructs the ScannerSequence faithfully (the round-trip safety net). With no ``system`` the file's
    raster IS the sequence grid (one sample per raster, exact). On a real ``system`` (``make_system(scanner)``)
    the grid must be a whole number of the scanner's gradient rasters -- refused otherwise, naming the grids
    that would fit -- and each step is played as a linear ramp between its samples on that raster: what a
    scanner does with a coarser grid, and what ``from_pulseq`` reads back exactly at the grid's own nodes.
    """
    pp = _require_pypulseq()
    dt = float(waveform.dt)
    # waveform.G is the PHYSICAL gradient, what a scanner plays; the pulses are their own blocks.
    G = np.asarray(waveform.G)[m].astype(float)
    sys = system or _permissive_system(dt)
    gamma_hz = float(getattr(sys, 'gamma', GAMMA_BAR))
    raster = float(getattr(sys, 'grad_raster_time', dt) or dt)
    up = dt / raster                                  # rasters per grid step
    if abs(up - round(up)) > 1e-6:
        TE, q = waveform.T, int(round(waveform.T / raster))
        below = next((m_ for m_ in range(int(up), 0, -1) if q % m_ == 0), None)
        above = next((m_ for m_ in range(int(up) + 1, q + 1) if q % m_ == 0), None)
        near = sorted(q // m_ + 1 for m_ in (below, above) if m_ is not None)
        raise ValueError(f"the grid dt = {dt*1e6:.2f} us is not a whole number of the {raster*1e6:.0f} us gradient "
                         f"raster of this system; at TE = {TE*1e3:.2f} ms the grids n_t = {near} are. Build on the "
                         f"raster rather than resample the sequence.")
    up = int(round(up))
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
                if up > 1:                                # a coarser grid: linear ramps between its samples
                    k = np.arange(len(wf)) * up
                    wf = np.interp(np.arange(k[-1] + 1), k, wf)
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
    if getattr(waveform, 'prescription', None) is not None:
        p = waveform.prescription
        seq.set_definition('FOV', list(p.fov_m))                                    # Pulseq's own field-of-view key
        seq.set_definition('dmipy_prescription', json.dumps(p.to_dict(), separators=(',', ':')))
    if getattr(waveform, 'readout_window', None) is not None:
        seq.set_definition('dmipy_readout_window', json.dumps(list(waveform.readout_window)))









