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
def to_pulseq(waveform, m=0, *, system=None, filename=None,
              excitation_flip_deg=90.0):
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
    G = np.asarray(waveform.G)[m].astype(float)      # (n_t, 3) T/m
    dt = float(waveform.dt)
    sys = system or _permissive_system(dt)
    gamma_hz = float(getattr(sys, 'gamma', GAMMA_HZ))
    seq = pp.Sequence(system=sys)

    # excitation
    seq.add_block(pp.make_block_pulse(
        flip_angle=np.deg2rad(excitation_flip_deg), duration=dt, system=sys))

    # gradient: one arbitrary block, all active channels concurrent.  Zero-pad the
    # endpoints (Pulseq requires arbitrary grads to start/end at 0).
    Ghz = G * gamma_hz                                # Hz/m
    grads = []
    for ci, ch in enumerate(('x', 'y', 'z')):
        col = Ghz[:, ci]
        if np.any(col):
            padded = np.concatenate([[0.0], col, [0.0]])
            grads.append(pp.make_arbitrary_grad(channel=ch, waveform=padded, system=sys))
    if grads:
        seq.add_block(*grads)
    seq.add_block(pp.make_adc(num_samples=1, duration=dt, system=sys))

    seq.set_definition('dmipy_dt', dt)
    seq.set_definition('dmipy_echo_idx', int(waveform.echo_idx))
    seq.set_definition('dmipy_n_t', int(G.shape[0]))
    seq.set_definition('dmipy_rf_events', _encode_rf_events(waveform.rf_events))

    if filename:
        seq.write(filename)
    return seq


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
    for ci in range(min(3, len(gw))):
        arr = np.asarray(gw[ci], dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            G[:, ci] = np.interp(t_grid + t0, arr[0], arr[1],
                                 left=0.0, right=0.0) / gamma_hz

    # RF schedule
    rf_events = _decode_rf_events(defs.get('dmipy_rf_events'))
    if rf_events is None:
        rf_events = ([{'t_s': float(t) - t0, 'flip_deg': 90.0, 'label': 'excitation'}
                      for t in t_exc] +
                     [{'t_s': float(t) - t0, 'flip_deg': 180.0, 'label': 'refocusing'}
                      for t in t_ref]) or None

    if 'dmipy_echo_idx' in defs:
        echo_idx = int(defs['dmipy_echo_idx'])
    else:
        ta = _event_times(t_adc)
        echo_idx = (int(round((float(ta[-1]) - t0) / dt)) if ta.size else n_t - 1)
    echo_idx = int(np.clip(echo_idx, 0, n_t - 1))

    return Waveform(G=jnp.asarray(G[None]), dt=dt, echo_idx=echo_idx,
                    rf_events=rf_events)


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
