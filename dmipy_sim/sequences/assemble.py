"""Assemblers x shapes: the one mechanics behind every sequence builder.

A builder is a thin call. It names a SHAPE -- what one encoding block looks like as a function of its
amplitude (a trapezoid lobe, a train of them, a cosine, a bipolar pair, the per-axis pairs of a b-tensor
encoding) -- and an ASSEMBLER -- which RF schedule the block is played around and where the block sits
relative to the pulses -- and hands both the measurement axis. Everything a
:class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence` needs is produced here once: the grid that
runs from t = 0 to TE, the PHYSICAL gradient, the schedule, the exact b, the ``Encoding``.

The assemblers:

* :class:`SpinEcho` -- a 90, one 180 at TE/2, the echo at TE. Every row's block pair is centred on the 180:
  the same block before and after it (what a scanner plays), which the 180 folds into ``G_eff = B, -B`` so
  ``q(TE) = 0`` sample for sample. A row's own gap between its blocks (``Delta - delta - ramp`` for a PGSE, the
  refocusing window for an OGSE) is kept; a TE longer than the minimum adds dead time symmetrically outside
  the pair.
* :class:`StimulatedEcho` -- a 90, a store, a recall; no 180. The block before the store and the same block
  after the recall, with the time transverse before the store equal to the time transverse after the recall
  (the stimulated echo's analogue of a 180 at TE/2, so the static field refocuses at the echo) and the mixing
  time ``TM`` between them; ``TE = 2 t_store + TM``.
* :class:`GradientEcho` -- a 90 and no refocusing pulse: the block must refocus itself (a bipolar pair, the
  per-axis pairs of an STE / PTE). It starts when the excitation lets it; the rest of TE is free precession.
* :class:`EchoTrain` -- a 90 and ``n_echoes`` refocusing pulses at ``(k + 1/2) TE``, an echo at every ``k TE``.
  The gradient is on wherever the pulses and readouts leave room, at one polarity or alternating per interval
  (Carr-Purcell either way in ``G_eff``); each contiguous stretch is one lobe with its own ramps.

Slew is a LIMIT, never a fork: a finite ``slew_rate`` gives every lobe ramps of ``g / slew`` (a cosine, whose
own slope must stay under the limit, gets ramped edges), ``np.inf`` gives vertical ones, and the structure is
the same either way. A block that cannot reach its amplitude inside its own span is refused, not clipped.

A builder passes either ``bvalues`` or ``gradient_strengths``. With ``bvalues`` the amplitude is found by
iteration (the ramps depend on it) and then scaled so ``bvalues == b_from_gradient(G_eff, dt)`` exactly; with
``gradient_strengths`` the b is whatever the played gradient integrates to. A timing budget
(:class:`~dmipy_sim.acquisition.timing.SequenceTiming`) makes the pulses finite and keeps the gradient out of
the lead-in, the refocusing windows and the readout tails; without one the pulses are instantaneous and
constrain nothing.
"""
from __future__ import annotations

import math

import numpy as np

from ..acquisition.rf import RFEvent, RFSchedule
from ..acquisition.scanner_sequence import Encoding, ScannerSequence
from ..constants import resolve_slew
from ..math.gradient_conversions import q_from_g
from ._helpers import _calc_b_from_waveform

__all__ = ["trapezoid", "trapezoid_train", "cosine", "bipolar", "axis_pairs",
           "SpinEcho", "StimulatedEcho", "GradientEcho", "EchoTrain", "assemble", "ramp_of"]

_B_ITER = 16          # amplitude iterations (the ramps depend on the amplitude)
_B_RTOL = 1e-10
_EPS_T = 1e-9         # seconds: placement tolerance against floating rounding


# ── shapes: unit-amplitude blocks sampled on the sequence grid ─────────────────────────────────────────────────

def ramp_of(g, slew_rate):
    """The ramp a lobe of amplitude ``g`` needs at ``slew_rate`` (0 in the instantaneous limit)."""
    slew, square = resolve_slew(slew_rate)
    return 0.0 if square or g <= 0.0 else float(g) / float(slew)


def trapezoid(delta, eps, dt):
    """One unit lobe: a ramp 0 -> 1 over ``eps``, flat until ``delta``, a ramp 1 -> 0 over ``eps`` -- the
    half-amplitude width is ``delta``, the physical span ``delta + eps``; ``eps = 0`` is the square limit.
    Sample ``k`` is the value at the middle of its step, ``(k + 1/2) dt`` -- the engine applies ``G[k]`` over
    ``[k dt, (k + 1) dt)``, so a lobe rasterises symmetrically however short its ramps."""
    delta, eps = float(delta), float(eps)
    if eps > delta + _EPS_T:
        raise ValueError(f"a lobe of {delta*1e3:.3f} ms (half-amplitude width) cannot ramp over {eps*1e3:.3f} ms: "
                         f"the ramp is longer than the lobe -- lower the amplitude, lengthen the lobe or raise the slew")
    n = max(1, int(math.floor((delta + eps) / dt + _EPS_T)))          # whole steps inside the span
    t = (np.arange(n) + 0.5) * dt
    if eps <= 0.0:
        return np.ones(n)
    a = np.ones(n)
    up = t < eps
    a[up] = t[up] / eps
    dn = t >= delta
    a[dn] = np.clip((delta + eps - t[dn]) / eps, 0.0, 1.0)
    return a


def trapezoid_train(n_lobes, lobe, eps, dt):
    """``n_lobes`` adjacent unit lobes of span ``lobe`` each, alternating in sign (+, -, +, ...): the trapezoidal
    OGSE block (Drobnjak et al. 2016); one lobe is a PGSE lobe. Each lobe ramps over ``eps`` at both ends."""
    if eps > lobe / 2.0 + _EPS_T:
        raise ValueError(f"a lobe of {lobe*1e3:.3f} ms cannot ramp up and down over {eps*1e3:.3f} ms each at this "
                         f"amplitude and slew: lower the amplitude, the frequency, or raise the slew")
    one = trapezoid(lobe - eps, eps, dt)
    n1 = len(one)
    out = np.empty(int(n_lobes) * n1)
    for k in range(int(n_lobes)):
        out[k * n1:(k + 1) * n1] = one if k % 2 == 0 else -one
    return out


def cosine(n_cycles, frequency, eps, dt):
    """``n_cycles`` whole periods of ``cos(2 pi f t)`` -- the frequency-selective OGSE block, DC-free in q -- with
    its edges ramped over ``eps`` (the cosine starts and ends at full amplitude; a scanner ramps into it)."""
    sigma = float(n_cycles) / float(frequency)
    n = max(2, int(math.floor(sigma / dt + _EPS_T)))
    t = (np.arange(n) + 0.5) * dt                                     # mid-step, as every shape
    a = np.cos(2.0 * np.pi * float(frequency) * t)
    if eps > 0.0:
        a = a * np.clip(t / eps, 0.0, 1.0) * np.clip((sigma - t) / eps, 0.0, 1.0)
    return a


def bipolar(delta, Delta, eps, dt):
    """A self-refocusing pair: a unit lobe at 0 and its negative ``Delta`` later (centre to centre)."""
    if float(Delta) < float(delta) + float(eps) - _EPS_T:
        raise ValueError("the two lobes of a bipolar pair overlap: Delta is shorter than the lobe with its ramp")
    lobe = trapezoid(delta, eps, dt)
    i1 = int(round(float(Delta) / dt))
    out = np.zeros(i1 + len(lobe))
    out[:len(lobe)] += lobe
    out[i1:] -= lobe
    return out


def axis_pairs(axes, duration, eps, dt):
    """Sequential self-refocusing pairs, one per axis in ``axes`` (rows, unit vectors), back to back over
    ``duration``: the canonical spherical (three axes) and planar (two) tensor encodings. Returns ``(n, 3)``;
    each axis's q returns to zero inside its own pair, so the off-diagonal B is exactly zero."""
    axes = np.asarray(axes, dtype=np.float64)
    n_seg = 2 * len(axes)
    seg = float(duration) / n_seg
    lobe = trapezoid(seg - eps, eps, dt)
    n1 = len(lobe)
    out = np.zeros((n_seg * n1, 3))
    for i, ax in enumerate(axes):
        a0 = (2 * i) * n1
        out[a0:a0 + n1] += lobe[:, None] * ax
        out[a0 + n1:a0 + 2 * n1] -= lobe[:, None] * ax
    return out


# ── assemblers: an RF schedule and where the blocks sit around it ──────────────────────────────────────────────

def _lead(timing):
    return 0.0 if timing is None else float(timing.t_lead)


def _tail(timing):
    return 0.0 if timing is None else float(timing.t_readout_pre_echo)


def _dur(timing, which):
    return 0.0 if timing is None else float(getattr(timing, which))


def _steps_before(t, dt):
    """How many steps ``[k dt, (k + 1) dt)`` end by time ``t``: the exclusive end index of a block that must be
    over by ``t``."""
    return int(math.floor(t / dt + _EPS_T))


def _first_step_at(t, dt):
    """The first step that starts at or after time ``t``."""
    return int(math.ceil(t / dt - _EPS_T))


# A placement says where a block goes. Sample k acts over the step [k dt, (k + 1) dt) (the engine's rule), and
# the readout sample n_t - 1 acts over nothing. ("end", t, factor): the block's last step ends by time t (it
# ends against a pulse, or at the readout); ("start", t, factor): its first step starts at or after t (it
# follows a pulse); ("at", i0, factor, n): exactly n samples from sample i0 (a stretch handed out to fill).


class SpinEcho:
    """90 at 0, 180 at TE/2, echo at TE; each row's block pair centred on the 180 with its own ``gap``."""

    def __init__(self, gap, timing=None):
        self.gap = gap                    # callable (m, g) -> seconds between the two blocks
        self.timing = timing

    def te_min(self, spans, g):
        gaps = np.array([self.gap(m, g[m]) for m in range(len(spans))])
        if np.any(gaps < -_EPS_T):
            raise ValueError("the two encoding blocks overlap: Delta is shorter than the lobe with its ramp")
        if self.timing is not None and np.any(gaps < self.timing.t_refocus - _EPS_T):
            raise ValueError(f"the gap between the blocks, {float(np.min(gaps))*1e3:.3f} ms, is narrower than the "
                             f"refocusing window {self.timing.t_refocus*1e3:.3f} ms: the 180 does not fit")
        return float(np.max(2.0 * spans + gaps)) + 2.0 * max(_lead(self.timing), _tail(self.timing))

    def layout(self, spans, g, TE, dt, n_t):
        schedule = (self.timing.rf_events(TE) if self.timing is not None else
                    RFSchedule([RFEvent(0.0, 90, 'Mz→Mxy'), RFEvent(TE / 2.0, 180, 'refocus')]))
        placements = []
        for m, span in enumerate(spans):
            gap = self.gap(m, g[m])
            placements.append([("end", TE / 2.0 - gap / 2.0, 1.0), ("start", TE / 2.0 + gap / 2.0, 1.0)])
        return schedule, placements


class StimulatedEcho:
    """90 at 0, store, recall; the block before the store and after the recall, ``TE = 2 t_store + TM``."""

    def __init__(self, TM, flips=(90.0, 90.0, 90.0), timing=None):
        self.TM = float(TM)
        self.flips = tuple(float(a) for a in flips)
        self.timing = timing

    def te_min(self, spans, g):
        w = _dur(self.timing, "t_excite")
        return 2.0 * float(np.max(spans)) + w + self.TM + 2.0 * max(_lead(self.timing), _tail(self.timing))

    def layout(self, spans, g, TE, dt, n_t):
        w = _dur(self.timing, "t_excite")
        s_max = float(np.max(spans))
        d = (TE - self.TM - 2.0 * s_max - w) / 2.0
        t_store = d + s_max + w / 2.0
        t_recall = t_store + self.TM
        a1, a2, a3 = self.flips
        schedule = RFSchedule([RFEvent(_dur(self.timing, "t_prep"), a1, 'Mz→Mxy', duration_s=w),
                               RFEvent(t_store, a2, 'store', duration_s=w),
                               RFEvent(t_recall, a3, 'recall', duration_s=w)])
        return schedule, [[("end", t_store - w / 2.0, 1.0), ("start", t_recall + w / 2.0, 1.0)] for _ in spans]


class GradientEcho:
    """90 at 0 and no refocusing pulse; the (self-refocusing) block starts when the excitation lets it."""

    def __init__(self, timing=None):
        self.timing = timing

    def te_min(self, spans, g):
        return _lead(self.timing) + float(np.max(spans)) + _tail(self.timing)

    def layout(self, spans, g, TE, dt, n_t):
        schedule = RFSchedule([RFEvent(_dur(self.timing, "t_prep"), 90, 'Mz→Mxy',
                                       duration_s=_dur(self.timing, "t_excite"))])
        return schedule, [[("start", _lead(self.timing), 1.0)] for _ in spans]


class EchoTrain:
    """90 at 0 and ``n_echoes`` refocusing pulses of ``beta_deg`` at ``(k + 1/2) TE``; the gradient fills every
    stretch the pulses and readouts leave, at one polarity or alternating per echo interval."""

    def __init__(self, n_echoes, TE, polarity="constant", beta_deg=180.0, timing=None):
        if polarity not in ("constant", "alternate"):
            raise ValueError(f"polarity must be 'constant' or 'alternate', got {polarity!r}")
        self.n_echoes, self.TE_echo = int(n_echoes), float(TE)
        self.polarity, self.beta = polarity, float(beta_deg)
        self.timing = timing

    def te_min(self, spans, g):
        return self.n_echoes * self.TE_echo

    def n_t_of(self, n_t_per_echo):
        """The grid: ``n_t_per_echo`` samples per interval, every echo on a sample."""
        return self.n_echoes * int(n_t_per_echo) + 1

    def windows(self):
        """Gradient-off intervals over the train: each refocusing window, and the same dead time -- the larger of
        the lead-in and the readout tail -- on both sides of every echo, so the two halves of every interval
        carry the same gradient and every echo refocuses."""
        tm = self.timing
        if tm is None:
            return []
        d = max(tm.t_lead, tm.t_readout_pre_echo)
        w = []
        for k in range(self.n_echoes):
            c = (k + 0.5) * self.TE_echo
            w += [(k * self.TE_echo, k * self.TE_echo + d), (c - tm.t_refocus / 2.0, c + tm.t_refocus / 2.0),
                  ((k + 1) * self.TE_echo - d, (k + 1) * self.TE_echo)]
        return w

    def layout(self, spans, g, TE, dt, n_t):
        tm = self.timing
        schedule = RFSchedule([RFEvent(_dur(tm, "t_prep"), 90, 'Mz→Mxy', duration_s=_dur(tm, "t_excite"))] +
                              [RFEvent((k + 0.5) * self.TE_echo, self.beta, 'refocus', duration_s=_dur(tm, "t_refocus"))
                               for k in range(self.n_echoes)])
        t = np.arange(n_t) * dt
        on = np.ones(n_t, bool)
        on[-1] = False                                                # the readout sample acts over nothing
        for a, b in self.windows():
            on &= ~((t >= a - _EPS_T) & (t <= b + _EPS_T))            # samples inside a window, as validate() reads it
        interval = np.minimum(np.floor(t / self.TE_echo + _EPS_T).astype(int), self.n_echoes - 1)
        pol = np.where(interval % 2 == 0, 1.0, -1.0) if self.polarity == "alternate" else np.ones(n_t)
        # one lobe per stretch, and a stretch never crosses an echo: each interval refocuses on its own
        key = np.where(on, pol * (interval + 1), 0.0)
        runs = []                                                     # (start, length, polarity) of each stretch
        i = 0
        while i < n_t:
            if key[i] == 0.0:
                i += 1
                continue
            j = i
            while j < n_t and key[j] == key[i]:
                j += 1
            runs.append(("at", i, float(np.sign(key[i])), j - i))
            i = j
        return schedule, [list(runs) for _ in spans]


# ── the driver ──────────────────────────────────────────────────────────────────────────────────────────────────

def _place(G, m, i0, block):
    n = block.shape[0]
    if i0 < 0 or i0 + n > G.shape[1] - 1:                             # the last sample is the readout: not a step
        raise ValueError(f"an encoding block of {n} samples at sample {i0} runs off the {G.shape[1]}-sample grid")
    G[m, i0:i0 + n] += block


def assemble(assembler, *, gradient_directions, span, bvalues=None, gradient_strengths=None, sample=None,
             fill=None, TE=None, n_t=1000, slew_rate=np.inf, family, build_spec, encoding=None, q_width=None,
             timing=None):
    """Build a validated :class:`ScannerSequence` from an assembler and a shape.

    ``span(m, g)`` is row ``m``'s block duration at amplitude ``g``; ``sample(m, g, dt)`` its unit block, ``(n,)``
    along the row's direction or ``(n, 3)``; ``fill(m, g, dt, n)`` a unit block of ``n`` samples for an assembler
    that hands out stretches to fill (the echo train). Exactly one of ``bvalues`` / ``gradient_strengths`` sets
    the amplitude; ``encoding(g, TE, te_min)`` returns the family's extra ``Encoding`` fields; ``q_width`` is the
    lobe width ``q = gamma g width / 2 pi`` reports.
    """
    dirs = np.asarray(gradient_directions, dtype=np.float64)
    if dirs.ndim == 1:
        dirs = dirs[None]
    n_m = dirs.shape[0]
    if bvalues is not None and gradient_strengths is not None:
        raise ValueError("give bvalues= or gradient_strengths=, not both")
    b_target = None if bvalues is None else np.broadcast_to(np.asarray(bvalues, np.float64), (n_m,)).copy()
    if b_target is not None and np.any(b_target < 0):
        raise ValueError("bvalues must be non-negative")
    if gradient_strengths is not None:
        g = np.broadcast_to(np.asarray(gradient_strengths, np.float64), (n_m,)).copy()
        if np.any(g < 0):
            raise ValueError("gradient_strengths must be non-negative")
    else:
        g = np.zeros(n_m)                                             # the square unit shape seeds the iteration

    def _build(g, amp):
        spans = np.array([float(span(m, g[m])) for m in range(n_m)])
        te_min = assembler.te_min(spans, g)
        te = te_min if TE is None else float(TE)
        if te < te_min - _EPS_T:
            raise ValueError(f"TE = {te*1e3:.3f} ms is below the {te_min*1e3:.3f} ms this encoding needs "
                             f"(its blocks, their gap and the pulse / readout windows)")
        n = int(n_t) if not hasattr(assembler, "n_t_of") else assembler.n_t_of(n_t)
        dt = te / (n - 1)
        schedule, placements = assembler.layout(spans, g, te, dt, n)
        G = np.zeros((n_m, n, 3), dtype=np.float64)
        for m in range(n_m):
            for kind, where, factor, *rest in placements[m]:
                blk = fill(m, g[m], dt, rest[0]) if kind == "at" else sample(m, g[m], dt)
                blk = np.asarray(blk, dtype=np.float64)
                if blk.ndim == 1:
                    blk = blk[:, None] * dirs[m]
                if kind == "at":
                    i0 = int(where)
                elif kind == "start":
                    i0 = _first_step_at(where, dt)
                else:
                    i0 = _steps_before(where, dt) - blk.shape[0]
                _place(G, m, i0, factor * amp[m] * blk)
        sign = schedule.sign(np.arange(n) * dt)[None, :, None]
        return G, dt, schedule, sign, te, te_min

    if b_target is None:
        G, dt, schedule, sign, te, te_min = _build(g, g)
    else:
        G, dt, schedule, sign, te, te_min = _build(np.zeros(n_m), np.ones(n_m))   # square unit blocks seed the amplitude
        b_unit = _calc_b_from_waveform(G * sign, dt)
        g = np.where((b_target > 0) & (b_unit > 0), np.sqrt(b_target / np.where(b_unit > 0, b_unit, 1.0)), 0.0)
        for _ in range(_B_ITER):                                      # the ramps follow the amplitude
            G, dt, schedule, sign, te, te_min = _build(g, g)
            b_num = _calc_b_from_waveform(G * sign, dt)
            live = (b_target > 0) & (b_num > 0)
            scale = np.where(live, np.sqrt(b_target / np.where(b_num > 0, b_num, 1.0)), 1.0)
            g = g * scale
            if np.all(np.abs(scale - 1.0) < _B_RTOL):
                break
        # the declared b IS the numeric b of the effective gradient the walk integrates
        b_num = _calc_b_from_waveform(G * sign, dt)
        live = (b_target > 0) & (b_num > 0)
        scale = np.where(live, np.sqrt(b_target / np.where(b_num > 0, b_num, 1.0)), 1.0)
        G = G * scale[:, None, None]
        g = g * scale
    b_out = _calc_b_from_waveform(G * sign, dt) if b_target is None else b_target
    extra = {} if encoding is None else dict(encoding(g, te, te_min))
    q = None if q_width is None else q_from_g(g, np.broadcast_to(np.asarray(q_width, np.float64), (n_m,)))
    enc = Encoding(bvalues=b_out, gradient_directions=dirs, TE=np.full(n_m, te), qvalues=q, gradient_strengths=g,
                   minimum_te=te_min, te_auto=TE is None, **extra)
    return ScannerSequence(G=G, dt=dt, rf=schedule, timing=timing, family=family, encoding=enc,
                           build_spec=build_spec).validate()
