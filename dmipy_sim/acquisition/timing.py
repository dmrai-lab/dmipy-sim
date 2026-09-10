"""The timing budget of a spin echo: where the gradient may not be on.

A scanner's sequence is not only its gradient and its pulses -- the pulses take time, the readout takes
time, a preparation may precede them -- and diffusion encoding is OFF during all of it. :class:`SequenceTiming`
is that budget for a spin echo whose 180 sits at ``TE/2`` (the refocus condition): the excitation lead-in,
the refocusing window (RF, and any crusher time around it), the readout tail before the echo, an optional
preparation. Everything else is derived from it -- the smallest echo time both encoding windows exist for,
the gradient-off windows and their mask on a grid, and the finite RF schedule the budget implies. The two
encoding windows it leaves are generally UNEQUAL, since the lead-in and the readout tail differ: any
pre/post-180 asymmetry of a designed waveform is a consequence of the budget, never a knob::

    [prep + excite] [== pre-180 encode ==] [180] [== post-180 encode ==] [readout -> echo]
    0               t_lead                 TE/2 -+ t_refocus/2          TE - t_ro_pre     TE

Times in seconds; ``t = 0`` is the excitation centre (echo time is centre-to-centre, as a scanner counts
it), so the excitation pulse straddles t = 0. Build it from a readout description (:meth:`from_readout`)
or from a real ``.seq`` (:meth:`from_pulseq`). A :class:`~dmipy_sim.acquisition.waveforms.Waveform` or
:class:`~dmipy_sim.sequences.Sequence` carries one as ``timing`` when it was built to a budget, and
``validate()`` then refuses gradient inside its windows.
"""
from dataclasses import dataclass, asdict

import numpy as np

from .rf import RFEvent, RFSchedule

__all__ = ["SequenceTiming"]


@dataclass(frozen=True)
class SequenceTiming:
    """The spin-echo timing budget: the windows the gradient must stay out of.

    ``t_excite`` is the 90's duration (encoding starts after it), ``t_refocus`` the gradient-off window
    around the 180 at ``TE/2`` (the RF and any crusher time), ``t_readout_pre_echo`` the readout's start
    to the echo (post-180 encoding must end before it), ``t_prep`` an optional preparation before the
    excitation, ``TE`` the native echo time when the budget came from a real sequence (``None`` = whatever
    a builder chooses at or above :meth:`min_TE`).
    """
    t_excite: float
    t_refocus: float
    t_readout_pre_echo: float
    t_prep: float = 0.0
    TE: float = None

    def __post_init__(self):
        for k in ("t_excite", "t_refocus", "t_readout_pre_echo", "t_prep"):
            v = float(getattr(self, k))
            if v < 0.0:
                raise ValueError(f"{k} must be >= 0, got {v}")
            object.__setattr__(self, k, v)
        if self.TE is not None:
            object.__setattr__(self, "TE", float(self.TE))
            if self.TE < self.min_TE() - 1e-12:
                raise ValueError(f"TE = {self.TE*1e3:.3f} ms is below min_TE = {self.min_TE()*1e3:.3f} ms for "
                                 f"this budget: an encoding window would vanish")

    @property
    def t_lead(self):
        """Dead time from t = 0 (the excitation centre) until encoding may begin."""
        return self.t_prep + self.t_excite

    def min_TE(self):
        """The smallest echo time for which both the pre- and the post-180 encoding windows exist."""
        return max(2.0 * (self.t_lead + self.t_refocus / 2.0),
                   2.0 * (self.t_readout_pre_echo + self.t_refocus / 2.0))

    def resolve_TE(self, TE=None):
        """The echo time to build to: ``TE`` if given, else the budget's own, else :meth:`min_TE`; refused below
        :meth:`min_TE`."""
        te = float(TE if TE is not None else (self.TE if self.TE is not None else self.min_TE()))
        if te < self.min_TE() - 1e-12:
            raise ValueError(f"TE = {te*1e3:.3f} ms is below min_TE = {self.min_TE()*1e3:.3f} ms for this budget: "
                             f"an encoding window would vanish")
        return te

    def windows(self, TE=None):
        """The gradient-off windows ``[(t0, t1, what), ...]`` for an echo at ``TE``: the lead-in, the refocusing
        window centred on ``TE/2``, the readout tail."""
        te = self.resolve_TE(TE)
        return [(0.0, self.t_lead, "excitation"),
                (te / 2.0 - self.t_refocus / 2.0, te / 2.0 + self.t_refocus / 2.0, "refocus"),
                (te - self.t_readout_pre_echo, te, "readout")]

    def encoding_windows(self, TE=None):
        """The two windows the gradient may be on: ``((pre0, pre1), (post0, post1))``."""
        te = self.resolve_TE(TE)
        return ((self.t_lead, te / 2.0 - self.t_refocus / 2.0),
                (te / 2.0 + self.t_refocus / 2.0, te - self.t_readout_pre_echo))

    def on_mask(self, t, TE=None):
        """``1.0`` where the gradient may be on at the times ``t``, ``0.0`` in an off window."""
        t = np.asarray(t, dtype=np.float64)
        on = np.ones_like(t)
        for t0, t1, _ in self.windows(TE):
            on[(t >= t0 - 1e-12) & (t <= t1 + 1e-12)] = 0.0
        return on

    def rf_events(self, TE=None):
        """The finite RF schedule the budget implies: a 90 of ``t_excite`` centred on ``t_prep`` (the excitation
        centre is t = 0 when there is no preparation) and a 180 of ``t_refocus`` centred on ``TE/2``."""
        te = self.resolve_TE(TE)
        return RFSchedule([RFEvent(self.t_prep, 90, 'Mz→Mxy', duration_s=self.t_excite),
                           RFEvent(te / 2.0, 180, 'refocus', duration_s=self.t_refocus)])

    @classmethod
    def from_readout(cls, *, t_excite, t_refocus, readout_duration, partial_fourier, t_prep=0.0, TE=None):
        """From a readout description: the echo (k-space centre) sits ``(pf - 0.5) / pf`` into the readout, so
        partial Fourier (``pf < 1``) shortens the post-180 window -- the mechanism that makes an optimum
        asymmetric."""
        pf = float(partial_fourier)
        if not (0.5 <= pf <= 1.0):
            raise ValueError(f"partial_fourier must be in [0.5, 1.0]; got {pf}")
        return cls(float(t_excite), float(t_refocus), float(readout_duration) * (pf - 0.5) / pf, float(t_prep), TE)

    @classmethod
    def from_pulseq(cls, src):
        """Read the budget (and the native ``TE``) from a Pulseq ``.seq`` -- the first RF block the 90, the second
        the 180, one ADC -- through :func:`dmipy_sim.sequences.pulseq.pulseq_timing`."""
        from ..sequences.pulseq import pulseq_timing
        d = pulseq_timing(src)
        return cls(t_excite=d['t_excite'], t_refocus=d['t_refocus'], t_readout_pre_echo=d['t_readout_pre_echo'],
                   TE=d['TE'])

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
