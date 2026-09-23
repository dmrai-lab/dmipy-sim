"""Coherence pathways of a pulse sequence: the extended phase graph, enumerated.

A sequence of RF pulses and gradient windings does not produce one signal pathway but many. A perfect
refocusing pulse leaves a single spin-echo pathway; any other flip angle splits the magnetisation at every
pulse, and the branches that come back to coherence order zero at a readout all contribute to the echo --
each having spent a different part of the sequence transverse (decaying at T2, accruing gradient phase) or
stored longitudinally (decaying at T1, accruing none). The amplitudes of those branches are what an
extended phase graph computes, and they are the weights a replay needs; they are not free parameters and
they are not to be written down by hand.

**What this is for.** :func:`enumerate_pathways` turns a :class:`Schedule` into a list of :class:`Pathway`,
each carrying its complex amplitude ``eta`` and, laid onto a trajectory's time grid, the two arrays a replay
consumes: :meth:`Pathway.chi_perp` (1 while transverse, 0 while stored -- which relaxation applies, and
whether wall contact accrues) and :meth:`Pathway.eps_P` (+1 / -1 / 0 -- the sign with which phase
accumulates, so the pathway's own effective gradient is ``G * eps_P``). One walk then answers every pathway:

    S = sum_p  eta_p * replay(trajectory, ..., chi_perp=p.chi_perp(dt), eps_P=p.eps_P(dt))

**Phases are the lab's.** ``phase_deg`` is the B1 phase, 0 about x and 90 about y, for every pulse
including the excitation, and the equilibrium magnetisation is tipped into the transverse plane exactly as
that pulse leaves it. So the CPMG geometry -- refocusing about the axis the magnetisation lies along -- is a
train whose refocusing pulses are 90 degrees from its excitation, and that is what this expresses. An
excitation's own phase multiplies every pathway alike, so it cancels from any ratio and never moves an echo's
magnitude; a perfect refocusing train gives ``|eta| = 1`` at every echo, its sign alternating with each pulse.

**The idealisations.** Pulses are instantaneous, windings are integer multiples of one coherence order, and
relaxation is applied by the consumer rather than here, so the amplitudes depend on the flip angles and
phases alone and not on the substrate. Reference: Weigel M (2015), *Extended phase graphs: dephasing, RF
pulses, and echoes*, JMRI 41(2):266-295, doi:10.1002/jmri.24619.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Pulse", "Winding", "Schedule", "Pathway", "enumerate_pathways", "splice_schedule",
           "cpmg_schedule", "ste_schedule", "ste_amplitude", "pathway_weight"]


@dataclass(frozen=True)
class Pulse:
    """An instantaneous RF pulse on the coherence lattice: its flip and its B1 phase, in degrees.

    ``phase_deg`` is the B1 phase in the lab, 0 about x and 90 about y, and it is read on every pulse
    including the excitation.
    """
    flip_deg: float
    phase_deg: float = 0.0


@dataclass(frozen=True)
class Winding:
    """A free-precession interval: the gradient winds every transverse coherence by ``dk`` orders.

    ``dk`` is the interval's net winding in whole coherence orders, signed by the lobe's polarity; a
    crusher or an unbalanced readout is simply a winding of more than one. ``duration`` is the interval's
    length in seconds, which is what lays a pathway's gates onto a trajectory grid, and ``readout`` marks an
    echo acquisition at the end of the interval.
    """
    dk: int
    duration: float = 0.0
    readout: bool = False


@dataclass(frozen=True)
class Schedule:
    """A sequence as an ordered timeline of :class:`Pulse` and :class:`Winding` events.

    The gradient's shape lives on the :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence`;
    this is the RF and winding structure alone, which is what decides which pathways exist.
    """
    events: tuple


@dataclass(frozen=True)
class Pathway:
    """One branch that reaches coherence order zero at a readout: its amplitude and its history.

    ``intervals`` is ``((state, duration), ...)`` over the windings up to the readout, with ``state`` one of
    ``'F+'``, ``'F-'`` (transverse) or ``'Z'`` (stored).
    """
    eta: complex
    intervals: tuple
    readout_idx: int

    def _lay(self, dt, value):
        segs = [np.full(max(1, int(round(d / float(dt)))), value(st), float) for st, d in self.intervals]
        return np.concatenate(segs) if segs else np.zeros(0)

    def chi_perp(self, dt):
        """The transverse gate on a grid of step ``dt``: 1 while this pathway is transverse, 0 while stored."""
        return self._lay(dt, lambda st: 1.0 if st in ("F+", "F-") else 0.0)

    def eps_P(self, dt):
        """The sign with which phase accumulates on a grid of step ``dt``: +1 on ``F+``, -1 on ``F-``, 0 stored.

        The pathway's own effective gradient is the sequence's ``G`` times this array.
        """
        return self._lay(dt, lambda st: {"F+": 1.0, "F-": -1.0, "Z": 0.0}[st])


def _branch(paths, flip_deg, phase_deg, threshold):
    """Every pathway through one pulse of flip ``flip_deg`` and phase ``phase_deg`` (Weigel's operator).

    At phase zero this is the rotation about x; a phase ``phi`` carries the three factors ``exp(2i phi)`` on
    the transverse-to-conjugate term and ``exp(+-i phi)`` on the terms that couple to and from the
    longitudinal store.
    """
    a, phi = np.radians(float(flip_deg)), np.radians(float(phase_deg))
    c2, s2, sa, ca = np.cos(a / 2) ** 2, np.sin(a / 2) ** 2, np.sin(a), np.cos(a)
    e1, e2 = np.exp(1j * phi), np.exp(2j * phi)
    out = []
    for k, st, amp, hist in paths:
        if st == "F+":
            cand = ((k, "F+", c2 * amp, hist), (k, "F-", np.conj(e2) * s2 * amp, hist),
                    (k, "Z", -0.5j * sa * np.conj(e1) * amp, hist))
        elif st == "F-":
            cand = ((k, "F+", e2 * s2 * amp, hist), (k, "F-", c2 * amp, hist),
                    (k, "Z", 0.5j * sa * e1 * amp, hist))
        else:
            cand = ((k, "F+", -1j * sa * e1 * amp, hist), (k, "F-", 1j * sa * np.conj(e1) * amp, hist),
                    (k, "Z", ca * amp, hist))
        out.extend(c for c in cand if abs(c[2]) >= threshold)
    return out


def _wind_once(paths, step):
    """One unit winding: transverse orders move by ``step``, the store does not, and an order crossing zero
    becomes its conjugate partner (``F+`` at order 0 wound down is ``F-`` at order 1)."""
    out = []
    for k, st, amp, hist in paths:
        if st == "Z":
            out.append((k, "Z", amp, hist))
        elif st == "F+":
            nk = k + step
            out.append((nk, "F+", amp, hist) if nk >= 0 else (1, "F-", np.conj(amp), hist))
        else:
            nk = k - step
            out.append((nk, "F-", amp, hist) if nk >= 0 else (1, "F+", np.conj(amp), hist))
    return out


def _wind(paths, dk):
    """A winding of any whole number of orders, as the unit step repeated."""
    step = 1 if dk > 0 else -1
    for _ in range(abs(int(dk))):
        paths = _wind_once(paths, step)
    return paths


def enumerate_pathways(schedule, threshold=1e-2):
    """Every pathway of ``schedule`` that reaches coherence order zero at a readout.

    Walks the timeline once from equilibrium, branching at each :class:`Pulse` and winding at each
    :class:`Winding`, and collects the transverse coherence at order zero wherever a readout is marked. A
    branch whose amplitude falls below ``threshold`` is dropped, so the cost is bounded by what actually
    contributes. Returns the pathways sorted by readout and then by descending amplitude.
    """
    paths, out, readout_idx, excited = [], [], 0, False
    for ev in schedule.events:
        if isinstance(ev, Pulse):
            if not excited:
                # The excitation is seeded rather than branched: at order zero the two transverse states are
                # one hermitian coherence, and branching would count the initial magnetisation twice. The
                # seed is the physical one a pulse of this flip and phase makes out of equilibrium,
                # ``-i exp(i phi) sin(a)``, so that a phase is a LAB B1 phase and a refocusing pulse 90
                # degrees from the excitation is the CPMG geometry.
                a, phi = np.radians(float(ev.flip_deg)), np.radians(float(ev.phase_deg))
                paths = [p for p in ((0, "F+", -1j * np.exp(1j * phi) * np.sin(a), ()),
                                     (0, "Z", complex(np.cos(a)), ()))
                         if abs(p[2]) >= threshold]
                excited = True
            else:
                paths = _branch(paths, ev.flip_deg, ev.phase_deg, threshold)
        else:
            paths = [(k, st, amp, hist + ((st, ev.duration),)) for k, st, amp, hist in paths]
            paths = _wind(paths, ev.dk)
            if ev.readout:
                out.extend(Pathway(eta=amp, intervals=hist, readout_idx=readout_idx)
                           for k, st, amp, hist in paths if k == 0 and st in ("F+", "F-") and abs(amp) >= threshold)
                readout_idx += 1
    out.sort(key=lambda p: (p.readout_idx, -abs(p.eta)))
    return out


def splice_schedule(n_echoes, beta_deg, ESP=1.0, refocus_phase_deg=90.0):
    """A SPLICE train: the split acquisition of fast spin-echo signals (Schick 1997).

    The readout gradient is **prolonged and unbalanced** -- the pre-phaser carries a quarter of the area the
    readout then plays, not the half an ordinary fast spin echo uses -- so the interval winds three orders
    instead of two and TWO echoes form in it rather than one. Written as the winding structure alone:

        pre-phaser   +1 over ESP / 4
        refocus      beta
        E1           +1 over ESP / 4   read here
        E2           +1 over ESP / 2   read here

    What the two echoes separate is **not** spin echoes from stimulated echoes: each family carries both.
    They are the two CONJUGATION PARITIES -- pathways refocused an even and an odd number of times -- which
    is why the split works at all after a diffusion preparation. The preparation leaves every spin an
    arbitrary phase, and that phase enters the two families as ``+phi`` and ``-phi``, constant within each,
    so each family's MAGNITUDE is insensitive to it. The two are reconstructed separately and their
    magnitude images summed. That is what lets this family violate the CPMG condition and survive.

    At ``beta_deg = 180`` the split degenerates: nothing is stored along z, one pathway survives, and it
    lands alternately in one family and the other (1, 0, 1, 0 against 0, 1, 0, 1), so each k-space would get
    only every second line. A real SPLICE train therefore runs below 180.

    Amplitudes reproduce the reference implementation of Rahbek et al. 2023 (MRM 89:1469, ``epg_splice.m``)
    exactly; the first few have closed forms: ``sin^2(beta/2)`` and ``(1/2) sin^2(beta)`` for E1's first two
    echoes, ``sin^4(beta/2)`` and ``sin^2(beta/2) sin^2(beta)`` for E2's second and third.
    """
    ev = [Pulse(90.0)]
    for _ in range(int(n_echoes)):
        ev += [Winding(+1, float(ESP) / 4),
               Pulse(float(beta_deg), float(refocus_phase_deg)),
               Winding(+1, float(ESP) / 4, readout=True),
               Winding(+1, float(ESP) / 2, readout=True)]
    return Schedule(tuple(ev))


def cpmg_schedule(n_echoes, beta_deg=180.0, TE=1.0, refocus_phase_deg=90.0):
    """A CPMG train: an excitation, then ``n_echoes`` refocusing pulses each between two unit windings.

    ``refocus_phase_deg`` is the refocusing pulses' B1 phase against an excitation about x: 90 is the CPMG
    condition, refocusing about the axis the magnetisation lies along, and 0 is not. The two agree at a
    perfect 180 and part company at every other flip, which is what the condition is about.
    """
    ev = [Pulse(90.0)]
    for _ in range(int(n_echoes)):
        ev += [Winding(+1, TE / 2), Pulse(float(beta_deg), float(refocus_phase_deg)),
               Winding(+1, TE / 2, readout=True)]
    return Schedule(tuple(ev))


def ste_schedule(alpha1_deg=90.0, alpha2_deg=90.0, alpha3_deg=90.0, delta=1.0, TM=1.0, spoil=0):
    """A three-pulse stimulated echo: excite, dephase, store, mix, recall, rephase, read.

    Both gradient lobes wind the SAME way, as a stimulated echo's lobes physically do; there is no 180 to
    fold, and what rephases the stored magnetisation is the recall pulse turning it into the conjugate
    coherence, which then winds back down to order zero. That is why the echoing pathway reads
    ``F+ / Z / F-`` and its ``eps_P`` is ``+1 / 0 / -1``: the sign the replay needs lives on the pathway,
    not on the lobe.

    ``spoil`` is the crusher's winding over the mixing time, in whole coherence orders. Zero leaves the
    competing transverse branches in place and they contribute to the echo. A crusher removes them only if
    the rest of the sequence cannot wind them back to order zero, so its size is not free: on this schedule
    a winding of 2 is rewound by the two lobes and returns an unwanted pathway to the readout, which is the
    ordinary reason a real crusher's area is chosen with care rather than merely made large.
    """
    return Schedule((Pulse(float(alpha1_deg)), Winding(+1, delta), Pulse(float(alpha2_deg)),
                     Winding(int(spoil), TM), Pulse(float(alpha3_deg)), Winding(+1, delta, readout=True)))


def ste_amplitude(alpha1_deg=90.0, alpha2_deg=90.0, alpha3_deg=90.0):
    """The stimulated echo's amplitude: ``0.5 sin(a1) sin(a2) sin(a3)``, as the pathway product gives it.

    The one half of a stimulated echo is this number at three 90 degree pulses. It is the magnitude of the
    stored pathway's rotation product, not a factor applied by hand.
    """
    stored = [p for p in enumerate_pathways(ste_schedule(alpha1_deg, alpha2_deg, alpha3_deg, spoil=4),
                                            threshold=1e-12)
              if any(st == "Z" for st, _ in p.intervals)]
    return float(abs(stored[0].eta)) if stored else 0.0


def pathway_weight(sequence):
    """The amplitude of the coherence pathway ``sequence``'s readout is, derived from its RF schedule.

    Every case here is ENUMERATED rather than written down, and the cases are exactly those a single number
    can express:

    * no refocusing and no store -- a gradient echo, the magnetisation the excitation ``alpha`` tips,
      ``sin(alpha)``, 1 at a 90;
    * one refocused echo at flip ``beta`` -- ``sin(alpha) sin^2(beta/2)``, one pathway, and 1 at a 90 and a
      perfect 180;
    * a store and a recall -- the stimulated echo, :func:`ste_amplitude`, which a real sequence isolates by
      crushing the rest;
    * a train of perfect 180s -- every echo is the same single pathway, ``sin(alpha)``.

    The excitation is the pulse labelled ``excite``, else the schedule's first pulse, at the flip it carries: a
    transmit scale that acts on every pulse of a sequence scales the excitation too, and its ``sin`` is part of
    the amplitude (dmipy-sim#391).

    A train whose refocusing pulses are NOT 180 is **refused**. Its echoes differ from one another (a six-echo
    train at 120 degrees runs 0.75, 0.94, 0.84, 0.86, 0.88, 0.86) and each is a sum over several pathways, so
    no single amplitude describes the readout: that needs the pathway sum, which is dmipy-sim#307, and on the
    vector route it also needs the crusher the replay ignores, which is #305. Returning 1 there would be a
    confident wrong answer of up to 25 per cent on the first echo alone.

    Duck-typed on ``stimulated_echo``, ``rf`` and ``readout``, so it reads a
    :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence` without importing one.
    """
    from .rf import role_of
    rf = getattr(sequence, "rf", None) or ()
    if getattr(sequence, "stimulated_echo", False):
        flips = {}
        for e in rf:
            role = role_of(e)
            if role in ("excite", "store", "recall") and role not in flips:
                flips[role] = float(e.flip_deg)
        if set(flips) != {"excite", "store", "recall"}:
            raise ValueError("a stimulated echo needs an excite, a store and a recall in its RF schedule; "
                             f"this one labels {sorted(flips)}")
        return ste_amplitude(flips["excite"], flips["store"], flips["recall"])

    excite = [float(e.flip_deg) for e in rf if role_of(e) == "excite"] or [float(e.flip_deg) for e in rf if float(e.flip_deg) != 0.0]
    alpha = excite[0] if excite else 90.0
    tipped = float(abs(np.sin(np.radians(alpha))))            # what the excitation puts in the plane
    refocus = [float(e.flip_deg) for e in rf if role_of(e) == "refocus"]
    imperfect = [b for b in refocus if abs(b - 180.0) > 1e-6]
    if not imperfect:
        return tipped                   # no refocusing at all, or every pulse a perfect 180: one pathway, whole
    n_readout = len(tuple(getattr(sequence, "readout", ()) or ()))
    if n_readout > 1 or len(refocus) > 1:
        raise ValueError(
            f"this schedule refocuses {len(refocus)} time(s) at {sorted(set(imperfect))} degrees and reads "
            f"{n_readout} echo(es): its readout is a SUM over several coherence pathways, of different "
            "amplitudes at each echo, so no single amplitude describes it. The pathway sum is dmipy-sim#307 "
            "(enumerate_pathways gives the terms); on the vector-Bloch route it also needs the crusher that "
            "route ignores, dmipy-sim#305. Use a perfect 180, or a single refocused echo.")
    sch = Schedule((Pulse(alpha), Winding(+1, 1.0), Pulse(imperfect[0], 90.0), Winding(+1, 1.0, readout=True)))
    return float(abs(sum(p.eta for p in enumerate_pathways(sch, threshold=1e-12))))
