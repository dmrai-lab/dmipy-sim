"""An RF train's response as a sum over coherence pathways, so that a pose expansion can carry it.

The vector-Bloch route propagates the magnetisation, which is why it needs one rotation per slot and
refuses an orientation distribution (dmipy-sim#338): a propagation cannot be expanded over poses the way a
phase sum can. But it does not have to be the thing that is expanded.

What a train delivers at a readout is a sum over coherence pathways,

    S(pose, kappa) = sum_p  eta_p(kappa) * E_p(pose)

and the two factors are independent. ``eta_p`` is the pathway's RF amplitude -- a product of the pulses'
own transition coefficients, so a function of the flip angles and therefore of the transmit scale ALONE.
``E_p`` is the ensemble phase of the same walk under that pathway's own gradient gate: ``+1`` while the
pathway is transverse, ``-1`` where a pulse has conjugated it, ``0`` while it is stored along z. That is an
ordinary phase sum, and the closed-form pose expansion already builds it.

Three things follow, and they are why this exists.

A pose expansion becomes possible for an RF-aware replay, so a train reaches an ODF phantom -- a brain --
rather than only a frames-mode one.

A transmit map costs a RE-WEIGHTING rather than a re-expansion: the expansions do not depend on kappa at
all, so a phantom whose every voxel has its own transmit scale builds one set and weights it per voxel.

And the result is less noisy than the route it replaces. The Bloch route models the voxel-scale crusher by
giving every walker a random phase, which carries its own Monte-Carlo scatter; a pathway gate is
deterministic, so this is that route's exact ensemble limit rather than a sample of it.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

__all__ = ["TrainResponse", "gate_sign", "train_response", "sequence_events"]

#: A pathway's coherence sign while it is in each state: transverse, conjugated, or stored.
_SIGN = {"F+": +1.0, "F-": -1.0, "Z": 0.0}


def gate_sign(gate, edges, n_t, dt):
    """The coherence sign of one microscopic gate at every sample of the grid, ``(n_t,)``.

    ``gate`` is the tuple of states (``"F+"``, ``"F-"``, ``"Z"``) a pathway was in over the consecutive
    intervals ``edges`` cut the timeline into. This is the whole of what distinguishes one gate's phase from
    another's: the SAME walk, read with a different sign pattern. Zero where the pathway is stored along z,
    because stored magnetisation accumulates no gradient phase -- which is exactly why a stimulated echo
    carries the diffusion weighting it does and not the weighting of the interval it slept through.
    """
    t = np.arange(int(n_t)) * float(dt)
    sign = np.zeros(int(n_t))
    for k, kind in enumerate(gate):
        if k + 1 >= len(edges):
            break
        sign[(t >= edges[k]) & (t <= edges[k + 1] + 1e-12)] = _SIGN[kind]
    return sign


class TrainResponse:
    """A pack's response to an RF train over every pose AND every transmit scale.

    Holds one pose expansion per microscopic GATE -- the sign pattern a coherence pathway wore while the
    gradient was on. :meth:`at` weights them for a transmit scale and returns one
    :class:`~dmipy_sim.replay.PoseResponse` per readout, which composes with an orientation distribution
    exactly as any other does.

    The expansions are the expensive part and they do not depend on the transmit scale at all, so a second
    scale costs a state propagation and a handful of multiply-adds.
    """

    def __init__(self, parts, waveform, pulse_times, TE, n_orders=None, readouts=(), tau=None, durations=()):
        self.parts = dict(parts)                  # {gate: PoseResponse}
        self.waveform, self.pulse_times, self.TE = waveform, tuple(pulse_times), float(TE)
        self.n_orders = n_orders
        self.readouts = tuple(readouts)
        #: each gate's SIGNED TRANSVERSE time over the preparation: what a uniform field offset
        #: multiplies. Zero for the refocused pathway, which is why a spin echo refocuses an
        #: offset and a stimulated one does not.
        self.tau = dict(tau or {})
        self.durations = tuple(durations)

    @property
    def n_gates(self):
        return len(self.parts)

    def weights(self, b1_scale=1.0, dw=0.0):
        """``{gate: amplitude per readout}`` at this transmit scale and field offset.

        ``dw`` (rad/s) is a UNIFORM off-resonance carried through the train. It is not a phase applied at the
        end: off-resonance is gated like the gradient, so a pathway that spent an interval along z accrues
        none of it, and the train's own pulses then mix what is left. A drifting magnet is exactly this --
        uniform in space, so one propagation serves a whole image.
        """
        from ..acquisition.epg_state import train_weights
        prep, train, on, ro, _edges = sequence_events(self.waveform, b1_scale)
        return train_weights(prep, train, int(self.n_orders), lambda i: on[i] if i < len(on) else False, ro,
                             dw=dw, durations=self.durations)

    def at(self, b1_scale=1.0, echo=-1, dw=0.0):
        """The pose expansion of the whole train at this transmit scale and field offset, at ``echo``."""
        from .replay import PoseResponse
        w = self.weights(b1_scale, dw=dw)
        first = max(self.parts.values(), key=lambda r: r.lmax)      # the widest band carries the sum
        coeffs = np.zeros_like(np.asarray(first.coeffs, np.complex128))
        misfit = np.zeros_like(np.asarray(first.misfit, float))
        for gate, resp in self.parts.items():
            amp = w.get(gate)
            if amp is None or not len(amp):
                continue
            e = complex(amp[int(echo)])
            if dw:                                    # the preparation's own share, gated per pathway
                e = e * np.exp(1j * float(dw) * self.tau.get(gate, 0.0))
            if e == 0:
                continue
            c = np.asarray(resp.coeffs, np.complex128)
            if c.shape[1] < coeffs.shape[1]:          # a narrower gate: zero above its own band
                c = np.pad(c, ((0, 0), (0, coeffs.shape[1] - c.shape[1])))
            coeffs = coeffs + e * c
            misfit = misfit + abs(e) * np.asarray(resp.misfit, float)
        out = PoseResponse(coeffs, first.lmax, first.nmax, misfit, first.floor,
                           first.phase_amplitude, first.n_samples)
        out.route = first.route
        return out

    def __repr__(self):
        return (f"TrainResponse(gates={self.n_gates}, echoes={len(self.readouts)}, "
                f"lmax={next(iter(self.parts.values())).lmax})")


def sequence_events(waveform, b1_scale=1.0):
    """``(prep_events, train_events, gradient_on, readout_positions, edges)`` for a waveform.

    The timeline is cut at every pulse AND every readout, because an echo forms part way through the gap
    between two pulses and a state has to be read there rather than at the next pulse. ``gradient_on`` says
    which of the resulting intervals carry a microscopic gradient -- the only ones that can tell coherence
    pathways apart. The preparation is everything up to and including the last of those; the rest is the
    train, however long it runs.
    """
    rf = [e for e in (waveform.rf or ()) if float(e.flip_deg) != 0.0]
    if not rf:
        raise ValueError("a pathway sum describes an RF train, and this waveform carries no pulses")
    dt, n_t = float(waveform.dt), int(waveform.n_t)
    TE = (n_t - 1) * dt
    pulses = {round(float(e.t_s), 12): e for e in rf}
    reads = {round(float(i) * dt, 12) for i in (waveform.readout or (n_t - 1,))}
    marks = sorted(set(pulses) | reads | {round(TE, 12)})
    edges = [m for m in marks]
    if edges[0] > 0.0:
        edges = [0.0] + edges

    G = np.abs(np.asarray(waveform.G, np.float64)).sum(axis=(0, 2))
    t = np.arange(n_t) * dt
    on = []
    for k in range(len(edges) - 1):
        m = (t >= edges[k] - 1e-12) & (t <= edges[k + 1] + 1e-12)
        on.append(bool(np.any(G[m] > 0)))
    last_grad = max([i for i, v in enumerate(on) if v], default=-1)

    events, read_at = [], []
    for k in range(len(edges)):
        e = pulses.get(edges[k])
        if e is not None:
            events.append(("pulse", float(e.flip_deg) * float(b1_scale),
                           float(getattr(e, "axis_deg", 0.0) or 0.0)))
        if k < len(edges) - 1:
            events.append(("interval", +1, k))
            if round(edges[k + 1], 12) in reads:
                read_at.append(len(events) - 1)
    cut = 0
    for i, ev in enumerate(events):
        if ev[0] == "interval" and ev[2] == last_grad:
            cut = i + 1
            break
    prep, train = events[:cut], events[cut:]
    ro = [i - cut for i in read_at if i >= cut]
    return prep, train, on, ro, edges


def train_response(pack, waveform, *, keep=(None, 0), b1_reference=1.0, n_orders=None, **kw):
    """Build a :class:`TrainResponse` by the STATE route: one pose expansion per microscopic gate.

    The pathway explosion lives entirely in the weights, which a configuration-state propagation handles in
    quadratic time -- a seventy-echo train costs ten milliseconds and would be 10^68 pathways enumerated.
    What is expanded over poses is one walk per distinct GATE, and for a diffusion-prepared train there are
    a handful of those however long the train is.
    """
    from ..acquisition.epg_state import split_by_gate
    prep, train, on, ro, edges = sequence_events(waveform, b1_reference)
    n_orders = int(n_orders if n_orders is not None else len(train) + 4)
    gates = split_by_gate(prep, n_orders, lambda i: on[i] if i < len(on) else False)
    G = np.asarray(waveform.G, np.float64)
    n_t, dt = int(waveform.n_t), float(waveform.dt)
    signs = {gate: gate_sign(gate, edges, n_t, dt) for gate in gates}
    gated = {gate: replace(waveform, G=(G * s[None, :, None]).astype(np.float32), rf=None,
                           family="waveform", crusher=None, readout=None)
             for gate, s in signs.items()}

    # The gates do not reach the same band: one that spends an interval STORED accumulates no phase there
    # and so is smoother. They are summed at the widest of them, which is exact rather than a choice --
    # `so3_index` lays coefficients out with `l` ascending, so a narrower gate's are a prefix of a wider
    # gate's and everything above its own band is genuinely zero.
    parts = {g: pack.pose_response(w, keep=keep, **kw) for g, w in gated.items()}
    durations = [edges[k + 1] - edges[k] for k in range(len(edges) - 1)]
    # each gate's SIGNED transverse time over the preparation, from the interval durations themselves rather
    # than from a sample count, so that a refocused pathway's tau is exactly zero on any grid
    tau = {gate: float(sum(_SIGN[kind] * durations[k] for k, kind in enumerate(gate) if k < len(durations)))
           for gate in gates}
    return TrainResponse(parts, waveform, edges[:-1], edges[-1], n_orders=n_orders, readouts=ro,
                         tau=tau, durations=durations)
