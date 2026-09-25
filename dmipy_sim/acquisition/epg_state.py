"""EPG as a state vector rather than a list of pathways: what a long train needs.

:func:`~dmipy_sim.acquisition.epg.enumerate_pathways` walks every branch, which is exact and exponential --
49,714 pathways reach the last readout of a twelve-echo train at 120 degrees, and pruning does not rescue
it (a threshold that keeps a hundredth of each amplitude keeps three per cent of their sum). A real train
is seventy echoes.

The configuration-state formulation carries the same information in a vector over coherence orders, and
costs one matrix multiply per pulse and one shift per interval. What it gives up is the pathway labels --
and those are exactly what a replay needs, because a pathway's label is what says how much gradient phase
it accumulated.

:func:`split_by_gate` is the reconciliation. A pathway's microscopic phase comes only from intervals where
the gradient is ON, so pathways that differ only in what they did while it was OFF share a phase. For a
diffusion-prepared train -- a preparation that encodes, then a train that reads -- that collapses tens of
thousands of pathways onto a handful of distinct gates, and the explosion stays entirely inside the
pose-independent weights where a state vector handles it in quadratic time.
"""
from __future__ import annotations

import numpy as np

__all__ = ["EPGState", "rf_operator", "split_by_gate", "train_weights"]


def rf_operator(flip_deg, phase_deg=0.0):
    """The 3x3 mixing a pulse applies to ``(F+, F-, Z)`` at every coherence order (Weigel 2015, eq. 2.4)."""
    a = np.deg2rad(float(flip_deg))
    p = np.deg2rad(float(phase_deg))
    ca2, sa2 = np.cos(a / 2) ** 2, np.sin(a / 2) ** 2
    sa, ca = np.sin(a), np.cos(a)
    e1, e2 = np.exp(1j * p), np.exp(2j * p)
    return np.array([
        [ca2,            e2 * sa2,         -1j * e1 * sa],
        [np.conj(e2) * sa2, ca2,            1j * np.conj(e1) * sa],
        [-0.5j * np.conj(e1) * sa, 0.5j * e1 * sa, ca],
    ], dtype=np.complex128)


class EPGState:
    """The configuration states of a spin ensemble: ``F`` over coherence orders ``-N..N`` and ``Z`` over
    ``0..N``.

    Stored over the FULL integer range rather than folded, because folding needs a reality condition applied
    at exactly the right moment and getting that wrong is silent -- it reproduces the first echo of a train
    and then quietly drifts. ``F[0]`` is the observable transverse magnetisation.
    """

    def __init__(self, n_orders):
        self.n = int(n_orders)
        self.F = np.zeros(2 * self.n + 1, np.complex128)      # index n + k holds order k
        self.Z = np.zeros(self.n + 1, np.complex128)

    @classmethod
    def equilibrium(cls, n_orders):
        st = cls(n_orders)
        st.Z[0] = 1.0
        return st

    def copy(self):
        out = EPGState(self.n)
        out.F, out.Z = self.F.copy(), self.Z.copy()
        return out

    def pulse(self, flip_deg, phase_deg=0.0):
        """Mix ``(F_k, F_{-k}^*, Z_k)`` at every order -- the one place the three states talk to each other."""
        T = rf_operator(flip_deg, phase_deg)
        n = self.n
        Fp = self.F[n:]                                  # orders 0..N
        Fm = np.conj(self.F[n::-1])                      # conj(F_{-k}) for k = 0..N
        Z = self.Z
        new_p = T[0, 0] * Fp + T[0, 1] * Fm + T[0, 2] * Z
        new_m = T[1, 0] * Fp + T[1, 1] * Fm + T[1, 2] * Z
        new_z = T[2, 0] * Fp + T[2, 1] * Fm + T[2, 2] * Z
        self.F[n:] = new_p
        self.F[n::-1] = np.conj(new_m)
        self.Z = new_z
        return self

    def shift(self, dk=1):
        """Wind every transverse coherence by ``dk`` orders: what an interval's gradient does at voxel scale.

        ``Z`` does not move, which is the whole reason a stored pathway comes back where it left."""
        k = int(dk)
        if k:
            self.F = np.roll(self.F, k)
            if k > 0:
                self.F[:k] = 0.0
            else:
                self.F[k:] = 0.0
        return self

    def off_resonance(self, dt, dw):
        """Advance EVERY transverse state by ``dw * dt`` radians, and the stored ones by nothing.

        Off-resonance is gated like the gradient in one respect only -- magnetisation parked along z
        accumulates none of it -- and NOT in the other. The dephasing index does not enter. This array holds
        the F+ coefficients over the whole integer range, and they are all ordinary transverse
        magnetisation, so a uniform offset advances them all by the same angle whatever their winding.

        What reverses the accumulated phase is the RF, not the index: a refocusing pulse conjugates the
        state, which negates everything accrued so far. That is why a spin echo refocuses an offset and a
        gradient echo does not, and why the answer is the same whether a bipolar pair was wound +/- or -/+.
        Keying the sign to the index instead gets the second of those backwards -- it reports zero for a
        gradient echo wound negative-first, where the truth is the same phase as positive-first.
        """
        if dw:
            self.F *= np.exp(1j * float(dw) * float(dt))
        return self

    def relax(self, dt, T1=None, T2=None):
        """Per-interval relaxation; longitudinal order zero regrows toward 1."""
        if T2:
            self.F *= np.exp(-float(dt) / float(T2))
        if T1:
            e1 = np.exp(-float(dt) / float(T1))
            self.Z *= e1
            self.Z[0] += 1.0 - e1
        return self

    @property
    def signal(self):
        """The observable transverse magnetisation: the order-zero ``F``."""
        return complex(self.F[self.n])


def split_by_gate(events, n_orders, gradient_on):
    """Propagate through ``events``, splitting the state whenever the gradient is ON.

    ``events`` is a list of ``("pulse", flip, phase)`` and ``("interval", dk, index)``. ``gradient_on`` says
    whether interval ``index`` carries a microscopic gradient. Where it does, the state is split into its
    three kinds -- because they accumulate ``+phase``, ``-phase`` and nothing -- and the kind is appended to
    that branch's gate. Where it does not, nothing splits: that interval cannot tell pathways apart.

    Returns ``{gate: EPGState}``. For a diffusion-prepared train the gradient is on only in the preparation,
    so this terminates with a handful of branches however long the train that follows is.
    """
    branches = {(): EPGState.equilibrium(n_orders)}
    for ev in events:
        if ev[0] == "pulse":
            for st in branches.values():
                st.pulse(ev[1], ev[2])
            continue
        dk, idx = ev[1], ev[2]
        if not gradient_on(idx):
            for st in branches.values():
                st.shift(dk)
            continue
        out = {}
        n = n_orders
        for gate, st in branches.items():
            for kind, sl in (("F+", slice(n, None)), ("F-", slice(None, n)), ("Z", None)):
                part = EPGState(n)
                if kind == "Z":
                    part.Z = st.Z.copy()
                else:
                    part.F[sl] = st.F[sl]
                if not np.any(np.abs(part.F)) and not np.any(np.abs(part.Z)):
                    continue
                part.shift(dk)
                key = gate + (kind,)
                if key in out:
                    out[key].F += part.F
                    out[key].Z += part.Z
                else:
                    out[key] = part
        branches = out
    return branches


def train_weights(prep_events, train_events, n_orders, gradient_on, readouts, dw=0.0, durations=None):
    """``{gate: [amplitude at each readout]}``: what each microscopic gate contributes to each echo.

    The preparation is split by gate; each branch is then propagated through the train as a state vector,
    which is where the pathway explosion would otherwise live and where it costs nothing.
    """
    branches = split_by_gate(prep_events, n_orders, gradient_on)
    out = {}
    for gate, st in branches.items():
        s = st.copy()
        amps, ro = [], set(readouts)
        for i, ev in enumerate(train_events):
            if ev[0] == "pulse":
                s.pulse(ev[1], ev[2])
            else:
                s.shift(ev[1])
                if dw and durations is not None:
                    s.off_resonance(durations[ev[2]] if ev[2] < len(durations) else 0.0, dw)
            if i in ro:
                amps.append(s.signal)
        out[gate] = np.asarray(amps, np.complex128)
    return out
