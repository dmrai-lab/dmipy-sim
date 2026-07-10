"""Physical acquisition sequences (sim-owned forward-signal definition).

``Sequence`` carries the real ``G(t)`` + per-measurement encoding (gradient
directions, b-values, q-values, gradient strengths, timing) and the directly
derived ``btensor`` / ``instantaneous`` view.  The module-level constructors are
the canonical API::

    seq = dmipy_sim.sequences.pgse(bvalues, gdirs, delta, Delta, slew_rate=200.)
    seq.G, seq.dt, seq.bvalues, seq.btensor()      # physical / forward
    seq.instantaneous()                            # square / infinite-slew limit

dmipy-fit's AcquisitionScheme consumes a Sequence (fit eats sim's real
constructors) and adds the analytical shell / SH / rotational-harmonics layer.
"""
from .sequence import Sequence
from .pulseq import from_pulseq, to_pulseq, make_system, PULSEQ_SYSTEMS

pgse = Sequence.from_pgse
cpmg = Sequence.from_cpmg
ogse = Sequence.from_ogse
ste = Sequence.from_btensor_ste
pte = Sequence.from_btensor_pte
from_waveform = Sequence.from_waveform

__all__ = [
    'Sequence', 'pgse', 'cpmg', 'ogse', 'ste', 'pte',
    'from_waveform',
    'from_pulseq', 'to_pulseq', 'make_system', 'PULSEQ_SYSTEMS',
]
