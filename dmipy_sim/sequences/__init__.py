"""The sequence builders and the Pulseq bridge.

Every builder returns a :class:`~dmipy_replay.acquisition.scanner_sequence.ScannerSequence` -- the physical
gradient, the RF schedule, the readout, the per-measurement encoding an analytical layer reads::

    seq = dmipy_sim.sequences.pgse(gdirs, delta, Delta, bvalues=bvalues, slew_rate=200.)
    seq.G, seq.G_eff, seq.dt, seq.encoding.bvalues, seq.btensor()
    dmipy_sim.sequences.instantaneous(seq)          # the square / infinite-slew limit

dmipy-fit's AcquisitionScheme consumes one (fit eats sim's real builders) and adds the analytical shell / SH /
rotational-harmonics layer.
"""
from .pulseq import to_pulseq, make_system, PULSEQ_SYSTEMS

__all__ = [ 'to_pulseq', 'make_system', 'PULSEQ_SYSTEMS',
]


def __getattr__(name):
    from .._moved import refuse
    raise refuse(__name__, name)
