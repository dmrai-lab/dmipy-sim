"""The Pulseq writer: :func:`~dmipy_sim.sequences.pulseq.to_pulseq` writes a
:class:`~dmipy_replay.acquisition.scanner_sequence.ScannerSequence` as a ``.seq`` file for a scanner's system
(:func:`~dmipy_sim.sequences.pulseq.make_system`, :data:`~dmipy_sim.sequences.pulseq.PULSEQ_SYSTEMS`).

The builders and the Pulseq reader are dmipy-replay's (``dmipy_replay.sequences``, ``dmipy_replay.sequences.pulseq``)::

    seq = dmipy_replay.pgse(gdirs, delta, Delta, bvalues=bvalues, slew_rate=200.)
    to_pulseq(seq, filename="pgse.seq")
"""
from .pulseq import to_pulseq, make_system, PULSEQ_SYSTEMS

__all__ = ['to_pulseq', 'make_system', 'PULSEQ_SYSTEMS']


def __getattr__(name):
    from .._moved import refuse
    raise refuse(__name__, name)
