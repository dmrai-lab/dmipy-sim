"""The sequence builders and the Pulseq bridge.

Every builder returns a :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence` -- the physical
gradient, the RF schedule, the readout, the per-measurement encoding an analytical layer reads::

    seq = dmipy_sim.sequences.pgse(gdirs, delta, Delta, bvalues=bvalues, slew_rate=200.)
    seq.G, seq.G_eff, seq.dt, seq.encoding.bvalues, seq.btensor()
    dmipy_sim.sequences.instantaneous(seq)          # the square / infinite-slew limit

dmipy-fit's AcquisitionScheme consumes one (fit eats sim's real builders) and adds the analytical shell / SH /
rotational-harmonics layer.
"""
from .builders import (pgse, pgste, gre, cpmg, ogse, ste, pte, from_waveform, from_btensor_waveform,
                       from_pgste_waveform, instantaneous, to_gradient_array)
from .pulseq import from_pulseq, to_pulseq, make_system, PULSEQ_SYSTEMS

__all__ = [
    'pgse', 'pgste', 'gre', 'cpmg', 'ogse', 'ste', 'pte',
    'from_waveform', 'from_btensor_waveform', 'from_pgste_waveform', 'instantaneous', 'to_gradient_array',
    'from_pulseq', 'to_pulseq', 'make_system', 'PULSEQ_SYSTEMS',
]
