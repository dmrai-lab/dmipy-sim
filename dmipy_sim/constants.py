"""Physical constants for dmipy-sim."""
import numpy as np

# Proton gyromagnetic ratio in rad / (s * T)
# Matches disimpy's value exactly.
GAMMA = 267.513e6

# Default gradient slew rate (T/m/s) for the physical sequence constructors.
# dmipy-sim is the forward truth -- a real scanner -- so its sequences are
# slew-limited (realizable) BY DEFAULT; ``slew_rate=np.inf`` requests the
# idealized instantaneous (square) limit used for A/B against dmipy-fit's
# analytic solutions.  200 T/m/s (= 200 mT/m/ms) is a typical clinical maximum
# (e.g. Siemens Prisma); see dmipy_sim.sequences.pulseq.PULSEQ_SYSTEMS.
DEFAULT_SLEW_RATE = 200.0


def resolve_slew(slew_rate):
    """Validate the ``slew_rate`` knob, returning ``(value, is_instantaneous)``.

    ``slew_rate`` must be a POSITIVE rate in T/m/s (realistic, slew-limited) or
    ``np.inf`` for the idealized instantaneous (square) limit.  ``None`` is NOT
    accepted: a "no slew limit" sentinel reading as square is error-prone, so the
    square limit must be requested explicitly with ``np.inf``.
    """
    if slew_rate is None or not (slew_rate > 0):
        raise ValueError(
            "slew_rate must be a positive number (T/m/s) for a realistic "
            "slew-limited gradient, or np.inf for the idealized instantaneous "
            "(square) limit; got {!r}.".format(slew_rate))
    return slew_rate, (not np.isfinite(slew_rate))
