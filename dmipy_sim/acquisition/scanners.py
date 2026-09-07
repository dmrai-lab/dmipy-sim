"""Scanner classes (peak gradient, slew) and the save-interval rule a persistent walk derives its grid from.

The classes are those of the replay band-limit certificate (dmipy-sim #143): the same six that size K per
scanner. ``save_interval`` is the rule for ``dt_save``: between two saves the walker moves and the replay's phase
integral treats it as sitting at the sample; for a Brownian path the phase error has variance
``(2/3) gamma^2 D dt^2 int |G|^2 dt`` and the relative signal error is half of it. The adversarial deliverable
waveform is ``G = Gmax`` for the whole echo time, so the scanner enters only through ``Gmax`` (as it enters the
K bound only through slew). Held to a fraction ``f`` of the Monte-Carlo floor ``1 / sqrt(N_w)``:

    dt <= sqrt( 3 f / ( sqrt(N_w) gamma^2 D Gmax^2 T ) )

Geometry sets nothing here: walls only shrink displacements, occupancy and wall contact are accumulated per save
by the walk itself. The field tier samples the field basis at the saves; its error is second order in ``dt`` and
was measured at 3e-4 in |E| at dt = 50 us in the harshest catalogued case (7 T, chi 1.06e-6, TE 40 ms), so a
walk with a field source is capped at ``FIELD_DT_CAP`` (#143). The codec needs only ``n_t >= 2K + 2``.
"""
import numpy as np

from ..constants import GAMMA

#: (peak gradient T/m, peak slew T/m/s) per scanner class, as in the band-limit certificate.
SCANNERS = {
    "prisma": (0.08, 200.0),
    "magnus": (0.20, 500.0),
    "connectom": (0.30, 600.0),
    "bruker_bga_s": (0.75, 6000.0),
    "micro_insert": (1.50, 10000.0),
    "extreme_insert": (3.00, 20000.0),
}

#: The save interval at which the field tier's save-grid error is < 0.2 of a 200k-walker floor in the harshest
#: catalogued case (measured, #143).
FIELD_DT_CAP = 5e-5


def scanner_limits(scanner):
    """``(G_max, slew_max)`` of a scanner class name or of an explicit ``(G_max, slew_max)`` pair."""
    if isinstance(scanner, str):
        try:
            return SCANNERS[scanner.lower()]
        except KeyError:
            raise ValueError(f"unknown scanner class {scanner!r}; known: {sorted(SCANNERS)} (or give (G_max, slew_max))")
    g, s = scanner
    return float(g), float(s)


def save_interval(T_max, n_walkers, scanner="connectom", *, D=2e-9, floor_fraction=0.1, field=False):
    """The save interval ``dt`` a persistent walk of ``T_max`` with ``n_walkers`` needs so that the in-step path
    integration error of any waveform ``scanner`` can deliver stays below ``floor_fraction`` of the Monte-Carlo
    floor; capped at :data:`FIELD_DT_CAP` when the walk carries a field source. ``D`` is the fastest pool's
    diffusivity (free water is the worst case)."""
    G_max, _ = scanner_limits(scanner)
    T_max, n_walkers, D = float(T_max), float(n_walkers), float(D)
    if T_max <= 0 or n_walkers <= 0 or D <= 0 or G_max <= 0 or floor_fraction <= 0:
        raise ValueError("T_max, n_walkers, D, G_max and floor_fraction must be positive")
    dt = np.sqrt(3.0 * floor_fraction / (np.sqrt(n_walkers) * GAMMA ** 2 * D * G_max ** 2 * T_max))
    if field:
        dt = min(dt, FIELD_DT_CAP)
    # a whole number of saves over T_max, never coarser than the rule
    n_t = int(np.ceil(T_max / dt)) + 1
    return T_max / (n_t - 1)
