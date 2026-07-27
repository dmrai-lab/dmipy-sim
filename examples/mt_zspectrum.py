#!/usr/bin/env python3
"""Emergent MT Z-spectrum vs the analytic two-pool oracle — the turnkey sweep.

Runs :func:`dmipy_sim.emergent_z_spectrum` (a real forward vector-Bloch Monte-Carlo walk:
spins bind to the wall, the short-T2 bound pool saturates off-resonance) over a set of
saturation offsets on ONE substrate, and overlays the analytic two-pool Bloch--McConnell
Z-spectrum (:func:`dmipy_sim.mt.mt_z_spectrum`) it is validated against. No super-Lorentzian
lineshape is assumed on either side — the MT dip emerges from real short-T2b spins.

Heavy Monte-Carlo -> GPU-recommended.  Reduce ``N_WALKERS`` / ``OFFSETS`` for a quick look,
or override on the CLI:  ``python examples/mt_zspectrum.py --walkers 2000``.

    python examples/mt_zspectrum.py [--walkers N] [--offsets a,b,c] [--no-plot]
"""
import argparse

import numpy as np

from dmipy_sim import Sphere, emergent_z_spectrum
from dmipy_sim import mt

# well-mixed sphere (R^2/D mixing time << 1/k_f), broad bound pool (T2b ~ 10 us)
R, D = 2e-6, 2e-9
K_F, K_R = 40.0, 80.0                         # forward / backward exchange (s^-1)
T2A, T1A, T2B, T1B = 80e-3, 1.0, 1e-5, 1.0
W1_HZ, T_SAT, DT = 200.0, 0.025, 2e-5
S_OVER_V = 3.0 / R                            # sphere
KAPPA_MT = mt.kappa_MT_from_forward_rate(K_F, S_OVER_V)   # k_f = kappa*(S/V)
DWELL = 1.0 / K_R
OFFSETS = np.array([0.0, 500.0, 1000.0, 3000.0, 8000.0])


def oracle_total(offsets):
    """Two-pool oracle TOTAL longitudinal (free + bound), the quantity the walker-mean Mz
    reports; normalised to equilibrium."""
    kw = dict(w1_hz=W1_HZ, t_sat=T_SAT, T1a=T1A, T2a=T2A, T1b=T1B, T2b=T2B, k_f=K_F, k_r=K_R)
    za = np.atleast_1d(mt.mt_z_spectrum(offsets, read_pool="a", **kw))
    zb = np.atleast_1d(mt.mt_z_spectrum(offsets, read_pool="b", **kw))
    m0b = K_F / K_R
    return (za + m0b * zb) / (1.0 + m0b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walkers", type=int, default=4000)
    ap.add_argument("--offsets", type=str, default=None, help="comma-separated Hz")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    offsets = (np.array([float(x) for x in args.offsets.split(",")])
               if args.offsets else OFFSETS)

    print(f"emergent MT Z-spectrum: sphere R={R*1e6:.1f} um, k_f={K_F:.0f}/s, k_r={K_R:.0f}/s, "
          f"f_b={mt.bound_fraction(KAPPA_MT, DWELL, S_OVER_V):.3f}")
    mc = emergent_z_spectrum(offsets, Sphere(radius=R), n_walkers=args.walkers,
                             diffusivity=D, w1_hz=W1_HZ, t_sat=T_SAT, dt=DT,
                             T2=T2A, T1=T1A, kappa_MT=KAPPA_MT, dwell_time=DWELL,
                             T2_bound=T2B, T1_bound=T1B, equilibrate_binding="auto", seed=3)
    an = oracle_total(offsets)
    rel = np.abs(mc - an) / np.maximum(np.abs(an), 0.05)

    print(f"\n{'offset (Hz)':>12} {'emergent Mz':>12} {'oracle Mz':>10} {'rel.diff':>9}")
    for o, m, a, r in zip(offsets, mc, an, rel):
        print(f"{o:12.0f} {m:12.3f} {a:10.3f} {r:9.1%}")
    print(f"\nmax relative difference: {rel.max():.1%}  (emergent vs analytic oracle)")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.plot(offsets, an, "-", color="#7a8499", lw=2, label="analytic two-pool oracle")
            ax.plot(offsets, mc, "o", color="#4af0c4", ms=8, label="emergent Monte-Carlo")
            ax.set_xlabel("saturation offset (Hz)"); ax.set_ylabel(r"free-pool $M_z / M_0$")
            ax.set_title("Emergent MT Z-spectrum vs analytic oracle")
            ax.legend(frameon=False)
            fig.tight_layout()
            out = "mt_zspectrum.png"
            fig.savefig(out, dpi=120)
            print(f"wrote {out}")
        except ImportError:
            print("(matplotlib not available — skipped plot)")


if __name__ == "__main__":
    main()
