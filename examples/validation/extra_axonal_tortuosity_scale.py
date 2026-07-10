"""Extra-axonal tortuosity: same volume fraction, different emergent D_perp by scale.

Positive control for time-dependent extra-axonal diffusion (Novikov--Fieremans structural
disorder; Burcaw-Fieremans-Novikov 2015). At a FIXED cylinder volume fraction (same v_ic),
the DC tortuosity limit D_inf/D depends only on the packing (scale-invariant), but the
correlation time t_c ~ R^2/(2 D) scales with cylinder size. So at a fixed diffusion time
Delta, the emergent perpendicular diffusivity D_perp depends on the pack SIZE:

  * small pack (sub-micron axons):  t_c << Delta  ->  D_perp on the DC plateau (~0.5 D),
    time-INDEPENDENT (Burcaw A ~ 0).  <-- the white-matter-at-clinical-Delta regime.
  * large pack (~10 um):            t_c ~ Delta   ->  D_perp caught mid-crossover, elevated
    toward D_free and Delta-DEPENDENT (A > 0).

This is why a fixed tortuosity (lambda_perp = D*(1-f), a G2Zeppelin) suffices for sub-micron
WM at clinical Delta, while a temporal Zeppelin (G3TemporalZeppelin) is only needed when
t_c ~ Delta (large calibre, or mesoscopic disorder: beading / undulation / clustering).

Measured on the dedicated PackedCylinders extra-axonal geometry (all walkers extra, specular
reflection off fibres, unfolded phase), low b (Gaussian/cumulant ~ trajectory D), with the
step size resolved (step << smallest cylinder) at every scale to avoid tunnelling.

Run:  python extra_axonal_tortuosity_scale.py [--quick] [--outdir DIR]
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dmipy_sim.substrate.biophysical_constants import get_value
from dmipy_sim import pack_cylinders, PackedCylinders, simulate, pgse, set_b

HERE = os.path.dirname(os.path.abspath(__file__))
D = get_value('D_intra_axonal')                          # intrinsic diffusivity (m^2/s)
AL = get_value('gamma_shape_diameter')                   # OUTER (fibre) diameter Gamma
SC = get_value('gamma_scale_diameter')
F = 0.50                                                  # fibre volume fraction (== v_ic here)


def d_perp(scale, Delta, delta=3e-3, b=0.15e9, n_walk=30000, seed=0):
    """Emergent extra-axonal perpendicular D at diffusion time Delta for a pack scaled
    by ``scale`` (radii and cell scaled together -> volume fraction unchanged)."""
    base = np.maximum(np.random.default_rng(0).gamma(AL, SC, 40), 0.4e-6) / 2.0   # base radii
    radii = base * scale
    centers, L, vf = pack_cylinders(radii, target_vf=F, seed=0)
    geom = PackedCylinders(radii, centers, L)
    step_target = float(radii.min()) / 3.0               # resolve the smallest cylinder
    n_t = int(np.ceil((delta + Delta) / (step_target ** 2 / (6.0 * D))))
    wf = set_b(pgse(delta=delta, DELTA=Delta, G_magnitude=0.05,
                    bvecs=np.array([[1., 0, 0]], np.float32), n_t=n_t),
               np.array([b], np.float32))
    S = float(np.asarray(simulate(n_walk, diffusivity=D, waveform=wf, geometry=geom,
                                  seed=seed + 1, require_gpu=False)).ravel()[0])
    Rmean = float(np.mean(radii))
    return -np.log(S) / b, Rmean ** 2 / (2.0 * D), np.mean(2 * radii)


def main(quick=False, outdir=HERE):
    Delta = 20e-3
    scales = [1.0, 4.0, 16.0] if not quick else [1.0, 16.0]
    print(f"fixed Delta={Delta*1e3:.0f}ms, f={F} (same v_ic); D_free={D*1e9:.3f} um2/ms")
    rows = []
    for s in scales:
        Dp, tc, dmean = d_perp(s, Delta, n_walk=(8000 if quick else 30000))
        rows.append((s, dmean, tc, Dp))
        print(f"  scale x{s:<4.0f} mean d_out={dmean*1e6:6.2f}um  t_c~{tc*1e3:7.2f}ms  "
              f"D_perp/D_free={Dp/D:.3f}")
    s, dm, tc, dp = np.array(rows).T
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.axhline(1.0, color='0.6', ls='--', lw=1, label=r'$D_{\rm free}$')
    ax.plot(dm * 1e6, dp / D, 'o-', color='#1f77b4', lw=2, ms=7)
    ax.set_xscale('log')
    ax.set_xlabel(r'mean outer (fibre) diameter  ($\mu$m)  [same $v_{\rm ic}$]')
    ax.set_ylabel(r'emergent $D_\perp / D_{\rm free}$  at fixed $\Delta=20$ ms')
    ax.set_title('Same volume fraction, size-dependent $D_\\perp$\n(DC plateau $\\to$ crossover)')
    ax.set_ylim(0.45, 1.02); ax.grid(alpha=0.25, which='both'); ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(outdir, 'extra_axonal_tortuosity_scale.png')
    fig.savefig(out, dpi=140, bbox_inches='tight'); print('saved', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--outdir', default=HERE)
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    main(quick=a.quick, outdir=a.outdir)
