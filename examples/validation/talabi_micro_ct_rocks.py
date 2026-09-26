"""Talabi's random-walk NMR on the Imperial 2007 micro-CT rocks, on the same voxels.

Talabi (2008), *Pore Scale Simulation of NMR Response in Porous Media* (Imperial College London
PhD thesis, http://hdl.handle.net/10044/1/4261) ran a random walk for T2 on the segmented micro-CT
images of the Imperial 2007 collection (Dong & Blunt 2009, Figshare, CC BY 4.0 per record). It is
the one label-volume family with a published measurement made on the released voxels, so it is the
reference this geometry is built against: the same image, the same crop, the same D0, T2B and
surface relaxivity, and therefore the same **voxelised** surface -- the face area of a segmentation,
which is 3/2 of a smooth surface through the same voxels and is what both walks relax against.

Three rungs, each printed with the thesis number beside it:

1. **porosity and S/V** of the crop against Table 7-1 (sand packs) and Table 8-4 (Berea). His S/V is
   the 6-connected grain-face count divided by the PORE volume (thesis eq. 4.1), which is exactly
   :meth:`LabelVolume.surface_to_volume` of the walking pool.
2. **the T2 decay**, from one walk per rock under a gradient-free CPMG train, against the
   fast-diffusion rate ``rho S/V + 1/T2B`` (eq. 3.13) and its own single-exponential fit.
3. **the log-mean T2** of the decay's regularised inverse Laplace transform (eq. 3.16) against the
   simulated mean T2 of Table 7-2 / Table 8-5, and against his measured T2lm. The inversion here is
   the same penalty family as his, not the same estimator -- :func:`t2_distribution` says which, and
   what the difference is worth.

Because neither his walk nor this one carries a gradient (he zeroes the diffusion term, thesis
§7.6.1), the echo spacing is a sampling interval for S(t) and not a sequence parameter: the train
below samples every ``--sample-ms`` instead of at his 200 us, and the walk's step is set by
``--sub-echo`` samples per interval, each of which the engine divides into the sub-steps its rules ask
for (``--sub-steps`` overrides that, which is how the rules are measured).

Data (not in the repository; ~4 MB per archive, CC BY 4.0):

    https://figshare.com/articles/dataset/LV60A_sandpack/1153795   -> LV60A.nhdr + LV60A.raw
    https://figshare.com/articles/dataset/F42A_sandstone/1189259   -> F42A.nhdr  + F42A.raw
    https://figshare.com/articles/dataset/Berea_Sandstone/1153794  -> Berea.nhdr + Berea.raw

Run:

    python examples/validation/talabi_micro_ct_rocks.py --data DIR [--rocks LV60A F42A Berea]
"""
import argparse
import os
import time

import numpy as np

from dmipy_sim import cpmg, simulate_cpmg
from dmipy_sim.spec import label_volume_spec, geometry_from_spec

# ── Talabi's parameters, thesis Table 8-2 and §7.6 ───────────────────────────────────────────
D0 = 2.07e-9          # m^2/s, brine at 308 K (thesis eq. 7.1)
T2B = 3.1             # s, bulk brine relaxation (eq. 7.2)
CROP = 300            # the central cube he simulated on ("maximum voxel size ... is 300", App. A-1)

#: Per rock: the surface relaxivity he used (m/s), whether he FITTED it to his own CPMG or took it
#: from the literature, his porosity and S/V of the crop, and his simulated and measured mean T2.
TALABI = {
    "LV60A": dict(rho=41e-6, rho_from="fitted to the LV60Y CPMG (thesis §7.6.2)",
                  porosity=0.377, s_over_v=57670, T2_sim=0.512, T2_exp=0.496,
                  table="Table 7-1 / Table 7-2"),
    "LV60B": dict(rho=41e-6, rho_from="fitted to the LV60Y CPMG (thesis §7.6.2)",
                  porosity=0.368, s_over_v=61090, T2_sim=0.488, T2_exp=0.496, crop=338,
                  table="Table 7-1 / Table 7-2"),
    "LV60C": dict(rho=41e-6, rho_from="fitted to the LV60Y CPMG (thesis §7.6.2)",
                  porosity=0.372, s_over_v=61590, T2_sim=0.471, T2_exp=0.496,
                  table="Table 7-1 / Table 7-2"),
    "F42A": dict(rho=41e-6, rho_from="fitted to the F42Y CPMG (thesis §7.6.2)",
                 porosity=0.330, s_over_v=43770, T2_sim=0.677, T2_exp=0.668,
                 table="Table 7-1 / Table 7-2"),
    "F42B": dict(rho=41e-6, rho_from="fitted to the F42Y CPMG (thesis §7.6.2)",
                 porosity=0.333, s_over_v=44930, T2_sim=0.654, T2_exp=0.668,
                 table="Table 7-1 / Table 7-2"),
    "F42C": dict(rho=41e-6, rho_from="fitted to the F42Y CPMG (thesis §7.6.2)",
                 porosity=0.331, s_over_v=45760, T2_sim=0.647, T2_exp=0.668,
                 table="Table 7-1 / Table 7-2"),
    "Berea": dict(rho=15e-6, rho_from="a literature value for sandstone (thesis §8.3); not fitted",
                  porosity=0.196, s_over_v=118960, T2_sim=0.584, T2_exp=None,
                  table="Table 8-4 / Table 8-5"),
}


def central_crop(shape, n):
    """The ``(i0, j0, k0, i1, j1, k1)`` central cube of side ``n`` voxels of an image of ``shape``.

    Talabi cropped "a central cubic section of voxel size 300^3" (thesis §8.2, §9.1) and never
    recorded the offset, so the centre is the one reconstruction his own porosities support.
    """
    lo = [(int(s) - int(n)) // 2 for s in shape]
    if any(v < 0 for v in lo):
        raise ValueError(f"an image of {shape} has no central {n}^3 cube")
    return tuple(lo) + tuple(v + int(n) for v in lo)


def t2_distribution(t, S, T2_grid, lam=0.1):
    """Amplitudes of ``S(t) = sum_i a_i exp(-t / T2_i)``, ``a_i >= 0``, by non-negative least squares
    with a second-difference penalty of weight ``lam`` on a log-spaced grid.

    This is the same penalty FAMILY as the thesis's inversion and not the same estimator. Talabi's
    App. A-3 (after Chen et al. 1999) is an unconstrained normal-equation solve,
    ``A = (K'K + lam^2 W)^-1 K' M``, with ``W`` a fourth-derivative matrix, ``lam`` fixed by his
    eq. A.2.6 and 100 grid points from 0.1 ms; this is a non-negative solve, a second-difference
    penalty, ``lam`` fixed at 0.1 and 60 points from 1 ms. So the log-mean below is comparable to his
    Table 7-2 rather than computed the same way.

    How much of the comparison that costs is measured rather than argued: over ``lam`` in
    [0.003, 1] and five grids (40/60/100/150 points, from 0.1 ms and from 1 ms), the log-mean of this
    walk's LV60A decay spans 479.6-494.6 ms about a median of 487.8 (3.1 %) and Berea's 519.0-541.7
    about 539.3 (4.2 %). The inversion is therefore not what puts Berea 7.5 % under his number, and it
    is about half of what would be needed to explain it.

    The log-mean of the result (his eq. 3.16) is what Table 7-2 reports as the mean T2.
    """
    from scipy.optimize import nnls
    K = np.exp(-t[:, None] / T2_grid[None, :])
    n = len(T2_grid)
    W = np.zeros((n, n))
    for i in range(1, n - 1):
        W[i, i - 1:i + 2] = (1.0, -2.0, 1.0)
    A = np.vstack([K, lam * np.linalg.norm(K) / max(np.linalg.norm(W), 1e-30) * W])
    b = np.concatenate([S, np.zeros(n)])
    a, _ = nnls(A, b)
    return a


def log_mean_T2(T2_grid, a):
    """``exp(sum a_i log T2_i / sum a_i)`` -- Talabi's eq. 3.16."""
    w = a.sum()
    return float(np.exp((a * np.log(T2_grid)).sum() / w)) if w > 0 else float("nan")


def run_rock(name, data_dir, *, n_walkers, T_max, sample_ms, sub_echo, walker_batch, seed=0,
             sub_steps=None):
    ref = TALABI[name]
    path = os.path.join(data_dir, f"{name}.nhdr")
    from dmipy_sim.io.label_volume import read_label_volume
    shape = read_label_volume(path).labels.shape
    crop = central_crop(shape, ref.get("crop", CROP))
    spec = label_volume_spec(path, pools={0: "free", 1: "grain"}, crop=crop,
                             rho=ref["rho"], D=D0, T2=T2B,
                             id=f"imperial2007/{name.lower()}",
                             source="Imperial College 2007 micro-CT collection (Dong & Blunt 2009), CC BY 4.0")
    g = geometry_from_spec(spec)
    phi, sv = g.porosity(), g.surface_to_volume()

    n_echoes = int(round(T_max / (sample_ms * 1e-3)))
    seq = cpmg(n_echoes, sample_ms * 1e-3, n_t_per_echo=int(sub_echo))
    dt = seq.dt
    # The WALK's step, which is what the physics is resolved at: the engine divides the sequence's dt
    # into `resolve_sub_steps` fine steps (here the surface-relaxivity rule's quarter of a voxel), so
    # sqrt(6 D dt) is the save interval's displacement and not the step. `sub_steps` overrides the
    # rule, which is how the rule itself is measured.
    from dmipy_sim.engine.physics import resolve_sub_steps
    n_sub = int(sub_steps) if sub_steps else resolve_sub_steps(g, D0, dt, surface=True)
    step = float(np.sqrt(6 * D0 * dt / n_sub))
    t0 = time.time()
    S = simulate_cpmg(n_walkers, D0, seq, g, T2=T2B, seed=seed, sub_steps=sub_steps,
                      walker_batch_size=walker_batch).ravel()
    wall = time.time() - t0
    t = np.arange(1, n_echoes + 1) * sample_ms * 1e-3

    rate_fd = ref["rho"] * sv + 1.0 / T2B                       # thesis eq. 3.13
    m = (t > 0.05) & (t < min(0.8, T_max)) & (S > 1e-3)
    rate_fit = -np.polyfit(t[m], np.log(S[m]), 1)[0]
    T2_grid = np.logspace(np.log10(1e-3), np.log10(10.0), 60)
    a = t2_distribution(t, S, T2_grid)
    return dict(name=name, spec=spec, phi=phi, s_over_v=sv, ref=ref, t=t, S=S, dt=dt, step=step,
                n_walkers=n_walkers, wall=wall, sub_steps=n_sub,
                T2_fd=1.0 / rate_fd, T2_fit=1.0 / rate_fit, T2_lm=log_mean_T2(T2_grid, a),
                floor=1.0 / np.sqrt(n_walkers))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="directory holding <rock>.nhdr and <rock>.raw")
    p.add_argument("--rocks", nargs="+", default=["LV60A", "F42A", "Berea"], choices=sorted(TALABI))
    p.add_argument("--walkers", type=int, default=200_000)
    p.add_argument("--T-max", type=float, default=3.0, help="seconds (his plotted window)")
    p.add_argument("--sample-ms", type=float, default=1.0, help="S(t) sampling interval, ms")
    p.add_argument("--sub-echo", type=int, default=2, help="sequence samples per sampling interval")
    p.add_argument("--sub-steps", type=int, default=None,
                   help="override the engine's sub-step count (for measuring the step rule itself)")
    p.add_argument("--walker-batch", type=int, default=25_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default=None, help="write the decays to this .npz")
    args = p.parse_args()

    rows = []
    for name in args.rocks:
        r = run_rock(name, args.data, n_walkers=args.walkers, T_max=args.T_max,
                     sample_ms=args.sample_ms, sub_echo=args.sub_echo, sub_steps=args.sub_steps,
                     walker_batch=args.walker_batch, seed=args.seed)
        rows.append(r)
        ref = r["ref"]
        print(f"\n=== {name}  ({ref['table']};  rho = {ref['rho'] * 1e6:.0f} um/s, {ref['rho_from']}) ===")
        print(f"  crop {r['spec'].walls[0].surface.crop}   voxel "
              f"{r['spec'].walls[0].surface.voxel_size[0] * 1e6:.4g} um   "
              f"walkers {r['n_walkers']:,}   dt {r['dt'] * 1e6:.1f} us x {r['sub_steps']} sub-steps   "
              f"step {r['step'] * 1e6:.3f} um   {r['wall']:.1f} s")
        print(f"  porosity      ours {r['phi']:.4f}      Talabi {ref['porosity']:.4f}"
              f"      ({100 * (r['phi'] / ref['porosity'] - 1):+.2f} %)")
        print(f"  S/V (1/m)     ours {r['s_over_v']:.0f}       Talabi {ref['s_over_v']:.0f}"
              f"       ({100 * (r['s_over_v'] / ref['s_over_v'] - 1):+.2f} %)")
        print(f"  T2 fast-diffusion 1/(rho S/V + 1/T2B) {r['T2_fd'] * 1e3:.1f} ms")
        print(f"  T2 single-exponential fit             {r['T2_fit'] * 1e3:.1f} ms")
        print(f"  T2 log-mean of the inverted decay     {r['T2_lm'] * 1e3:.1f} ms"
              f"      Talabi simulated {ref['T2_sim'] * 1e3:.0f} ms"
              + (f", measured {ref['T2_exp'] * 1e3:.0f} ms" if ref["T2_exp"] else ""))

    print("\n%-7s %9s %9s %11s %11s %9s %9s %9s" % ("rock", "phi", "phi(T)", "S/V", "S/V(T)",
                                                    "T2lm", "T2sim(T)", "T2exp(T)"))
    for r in rows:
        ref = r["ref"]
        print("%-7s %9.4f %9.4f %11.0f %11.0f %9.1f %9.0f %9s"
              % (r["name"], r["phi"], ref["porosity"], r["s_over_v"], ref["s_over_v"],
                 r["T2_lm"] * 1e3, ref["T2_sim"] * 1e3,
                 f"{ref['T2_exp'] * 1e3:.0f}" if ref["T2_exp"] else "-"))
    if args.save:
        np.savez(args.save, **{f"{r['name']}_S": r["S"] for r in rows},
                 **{f"{r['name']}_t": r["t"] for r in rows})
        print("wrote", args.save)


if __name__ == "__main__":
    main()
