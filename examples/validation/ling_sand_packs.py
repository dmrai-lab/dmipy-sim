"""Ling et al. 2022's NMR random walk on their quartz/garnet sand packs, on the same voxels.

Ling, Hussaini, Elsayed, Connolly, El-Husseiny, Mahmoud, May & Johns (2022), *Model Synthetic Samples
for Validation of NMR Signal Simulations*, Transport in Porous Media 142(3):623-639,
https://doi.org/10.1007/s11242-022-01764-w (CC BY 4.0) measured a CPMG on five resin-set packs of
sieved quartz and garnet sand, then micro-CT imaged **those same packs** and ran a random walk on the
images. Both halves are released (figshare https://doi.org/10.6084/m9.figshare.17161730, CC BY 4.0,
`450CubicEqSized.zip` and `CPMG data.zip`), which makes this the one label-volume reference where the
measurement and the geometry are the same physical object.

Three rungs, each printed with their number beside it:

1. **porosity and S/V** of the released 450^3 sub-volume. The paper reports neither per sample, so
   these are ours and are stated as such; the S/V is the 6-connected grain-face count over the pore
   volume, which is the **Manhattan** surface of the segmentation and is what both walks relax against.
2. **the T2 decay**, from one walk per pack under a gradient-free CPMG train, against the
   fast-diffusion rate ``rho S/V + 1/T2B``. That formula holds only for ``rho (V/S) / D << 1``, which is
   0.07 for the quartz pack and 0.60 for the garnet one -- so the garnet pack is where the walk is not
   optional, and the two numbers printed side by side show it.
3. **the log-mean T2** of the decay's regularised inverse Laplace transform, against the log-mean of
   Ling's own released ``te`` = 100 us echo train inverted the same way. Their paper prints no T2
   number: its comparison is Fig. 5, a figure. So the measured number here is computed from their
   released data by :func:`measured_log_mean` and not transcribed from the paper.

Their surface relaxivities are **theirs**: 12.5 um/s (quartz) and 98.5 um/s (garnet), fitted by them to
their own te = 100 us T2 distributions (SS3.2) by overlaying whole distributions, not by matching any
number -- the paper prints no T2 value anywhere -- with an independent maximal-ball estimate of 22 and
80 um/s in the same section. The bulk T2 of the brine is **ours**: the paper states none. The
diffusivity is theirs only in the sense that 2.3e-9 is the "e.g." beside their step equation; see D0.

A pack of two minerals has two of their relaxivities on two walls, and this geometry accumulates one
boundary local time over both, so ``--rho`` is one number and a mixture is walked at a stated single
value for the code's sake, never as a claim about their simulation.

Data (not in the repository; 37 MB and 20 MB zipped, CC BY 4.0):

    https://doi.org/10.6084/m9.figshare.17161730  -> 450CubicEqSized.zip, CPMG data.zip

Run:

    python examples/validation/ling_sand_packs.py --data DIR [--samples 6_Q100 1_G100] [--measured DIR]
"""
import argparse
import glob
import os
import time

import numpy as np

from dmipy_sim import cpmg, simulate_cpmg
from dmipy_sim.spec import geometry_from_spec, label_volume_spec

# The inversion is the one the Imperial rocks validation owns: the same estimator on both decays, and
# that module states what its penalty family and grid are worth. It is imported, never copied.
from examples.validation.talabi_micro_ct_rocks import log_mean_T2, t2_distribution

# The paper states NO diffusivity for its walk. SS2.4 introduces D0 in the step-length equation with an
# example -- "e.g. for water at 25 C, D0 = 2.3e-9 m^2/s" -- and SS2.5 uses 2.2e-9 for "the self-diffusion
# coefficient of the brine solutions" in a diffusion-length check. Neither sentence says which the
# simulations took. 2.3e-9 is the value used here because it is the one attached to the walk's own
# equation; the 4.5 % to 2.2e-9 is an uncertainty this family carries, not one it resolves.
D0 = 2.3e-9       # m^2/s, the "e.g." beside their step-length equation (SS2.4, Eq. 6)
D0_ALT = 2.2e-9   # m^2/s, the brine value in their diffusion-length check (SS2.5)
T2B = 3.0         # s, the brine's bulk T2 -- OURS: the paper states none

#: Their fitted surface relaxivity per mineral (SS3.2), and the independent maximal-ball estimate.
RHO = {"quartz": 12.5e-6, "garnet": 98.5e-6}
RHO_MAXIMAL_BALL = {"quartz": 22e-6, "garnet": 80e-6}

#: Per released image: the label map and the mineral of each solid label. The file names carry the
#: recipe (``6_Q100_Extracted_Voxel-3.93micro_X-450_Y-450_Z-450_Labels_0-Pore_1-Quartz_ASCII.am``), and
#: their indices are the DATASET's, which are not the paper's Table 1 sample ids (the paper numbers
#: 1 = Q100 .. 5 = G100; the deposit ships 1_G100, 2_Q25G75, 4_Q50G50, 5_Q75G25, 6_Q100 and no 3). The
#: composition in the file name is what identifies a pack here.
LING = {
    "6_Q100": dict(pools={0: "free", 1: "quartz"}, composition="100 % quartz (v/v)"),
    "5_Q75G25": dict(pools={0: "free", 1: "quartz", 2: "garnet"}, composition="75 % quartz / 25 % garnet"),
    "4_Q50G50": dict(pools={0: "free", 1: "quartz", 2: "garnet"}, composition="50 % quartz / 50 % garnet"),
    "2_Q25G75": dict(pools={0: "free", 1: "quartz", 2: "garnet"}, composition="25 % quartz / 75 % garnet"),
    "1_G100": dict(pools={0: "free", 1: "garnet"}, composition="100 % garnet (v/v)"),
}


def image_path(sample, data_dir):
    """The released ``.am`` of one pack, found by its dataset id."""
    hits = sorted(glob.glob(os.path.join(data_dir, f"{sample}_*.am")))
    if len(hits) != 1:
        raise FileNotFoundError(f"{data_dir}: {len(hits)} files match {sample}_*.am ({hits})")
    return hits[0]


def default_rho(sample):
    """Their relaxivity for a pack of one mineral; for a mixture there is no single one, so the caller
    states it."""
    minerals = {n for n in LING[sample]["pools"].values() if n != "free"}
    return RHO[minerals.pop()] if len(minerals) == 1 else None


def measured_log_mean(sample, cpmg_dir, *, te=100e-6, sample_ms=1.0, grid=None):
    """The log-mean T2 of THEIR released echo train at echo spacing ``te``, on the ``sample_ms`` grid.

    ``cpmg_dir`` holds one ``<sample>.npz`` per pack, the 32 ``(time, signal)`` column pairs of the
    released workbook (``CPMG data.zip``, ``<sample>-Int Grad.xlsx``) as ``t<i>`` / ``s<i>`` with the
    echo spacings in ``te``. Only the echoes that land on the grid our own decay is sampled on are kept,
    so the two log-means differ by their decays and not by their sampling.
    """
    z = np.load(os.path.join(cpmg_dir, f"{sample}.npz"))
    tes = np.asarray(z["te"], float)
    i = int(np.nanargmin(np.abs(tes - te)))
    if not np.isfinite(tes[i]) or abs(tes[i] - te) > 1e-9:
        raise ValueError(f"{sample}: no released train at te = {te * 1e6:g} us (have {tes * 1e6} us)")
    t, S = np.asarray(z[f"t{i}"], float), np.asarray(z[f"s{i}"], float)
    n = t / (sample_ms * 1e-3)
    k = np.flatnonzero(np.isclose(np.round(n), n, atol=1e-6))
    g = np.logspace(-3, 1, 60) if grid is None else grid
    return dict(T2_lm=log_mean_T2(g, t2_distribution(t[k], S[k] / S[0], g)), te=float(tes[i]),
                n_echoes=int(t.size), t_last=float(t[-1]), n_inverted=int(k.size),
                S_last_over_S0=float(S[-1] / S[0]))


def run_pack(sample, data_dir, *, rho=None, n_walkers=200_000, T_max=3.5, sample_ms=1.0,
             D0=D0, T2B=T2B, walker_batch=20_000, seed=0, sub_steps=None, folds=1):
    """One direct fused walk: the substrate with ``rho`` baked in, its decay, and the decay's log-mean.

    This is the OTHER route to the quantity a replay pack serves as a knob, which is what makes it the
    reference a pack is gated against: same image, same D0, same T2B, same inversion, ``rho`` in the
    walk instead of in the replay.

    ``folds`` splits the walkers into that many independent walks on consecutive seeds and averages
    their decays, which is the same estimator as one walk of ``n_walkers`` at the same cost. It is a
    convenience for walking a count larger than one device pass, and it is NOT the log-mean's error bar:
    a fold spread is a seed-dependent estimate of one (measured on 1_G100: 0.00189-0.00589 over ten
    seeds at six folds), and a gate threshold cannot be seed-dependent. The error bar is the ANALYTIC
    standard error of the log-mean -- the delta method over walkers, no folds, no seed and no coverage
    factor -- which the family's ``records/design.json`` derives and its gate uses. ``1 / sqrt(N)`` is
    the floor of the SIGNAL and is reported beside it as exactly that.
    """
    ref = LING[sample]
    rho = default_rho(sample) if rho is None else float(rho)
    if rho is None:
        raise ValueError(f"{sample} has two mineral surfaces and no single relaxivity; pass rho=")
    path = image_path(sample, data_dir)
    # The substrate goes to the ORIGIN. These images are deposited at the micro-CT stage's absolute
    # coordinates, near (-717, -717, -458) mm, where a float32 ulp is 1/65 of the 3.93 um voxel and
    # `LabelVolume` refuses to walk them (`NUDGE_FRACTION_MAX`): the nudge that keeps a walker off a face
    # is no longer small against a voxel. Translating the volume is physically nothing.
    from dmipy_sim.io.label_volume import read_label_volume
    vol = read_label_volume(path)
    origin = -0.5 * np.asarray(vol.labels.shape, float) * np.asarray(vol.voxel_size, float)
    spec = label_volume_spec(path, pools=dict(ref["pools"]), rho=rho, D=D0, T2=T2B, origin=origin,
                             id=f"ling2022/{sample.lower()}",
                             source="Ling et al. 2022 model synthetic sediment samples, figshare "
                                    "doi:10.6084/m9.figshare.17161730, CC BY 4.0")
    g = geometry_from_spec(spec)
    phi, sv = g.porosity(), g.surface_to_volume()

    n_echoes = int(round(T_max / (sample_ms * 1e-3)))
    seq = cpmg(n_echoes, sample_ms * 1e-3, n_t_per_echo=2)
    dt = seq.dt
    from dmipy_sim.engine.physics import resolve_sub_steps
    n_sub = int(sub_steps) if sub_steps else resolve_sub_steps(g, D0, dt, surface=True)
    step = float(np.sqrt(6 * D0 * dt / n_sub))
    n_each = int(n_walkers) // int(folds)
    t0 = time.time()
    parts = [simulate_cpmg(n_each, D0, seq, g, T2=T2B, seed=seed + i, sub_steps=sub_steps,
                           walker_batch_size=walker_batch).ravel() for i in range(int(folds))]
    wall = time.time() - t0
    S = np.mean(parts, axis=0)
    t = np.arange(1, n_echoes + 1) * sample_ms * 1e-3
    grid = np.logspace(-3, 1, 60)
    lm = log_mean_T2(grid, t2_distribution(t, S, grid))
    lms = [log_mean_T2(grid, t2_distribution(t, p, grid)) for p in parts] if len(parts) > 1 else []
    # Reported, never a tolerance: the spread of the folds when there are any, and the SIGNAL's shot
    # noise. The log-mean's error bar is analytic and lives with the gate.
    fold_spread = float(np.std(lms, ddof=1) / np.sqrt(len(lms)) / lm) if len(lms) > 1 else None
    return dict(name=sample, spec=spec, phi=phi, s_over_v=sv, rho=rho, t=t, S=S, dt=dt, step=step,
                n_walkers=int(n_each * len(parts)), n_folds=int(len(parts)), wall=wall, sub_steps=n_sub,
                T2_fd=1.0 / (rho * sv + 1.0 / T2B), rho_V_over_S_over_D=rho / sv / D0,
                T2_lm=lm, T2_lm_folds=[float(x) for x in lms], fold_spread=fold_spread,
                signal_floor=1.0 / np.sqrt(n_each * len(parts)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="directory holding the released <sample>_*.am images")
    p.add_argument("--measured", default=None,
                   help="directory holding <sample>.npz, their released echo trains")
    p.add_argument("--samples", nargs="+", default=["6_Q100", "1_G100"], choices=sorted(LING))
    p.add_argument("--rho", type=float, default=None, help="m/s; default is their value for the mineral")
    p.add_argument("--walkers", type=int, default=200_000)
    p.add_argument("--folds", type=int, default=1,
                   help="independent walks to average (a convenience for a large count, not an error bar)")
    p.add_argument("--T-max", type=float, default=3.5, help="seconds (their longest released train)")
    p.add_argument("--sample-ms", type=float, default=1.0, help="S(t) sampling interval, ms")
    p.add_argument("--sub-steps", type=int, default=None,
                   help="override the engine's sub-step count (for measuring the step rule itself)")
    p.add_argument("--walker-batch", type=int, default=20_000)
    p.add_argument("--T2B", type=float, default=T2B, help="s; OURS, the paper states none")
    p.add_argument("--D0", type=float, default=D0, help="m^2/s")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default=None, help="write the decays to this .npz")
    args = p.parse_args()

    rows = []
    for name in args.samples:
        r = run_pack(name, args.data, rho=args.rho, n_walkers=args.walkers, T_max=args.T_max,
                     sample_ms=args.sample_ms, sub_steps=args.sub_steps, D0=args.D0, T2B=args.T2B,
                     walker_batch=args.walker_batch, seed=args.seed, folds=args.folds)
        if args.measured:
            r["measured"] = measured_log_mean(name, args.measured, sample_ms=args.sample_ms)
        rows.append(r)
        surf = r["spec"].walls[0].surface
        print(f"\n=== {name}  ({LING[name]['composition']};  rho = {r['rho'] * 1e6:g} um/s, "
              f"{'Ling SS3.2, fitted by them' if args.rho is None else 'stated on the command line'}) ===")
        print(f"  image {os.path.basename(surf.file)}  voxel {surf.voxel_size[0] * 1e6:.5g} um  "
              f"walkers {r['n_walkers']:,}  dt {r['dt'] * 1e6:.1f} us x {r['sub_steps']} sub-steps  "
              f"step {r['step'] * 1e6:.3f} um  {r['wall']:.1f} s")
        print(f"  porosity (ours; the paper reports none per sample)   {r['phi']:.4f}")
        print(f"  S/V (1/m) (ours, the Manhattan surface)              {r['s_over_v']:.0f}")
        print(f"  rho (V/S) / D                                        {r['rho_V_over_S_over_D']:.3f}"
              f"   (<< 1 is where the fast-diffusion formula holds)")
        print(f"  T2 fast-diffusion 1/(rho S/V + 1/T2B)                {r['T2_fd'] * 1e3:.1f} ms")
        print(f"  T2 log-mean of the inverted decay                    {r['T2_lm'] * 1e3:.1f} ms"
              + (f"   ({len(r['T2_lm_folds'])} folds, spread {r['fold_spread'] * 100:.2f} %)"
                 if r["T2_lm_folds"] else f"   (signal floor {r['signal_floor'] * 100:.2f} %)"))
        if "measured" in r:
            m = r["measured"]
            print(f"  Ling's MEASURED train, same inversion                {m['T2_lm'] * 1e3:.1f} ms"
                  f"   (te {m['te'] * 1e6:.0f} us, {m['n_echoes']:,} echoes to {m['t_last']:g} s, "
                  f"{100 * (r['T2_lm'] / m['T2_lm'] - 1):+.1f} %)")

    print("\n%-10s %9s %11s %8s %9s %9s %9s" % ("pack", "phi", "S/V", "rhoVS/D", "T2fd", "T2lm", "theirs"))
    for r in rows:
        theirs = f"{r['measured']['T2_lm'] * 1e3:.1f}" if "measured" in r else "-"
        print("%-10s %9.4f %11.0f %8.3f %9.1f %9.1f %9s"
              % (r["name"], r["phi"], r["s_over_v"], r["rho_V_over_S_over_D"], r["T2_fd"] * 1e3,
                 r["T2_lm"] * 1e3, theirs))
    if args.save:
        np.savez(args.save, **{f"{r['name']}_S": r["S"] for r in rows},
                 **{f"{r['name']}_t": r["t"] for r in rows})
        print("wrote", args.save)


if __name__ == "__main__":
    main()
