"""What the Swoop's magnet costs the BATMAN brain: the ADC bias it imposes, per voxel (dmipy-sim#322 PR 8).

A 64 mT permanent magnet encodes diffusion with its own field gradient, on during every pulse and every
dead time, and the gradient coils add a concomitant term on top. Neither is noise: both are deterministic
functions of where a voxel sits in the bore, so both are replayable -- and both are far above the floor the
simulation itself can resolve.

Run it: `python examples/rph/swoop_brain_bias.py`. It needs no pack and no walk; the bias is a property of
the acquisition and the magnet, not of the substrate.

Recorded numbers, 96 x 96 x 60 at 2.5 mm centred in the bore, 18 directions at b = 945 s/mm^2,
delta / Delta = 35 / 42 ms, against the pack floor 0.0048:

    inside the law's 8 cm anchor        136,584 of 552,960 voxels (25 %)
    worst |ADC bias|                    15.38 %     (the paper measures up to 16.1 %)
    median voxel, worst direction        4.14 %
    direction-averaged, worst voxel      0.98 %
    voxels above the pack's own floor   97.57 %

    by radius      0-2 cm    2,176 voxels   worst  1.51 %   median 0.47 %   above floor  48.1 %
                   2-4 cm   15,080 voxels   worst  4.53 %   median 1.31 %   above floor  86.7 %
                   4-6 cm   40,600 voxels   worst  9.07 %   median 3.06 %   above floor  99.6 %
                   6-8 cm   78,728 voxels   worst 15.38 %   median 5.85 %   above floor 100.0 %

Three things those numbers say.

The bias clears the simulation's own noise floor in 97.6 % of voxels, so it is almost never the thing that
gets lost in the Monte-Carlo error -- but it does not clear it everywhere, and where it fails is not random.
Inside 2 cm of isocentre only about half the voxels clear the floor, because a linearly shimmed magnet has no
first-order field variation there: every harmonic of order two and above has zero gradient at the origin, so
the background gradient grows from nothing. The magnet genuinely does not encode at its own centre.

Averaging over directions hides it. 15.4 % per direction becomes 1.0 % once averaged, since the cross term
flips sign with the diffusion direction. A tensor fit sees the per-direction number, not the average.

The concomitant term is the small half, an order of magnitude below the magnet's own gradient.

At 3 T there is nothing to compute. A shimmed superconducting magnet publishes no field shape because its
residual is parts per million rather than parts per thousand, so the catalogue carries none and this returns
`None` -- which is the comparison, not a gap in it.
"""
import numpy as np

from dmipy_sim import sequences
from dmipy_sim.acquisition.scanners import ScannerLimits
from dmipy_sim.phantom import Grid
from dmipy_sim.phantom.bore import delivered_b_map

SHAPE, VOXEL_M = (96, 96, 60), 2.5e-3
PACK_FLOOR = 0.0048                 # fill/swoop/cactus_1s.rpk, the experiment's own substrate


def brain_grid(shape=SHAPE, voxel_m=VOXEL_M, tilt_deg=2.58):
    """The BATMAN acquisition's matrix, centred in the bore and carrying its prescribed obliquity.

    A head is positioned at isocentre, so the volume is centred rather than left where a crop's origin put
    it. The tilt is the tutorial's own 2.58 degrees about x: a field law is a function of position in the
    BORE, so an oblique grid has to be rotated into it or the law is evaluated at the wrong place.
    """
    c, s = np.cos(np.radians(tilt_deg)), np.sin(np.radians(tilt_deg))
    R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])          # grid axes -> scanner axes
    grid = Grid(shape=shape, voxel_size_m=(voxel_m,) * 3,
                origin_m=tuple(-0.5 * (n - 1) * voxel_m for n in shape),
                isocenter_m=(0.0, 0.0, 0.0))
    return grid, R


def swoop_protocol(n_dirs=18, seed=0):
    """The Swoop's diffusion measurement: b = 945 s/mm^2 at delta / Delta = 35 / 42 ms."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_dirs, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return sequences.pgse(dirs.tolist(), 35e-3, 42e-3, bvalues=[0.945e9] * n_dirs, n_t=400)


def bias_map(scanner, grid, R, sequence, **kw):
    """``(voxels, bias)`` -- the fractional ADC bias per voxel per direction, for the voxels the magnet's
    law is actually anchored over. Beyond that radius the law is an extrapolation and is refused rather
    than guessed, so those voxels are not returned."""
    if getattr(scanner, "b0_harmonic_Z2", None) is None:
        return grid.every_voxel, None          # no law: nothing to anchor, and nothing to bias
    pos = grid.positions_m(grid.every_voxel)
    inside = np.linalg.norm(pos - np.asarray(grid.isocenter_m), axis=-1) < scanner.b0_validity_radius
    vox = grid.every_voxel[inside]
    delivered = delivered_b_map(scanner, grid, sequence, to_scanner=R, voxels=vox, **kw)
    if delivered is None:
        return vox, None
    return vox, delivered / sequence.b() - 1.0


def main():
    swoop = ScannerLimits.of("swoop")
    grid, R = brain_grid()
    seq = swoop_protocol()
    pos = grid.positions_m(grid.every_voxel)
    rad = np.linalg.norm(pos - np.asarray(grid.isocenter_m), axis=-1)

    vox, bias = bias_map(swoop, grid, R, seq)
    inside = rad < swoop.b0_validity_radius
    worst_dir = np.abs(bias).max(axis=1)
    print(f"grid {grid.shape} = {rad.size:,} voxels at {VOXEL_M*1e3:.1f} mm, "
          f"{np.degrees(np.arccos((np.trace(R)-1)/2)):.2f} deg oblique")
    print(f"inside the law's {swoop.b0_validity_radius*100:.0f} cm anchor: {inside.sum():,} ({inside.mean():.0%})\n")
    print(f"  worst |ADC bias|                   {np.abs(bias).max():6.1%}")
    print(f"  median voxel, worst direction      {np.median(worst_dir):6.1%}")
    print(f"  direction-averaged, worst voxel    {np.abs(bias.mean(axis=1)).max():6.1%}")
    above = worst_dir > PACK_FLOOR
    print(f"  voxels above the pack floor        {above.mean():6.3%}"
          f"   ({(~above).sum()} below, a null surface of this direction set)\n")
    for lo, hi in ((0.0, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.08)):
        m = (rad[inside] >= lo) & (rad[inside] < hi)
        if m.any():
            print(f"    {lo*100:4.0f}-{hi*100:2.0f} cm {m.sum():7,} voxels   worst {np.abs(bias[m]).max():5.1%}"
                  f"   median {np.median(worst_dir[m]):5.1%}")

    _v, bg = bias_map(swoop, grid, R, seq, concomitant=False)
    print(f"\n  the magnet's gradient alone        {np.abs(bg).max():7.3%}")
    print(f"  with the coils' Maxwell term       {np.abs(bias).max():7.3%}")
    print(f"  so the Maxwell term is worth       {np.abs(bias).max() - np.abs(bg).max():7.3%} here, "
          f"and up to {np.abs(bias - bg).max():.3%} in some voxel")

    _v, none = bias_map(ScannerLimits.of("prisma"), grid, R, seq)
    print(f"\n  the same brain at 3 T: {none if none is None else 'a law'} -- a shimmed superconducting "
          f"magnet publishes no shape,\n  its residual being parts per million rather than parts per "
          f"thousand. That is the comparison.")


if __name__ == "__main__":
    main()
