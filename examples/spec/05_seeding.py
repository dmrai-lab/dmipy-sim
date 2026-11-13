"""Seeding: which pools get walkers, and in what proportion.

The spec states which pools are seeded, by what rule, and how the walkers are weighted. It is separate from
the pools' water fractions on purpose: the fractions say how many protons a pool HAS, the seeding says where
this particular walk puts its samples. A pool with water that is not seeded is simply not sampled by this
walk; a pool without water cannot be seeded at all.

Getting this wrong is quiet. A packed cell seeded only outside its cylinders is a perfectly valid extra-axonal
experiment, and it is also the wrong substrate if you meant to measure both pools -- the signal comes back
looking like a plausible tissue either way.

Assumes rung 02.
"""
import numpy as np

from dmipy_sim import Cylinder, MyelinatedCylinder, PackedCylinders, pack_cylinders

radii = np.full(4, 2e-6)
centers, L, _ = pack_cylinders(radii, target_vf=0.3, seed=0)

cases = [("isolated cylinder", Cylinder(5e-6, (0, 0, 1))),
         ("packed cylinders", PackedCylinders(radii, centers, L)),
         ("packed, extra only", PackedCylinders(radii, centers, L, pool="extra")),
         ("myelinated cylinder", MyelinatedCylinder(2e-6, 2.8e-6, (0, 0, 1), D_intra=1.7e-9, D_extra=2e-9))]
for name, g in cases:
    s = g.spec
    seeded = [s.pool(i).name for i in s.seeding.pools]
    print(f"{name:22s} seeds {str(seeded):32s} rule {s.seeding.rule!r} weights {s.seeding.weights!r}")

print("\n  uniform_by_volume  walkers are placed by the volume each pool occupies, so the counts follow the")
print("                     geometry rather than being chosen")
print("  weights            'water_fraction' multiplies each walker by its pool's share of the protons, so a")
print("                     half-dry pool contributes half as much signal without needing half the walkers")
print("\nFor a pack that will become a voxel grid, `spec.StratifiedByVoxel(grid=, walkers_per_voxel=)` seeds the")
print("same count of every pool in each voxel instead, so a voxel's floor is set by its count rather than by")
print("Poisson chance -- which is what `Phantom.partition` needs.")
