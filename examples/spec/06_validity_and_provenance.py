"""What the spec claims it is good for, and where it came from.

Two blocks carry no physics and decide a great deal. `validity` states the smallest feature the substrate
has, the narrowest gap between objects, and which TIERS the spec can support -- and the tier list is what
later steps check before they record a channel. `provenance` states where the substrate came from: the
producer, the files it cites with their hashes, and every transformation applied on the way in.

The point of both is that a result stays traceable without the script that made it. A pack embeds its spec,
so a pack found on a hub still says which files it came from and what it may be replayed for.

Assumes rung 03.
"""
import numpy as np

from dmipy_sim import PackedCylinders, pack_cylinders

radii = np.array([1.0e-6, 2.0e-6, 1.5e-6])
centers, L, vf = pack_cylinders(radii, target_vf=0.25, seed=0)
spec = PackedCylinders(radii, centers, L, surface_relaxivity_t2=1e-6).spec

v = spec.validity
print(f"smallest feature {v.smallest_feature*1e6:.3f} um   narrowest gap "
      f"{'-' if v.min_gap is None else format(v.min_gap*1e6, '.3f') + ' um'}")
print(f"tiers            {v.tiers}")
print("\nThe tier list is a claim, and it is read rather than assumed:")
print("  gradient    positions are recorded, so any waveform can be replayed")
print("  relaxation  the pool each walker is in is recorded, so per-pool T2 and T1 can be applied")
print("  surface     the time spent against a wall is recorded, so a surface relaxivity can be applied")
print("  field       a susceptibility source exists and its basis can be built, so a field can be applied")
print("A tier the spec does not list is not silently skipped later; asking for it raises.")

print(f"\nsmallest feature is what the step rule divides: a walker's step must resolve {v.smallest_feature*1e6:.3f} um,")
print("and the narrowest gap is what a step may not cross in one move")

prov = spec.provenance or {}
print(f"\nprovenance keys: {sorted(prov)}")
for k in ("source", "created"):
    if k in prov:
        print(f"   {k}: {prov[k]}")
print(f"realisation: {sorted((spec.realisation or {}))}")
print("\nA producer that reads a published dataset records each file with its sha256 and every transformation")
print("it applied, so the substrate can be rebuilt from the same inputs and checked against them.")
