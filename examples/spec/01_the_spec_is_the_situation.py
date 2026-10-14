"""A substrate is a spec, and the spec is the whole situation.

Every driver in this library takes its substrate as a `SubstrateSpec`: a domain with a boundary per axis, the
pools of water in it, the walls between them, which pools are seeded, what the spec claims to be valid for,
and where it came from. A geometry object is one spelling of a spec, not a separate thing -- `geometry.spec`
writes out what the constructor left implicit, and a substrate with no spec spelling is refused.

This matters before anything else because the spec fixes what every later step can do. A pool without a
diffusivity cannot be walked. A wall without a permeability cannot exchange. A spec whose validity does not
list a tier cannot carry it into a pack. Nothing downstream recovers information the spec did not state.

Assumes nothing.
"""
import json

from dmipy_sim import Cylinder

spec = Cylinder(radius=5e-6, orientation=(0.0, 0.0, 1.0)).spec

print(f"id          {spec.id}")
print(f"description {spec.description}")
print(f"domain      {spec.domain.box_min} to {spec.domain.box_max} m, boundary {spec.domain.boundary}")
print(f"frame       axis {spec.frame.axis}")
print(f"seeding     pools {spec.seeding.pools}, rule {spec.seeding.rule!r}, weights {spec.seeding.weights!r}")
print(f"validity    smallest feature {spec.validity.smallest_feature:.2e} m, tiers {spec.validity.tiers}")
print("pools")
for p in spec.pools:
    print(f"   {p.id} {p.name:8s} D {p.D}   water fraction {p.water_fraction}   T2 {p.T2}   T1 {p.T1}")
print("walls")
for w in spec.walls:
    print(f"   {w.name:10s} {w.surface.kind:10s} inside pool {w.inside_pool} -> outside pool {w.outside_pool}")

# The spec is a file. Saving it records the situation; loading it is enough to walk again, with no code in
# between, which is what makes a result reproducible by someone who does not have your script.
spec.save("/tmp/cylinder.sub.json")
print(f"\nsaved {len(json.load(open('/tmp/cylinder.sub.json')))} top-level fields to a .sub.json")

# An isolated cylinder declares an OPEN box four radii wide and seeds only the lumen: the outside is void
# unless a permeability makes it reachable. The description says so, and the seeding proves it.
print(f"\nthe outside pool exists but holds no water: "
      f"{[(p.name, p.water_fraction) for p in spec.pools]}")
