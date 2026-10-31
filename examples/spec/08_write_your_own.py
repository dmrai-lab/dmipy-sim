"""Writing a spec, and every way it is refused.

A producer for a new substrate emits a spec and constructs nothing. This is that, at its smallest: one
sphere, two pools, one wall. The second half is the more useful part -- `validate` refuses a spec that does
not describe a walkable situation, and it names the field it refused, so the failures below are the contract
stated as errors.

Assumes rungs 02 and 03.
"""
from dataclasses import replace

from dmipy_sim.spec import SpecError, geometry_from_spec
from dmipy_sim.spec.substrate import Domain, Pool, Seeding, Surface, SubstrateSpec, Validity, Wall

R = 5e-6
good = SubstrateSpec(
    id="cookbook/one-sphere",
    domain=Domain([-4 * R] * 3, [4 * R] * 3, ["open"] * 3),
    pools=[Pool(0, "extra", None, water_fraction=0.0), Pool(1, "intra", 2e-9, water_fraction=1.0)],
    walls=[Wall("membrane", Surface("sphere", center=[0.0] * 3, radius=R), inside_pool=1, outside_pool=None)],
    seeding=Seeding([1]),
    validity=Validity(R, ["gradient"]),
    description="one impermeable sphere in an open box; the outside is void",
).validate()
print(f"valid: {good.id} -> {type(geometry_from_spec(good)).__name__}")

# Each of these is a way the situation does not hold together. The message names the field.
def refused(what, spec):
    try:
        spec.validate()
    except (SpecError, ValueError, KeyError) as e:
        print(f"  {what:34s} {type(e).__name__}: {str(e)[:96]}")
    else:
        print(f"  {what:34s} ACCEPTED")

print("\nrefused:")
refused("a wall between unknown pools", replace(good, walls=[replace(good.walls[0], inside_pool=7)]))
refused("seeding a pool that does not exist", replace(good, seeding=Seeding([9])))
refused("seeding a pool with no water", replace(good, seeding=Seeding([0])))
refused("two pools sharing an id", replace(good, pools=[good.pools[1], good.pools[1]]))
refused("a box that is not a box", replace(good, domain=Domain([0] * 3, [-1e-6] * 3, ["open"] * 3)))
refused("a boundary kind that is not one", replace(good, domain=Domain([-1e-5] * 3, [1e-5] * 3, ["squishy"] * 3)))

# Not everything unwalkable is caught HERE. A pool that is seeded but has no diffusivity passes `validate`
# and is refused by the walk instead, which is a division worth knowing: `validate` checks that the
# situation is well formed, the walk checks that it can be executed (dmipy-sim#318).
unwalkable = replace(good, pools=[good.pools[0], replace(good.pools[1], D=None)]).validate()
print(f"\naccepted by validate, refused by the walk: pool {unwalkable.pool(1).name!r} has D = "
      f"{unwalkable.pool(1).D}")
try:
    from dmipy_sim.spec import walk_spec
    walk_spec(unwalkable, 10, T_max=1e-3, dt_save=5e-4, require_gpu=False)
except Exception as e:
    print(f"  walk_spec: {type(e).__name__}: {str(e)[:88]}")

print("\nThe point is not the list. It is that a spec which passes describes a situation that is well formed,")
print("and the walk then refuses what it cannot execute -- so a malformed substrate is stopped at one of two")
print("named places rather than producing a signal that looks reasonable.")
