"""Walls: the surface between two pools, and the physics that lives on it.

A wall names a surface and the pool on each side of it. Everything that happens AT a boundary is declared
here rather than on the pools: how permeable it is in each direction, how relaxing it is on each side, and
how it binds. The asymmetry is deliberate -- a membrane can let water out faster than in, and can relax the
inside differently from the outside -- so the spec records a value per direction and per side rather than one
number per wall.

Assumes rung 02.
"""
import numpy as np

from dmipy_sim import MyelinatedCylinder, PackedCylinders, pack_cylinders

myelin = MyelinatedCylinder(2.0e-6, 2.8e-6, (0, 0, 1), D_intra=1.7e-9, D_extra=2.0e-9,
                            kappa_inner=1e-5, kappa_outer=2e-5).spec
radii = np.full(4, 2e-6)
centers, L, _ = pack_cylinders(radii, target_vf=0.3, seed=0)
packed = PackedCylinders(radii, centers, L, surface_relaxivity_t2=1.2e-6, permeability=1e-5).spec

for name, spec in (("a myelinated cylinder", myelin), ("a periodic packing", packed)):
    print(f"{name}: {len(spec.walls)} wall(s)")
    for w in spec.walls:
        out = "the void" if w.outside_pool is None else f"pool {w.outside_pool}"
        print(f"   {w.name:10s} {w.surface.kind:10s} pool {w.inside_pool} | {out}")
        print(f"      permeability        in->out {w.permeability.in_to_out:g}  out->in {w.permeability.out_to_in:g} m/s")
        print(f"      surface relaxivity  inside  {w.surface_relaxivity.inside:g}  outside {w.surface_relaxivity.outside:g} m/s")

print("\nWhat a wall decides:")
print("  the surface     a shape, or a file with its hash -- a spec embedded in a shard cannot carry 12,196")
print("                  centerlines, so it CITES the file it came from")
print("  inside/outside  which pools it separates; None outside means the void, reachable only if permeable")
print("  permeability    baked into the WALK: one walk per value, because it changes the paths")
print("  relaxivity      recorded as a contact channel and applied at REPLAY: one walk serves every value")
print("  mt_reactivity   the same, for binding")
