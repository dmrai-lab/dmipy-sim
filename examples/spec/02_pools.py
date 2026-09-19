"""Pools: where the water is, what it is made of, and what the spec means by "nominal".

A pool is a population of water with its own diffusivity and its own share of the protons. Pool ids follow
one convention everywhere: 0 is the extra-cellular or free pool, enclosed pools are positive, 1 is the lumen
and 2 the myelin. Every per-pool quantity downstream -- the occupancy channel, per-pool T2 and T1 at replay,
the compartment a walker is selected by -- is indexed by that id.

A pool may also carry NOMINAL values: the T2, T1 and susceptibility the producer calibrated it at. They are
not applied by anything. A pack carries no physical value at all; `pack.nominal` reads these back so a
published substrate reproduces its own paper, and any other tissue is a replay knob.

Assumes rung 01.
"""
from dmipy_sim import MyelinatedCylinder
from dmipy_sim.compartments import Compartments, Pool

# Per-compartment properties have ONE spelling: a `Compartments` of `Pool`s by name. Each pool carries what
# is true of that water -- its diffusivity, and the relaxation it was calibrated at.
# A water fraction is set on every pool or on none: a partial set would leave the others implicit, and the
# proton budget of the substrate has to add up.
tissue = Compartments(intra=Pool(D=1.7e-9, T2=0.080, T1=1.2, water_fraction=1.0),
                      extra=Pool(D=2.0e-9, T2=0.060, T1=1.0, water_fraction=1.0),
                      myelin=Pool(D=0.0, T2=0.010, T1=0.44, water_fraction=0.4))
g = MyelinatedCylinder(inner_radius=2.0e-6, outer_radius=2.8e-6, orientation=(0, 0, 1),
                       D_intra=1.7e-9, D_extra=2.0e-9, compartments=tissue)
spec = g.spec

print(f"{'id':>2} {'name':8s} {'D (m^2/s)':>10} {'water':>6} {'T2':>6} {'T1':>6}  susceptibility")
for p in spec.pools:
    chi = "-" if p.susceptibility is None else (f"chi_iso {p.susceptibility.chi_iso}, "
                                                f"director {p.susceptibility.director!r}")
    print(f"{p.id:2d} {p.name:8s} {str(p.D):>10} {p.water_fraction:6.2f} "
          f"{'-' if p.T2 is None else format(p.T2, '.3f'):>6} "
          f"{'-' if p.T1 is None else format(p.T1, '.3f'):>6}  {chi}")

print("\nWhat each field decides:")
print("  D              zero freezes the pool -- myelin here is a shell walkers sit in and do not move through")
print("  water_fraction the pool's share of the protons; zero means it holds none, so no walker belongs in it")
print("  T2 / T1        nominal only. A pack stores no relaxation; `pack.nominal` returns these as a Tissue")
print("  susceptibility marks the pool as a FIELD SOURCE. Its chi is a replay knob; the director shapes the")
print("                 field basis, so the walk must record the path channel for a field replay to be possible")

# The seeding follows the water, not the geometry: a pool with no water is not seeded, and a frozen pool is.
print(f"\nseeded pools {spec.seeding.pools} of {[p.id for p in spec.pools]}, weighted by {spec.seeding.weights!r}")
print(f"field-source pools: {[p.name for p in spec.field_source_pools]}")
