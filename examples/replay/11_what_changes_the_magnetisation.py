"""What the water carries, as opposed to where it goes -- and why one walk answers all of it.

Relaxation, surface relaxivity and susceptibility are properties of the substrate, not of the scanner. They
still do not move a walker: they change the magnetisation it carries along a path it would have taken anyway.
So the walk records the EXPOSURE -- which pool a walker was in at each save, how long it spent against a
wall, what field it passed through -- and the magnitude is applied afterwards.

That is the whole trick, and it is why a tissue is a replay knob despite being a property of the tissue: one
walk, then every T2, every relaxivity, every field, at the cost of an arithmetic pass over stored numbers.

Assumes rung 10.
"""
import numpy as np

from dmipy_sim import Cylinder, pgse, simulate_trajectories
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec.tissue import Tissue

pore = Cylinder(radius=4e-6, orientation=(0, 0, 1))
walk = simulate_trajectories(4_000, 2e-9, pore, T_max=0.06, dt_save=2e-4, seed=0, require_gpu=False)
pack = build_replay_pack(walk, id="cookbook/pore", license="CC-BY-4.0", citation="the cookbook", K=32)
seq = pgse([[1.0, 0.0, 0.0]], 0.008, 0.030, bvalues=[1e9], TE=0.05, n_t=600)

print(f"channels this walk recorded: {sorted(pack.meta.get('compression', {}).get('channels', {}))}")
print(f"\n{'tissue applied at replay':34s} {'S':>8}")
print(f"{'none (bare diffusion)':34s} {float(np.asarray(pack.replay(seq))[0]):8.4f}")
for T2 in (0.08, 0.04):
    S = float(np.asarray(pack.replay(seq, tissue=Tissue(T2=T2)))[0])
    print(f"{f'T2 = {T2*1e3:.0f} ms':34s} {S:8.4f}")
for rho in (1e-6, 1e-5):
    S = float(np.asarray(pack.replay(seq, tissue=Tissue(T2=0.08, rho=rho, D=2e-9)))[0])
    print(f"{f'T2 = 80 ms, relaxivity {rho*1e6:.0f} um/s':34s} {S:8.4f}")

print("\nOne walk produced every line. The walkers are identical in all of them; what differs is the weight")
print("each one carries to the readout. A field is the same kind of knob -- the walk stores the field along")
print("the path and the susceptibility and B0 multiply it at replay -- shown in the susceptibility rung.")
print("\nWhat a replay refuses: a tier the walk did not record. Asking for a relaxivity on a walk with no")
print("contact channel raises rather than returning the unrelaxed number.")
