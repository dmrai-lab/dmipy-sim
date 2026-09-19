"""The spins do this anyway. The scanner only asks.

    "Diffusion MRI altered this situation by encoding a physical process that EXISTS INDEPENDENTLY OF THE
    SCANNER: the thermally driven displacement of water molecules."
        -- Le Bihan, 40 years of Diffusion MRI in the Brain, Imaging Neuroscience 2026,
           doi 10.1162/IMAG.a.1365

That sentence opens the field's own history, and replay is what follows from taking it literally in a
simulator. If the displacement exists independently of the scanner, then a simulation of it does too, and an
acquisition is something you ask of it afterwards rather than something you must decide before starting.

Water in a substrate diffuses, meets walls, crosses membranes where they are permeable, accumulates time in
contact with surfaces, and sits in whatever field the tissue's own susceptibility produces. None of that
depends on a scanner being switched on. A scanner applies gradients and pulses and reads out; at any field
and gradient a human scanner can deliver, it does not move the water.

So the trajectory is PRIOR to the measurement, and an acquisition is a question asked of a record that
already exists. That is not an optimisation and not an approximation -- it is a statement about what depends
on what, and it is why one walk can answer many acquisitions.

This rung demonstrates it rather than asserting it: one walk is replayed under three different acquisitions,
and each answer is compared against an independent forward simulation of the same substrate with the same
seed, which is what the engine would have done had it been told the sequence in advance.

Assumes part I.
"""
import numpy as np

from dmipy_sim import Cylinder, ogse, pgse, simulate, simulate_trajectories
from dmipy_sim.replay.bank import build_replay_pack

pore = Cylinder(radius=4e-6, orientation=(0.0, 0.0, 1.0))
D, N, T = 2e-9, 4_000, 0.06

# The walk. It is told the substrate, the diffusivity and how long to go -- and nothing about an acquisition.
walk = simulate_trajectories(N, D, pore, T_max=T, dt_save=2e-4, seed=0, require_gpu=False)
pack = build_replay_pack(walk, id="cookbook/pore", license="CC-BY-4.0", citation="the cookbook", K=32)
print(f"one walk: {pack.n_walkers} walkers, {pack.n_t} saves, and no acquisition has been mentioned yet")

# Three questions, asked afterwards, of the same record.
asked = {
    "PGSE across the pore": pgse([[1.0, 0, 0]], 0.008, 0.030, bvalues=[1e9], TE=0.05, n_t=600),
    "PGSE along it":        pgse([[0.0, 0, 1]], 0.008, 0.030, bvalues=[1e9], TE=0.05, n_t=600),
    "OGSE, 50 Hz, across":  ogse([[1.0, 0, 0]], 50.0, 0.020, bvalues=[1e9], TE=0.05, n_t=600),
}
print(f"\n{'acquisition':24s} {'replayed':>10} {'simulated':>10} {'difference':>11}")
for name, seq in asked.items():
    replayed = float(np.asarray(pack.replay(seq))[0])
    # the same substrate, the same seed, but the engine told the sequence in advance
    forward = float(simulate(N, D, seq, pore, seed=0, require_gpu=False)[0])
    print(f"{name:24s} {replayed:10.4f} {forward:10.4f} {abs(replayed-forward):11.5f}")

print(f"\nThe Monte-Carlo floor of this walk is {pack.meta['fidelity']['floor_max']:.4f}, and the codec error")
print(f"is {pack.meta['fidelity']['err_max']:.2e}. The differences above sit at that level, which is the")
print("statement being made: asking afterwards costs the compression, not the physics.")
