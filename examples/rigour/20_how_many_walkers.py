"""Sizing a run: how precise the answer is, and how many walkers that costs.

A Monte-Carlo signal has a floor. Below it the number is noise, and no care downstream recovers anything. So
the first question about a simulated measurement is not whether it is right but how precise it is, and a pack
answers it with two numbers that are often confused.

`noise_floor` is one over the square root of the walker count: the analytic expectation, free, and the same
for every substrate and every sequence. `floor_max` is measured -- the ensemble is split in half and the two
halves' signals are compared over a battery of acquisitions, with no reference and no analytic solution
needed -- and it is the WORST disagreement in that battery.

The measured worst sits above the analytic expectation, and the ratio between them is the number worth
carrying. It is what the substrate and the battery do to the ideal law: some waveforms are far more sensitive
to which walkers you happened to draw than others. Size a run from the law, then quote what was measured.

Assumes rung 16.
"""
import numpy as np

from dmipy_sim import Cylinder, simulate_trajectories
from dmipy_sim.replay.bank import build_replay_pack

pore = Cylinder(radius=4e-6, orientation=(0, 0, 1))
print(f"{'walkers':>9} {'1/sqrt(N)':>11} {'measured worst':>15} {'ratio':>7}")
ratios = []
for N in (500, 2_000, 8_000):
    walk = simulate_trajectories(N, 2e-9, pore, T_max=0.03, dt_save=2e-4, seed=0, require_gpu=False)
    pack = build_replay_pack(walk, id=f"cookbook/n{N}", license="CC-BY-4.0", citation="the cookbook", K=32)
    law, worst = float(pack.fidelity["noise_floor"]), float(pack.fidelity["floor_max"])
    ratios.append(worst / law)
    print(f"{N:9d} {law:11.5f} {worst:15.5f} {ratios[-1]:7.2f}")

print(f"\nThe measured floor runs {min(ratios):.1f} to {max(ratios):.1f} times the ideal law here. That margin is")
print("what a battery over a real substrate costs, and it is why the certificate reports a measurement")
print("rather than the formula: the formula is the same for every pack ever made.")

print(f"\nSizing from the law, then applying the margin ({np.mean(ratios):.1f} times):")
for target in (5e-3, 1e-3):
    n_law = (1.0 / target) ** 2
    print(f"  a measured floor of {target:.4f} needs about {n_law * np.mean(ratios)**2:,.0f} walkers "
          f"({n_law:,.0f} if the law held exactly)")

print("\nThe production CACTUS walk behind the low-field experiment ran 119,999 walkers and certified at")
print("0.0048, where the law alone predicts 0.0029: a ratio of 1.7, inside the range above. Sizing a run")
print("on the formula and reporting the measurement is the practice this supports.")
