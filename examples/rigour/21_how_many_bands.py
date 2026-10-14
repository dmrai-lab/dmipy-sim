"""Choosing the band count: compress until the codec's error disappears under the walk's own noise, then stop.

A pack stores each path as K sine bands. K is the only thing that sets its size, and it is the one parameter
a producer has to choose. Choosing it by taste gives either a file that is larger than it needs to be or one
that quietly cannot answer the acquisition it will be asked.

There is a criterion that removes the taste. The walk already has a Monte-Carlo floor (rung 20), and a codec
error below that floor is invisible in any use of the walk: it cannot be distinguished from having drawn
different walkers. So raise K until the error sits under the floor and stop, because beyond that point the
extra bands buy accuracy nobody can measure.

The band count is also a frequency. K bands over a walk of duration T resolve up to K/(2T), so the criterion
has a physical reading: the pack must resolve the fastest thing the acquisition asks about. A sequence with
rapid oscillating gradients needs more bands than a single diffusion time at the same precision, which is why
the certificate is measured over a battery rather than over one waveform.

Assumes rung 20.
"""
import numpy as np

from dmipy_sim import Cylinder, simulate_trajectories
from dmipy_sim.replay.bank import build_replay_pack

pore = Cylinder(radius=4e-6, orientation=(0, 0, 1))
walk = simulate_trajectories(4_000, 2e-9, pore, T_max=0.03, dt_save=2e-4, seed=0, require_gpu=False)

print(f"{'K':>5} {'Hz resolved':>12} {'codec error':>12} {'floor':>9} {'error / floor':>14}  verdict")
for K in (4, 8, 16, 32, 64):
    pack = build_replay_pack(walk, id=f"cookbook/k{K}", license="CC-BY-4.0", citation="the cookbook", K=K)
    err, floor = float(pack.fidelity["err_max"]), float(pack.fidelity["floor_max"])
    verdict = "the codec is the error" if err > floor else "the walk is the error"
    print(f"{K:5d} {pack.temporal_bandwidth_hz:12.0f} {err:12.2e} {floor:9.5f} {err/floor:14.3f}  {verdict}")

print("\nOnce the ratio drops below one the compression has stopped mattering, and every further doubling of K")
print("doubles the file for accuracy that the walk's own noise already hides. That is the whole rule.")
print("\nThe two numbers move for different reasons, which is why both are in the certificate: the floor")
print("falls only by walking more walkers, the codec error only by storing more bands. A pack that is short")
print("of the first cannot be rescued by the second.")
