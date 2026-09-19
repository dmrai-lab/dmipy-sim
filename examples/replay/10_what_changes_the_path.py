"""Three things change where the water goes, and each needs its own walk.

The boundary of replay is not "substrate versus scanner". It is between what changes the PATH and what
changes the magnetisation carried along it. Exactly three things move a walker:

  the diffusivity   how far it goes per unit time
  the geometry      what it runs into
  the permeability  whether it gets through

Change one and the situation is physically different, so the trajectory that describes it is a different
trajectory. No amount of post-processing recovers it from the first one, and the library does not pretend
otherwise: these are constructor arguments on the substrate, not knobs on a replay.

Assumes rung 09.
"""
import numpy as np

from dmipy_sim import Cylinder, pgse, simulate

seq = pgse([[1.0, 0.0, 0.0]], 0.008, 0.030, bvalues=[1e9], TE=0.05, n_t=600)
N = 4_000

cases = [
    ("the walk as written",        4e-6, 2.0e-9, None),
    ("half the diffusivity",       4e-6, 1.0e-9, None),
    ("half the radius",            2e-6, 2.0e-9, None),
    ("a permeable membrane",       4e-6, 2.0e-9, 2e-5),
]
base = None
for name, R, D, kappa in cases:
    g = Cylinder(radius=R, orientation=(0, 0, 1), permeability=kappa)
    S = float(simulate(N, D, seq, g, seed=0, require_gpu=False)[0])
    base = S if base is None else base
    print(f"{name:24s} S = {S:.4f}   {'':8s}" if kappa is None and R == 4e-6 and D == 2e-9
          else f"{name:24s} S = {S:.4f}   changed by {abs(S-base):.4f}")

print("\nEach line above is its OWN walk, and has to be: the walkers went somewhere else.")
print("Compare that with the next rung, where one walk answers every question without moving a walker.")
