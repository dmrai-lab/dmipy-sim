"""The domain and what a walker meets at its edge.

The domain is a box and a boundary condition per axis, and the three kinds mean different things:

  periodic   the cell tiles space. A walker that leaves one face returns at the opposite one, and the
             geometry is queried at the wrapped position -- but the position the walk RECORDS keeps going,
             because the gradient phase integrates the real path, not the folded one.
  reflect    the face is a wall. Nothing leaves.
  open       the face is nothing. A walker crosses it and keeps diffusing, which is what an isolated object
             in free water needs.

Choosing wrongly changes the answer rather than raising: a packing declared open leaks its extra-cellular
walkers away from the substrate, and one declared reflecting confines them in a box that is not there.

Assumes rung 03.
"""
import numpy as np

from dmipy_sim import Cylinder, PackedCylinders, pack_cylinders, pgse, simulate

radii = np.full(4, 2e-6)
centers, L, _ = pack_cylinders(radii, target_vf=0.3, seed=0)
packed = PackedCylinders(radii, centers, L)

for name, spec in (("isolated cylinder", Cylinder(5e-6, (0, 0, 1)).spec), ("periodic packing", packed.spec)):
    d = spec.domain
    side = np.subtract(d.box_max, d.box_min)
    print(f"{name:20s} box {side[0]*1e6:5.1f} x {side[1]*1e6:5.1f} x {side[2]*1e6:5.1f} um   boundary {d.boundary}")

# The recorded path is continuous even where the cell is periodic. Over 20 ms a free walker travels far
# further than this 13 um cell, and the walk keeps that displacement rather than folding it.
seq = pgse([[1.0, 0.0, 0.0]], 0.005, 0.015, bvalues=[1e9], n_t=200)
_, pos = simulate(2_000, 2e-9, seq, packed, seed=0, require_gpu=False, return_positions=True)
travel = np.abs(pos[:, 0] - pos[:, 0].mean())
print(f"\ncell side {L*1e6:.1f} um; largest recorded |x| excursion {travel.max()*1e6:.1f} um")
print("the position is not wrapped, so the phase is integrated along the real path; only the geometry query wraps")
