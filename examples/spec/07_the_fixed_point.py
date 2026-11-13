"""A geometry and its spec are the same thing, and the round trip proves it.

`geometry.spec` writes out what a constructor left implicit. `spec.geometry_from_spec` builds the geometry
back. The two are a fixed point: the rebuilt geometry walks to the same positions, bit for bit, from the same
seed. That is what lets a spec be the only thing a result needs to carry -- there is no information in the
object that the file does not have.

Assumes rung 01.
"""
import numpy as np

from dmipy_sim import PackedCylinders, pack_cylinders, pgse, simulate
from dmipy_sim.spec import geometry_from_spec, load_spec

radii = np.full(3, 2e-6)
centers, L, _ = pack_cylinders(radii, target_vf=0.25, seed=0)
original = PackedCylinders(radii, centers, L, surface_relaxivity_t2=1e-6)

spec = original.spec
spec.save("/tmp/packed.sub.json")
rebuilt = geometry_from_spec(load_spec("/tmp/packed.sub.json"))
print(f"rebuilt a {type(rebuilt).__name__} from the file alone")

seq = pgse([[1.0, 0.0, 0.0]], 0.005, 0.015, bvalues=[1e9], n_t=200)
a = simulate(2_000, 2e-9, seq, original, seed=0, require_gpu=False, return_positions=True)[1]
b = simulate(2_000, 2e-9, seq, rebuilt, seed=0, require_gpu=False, return_positions=True)[1]
print(f"same walk: max |difference| in the final positions = {np.abs(np.asarray(a) - np.asarray(b)).max():.3e} m")

# Every driver takes the spec directly, so the geometry object is never the thing you have to keep.
S = simulate(2_000, 2e-9, seq, "/tmp/packed.sub.json", seed=0, require_gpu=False)
print(f"and a driver takes the FILE as its substrate: S = {float(S[0]):.4f}")
