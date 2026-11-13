"""A field is a tissue number times a machine number, and neither one alone is a field.

The previous rungs applied every knob the sample owns. The scanner owns one too, and it is the only
replay input that is not a property of the water or of the question asked about it: the static field the
magnet holds. It enters as `scanner=`, beside `tissue=`, and the two are multiplied.

That multiplication is why neither is a field on its own. A susceptibility with no magnet to polarise it
shifts nothing, and a magnet over water that does not respond shifts nothing either. The walk is prior to
both: it recorded the field BASIS -- the shape of the perturbation the geometry would make, per unit
susceptibility, per unit field -- so the same stored path answers every (chi, B0) pair by arithmetic.

`FieldGrid` says it exactly: "No susceptibility value lives here: the grid is the substrate's shape, the
field is a replay knob."

The sequence still decides how much of it survives to the echo, which is the second half of this rung: a
gradient echo keeps the static dephasing, a spin echo refocuses it and leaves only what diffusion made
irreversible. Same walk, same tissue, same field -- a different question.

Assumes rung 13. The knob table is `docs/replay-guide/README.md`.
"""
import numpy as np

import dmipy_sim as d
from dmipy_sim.acquisition.scanners import ScannerLimits
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec.tissue import Tissue

D, TE = 2e-9, 0.012
DIR, BVAL, DELTA, BIG_DELTA = [[1.0, 0.0, 0.0]], [1e9], 0.003, 0.006

# A sheathed axon: myelin is the field source, so this substrate HAS a susceptibility to declare.
# `susc_path_K` records the field along the path (C3); the basis is derived from the geometry alone.
axon = d.PackedMyelinatedCylinders([1.0e-6], 0.7, [[0.0, 0.0]], 30e-6, N_max=2, D_intra=D, D_extra=D)
walk = d.simulate_trajectories(1_500, D, axon, TE, 4e-4, seed=0, require_gpu=False)
pack = build_replay_pack(walk, id="cookbook/sheath", license="CC-BY-4.0", citation="the cookbook",
                         K=32, susc_path_K=32)
print(f"channels this walk recorded: {sorted(pack.meta['compression']['channels'])}")

myelin = Tissue(chi_iso=-9.0e-6)

# Neither half is a field by itself.
fid = d.gre(TE, gradient_directions=DIR, bvalues=BVAL, delta=DELTA, Delta=BIG_DELTA, n_t=300)
bare = float(np.abs(np.asarray(pack.replay(fid))[0]))
chi_only = float(np.abs(np.asarray(pack.replay(fid, tissue=myelin))[0]))
print(f"\na susceptibility with no magnet: {chi_only:.4f} against bare {bare:.4f} -- the same number")

# `scanner=` is an object as readily as a number: the catalogue carries the field of a real magnet.
swoop = ScannerLimits.of("swoop")
print(f"{swoop.name} from the catalogue: {swoop.field_T} T")

spin_echo = d.pgse(DIR, DELTA, BIG_DELTA, bvalues=BVAL, TE=TE, n_t=300)
ref = [float(np.abs(np.asarray(pack.replay(s))[0])) for s in (fid, spin_echo)]
print(f"\nthe same b = {BVAL[0]/1e9:.0f} e9 s/m2, with the field switched on. The columns are the signal as a")
print("fraction of its own no-field value, so what is left is the field's doing alone.")
print(f"\n{'scanner':22s} {'gradient echo':>14s} {'spin echo':>11s}")
for label, scanner in ((swoop.name, swoop), ("1.5 T", 1.5), ("3 T", 3.0), ("7 T", 7.0)):
    S = [float(np.abs(np.asarray(pack.replay(s, tissue=myelin, scanner=scanner))[0])) for s in (fid, spin_echo)]
    print(f"{label:22s} {S[0]/ref[0]:14.4f} {S[1]/ref[1]:11.4f}")
print("\nthe gradient echo loses a tenth of its signal by 7 T; the spin echo holds its own to a part in a")
print("thousand, because a static field is exactly what a 180 puts back. What little it does lose is the")
print("part diffusion made irreversible -- the walkers did not stay where the field could be undone.")
print("At 64 mT there is barely anything to refocus in the first place, which is the low-field bargain.")

# What is refused, and the one case that is legitimately zero.
for label, pk, tissue, scanner in (
        ("a field with no susceptibility", pack, Tissue(T2=0.08), 3.0),
        ("a susceptible substrate whose pack carries no C3",
         build_replay_pack(walk, id="cookbook/sheath-nofield", license="CC-BY-4.0",
                           citation="the cookbook", K=32, field=False), myelin, 3.0)):
    try:
        pk.replay(fid, tissue=tissue, scanner=scanner)
        print(f"\n{label}: NOT REFUSED")
    except ValueError as e:
        print(f"\n{label}:\n  {str(e).split(';')[0]}")

plain = build_replay_pack(
    d.simulate_trajectories(500, D, d.Cylinder(radius=4e-6, orientation=(0, 0, 1)), TE, 4e-4,
                            seed=0, require_gpu=False),
    id="cookbook/pore", license="CC-BY-4.0", citation="the cookbook", K=32)
S = float(np.abs(np.asarray(plain.replay(fid, tissue=myelin, scanner=3.0))[0]))
print(f"\na substrate with no field source at all is not refused: S = {S:.4f}. Its field is zero because")
print("the spec declares no susceptible pool, and zero is the right answer rather than a missing tier.")
