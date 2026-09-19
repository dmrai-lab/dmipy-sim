"""The sequence leaves the simulator as a file a scanner can play, and comes back as the same object.

A simulated acquisition is only worth something if it is the acquisition that will be run. Pulseq is the
interchange format that makes that checkable: a `.seq` file is what the scanner executes, so exporting to it
and reading it back is a round trip through the same representation the hardware sees.

The round trip is not decorative. It is what catches a sequence that exists only inside the simulator --
gradients between raster samples, an RF pulse the system cannot deliver, a b-value that depended on an
idealisation. What survives the export is what a scanner could play; what comes back carries the timing the
file states rather than the timing the builder intended, and the two agreeing is the check.

Assumes rung 17.
"""
import numpy as np

from dmipy_sim import pgse, sequences

seq = pgse([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], 0.012, 0.030, bvalues=[1e9, 2e9], TE=0.06, n_t=800,
           slew_rate=120.0)
print(f"built: {np.asarray(seq.G).shape[0]} measurements, b = "
      f"{', '.join(f'{v/1e6:.0f}' for v in np.asarray(seq.b()))} s/mm^2, "
      f"peak {float(np.abs(seq.G).max())*1e3:.1f} mT/m")

print(f"\nsystems the exporter knows: {', '.join(sorted(sequences.PULSEQ_SYSTEMS))}")

# The first thing the export checks is that the sequence lives on the scanner's raster. A simulation grid
# chosen for the physics is not automatically a grid the hardware can play, and the exporter says which are.
try:
    sequences.to_pulseq(seq, 0, system=sequences.make_system("prisma"), filename="rejected.seq")
except ValueError as e:
    print(f"\nrefused, and told what would work:\n  {str(e).splitlines()[0]}")

seq = pgse([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], 0.012, 0.030, bvalues=[1e9, 2e9], TE=0.06, n_t=1001,
           slew_rate=120.0)
print(f"\nrebuilt on the raster: dt = {float(seq.dt)*1e6:.1f} us, b = "
      f"{', '.join(f'{v/1e6:.0f}' for v in np.asarray(seq.b()))} s/mm^2")
# A .seq file holds one measurement, which is what a scanner plays; a protocol is a file per direction.
system = sequences.make_system("connectom")     # the peak above is past a Prisma's 80 mT/m
print(f"\n{'measurement':>12s} {'b built':>10s} {'b from the file':>16s} {'raster':>10s}")
for m in range(np.asarray(seq.G).shape[0]):
    name = f"cookbook_{m}.seq"
    sequences.to_pulseq(seq, m, system=system, filename=name)
    back = sequences.from_pulseq(name)
    b_built = float(np.asarray(seq.b())[m]) / 1e6
    b_file = float(np.asarray(back.b()).reshape(-1)[0]) / 1e6
    print(f"{m:12d} {b_built:10.1f} {b_file:16.1f} {float(back.dt)*1e6:9.1f} us"
          f"   ({abs(b_file - b_built) / b_built * 100:.2f} % apart)")

print("\nThe b-values come back within a fifth of a percent, and the residual is the point rather than an")
print("error: the file holds the waveform as blocks on the hardware raster, so what returns is the b the")
print("scanner would actually deliver. An export that wrote delta, Delta and b instead would round-trip")
print("exactly and tell you nothing, because it would be comparing the request with itself.")
