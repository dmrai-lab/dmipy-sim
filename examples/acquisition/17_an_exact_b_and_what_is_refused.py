"""Asking for a b-value is asking for a waveform, and the builder either realises it or says why not.

A b-value is not a parameter of a sequence. It is a functional of the gradient waveform, and for a given
timing and a given scanner there is one amplitude that produces it -- or none. The builders solve for that
amplitude on the scanner's own raster and with its own ramps, so the b that comes back is the b of the
waveform that was actually built, not the b of an idealised rectangle.

That distinction is the whole reason a builder can refuse. A trapezoid that must ramp up and down inside its
own duration has a largest b it can reach; a pulse that must sit on a raster cannot sit between two samples;
an echo time shorter than the pulses and lobes it has to contain does not exist. Each of those is a refusal
with a reason, and each of them is a silently wrong number in a simulator that just multiplies
gamma^2 G^2 delta^2 (Delta - delta/3) and moves on.

Assumes part I; no walk is needed, because nothing here has met water yet.
"""
import numpy as np

from dmipy_sim import pgse
from dmipy_sim.acquisition.scanners import ScannerLimits

GAMMA = 267.513e6
asked = 1e9
seq = pgse([[1.0, 0.0, 0.0]], 0.008, 0.030, bvalues=[asked], TE=0.05, n_t=600)
realised = float(np.asarray(seq.b())[0])
peak = float(np.abs(seq.G).max())
ideal = np.sqrt(asked / (GAMMA**2 * 0.008**2 * (0.030 - 0.008 / 3)))
print(f"asked for b = {asked/1e6:.0f} s/mm^2, realised {realised/1e6:.1f} s/mm^2 at a peak of {peak*1e3:.1f} mT/m")
print(f"  the ideal-rectangle formula would need {ideal*1e3:.1f} mT/m; the difference is the ramps")

print("\nwhat a scanner can deliver, from the catalogue rather than from a number in this file:")
for name in ("swoop", "prisma", "connectom"):
    s = ScannerLimits.of(name)
    g_max, slew = s.gradient_limits
    verdict = "can run it" if peak <= g_max else f"cannot: needs {peak*1e3:.1f} mT/m"
    print(f"  {s.name:32s} {s.field_T:5.3f} T  {g_max*1e3:6.1f} mT/m  {slew:6.1f} T/m/s   {verdict}")

print("\nrefusals, each of which would otherwise be a plausible wrong number:")
cases = (
    ("a b this timing cannot reach", dict(delta=0.008, Delta=0.030, bvalues=[5e11], TE=0.05)),
    ("an echo time shorter than its lobes", dict(delta=0.008, Delta=0.030, bvalues=[1e9], TE=0.02)),
    ("a diffusion time shorter than the pulse", dict(delta=0.030, Delta=0.008, bvalues=[1e9], TE=0.05)),
)
for label, kw in cases:
    try:
        pgse([[1.0, 0, 0]], kw["delta"], kw["Delta"], bvalues=kw["bvalues"], TE=kw["TE"], n_t=600)
        print(f"  {label:40s} NOT REFUSED")
    except (ValueError, RuntimeError) as e:
        print(f"  {label:40s} {str(e).splitlines()[0][:92]}")

print("\nThe sequence that survives carries its own waveform, so everything downstream -- the replay, the")
print("certificate, the Pulseq export -- reads the same object and cannot disagree about what was run.")
