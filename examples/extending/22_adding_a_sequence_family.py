"""Adding a sequence the engine has never seen, and the one invariant that keeps it honest.

A sequence family is not a new waveform function. It is an ASSEMBLER: an object that says where the pulses go
and where the encoding blocks sit relative to them, and that is all. Everything else -- the grid, the exact
b-value, the ramps, the refusals, the Pulseq export, both replay routes -- is produced once by `assemble` and
inherited.

The protocol is two methods. `te_min(spans, g)` is the shortest echo time the shape fits in, and `layout` 
returns the RF schedule and, per measurement, where each block is placed. A multi-echo family adds `windows`
and `n_t_of`. Fifteen lines gets a family that everything downstream already understands.

The invariant worth learning is the second half of this rung. The echo time is NOT a label you attach to a
sequence. It is derived from the pulses, and a readout that disagrees with what the schedule actually
refocuses is refused. That is the check that catches the most plausible error in this whole area: moving a
pulse, for a raster or for a timing budget, and leaving the echo time where it was.

Assumes rung 18.
"""
import numpy as np

from dmipy_sim.acquisition.rf import RFEvent, RFSchedule
from dmipy_sim.sequences.assemble import assemble, ramp_of, trapezoid


class AsymmetricSpinEcho:
    """90 at 0, one 180 at ``frac * TE``, so the echo forms at ``2 * frac * TE``: a PGSE pair straddles the 180."""

    def __init__(self, gap, frac=0.5):
        self.gap, self.frac, self.timing = gap, frac, None

    def te_min(self, spans, g):
        gaps = np.array([self.gap(m, g[m]) for m in range(len(spans))])
        if np.any(gaps < 0.0):
            raise ValueError("the two encoding blocks overlap: Delta is shorter than the lobe with its ramp")
        return float(np.max(2.0 * spans + gaps)) / (2.0 * self.frac)

    def layout(self, spans, g, TE, dt, n_t):
        t180 = self.frac * TE
        schedule = RFSchedule([RFEvent(0.0, 90, "Mz→Mxy"), RFEvent(t180, 180, "refocus")])
        placements = [[("end", t180 - self.gap(m, g[m]) / 2.0, 1.0),
                       ("start", t180 + self.gap(m, g[m]) / 2.0, 1.0)] for m in range(len(spans))]
        return schedule, placements


delta, Delta, slew, TE = 0.008, 0.030, 200.0, 0.08
eps = lambda m, g: ramp_of(g, slew)


def build(frac):
    return assemble(
        AsymmetricSpinEcho(gap=lambda m, g: Delta - delta - eps(m, g), frac=frac),
        gradient_directions=[[1.0, 0.0, 0.0]], bvalues=[1e9], TE=TE, n_t=800, slew_rate=slew,
        family="asymmetric_se", q_width=np.array([delta]),
        span=lambda m, g: delta + eps(m, g),
        sample=lambda m, g, dt: trapezoid(delta, eps(m, g), dt),
        build_spec=("asymmetric_se", dict(frac=frac)))


seq = build(0.5)
echo = float(np.asarray(seq.readout)[0]) * float(seq.dt)
print(f"a family the engine has never seen, in fifteen lines:")
print(f"  180 at {0.5*TE*1e3:.1f} ms, echo at {echo*1e3:.1f} ms, b realised "
      f"{float(np.asarray(seq.b())[0])/1e6:.0f} s/mm^2 at {float(np.abs(seq.G).max())*1e3:.1f} mT/m")
print(f"  and it is a ScannerSequence like any other: {seq.G.shape[0]} measurement on {seq.G.shape[1]} samples")

print("\nNow move the pulse and leave the echo time alone, which is the mistake this area invites:")
for frac in (0.4, 0.3):
    try:
        build(frac)
        print(f"  180 at {frac*TE*1e3:.0f} ms: NOT REFUSED")
    except ValueError as e:
        print(f"  180 at {frac*TE*1e3:.0f} ms: {str(e).splitlines()[0]}")

print("\nThe readout is derived from the pulses, not declared beside them. A family whose echoes genuinely")
print("are not at TE says so through `windows`, which is how the refocusing train reports twelve of them;")
print("what it cannot do is claim an echo its own schedule does not form.")
print("\nThat is the argument for extending through the assembler rather than around it. A waveform built by")
print("hand carries no schedule, so nothing checks it, and the sequence that results is wrong in the one way")
print("that still produces a plausible number.")
