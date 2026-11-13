"""A refocusing train, and why its crusher is physics rather than housekeeping.

A train of refocusing pulses does not produce one echo per pulse. Each pulse splits the magnetisation into
coherence pathways, and what arrives at an echo is a sum over every pathway that happens to be rephased
there. At a nominal 180 those pathways are degenerate and the sum is exactly the spin echo; at any other
flip angle they are not, and the echo amplitude becomes a property of the whole pathway tree.

A crusher is what makes the sum finite. It dephases everything across a voxel except the pathways whose
accumulated winding cancels, so the train selects instead of accumulating. Leave it out and the pathways
recombine, the echoes stop depending on the flip angle, and a simulation of a reduced-flip train returns
approximately the answer for a perfect one -- which is the wrong answer, arrived at plausibly.

Two independent routes say what the train does. `epg` enumerates the pathways and gives the amplitude in
closed form for the cases where one exists. The Bloch replay propagates each walker's magnetisation vector
through the actual pulses and the actual crusher windings. They are different calculations of the same
thing, which is what makes the agreement worth printing.

Assumes rung 17.
"""
import numpy as np

from dmipy_sim import sequences
from dmipy_sim.acquisition import epg

n_echoes, TE_echo = 6, 0.010

print(f"{'refocusing flip':>16s} {'closed-form amplitude':>22s}   {'what the scalar route does'}")
for beta in (180.0, 150.0, 120.0, 90.0):
    seq = sequences.cpmg(n_echoes, TE_echo, beta_deg=beta, n_t_per_echo=40)
    try:
        print(f"{beta:13.0f} deg {float(np.real(epg.pathway_weight(seq))):22.4f}   the pathways are degenerate here")
    except (ValueError, NotImplementedError):
        print(f"{beta:13.0f} deg {'refused':>22s}   no closed form: this needs the Bloch route")

print("\nThe refusal is the point. A six-echo train below 180 degrees has no expressible amplitude, so the")
print("scalar replay declines to give one instead of returning the perfect-train number with a straight face.")

print("\nA single refocused echo is the case with a closed form, sin^2(beta/2), and the enumeration")
print("reproduces it rather than being told it:")
for beta in (180.0, 150.0, 120.0, 90.0):
    paths = epg.enumerate_pathways(epg.cpmg_schedule(1, beta_deg=beta, TE=TE_echo), threshold=1e-6)
    amp = abs(sum(p.eta for p in paths if p.readout_idx == 0))   # the pathways add coherently, not in magnitude
    print(f"  beta {beta:5.0f} deg: enumerated {amp:8.5f}   sin^2(beta/2) {np.sin(np.radians(beta)/2)**2:8.5f}")

print("\nThe stimulated echo is the other closed form the enumeration has to reproduce, 0.5 sin a1 sin a2 sin a3:")
for a in (90.0, 60.0, 45.0):
    # spoil=4: a crusher the schedule cannot rewind, so the stimulated echo is the only pathway left
    paths = epg.enumerate_pathways(epg.ste_schedule(a, a, a, delta=0.01, TM=0.02, spoil=4), threshold=1e-6)
    amp = abs(sum(p.eta for p in paths if p.readout_idx == 0))
    print(f"  three {a:4.0f}-degree pulses: enumerated {amp:8.5f}   formula "
          f"{epg.ste_amplitude(a, a, a):8.5f}")

print("\nWhat a builder declares. `splice` is the diffusion-prepared train an ultra-low-field scanner plays,")
print("and it is the first builder that has to state a crusher, because without one its reduced-flip echoes")
print("would be indistinguishable from a perfect train's:")
sp = sequences.splice([[1.0, 0, 0]], 0.035, 0.042, 4, 0.010, bvalues=[0.945e9], TE_prep=0.084,
                      beta_deg=120.0, n_t_per_echo=20, slew_rate=22.0)
crusher = sp.crusher
print(f"  {len(crusher['windows_s'])} crusher windows of {crusher['n_cycles']:.0f} winding cycles each, one "
      f"straddling every refocusing pulse")
print(f"  b realised {float(np.asarray(sp.b())[0])/1e6:.0f} s/mm^2 at a peak of {float(np.abs(sp.G).max())*1e3:.1f} mT/m")
print(f"  echoes at {', '.join(f'{t*1e3:.0f}' for t in np.asarray(sp.readout) * float(sp.dt))} ms")
