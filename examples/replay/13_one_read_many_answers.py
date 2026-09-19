"""A study: every tissue and every field the question needs, in one pass over the pack.

The previous rung applied one tissue at a time, which is the right way to see what a knob does and the wrong
way to run an experiment. A study states the whole grid up front -- a protocol, a list of tissues, a list of
fields, and which pairs of them are wanted -- and the pack is read once for all of it.

That matters because reading is the cost. The arithmetic per pair is a contraction over stored coefficients
and is cheap; getting those coefficients off disk, or off the network, is not. A study is therefore not a
convenience wrapper around a loop: it is the object that lets the reader decide what to fetch by looking at
the most demanding pair in the whole grid, and fetch it once.

The two lists are not the same kind of thing, and the study keeps them apart: the tissue is the sample and
the scanner is the machine. A susceptibility is a property of myelin, the field it sits in is a property of
the magnet, and the phase is their product -- so they are two axes of a grid, not one merged setting.

The grid is the cross product by default, and here it must not be: a field asked of a tissue that does not
respond is refused rather than quietly returning the unshifted number, so the pairs that mean something are
named with `pairs=`. A study of four tissues and two magnets is five questions, not eight.

The pairs also decide which channels are touched. A study whose tissues all have `rho = None` never reads the
contact channel; add one relaxivity and every pair in the study rides along on the rows that tissue needed.
`needs_relaxation`, `needs_contact` and `needs_field` are the study answering that question about itself
before anything is read.

Assumes rung 11.
"""
import numpy as np

import dmipy_sim as d
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.study import Acquisition, Protocol, Study
from dmipy_sim.spec.tissue import Tissue

D, TE = 2e-9, 0.012

# A sheathed axon, so that a susceptibility has something to be a property OF: myelin is the field source.
pore = d.PackedMyelinatedCylinders([1.0e-6], 0.7, [[0.0, 0.0]], 30e-6, N_max=2, D_intra=D, D_extra=D)
walk = d.simulate_trajectories(1_500, D, pore, TE, 4e-4, seed=0, require_gpu=False)
pack = build_replay_pack(walk, id="cookbook/sheath", license="CC-BY-4.0", citation="the cookbook",
                         K=32, susc_path_K=32)

across_and_along = d.pgse([[1.0, 0, 0], [0, 0, 1.0]], 0.003, 0.006, bvalues=[1e9, 1e9], TE=TE, n_t=300)
protocol = Protocol([Acquisition(across_and_along, name="across and along")])

tissues = [None, Tissue(T2=0.08), Tissue(T2=0.08, rho=1e-5, D=D), Tissue(T2=0.08, chi_iso=-9.0e-6)]
labels = ["bare diffusion", "T2 80 ms", "T2 80 ms, relaxivity 10 um/s", "T2 80 ms, myelin susceptibility"]
scanners = [None, 3.0]
# Only the susceptible tissue has anything to do with a magnet, so only it is paired with one.
pairs = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)]
study = Study(protocol, tissues=tissues, scanners=scanners, pairs=pairs)

print(f"the study asks for {len(study)} pairs of {protocol.n_meas} measurements, chosen from the "
      f"{len(tissues)} x {len(scanners)} grid")
print(f"  relaxation needed: {study.needs_relaxation}   contact: {study.needs_contact}   field: {study.needs_field}")

S = np.asarray(pack.study(study))                       # (pairs, measurements), one read
print(f"\n{'tissue':34s} {'scanner':>9s} {'across':>8} {'along':>8}")
for k, row in enumerate(S):
    i, j = study.pairs[k]
    print(f"{labels[i]:34s} {('-' if scanners[j] is None else f'{scanners[j]:.0f} T'):>9s} {row[0]:8.4f} {row[1]:8.4f}")

one_at_a_time = np.array([np.asarray(pack.replay(across_and_along, tissue=t, scanner=s))
                          for t, s in (study.resolved(k) for k in range(len(study)))])
print(f"\nSame numbers one at a time, max difference {np.abs(S - one_at_a_time).max():.2e}: the study is not a")
print("different calculation, it is the same one arranged so the pack is read once.")

print("\nOnly the last tissue has a chi, so only it was asked at a field -- and its two rows are the same walk,")
print("the same susceptibility, and a different magnet. A field is the one tier that needs a value from BOTH")
print("lists before it does anything at all, which is why the grid is not rectangular.")
print("The two rows barely differ because this protocol is a spin echo, and a 180 is precisely what puts a")
print("static field back; rung 25 takes that apart on a sequence that keeps it.")

try:
    pack.study(Study(protocol, tissues=tissues, scanners=scanners))
    print("\nthe full cross product: NOT REFUSED")
except ValueError as e:
    print(f"\nasking for the full {len(tissues)} x {len(scanners)} cross product instead:\n  {e}")

print("\nWhat the record says it did:")
meta = study.to_meta()
for k, pair in enumerate(meta["pairs"]):
    t = pair["tissue"]
    print(f"  pair {k}: scanner {pair['scanner']}, tissue "
          f"{'none' if t is None else {kk: vv for kk, vv in t.items() if vv is not None}}")
