"""A study: every tissue and every field the question needs, in one pass over the pack.

The previous rung applied one tissue at a time, which is the right way to see what a knob does and the wrong
way to run an experiment. A study states the whole grid up front -- a protocol, a list of tissues, a list of
fields, and which pairs of them are wanted -- and the pack is read once for all of it.

That matters because reading is the cost. The arithmetic per pair is a contraction over stored coefficients
and is cheap; getting those coefficients off disk, or off the network, is not. A study is therefore not a
convenience wrapper around a loop: it is the object that lets the reader decide what to fetch by looking at
the most demanding pair in the whole grid, and fetch it once.

The pairs also decide which channels are touched. A study whose tissues all have `rho = None` never reads the
contact channel; add one relaxivity and every pair in the study rides along on the rows that tissue needed.
`needs_relaxation`, `needs_contact` and `needs_field` are the study answering that question about itself
before anything is read.

Assumes rung 11.
"""
import numpy as np

from dmipy_sim import Cylinder, pgse, simulate_trajectories
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.study import Acquisition, Protocol, Study
from dmipy_sim.spec.tissue import Tissue

pore = Cylinder(radius=4e-6, orientation=(0, 0, 1))
walk = simulate_trajectories(4_000, 2e-9, pore, T_max=0.06, dt_save=2e-4, seed=0, require_gpu=False)
pack = build_replay_pack(walk, id="cookbook/pore", license="CC-BY-4.0", citation="the cookbook", K=32)

across_and_along = pgse([[1.0, 0, 0], [0, 0, 1.0]], 0.008, 0.030, bvalues=[1e9, 1e9], TE=0.05, n_t=600)
protocol = Protocol([Acquisition(across_and_along, name="across and along")])

tissues = [None, Tissue(T2=0.08), Tissue(T2=0.04), Tissue(T2=0.08, rho=1e-5, D=2e-9)]
labels = ["bare diffusion", "T2 80 ms", "T2 40 ms", "T2 80 ms, relaxivity 10 um/s"]
study = Study(protocol, tissues=tissues)

print(f"the study asks for {len(study)} pairs of {protocol.n_meas} measurements")
print(f"  relaxation needed: {study.needs_relaxation}   contact: {study.needs_contact}   field: {study.needs_field}")

S = np.asarray(pack.study(study))                       # (pairs, measurements), one read
print(f"\n{'tissue':32s} {'across':>8} {'along':>8}")
for label, row in zip(labels, S):
    print(f"{label:32s} {row[0]:8.4f} {row[1]:8.4f}")

one_at_a_time = np.array([np.asarray(pack.replay(across_and_along, tissue=t)) for t in tissues])
print(f"\nSame numbers one at a time, max difference {np.abs(S - one_at_a_time).max():.2e}: the study is not a")
print("different calculation, it is the same one arranged so the pack is read once.")

print("\nWhat the record says it did:")
meta = study.to_meta()
for k, pair in enumerate(meta["pairs"]):
    t = pair["tissue"]
    print(f"  pair {k}: tissue {'none' if t is None else {kk: vv for kk, vv in t.items() if vv is not None}}")
