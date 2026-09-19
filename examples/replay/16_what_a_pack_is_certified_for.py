"""A pack travels; what travels with it is the statement of what it can answer.

A replay pack is a file. It is meant to be written once, published, and read by someone who did not run the
walk and cannot inspect it. So the file has to answer, by itself, three questions that a bare array of
numbers cannot: how accurate is it, which physics did the walk record, and which acquisitions is that
accuracy claimed for.

`fidelity` answers the first, and the number that matters is a comparison rather than an absolute: the
codec's error against the Monte-Carlo floor of the walk it compressed. A compression whose error sits under
the noise the walk already has is invisible in any use of that walk. `replay_envelope` answers the other two
-- which tiers are in the file, and over what range of acquisitions the fidelity was measured.

This is the part that makes replay publishable rather than merely fast. A number from a pack outside its
envelope is not wrong in a way the file can detect, so the file states the envelope and the consumer checks
it.

Assumes rung 12.
"""
import numpy as np

from dmipy_sim import Cylinder, pgse, simulate_trajectories
from dmipy_sim.replay import ReplayPack
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec.tissue import Tissue

pore = Cylinder(radius=4e-6, orientation=(0, 0, 1))
walk = simulate_trajectories(4_000, 2e-9, pore, T_max=0.06, dt_save=2e-4, seed=0, require_gpu=False)
pack = build_replay_pack(walk, id="cookbook/pore", license="CC-BY-4.0", citation="the cookbook", K=32)

path = "cookbook_pore.rpk"
pack.save(path)
reloaded = ReplayPack.load(path)                       # what a reader gets, with no access to the walk

f = reloaded.fidelity
print(f"pack {reloaded.id}, licence {reloaded.license}")
print(f"  codec error {f['err_max']:.2e} against a Monte-Carlo floor of {f['floor_max']:.4f}")
print(f"  certified: {f['certified']}, within twice the floor: {f['within_2x_floor']}")
print(f"  bands {reloaded.K}, so {reloaded.temporal_bandwidth_hz:.0f} Hz resolved over this walk")

print("\nthe error by sequence family, which is where a single worst case would hide the structure:")
for family, v in sorted(f["per_family"].items()):
    print(f"  {family:8s} error {v['err_max']:.2e}   floor {v['floor_max']:.4f}")

env = reloaded.replay_envelope
flags = {"diffusivity_fixed"}                          # a statement about the walk, not a tier it carries
print(f"\ntiers this walk recorded: {', '.join(k for k, v in env.items() if v is True and k not in flags)}")
print(f"tiers it did not: {', '.join(k for k, v in env.items() if v is False and k not in flags)}")
print(f"acquisitions the fidelity was measured over: {env['acquisition']}")

seq = pgse([[1.0, 0.0, 0.0]], 0.008, 0.030, bvalues=[1e9], TE=0.05, n_t=600)
print(f"\nthe reloaded pack answers as the original did: "
      f"{float(np.asarray(reloaded.replay(seq, tissue=Tissue(T2=0.08)))[0]):.4f} against "
      f"{float(np.asarray(pack.replay(seq, tissue=Tissue(T2=0.08)))[0]):.4f}")

print("\n`diffusivity_fixed` in the envelope is the honest entry. The theory says a pack replays at any")
print("diffusivity, because that is a rescaling of the clock and not a different walk (rung 10); the")
print("implementation does not do it yet, so the file says the walk's diffusivity is the one it answers at.")
print("An envelope that claimed the capability would be the dangerous kind of wrong.")
