"""What a walk records, and what each record makes possible later.

A walk stores channels, and each one is an exposure: a quantity accumulated along the path that some later
physics multiplies. They are what turn "the spins did this anyway" into something a replay can use.

  C0  positions            the path itself -- every gradient waveform is a contraction against it
  C1  compartment          which pool each walker was in, so relaxation can be applied per pool
  C2  boundary local time  how long each walker spent against a wall, so a surface relaxivity can be applied
  C3  field along the path what field the walker passed through, so a susceptibility at any B0 can be applied
  C4  bound fraction       time spent bound, for magnetization transfer

A channel costs storage in the walk and cannot be reconstructed afterwards. A tier the walk did not record is
refused at replay rather than skipped, which is the behaviour you want: a missing surface term should not
quietly return the unrelaxed signal.

Assumes rung 11.
"""
import numpy as np

from dmipy_sim import Cylinder, pgse, simulate_trajectories
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec.tissue import Tissue

pore = Cylinder(radius=4e-6, orientation=(0, 0, 1))
seq = pgse([[1.0, 0.0, 0.0]], 0.008, 0.030, bvalues=[1e9], TE=0.05, n_t=600)

full = simulate_trajectories(2_000, 2e-9, pore, T_max=0.06, dt_save=2e-4, seed=0, require_gpu=False)
bare = simulate_trajectories(2_000, 2e-9, pore, T_max=0.06, dt_save=2e-4, seed=0, require_gpu=False, tiers=())
print(f"tiers='all' recorded : positions, compartment {full.compartment is not None}, "
      f"boundary local time {full.boundary_local_time is not None}")
print(f"tiers=()    recorded : positions, compartment {bare.compartment is not None}, "
      f"boundary local time {bare.boundary_local_time is not None}")

p_full = build_replay_pack(full, id="cookbook/full", license="x", citation="the cookbook", K=32)
p_bare = build_replay_pack(bare, id="cookbook/bare", license="x", citation="the cookbook", K=32)
print(f"\nchannels in the pack: full {sorted(p_full.meta['compression']['channels'])}")
print(f"                      bare {sorted(p_bare.meta['compression'].get('channels', {}))}   (none: only the path)")

relaxivity = Tissue(T2=0.08, rho=1e-5, D=2e-9)
print(f"\nsurface relaxivity on the full pack: {float(np.asarray(p_full.replay(seq, tissue=relaxivity))[0]):.4f}")
try:
    p_bare.replay(seq, tissue=relaxivity)
except ValueError as e:
    print(f"on the bare pack:                    refused -- {str(e)[:78]}")

print("\nThe refusal is the point. The bare pack can still answer any gradient question, because C0 is there;")
print("it cannot answer a surface question, and it says so instead of returning the number without the term.")
