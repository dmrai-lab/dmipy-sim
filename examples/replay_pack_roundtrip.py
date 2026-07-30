"""Replay packs (.rpk): walk once, replay any acquisition — then share it.

The Monte-Carlo REPLAY workflow in dmipy-sim end to end:

  1. walk the substrate ONCE                 (simulate_trajectories)
  2. compress it into a replay pack           (build_replay_pack -> .rpk)
  3. write / read the .rpk container           (safetensors; the [bank] extra)
  4. replay ANY waveform / b-value             (ReplayPack.replay) -- no re-simulation

The walker trajectory depends only on (geometry, diffusivity, seed); the gradient
waveform, b-value, relaxation and surface relaxivity are replay *knobs*. So one walk
serves every acquisition below, reconstructed post-hoc instead of re-simulated. The
replayed signal matches the forward engine to the Monte-Carlo noise floor.

Needs the bank extra:  pip install -e ".[bank]"
Run:                    python examples/replay_pack_roundtrip.py
"""
import numpy as np

from dmipy_sim import (simulate_trajectories, simulate, pgse, set_b, Sphere,
                       master_from_walk, build_replay_pack, read_rpk)

D = 2e-9
T2 = 40e-3
geom = Sphere(radius=6e-6)
N = 8000
T_max = 60e-3
OUT = "/tmp/sphere-r6.rpk"

# 1) ONE walk to T_max (every shorter TE is a leading prefix of this walk).
result = simulate_trajectories(N, D, geom, T_max=T_max, dt_save=1e-3,
                               seed=7, save_relaxation_data=True)

# 2) master dict -> compressed, self-certifying replay pack. K=None auto-selects the
#    smallest number of low-rank modes that keeps the *measured* replay error within
#    2x the Monte-Carlo split-half floor across the default acquisition envelope.
master = master_from_walk(result, D=D, T2_per_comp=[T2, T2])
pack = build_replay_pack(master, id="demo/sphere-r6", method="lowrank", K=None,
                         license="CC-BY-4.0", citation="dmipy-sim replay demo",
                         out_path=OUT)
print(pack)
print(f"fidelity within 2x MC floor: {pack.fidelity.get('within_2x_floor')}\n")

# 3) container round-trip (this is exactly what stage_pack/publish upload and
#    pull downloads from the substrate bank).
loaded = read_rpk(OUT)

# 4) replay a b-value sweep off the SINGLE walk; cross-check the forward engine.
print(f"{'b [s/mm2]':>10} {'replay':>9} {'engine':>9} {'|delta|':>9}")
worst = 0.0
for b_si in (1e9, 2e9, 3e9):
    wf = set_b(pgse(delta=0.008, DELTA=0.03, G_magnitude=0.2,
                    bvecs=[[1, 0, 0]], n_t=200), b_si)
    s_replay = float(np.real(loaded.replay(wf, relaxation=True))[0])
    s_engine = float(np.real(simulate(N, D, wf, geom, seed=7, T2=T2,
                                      engine="fused"))[0])
    worst = max(worst, abs(s_replay - s_engine))
    print(f"{b_si/1e6:10.0f} {s_replay:9.4f} {s_engine:9.4f} "
          f"{abs(s_replay - s_engine):9.1e}")

print(f"\nworst |replay - engine| = {worst:.2e} (MC noise floor ~ 1/sqrt(N) = "
      f"{1/np.sqrt(N):.2e})")
assert worst < 0.05, "replay drifted from the forward engine beyond the noise floor"
print("OK: one walk replays every acquisition, matching the forward engine.")
