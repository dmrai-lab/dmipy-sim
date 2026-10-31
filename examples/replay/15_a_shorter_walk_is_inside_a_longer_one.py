"""One long walk answers every shorter echo time, because a prefix of a walk is a walk.

A trajectory to 100 ms contains the trajectory to 50 ms: the first half of the record is exactly what the
engine would have produced had it been told to stop there, since nothing later in a walk influences anything
earlier. So a pack built once at the longest echo time a study needs also serves every shorter one, and a
multi-TE protocol is one walk rather than one walk per TE.

The bands are what makes this a real operation rather than a slice. A pack stores each path as K sine bands
over its duration T, which resolve frequencies up to K/(2T). Cutting the duration to T' and keeping K bands
RAISES the resolved frequency, so a prefix is never less faithful per unit time than its parent; asking for
the same bands per second on the shorter walk needs only K' = K T'/T. `prefix` recomputes the coefficients on
the shorter support and re-certifies there, rather than reinterpreting the parent's numbers.

Assumes rung 13.
"""
import numpy as np

from dmipy_sim import Cylinder, pgse, simulate_trajectories
from dmipy_sim.replay.bank import build_replay_pack

pore = Cylinder(radius=4e-6, orientation=(0, 0, 1))
D, N, dt_save = 2e-9, 4_000, 2e-4

long_walk = simulate_trajectories(N, D, pore, T_max=0.10, dt_save=dt_save, seed=0, require_gpu=False)
pack = build_replay_pack(long_walk, id="cookbook/pore-100ms", license="CC-BY-4.0", citation="the cookbook", K=64)
T = (pack.n_t - 1) * pack.dt
print(f"one walk to {T*1e3:.0f} ms: {pack.n_walkers} walkers, K = {pack.K} bands, "
      f"{pack.temporal_bandwidth_hz:.0f} Hz resolved")

print(f"\n{'echo time':>10} {'K':>5} {'Hz resolved':>12} {'S (from the prefix)':>20} {'S (walked to that TE)':>22}")
for TE in (0.03, 0.05, 0.08):
    short = pack.prefix(TE)
    seq = pgse([[1.0, 0.0, 0.0]], 0.006, TE - 0.012, bvalues=[1e9], TE=TE, n_t=600)
    from_prefix = float(np.asarray(short.replay(seq))[0])
    # the same substrate and seed, walked only as far as this TE: what the prefix claims to equal
    own = simulate_trajectories(N, D, pore, T_max=TE, dt_save=dt_save, seed=0, require_gpu=False)
    own_pack = build_replay_pack(own, id="cookbook/own", license="CC-BY-4.0", citation="the cookbook", K=64)
    walked = float(np.asarray(own_pack.replay(seq))[0])
    print(f"{TE*1e3:9.0f} ms {short.K:5d} {short.temporal_bandwidth_hz:12.0f} {from_prefix:20.4f} {walked:22.4f}")

print("\nThe right-hand column is the walk the prefix replaces. They agree at the codec's level, which is the")
print("claim: the shorter experiment did not need its own simulation.")
print("\nA prefix is refused when the echo time is not on the pack's save grid, and when the walk is shorter")
print("than the acquisition asks for -- the one thing a replay genuinely cannot invent is more time.")
try:
    pack.prefix(0.2)
except ValueError as e:
    print(f"  asking for 200 ms of a {T*1e3:.0f} ms walk: {str(e).splitlines()[0]}")
