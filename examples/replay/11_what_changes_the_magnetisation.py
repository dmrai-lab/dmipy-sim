"""What the water carries, as opposed to where it goes -- and why one walk answers all of it.

Relaxation, surface relaxivity and susceptibility are properties of the substrate, not of the scanner. They
still do not move a walker: they change the magnetisation it carries along a path it would have taken anyway.
That is a different statement from the previous rung's -- there the diffusivity moved the walker but only by
rescaling the clock; here nothing moves at all, and the walk's own numbers are reused as they stand.
So the walk records the EXPOSURE -- which pool a walker was in at each save, how long it spent against a
wall, what field it passed through -- and the magnitude is applied afterwards.

That is the whole trick, and it is why a tissue is a replay knob despite being a property of the tissue: one
walk, then every T2, every relaxivity, every field, at the cost of an arithmetic pass over stored numbers.

Assumes rung 10.
"""
import numpy as np

from dmipy_sim import Cylinder, pgse, simulate_trajectories
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec.tissue import Tissue

pore = Cylinder(radius=4e-6, orientation=(0, 0, 1))
walk = simulate_trajectories(4_000, 2e-9, pore, T_max=0.06, dt_save=2e-4, seed=0, require_gpu=False)
pack = build_replay_pack(walk, id="cookbook/pore", license="CC-BY-4.0", citation="the cookbook", K=32)
seq = pgse([[1.0, 0.0, 0.0]], 0.008, 0.030, bvalues=[1e9], TE=0.05, n_t=600)

print(f"channels this walk recorded: {sorted(pack.meta.get('compression', {}).get('channels', {}))}")
print(f"\n{'tissue applied at replay':34s} {'S':>8}")
print(f"{'none (bare diffusion)':34s} {float(np.asarray(pack.replay(seq))[0]):8.4f}")
for T2 in (0.08, 0.04):
    S = float(np.asarray(pack.replay(seq, tissue=Tissue(T2=T2)))[0])
    print(f"{f'T2 = {T2*1e3:.0f} ms':34s} {S:8.4f}")
for rho in (1e-6, 1e-5):
    S = float(np.asarray(pack.replay(seq, tissue=Tissue(T2=0.08, rho=rho, D=2e-9)))[0])
    print(f"{f'T2 = 80 ms, relaxivity {rho*1e6:.0f} um/s':34s} {S:8.4f}")

print("\nOne walk produced every line. The walkers are identical in all of them; what differs is the weight")
print("each one carries to the readout. A field is the same kind of knob -- the walk stores the field along")
print("the path and the susceptibility and B0 multiply it at replay -- shown in the susceptibility rung.")
print("\nWhat a replay refuses: a tier the walk did not record. Asking for a relaxivity on a walk with no")
print("contact channel raises rather than returning the unrelaxed number.")

# ---------------------------------------------------------------------------------------------------------
# And what rung 10's time scaling does to these tiers, which is the question this rung invites.
#
# Speeding up the clock compresses the time a spin spends against a wall or inside a field gradient, per save
# interval. What answers it is that the acquisition is fixed in ITS OWN time, so it spans proportionately MORE
# save intervals: a 180 at TE/2 lands further along the recorded path, a storage period covers a different
# stretch of it, and a wall is met more often. Nothing cancels. Every tier's answer changes, which is what a
# higher diffusivity is.
print("\n" + "-" * 104)
print("what each channel stores at save k, walked at D and at 2D on the halved clock -- the same path:\n")

D2, T2_, dt2, a = 2e-9, 0.02, 1e-4, 2.0
wall = Cylinder(4e-6, (0, 0, 1), surface_relaxivity_t2=1e-5)
slow = simulate_trajectories(600, D2, wall, T_max=T2_, dt_save=dt2, seed=0, require_gpu=False)
fast = simulate_trajectories(600, a * D2, wall, T_max=T2_ / a, dt_save=dt2 / a, seed=0, require_gpu=False)
for label, name in (("positions", "positions"), ("boundary local time", "boundary_local_time"),
                    ("compartment occupancy", "compartment")):
    x, y = np.asarray(getattr(slow, name), float), np.asarray(getattr(fast, name), float)
    print(f"  {label:22s} max |difference| {np.abs(x - y).max():.3e}")

# The surface term is the one that LOOKS like it cancels, because its weight carries a 1/D. Over a fixed echo
# time the accumulated exposure doubles and the divisor doubles, so the MEAN exposure is invariant. The signal
# is not the mean: it is an ensemble average of exp(exposure), and that distribution changes shape.
TE, dt, rho, N = 2e-4, 1e-6, 3.5e-3, 3000
w = simulate_trajectories(N, D2, Cylinder(4e-6, (0, 0, 1), surface_relaxivity_t2=rho),
                          T_max=2 * TE, dt_save=dt, seed=0, require_gpu=False)
ell = np.asarray(w.boundary_local_time, float)
n = int(round(TE / dt))
x_D = (rho / D2) * ell[:, :n + 1].sum(axis=1)            # exposure over TE at D
x_2D = (rho / (2 * D2)) * ell.sum(axis=1)                # over TE at 2D: twice as far along, twice the divisor
print(f"\nsurface relaxivity over the same {TE*1e3:g} ms echo time:")
print(f"  exposure at  D: mean {x_D.mean():7.4f}  spread {x_D.std():.4f}")
print(f"  exposure at 2D: mean {x_2D.mean():7.4f}  spread {x_2D.std():.4f}   <- same mean, narrower")
print(f"  signal    at  D: {np.exp(x_D).mean():.6f}")
print(f"  signal    at 2D: {np.exp(x_2D).mean():.6f}   <- NOT the same number")

print("""
So the surface term does not cancel either. Its mean exposure does, which is why it looks as though it
should; but faster diffusion mixes the walkers better, the spread of exposure across them narrows, and the
average of an exponential moves when its distribution changes shape even at a fixed mean. The two agree only
in the well-mixed limit, where the spread is small enough that the mean is the whole answer.

What IS exact is the replay. Reading the recorded path further along and dividing by the new diffusivity is
bit-identical to a walk actually done at that diffusivity, walker for walker. The replay reproduces the
change rather than cancelling it -- which is the only property it needs.""")
