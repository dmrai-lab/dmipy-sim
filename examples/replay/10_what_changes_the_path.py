"""What actually changes the path, and what is only a change of clock.

It is tempting to say the walk fixes the diffusivity, the geometry and the permeability, and that changing
any of them means walking again. That is wrong about two of the three, and the reason is one line of the
position update: the diffusivity enters only through sqrt(2 D) dW. So a path walked at D and read on a time
axis divided by a IS a path walked at a D -- the same spatial path, the same wall encounters, the same
sub-steps. Diffusivity is a time scaling, not a different physical situation.

Across a membrane the crossings are part of the path, and what a walk realises is the crossing probability
per encounter, proportional to kappa sqrt(dt / D). So the same path is the walk at (a D, a kappa) and at no
other pair: the permeability comes along with the scaling rather than blocking it.

That leaves the GEOMETRY as the thing that genuinely changes where the water goes.

This is also what a change of temperature does. Water's diffusivity rises about 2.5 % per kelvin, so for an
impermeable substrate a pack walked at room temperature replays a brain at 37 C exactly. A permeability has
its own activation energy and does not follow D by the same factor, so for a permeable substrate the scaling
is an approximation and a pack meant for a temperature is walked at it.

Assumes rung 09. Theory: `replayable_mc/sections/theory.tex`, "Diffusivity is a time scaling, which is what
temperature does".
"""
import numpy as np

from dmipy_sim import Cylinder, pgse, simulate, simulate_trajectories

pore = Cylinder(radius=4e-6, orientation=(0, 0, 1))
D, T, dt, a = 2e-9, 0.04, 2e-4, 2.0

# The claim, checked: the walk at 2D over half the time on half the grid is the SAME PATH.
slow = simulate_trajectories(1_000, D, pore, T_max=T, dt_save=dt, seed=0, require_gpu=False)
fast = simulate_trajectories(1_000, a * D, pore, T_max=T / a, dt_save=dt / a, seed=0, require_gpu=False)
p, q = np.asarray(slow.positions), np.asarray(fast.positions)
print(f"walked at  D over {T*1e3:.0f} ms on a {dt*1e6:.0f} us grid: {p.shape[1]} saves")
print(f"walked at 2D over {T/a*1e3:.0f} ms on a {dt/a*1e6:.0f} us grid: {q.shape[1]} saves")
print(f"max |difference| between the two paths: {np.abs(p - q).max():.1e} m   -- the same walk, a different clock")

# The geometry is the one that moves the water somewhere else.
seq = pgse([[1.0, 0.0, 0.0]], 0.008, 0.030, bvalues=[1e9], TE=0.05, n_t=600)
for R in (4e-6, 2e-6):
    S = float(simulate(4_000, D, seq, Cylinder(radius=R, orientation=(0, 0, 1)), seed=0, require_gpu=False)[0])
    print(f"\nradius {R*1e6:.0f} um: S = {S:.4f}")
print("a different radius is a different substrate, and no rescaling of the clock reaches it")

print("\nWhat the code does today: the rescaling is NOT applied at replay (dmipy-sim#289), so a pack answers")
print("at the diffusivity it was walked with and another D means another walk. That is an implementation gap,")
print("not the boundary -- the boundary is the geometry, and the certificate's limit on how far a save grid")
print("may be stretched before its band no longer covers the acquisition.")
