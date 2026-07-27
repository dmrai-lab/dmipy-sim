"""Susceptibility off-resonance fields in the forward Bloch engine.

Three ways to define a susceptibility source, each yielding an off-resonance field
ΔBz(r) that the walk accrues as an extra z-precession γ·ΔBz(r)·dt:

  1. isotropic sphere perturbers (grey-matter iron / vasculature)
  2. anisotropic hollow-cylinder myelin (Wharton & Bowtell), closed form
  3. an arbitrary distribution voxelised onto a grid (the mesh route)

Run:  python examples/susceptibility_field.py
"""
import numpy as np
import jax, jax.numpy as jnp

from dmipy_sim import (simulate_bloch, FreeDiffusion, GAMMA,
                       SusceptibilitySources, MyelinSusceptibility, GridSusceptibility,
                       dipole_field, myelin_susceptibility_tensor, mesh_shapes as ms)

# ---- 1. analytic myelin field: zero at parallel B0, non-zero at perpendicular ----
a, b, L = 2e-6, 3e-6, 20e-6
for theta, name in [(0.0, "B0 || fibre"), (np.pi / 2, "B0 _|_ fibre")]:
    prov = MyelinSusceptibility(centers=[[0., 0.]], inner_radii=[a], outer_radii=[b],
                                L=L, delta_chi_a=-0.1e-6, B0=7.0, theta=theta, periodic=False)
    f = jax.jit(prov.delta_bz_fn())
    intra = float(f(jnp.array([0.2e-6, 0., 0.])))       # inside the lumen
    print(f"myelin intra ΔBz ({name:12s}) = {intra: .3e} T")

# ---- 2. grid (mesh) route: hollow-cylinder intra offset = (chi_A/2) ln(1/g) ----
X, Y, Z, vs, org = ms.grid_axes(24e-6, 96)
mask, radial = ms.straight_myelin_source(X, Y, Z, a, b)
dB = dipole_field(myelin_susceptibility_tensor(mask, radial, chi_aniso=-0.1e-6),
                  vs, [1, 0, 0], B0=7.0)
rho = np.sqrt(X ** 2 + Y ** 2)
print(f"grid intra ΔBz (perp)         = {dB[rho < 0.6 * a].mean(): .3e} T  "
      f"(analytic {-0.1e-6 * 7.0 * 0.5 * np.log(b / a): .3e})")

# ---- 3. forward Bloch spin echo: uniform field refocuses; diffusion attenuates ----
dt, TE = 1e-4, 6e-3
n_t = int(round(TE / dt)) + 1
from types import SimpleNamespace
wf = SimpleNamespace(G=np.zeros((1, n_t, 3)), dt=dt)
exc = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': 90.0}]
se = exc + [{'t_s': TE / 2, 'flip_deg': 180.0, 'axis_deg': 0.0}]
src = SusceptibilitySources(centers=[[0., 0., 0.]], radii=[2e-6], delta_chi=3e-6, B0=7.0)
frozen = abs(simulate_bloch(4000, 1e-13, wf, FreeDiffusion(), se, seed=0, susceptibility=src)[0])
moving = abs(simulate_bloch(4000, 2e-9, wf, FreeDiffusion(), se, seed=0, susceptibility=src)[0])
print(f"spin-echo |S| frozen spins    = {frozen:.3f}  (static field refocused)")
print(f"spin-echo |S| diffusing spins = {moving:.3f}  (susceptibility x diffusion)")
