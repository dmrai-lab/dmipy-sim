"""Canonical-WM parity, entered from the dmipy-sim side.

Build the white-matter substrate from the dmipy-sim base API + catalogue, run the Monte-Carlo
forward, then pull the ANALYTICAL forward from the dmipy-fit factory and show they agree. This
is the mirror of dmipy-fit's flagship (examples/flagship_canonical_wm/), which enters from the
fit side. It demonstrates the shared substrate/sequence interface -- agreement is the interface
contract working, NOT a proof of correctness (the engines are correlated).

Fast surface-relaxivity check at b=0 (gradient-free): surface ON vs OFF, sim MC vs fit
analytical. The step-resolved *diffusion* parity is the full flagship on the fit side.

dmipy-fit is an OPTIONAL dependency of this EXAMPLE only (never of the dmipy-sim package):
    pip install dmipy-sim[examples]      # pulls dmipy-fit
Run:  python canonical_wm_parity_from_sim.py
"""
import sys
import numpy as np

try:
    from dmipy_fit.white_matter.composition import build_white_matter_model
    from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
except ImportError:
    print("This cross-engine example needs the analytical engine dmipy-fit.\n"
          "Install it with:  pip install dmipy-sim[examples]   (or pip install dmipy-fit)\n"
          "Skipping.")
    sys.exit(0)

from dmipy_sim.substrate.biophysical_constants import canonical_white_matter, get_value
from dmipy_sim import pack_myelinated_cylinders, PackedMyelinatedCylinders, simulate, pgse

C = canonical_white_matter(3.0)
D, G, RHO = C['D_intra'], C['g_ratio'], C['rho2']
T2I, T2E, T2M = C['T2_intra'], C['T2_extra'], C['T2_myelin']
AL, SC = get_value('gamma_shape_diameter'), get_value('gamma_scale_diameter')
FA = 0.55                                                  # fibre volume fraction
TE = 0.04

# --- sim side: build the substrate from the base API + catalogue Gamma ---
def geom(surf, seed=0):
    d_out = np.maximum(np.random.default_rng(seed).gamma(AL, SC, 40), 0.4e-6)
    inner, gr, cen = pack_myelinated_cylinders(inner_radii=G * d_out / 2,
        g_ratios=np.full(40, G), target_packing=FA, seed=seed)
    cell = float(np.sqrt(np.pi * np.sum((inner / gr) ** 2) / FA))
    g = PackedMyelinatedCylinders(inner_radii=inner, g_ratios=gr, centers=cen, cell_size=cell,
        N_max=len(inner) + 1, D_intra=D, D_extra=D, D_myelin=0.0,
        T2_intra=T2I, T2_extra=T2E, T2_myelin=T2M,
        rho_inner=(RHO if surf else 0.0), rho_outer=(RHO if surf else 0.0),
        kappa_inner=0.0, kappa_outer=0.0)
    g.surface_substep_frac = (2.0 if surf else 0.0)
    return g

wf0 = pgse(delta=TE / 2 - 1e-4, DELTA=TE / 2, G_magnitude=0.0,
           bvecs=np.array([[0, 0, 1.]], np.float32), n_t=3000)   # b=0, gradient-free

# --- fit side: analytical forward from the factory (same substrate) ---
scheme = acquisition_scheme_from_bvalues(np.array([0.]), np.array([[0, 0, 1.]]),
                                         delta=0.01, Delta=0.02, TE=TE)
def analytic(rho):
    model, p = build_white_matter_model()
    p = dict(p); p['OccupancyGatedModel_1_mu'] = np.array([0.0, 0.0])
    p['OccupancyGatedModel_1_surface_relaxivity'] = rho
    p['OccupancyGatedModel_2_surface_relaxivity'] = rho
    return float(np.asarray(model(scheme, **p)).ravel()[0])

print("surface   fit S0    mc S0     |diff|")
for surf, rho in [(False, 0.0), (True, RHO)]:
    s_mc = float(np.asarray(simulate(5000, waveform=wf0, geometry=geom(surf),
                                     seed=1, require_gpu=False)).ravel()[0])
    s_fit = analytic(rho)
    print(f"  {'ON ' if surf else 'OFF'}    {s_fit:.4f}    {s_mc:.4f}    {abs(s_fit-s_mc):.4f}")
    assert abs(s_fit - s_mc) < 0.02, "cross-engine S0 parity gap too large"
print("PASS: sim MC and fit analytical agree on S0 in both surface states.")
