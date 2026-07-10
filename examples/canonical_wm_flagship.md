---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Flagship: canonical white matter, from the forward-truth engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmrai-lab/dmipy-sim/blob/main/examples/canonical_wm_flagship.ipynb)

dmipy-sim is the **forward-truth engine**: it random-walks spins through a real geometric
substrate and accumulates phase under a gradient waveform, with no analytical shortcut. This
flagship builds a **canonical white-matter substrate** — packed myelinated cylinders with a
histology-calibrated diameter distribution — runs the Monte-Carlo forward, and shows two things:

1. **surface relaxivity** measurably lowers the signal (the myelin wall is a sink), and
2. the MC signal **agrees with the analytical model in dmipy-fit** built from the *same*
   substrate — the shared substrate/sequence interface working end to end.

Cross-engine agreement is the interface contract holding, **not** a proof of correctness: the two
engines are correlated. It is necessary evidence, not sufficient. The step-resolved *diffusion*
parity is the companion flagship on the [dmipy-fit side](https://github.com/dmrai-lab/dmipy-fit/tree/main/examples/flagship_canonical_wm).

```{code-cell} ipython3
# On Colab this installs both engines (public on GitHub); locally it is a no-op.
# Once dmipy is on PyPI:  pip install "dmipy[examples]"
import importlib.util, subprocess, sys
if importlib.util.find_spec("dmipy_fit") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "dmipy-sim @ git+https://github.com/dmrai-lab/dmipy-sim.git",
                    "dmipy-fit @ git+https://github.com/dmrai-lab/dmipy-fit.git"], check=True)
```

## The canonical substrate — from the constants catalogue

Every tissue constant comes from the cited `biophysical_constants` catalogue (one source of
truth), not hard-coded here: intra/extra/myelin diffusivities and T2, the g-ratio, the Gamma
diameter distribution, and the surface relaxivity `rho2`.

```{code-cell} ipython3
import numpy as np
from dmipy_sim.substrate.biophysical_constants import canonical_white_matter, get_value
from dmipy_sim import pack_myelinated_cylinders, PackedMyelinatedCylinders, simulate, pgse

C = canonical_white_matter(3.0)                       # at 3 T
D, G, RHO = C['D_intra'], C['g_ratio'], C['rho2']
T2I, T2E, T2M = C['T2_intra'], C['T2_extra'], C['T2_myelin']
AL, SC = get_value('gamma_shape_diameter'), get_value('gamma_scale_diameter')
FA, TE = 0.55, 0.04                                   # fibre volume fraction, echo time (s)
print(f"D_intra={D:.2e} m^2/s  g-ratio={G:.2f}  rho2={RHO:.1e} m/s  TE={TE*1e3:.0f} ms")
```

```{code-cell} ipython3
def build_substrate(surface_on, seed=0):
    """Pack myelinated cylinders with a Gamma outer-diameter distribution."""
    d_out = np.maximum(np.random.default_rng(seed).gamma(AL, SC, 40), 0.4e-6)
    inner, gr, cen = pack_myelinated_cylinders(
        inner_radii=G * d_out / 2, g_ratios=np.full(40, G),
        target_packing=FA, seed=seed)
    cell = float(np.sqrt(np.pi * np.sum((inner / gr) ** 2) / FA))
    g = PackedMyelinatedCylinders(
        inner_radii=inner, g_ratios=gr, centers=cen, cell_size=cell,
        N_max=len(inner) + 1, D_intra=D, D_extra=D, D_myelin=0.0,
        T2_intra=T2I, T2_extra=T2E, T2_myelin=T2M,
        rho_inner=(RHO if surface_on else 0.0),
        rho_outer=(RHO if surface_on else 0.0),
        kappa_inner=0.0, kappa_outer=0.0)
    g.surface_substep_frac = (2.0 if surface_on else 0.0)
    return g
```

## Monte-Carlo forward: surface relaxivity OFF vs ON

A gradient-free `pgse` (b=0) isolates the pure T2 / surface-relaxivity attenuation — no diffusion
weighting — so the effect of the myelin wall as a relaxation sink is visible directly.

```{code-cell} ipython3
wf0 = pgse(delta=TE / 2 - 1e-4, DELTA=TE / 2, G_magnitude=0.0,
           bvecs=np.array([[0, 0, 1.]], np.float32), n_t=3000)
S_off = float(np.asarray(simulate(5000, waveform=wf0, geometry=build_substrate(False),
                                   seed=1, require_gpu=False)).ravel()[0])
S_on  = float(np.asarray(simulate(5000, waveform=wf0, geometry=build_substrate(True),
                                   seed=1, require_gpu=False)).ravel()[0])
print(f"MC S0  surface OFF = {S_off:.4f}")
print(f"MC S0  surface ON  = {S_on:.4f}   (surface relaxivity removes {100*(S_off-S_on)/S_off:.1f}%)")
```

## Cross-engine check: the same substrate, the analytical forward

The analytical white-matter model in dmipy-fit is built from the *same* catalogue and consumes
the *same* substrate description — no conversion layer. It should reproduce the MC S₀.

```{code-cell} ipython3
from dmipy_fit.white_matter.composition import build_white_matter_model
from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues

scheme = acquisition_scheme_from_bvalues(np.array([0.]), np.array([[0, 0, 1.]]),
                                         delta=0.01, Delta=0.02, TE=TE)

def analytic_S0(rho):
    model, p = build_white_matter_model()
    p = dict(p); p['OccupancyGatedModel_1_mu'] = np.array([0.0, 0.0])
    p['OccupancyGatedModel_1_surface_relaxivity'] = rho    # intra-pore wall
    p['OccupancyGatedModel_2_surface_relaxivity'] = rho    # exterior wall
    return float(np.asarray(model(scheme, **p)).ravel()[0])

print("surface   fit S0    MC S0     |diff|")
for on, rho, mc in [(False, 0.0, S_off), (True, RHO, S_on)]:
    fit = analytic_S0(rho)
    print(f"  {'ON ' if on else 'OFF'}    {fit:.4f}    {mc:.4f}    {abs(fit-mc):.4f}")
    assert abs(fit - mc) < 0.02, "cross-engine S0 parity gap too large"
print("\nsim MC and fit analytical agree on S0 in both surface states.")
```

## Takeaway

- dmipy-sim builds a **real** canonical-WM substrate and runs the physics forward — the
  ground truth the analytical models are checked against.
- Surface relaxivity is a first-class, baked-in effect (myelin wall as a T2 sink).
- The MC and analytical forwards, built from one shared substrate, agree — the interface contract.
  For the full step-resolved diffusion parity, see the dmipy-fit flagship.
