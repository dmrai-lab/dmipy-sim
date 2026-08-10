#!/usr/bin/env python3
"""FEXI apparent-exchange-rate (AXR) demo — WORK IN PROGRESS, see FEXI_AXR_NOTES.md.

Runs the ``dmipy_sim.pulse_sequence.fexi`` stimulated-echo filter-exchange sequence on a
permeable packed substrate through the vector-Bloch engine (which now models membrane
permeation), sweeps the mixing time, and fits the AXR recovery model

    ADC'(t_m) = ADC_eq * (1 - sigma * exp(-AXR * t_m))          (Lasič 2011; Kiselev & Li 2026)

for an exchanging (kappa>0) vs a non-exchanging (kappa=0) substrate. With exchange the filtered
ADC' recovers toward equilibrium at rate AXR; without it, ADC' is flat (the null).

STATUS: the sequence + engine are correct, but a *clean* AXR curve needs a substrate with a
strong (>=2x) inter-pool ADC contrast. Packed cylinders (single bulk D) give only a weak filter
(sigma ~ 0.2) and a noisy curve; packed SPHERES (3D confinement -> much lower intra ADC) are more
promising and are the default here. A realistic grey-matter mesh (ConCeG, arXiv:2607.03286, via
Mesh.from_ply) is the faithful substrate. Do NOT use per-compartment D + permeability together —
dmipy-sim rejects a diffusivity discontinuity across a permeable wall (the FEXI contrast is
geometric restriction, not a bulk-D difference). See FEXI_AXR_NOTES.md for the full findings.

Run (GPU strongly recommended — this is a heavy permeable MC):
    NVLIBS=$(find ~/.local/lib/python3.11/site-packages/nvidia -name '*.so*' -path '*/lib/*' \
            | sed 's:/[^/]*$::' | sort -u | tr '\n' ':')
    LD_LIBRARY_PATH="$NVLIBS" XLA_PYTHON_CLIENT_PREALLOCATE=false \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 JAX_PLATFORMS=cuda python examples/fexi_axr_demo.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.optimize import curve_fit

from dmipy_sim.geometries import pack_spheres, PackedSpheres
from dmipy_sim.pulse_sequence import fexi, run_bloch_sequence   # <- library constructor (no handroll)

# ---- substrate: uniform-D packed spheres (3D restriction gives the intra/extra ADC contrast) ----
D = 2e-9                                        # bulk diffusivity (m^2/s)
R = 3e-6                                         # sphere radius (m); smaller -> stronger restriction
radii = np.full(60, R)
centers, L, vf = pack_spheres(radii, target_vf=0.35, seed=0)    # RSA monodisperse limit ~0.38

# ---- FEXI acquisition (bipolar filter + STE storage + bipolar detection) ----
DELTA_LOBE, DELTA_SEP = 4e-3, 20e-3             # delta / Delta (lobe separation = diffusion time)
G_FILTER = 0.4                                  # filter gradient (T/m); tune for sigma ~ 0.3-0.6
G_DETECT = 0.22                                 # detection gradient (one nonzero b + b=0)
T_MIX = np.array([10, 40, 90, 160, 260]) * 1e-3
N_WALKERS, SEEDS, DT = 40_000, (1, 2, 3), 1e-4


def _S(seq, geom):
    return np.mean([abs(complex(run_bloch_sequence(seq, N_WALKERS, D, geom, seed=s,
                                                    require_gpu=True)[0])) for s in SEEDS])


def adc_prime(geom, g_filter, t_mix):
    s0 = fexi(DELTA_LOBE, t_mix, DT, Delta=DELTA_SEP, g_filter=g_filter, g_detect=[0.0])
    sb = fexi(DELTA_LOBE, t_mix, DT, Delta=DELTA_SEP, g_filter=g_filter, g_detect=[G_DETECT])
    return -np.log(_S(sb, geom) / _S(s0, geom)) / sb.b_detect[0]


model = lambda t, eq, sigma, axr: eq * (1.0 - sigma * np.exp(-axr * t))

for kappa, tag in [(0.0, "kappa=0 (no exchange)"), (5e-5, "kappa=5e-5 (exchange)")]:
    geom = PackedSpheres(radii=radii, centers=centers, L=L,
                         permeability=(None if kappa == 0 else kappa))
    eq = adc_prime(geom, 0.0, 60e-3)                              # equilibrium (filter off)
    adcp = np.array([adc_prime(geom, G_FILTER, tm) for tm in T_MIX])
    line = f"{tag}: eq={eq*1e9:.2f}  ADC'(t_m)=" + " ".join(f"{a*1e9:.2f}" for a in adcp)
    try:
        p, _ = curve_fit(model, T_MIX, adcp, p0=[eq, 0.4, 5.0], maxfev=40000,
                         bounds=([0, 0, 0], [5e-9, 1, 300]))
        line += f"   -> sigma={p[1]:.2f}, AXR={p[2]:.1f} s^-1"
    except Exception as e:
        line += f"   (fit failed: {e})"
    print(line, flush=True)
