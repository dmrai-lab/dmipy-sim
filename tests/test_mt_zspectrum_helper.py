"""Fast-tier guard for the turnkey ``emergent_z_spectrum`` helper.

The physics (emergent Z-spectrum vs the two-pool oracle) is validated in the slow
``test_mt_zspectrum.py``.  Here we only check that the helper is a *faithful wrapper*:
over a tiny MC run it must reproduce a hand-written ``simulate_bloch`` offset sweep with
identical parameters and seed, and return the right shape.  Small/`off`-equilibrated so it
stays in the ~1-min fast tier.
"""
from types import SimpleNamespace

import numpy as np

from dmipy_sim import Sphere, simulate_bloch, emergent_z_spectrum

# tiny, deterministic config (fast tier: small N, short saturation, no burn-in)
R, D = 2e-6, 1e-9
CFG = dict(n_walkers=400, diffusivity=D, w1_hz=150.0, t_sat=8e-3, dt=4e-5,
           T2=60e-3, T1=1.0, kappa_MT=3e-5, dwell_time=1.0 / 60.0, T2_bound=1e-5,
           equilibrate_binding="off", seed=1)
OFFSETS = np.array([0.0, 4000.0])


def _manual_sweep():
    n_t = int(round(CFG["t_sat"] / CFG["dt"])) + 1
    wf = SimpleNamespace(G=np.zeros((1, n_t, 3)), dt=CFG["dt"])
    flip = 360.0 * CFG["w1_hz"] * CFG["t_sat"]
    out = []
    for off in OFFSETS:
        rf = [{"t_s": CFG["t_sat"] / 2, "flip_deg": flip, "axis_deg": 0.0,
               "duration_s": CFG["t_sat"], "offset_hz": float(off)}]
        _, mz = simulate_bloch(CFG["n_walkers"], D, wf, Sphere(radius=R), rf,
                               T2=CFG["T2"], T1=CFG["T1"], kappa_MT=CFG["kappa_MT"],
                               dwell_time=CFG["dwell_time"], T2_bound=CFG["T2_bound"],
                               return_mz=True, equilibrate_binding="off", seed=CFG["seed"])
        out.append(float(mz[0]))
    return np.array(out)


def test_emergent_z_spectrum_matches_manual_sweep():
    z = emergent_z_spectrum(OFFSETS, Sphere(radius=R), **CFG)
    assert z.shape == OFFSETS.shape
    assert np.all(np.isfinite(z))
    # faithful wrapper: identical args + seed -> identical result as a hand sweep
    np.testing.assert_allclose(z, _manual_sweep(), atol=1e-6)


def test_emergent_z_spectrum_scalar_offset():
    z = emergent_z_spectrum(0.0, Sphere(radius=R), **CFG)
    assert z.shape == (1,) and np.isfinite(z[0])
