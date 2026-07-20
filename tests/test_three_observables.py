"""Three observables of one wall: the deconfounding (Piece F of the MT staging ladder).

Surface relaxivity ``rho`` and MT reactivity ``kappa_MT`` are BOTH wall reactivities
that attenuate the free-water transverse signal through the same boundary-local-time
channel, so to first order they are degenerate there: a (rho only) wall and a
(rho' + kappa) wall tuned to the same total reactivity give the SAME
diffusion/relaxation signal.  The off-resonance Z-spectrum / MTR is the third
observable that LIFTS the degeneracy -- only the MT wall has a broad bound pool that
saturates off-resonance.  Both act in ONE forward simulate_bloch call on ONE wall.

Heavy Monte-Carlo -> GPU-recommended; marked slow.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from dmipy_sim import Sphere, simulate_bloch, spin_echo, run_bloch_sequence

pytestmark = pytest.mark.slow

R, D = 5e-6, 1e-9
S_OVER_V = 3.0 / R                       # sphere
T2A, T1A = 80e-3, 1.0
# tuned so rho_A = rho_B + kappa_B  ->  same total surface reactivity (same wall loss).
# The binding transverse loss is k_f = kappa_B*(S/V) in the short-T2b limit, independent
# of the dwell, so the dwell (k_r) is free to set the bound-pool size / MTR without
# touching the Panel-A degeneracy.
RHO_A = 2.0e-5                           # pure surface-relaxivity wall
RHO_B, KAPPA_B = 5.0e-6, 1.5e-5        # mixed wall (rho + MT), same total reactivity
DWELL, T2B = 1.0 / 40.0, 1e-5          # k_r = 40 /s -> f_b ~ 0.18


# ── Panel A: the diffusion / relaxation signal is DEGENERATE across rho <-> kappa ─
def test_diffusion_signal_degenerate():
    dt, TE = 6e-5, 40e-3                 # fine enough to resolve the wall for both paths
    seq = spin_echo(TE, dt)
    S_rho = abs(run_bloch_sequence(seq, 6000, D, Sphere(radius=R), T2=T2A,
                                   surface_relaxivity=RHO_A, seed=1)[0])
    S_mt = abs(run_bloch_sequence(seq, 6000, D, Sphere(radius=R), T2=T2A,
                                  surface_relaxivity=RHO_B, kappa_MT=KAPPA_B,
                                  dwell_time=DWELL, T2_bound=T2B, seed=1)[0])
    assert S_rho > 0.1 and S_mt > 0.1                       # both attenuated, not dead
    assert abs(S_rho - S_mt) / S_rho < 0.08                # indistinguishable diffusion signal


# ── Panel B: the off-resonance MTR LIFTS the degeneracy ─────────────────────────
def test_zspectrum_lifts_degeneracy():
    # far off-resonance (6 kHz): the ~2 Hz-wide free-water line is fully spared, but
    # the ~16 kHz-wide bound pool (T2b ~ 10 us) still saturates -> the dip is
    # MT-SPECIFIC, cleanly separating the two walls that gave the same diffusion signal.
    dt, t_sat, w1_hz, offset = 2e-5, 25e-3, 300.0, 6000.0
    n_t = int(round(t_sat / dt)) + 1
    wf = SimpleNamespace(G=np.zeros((1, n_t, 3)), dt=dt)

    def mz(surface_relaxivity, kappa_MT, w1):
        flip = 360.0 * w1 * t_sat
        rf = [{'t_s': t_sat / 2, 'flip_deg': flip, 'axis_deg': 0.0,
               'duration_s': t_sat, 'offset_hz': offset}]
        kw = dict(T2=T2A, T1=T1A, return_mz=True, seed=2,
                  surface_relaxivity=surface_relaxivity)
        if kappa_MT > 0:
            kw.update(kappa_MT=kappa_MT, dwell_time=DWELL, T2_bound=T2B, T1_bound=1.0)
        return float(simulate_bloch(4000, D, wf, Sphere(radius=R), rf, **kw)[1][0])

    mtr_rho = 1.0 - mz(RHO_A, 0.0, w1_hz) / mz(RHO_A, 0.0, 0.0)      # rho-only wall (ref: w1=0)
    mtr_mt = 1.0 - mz(RHO_B, KAPPA_B, w1_hz) / mz(RHO_B, KAPPA_B, 0.0)  # rho + MT wall
    assert mtr_rho < 0.015                                  # narrow free line spared far off-res
    assert mtr_mt > 0.045                                   # MT bound pool saturates off-res
    assert mtr_mt > 4.0 * max(mtr_rho, 1e-3)               # the degeneracy is LIFTED
