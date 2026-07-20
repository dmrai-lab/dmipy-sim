"""Three observables of one wall: surface relaxivity, MT, and the Z-spectrum.

Two white-matter walls are tuned to the SAME total surface reactivity, hence the SAME
diffusion / relaxation signal (Panel A -- degenerate): one is pure surface relaxivity
(rho only), the other splits the same reactivity between rho and an MT bound pool
(rho' + kappa_MT).  The off-resonance Z-spectrum (Panel B) is the third observable
that LIFTS the degeneracy: only the MT wall has a broad, short-T2b bound pool that
saturates far off-resonance, so its Z-spectrum has broad wings the rho wall lacks.

Everything is emergent from the forward vector-Bloch engine (no replay, no
susceptibility): a free spin sticks at the wall, freezes, exchanges; an off-resonance
CW pulse saturates the bound pool and the walk carries the transfer to the free pool.

Run (GPU strongly recommended; set LD_LIBRARY_PATH to the venv's nvidia/*/lib):
    python examples/three_observables.py
Writes examples/three_observables.png.  This is a local figure -- nothing is uploaded.
"""
from types import SimpleNamespace
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dmipy_sim import Sphere, simulate_bloch, spin_echo, run_bloch_sequence

# ── one wall, two ways to make the SAME diffusion signal ────────────────────────
R, D = 5e-6, 1e-9
T2A, T1A = 80e-3, 1.0
RHO_A = 2.0e-5                              # pure surface-relaxivity wall
RHO_B, KAPPA_B = 5.0e-6, 1.5e-5           # rho + MT wall, same total reactivity
DWELL, T2B = 1.0 / 40.0, 1e-5
N = 4000
RHO = dict(color="#2563eb", label="surface relaxivity only  (ρ)")
MT = dict(color="#e0651a", label="surface relaxivity + MT  (ρ' + κ$_{MT}$)")


def _mz(offset_hz, w1_hz, *, kappa_MT, rho):
    t_sat, dt = 25e-3, 2e-5
    n_t = int(round(t_sat / dt)) + 1
    wf = SimpleNamespace(G=np.zeros((1, n_t, 3)), dt=dt)
    rf = [{'t_s': t_sat / 2, 'flip_deg': 360.0 * w1_hz * t_sat, 'axis_deg': 0.0,
           'duration_s': t_sat, 'offset_hz': offset_hz}]
    kw = dict(T2=T2A, T1=T1A, return_mz=True, seed=2, surface_relaxivity=rho)
    if kappa_MT > 0:
        kw.update(kappa_MT=kappa_MT, dwell_time=DWELL, T2_bound=T2B, T1_bound=1.0)
    return float(simulate_bloch(N, D, wf, Sphere(radius=R), rf, **kw)[1][0])


def main():
    # Panel A: the diffusion / relaxation (spin-echo) signal -- degenerate
    se = spin_echo(40e-3, 6e-5)
    S_rho = abs(run_bloch_sequence(se, N, D, Sphere(radius=R), T2=T2A,
                                   surface_relaxivity=RHO_A, seed=1)[0])
    S_mt = abs(run_bloch_sequence(se, N, D, Sphere(radius=R), T2=T2A,
                                  surface_relaxivity=RHO_B, kappa_MT=KAPPA_B,
                                  dwell_time=DWELL, T2_bound=T2B, seed=1)[0])

    # Panel B: the Z-spectrum -- Mz after a CW saturation vs offset (w1 = 300 Hz)
    offsets = np.array([-10000, -6000, -4000, -2500, -1500, -800, -300, 0,
                        300, 800, 1500, 2500, 4000, 6000, 10000], float)
    w1 = 300.0
    ref_rho = _mz(0.0, 0.0, kappa_MT=0.0, rho=RHO_A)
    ref_mt = _mz(0.0, 0.0, kappa_MT=KAPPA_B, rho=RHO_B)
    Z_rho = np.array([_mz(o, w1, kappa_MT=0.0, rho=RHO_A) for o in offsets]) / ref_rho
    Z_mt = np.array([_mz(o, w1, kappa_MT=KAPPA_B, rho=RHO_B) for o in offsets]) / ref_mt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.3))

    axA.bar([0, 1], [S_rho, S_mt], color=[RHO["color"], MT["color"]], width=0.6)
    for x, s in zip((0, 1), (S_rho, S_mt)):
        axA.text(x, s + 0.01, f"{s:.3f}", ha="center", va="bottom", fontsize=11)
    axA.set_xticks([0, 1])
    axA.set_xticklabels(["ρ only", "ρ' + κ$_{MT}$"])
    axA.set_ylabel("spin-echo signal $|S|$  (TE = 40 ms)")
    axA.set_ylim(0, max(S_rho, S_mt) * 1.25)
    axA.set_title("A  Diffusion / relaxation signal: DEGENERATE", fontsize=11, loc="left")

    axB.plot(offsets / 1e3, Z_rho, "o-", color=RHO["color"], label=RHO["label"], lw=2, ms=5)
    axB.plot(offsets / 1e3, Z_mt, "s-", color=MT["color"], label=MT["label"], lw=2, ms=5)
    axB.set_xlabel("saturation offset (kHz)")
    axB.set_ylabel("$M_z / M_0$  (Z-spectrum)")
    axB.set_ylim(0, 1.05)
    axB.axhline(1.0, color="0.7", lw=0.8, ls=":")
    axB.legend(frameon=False, fontsize=9, loc="lower center")
    axB.set_title("B  Off-resonance Z-spectrum: LIFTS the degeneracy", fontsize=11, loc="left")

    fig.suptitle("Three observables of one wall (emergent, forward vector-Bloch)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(os.path.dirname(__file__), "three_observables.png")
    fig.savefig(out, dpi=140)
    print(f"diffusion signal:  rho={S_rho:.3f}  rho+MT={S_mt:.3f}  "
          f"(rel diff {abs(S_rho-S_mt)/S_rho*100:.1f}%)")
    print(f"Z-spectrum wings @ 6 kHz:  rho={Z_rho[-2]:.3f}  rho+MT={Z_mt[-2]:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
