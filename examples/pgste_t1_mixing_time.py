"""PGSTE stimulated echo: T1 decay across the mixing time, from the direct simulation.

A pulsed-gradient stimulated echo (PGSTE) stores the magnetisation along the
longitudinal axis during a mixing time ``TM`` (the gradient is off).  While
stored there is no transverse (T2) loss and no surface-relaxivity loss -- only
T1 acts.  The stimulated echo stores half the magnetisation, an idealized 0.5
amplitude factor.  For free diffusion the direct Monte-Carlo forward therefore
returns

    S(TM) = 0.5 * exp(-b*D) * exp(-TM/T1),

so at fixed b-value the mixing-time series is a clean single-exponential T1 decay
riding on the constant diffusion attenuation.  This example sweeps ``TM`` and
compares the simulated signal against that closed form.

CPU-only, self-contained.  Run:  python examples/pgste_t1_mixing_time.py
"""
import numpy as np

from dmipy_sim import simulate, FreeDiffusion, set_b, calc_b, pgste

D = 2e-9            # m^2/s, free diffusivity
N_WALKERS = 60_000  # raise for tighter statistics
SEED = 11
T1 = 800e-3         # s, longitudinal relaxation time
B = 1.0e9           # s/m^2 target b-value (held fixed across the sweep)
DELTA = 5e-3        # s, gradient-lobe duration
TM_VALUES = np.array([5e-3, 20e-3, 40e-3, 80e-3, 160e-3])   # s, mixing times

bvecs = np.array([[1.0, 0.0, 0.0]])


def run():
    sim, model = [], []
    print(f"PGSTE T1 mixing-time sweep  (D={D:.1e} m^2/s, T1={T1*1e3:.0f} ms, "
          f"b={B:.1e} s/m^2)\n")
    print(f"  {'TM [ms]':>8}  {'b_eff [s/m^2]':>14}  {'S_sim':>8}  {'S_model':>8}")
    for TM in TM_VALUES:
        wf = set_b(pgste(delta=DELTA, TM=TM, G_magnitude=1.0, bvecs=bvecs, n_t=800),
                   np.array([B]))
        b_eff = calc_b(wf)[0]
        S = float(simulate(N_WALKERS, D, wf, FreeDiffusion(), seed=SEED,
                           T1=T1, require_gpu=False)[0])
        S_model = 0.5 * np.exp(-b_eff * D) * np.exp(-TM / T1)
        sim.append(S)
        model.append(S_model)
        print(f"  {TM*1e3:8.0f}  {b_eff:14.3e}  {S:8.4f}  {S_model:8.4f}")

    sim, model = np.array(sim), np.array(model)
    print(f"\nmax |S_sim - S_model| = {np.max(np.abs(sim - model)):.4f} "
          f"(MC noise floor ~ {1.0/np.sqrt(N_WALKERS):.4f})")
    return sim, model


def plot(sim, model):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(TM_VALUES * 1e3, model, "-", color="0.4",
            label="0.5·exp(-b·D)·exp(-TM/T1)")
    ax.plot(TM_VALUES * 1e3, sim, "o", color="C0", label="direct simulation")
    ax.set_xlabel("mixing time TM [ms]")
    ax.set_ylabel("stimulated-echo signal")
    ax.set_title("PGSTE T1 decay across the mixing time")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = "pgste_t1_mixing_time.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    sim, model = run()
    plot(sim, model)
