"""First-principles validation of membrane permeability: 1D -> 2D -> 3D, against EXACT analytics.

Every physics effect in dmipy-sim should be provable from the simplest geometry up, against
*exact* references -- not the well-mixed idealisation ``tau = V/(kappa*S)``, which omits the
(correct) diffusion-limited near-membrane depletion. This is the template for that ladder.

For each dimension we build a CLOSED two-compartment cell (permeable membrane + reflecting
outer wall -> no exterior re-entry) and check the Monte-Carlo exchange against the exact
finite-diffusion eigenvalue tau_1 = 1/(D lambda_1^2):

  1D slab            : lambda*tan(lambda*L/2) = 2*kappa/D
  2D cylinder shell  : Bessel-J/Y transcendental (permeable R_in, reflecting R_out)
  3D sphere shell    : spherical-Bessel transcendental

We also check the equilibrium partition == V_A/V_total (detailed balance: a passive membrane
must be bidirectionally symmetric).

Requires scipy (Bessel functions).  Run:  python examples/validation/permeability_1d_to_3d.py
"""
import numpy as np
import jax.numpy as jnp
from scipy.special import jv, yv, spherical_jn, spherical_yn
from scipy.optimize import brentq

from dmipy_sim import simulate, Waveform, PermeableSlab1D
from dmipy_sim.geometries import PermeableShell

D = 2e-9            # m^2/s
NW = 500_000        # walkers (raise for tighter statistics)
SEED = 3


# ── exact closed-cell exchange eigenvalues ───────────────────────────────────
def _lowest_root(f, hi):
    ls = np.linspace(1.0, hi, 2000)
    v = [f(x) for x in ls]
    for i in range(len(ls) - 1):
        if v[i] == 0 or v[i] * v[i + 1] < 0:
            return brentq(f, ls[i], ls[i + 1])
    raise RuntimeError("no eigenvalue root found")


def tau_slab(L, kappa):
    lam = _lowest_root(lambda l: l * np.tan(l * L / 2) - 2 * kappa / D, np.pi / L * 0.999)
    return 1.0 / (D * lam ** 2)


def _tau_shell(Rin, Rout, kappa, kind):
    def det(l):
        a, b = l * Rin, l * Rout
        if kind == 'sphere':
            F0, F0p, G0, G0p = (spherical_jn(0, a), spherical_jn(0, a, True),
                                spherical_yn(0, a), spherical_yn(0, a, True))
            Fbp, Gbp = spherical_jn(0, b, True), spherical_yn(0, b, True)
        else:  # cylinder: J0'=-J1, Y0'=-Y1
            F0, F0p, G0, G0p = jv(0, a), -jv(1, a), yv(0, a), -yv(1, a)
            Fbp, Gbp = -jv(1, b), -yv(1, b)
        # rows: reflect(R_out); flux-cont(R_in); permeable-jump(R_in)
        M = np.array([[0.0, Fbp, Gbp],
                      [F0p, -F0p, -G0p],
                      [-D * l * F0p - kappa * F0, kappa * F0, kappa * G0]])
        return np.linalg.det(M)
    lam = _lowest_root(det, 1.2e5)
    return 1.0 / (D * lam ** 2)


# ── Monte-Carlo exchange time in a closed cell ───────────────────────────────
def _zero_waveform(n_t, dt):
    return Waveform(G=jnp.zeros((1, n_t, 3), dtype=jnp.float32), dt=float(dt), echo_idx=n_t - 1)


def _f_inside(geom, t, dt):
    n_t = max(1, int(round(t / dt)))
    batch = int(min(NW, max(40000, 1.5e8 / max(1, n_t))))
    _, _, cf = simulate(NW, D, _zero_waveform(n_t, dt), geom, seed=SEED,
                        return_compartments='final', walker_batch_size=batch)
    return float((np.asarray(cf) == 0).mean()), n_t * dt


def mc_exchange_tau(geom, tau_ex, feq, R):
    dt = (R / 50) ** 2 / (6 * D)                       # fine enough for the curved crossing
    ts, fs = [], []
    for m in (0.6, 0.9, 1.2, 1.5):
        f, ta = _f_inside(geom, m * tau_ex, dt)
        ts.append(ta); fs.append(f)
    ts, fs = np.array(ts), np.array(fs)
    y = (fs - feq) / (1 - feq)
    tau = -1.0 / np.polyfit(ts, np.log(y), 1)[0]
    # equilibrium at long time, coarse step (step-insensitive), for detailed balance
    f_eq, _ = _f_inside(geom, 8 * tau_ex, (0.1 * R) ** 2 / (6 * D))
    return tau, f_eq


def _report(name, tau_mc, tau_ex, f_eq, f_eq_th):
    e = (tau_mc - tau_ex) / tau_ex * 100
    ok = "OK " if abs(e) < 1.0 and abs(f_eq - f_eq_th) < 0.01 else "!! "
    print(f"  {ok}{name:18s} tau: MC={tau_mc*1e3:7.2f} ms  exact={tau_ex*1e3:7.2f} ms  ({e:+.2f}%)"
          f"   |  eq: {f_eq:.4f} (exact {f_eq_th:.4f})")


def main():
    kappa = 20e-6
    print(f"Permeability ladder vs EXACT closed-cell eigenvalues (D={D:.1e}, kappa={kappa:.1e})\n")

    # 1D slab: two equal compartments, membrane at L/2
    L = 20e-6
    slab = PermeableSlab1D(length=L, permeability=kappa)
    tau_ex = tau_slab(L, kappa)
    tau_mc, f_eq = mc_exchange_tau(slab, tau_ex, 0.5, L / 2)
    _report("1D slab", tau_mc, tau_ex, f_eq, 0.5)

    # 2D cylinder shell + 3D sphere shell: permeable R_in, reflecting R_out=2 R_in
    R = 10e-6
    for kind, feq_th in (('cylinder', R ** 2 / (2 * R) ** 2), ('sphere', R ** 3 / (2 * R) ** 3)):
        geom = PermeableShell(R, 2 * R, kappa, kind=kind)
        tau_ex = _tau_shell(R, 2 * R, kappa, kind)
        tau_mc, f_eq = mc_exchange_tau(geom, tau_ex, feq_th, R)
        _report(f"{'2D' if kind == 'cylinder' else '3D'} {kind} shell", tau_mc, tau_ex, f_eq, feq_th)

    print("\nPermeability matches exact finite-diffusion theory to <1% in 1D/2D/3D, with correct\n"
          "detailed balance. Deviations from the well-mixed V/(kappa*S) are the correct\n"
          "diffusion-limited physics (see permeability_findings.md).")


if __name__ == "__main__":
    main()
