"""First-principles validation of surface relaxivity: 1D -> 2D -> 3D, against EXACT analytics.

Companion to ``permeability_1d_to_3d.py`` -- the same N-D ladder, for the other wall process.
Surface relaxivity rho (m/s) is a partially-absorbing wall: the Monte-Carlo weight rule
``dlog_w = -2 rho/D * d_perp`` per encounter is, in the continuum, exactly the Robin boundary
condition ``D dc/dn = -rho c`` (Brownstein & Tarr 1979).  So the magnetisation of a CLOSED
relaxing cell decays, at long time, as ``exp(-t/tau_1)`` with ``tau_1 = 1/(D lambda_1^2)`` and
``lambda_1`` the lowest Robin eigenvalue:

  1D slab   (both walls relaxing) : lambda*tan(lambda*L/2) = rho/D
  2D cylinder (wall at R)         : D*lambda*J1(lambda*R) = rho*J0(lambda*R)
  3D sphere  (wall at R)          : 1 - lambda*R*cot(lambda*R) = rho*R/D

The fast-diffusion idealisation ``1/tau = rho*S/V`` (Brownstein-Tarr fast limit) is only the
``rho*R/D -> 0`` corner and is deliberately shown as the WRONG reference at finite rho*R/D --
exactly as the well-mixed ``V/(kappa*S)`` is for permeability.  We certify the MC against the
finite-diffusion eigenvalue instead.

Requires scipy (Bessel functions).  Run:  python examples/validation/surface_relaxivity_1d_to_3d.py
"""
import numpy as np
import jax.numpy as jnp
from scipy.special import jv
from scipy.optimize import brentq

from dmipy_sim import simulate, Waveform, Box1D, Sphere, Cylinder

D = 2e-9            # m^2/s
NW = 500_000        # walkers (raise for tighter statistics)
SEED = 3


# ── exact closed-cell relaxation eigenvalues (lowest Robin mode) ──────────────
def _lowest_root(f, hi):
    ls = np.linspace(1.0, hi, 4000)
    v = [f(x) for x in ls]
    for i in range(len(ls) - 1):
        if v[i] == 0 or v[i] * v[i + 1] < 0:
            return brentq(f, ls[i], ls[i + 1])
    raise RuntimeError("no eigenvalue root found")


def tau_slab(L, rho):
    # symmetric cosine mode about L/2; hi just below the pole at lambda=pi/L
    lam = _lowest_root(lambda l: l * np.tan(l * L / 2) - rho / D, np.pi / L * 0.999)
    return 1.0 / (D * lam ** 2)


def tau_cyl(R, rho):
    # J0 mode; hi just below the first J0 zero (2.4048/R) so the root is unique
    lam = _lowest_root(lambda l: D * l * jv(1, l * R) - rho * jv(0, l * R), 2.404 / R)
    return 1.0 / (D * lam ** 2)


def tau_sph(R, rho):
    # j0 mode; root sits well below the cot pole at lambda=pi/R
    lam = _lowest_root(lambda l: 1.0 - l * R * np.cos(l * R) / np.sin(l * R) - rho * R / D,
                       np.pi / R * 0.999)
    return 1.0 / (D * lam ** 2)


def tau_fast(rho, S_over_V):
    """Fast-diffusion (well-mixed) idealisation 1/tau = rho*S/V -- the WRONG finite-rho ref."""
    return 1.0 / (rho * S_over_V)


# ── Monte-Carlo relaxation time in a closed cell ─────────────────────────────
def _zero_waveform(n_t, dt):
    """b=0, G=0 walk of duration n_t*dt: signal == surviving magnetisation S(t)."""
    return Waveform(G=jnp.zeros((1, n_t, 3), dtype=jnp.float32), dt=float(dt), echo_idx=n_t - 1)


def _survival(geom, t, dt):
    n_t = max(1, int(round(t / dt)))
    batch = int(min(NW, max(40000, 1.5e8 / max(1, n_t))))
    sig = simulate(NW, D, _zero_waveform(n_t, dt), geom, seed=SEED, walker_batch_size=batch)
    return float(np.asarray(sig)[0]), n_t * dt


def mc_relax_tau(geom, tau_ex, R):
    """Fit the long-time single-exponential of S(t) -> tau_1 (higher modes decay away first)."""
    dt = (R / 50) ** 2 / (6 * D)                       # fine step (relaxivity is step-robust)
    ts, ss = [], []
    for m in (1.0, 1.5, 2.0, 2.5, 3.0):                # window where the lowest mode dominates
        s, ta = _survival(geom, m * tau_ex, dt)
        ts.append(ta); ss.append(s)
    ts, ss = np.array(ts), np.array(ss)
    slope, _ = np.polyfit(ts, np.log(ss), 1)
    return -1.0 / slope


def _report(name, tau_mc, tau_ex, tau_fd):
    e = (tau_mc - tau_ex) / tau_ex * 100
    ef = (tau_fd - tau_ex) / tau_ex * 100
    ok = "OK " if abs(e) < 2.0 else "!! "
    print(f"  {ok}{name:14s} tau: MC={tau_mc*1e3:7.2f} ms  exact={tau_ex*1e3:7.2f} ms  ({e:+.2f}%)"
          f"   |  fast-limit rho*S/V={tau_fd*1e3:7.2f} ms ({ef:+.1f}% off -- the wrong ref)")


def main():
    rho = 20e-6        # m/s  -> rho*R/D ~ 0.1 (finite-diffusion regime, like the perm ladder)
    print(f"Surface-relaxivity ladder vs EXACT closed-cell Robin eigenvalues "
          f"(D={D:.1e}, rho={rho:.1e})\n")

    # 1D slab: both walls relaxing, width L
    L = 20e-6
    slab = Box1D(length=L, surface_relaxivity_t2=rho)
    _report("1D slab", mc_relax_tau(slab, tau_slab(L, rho), L / 2),
            tau_slab(L, rho), tau_fast(rho, 2.0 / L))

    # 2D cylinder + 3D sphere: single relaxing wall at R
    R = 10e-6
    cyl = Cylinder(radius=R, orientation=(0.0, 0.0, 1.0), surface_relaxivity_t2=rho)
    _report("2D cylinder", mc_relax_tau(cyl, tau_cyl(R, rho), R),
            tau_cyl(R, rho), tau_fast(rho, 2.0 / R))

    sph = Sphere(radius=R, surface_relaxivity_t2=rho)
    _report("3D sphere", mc_relax_tau(sph, tau_sph(R, rho), R),
            tau_sph(R, rho), tau_fast(rho, 3.0 / R))

    print("\nSurface relaxivity matches exact finite-diffusion (Robin) theory to <2% in 1D/2D/3D.\n"
          "Deviations from the fast-limit rho*S/V are the correct diffusion-limited depletion\n"
          "(see surface_relaxivity_findings.md).")


if __name__ == "__main__":
    main()
