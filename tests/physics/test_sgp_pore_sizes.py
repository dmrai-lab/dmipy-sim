"""SGP pore-size recovery tests.

Under short gradient pulse (SGP) + long diffusion time, the MC signal must
exactly match the Fourier transform of the uniform pore density:

  Box1D (separation d):
    E(q) = (2·sin(q·d/2) / (q·d))²
    first zero at q·d = 2π  →  q_min = 2π/d
    RTPP = (1/π) ∫₀^∞ E(q) dq = 1/d

  Cylinder (radius R, q perpendicular to axis):
    E(q) = (2·J₁(q·R) / (q·R))²
    first zero at q·R = 3.8317  →  q_min = 3.8317/R
    RTAP = (1/2π) ∫₀^∞ E(q)·q dq = 1/(πR²)

  Sphere (radius R):
    E(q) = (3·j₁(q·R) / (q·R))²   where j₁(x) = (sin x − x cos x)/x²
    first zero at q·R = 4.4934  →  q_min = 4.4934/R
    RTOP = (1/2π²) ∫₀^∞ E(q)·q² dq = 3/(4πR³)

where q = γ·G·δ in rad/m (NOT cycles/m).  This is the convention used
by the dmipy-sim physics engine (dphi = γ·dt·G·r) so no 2π factors appear
in the Bessel arguments.

SGP conditions enforced per test:
  δ = 0.001 × R²/D  (0.1% of the characteristic time — strong SGP)
  Δ = 2.5 × R²/D    (the slowest non-uniform mode of the pore, e^{-μ²DΔ/R²} with μ = 1.84 (cylinder), 2.08
                     (sphere), has decayed to 2e-4 and 2e-5, a tenth of the Monte Carlo error or less, so the two
                     pulses see independent uniform positions)
  dt = δ            (one sample per pulse; the walk keeps a uniform density at any step, the pulse width is what
                     the sampling resolves)

The problem is scale-free in R²/D, so one pore size per geometry.

Size: the 32 q-points share the walkers, so they are correlated; their covariance is known in closed form
(``_form_factor_covariance``). The unweighted least-squares radius then has the standard deviation
``_fit_sd / sqrt(N)`` (0.84 and 0.87 for the cylinder and the sphere; a simulation of the closed-form walkers gives
0.84 and 0.85), and N is the walker count at which a 4 % error in R fails the 2 % tolerance with the margin the noise
passes it: ``n_resolving(0.04, sd)``, 19,200 and 20,600 walkers. Every q-point must also lie within the Bonferroni
quantile of its own standard error. The return probabilities RTAP = 1/(πR²) and RTOP = 3/(4πR³) are functions of the
fitted radius alone and are not asserted separately.

Both tests run the oracle ``simulate_fused`` (the full-resolution walk): the waveform's gradient is two pulses of
0.1 % of the diffusion time, far above the temporal band of a replay pack.

Each is 5e7 walker-steps, about 40 s on four CPU cores.
"""

import numpy as np
import pytest
from pathlib import Path
from scipy.special import j1
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dmipy_sim import Cylinder, Sphere
from dmipy_sim.engine._fused import simulate_fused
from dmipy_replay import pgse
from dmipy_replay.acquisition.scanner_sequence import ScannerSequence
import jax.numpy as jnp

from tests.conftest import n_resolving, se_tol, z_family

# ── Constants ────────────────────────────────────────────────────────────────
D = 2e-9          # m²/s — intrinsic diffusivity (standard dmipy-sim value)
SEED = 42
R_ERROR_RESOLVED = 0.04     # the pore-size error that must fail the 2 % tolerance
N_Q = 32          # q-points per sweep
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ── SGP regime helpers ────────────────────────────────────────────────────────

def _sgp_params(pore_size_m):
    """Return (delta, DELTA, n_t) that satisfy SGP + long-time conditions.

    δ = 0.001 × τ   where τ = pore_size² / D   (strong SGP condition)
    Δ = 2.5 × τ                                (long-time limit, see the module docstring)
    n_t: dt = δ, one sample per pulse.
    """
    tau = pore_size_m ** 2 / D
    delta = 0.001 * tau
    DELTA = 2.5 * tau
    T_total = DELTA + delta
    n_t = int(T_total / delta) + 2
    return delta, DELTA, n_t


def _build_sgp_waveform(q_values, bvec, delta, DELTA, n_t):
    """Build a multi-measurement PGSE waveform for a q-sweep.

    Parameters
    ----------
    q_values : (n_q,) array in rad/m  (q = γ·G·δ)
    bvec     : (3,) unit vector for gradient direction
    delta, DELTA : floats, seconds
    n_t      : int, time resolution

    Returns
    -------
    ScannerSequence with G scaled so each measurement corresponds to q_values[i].
    """
    n_q = len(q_values)
    bvecs = np.tile(bvec, (n_q, 1)).astype(np.float32)

    # Template with G_magnitude=1 T/m; actual magnitude set by scaling below.
    wf_template = pgse(bvecs, delta, DELTA, gradient_strengths=1.0, n_t=n_t, slew_rate=np.inf)
    b_template = wf_template.b()  # (n_q,) — same for all measurements

    # b = q² · (Δ - δ/3) — exact for PGSE in the SGP limit
    b_target = q_values ** 2 * (DELTA - delta / 3.0)

    # Scale G: G_new = G_template · √(b_target / b_template)
    # For q=0 (b_target=0), set scale=0 so G=0.
    safe_b = np.where(b_template > 0, b_template, 1.0)
    scale = np.sqrt(np.where(b_target > 0, b_target / safe_b, 0.0)
                    ).astype(np.float32)

    G = np.array(wf_template.G)  # (n_q, n_t, 3)
    G_scaled = G * scale[:, None, None]
    return ScannerSequence(G=jnp.array(G_scaled.astype(np.float32)),
                    dt=wf_template.dt,
                    readout=wf_template.readout)


# ── Theoretical form factors ──────────────────────────────────────────────────
# q is in rad/m throughout (q = γ·G·δ, NOT q = γGδ/(2π)).

def e_slab(q, d):
    """E(q) = (2·sin(q·d/2) / (q·d))²   (Box1D, uniform density on [0,d]).

    First zero at q·d = 2π  (q = 2π/d).
    Limit as q→0: 1.
    """
    qd = q * d
    return np.where(np.abs(qd) < 1e-12, 1.0, (2 * np.sin(qd / 2) / qd) ** 2)


def e_cylinder(q, R):
    """E(q) = (2·J₁(q·R) / (q·R))²   (Cylinder, uniform disk, q ⊥ axis).

    First zero at q·R = 3.8317  (q = 3.8317/R).
    Limit as q→0: 1.
    """
    qR = q * R
    return np.where(qR < 1e-12, 1.0, (2 * j1(qR) / qR) ** 2)


def e_sphere(q, R):
    """E(q) = (3·j₁(q·R) / (q·R))²   (Sphere, uniform ball).

    j₁(x) = (sin x − x cos x) / x²  (spherical Bessel, order 1).
    First zero at q·R = 4.4934  (q = 4.4934/R).
    Limit as q→0: 1.
    """
    qR = q * R
    j1v = np.where(np.abs(qR) < 1e-12,
                   qR / 3.0,
                   (np.sin(qR) - qR * np.cos(qR)) / qR ** 2)
    return np.where(qR < 1e-12, 1.0, (3 * j1v / qR) ** 2)


# ── Statistics of the q-sweep ─────────────────────────────────────────────────

def _form_factor_covariance(e_fn, R, q):
    """The covariance of one walker's cos(q_i X) and cos(q_j X), X the displacement between the two pulses.

    At long time X is the difference of two independent uniform positions, so ``<cos(q X)> = E(q)`` and the product
    of cosines is ``(cos((q_i - q_j) X) + cos((q_i + q_j) X)) / 2``: the covariance is
    ``(E(|q_i - q_j|) + E(q_i + q_j)) / 2 - E(q_i) E(q_j)``. The mean over N walkers has 1/N of it."""
    with np.errstate(all="ignore"):
        e = e_fn(q, R)
        qi, qj = q[:, None], q[None, :]
        return (e_fn(np.abs(qi - qj), R) + e_fn(qi + qj, R)) / 2.0 - np.outer(e, e)


def _fit_sd(e_fn, R, q):
    """``sqrt(N) sigma(R_fit) / R``: the standard deviation of the unweighted least-squares radius from N walkers,
    ``a^T C a`` with the weights ``a = g / (g . g)`` of the linearised fit (``g = dE/dR``) and C the covariance of the
    q-points."""
    h = 1e-6 * R
    with np.errstate(all="ignore"):
        g = (e_fn(q, R + h) - e_fn(q, R - h)) / (2.0 * h)
    a = g / (g @ g)
    return float(np.sqrt(a @ _form_factor_covariance(e_fn, R, q) @ a)) / R


# ── Diagnostic figure ─────────────────────────────────────────────────────────

def _save_figure(q_values, E_mc, sigma_q, E_theory_fn, q_min_true,
                 geometry_name, size_um, size_fit_um, z_max):
    """Save a diagnostic figure (even if the test passes)."""
    q_units = q_values / q_min_true   # normalised to first zero

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.errorbar(q_units[1:], E_mc[1:], yerr=sigma_q, fmt="o", ms=2,
                alpha=0.6, label="MC", color="steelblue")
    q_fine = np.linspace(0, q_units[-1] * 1.02, 500)
    ax.plot(q_fine, E_theory_fn(q_fine * q_min_true), "r-", lw=2, label="Theory")
    ax.axvline(1.0, color="green", ls="--", lw=1.2, label="q_min (true)")
    ax.set_xlabel("q / q_min")
    ax.set_ylabel("E(q)")
    ax.set_title(
        f"{geometry_name} {size_um:.0f} μm\n"
        f"fit = {size_fit_um:.2f} μm   max |z| = {z_max:.2f}"
    )
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.1)

    fig.tight_layout()
    fname = FIGURES_DIR / f"sgp_{geometry_name.lower()}_{size_um:.0f}um.png"
    fig.savefig(fname, dpi=100)
    plt.close(fig)


def _recover_pore_size(e_fn, geometry, R, q_first_zero, name):
    """Walk the pore, sweep q, and assert (1) every q-point is within the Bonferroni quantile of its own standard error
    of the closed form and (2) the least-squares radius is within ``se_tol`` of the true one."""
    delta, DELTA, n_t = _sgp_params(R)
    q_values = np.linspace(0.0, 2.0 * q_first_zero, N_Q)
    q = q_values[1:]                                     # q = 0 is exactly 1, with no spread
    sd_R = _fit_sd(e_fn, R, q)
    n = n_resolving(R_ERROR_RESOLVED, sd_R)
    wf = _build_sgp_waveform(q_values, np.array([1.0, 0.0, 0.0]), delta, DELTA, n_t)
    E_mc = np.array(simulate_fused(n, D, wf, geometry, seed=SEED), dtype=np.float64)
    with np.errstate(all="ignore"):                      # q = 0 is the removable 0/0 of the form factor
        E_theory = e_fn(q_values, R)

    sigma_q = np.sqrt(np.diag(_form_factor_covariance(e_fn, R, q)) / n)
    z = np.abs(E_mc[1:] - E_theory[1:]) / sigma_q
    popt, _ = curve_fit(e_fn, q, E_mc[1:], p0=[R], bounds=(R * 0.5, R * 2.0))
    R_fit = float(popt[0])
    _save_figure(q_values, E_mc, sigma_q, lambda x: e_fn(x, R), q_first_zero, name, R * 1e6, R_fit * 1e6, z.max())

    assert z.max() < z_family(len(q)), (
        f"{name} R={R * 1e6} μm: E(q) at q/q_min = {q[z.argmax()] / q_first_zero:.2f} is {z.max():.2f} standard errors "
        f"from the closed form (limit {z_family(len(q)):.2f}) — systematic physics error. Check units, gradient direction.")
    err_R = abs(R_fit - R) / R
    assert err_R < se_tol(n, sd_R), (
        f"{name} R={R * 1e6} μm: fit recovered R = {R_fit * 1e6:.3f} μm, relative error = {err_R:.4f} "
        f"> {se_tol(n, sd_R):.4f} (N = {n})")


# ─────────────────────────────────────────────────────────────────────────────
# TestCylinderSGP
# ─────────────────────────────────────────────────────────────────────────────

class TestCylinderSGP:
    """Pore size recovery for cylinders (Cylinder).

    Cylinder axis along z; gradient along x (perpendicular).
    SGP formula: E(q) = (2·J₁(q·R) / (q·R))²
    First zero: q·R = 3.8317  →  q_min = 3.8317/R.
    RTAP = (1/2π) ∫₀^∞ E(q)·q dq = 1/(πR²).
    """

    @pytest.mark.parametrize("R_um", [5.0])
    def test_pore_size_recovery(self, R_um):
        R = R_um * 1e-6  # metres
        _recover_pore_size(e_cylinder, Cylinder(radius=R, orientation=[0, 0, 1.0]), R, 3.8317 / R, "Cylinder")


# ─────────────────────────────────────────────────────────────────────────────
# TestSphereSGP
# ─────────────────────────────────────────────────────────────────────────────

class TestSphereSGP:
    """Pore size recovery for spheres (Sphere).

    SGP formula: E(q) = (3·j₁(q·R) / (q·R))²
    First zero: q·R = 4.4934  →  q_min = 4.4934/R.
    RTOP = (1/2π²) ∫₀^∞ E(q)·q² dq = 3/(4πR³).
    """

    @pytest.mark.parametrize("R_um", [6.0])
    def test_pore_size_recovery(self, R_um):
        R = R_um * 1e-6  # metres
        _recover_pore_size(e_sphere, Sphere(radius=R), R, 4.4934 / R, "Sphere")
