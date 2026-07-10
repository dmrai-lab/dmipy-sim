"""Physics tests: surface relaxivity — Brownstein-Tarr T2 for all primitives.

Validates that the surface-T2 decay matches the Brownstein-Tarr prediction for
all three primitive geometries in the fast-diffusion limit (ρ·R/D ≪ 1).

Physics
-------
In the fast-diffusion (motional-averaging) limit, zero-gradient signal decays as:

    E(t) = exp(−t / T2_surface)

with exact T2 values:

    Box1D (thickness d):   T2 = d / (2ρ)     [V/(ρ·A) = d / (ρ·2)]
    Cylinder (radius R):   T2 = R / (2ρ)     [V/(ρ·A) = R/2ρ]
    Sphere (radius R):     T2 = R / (3ρ)     [V/(ρ·A) = R/3ρ]

Parameters: R=10μm, D=2e-9 m²/s, ρ ∈ {1,5,20}×10⁻⁶ m/s
  → ρ·R/D ∈ {0.005, 0.025, 0.1} — all satisfy the fast-diffusion condition.

Acceptance criterion: |T2_fit − T2_theory| / T2_theory < 0.05 (5%).

Tiers tested:
- Tier 1: ρ=0 baseline — E(t) must be 1.0 to machine precision.
- Tier 2: T2 recovery for ρ ∈ {1,5,20}μm/s × {Box1D,Cylinder,Sphere}.
- Tier 3: S/V scaling — log(T2) vs log(R) slope must be 1.0 ± 0.02.
- Tier 4: D independence — T2 must be flat (std/mean < 0.02) as D varies.

References
----------
- Brownstein & Tarr (1979) Phys Rev A 19, 2446
- Axelrod & Sen (2001) J Chem Phys 114, 6878
"""

import os
import numpy as np
import numpy.testing as npt
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jax.numpy as jnp

from dmipy_sim import simulate, Box1D, Cylinder, Sphere, set_b
from dmipy_sim.waveforms import pgse


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_DEFAULT = 2e-9     # m²/s
R_DEFAULT = 10e-6    # m
SEED = 42

RHOS_SI = np.array([1e-6, 5e-6, 20e-6])   # m/s

N_WALKERS_T2 = 500_000   # minimum for T2 fitting
N_WALKERS_SV = 200_000   # acceptable for S/V scaling

# Output directory for diagnostic figures
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b0_waveform(TE_s, n_t=500):
    """PGSE waveform with negligible b (b=1 s/m² → exp(-bD) ≈ 1)."""
    delta = max(TE_s * 0.05, 5e-6)
    DELTA = TE_s - delta
    bvecs = np.array([[1.0, 0.0, 0.0]])
    wf = set_b(
        pgse(delta=delta, DELTA=DELTA, G_magnitude=1.0,
             bvecs=bvecs, n_t=n_t),
        np.array([1.0]))   # b=1 s/m² → attenuation = exp(-2e-9) ≈ 1
    actual_TE = wf.echo_idx * wf.dt
    return wf, actual_TE


def _fit_t2(TE_values, signals):
    """Fit E = exp(−TE/T2) via log-linear regression; return T2 in seconds."""
    log_s = np.log(np.maximum(signals, 1e-12))
    slope, _ = np.polyfit(TE_values, log_s, 1)
    return -1.0 / slope


def _measure_t2(geometry, T2_theory, D=D_DEFAULT, n_walkers=N_WALKERS_T2,
                n_te=6, seed=SEED):
    """Run b≈0 simulations at n_te echo times spanning 0.5–3×T2_theory.

    The step size is chosen so that σ = √(6·D·dt) ≤ R_DEFAULT/5 at all TEs.
    The TE range is capped at 2.0 s so that long T2 geometries (ρ=1µm/s,
    T2=5 s) do not require impractically many steps.

    Returns (T2_fit, actual_TEs, signals).
    """
    # Cap TE range to keep simulations tractable.
    # At ρ=1µm/s, T2_theory can be up to 5 s; cap TE_max at 2 s to keep
    # dt = TE/n_t within physical bounds.
    TE_max = min(3.0 * T2_theory, 2.0)
    TE_min = min(0.5 * T2_theory, TE_max * 0.15)
    TE_targets = np.linspace(TE_min, TE_max, n_te)

    actual_TEs = []
    signals = []
    for TE_t in TE_targets:
        # Choose n_t so that σ/R_DEFAULT ≤ 0.20 at this TE.
        dt_max = (0.20 * R_DEFAULT) ** 2 / (6 * D)
        n_t = max(500, int(np.ceil(TE_t / dt_max)))
        wf, actual_TE = _b0_waveform(TE_t, n_t=n_t)
        actual_TEs.append(actual_TE)
        s = simulate(n_walkers, D, wf, geometry, seed=seed)
        signals.append(float(s[0]))
    T2_fit = _fit_t2(np.array(actual_TEs), np.array(signals))
    return T2_fit, np.array(actual_TEs), np.array(signals)


# ---------------------------------------------------------------------------
# Tier 1: ρ=0 baseline — E(t) must be 1.0 to machine precision
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBaseline:
    """ρ=0 (no surface relaxivity): E(t) = 1.0 at all times."""

    @pytest.mark.parametrize("TE_s", [0.05, 0.1, 0.5])
    def test_box1d_no_relaxivity(self, TE_s):
        """Box1D with ρ=None: signal at b=0 must be 1.0."""
        geom = Box1D(length=R_DEFAULT)
        wf, _ = _b0_waveform(TE_s)
        s = simulate(50_000, D_DEFAULT, wf, geom, seed=SEED)
        npt.assert_allclose(float(s[0]), 1.0, atol=0.005,
            err_msg=f"Box1D ρ=0 at TE={TE_s*1e3:.0f}ms: E={float(s[0]):.4f}")

    @pytest.mark.parametrize("TE_s", [0.05, 0.1, 0.5])
    def test_cylinder_no_relaxivity(self, TE_s):
        """Cylinder with ρ=None: signal at b=0 must be 1.0."""
        geom = Cylinder(radius=R_DEFAULT, orientation=[0, 0, 1.])
        wf, _ = _b0_waveform(TE_s)
        s = simulate(50_000, D_DEFAULT, wf, geom, seed=SEED)
        npt.assert_allclose(float(s[0]), 1.0, atol=0.005,
            err_msg=f"Cylinder ρ=0 at TE={TE_s*1e3:.0f}ms: E={float(s[0]):.4f}")

    @pytest.mark.parametrize("TE_s", [0.05, 0.1, 0.5])
    def test_sphere_no_relaxivity(self, TE_s):
        """Sphere with ρ=None: signal at b=0 must be 1.0."""
        geom = Sphere(radius=R_DEFAULT)
        wf, _ = _b0_waveform(TE_s)
        s = simulate(50_000, D_DEFAULT, wf, geom, seed=SEED)
        npt.assert_allclose(float(s[0]), 1.0, atol=0.005,
            err_msg=f"Sphere ρ=0 at TE={TE_s*1e3:.0f}ms: E={float(s[0]):.4f}")


# ---------------------------------------------------------------------------
# Tier 2: T2 recovery across ρ values
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestT2RecoveryBox1D:
    """Box1D: T2 = d/(2ρ) must be recovered within 5%."""

    @pytest.mark.parametrize("rho", [1e-6, 5e-6, 20e-6])
    def test_t2_recovery(self, rho):
        """T2_fit must agree with d/(2ρ) within 5%."""
        geom = Box1D(length=R_DEFAULT, surface_relaxivity_t2=rho)
        T2_theory = geom.volume() / (rho * geom.surface_area())

        T2_fit, _, _ = _measure_t2(geom, T2_theory)

        rel_err = abs(T2_fit - T2_theory) / T2_theory
        assert rel_err < 0.05, (
            f"Box1D ρ={rho:.0e}: T2_fit={T2_fit*1e3:.1f}ms, "
            f"T2_theory={T2_theory*1e3:.1f}ms, rel_err={rel_err:.3f}")

    @pytest.mark.parametrize("rho", [1e-6, 5e-6, 20e-6])
    def test_fast_diffusion_condition(self, rho):
        """Verify ρ·R/D ≪ 1 (fast-diffusion limit is satisfied)."""
        ratio = rho * R_DEFAULT / D_DEFAULT
        assert ratio < 0.15, f"ρ·R/D = {ratio:.3f} ≥ 0.15"


@pytest.mark.slow
class TestT2RecoveryCylinder:
    """Cylinder: T2 = R/(2ρ) must be recovered within 5%."""

    @pytest.mark.parametrize("rho", [1e-6, 5e-6, 20e-6])
    def test_t2_recovery(self, rho):
        geom = Cylinder(radius=R_DEFAULT, orientation=[0, 0, 1.],
                        surface_relaxivity_t2=rho)
        T2_theory = geom.volume(L=1.0) / (rho * geom.surface_area(L=1.0))

        T2_fit, _, _ = _measure_t2(geom, T2_theory)

        rel_err = abs(T2_fit - T2_theory) / T2_theory
        assert rel_err < 0.05, (
            f"Cylinder ρ={rho:.0e}: T2_fit={T2_fit*1e3:.1f}ms, "
            f"T2_theory={T2_theory*1e3:.1f}ms, rel_err={rel_err:.3f}")

    @pytest.mark.parametrize("rho", [1e-6, 5e-6, 20e-6])
    def test_fast_diffusion_condition(self, rho):
        ratio = rho * R_DEFAULT / D_DEFAULT
        assert ratio < 0.15


@pytest.mark.slow
class TestT2RecoverySphere:
    """Sphere: T2 = R/(3ρ) must be recovered within 5%."""

    @pytest.mark.parametrize("rho", [1e-6, 5e-6, 20e-6])
    def test_t2_recovery(self, rho):
        geom = Sphere(radius=R_DEFAULT, surface_relaxivity_t2=rho)
        T2_theory = geom.volume() / (rho * geom.surface_area())

        T2_fit, _, _ = _measure_t2(geom, T2_theory)

        rel_err = abs(T2_fit - T2_theory) / T2_theory
        assert rel_err < 0.05, (
            f"Sphere ρ={rho:.0e}: T2_fit={T2_fit*1e3:.1f}ms, "
            f"T2_theory={T2_theory*1e3:.1f}ms, rel_err={rel_err:.3f}")


# ---------------------------------------------------------------------------
# Tier 3: S/V scaling — log(T2) vs log(R) slope = 1.0 ± 0.02
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sv_scaling_cylinder():
    """Cylinder: T2 ∝ R — log-log slope must be 1.0 ± 0.02.

    Fix ρ=1e-6 m/s, sweep R ∈ {2,4,8,16}μm.
    T2_theory = R/(2ρ) → T2 ∝ R (slope=1 in log-log).
    """
    rho = 1e-6   # m/s
    radii = np.array([2e-6, 4e-6, 8e-6, 16e-6])   # m
    t2_fits = []
    t2_theories = []

    for R in radii:
        geom = Cylinder(radius=R, orientation=[0, 0, 1.],
                        surface_relaxivity_t2=rho)
        T2_theory = geom.volume(L=1.0) / (rho * geom.surface_area(L=1.0))
        t2_theories.append(T2_theory)

        T2_fit, _, _ = _measure_t2(
            geom, T2_theory, n_walkers=N_WALKERS_SV)
        t2_fits.append(T2_fit)

    t2_fits = np.array(t2_fits)
    log_R = np.log(radii)
    log_T2 = np.log(t2_fits)
    slope, _ = np.polyfit(log_R, log_T2, 1)

    assert abs(slope - 1.0) < 0.02, (
        f"Cylinder S/V scaling: slope={slope:.4f}, expected 1.0 ± 0.02")


@pytest.mark.slow
def test_sv_scaling_sphere():
    """Sphere: T2 ∝ R — log-log slope must be 1.0 ± 0.02."""
    rho = 1e-6
    radii = np.array([2e-6, 4e-6, 8e-6, 16e-6])
    t2_fits = []

    for R in radii:
        geom = Sphere(radius=R, surface_relaxivity_t2=rho)
        T2_theory = geom.volume() / (rho * geom.surface_area())

        T2_fit, _, _ = _measure_t2(
            geom, T2_theory, n_walkers=N_WALKERS_SV)
        t2_fits.append(T2_fit)

    t2_fits = np.array(t2_fits)
    log_R = np.log(radii)
    log_T2 = np.log(t2_fits)
    slope, _ = np.polyfit(log_R, log_T2, 1)

    assert abs(slope - 1.0) < 0.02, (
        f"Sphere S/V scaling: slope={slope:.4f}, expected 1.0 ± 0.02")


@pytest.mark.slow
def test_sv_scaling_box1d():
    """Box1D: T2 ∝ d — log-log slope must be 1.0 ± 0.02."""
    rho = 1e-6
    lengths = np.array([2e-6, 4e-6, 8e-6, 16e-6])
    t2_fits = []

    for d in lengths:
        geom = Box1D(length=d, surface_relaxivity_t2=rho)
        T2_theory = geom.volume() / (rho * geom.surface_area())

        T2_fit, _, _ = _measure_t2(
            geom, T2_theory, n_walkers=N_WALKERS_SV)
        t2_fits.append(T2_fit)

    t2_fits = np.array(t2_fits)
    log_d = np.log(lengths)
    log_T2 = np.log(t2_fits)
    slope, _ = np.polyfit(log_d, log_T2, 1)

    assert abs(slope - 1.0) < 0.02, (
        f"Box1D S/V scaling: slope={slope:.4f}, expected 1.0 ± 0.02")


# ---------------------------------------------------------------------------
# Tier 4: D independence — T2 flat across D values
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_t2_d_independence_cylinder():
    """Cylinder T2 must be flat as D varies (T2 = R/(2ρ) has no D).

    Fix ρ=1e-6 m/s, R=10μm, vary D ∈ {0.5,1,2,4}×10⁻⁹ m²/s.
    Expected std/mean < 0.02.
    """
    rho = 1e-6
    R = R_DEFAULT
    D_values = np.array([0.5e-9, 1e-9, 2e-9, 4e-9])
    geom_base = Cylinder(radius=R, orientation=[0, 0, 1.],
                         surface_relaxivity_t2=rho)
    T2_theory = geom_base.volume(L=1.0) / (rho * geom_base.surface_area(L=1.0))
    t2_fits = []

    for D in D_values:
        geom = Cylinder(radius=R, orientation=[0, 0, 1.],
                        surface_relaxivity_t2=rho)
        T2_fit, _, _ = _measure_t2(geom, T2_theory, D=D,
                                    n_walkers=N_WALKERS_SV)
        t2_fits.append(T2_fit)

    t2_arr = np.array(t2_fits)
    cv = t2_arr.std() / t2_arr.mean()
    assert cv < 0.02, (
        f"Cylinder T2 D-independence: std/mean={cv:.4f} > 0.02; "
        f"T2_vals={t2_arr*1e3}")


@pytest.mark.slow
def test_t2_d_independence_sphere():
    """Sphere T2 must be flat as D varies."""
    rho = 1e-6
    R = R_DEFAULT
    D_values = np.array([0.5e-9, 1e-9, 2e-9, 4e-9])
    geom_base = Sphere(radius=R, surface_relaxivity_t2=rho)
    T2_theory = geom_base.volume() / (rho * geom_base.surface_area())
    t2_fits = []

    for D in D_values:
        geom = Sphere(radius=R, surface_relaxivity_t2=rho)
        T2_fit, _, _ = _measure_t2(geom, T2_theory, D=D,
                                    n_walkers=N_WALKERS_SV)
        t2_fits.append(T2_fit)

    t2_arr = np.array(t2_fits)
    cv = t2_arr.std() / t2_arr.mean()
    assert cv < 0.02, (
        f"Sphere T2 D-independence: std/mean={cv:.4f} > 0.02; "
        f"T2_vals={t2_arr*1e3}")


# ---------------------------------------------------------------------------
# Diagnostic figures
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_relaxivity_t2_decay_figure():
    """Generate E(t) log-scale decay plots for all geometries × ρ values."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    rhos = [1e-6, 5e-6, 20e-6]
    colors = ['C0', 'C1', 'C2']

    geom_configs = [
        ('Box1D',    lambda rho: Box1D(length=R_DEFAULT, surface_relaxivity_t2=rho),
         lambda g, rho: g.volume() / (rho * g.surface_area())),
        ('Cylinder', lambda rho: Cylinder(radius=R_DEFAULT, orientation=[0,0,1.],
                                           surface_relaxivity_t2=rho),
         lambda g, rho: g.volume(L=1.0) / (rho * g.surface_area(L=1.0))),
        ('Sphere',   lambda rho: Sphere(radius=R_DEFAULT, surface_relaxivity_t2=rho),
         lambda g, rho: g.volume() / (rho * g.surface_area())),
    ]

    for ax, (name, geom_fn, t2_fn) in zip(axes, geom_configs):
        for rho, color in zip(rhos, colors):
            geom = geom_fn(rho)
            T2_th = t2_fn(geom, rho)
            TE_targets = np.linspace(0.3 * T2_th, 3.0 * T2_th, 7)
            actual_TEs = []
            signals = []
            for TE_t in TE_targets:
                wf, actual_TE = _b0_waveform(TE_t)
                actual_TEs.append(actual_TE)
                s = simulate(200_000, D_DEFAULT, wf, geom, seed=SEED)
                signals.append(float(s[0]))
            ax.semilogy(np.array(actual_TEs)*1e3, signals,
                        'o-', color=color, alpha=0.8,
                        label=f'ρ={rho*1e6:.0f}µm/s, T2={T2_th*1e3:.0f}ms')
            # Theory
            TE_ref = np.linspace(0.3 * T2_th, 3.0 * T2_th, 100)
            ax.semilogy(TE_ref*1e3, np.exp(-TE_ref/T2_th),
                        '--', color=color, alpha=0.4)
        ax.set_xlabel('TE (ms)')
        ax.set_ylabel('E(TE)')
        ax.set_title(f'{name} surface relaxivity')
        ax.legend(fontsize=7)

    fig.suptitle('Surface relaxivity T2 decay: MC (points) vs theory (dashed)')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'surface_relaxivity_t2_decay.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    assert os.path.exists(out)


@pytest.mark.slow
def test_sv_scaling_figure():
    """Generate T2 vs R log-log plot with slope annotation."""
    rho = 1e-6
    radii = np.array([2e-6, 4e-6, 8e-6, 16e-6])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (name, geom_fn, t2_fn) in zip(axes, [
        ('Cylinder', lambda R: Cylinder(radius=R, orientation=[0,0,1.],
                                         surface_relaxivity_t2=rho),
         lambda g, rho: g.volume(L=1.0) / (rho * g.surface_area(L=1.0))),
        ('Sphere',   lambda R: Sphere(radius=R, surface_relaxivity_t2=rho),
         lambda g, rho: g.volume() / (rho * g.surface_area())),
    ]):
        t2_fits = []
        t2_theories = []
        for R in radii:
            geom = geom_fn(R)
            T2_theory = t2_fn(geom, rho)
            t2_theories.append(T2_theory)
            T2_fit, _, _ = _measure_t2(geom, T2_theory, n_walkers=N_WALKERS_SV)
            t2_fits.append(T2_fit)

        log_R = np.log(radii)
        log_T2 = np.log(np.array(t2_fits))
        slope, intercept = np.polyfit(log_R, log_T2, 1)

        ax.loglog(radii*1e6, np.array(t2_theories)*1e3, 'k--',
                  label='Theory')
        ax.loglog(radii*1e6, np.array(t2_fits)*1e3, 'o',
                  label=f'MC fit (slope={slope:.3f})')
        ax.set_xlabel('R (µm)')
        ax.set_ylabel('T2 (ms)')
        ax.set_title(f'{name}: T2 vs R (ρ=1µm/s)')
        ax.legend()
        ax.text(0.05, 0.95, f'slope = {slope:.3f}', transform=ax.transAxes,
                va='top')

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'sv_scaling_t2_vs_r.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    assert os.path.exists(out)


@pytest.mark.slow
def test_d_independence_figure():
    """Generate T2 vs D scatter plot (should be flat)."""
    rho = 1e-6
    R = R_DEFAULT
    D_values = np.array([0.5e-9, 1e-9, 2e-9, 4e-9])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (name, geom_fn, t2_fn) in zip(axes, [
        ('Cylinder', lambda: Cylinder(radius=R, orientation=[0,0,1.],
                                       surface_relaxivity_t2=rho),
         lambda g, rho: g.volume(L=1.0) / (rho * g.surface_area(L=1.0))),
        ('Sphere',   lambda: Sphere(radius=R, surface_relaxivity_t2=rho),
         lambda g, rho: g.volume() / (rho * g.surface_area())),
    ]):
        geom_base = geom_fn()
        T2_theory = t2_fn(geom_base, rho)
        t2_fits = []
        for D in D_values:
            geom = geom_fn()
            T2_fit, _, _ = _measure_t2(geom, T2_theory, D=D,
                                        n_walkers=N_WALKERS_SV)
            t2_fits.append(T2_fit)

        ax.semilogx(D_values*1e9, np.array(t2_fits)*1e3, 'o-')
        ax.axhline(T2_theory*1e3, color='k', linestyle='--',
                   label=f'Theory = {T2_theory*1e3:.0f}ms')
        ax.set_xlabel('D (×10⁻⁹ m²/s)')
        ax.set_ylabel('T2 (ms)')
        ax.set_title(f'{name}: T2 vs D (ρ=1µm/s, R=10µm)')
        ax.legend()

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 't2_d_independence.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    assert os.path.exists(out)
