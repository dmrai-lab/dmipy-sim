"""Kärger two-compartment exchange: MC (permeable packed spheres) vs the analytical formula.

A mixed population (f inside a reflecting sphere, 1−f outside) is walked on a periodic
``PackedSpheres`` substrate with a permeable membrane.  The PGSE-encoded signal is compared to
the Kärger exchange formula built from the separately-simulated intra/extra attenuations.

Protocol
--------
1. E_intra  : isolated reflecting Sphere, intra walkers.
2. E_extra  : PackedSpheres exterior, extra walkers.
3. E_mc     : mixed r0 on the permeable PackedSpheres (exchange active).
4. E_analyt : _karger_formula(R1, R2, κ, f, t_d),  t_d = Δ − δ/3.
5. assert |E_mc − E_analyt| ≤ atol.
"""

import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
import numpy.testing as npt

from dmipy_sim import simulate, PackedSpheres, set_b
from dmipy_sim.geometries import Sphere
from dmipy_sim.waveforms import pgse, pgste

import pytest
# Cross-engine parity: MC (this repo) vs the analytical Kärger formula in dmipy-fit. dmipy-fit is
# NOT a dependency of dmipy-sim, so skip the whole module at collection when it is absent —
# otherwise a bare import errors on a standalone dmipy-sim checkout / CI.
_exchange_models = pytest.importorskip(
    "dmipy_fit.signal_models.exchange_models",
    reason="cross-engine Kärger parity needs the analytical engine dmipy-fit",
)
_karger_formula = _exchange_models._karger_formula
_karger_propagator_ste = _exchange_models._karger_propagator_ste


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED   = 42
N      = 200_000
R      = 5e-6     # m  — sphere radius
F      = 0.30     # intra volume fraction
D      = 2e-9     # m²/s — diffusivity (same in both compartments)

# L such that (4π/3 R³) / L³ = F
L_BOX  = R * ((4 * np.pi / 3) / F) ** (1 / 3)

# PGSE parameters
DELTA_PGSE  = 20e-3    # s  — diffusion time (Δ)
DELTA_SHORT = 5e-3     # s  — encoding duration (δ)
B_PGSE = np.array([500e6, 1000e6, 2000e6, 3000e6])  # s/m²

# PGSTE parameters — same encoding duration δ as PGSE; the diffusion time is now
# spanned by the mixing time TM (magnetisation stored longitudinally) plus the
# two lobes.  TM is kept short enough that κ·t_d_pgste stays in the Kärger regime.
TM_STE  = 20e-3    # s  — mixing time (longitudinal storage)
B_PGSTE = B_PGSE   # s/m² — reuse the PGSE shells

# Exchange surface permeabilities
KAPPA_SURF_SLOW = 3e-6    # m/s  → κ = 1.8 s⁻¹,  κ·t_d ≈ 0.033 (slow exchange)
KAPPA_SURF_FAST = 1e-5    # m/s  → κ = 6.0 s⁻¹,  κ·t_d ≈ 0.110 (fast exchange)
KAPPA_SURF_ZERO = 1e-8    # m/s  → essentially zero exchange (for no-exchange test)

# Tolerance
ATOL = max(0.03, 3.0 / np.sqrt(N))

# Very-slow-exchange (tight) regime
N_TIGHT           = 500_000
KAPPA_SURF_VTIGHT = 5e-7   # m/s → κ = 0.30 s⁻¹, κ·t_d ≈ 0.0055
ATOL_TIGHT        = max(0.012, 4.0 / np.sqrt(N_TIGHT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _karger_rate(kappa_surf):
    """Surface permeability (m/s) → Kärger exchange rate (s⁻¹).  Sphere: κ = 3·κ_surf/R."""
    return 3.0 * kappa_surf / R


def _r0_inside_sphere(n, radius, seed):
    """Rejection-sample n positions uniformly inside sphere of radius R at origin."""
    rng = np.random.default_rng(seed)
    accepted = []
    while sum(len(a) for a in accepted) < n:
        pts = rng.uniform(-radius, radius, (max(n * 4, 1024), 3))
        accepted.append(pts[np.linalg.norm(pts, axis=1) < radius])
    return np.concatenate(accepted)[:n].astype(np.float32)


def _geom_sphere():
    """Isolated reflecting sphere for intra-compartment signal."""
    return Sphere(radius=R)


def _geom_packed(kappa_surf=None):
    """Periodic packed sphere.  kappa_surf=None → fully impermeable exterior."""
    return PackedSpheres(
        radii=np.array([R]),
        centers=np.array([[0., 0., 0.]]),
        L=L_BOX,
        permeability=kappa_surf,
    )


def _pgse_waveform(b_values):
    """PGSE waveform at given b-values (s/m²), all along x-axis."""
    bvecs = np.tile([1., 0., 0.], (len(b_values), 1))
    wf = pgse(delta=DELTA_SHORT, DELTA=DELTA_PGSE, G_magnitude=1.0,
              bvecs=bvecs, n_t=500)
    return set_b(wf, b_values)


def _pgste_waveform(b_values):
    """PGSTE waveform at given b-values (s/m²), all along x-axis.

    Ideal instantaneous pulses; the returned waveform carries ``stimulated_echo``
    so :func:`dmipy_sim.simulate` applies the idealized 0.5 stimulated-echo factor.
    """
    bvecs = np.tile([1., 0., 0.], (len(b_values), 1))
    wf = pgste(delta=DELTA_SHORT, TM=TM_STE, G_magnitude=1.0,
               bvecs=bvecs, n_t=500)
    return set_b(wf, b_values)


# ---------------------------------------------------------------------------
# Test 1: near-impermeable → MC signal ≈ f·E_intra + (1-f)·E_extra
# ---------------------------------------------------------------------------

def test_pgse_no_exchange_is_mixture():
    """Near-zero κ: MC(mixed r0, permeate≈0) ≈ f·E_intra + (1−f)·E_extra."""
    import jax

    b_vals = np.array([0.0, 500e6, 1000e6, 1500e6, 2000e6])
    wf = _pgse_waveform(b_vals)

    geom_sphere = _geom_sphere()
    geom_packed = _geom_packed()
    geom_zero   = _geom_packed(kappa_surf=KAPPA_SURF_ZERO)

    n_intra = int(round(F * N))
    n_extra = N - n_intra

    r_intra = _r0_inside_sphere(n_intra, R, seed=SEED)
    E_intra = simulate(n_intra, diffusivity=D, waveform=wf,
                       geometry=geom_sphere, seed=SEED, r0=r_intra)

    key = jax.random.PRNGKey(SEED + 10)
    r_extra = np.array(geom_packed.init_positions(n_extra, key), dtype=np.float32)
    E_extra = simulate(n_extra, diffusivity=D, waveform=wf,
                       geometry=geom_packed, seed=SEED + 1, r0=r_extra)

    E_mixture = F * E_intra + (1 - F) * E_extra

    r0_mixed = np.concatenate([r_intra, r_extra], axis=0)
    E_mc = simulate(N, diffusivity=D, waveform=wf, geometry=geom_zero,
                    seed=SEED + 2, r0=r0_mixed)

    npt.assert_allclose(
        E_mc, E_mixture, atol=ATOL,
        err_msg=(f"No-exchange mixture mismatch:\n"
                 f"  E_mc      = {E_mc}\n"
                 f"  E_mixture = {E_mixture}\n"
                 f"  diff      = {E_mc - E_mixture}"),
    )


# ---------------------------------------------------------------------------
# Tests 2–3: PGSE Kärger (slow / fast exchange)
# ---------------------------------------------------------------------------

def test_pgse_slow_exchange_kappa_formula():
    """PGSE slow exchange (κ_surf = 3e-6 m/s, κ·t_d ≈ 0.033)."""
    _run_pgse_karger_test(KAPPA_SURF_SLOW, B_PGSE, N, ATOL)


def test_pgse_fast_exchange_kappa_formula():
    """PGSE fast exchange (κ_surf = 1e-5 m/s, κ·t_d ≈ 0.11)."""
    _run_pgse_karger_test(KAPPA_SURF_FAST, B_PGSE, N, ATOL)


def test_pgse_very_slow_exchange_tight():
    """PGSE very slow exchange (κ_surf=5e-7 m/s, κ·t_d≈0.0055), N=500k, atol=0.012."""
    _run_pgse_karger_test(KAPPA_SURF_VTIGHT, B_PGSE, N_TIGHT, ATOL_TIGHT)


def _run_pgse_karger_test(kappa_surf, b_values, n_walkers, atol):
    """Shared logic for PGSE Kärger tests."""
    import jax

    wf = _pgse_waveform(b_values)
    geom_sphere = _geom_sphere()
    geom_packed = _geom_packed()
    geom_perm   = _geom_packed(kappa_surf=kappa_surf)
    kappa = _karger_rate(kappa_surf)
    t_d   = DELTA_PGSE - DELTA_SHORT / 3.0

    n_intra = int(round(F * n_walkers))
    n_extra = n_walkers - n_intra

    # Intra-only, Sphere → E_intra
    r_intra = _r0_inside_sphere(n_intra, R, seed=SEED)
    E_intra = simulate(n_intra, diffusivity=D, waveform=wf,
                       geometry=geom_sphere, seed=SEED, r0=r_intra)

    # Extra-only, PackedSpheres → E_extra
    key = jax.random.PRNGKey(SEED + 10)
    r_extra = np.array(geom_packed.init_positions(n_extra, key), dtype=np.float32)
    E_extra = simulate(n_extra, diffusivity=D, waveform=wf,
                       geometry=geom_packed, seed=SEED + 1, r0=r_extra)

    # Mixed, permeable → E_mc_exchange
    r0_mixed = np.concatenate([r_intra, r_extra], axis=0)
    E_mc = simulate(n_walkers, diffusivity=D, waveform=wf, geometry=geom_perm,
                    seed=SEED + 2, r0=r0_mixed)

    # Analytical Kärger formula
    R1 = np.clip(-np.log(np.clip(E_intra, 1e-10, None)), 0, 10)
    R2 = np.clip(-np.log(np.clip(E_extra, 1e-10, None)), 0, 10)
    E_analytical = _karger_formula(R1, R2, kappa=kappa, f=F, t_d=t_d)

    npt.assert_allclose(
        E_mc, E_analytical, atol=atol,
        err_msg=(f"PGSE Kärger mismatch (κ_surf={kappa_surf:.1e} m/s, "
                 f"κ={kappa:.3f} s⁻¹, κ·t_d={kappa*t_d:.5f}):\n"
                 f"  b         = {b_values * 1e-6} s/mm²\n"
                 f"  E_mc      = {E_mc}\n"
                 f"  E_analyt. = {E_analytical}\n"
                 f"  diff      = {E_mc - E_analytical}"),
    )


# ---------------------------------------------------------------------------
# Tests 4–6: PGSTE Kärger (slow / fast / very-slow exchange)
# ---------------------------------------------------------------------------
# The stimulated echo stores the magnetisation longitudinally during the mixing
# time, so the MC signal carries the idealized 0.5 stimulated-echo factor (E_mc at
# b=0 is 0.5, not 1).  The Kärger formula works on per-compartment attenuation
# relative to the unweighted signal, so the single-compartment MC signals are
# divided by 0.5 before forming R1/R2 and the formula result is multiplied by 0.5
# to restore the stimulated-echo factor.  The effective diffusion time is
# t_d_pgste = TM + 2·δ/3 (mixing time plus the finite-pulse correction of the two
# lobes), matching the analytical PGSTE mixing-time path.


def test_pgste_slow_exchange_kappa_formula():
    """PGSTE slow exchange (κ_surf = 3e-6 m/s, κ·t_d ≈ 0.042)."""
    _run_pgste_karger_test(KAPPA_SURF_SLOW, B_PGSTE, N, ATOL)


def test_pgste_fast_exchange_kappa_formula():
    """PGSTE fast exchange (κ_surf = 1e-5 m/s, κ·t_d ≈ 0.14)."""
    _run_pgste_karger_test(KAPPA_SURF_FAST, B_PGSTE, N, ATOL)


def test_pgste_very_slow_exchange_tight():
    """PGSTE very slow exchange (κ_surf=5e-7 m/s), N=500k, atol=0.012."""
    _run_pgste_karger_test(KAPPA_SURF_VTIGHT, B_PGSTE, N_TIGHT, ATOL_TIGHT)


def _run_pgste_karger_test(kappa_surf, b_values, n_walkers, atol):
    """Shared logic for PGSTE Kärger tests (mirrors ``_run_pgse_karger_test``)."""
    import jax

    wf = _pgste_waveform(b_values)
    geom_sphere = _geom_sphere()
    geom_packed = _geom_packed()
    geom_perm   = _geom_packed(kappa_surf=kappa_surf)
    kappa = _karger_rate(kappa_surf)
    t_d   = TM_STE + 2.0 * DELTA_SHORT / 3.0   # t_d_pgste = TM + 2δ/3

    n_intra = int(round(F * n_walkers))
    n_extra = n_walkers - n_intra

    # Intra-only, Sphere → E_intra (carries the 0.5 stimulated-echo factor)
    r_intra = _r0_inside_sphere(n_intra, R, seed=SEED)
    E_intra = simulate(n_intra, diffusivity=D, waveform=wf,
                       geometry=geom_sphere, seed=SEED, r0=r_intra)

    # Extra-only, PackedSpheres → E_extra (carries the 0.5 stimulated-echo factor)
    key = jax.random.PRNGKey(SEED + 10)
    r_extra = np.array(geom_packed.init_positions(n_extra, key), dtype=np.float32)
    E_extra = simulate(n_extra, diffusivity=D, waveform=wf,
                       geometry=geom_packed, seed=SEED + 1, r0=r_extra)

    # Mixed, permeable → E_mc_exchange (carries the 0.5 stimulated-echo factor)
    r0_mixed = np.concatenate([r_intra, r_extra], axis=0)
    E_mc = simulate(n_walkers, diffusivity=D, waveform=wf, geometry=geom_perm,
                    seed=SEED + 2, r0=r0_mixed)

    # Remove the 0.5 stimulated-echo factor to get per-compartment attenuation,
    # then restore it on the analytical result.
    E_intra_norm = np.clip(E_intra / 0.5, 1e-10, 1.0)
    E_extra_norm = np.clip(E_extra / 0.5, 1e-10, 1.0)
    R1 = np.clip(-np.log(E_intra_norm), 0, 10)
    R2 = np.clip(-np.log(E_extra_norm), 0, 10)
    E_analytical = 0.5 * _karger_formula(R1, R2, kappa=kappa, f=F, t_d=t_d)

    npt.assert_allclose(
        E_mc, E_analytical, atol=atol,
        err_msg=(f"PGSTE Kärger mismatch (κ_surf={kappa_surf:.1e} m/s, "
                 f"κ={kappa:.3f} s⁻¹, κ·t_d={kappa*t_d:.5f}):\n"
                 f"  b         = {b_values * 1e-6} s/mm²\n"
                 f"  E_mc      = {E_mc}\n"
                 f"  E_analyt. = {E_analytical}\n"
                 f"  diff      = {E_mc - E_analytical}"),
    )

    # Also exercise the matrix-exponential STE propagator (the path
    # X0GeneralizedKarger takes once per-compartment relaxation is present) against
    # the same MC signal, so the analytic engine's per-lobe timing is covered too.
    # The two encoding lobes are each δ long (transverse total 2δ) with the storage
    # window TM in between; the second-lobe transverse time is dt6 = δ (read from
    # the geometry, not reconstructed from TE). Feed the MC-measured per-compartment
    # attenuations as effective diffusivities (D_eff = R/b over the two lobes) and
    # let the propagator handle exchange over the full 2δ + TM history.
    E_prop = np.empty_like(E_mc)
    for i, bv in enumerate(b_values):
        bv = float(bv)
        if bv < 1e3:
            E_prop[i] = 0.5                      # b0: no diffusion, keeps the STE 0.5
            continue
        D1_eff = float(R1[i]) / bv
        D2_eff = float(R2[i]) / bv
        M_TE = _karger_propagator_ste(
            D1_eff, D2_eff, 1e10, 1e10, 1e10, 1e10, kappa, F,
            bv / 2.0, bv / 2.0, DELTA_SHORT, TM_STE, 0.0, DELTA_SHORT)
        E_prop[i] = float(np.sum(M_TE))

    npt.assert_allclose(
        E_mc, E_prop, atol=atol,
        err_msg=(f"PGSTE Kärger STE-propagator mismatch (κ_surf={kappa_surf:.1e} "
                 f"m/s, κ={kappa:.3f} s⁻¹, dt6=δ={DELTA_SHORT:.4f} s):\n"
                 f"  b       = {b_values * 1e-6} s/mm²\n"
                 f"  E_mc    = {E_mc}\n"
                 f"  E_prop. = {E_prop}\n"
                 f"  diff    = {E_mc - E_prop}"),
    )


# ---------------------------------------------------------------------------
# Test 7: pure-waveform consistency of the PGSTE mixing-time mask (no MC)
# ---------------------------------------------------------------------------

def test_pgste_waveform_mixing_time_mask():
    """The ``pgste`` waveform gates transverse coherence to the two lobes only.

    The binary transverse-coherence mask must be True during the two encoding
    lobes (≈ 2δ) and False during the mixing time (≈ TM), and the waveform must
    advertise the stimulated echo.  This pins the timing the MC signal relies on
    without running any walkers.
    """
    b_values = np.array([1000e6])
    wf = _pgste_waveform(b_values)

    assert getattr(wf, 'stimulated_echo', False) is True
    npt.assert_allclose(wf.TM, TM_STE)

    chi = np.asarray(wf.chi_perp)
    n_stored = int((~chi).sum())          # longitudinal storage steps
    n_transverse = int(chi.sum())         # transverse (encoding) steps

    # Stored fraction ≈ TM / (2δ + TM); transverse fraction ≈ 2δ / (2δ + TM).
    T_total = 2.0 * DELTA_SHORT + TM_STE
    npt.assert_allclose(n_stored / len(chi), TM_STE / T_total, atol=0.02)
    npt.assert_allclose(n_transverse / len(chi),
                        2.0 * DELTA_SHORT / T_total, atol=0.02)
