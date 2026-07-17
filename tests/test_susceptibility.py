"""Susceptibility (off-resonance) physics — analytical field truth + gating.

The off-resonance field is a first-class substrate effect (geometry.off_resonance),
co-equal with surface relaxivity and permeability.  A uniformly-magnetised sphere
(radius a, susceptibility difference Δχ, field B0) produces, outside it,

    ΔBz(r) = (Δχ·B0/3)·a³·(3cos²θ − 1)/r³ ,   θ = angle(r − c, B0=z)   (Schenck 1996)

Tests:
  * FIELD: SusceptibilitySources.delta_bz_fn reproduces the closed-form dipole
    (superposition + interior clamp).
  * REFOCUS: refocus_sign flips at the last RF; near-static spins refocus the static
    field (spin/stimulated echo) -> E ≈ 1.
  * GATING: for diffusing spins PGSTE (transverse only during the two 2δ encodings)
    retains MORE susceptibility signal than PGSE (transverse the whole TE).
  * COMPOSE: off_resonance co-acts with surface relaxivity + permeability (one walk).
"""
import numpy as np
import pytest
import jax

from dmipy_sim import (simulate, set_b, FreeDiffusion, PackedSpheres,
                       SusceptibilitySources)
from dmipy_sim.susceptibility import refocus_sign
from dmipy_sim.waveforms import pgse, pgste
from dmipy_sim.constants import GAMMA

DELTA_ENC = 5e-3
DELTA = 40e-3
TM = DELTA - DELTA_ENC
BOX = 12e-6


def _dbz_analytic(r, centers, radii, delta_chi, B0):
    """Closed-form ΔBz(r) summed over magnetised spheres (∥B0=z), interior clamped."""
    d = r[None, :] - centers                      # (P, 3)
    dist2 = np.maximum(np.sum(d * d, axis=1), radii ** 2)
    cos2 = d[:, 2] ** 2 / dist2
    coeff = (delta_chi * B0 / 3.0) * radii ** 3
    return np.sum(coeff * (3.0 * cos2 - 1.0) / dist2 ** 1.5)


def test_field_matches_closed_form_dipole():
    rng = np.random.default_rng(0)
    P = 5
    centers = rng.uniform(-BOX, BOX, (P, 3))
    radii = rng.uniform(1e-6, 2e-6, P)
    delta_chi, B0 = 1.0e-5, 3.0
    src = SusceptibilitySources(centers=centers, radii=radii, delta_chi=delta_chi, B0=B0)
    dbz = src.delta_bz_fn()
    for r in rng.uniform(-BOX, BOX, (20, 3)):
        got = float(dbz(r))
        exp = _dbz_analytic(r, centers, radii, delta_chi, B0)
        # single-precision JAX field vs float64 reference
        assert abs(got - exp) <= 1e-4 * abs(exp) + 1e-12, (got, exp)


def test_field_axial_equatorial_ratio_is_minus_two():
    a = 1.5e-6
    src = SusceptibilitySources(centers=[[0, 0, 0]], radii=[a], delta_chi=1e-5, B0=3.0)
    dbz = src.delta_bz_fn()
    axial = float(dbz(np.array([0.0, 0.0, 3 * a])))       # θ=0  -> 3cos²-1 = +2
    equat = float(dbz(np.array([3 * a, 0.0, 0.0])))       # θ=90 -> 3cos²-1 = -1
    assert axial / equat == pytest.approx(-2.0, rel=1e-4)


def test_field_superposition():
    a = 1.5e-6
    c1, c2 = np.array([2e-6, 0, 0]), np.array([-3e-6, 1e-6, 0])
    kw = dict(radii=[a], delta_chi=1e-5, B0=3.0)
    f1 = SusceptibilitySources(centers=[c1], **kw).delta_bz_fn()
    f2 = SusceptibilitySources(centers=[c2], **kw).delta_bz_fn()
    f12 = SusceptibilitySources(centers=[c1, c2], radii=[a, a],
                                delta_chi=1e-5, B0=3.0).delta_bz_fn()
    r = np.array([0.0, 0.0, 0.0])
    assert float(f12(r)) == pytest.approx(float(f1(r)) + float(f2(r)), rel=1e-5)


def test_refocus_sign_flips_at_last_rf():
    wf = pgste(delta=DELTA_ENC, TM=TM, G_magnitude=1.0, bvecs=[[1, 0, 0]],
               n_t=400, slew_rate=np.inf)
    s = refocus_sign(wf)
    assert s.shape == (wf.G.shape[1],)
    assert set(np.unique(s)).issubset({-1.0, 1.0})
    idx = int(round(wf.rf_events[-1]['t_s'] / wf.dt))
    assert np.all(s[:idx] == 1.0) and np.all(s[idx:] == -1.0)


def _atten(seq, sources, D, seed=7, N=6000):
    """Engine attenuation E = S(field)/S0(no field) at b=0 (isolates the field)."""
    wf = (pgse(delta=DELTA_ENC, DELTA=DELTA, G_magnitude=1.0, bvecs=[[1, 0, 0]],
               n_t=1000, slew_rate=np.inf) if seq == 'pgse'
          else pgste(delta=DELTA_ENC, TM=TM, G_magnitude=1.0, bvecs=[[1, 0, 0]],
                     n_t=1000, slew_rate=np.inf))
    wf = set_b(wf, 0.0)
    rng = np.random.default_rng(1)
    r0 = rng.uniform(-BOX, BOX, (N, 3)).astype(np.float32)
    S = float(np.asarray(simulate(N, diffusivity=D, waveform=wf,
                                  geometry=FreeDiffusion(off_resonance=sources),
                                  seed=seed, r0=r0, require_gpu=False))[0])
    S0 = float(np.asarray(simulate(N, diffusivity=D, waveform=wf,
                                   geometry=FreeDiffusion(),
                                   seed=seed, r0=r0, require_gpu=False))[0])
    return S / S0


@pytest.fixture(scope="module")
def sources():
    rng = np.random.default_rng(1)
    return SusceptibilitySources(centers=rng.uniform(-BOX, BOX, (60, 3)),
                                 radii=np.full(60, 1.5e-6), delta_chi=1e-5, B0=3.0)


def test_static_field_refocuses(sources):
    # near-static spins (D→0): the echo refocuses the static field -> E ≈ 1
    for seq in ('pgse', 'pgste'):
        assert _atten(seq, sources, D=1e-18) == pytest.approx(1.0, abs=0.01)


def test_pgste_gates_more_than_pgse(sources):
    # diffusing: PGSTE stores longitudinally during T_M, so it accrues less
    # off-resonance phase and retains MORE signal than PGSE.
    D = 1.0e-9
    e_pgse = _atten('pgse', sources, D)
    e_pgste = _atten('pgste', sources, D)
    assert e_pgste > e_pgse + 0.05


def test_off_resonance_composes_with_membrane_effects():
    # susceptibility + permeability + surface relaxivity on one substrate must run
    # (the first-class effect is not mutually exclusive with membrane physics).
    src = SusceptibilitySources(centers=[[8e-6, 0, 0]], radii=[1.5e-6],
                                delta_chi=1e-5, B0=3.0)
    geom = PackedSpheres(radii=[5e-6], centers=[[0., 0., 0.]], L=24e-6,
                         surface_relaxivity_t2=1e-6, permeability=2e-5,
                         off_resonance=src)
    wf = set_b(pgse(delta=DELTA_ENC, DELTA=DELTA, G_magnitude=1.0, bvecs=[[1, 0, 0]],
                    n_t=600, slew_rate=np.inf), 1e9)
    sig = float(np.asarray(simulate(2000, diffusivity=1e-9, waveform=wf, geometry=geom,
                                    seed=0, T2=60e-3, require_gpu=False))[0])
    assert np.isfinite(sig) and 0.0 <= sig <= 1.0
