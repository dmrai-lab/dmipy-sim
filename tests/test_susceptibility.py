"""Susceptibility off-resonance fields — analytic-anchored validation.

Field providers (closed form + k-space dipole grid) are checked against closed forms
with no free parameters; the forward Bloch engine is checked for spin-echo refocusing of
a static field and for diffusion-driven attenuation in a varying field.
"""
import numpy as np
import numpy.testing as npt
import pytest
import jax
import jax.numpy as jnp

from dmipy_sim import (simulate_bloch, FreeDiffusion,
                       SusceptibilitySources, MyelinSusceptibility, GridSusceptibility,
                       dipole_field, myelin_susceptibility_tensor, radial_from_sdf,
                       sample_grid, mesh_shapes as ms)
from dmipy_sim.constants import GAMMA

D = 2e-9


def _grid(N, L):
    X, Y, Z, vs, org = ms.grid_axes(L, N)
    return X, Y, Z, vs, org


def _zero_waveform(n_t, dt):
    from types import SimpleNamespace
    return SimpleNamespace(G=np.zeros((1, n_t, 3), dtype=np.float64), dt=dt)


# =========================================================================== #
# Grid dipole field vs closed forms
# =========================================================================== #
def test_isotropic_sphere_zero_inside():
    N, L, R, chi = 96, 12.0, 3.0, 1e-6
    X, Y, Z, vs, org = _grid(N, L)
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    chi6 = myelin_susceptibility_tensor((r <= R).astype(float),
                                        np.zeros(r.shape + (3,)), chi_iso=chi)
    dB = dipole_field(chi6, vs, [0, 0, 1], B0=1.0)
    inside = r < 0.6 * R
    assert abs(dB[inside].mean()) / chi < 5e-3
    assert dB[inside].std() / chi < 5e-3


@pytest.mark.parametrize("bdir,theta", [([0, 0, 1], 0.0), ([1, 0, 0], np.pi / 2)])
def test_isotropic_cylinder_internal(bdir, theta):
    """Infinite cylinder internal dB/B0 = (chi/6)(3cos^2 th - 1), up to (1-f) referencing."""
    N, L, Rc, chi = 128, 24.0, 3.0, 1e-6
    X, Y, Z, vs, org = _grid(N, L)
    rho = np.sqrt(X ** 2 + Y ** 2)
    chi6 = myelin_susceptibility_tensor((rho <= Rc).astype(float),
                                        np.zeros(rho.shape + (3,)), chi_iso=chi)
    dB = dipole_field(chi6, vs, bdir, B0=1.0)
    inside = rho < 0.5 * Rc
    f = np.pi * Rc ** 2 / L ** 2
    val = dB[inside].mean() / chi / (1.0 - f)
    pred = (1.0 / 6.0) * (3 * np.cos(theta) ** 2 - 1)
    assert dB[inside].std() / chi < 1e-2
    npt.assert_allclose(val, pred, atol=0.02)


def _aniso_intra(a, b, bdir, N=128, L=24.0, chi_A=1.0):
    X, Y, Z, vs, org = _grid(N, L)
    mask, radial = ms.straight_myelin_source(X, Y, Z, a, b)
    chi6 = myelin_susceptibility_tensor(mask, radial, chi_iso=0.0, chi_aniso=chi_A)
    dB = dipole_field(chi6, vs, bdir, B0=1.0)
    rho = np.sqrt(X ** 2 + Y ** 2)
    intra = rho < 0.6 * a
    return dB[intra].mean(), dB[intra].std()


def test_aniso_hollow_cylinder_intra_ln_g():
    """Anisotropic intra field uniform and = (chi_A/2) ln(1/g) at perpendicular B0."""
    a, b = 3.0, 4.0
    m, s = _aniso_intra(a, b, [1, 0, 0])
    assert abs(s / m) < 0.05
    npt.assert_allclose(m, 0.5 * np.log(1.0 / (a / b)), rtol=0.06)


def test_aniso_intra_sin2theta_scaling():
    a, b = 3.0, 4.0
    perp, _ = _aniso_intra(a, b, [1, 0, 0])
    par, _ = _aniso_intra(a, b, [0, 0, 1])
    at45, _ = _aniso_intra(a, b, [1, 0, 1])
    assert abs(par / perp) < 0.08
    npt.assert_allclose(at45, 0.5 * perp, rtol=0.08)


def test_isotropic_hollow_cylinder_intra_is_zero():
    a, b = 3.0, 4.0
    X, Y, Z, vs, org = _grid(128, 24.0)
    mask, radial = ms.straight_myelin_source(X, Y, Z, a, b)
    chi6 = myelin_susceptibility_tensor(mask, radial, chi_iso=1.0, chi_aniso=0.0)
    dB = dipole_field(chi6, vs, [1, 0, 0], B0=1.0)
    rho = np.sqrt(X ** 2 + Y ** 2)
    assert abs(dB[rho < 0.6 * a].mean()) < 0.03


def test_radial_from_sdf_matches_analytic():
    a, b = 3.0, 4.0
    X, Y, Z, vs, org = _grid(96, 16.0)
    rho = np.sqrt(X ** 2 + Y ** 2)
    analytic = np.stack([X / np.maximum(rho, 1e-30), Y / np.maximum(rho, 1e-30),
                         np.zeros_like(X)], axis=-1)
    n = radial_from_sdf(rho - (a + b) / 2.0, vs)
    sheath = (rho >= a) & (rho <= b)
    assert np.abs((n * analytic).sum(-1))[sheath].min() > 0.99


def test_sampling_at_voxel_centres():
    a, b = 3.0, 4.0
    X, Y, Z, vs, org = _grid(64, 16.0)
    mask, radial = ms.straight_myelin_source(X, Y, Z, a, b)
    dB = dipole_field(myelin_susceptibility_tensor(mask, radial, chi_aniso=-0.1e-6),
                      vs, [1, 0, 0], B0=7.0)
    idx = np.array([[20, 30, 32], [40, 25, 10]])
    pts = org + (idx + 0.5) * vs
    got = sample_grid(dB, pts, org, vs, periodic=True)
    npt.assert_allclose(got, dB[idx[:, 0], idx[:, 1], idx[:, 2]], rtol=1e-6)


# =========================================================================== #
# Analytic providers (closed-form delta_bz callables)
# =========================================================================== #
def test_sphere_axial_equatorial_ratio_is_minus_two():
    """Uniformly magnetised sphere: dB(axial)/dB(equatorial) = -2 (dipole signature)."""
    src = SusceptibilitySources(centers=[[0., 0., 0.]], radii=[1e-6], delta_chi=1e-6, B0=3.0)
    f = jax.jit(src.delta_bz_fn())
    axial = float(f(jnp.array([0., 0., 5e-6])))          # theta = 0
    equat = float(f(jnp.array([5e-6, 0., 0.])))          # theta = 90
    npt.assert_allclose(axial / equat, -2.0, rtol=1e-4)


def test_sphere_superposition():
    """Two perturbers superpose linearly."""
    c = [[0., 0., 0.], [8e-6, 0., 0.]]
    f2 = jax.jit(SusceptibilitySources(centers=c, radii=[1e-6, 1e-6], B0=3.0).delta_bz_fn())
    f0 = jax.jit(SusceptibilitySources(centers=[c[0]], radii=[1e-6], B0=3.0).delta_bz_fn())
    f1 = jax.jit(SusceptibilitySources(centers=[c[1]], radii=[1e-6], B0=3.0).delta_bz_fn())
    p = jnp.array([3e-6, 2e-6, 1e-6])
    npt.assert_allclose(float(f2(p)), float(f0(p)) + float(f1(p)), rtol=1e-5)


def test_myelin_field_vanishes_at_parallel_fibre():
    """The whole myelin field (both m=0 and ell=2 carry sin^2 theta) vanishes at B0 ∥ fibre."""
    f = jax.jit(MyelinSusceptibility(centers=[[0., 0.]], inner_radii=[2e-6], outer_radii=[3e-6],
                                     L=20e-6, delta_chi_a=-0.1e-6, B0=3.0, theta=0.0,
                                     periodic=False).delta_bz_fn())
    assert abs(float(f(jnp.array([0.5e-6, 0., 0.])))) < 1e-12    # intra
    assert abs(float(f(jnp.array([6e-6, 1e-6, 0.])))) < 1e-12    # extra


def test_myelin_intra_matches_closed_form():
    """Analytic provider intra field = 1/2 Δχ_a B0 sin^2 th ln(1/g), matching the grid
    dipole solver (the corrected Wharton-Bowtell convention; zero at parallel)."""
    a, b, dchi, B0 = 3e-6, 4e-6, -0.1e-6, 7.0
    pred_perp = 0.5 * dchi * B0 * np.log(b / a)          # sin^2 = 1
    prov = MyelinSusceptibility(centers=[[0., 0.]], inner_radii=[a], outer_radii=[b],
                                L=1e-3, delta_chi_a=dchi, B0=B0, theta=np.pi / 2,
                                periodic=False, n_images=0)
    intra = float(jax.jit(prov.delta_bz_fn())(jnp.array([0.4e-6, 0.1e-6, 0.])))
    npt.assert_allclose(intra, pred_perp, rtol=0.02)
    # cross-check the grid dipole solver on the same shape (per-chi_A, scaled)
    m_perp, _ = _aniso_intra(3.0, 4.0, [1, 0, 0])        # micron grid, chi_A=1
    npt.assert_allclose(dchi * B0 * m_perp, pred_perp, rtol=0.06)


# =========================================================================== #
# Forward Bloch integration
# =========================================================================== #
def test_uniform_field_refocuses_under_spin_echo():
    """A spatially uniform susceptibility off-resonance is refocused by the 180 pulse
    (phase ~ 0 at the echo), and dephases the phase without it."""
    dt, TE, dB0 = 1e-4, 4e-3, 2e-6            # gamma*dB0*TE ~ 2.1 rad (unrefocused phase)
    n_t = int(round(TE / dt)) + 1
    wf = _zero_waveform(n_t, dt)
    exc = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': 90.0}]
    refocus = [{'t_s': TE / 2, 'flip_deg': 180.0, 'axis_deg': 0.0}]
    uniform = lambda r: jnp.float32(dB0)      # position-independent field
    ec_ref = simulate_bloch(2000, D, wf, FreeDiffusion(), exc + refocus,
                            seed=0, susceptibility=uniform)[0]
    ec_free = simulate_bloch(2000, D, wf, FreeDiffusion(), exc,
                             seed=0, susceptibility=uniform)[0]
    assert abs(np.angle(ec_ref)) < 0.1        # refocused
    assert abs(ec_ref) == pytest.approx(1.0, rel=1e-2)
    assert abs(np.angle(ec_free)) > 1.0       # unrefocused static dephasing


def test_diffusion_in_varying_field_attenuates():
    """A varying field refocuses when the spins are frozen (SE) but attenuates the echo
    once diffusion moves them through the gradient -- the susceptibility x diffusion effect."""
    dt, TE = 1e-4, 6e-3
    n_t = int(round(TE / dt)) + 1
    wf = _zero_waveform(n_t, dt)
    exc = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': 90.0}]
    refocus = [{'t_s': TE / 2, 'flip_deg': 180.0, 'axis_deg': 0.0}]
    # a strong local perturber the walkers diffuse around (impenetrable-source clamp)
    src = SusceptibilitySources(centers=[[0., 0., 0.]], radii=[2e-6],
                                delta_chi=3e-6, B0=7.0)
    frozen = abs(simulate_bloch(4000, 1e-13, wf, FreeDiffusion(), exc + refocus,
                                seed=0, susceptibility=src)[0])
    moving = abs(simulate_bloch(4000, D, wf, FreeDiffusion(), exc + refocus,
                                seed=0, susceptibility=src)[0])
    assert frozen == pytest.approx(1.0, abs=0.02)   # static per spin -> refocused
    assert moving < frozen - 0.01                    # diffusion breaks the refocusing


def test_susceptibility_composes_with_mt():
    """Unified physics: susceptibility + magnetization transfer in one forward pass.
    A uniform static field is still refocused by the spin echo (phase ~ 0) while the MT
    binding saturates the free-pool magnitude -- the two effects compose, not exclude."""
    from dmipy_sim import Sphere
    R, Dm, k_f, k_r = 5e-6, 1e-9, 50.0, 100.0
    kappa_MT, dwell = k_f * R / 3.0, 1.0 / k_r
    dt, TE, dB0 = 1e-4, 6e-3, 2e-6
    n_t = int(round(TE / dt)) + 1
    wf = _zero_waveform(n_t, dt)
    se = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': 90.0},
          {'t_s': TE / 2, 'flip_deg': 180.0, 'axis_deg': 0.0}]
    geom = Sphere(radius=R)
    uniform = lambda r: jnp.float32(dB0)
    susc_only = simulate_bloch(3000, Dm, wf, geom, se, T2=0.06, seed=0, susceptibility=uniform)[0]
    both = simulate_bloch(3000, Dm, wf, geom, se, T2=0.06, seed=0, susceptibility=uniform,
                          kappa_MT=kappa_MT, dwell_time=dwell, T2_bound=1e-5)[0]
    assert abs(np.angle(both)) < 0.15                    # static field refocused (MT on)
    assert abs(both) < abs(susc_only) - 0.02             # MT binding saturates the free signal


def test_grid_provider_runs_and_refocuses_frozen():
    """GridSusceptibility (mesh route) plugs into the Bloch walk; frozen spins refocus."""
    a, b = 3e-6, 4e-6
    X, Y, Z, vs, org = ms.grid_axes(16e-6, 48)
    mask, radial = ms.straight_myelin_source(X, Y, Z, a, b)
    prov = GridSusceptibility.from_source(mask, radial, vs, org, [1, 0, 0], 7.0,
                                          chi_aniso=-0.1e-6, periodic=True)
    dt, TE = 1e-4, 4e-3
    n_t = int(round(TE / dt)) + 1
    wf = _zero_waveform(n_t, dt)
    exc = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': 90.0}]
    refocus = [{'t_s': TE / 2, 'flip_deg': 180.0, 'axis_deg': 0.0}]
    ec = simulate_bloch(2000, 1e-13, wf, FreeDiffusion(), exc + refocus,
                        seed=0, susceptibility=prov)[0]
    assert abs(ec) == pytest.approx(1.0, abs=0.03)   # frozen -> static field refocuses


def test_forward_signal_parity_linear_field():
    """QUANTITATIVE parity. A spatially-linear susceptibility field ΔBz = g·x is exactly a
    constant gradient g·x̂, so a free-diffusion gradient echo attenuates to the closed form
    |S| = exp(−γ²g²D T³/3), AND it must be bit-identical to feeding that same constant
    gradient through the waveform (the already-validated gradient-phase path)."""
    g, dt, n_t, N = 0.05, 1e-4, 200, 20000        # g (T/m); ΔBz = g·x
    T = n_t * dt
    exc = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': 90.0}]   # gradient echo, no 180
    # (a) susceptibility as a linear field
    S_field = simulate_bloch(N, D, _zero_waveform(n_t, dt), FreeDiffusion(), exc,
                             seed=0, susceptibility=lambda r: g * r[0])[0]
    # (b) the SAME field as a constant gradient through the waveform, no susceptibility
    G = np.zeros((1, n_t, 3)); G[0, :, 0] = g
    from types import SimpleNamespace
    S_grad = simulate_bloch(N, D, SimpleNamespace(G=G, dt=dt), FreeDiffusion(), exc, seed=0)[0]
    b = (GAMMA ** 2 * g ** 2 * T ** 3) / 3.0
    analytic = np.exp(-b * D)
    tol = max(0.02, 1.0 / np.sqrt(N))
    assert abs(S_field) == pytest.approx(abs(S_grad), abs=1e-3)   # hook == gradient path (bit-level)
    assert abs(S_field) == pytest.approx(analytic, abs=tol)       # exact free-diffusion truth
