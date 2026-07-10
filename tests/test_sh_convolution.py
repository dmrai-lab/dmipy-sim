"""Tests for Item 8: SH convolution for orientation distributions.

Test suite verifies:
1. Isotropic ODF = orientation average (fundamental identity)
2. Watson kappa=0 → isotropic ODF
3. Watson kappa→∞ → single-fiber response
4. Legendre fit residuals within MC noise
5. Waveform rotation preserves b-value
7. SH convolution ordering invariant (linearity)
"""

import numpy as np
import pytest

from dmipy_sim import (
    pgse, set_b, Cylinder,
    compute_fiber_response, apply_odf,
    watson_odf_sh, isotropic_odf_sh,
    calc_b, rotate_waveform,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_WALKERS = 30_000   # enough for 1% MC noise (sigma ~ 1/sqrt(30000) ~ 0.6%)
SEED = 42


def make_pgse_waveform():
    """PGSE with b = [0, 500, 1000, 2000] s/mm^2, gradient along x."""
    delta, DELTA = 10e-3, 30e-3
    bvecs = np.array([
        [0., 0., 1.],   # along fiber axis (z): maximum signal
        [1., 0., 0.],   # perpendicular: minimum signal
        [1., 0., 0.],
        [1., 0., 0.],
    ], dtype=np.float32)
    bvals = np.array([0., 500., 1000., 2000.]) * 1e6   # s/m^2
    wf = pgse(delta=delta, DELTA=DELTA, G_magnitude=0.04,
              bvecs=bvecs, n_t=100)
    wf = set_b(wf, bvals)
    return wf


def make_cylinder():
    """Cylinder along z, radius 5 µm."""
    return Cylinder(radius=5e-6, orientation=[0., 0., 1.])


# ---------------------------------------------------------------------------
# Test 5: Waveform rotation preserves b-value (cheap, no simulation)
# ---------------------------------------------------------------------------

class TestWaveformRotation:

    def test_b_value_preserved_multiple_angles(self):
        """rotate_waveform(theta) must not change b-value."""
        wf = make_pgse_waveform()
        b_orig = calc_b(wf)   # (n_measurements,)

        for theta_deg in [0., 15., 30., 45., 60., 75., 90.]:
            theta = np.radians(theta_deg)
            wf_rot = rotate_waveform(wf, theta=theta)
            b_rot = calc_b(wf_rot)
            np.testing.assert_allclose(
                b_rot, b_orig, rtol=1e-4,
                err_msg=f"b-value changed at theta={theta_deg}°")

    def test_zero_rotation_identity(self):
        """theta=0 rotation must return identical G array."""
        wf = make_pgse_waveform()
        wf_rot = rotate_waveform(wf, theta=0.0)
        np.testing.assert_allclose(
            np.array(wf_rot.G), np.array(wf.G), atol=1e-6)

    def test_90_rotation_swaps_components(self):
        """theta=pi/2 maps z-axis to x-axis."""
        # A waveform with gradient purely along z
        bvecs_z = np.array([[0., 0., 1.]], dtype=np.float32)
        wf = pgse(delta=10e-3, DELTA=30e-3, G_magnitude=0.04, bvecs=bvecs_z, n_t=50)
        wf_rot = rotate_waveform(wf, theta=np.pi / 2)
        G_orig = np.array(wf.G)
        G_rot = np.array(wf_rot.G)
        # z-component should become x-component after 90-degree y-rotation
        # R_y(pi/2): [c,0,s; 0,1,0; -s,0,c] with c=0, s=1 → [0,0,1; 0,1,0; -1,0,0]
        # So G_orig_z → G_rot_x (the x-component of rotated = G_orig_z * sin(pi/2) = G_orig_z)
        # G_rot[m,t,0] = G_orig[m,t,0]*cos(pi/2) + G_orig[m,t,2]*sin(pi/2) = G_orig[m,t,2]
        np.testing.assert_allclose(G_rot[..., 0], G_orig[..., 2], atol=1e-6)


# ---------------------------------------------------------------------------
# Test 2: Watson kappa=0 → isotropic
# ---------------------------------------------------------------------------

class TestWatsonIsotropicLimit:

    def test_watson_kappa0_equals_isotropic(self):
        """watson_odf_sh(kappa=0) must equal isotropic_odf_sh() to machine precision."""
        lmax = 8
        iso = isotropic_odf_sh(lmax=lmax)
        wat = watson_odf_sh(kappa=0.0, lmax=lmax)

        np.testing.assert_allclose(
            wat, iso, atol=1e-8,
            err_msg="Watson kappa=0 should equal isotropic ODF")

    def test_isotropic_l0_value(self):
        """c_0^0 of isotropic ODF must equal 1/(2*sqrt(pi))."""
        iso = isotropic_odf_sh(lmax=8)
        expected = 1.0 / (2.0 * np.sqrt(np.pi))
        assert abs(iso[0] - expected) < 1e-12


# ---------------------------------------------------------------------------
# Test 4: Legendre fit residuals within MC noise
# ---------------------------------------------------------------------------

class TestLegendreResiduals:

    def test_fit_residuals_within_mc_noise(self):
        """Legendre fit residuals must be within MC noise floor."""
        wf = make_pgse_waveform()
        geom = make_cylinder()
        lmax = 8
        n_walkers = N_WALKERS

        fiber_response, thetas, E_theta = compute_fiber_response(
            geometry=geom,
            acquisition_scheme=wf,
            n_walkers=n_walkers,
            lmax=lmax,
            seed=SEED,
            diffusivity=2e-9,
        )

        # Reconstruct E from Legendre coefficients
        from scipy.special import eval_legendre
        cos_thetas = np.cos(thetas)
        n_orders = lmax // 2 + 1
        E_fit = np.zeros_like(E_theta)
        for j in range(n_orders):
            l = 2 * j
            P_l = eval_legendre(l, cos_thetas)
            E_fit += fiber_response[j][None, :] * P_l[:, None]

        residuals = np.abs(E_theta - E_fit)
        mc_noise = 3.0 / np.sqrt(n_walkers)   # 3-sigma MC noise floor
        max_residual = residuals.max()

        assert max_residual < mc_noise, (
            f"Max Legendre residual {max_residual:.4f} exceeds MC noise floor "
            f"{mc_noise:.4f} (3/sqrt({n_walkers}))")


# ---------------------------------------------------------------------------
# Test 1: Isotropic ODF = orientation average
# ---------------------------------------------------------------------------

class TestIsotropicIdentity:

    def test_isotropic_odf_equals_solid_angle_average(self):
        """apply_odf with isotropic ODF must equal the solid-angle average of E."""
        wf = make_pgse_waveform()
        geom = make_cylinder()
        lmax = 8
        n_walkers = N_WALKERS

        fiber_response, thetas, E_theta = compute_fiber_response(
            geometry=geom,
            acquisition_scheme=wf,
            n_walkers=n_walkers,
            lmax=lmax,
            seed=SEED,
            diffusivity=2e-9,
        )

        iso_sh = isotropic_odf_sh(lmax=lmax)
        E_sh = apply_odf(fiber_response, iso_sh, lmax=lmax)

        # The l=0 Legendre coefficient f_0 = (1/2) integral_{-1}^1 E(x) dx
        # For axisymmetric geometry this equals (1/2) integral_{-1}^1 E(x) dx
        # = the solid-angle average.
        # apply_odf(iso) should return f_0 * sqrt(4*pi) * (1/(2*sqrt(pi))) = f_0
        E_f0 = fiber_response[0]   # l=0 Legendre coefficient

        np.testing.assert_allclose(
            E_sh, E_f0, rtol=1e-10,
            err_msg="apply_odf with isotropic ODF must equal f_0 (l=0 Legendre coeff)")

        # Also check that E_f0 is physically reasonable:
        # The first measurement is gradient along z (parallel to fiber) at b=0 effectively,
        # or if b>0, signal is attenuated. Just check it's in a sensible range.
        # (The simulation in compute_fiber_response does not apply T2 by default)
        assert 0.0 < E_sh[0] <= 1.1, \
            f"b=0 isotropic signal out of range: {E_sh[0]:.3f}"

    def test_isotropic_matches_numerical_average(self):
        """apply_odf with isotropic ODF must match weighted average of E(theta_k)."""
        wf = make_pgse_waveform()
        geom = make_cylinder()
        lmax = 8
        n_walkers = N_WALKERS

        fiber_response, thetas, E_theta = compute_fiber_response(
            geometry=geom,
            acquisition_scheme=wf,
            n_walkers=n_walkers,
            lmax=lmax,
            seed=SEED,
            diffusivity=2e-9,
        )

        # Numerical solid-angle average: mean(E * sin(theta)) / mean(sin(theta))
        # on the upper hemisphere — but we're comparing to f_0, which IS the
        # solid-angle average computed via GL quadrature.
        # Instead: verify that E_sh (= f_0) approximates the true average
        # of E at the simulated angles weighted by GL sin(theta) weights.
        iso_sh = isotropic_odf_sh(lmax=lmax)
        E_sh = apply_odf(fiber_response, iso_sh, lmax=lmax)

        # Use a coarse numerical estimate: uniform average of E at simulated angles
        # (upper hemisphere only, biased but gives the right order of magnitude)
        sin_weights = np.sin(thetas)
        E_numerical_avg = np.average(E_theta, axis=0, weights=sin_weights)

        # Relative tolerance: allow 5% (limited by n_angles and MC noise)
        np.testing.assert_allclose(
            E_sh, E_numerical_avg, rtol=0.05,
            err_msg="Isotropic SH signal must be close to sin-weighted average")


# ---------------------------------------------------------------------------
# Test 3: Watson kappa→∞ → single fiber
# ---------------------------------------------------------------------------

class TestWatsonHighKappa:

    def test_large_kappa_approaches_parallel_signal(self):
        """Watson kappa=1000 signal must approach the theta=0 (parallel) response."""
        wf = make_pgse_waveform()
        geom = make_cylinder()
        lmax = 8
        n_walkers = N_WALKERS

        fiber_response, thetas, E_theta = compute_fiber_response(
            geometry=geom,
            acquisition_scheme=wf,
            n_walkers=n_walkers,
            lmax=lmax,
            seed=SEED,
            diffusivity=2e-9,
        )

        # The theta=0 signal: waveform along fiber axis → minimum restricted diffusion
        # For PGSE with large b perpendicular, the signal approaches 0.
        # With kappa=1000 (very concentrated along z), the signal should approach
        # the signal at theta=0.
        wat_sh = watson_odf_sh(kappa=1000.0, lmax=lmax)
        E_watson = apply_odf(fiber_response, wat_sh, lmax=lmax)

        # Reconstruct E at theta=0 from the Legendre fit
        from scipy.special import eval_legendre
        cos0 = np.cos(0.0)  # = 1.0
        n_orders = lmax // 2 + 1
        E_parallel_fit = np.zeros(wf.G.shape[0])
        for j in range(n_orders):
            E_parallel_fit += fiber_response[j] * eval_legendre(2 * j, cos0)

        # For measurements with significant signal contrast, Watson kappa=1000
        # must be within 2% of the parallel response
        # (measured at b=0 which should be ~1 regardless)
        np.testing.assert_allclose(
            E_watson, E_parallel_fit, rtol=0.02,
            err_msg="Watson kappa=1000 should match single-fiber (theta=0) response")


# ---------------------------------------------------------------------------
# Test 7: Linearity / ordering invariant
# ---------------------------------------------------------------------------

class TestOrderingInvariant:

    def test_convolve_after_average_equals_average_of_convolved(self):
        """apply_odf is linear: it must commute with averaging across draws.

        E_correct = apply_odf(mean(f_k)) must equal mean(apply_odf(f_k))
        to floating-point precision (this is exact by linearity).
        """
        wf = make_pgse_waveform()

        # Use two different radii to simulate two draws
        radii = [3e-6, 7e-6]
        lmax = 8
        n_walkers = N_WALKERS // 2   # fewer walkers per draw for speed

        odf_sh = watson_odf_sh(kappa=4.0, lmax=lmax)

        responses = []
        for k, r in enumerate(radii):
            geom = Cylinder(radius=r, orientation=[0., 0., 1.])
            fr, _, _ = compute_fiber_response(
                geometry=geom,
                acquisition_scheme=wf,
                n_walkers=n_walkers,
                lmax=lmax,
                seed=SEED + k * 37,
                diffusivity=2e-9,
            )
            responses.append(fr)

        # Correct: convolve after averaging
        mean_fr = np.mean(responses, axis=0)
        E_correct = apply_odf(mean_fr, odf_sh, lmax=lmax)

        # Also correct by linearity: average of convolved
        E_per_draw = [apply_odf(fr, odf_sh, lmax=lmax) for fr in responses]
        E_from_avg = np.mean(E_per_draw, axis=0)

        np.testing.assert_allclose(
            E_correct, E_from_avg, rtol=1e-10,
            err_msg="apply_odf must commute with averaging (linearity)")
