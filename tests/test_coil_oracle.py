"""Tier 1: a machine that exists only as geometry, used as an oracle (dmipy-sim#364).

Nothing here reads the catalogue or any fitted coefficient. The field is integrated from the Biot-Savart
law over an explicit wire, so agreement with the scanner model is evidence rather than bookkeeping.
"""
import numpy as np
import pytest

from dmipy_sim.acquisition import coil, maxwell, solid_harmonics as sh

R = 0.09


def test_the_oracle_reproduces_the_analytic_loop_it_has_no_business_knowing():
    """An oracle is worth nothing until its own error is known. The on-axis field of a circular loop is
    closed form, and the polyline integration must land on it."""
    a = 0.3
    P = np.stack([np.zeros(9), np.zeros(9), np.linspace(-0.2, 0.2, 9)], -1)
    num = coil.polyline_field(coil.circular_loop(a, 0.0), P)[:, 2]
    ana = coil.MU0 * a ** 2 / (2.0 * (a ** 2 + P[:, 2] ** 2) ** 1.5)
    assert np.abs(num / ana - 1.0).max() < 1e-5


@pytest.mark.parametrize("make", [coil.maxwell_pair, coil.golay_saddle])
def test_the_field_it_produces_satisfies_maxwell_in_the_interior(make):
    """div B = 0 and curl B = 0 where there is no current. This checks the ORACLE, not the model -- if it
    failed, everything downstream of it would be worthless."""
    c = make()
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.08, 0.08, (40, 3))
    h = 2e-4
    J = np.zeros((len(pts), 3, 3))
    for j in range(3):
        d = np.zeros(3); d[j] = h
        J[:, :, j] = (c.field(pts + d) - c.field(pts - d)) / (2 * h)
    scale = np.abs(J).max()
    assert np.abs(J[:, 0, 0] + J[:, 1, 1] + J[:, 2, 2]).max() / scale < 1e-5      # div B
    for i, j in ((2, 1), (0, 2), (1, 0)):
        assert np.abs(J[:, i, j] - J[:, j, i]).max() / scale < 1e-5               # curl B


def test_the_concomitant_field_comes_out_as_bernstein_derived_it():
    """The canonical reference (Bernstein et al. 1998) gives a z gradient's concomitant term as
    Gz^2 rho^2 / (8 B0). Here both transverse components are INTEGRATED and squared, with no symmetry
    parameter and no assumed geometry, so landing on that formula is an independent derivation of it --
    and it is what lets this oracle MEASURE the alpha the catalogue currently states from one paper."""
    mp = coil.maxwell_pair()
    B0 = 0.064
    g = coil.GradientCoil(mp, "z").nominal_gradient()
    q = np.array([[0.03, 0.02, 0.01], [-0.05, 0.01, 0.04], [0.02, -0.06, -0.02]])
    got = coil.concomitant_field(mp, q, B0)
    book = g ** 2 * (q[:, 0] ** 2 + q[:, 1] ** 2) / (8.0 * B0)
    assert np.abs(got / book - 1.0).max() < 0.01


def test_an_axially_symmetric_coil_produces_only_the_harmonics_its_symmetry_allows():
    """A Maxwell pair is axially symmetric, so no m != 0 term may survive. This is the check that caught a
    reporting error rather than a physics one: the raw fit gives Z3X = +0.23, which reads as a gross
    symmetry violation until one notices that Z3X is order four, so at 9 cm it contributes a hundred times
    less than the residual it is fitting. A coefficient is not a size."""
    p = coil.GradientCoil(coil.maxwell_pair(), "z").potential()
    assert set(p) == {"Z"}, f"an axially symmetric pair produced {sorted(p)}"
    assert p["Z"] == pytest.approx(1.0, abs=1e-3)
    assert len(coil.GradientCoil(coil.maxwell_pair(), "z").potential(significant=False)) == 18


def test_a_transverse_coil_produces_its_axis_and_the_nonlinearity_that_names_it():
    """A saddle coil's leading nonlinearity is Z2X -- the same term the catalogued B0 law needs, which is
    not a coincidence: both are the lowest harmonic odd in x and even in z."""
    p = coil.GradientCoil(coil.golay_saddle(), "x").potential()
    assert set(p) == {"X", "Z2X"}, f"got {sorted(p)}"
    assert p["X"] == pytest.approx(1.0, abs=1e-3)


def test_every_fitted_potential_is_harmonic_under_tier_zero():
    """The oracle and the model must be judged by the same rule, or agreement means nothing."""
    rng = np.random.default_rng(1)
    q = rng.uniform(-R, R, (30, 3))
    for c, ax in ((coil.maxwell_pair(), "z"), (coil.golay_saddle(), "x")):
        phi = coil.GradientCoil(c, ax).potential()
        r = maxwell.harmonic_residual(lambda p, d=phi: sh.evaluate(d, p), q)
        assert r < 1e-6, f"{ax} potential residual {r:.1e}"


def test_the_expansion_residual_is_truncation_and_says_where_the_basis_stops():
    """A truncated harmonic series must lose accuracy as R^order. A residual that does NOT fall with the
    probe radius would be the wire's discretisation instead, and would mean the oracle, not the basis, is
    the limit. Measured here it falls steeply, so order four is what bounds a 9 cm DSV -- which is a
    finding about OUR basis, produced by the oracle."""
    gc = coil.GradientCoil(coil.golay_saddle(), "x")
    rng = np.random.default_rng(2)
    u = rng.normal(size=(400, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    res = []
    for radius in (0.03, 0.06, 0.09):
        res.append(gc.residual(4, u * (rng.uniform(0, 1, (400, 1)) ** (1 / 3)) * radius))
    assert res[0] < res[1] < res[2], f"the residual does not grow with radius: {res}"
    assert res[2] / res[0] > 10.0, f"only {res[2]/res[0]:.1f}x over 3x in radius -- not truncation"


def test_a_real_coil_cannot_have_a_diagonal_gradient_tensor():
    """The structural claim behind #350, from geometry alone. No step in building this tensor could have
    produced a diagonal one, and its off-diagonals are the same size as its diagonal departure."""
    coils = {"x": coil.GradientCoil(coil.golay_saddle(), "x"),
             "y": coil.GradientCoil(coil.golay_saddle(), "x"),      # y is x rotated; same structure
             "z": coil.GradientCoil(coil.maxwell_pair(), "z")}
    rng = np.random.default_rng(3)
    q = rng.uniform(-R, R, (60, 3))
    L = coil.gradient_tensor(coils, q)
    off = np.abs(L[:, ~np.eye(3, dtype=bool)]).max()
    dev = np.abs(np.einsum("nii->ni", L) - 1.0).max()
    assert off > 0.2 * dev, f"off-diagonals {off:.4f} are negligible against {dev:.4f}"
    maxwell.require_gradient_tensor_admissible(lambda p: coil.gradient_tensor(coils, p), q, "the coil L")


def test_the_transverse_concomitant_term_is_bernstein_s_too_and_not_the_catalogue_s_alpha():
    """A second independent reproduction of the canonical result, and an honest statement of what this
    oracle does NOT cover.

    For a symmetric CYLINDRICAL gradient set Bernstein gives the transverse coils' concomitant term as
    (Gx^2 + Gy^2) z^2 / (2 B0) -- coefficient one. Integrating both transverse components of a saddle coil
    and fitting that form recovers exactly that, so the oracle reproduces the textbook for the transverse
    gradients as well as the axial one.

    The catalogue's concomitant_alpha = 1/2 is NOT this number and is not contradicted by it: it is de Vos's
    figure for PARALLEL PLATE coils in an open magnet whose B0 is not along the bore, which is a different
    geometry that nothing here builds. Testing that claim needs a bi-planar coil set and a transverse B0,
    and until this oracle has one the alpha the Swoop uses rests on a single paper."""
    gx = coil.golay_saddle()
    B0 = 0.064
    g = coil.GradientCoil(gx, "x").nominal_gradient()
    rng = np.random.default_rng(0)
    q = rng.uniform(-0.07, 0.07, (600, 3))
    A = np.stack([q[:, 2] ** 2, q[:, 1] ** 2, q[:, 0] * q[:, 2]], -1) * (g ** 2 / (2.0 * B0))
    c, *_ = np.linalg.lstsq(A, coil.concomitant_field(gx, q, B0), rcond=None)
    assert c[0] == pytest.approx(1.0, abs=0.05), f"the z^2 coefficient is {c[0]:.3f}, not Bernstein's 1"
