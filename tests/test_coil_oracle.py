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


@pytest.mark.parametrize("make", [coil.maxwell_pair, coil.golay_saddle, coil.biplanar_pair])
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
    # The y coil must be a REAL rotation of the x saddle. Using the same object for both made two columns
    # of L identical, so |det L| was 4.5e-18 -- and both the off-diagonal and the diagonal departure came
    # out ~1 by construction rather than by physics, which is not evidence of anything.
    rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    gy = coil.Coil([(v @ rz.T, i) for v, i in coil.golay_saddle().turns], name="golay_y")
    coils = {"x": coil.GradientCoil(coil.golay_saddle(), "x"),
             "y": coil.GradientCoil(gy, "y"),
             "z": coil.GradientCoil(coil.maxwell_pair(), "z")}
    rng = np.random.default_rng(3)
    q = rng.uniform(-R, R, (60, 3))
    L = coil.gradient_tensor(coils, q)
    off = np.abs(L[:, ~np.eye(3, dtype=bool)]).max()
    dev = np.abs(np.einsum("nii->ni", L) - 1.0).max()
    assert abs(np.linalg.det(L[0])) > 1e-3, "the three coils are not independent directions"
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


def test_alpha_is_measured_from_geometry_and_needs_no_magnet():
    """The symmetry parameter the catalogue states for the Swoop, measured instead of cited.

    alpha is how an AXIAL coil's divergence is shared between the two directions transverse to B0. It is
    fixed entirely by where the wires are, and cylindrical symmetry forces the even half -- which is what a
    Maxwell pair must therefore give.

    The part that makes the Swoop's claim testable at all: NOTHING about the magnet enters, only the
    DIRECTION of its field, because that is what defines transverse. A Halbach's B0 comes from magnetised
    blocks rather than free currents, and none of them are needed here."""
    assert coil.concomitant_alpha(coil.maxwell_pair()) == pytest.approx(0.5, abs=1e-3)


def test_the_concomitant_field_depends_on_which_way_b0_points():
    """On a Halbach B0 is TRANSVERSE to the bore, so the two components that enter |B_perp|^2 are not the
    two a cylindrical magnet's would be. Reading them as x and y regardless computes a real number for the
    wrong machine -- this asserts the two frames genuinely differ, so the argument cannot be dropped."""
    mp = coil.maxwell_pair()
    q = np.array([[0.03, 0.02, 0.01], [-0.05, 0.01, 0.04]])
    along = coil.concomitant_field(mp, q, 0.064, b0_axis=(0, 0, 1))
    across = coil.concomitant_field(mp, q, 0.064, b0_axis=(1, 0, 0))
    # atol=0 deliberately: these fields are ~1e-13 T, so np.allclose's default atol of 1e-8 calls any two
    # of them equal and the check would pass whatever the code did.
    assert not np.allclose(along, across, rtol=0.05, atol=0.0), "the b0_axis argument changes nothing"
    assert np.abs(across / along - 1.0).max() > 0.3


def test_a_transverse_coil_has_no_alpha_and_says_so():
    """alpha belongs to the coil whose gradient lies ALONG B0. Asked for one from a coil that is transverse
    in the given frame, a naive implementation returns a finite number -- 2.0 for this pair, outside the
    [0, 1] a shared divergence can occupy -- rather than refusing."""
    with pytest.raises(ValueError, match="AXIAL coil"):
        coil.concomitant_alpha(coil.maxwell_pair(), b0_axis=(1, 0, 0))


def test_both_published_alphas_are_one_geometry_at_two_parameter_points():
    """The catalogue's concomitant_alpha = 1/2 for the Swoop is an INFERENCE from a class statement -- de
    Vos gives 1/2 for parallel-plate coils in an open C- or H-shaped magnet and 0 for a Halbach array, and
    the Swoop is taken to be the former. This measures the class statement instead of citing it.

    Both values come out of the same coil family, so the unification is real. What sets alpha is the PAIR
    (width/gap, length/gap) and not the plate aspect ratio: at a fixed aspect ratio of three, alpha runs
    0.02 to 0.43 as the gap grows, because a pair whose gap dwarfs both plate dimensions degenerates to an
    axially symmetric dipole pair. An earlier version of this test asserted the aspect ratio was the whole
    physics, which is false and made the Swoop inference look better constrained than it is: it needs BOTH
    ratios, and neither is public."""
    B = (0.0, 1.0, 0.0)
    # square plates are four-fold symmetric about B0 at ANY gap, so 1/2 is forced there
    for gap in (0.05, 0.15, 0.45):
        sq = coil.biplanar_pair(half_gap=gap, width=0.30, length=0.30)
        assert coil.concomitant_alpha(sq, b0_axis=B) == pytest.approx(0.5, abs=1e-3)
        assert coil.cross_term(sq, b0_axis=B) < 1e-6           # genuinely symmetric, not merely reading 1/2

    # away from square, BOTH knobs move it
    by_aspect = [1 - coil.concomitant_alpha(coil.biplanar_pair(half_gap=0.15, width=0.30, length=0.30 * r),
                                            b0_axis=B) for r in (1.0, 2.0, 3.0, 10.0)]
    by_gap = [1 - coil.concomitant_alpha(coil.biplanar_pair(half_gap=g, width=0.30, length=0.90),
                                         b0_axis=B) for g in (0.05, 0.15, 0.45, 1.00)]
    assert by_aspect == sorted(by_aspect, reverse=True) and by_aspect[-1] < 0.01
    assert by_gap == sorted(by_gap), f"the gap must move alpha too, got {by_gap}"
    assert by_gap[-1] / by_gap[0] > 10.0, "alpha is not sensitive to the gap, contradicting the docstring"


def test_alpha_is_a_property_of_the_coil_and_not_of_the_frame():
    """Deriving the transverse frame from a global axis makes alpha a property of the coordinate system. An
    aspect-ratio-three pair reported 0.05 at one azimuth and 0.95 rotated ninety degrees about its own B0 --
    the same coil, the same physics, the whole range of the answer.

    alpha is now read along the transverse block's own PRINCIPAL axes, which is the only choice that depends
    on the coil. And alpha = 1/2 does not imply a symmetric coil: an asymmetric one read at 45 degrees to
    its principal axes reads 1/2 too, which is why cross_term exists to tell them apart."""
    rz = lambda t: np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    base = coil.biplanar_pair(width=0.30, length=0.90)
    B = (0.0, 1.0, 0.0)
    got = [coil.concomitant_alpha(coil.Coil([(v @ rz(np.deg2rad(d)).T, i) for v, i in base.turns]), b0_axis=B)
           for d in (0, 15, 30, 45, 60, 90)]
    assert np.ptp(got) < 1e-5, f"alpha moved with the coil's azimuth: {got}"

    tilted = coil.Coil([(v @ rz(np.deg2rad(45)).T, i) for v, i in base.turns])
    assert coil.concomitant_alpha(tilted, b0_axis=B, transverse_axis=(1, 0, 0)) == pytest.approx(0.5, abs=1e-3)
    assert coil.cross_term(tilted, b0_axis=B) > 0.5, "a 45-degree impostor must not look symmetric"


def test_alpha_zero_moves_the_concomitant_field_transversally_as_de_vos_describes():
    """de Vos's qualitative claim for alpha = 0 -- 'a single cross-term of double amplitude replaces the
    conventional pair' and the field 'grows transversally rather than axially' -- reproduced from geometry.

    At alpha = 1/2 the two transverse directions carry the term equally. At alpha = 0 it is carried entirely
    by one of them and VANISHES along the other, which is what 'a single term' means. This is a check on the
    source, not on us: an independent derivation agreeing with a paper whose DOI was wrong is worth having.
    """
    B0, G = 0.064, 0.023
    sq = coil.biplanar_pair(width=0.30, length=0.30)
    lg = coil.biplanar_pair(width=0.30, length=3.00)
    n = (0.0, 1.0, 0.0)
    along_x = np.array([[0.10, 0.0, 0.0]])
    along_z = np.array([[0.0, 0.0, 0.10]])

    sq_x = coil.concomitant_field(sq, along_x, B0, b0_axis=n, gradient_T_m=G)[0]
    sq_z = coil.concomitant_field(sq, along_z, B0, b0_axis=n, gradient_T_m=G)[0]
    lg_x = coil.concomitant_field(lg, along_x, B0, b0_axis=n, gradient_T_m=G)[0]
    lg_z = coil.concomitant_field(lg, along_z, B0, b0_axis=n, gradient_T_m=G)[0]

    assert sq_x == pytest.approx(sq_z, rel=1e-6)          # alpha = 1/2: the two directions are equivalent
    assert lg_z < 0.02 * lg_x                             # alpha = 0: one term survives, the other vanishes
    assert lg_x > 3.0 * sq_x                              # and the survivor is much larger


def test_a_requested_gradient_is_referred_to_the_axis_b0_lies_along():
    """Scaling a coil to a stated gradient read along a fixed z divides by nearly zero for any machine whose
    B0 is not the bore axis -- which is every Halbach, and the Swoop. It returned 1.1e8 uT before this.

    The refusal is for a genuinely degenerate case and not for any off-axis reading: a TRANSVERSE coil has
    no axial gradient at all (a Golay x-coil has ``dBz/dz = 3e-19 T/m`` at isocentre), so there is nothing
    to refer a requested gradient to. An axial coil read in a rotated frame is a different matter -- its
    along-frame gradient is half G, not zero -- and must still answer."""
    c = coil.biplanar_pair(width=0.30, length=0.30)
    got = coil.concomitant_field(c, np.array([[0.10, 0.0, 0.0]]), 0.064,
                                 b0_axis=(0, 1, 0), gradient_T_m=0.023)[0]
    assert 1e-6 < got < 1e-4, f"{got:.3e} T is not a physical concomitant field at 10 cm"

    with pytest.raises(ValueError, match="no gradient along b0_axis"):
        coil.concomitant_field(coil.golay_saddle(), np.array([[0.1, 0.0, 0.0]]), 0.064,
                               b0_axis=(0, 0, 1), gradient_T_m=0.023)


def test_the_probe_is_a_quadrature_and_not_a_random_cloud():
    """Solid harmonics are orthogonal over the sphere, but a RANDOM sample makes them only approximately so,
    and the residual correlation lets a large odd coefficient leak into a small even one. A symmetric Maxwell
    pair has exactly zero even content; 800 random points reported Z2 at 7e-4, and the Golay's Z2X moved
    3.40 to 3.51 across seeds -- a 3 per cent spread on a number quoted to four figures."""
    mp = coil.GradientCoil(coil.maxwell_pair(), "z")
    assert abs(mp.potential(significant=False)["Z2"]) < 1e-5, "even content leaked into a symmetric coil"

    z2x = [coil.GradientCoil(coil.golay_saddle(), "x", n_ang=a, n_shell=s).potential()["Z2X"]
           for a, s in ((100, 8), (160, 5), (80, 10), (200, 4))]
    assert np.ptp(z2x) / np.mean(z2x) < 5e-3, f"the coefficient moved with the quadrature split: {z2x}"


def test_a_dropped_term_reports_the_size_it_could_have_had():
    """potential() drops terms below its own residual, so a term under that is reported ABSENT rather than
    small -- and absence reads like a symmetry. A Maxwell pair whose loops differ by half a per cent has a
    genuine Z2 of -8e-3 and comes back perfectly linear. The floor is what makes that honest."""
    def asymmetric(dr, R=0.3):
        h = R * np.sqrt(3) / 2
        return coil.Coil([(coil.circular_loop(R * (1 + dr), +h), +1.0),
                          (coil.circular_loop(R, -h), -1.0)])

    gc = coil.GradientCoil(asymmetric(5e-3), "z")
    floor = gc.detection_floor()
    z = np.array([[0.0, 0.0, -2e-3], [0.0, 0.0, 0.0], [0.0, 0.0, 2e-3]])
    b = gc.coil.bz(z)
    true_z2 = (b[0] - 2 * b[1] + b[2]) / (2e-3) ** 2 / (4 * gc.nominal_gradient())

    assert "Z2" not in gc.potential(), "this pair's Z2 is below the residual and should be dropped"
    assert abs(true_z2) < floor["Z2"], "a term above the floor must not be dropped"
    assert floor["Z2"] < 1.0, "the floor is not informative"
    # and a term well above the floor does survive
    assert "Z2" in coil.GradientCoil(asymmetric(5e-2), "z").potential()


def test_the_gradient_guard_does_not_depend_on_the_winding_current():
    """g0 scales with the current, so an absolute floor let the SAME transverse coil refuse at 1 A and
    return 1/noise at 10 A. A physical refusal cannot turn on how hard the coil is driven."""
    for current in (1.0, 10.0, 1000.0):
        with pytest.raises(ValueError, match="no gradient along b0_axis"):
            coil.concomitant_field(coil.golay_saddle(current=current), np.array([[0.1, 0.0, 0.0]]),
                                   0.064, b0_axis=(0, 0, 1), gradient_T_m=0.023)
