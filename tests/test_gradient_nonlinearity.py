"""The gradient coils' own nonlinearity, measured from open data (dmipy-sim#322, the deferred item).

A gradient coil's field is only linear near isocentre; away from it a commanded gradient vector is
delivered mis-scaled and tilted. That is an ENCODING error -- it changes the b a voxel receives and the
direction it receives it along -- not a shading, which is why it belongs on the sequence rather than as a
layer. The Swoop publishes nothing about it; the coefficients here are regressed from the NIST dual-field
database and their limits are recorded with them.
"""
import numpy as np
import pytest

from dmipy_sim import sequences
from dataclasses import replace

from dmipy_sim.acquisition import maxwell, scanner_constants as scc, solid_harmonics
from dmipy_sim.acquisition.scanners import ScannerLimits
from dmipy_sim.phantom import Grid
from dmipy_sim.phantom.bore import gradient_tensor_map

AXES = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def _seq(n_t=300):
    return sequences.pgse(AXES, 35e-3, 42e-3, bvalues=[0.945e9] * 3, n_t=n_t)


def test_the_tensor_is_the_identity_at_isocentre_and_traceless_in_its_departure():
    """Two properties that are structural rather than incidental. It is the identity at isocentre because
    that is what defines the commanded gradient. And its departure is TRACELESS, because the measurement
    that produced it normalised each axis by the trace -- so a common mode, all three axes mis-scaled
    together, is not OBSERVABLE. The representation could state one -- three coils each carrying the same added
    curvature give tr L = 3(1 + e x) -- so what is missing is the data to set it, not the freedom to say it.
    This checks that the shipped tensor does not invent one."""
    s = ScannerLimits.of("swoop")
    np.testing.assert_allclose(s.gradient_tensor(np.zeros((1, 3)))[0], np.eye(3), atol=1e-15)
    coeffs = np.array([s.d_scale_x_dx, s.d_scale_y_dx, s.d_scale_z_dx])
    assert abs(coeffs.sum()) < 1e-3 * np.abs(coeffs).max(), "the departure is not traceless"
    for x in (0.05, -0.05, 0.08):
        L = s.gradient_tensor(np.array([[x, 0.0, 0.0]]))[0]
        assert np.linalg.det(L) == pytest.approx(1.0, abs=2e-3)      # unit determinant to first order
        # ON THE X AXIS the off-diagonals vanish identically (they go as y and z), which is why probing
        # only this line could never have told a diagonal L from an admissible one -- see the test below.
        np.testing.assert_allclose(L - np.diag(np.diag(L)), 0.0, atol=1e-15)


def test_the_delivered_b_is_the_squared_scale_so_a_gradient_error_arrives_doubled():
    """The reason a sub-percent coil error matters. b goes as the square of the gradient, so a fractional
    error `e` in what is delivered becomes `2e` in a diffusivity fitted against the nominal b. At 7 cm the
    catalogued 0.45 %/cm becomes a 6.5 % error in the y-encoded measurement."""
    s, seq = ScannerLimits.of("swoop"), _seq()
    x = 0.07
    L = s.gradient_tensor(np.array([[x, 0.0, 0.0]]))[0]
    got = seq.with_gradient_nonlinearity(L).b() / seq.b() - 1.0
    for i, axis in enumerate("xyz"):
        scale = 1.0 + getattr(s, f"d_scale_{axis}_dx") * x
        assert got[i] == pytest.approx(scale ** 2 - 1.0, rel=2e-3)   # exactly |L u|^2, to float32 G
    assert got[1] > 0.06 and got[2] < -0.05
    # and it reverses across the bore, because the coefficient is a gradient in x
    other = seq.with_gradient_nonlinearity(s.gradient_tensor(np.array([[-x, 0.0, 0.0]]))[0]).b() / seq.b() - 1.0
    assert np.sign(other[1]) == -np.sign(got[1])


def test_a_coil_error_vanishes_where_the_gradient_does_and_a_magnet_s_own_does_not():
    """The signature separating the two transforms, and the reason they cannot be merged. A magnet's own
    gradient is a constant added to G, so it is on through the pulses and the dead times alike -- a magnet
    does not switch off. A coil's nonlinearity RESCALES what was commanded, so wherever nothing is played it
    contributes nothing."""
    s, seq = ScannerLimits.of("swoop"), _seq()
    at = np.array([[0.06, 0.0, 0.0]])
    nl = seq.with_gradient_nonlinearity(s.gradient_tensor(at)[0])
    bg = seq.with_background_gradient([1.0e-3, 0.0, 0.0])          # any constant the magnet might add
    off = np.abs(np.asarray(seq.G)).max(axis=-1) < 1e-12                  # samples with nothing commanded
    assert off.any(), "the fixture has no dead time to test with"
    assert np.abs(np.asarray(nl.imposed_gradient)[off]).max() < 1e-12
    assert np.abs(np.asarray(bg.imposed_gradient)[off]).max() > 1e-6


def test_the_coefficients_carry_what_they_cannot_say():
    """A derived number is only as good as the limits recorded beside it. Both of these are structural, and
    a reader who takes the value without them will over-claim: the common mode is unobservable, and only the
    x dependence survives the argument from mirror symmetry."""
    leaf = scc.get_limit("hyperfine_swoop_64mT", "gradient_nonlinearity", "d_scale_y_dx")
    assert leaf["confidence"] == "derived" and leaf["source_key"] == "nist2025_dualfield_dwi"
    for phrase in ("TRACELESS", "mirror", "inference", "NOT excluded"):
        assert phrase in leaf["context"], f"the leaf does not record {phrase!r}"
    assert leaf["value"] == pytest.approx(0.4549, abs=1e-4)


def test_a_tensor_is_rotated_by_similarity_where_a_vector_is_not():
    """A position goes forward into the bore; the tensor evaluated there comes back as `R^T L R`, not as a
    single product. Getting it wrong leaves something still symmetric and still plausible, so the test has to
    check the value rather than the shape."""
    s = ScannerLimits.of("swoop")
    grid = Grid(shape=(3, 3, 3), voxel_size_m=(0.02,) * 3, origin_m=(-0.02,) * 3, isocenter_m=(0.0,) * 3)
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])    # grid -> scanner, 90 deg about z
    at = np.array([[0.0, 0.03, 0.0]])                                     # grid +y  ->  scanner +x
    L = gradient_tensor_map(s, grid, to_scanner=R)(at)[0]
    bore = s.gradient_tensor(at @ R.T)[0]
    np.testing.assert_allclose(L, R.T @ bore @ R, atol=1e-12)
    # the grid's y axis is the scanner's x, so the scanner's y-scaling lands on the grid's x
    assert L[0, 0] == pytest.approx(bore[1, 1])
    assert np.linalg.eigvalsh(L).tolist() == pytest.approx(np.linalg.eigvalsh(bore).tolist())


def test_it_is_refused_twice_and_absent_where_uncatalogued():
    s, seq = ScannerLimits.of("swoop"), _seq()
    L = s.gradient_tensor(np.array([[0.05, 0, 0]]))[0]
    with pytest.raises(ValueError, match="already carries"):
        seq.with_gradient_nonlinearity(L).with_gradient_nonlinearity(L)
    with pytest.raises(ValueError, match="one 3x3 tensor"):
        seq.with_gradient_nonlinearity(np.eye(2))
    grid = Grid(shape=(2, 2, 2), voxel_size_m=(0.02,) * 3, origin_m=(-0.01,) * 3, isocenter_m=(0.0,) * 3)
    for name in ("prisma", "connectom", "terra"):
        assert ScannerLimits.of(name).gradient_tensor(np.zeros((1, 3))) is None
        assert gradient_tensor_map(ScannerLimits.of(name), grid) is None


def test_the_tensor_tilts_off_axis_because_a_diagonal_one_cannot_be_a_field():
    """The #350 correction. A DIAGONAL L that is not the identity is impossible: a diagonal L makes each
    coil's B_z depend on its own axis alone, and Laplace then forces that dependence to be linear, so a
    diagonal L is the identity or it is nothing.

    That off-diagonals EXIST is therefore forced. Their VALUES are not: the measurement gives the diagonal
    only, and each coil admits a four-parameter family of harmonic completions on top of it. What ships is
    the minimum-norm, parity-preserving member -- see gradient_potentials -- so L_xy = a_y y is a modelling
    choice with a reason, not a measurement.

    An admissible alternative that breaks the y coil's reflection parity by as much as the measured effect
    moves L by the whole nonlinearity, and tier 0 cannot tell the two apart. This test pins the SHIPPED
    choice, which is what a regression test is for; it is not evidence that the choice is unique."""
    s = ScannerLimits.of("swoop")
    R = s.b0_validity_radius or 0.08

    # the two forced off-diagonal terms, read straight off the measured coefficients
    L = s.gradient_tensor(np.array([0.0, R, 0.0]))
    assert L[0, 1] == pytest.approx(s.d_scale_y_dx * R, rel=1e-9), "L_xy is not a_y y"
    L = s.gradient_tensor(np.array([0.0, 0.0, R]))
    assert L[0, 2] == pytest.approx(s.d_scale_z_dx * R, rel=1e-9), "L_xz is not a_z z"

    # The off-diagonal and diagonal departures are the SAME FUNCTION of position -- a_j times a coordinate
    # -- so at a point displaced equally in x and y they are equal. Asserting that pointwise says something
    # about the field; taking max|off| / max|diag| over a CUBE would only say the cube has equal sides
    # (halve its y and z and the ratio becomes 0.5), so it would be a test of the sampling box.
    e = R / np.sqrt(2.0)
    M = s.gradient_tensor(np.array([e, e, 0.0]))
    assert M[0, 1] == pytest.approx(s.d_scale_y_dx * e, rel=1e-9)      # L_xy = a_y y
    assert M[1, 1] - 1.0 == pytest.approx(s.d_scale_y_dx * e, rel=1e-9)  # L_yy - 1 = a_y x, equal here
    rng = np.random.default_rng(0)
    q = rng.uniform(-R, R, (400, 3))
    q = q[np.linalg.norm(q, axis=1) <= R]
    Ls = s.gradient_tensor(q)
    off = np.abs(Ls[:, ~np.eye(3, dtype=bool)]).max()
    assert off > 0.5 * np.abs(np.einsum("nii->ni", Ls) - 1.0).max()

    tilt = []
    for M in Ls:
        w, V = np.linalg.eigh(0.5 * (M + M.T))
        tilt.append(np.rad2deg(np.arccos(np.clip(np.abs(V[:, np.argmax(np.abs(w))]).max(), 0, 1))))
    assert np.median(tilt) > 15.0, f"the eigenframe barely rotates (median {np.median(tilt):.1f} deg)"


def test_the_tensor_is_admissible_under_tier_zero():
    """It is built from harmonic potentials, so it cannot express an inadmissible L -- and the check that
    refused the previous diagonal tensor now passes on this one."""
    s = ScannerLimits.of("swoop")
    rng = np.random.default_rng(1)
    q = rng.uniform(-0.045, 0.045, (200, 3))
    maxwell.require_gradient_tensor_admissible(s.gradient_tensor, q, "the Swoop L(r)")
    for j, phi in s.gradient_potentials().items():
        r = maxwell.harmonic_residual(lambda p, c=phi: solid_harmonics.evaluate(c, p), q[:20])
        assert r < 1e-6, f"coil {j}'s potential is not harmonic ({r:.1e})"


def test_the_one_free_parameter_is_bounded_and_attached_to_the_smallest_coefficient():
    """The x coil is the only one with freedom, because its own derivative is along the axis the
    mis-scaling depends on: d/dx (x + a_x x^2 / 2) needs x^2, which is not harmonic, so the curvature must
    be borrowed from y or from z. That share is the only thing chosen in this model.

    It costs little, and the catalogue says so rather than leaving it arbitrary: a_x is the smallest of the
    three coefficients, so the whole family spans a few per cent of the nonlinearity and essentially nothing
    of a b value.

    What this does NOT show is that the completion as a whole is nearly determined. lam is the one member of
    the freedom that is EXPOSED; the coil-parity assumption behind the off-diagonals is the larger one and is
    not bounded by the data at all. The b insensitivity below is also weaker than it reads: columns y and z
    carry no lam, so only the x direction is a live test of it."""
    s = ScannerLimits.of("swoop")
    assert abs(s.d_scale_x_dx) < 0.1 * max(abs(s.d_scale_y_dx), abs(s.d_scale_z_dx))

    rng = np.random.default_rng(2)
    q = rng.uniform(-0.045, 0.045, (300, 3))
    ends = [replace(s, gradient_completion=lam).gradient_tensor(q) for lam in (0.0, 1.0)]
    mid = replace(s, gradient_completion=0.5).gradient_tensor(q)
    spread = max(np.abs(e - mid).max() for e in ends)
    nonlin = np.abs(mid - np.eye(3)).max()
    assert spread < 0.05 * nonlin, f"the completion choice moves L by {spread/nonlin:.1%} of its own size"

    u = np.array([1.0, 0.0, 0.0])
    b = [np.sum(np.einsum("nij,j->ni", L, u) ** 2, -1) for L in ends]
    assert np.abs(b[0] - b[1]).max() < 1e-4, "the completion choice moves a b value"
