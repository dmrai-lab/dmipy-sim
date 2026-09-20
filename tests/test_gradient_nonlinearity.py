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
from dmipy_sim.acquisition import scanner_constants as scc
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
    together, is not merely unmeasured but inexpressible here. An L that could express one would be claiming
    more than the data supports."""
    s = ScannerLimits.of("swoop")
    np.testing.assert_allclose(s.gradient_tensor(np.zeros((1, 3)))[0], np.eye(3), atol=1e-15)
    coeffs = np.array([s.d_scale_x_dx, s.d_scale_y_dx, s.d_scale_z_dx])
    assert abs(coeffs.sum()) < 1e-3 * np.abs(coeffs).max(), "the departure is not traceless"
    for x in (0.05, -0.05, 0.08):
        L = s.gradient_tensor(np.array([[x, 0.0, 0.0]]))[0]
        assert np.linalg.det(L) == pytest.approx(1.0, abs=2e-3)      # unit determinant to first order
        np.testing.assert_allclose(L - np.diag(np.diag(L)), 0.0, atol=1e-15)   # diagonal: nothing tilts


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
