"""Tier 0 of the scanner model's evaluation harness: what Maxwell gives for free (dmipy-sim#364).

These checks need no oracle, no reference machine and no measurement. Each one is a property any field the
model emits must have because of what a magnetic field IS, and each is the check that would have caught a
defect this repository actually shipped.
"""
import numpy as np
import pytest

from dmipy_sim.acquisition import maxwell, solid_harmonics as sh
from dmipy_sim.acquisition.scanners import SCANNERS, ScannerLimits

P = 0.05 * np.array([[0.31, -0.47, 0.23], [-0.19, 0.11, -0.53], [0.41, 0.37, 0.17], [0.57, 0.29, -0.31]])


def test_the_residual_separates_a_field_from_a_non_field_by_ten_orders():
    """The check is worth nothing if its two populations overlap. A harmonic gives roundoff because its
    three second differences CANCEL; anything else gives an order-one number because they do not."""
    for f in (lambda q: 2 * q[:, 2] ** 2 - q[:, 0] ** 2 - q[:, 1] ** 2,
              lambda q: q[:, 0] * (4 * q[:, 2] ** 2 - q[:, 0] ** 2 - q[:, 1] ** 2)):
        assert maxwell.harmonic_residual(f, P) < 1e-8
    for f in (lambda q: np.sum(q ** 2, -1), lambda q: q[:, 2] ** 2):
        assert maxwell.harmonic_residual(f, P) == pytest.approx(1.0, abs=1e-6)


def test_the_residual_does_not_depend_on_the_stencil_width():
    """The flaw this formulation exists to avoid. Normalising by the field's VALUE makes the ratio fall as
    h^2, so any tolerance could be met by shrinking h and a non-field would pass by parameter choice."""
    r = [maxwell.harmonic_residual(lambda q: np.sum(q ** 2, -1), P, h=h) for h in (1e-3, 1e-4, 1e-5)]
    assert max(r) - min(r) < 1e-6, f"the verdict moved with the stencil: {r}"


def test_a_field_with_no_curvature_at_all_is_harmonic_and_not_a_violation():
    """x y is harmonic and its three second differences vanish IDENTICALLY, so an unguarded ratio divides
    roundoff by roundoff and reports a clean field as a violation."""
    assert maxwell.harmonic_residual(lambda q: q[:, 0] * q[:, 1], P) == 0.0
    assert maxwell.harmonic_residual(lambda q: q[:, 0] * 3.0 - 2.0, P) == 0.0


def test_the_c_r_squared_law_is_refused_by_name():
    """The #349 defect, as a regression. It is not an inaccurate field -- laplacian(r^2) = 6, so it puts a
    minimum at isocentre where the maximum principle forbids one, and every downstream number stays finite
    while being wrong."""
    with pytest.raises(ValueError, match="is not a magnetic field"):
        maxwell.require_harmonic(lambda q: np.sum(q ** 2, -1), P, "a c r^2 inhomogeneity law")


def test_every_term_of_the_standard_basis_passes_the_same_check_a_law_does():
    """A term and a law judged by different rules is two rules, and only one of them was checked."""
    worst = max(sh.check_harmonic(n) for n in sh.TERMS)
    assert worst < 1e-8, f"worst basis term residual {worst:.2e}"


def test_every_catalogued_field_law_is_a_field():
    """The catalogue's own laws, through the accessor a consumer uses."""
    checked = 0
    for name in SCANNERS:
        s = ScannerLimits.of(name)
        if not s.harmonic_law():
            continue
        R = 0.5 * (s.b0_validity_radius or 0.1)
        assert maxwell.require_harmonic(lambda q: sh.evaluate(s.harmonic_law(), q), R * P / 0.05,
                                        f"{name} law") < 1e-8
        checked += 1
    assert checked, "no catalogued scanner declares a field law, so this asserted nothing"


def test_a_machine_whose_transmit_axis_lies_along_its_field_is_refused():
    """Only the perpendicular component excites, so this machine would produce no signal. Both axes are
    plausible unit vectors, so the confusion never announces itself."""
    with pytest.raises(ValueError, match="not perpendicular"):
        maxwell.require_transverse((0, 0, 1), (0, 0.001, 1), "a confused machine")
    assert maxwell.require_transverse((0, 0, 1), (1, 0, 0), "a sane one") < 1e-12


def test_a_diagonal_gradient_tensor_that_varies_is_refused():
    """The #350 theorem, which needs no data. If L is diagonal then each coil's B_z depends on its own axis
    alone; Laplace forces that dependence to be linear; so a diagonal L is the identity or it is nothing.
    Our L(r) is diagonal AND varying, which is the same class of error as c r^2."""
    def diagonal_varying(q):
        L = np.tile(np.eye(3), (len(q), 1, 1))
        L[:, 1, 1] += 0.45 * q[:, 0]                     # the measured dL_yy/dx, diagonal only
        return L

    with pytest.raises(ValueError, match="diagonal.*yet varies in space"):
        maxwell.require_gradient_tensor_admissible(diagonal_varying, P, "L(r)")

    # the identity is admissible (no nonlinearity), and so is anything carrying off-diagonals
    maxwell.require_gradient_tensor_admissible(lambda q: np.tile(np.eye(3), (len(q), 1, 1)), P, "L")

    def with_off(q):
        L = diagonal_varying(q)
        L[:, 1, 0] += 0.45 * q[:, 1]
        return L
    off, dev = maxwell.require_gradient_tensor_admissible(with_off, P, "L")
    assert off > 0 and dev > 0
