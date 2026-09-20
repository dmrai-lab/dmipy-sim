"""Tier 0 of the scanner model's evaluation harness: what Maxwell gives for free (dmipy-sim#364).

Each check needs no oracle and no measurement, and each is the check that would have caught a defect this
repository actually shipped. Most of what is asserted here was found by an adversarial pass over the first
version, which was broken in four ways that all let a wrong physics result through; the tests are written as
those attacks so they cannot come back.
"""
import numpy as np
import pytest

from dmipy_sim.acquisition import maxwell, solid_harmonics as sh
from dmipy_sim.acquisition.scanners import SCANNERS, ScannerLimits

P = maxwell.probe_points(0.05)
ON_AXIS = np.array([[0.0, 0.0, 0.05], [0.0, 0.0, -0.03], [0.0, 0.0, 0.08]])


# ── the harmonic check ──────────────────────────────────────────────────────────────────────────────
def test_the_residual_separates_a_field_from_a_non_field():
    """A harmonic gives roundoff because its three second differences CANCEL; anything else gives an
    order-one number because they do not."""
    for f in (lambda q: sh.evaluate({"Z2": 1.0}, q), lambda q: sh.evaluate({"Z2X": 1.0}, q)):
        assert maxwell.harmonic_residual(f, P) < 1e-8
    for f in (lambda q: np.sum(q ** 2, -1), lambda q: q[:, 2] ** 2):
        assert maxwell.harmonic_residual(f, P) == pytest.approx(1.0, abs=1e-6)


def test_an_additive_offset_cannot_hide_a_violation():
    """The first version scaled its noise floor by ``max|f0|``, so a constant raised the floor above the
    violation and the SAME defect written in absolute tesla scored a clean zero. A real B0 field is mostly
    constant, so this was the losing form for the one law the check exists to catch."""
    for law in (lambda q: np.sum(q ** 2, -1),
                lambda q: 1e3 + np.sum(q ** 2, -1),
                lambda q: 3.0 * (1.0 + 1e-4 * np.sum(q ** 2, -1)),      # 1 ppm at 10 cm on a 3 T magnet
                lambda q: 42.577e6 * 3.0 * (1.0 + 1e-4 * np.sum(q ** 2, -1))):
        with pytest.raises(ValueError, match="is not a magnetic field"):
            maxwell.require_harmonic(law, P, "a c r^2 law")


def test_a_field_lost_in_its_own_offset_is_refused_rather_than_answered():
    """At an offset of 1e9 the variation is 1e-12 of the magnitude, so differencing it in double precision
    leaves only roundoff. Reporting "harmonic" there would be a guess wearing a number's clothes."""
    with pytest.raises(ValueError, match="no verdict is possible"):
        maxwell.harmonic_residual(lambda q: 1e9 + np.sum(q ** 2, -1), P)


def test_a_law_that_vanishes_on_the_probe_points_is_still_judged_correctly():
    """The mirror of the offset bug. Scaling by ``max|f0|`` makes the floor ZERO where the law vanishes, so
    roundoff was divided by roundoff and 8 of the 18 basis terms -- including the plain imaging gradients --
    were declared not magnetic fields when probed on the bore axis, which is the most natural place to
    probe."""
    refused = [n for n in sh.names_through(4)
               if maxwell.harmonic_residual(lambda q, m=n: sh.evaluate({m: 1.0}, q), ON_AXIS) > 1e-6]
    assert refused == [], f"{len(refused)} harmonic terms refused on the bore axis: {refused}"


def test_a_field_with_no_curvature_at_all_is_harmonic():
    """x y is harmonic and its pure second derivatives vanish IDENTICALLY, so an unguarded ratio divides
    roundoff by roundoff and reports a clean field as a violation."""
    assert maxwell.harmonic_residual(lambda q: q[:, 0] * q[:, 1], P) == 0.0
    assert maxwell.harmonic_residual(lambda q: q[:, 0] * 3.0 - 2.0, P) == 0.0


def test_the_verdict_does_not_depend_on_the_stencil_width():
    """Normalising by the field's VALUE makes the ratio fall as h^2, so any tolerance could be met by
    shrinking h and a non-field would pass by parameter choice."""
    r = [maxwell.harmonic_residual(lambda q: np.sum(q ** 2, -1), P, h=h) for h in (1e-3, 1e-4, 1e-5, 1e-6)]
    assert max(r) - min(r) < 1e-6, f"the verdict moved with the stencil: {r}"


def test_a_field_that_is_not_finite_is_refused_and_not_silently_passed():
    """A NaN compares False against every tolerance, so an unguarded check reports a law that blows up as a
    clean one. Both the law and the points are checked."""
    for law in (lambda q: np.sum(q ** 2, -1) + np.nan, lambda q: np.sum(q ** 2, -1) + np.inf):
        with pytest.raises(ValueError, match="not finite"):
            maxwell.require_harmonic(law, P, "a law that blows up")


def test_a_field_of_the_wrong_shape_is_refused_and_not_scored():
    """A caller handing back a VECTOR gets its three components summed against the three stencil axes and
    scores a clean zero -- a wrong answer that looks like a pass."""
    with pytest.raises(ValueError, match=r"\(n,\) values"):
        maxwell.harmonic_residual(lambda q: q, P)
    with pytest.raises(ValueError, match="non-empty"):
        maxwell.harmonic_residual(lambda q: np.sum(q ** 2, -1), np.zeros((0, 3)))


def test_every_term_of_the_standard_basis_passes_the_check_a_law_does():
    """A term and a law judged by different rules is two rules, and only one of them was checked."""
    assert max(sh.check_harmonic(n) for n in sh.TERMS) < 1e-8


def test_every_catalogued_field_law_is_a_field():
    checked = 0
    for name in SCANNERS:
        s = ScannerLimits.of(name)
        if not s.harmonic_law():
            continue
        R = 0.5 * (s.b0_validity_radius or 0.1)
        assert maxwell.require_harmonic(lambda q: sh.evaluate(s.harmonic_law(), q),
                                        maxwell.probe_points(R), f"{name} law") < 1e-8
        checked += 1
    assert checked, "no catalogued scanner declares a field law, so this asserted nothing"


# ── the transmit axis ───────────────────────────────────────────────────────────────────────────────
def test_a_machine_whose_transmit_axis_lies_along_its_field_is_refused():
    with pytest.raises(ValueError, match="not perpendicular"):
        maxwell.require_transverse((0, 0, 1), (0, 0.001, 1), "a confused machine")
    assert maxwell.require_transverse((0, 0, 1), (1, 0, 0), "a sane one") < 1e-12


def test_an_axis_that_is_not_a_direction_is_refused():
    """A zero or non-finite axis makes every angle a NaN, and a NaN passes every tolerance."""
    for b1 in ((0, 0, 0), (np.nan, 0, 0), (np.inf, 0, 0)):
        with pytest.raises(ValueError, match="not a direction"):
            maxwell.require_transverse((0, 0, 1), b1, "a machine with no transmit axis")
    with pytest.raises(ValueError, match="3-vectors"):
        maxwell.require_transverse((0, 1), (1, 0), "a two-dimensional machine")


# ── the gradient tensor ─────────────────────────────────────────────────────────────────────────────
def _diagonal_varying(q):
    """The #350 defect: the measured dL_yy/dx, carried diagonally."""
    L = np.tile(np.eye(3), (len(q), 1, 1))
    L[:, 1, 1] += 0.45 * q[:, 0]
    return L


def test_a_diagonal_gradient_tensor_that_varies_is_refused():
    """The theorem, which needs no data. A diagonal L makes each coil's B_z depend on its own axis alone;
    Laplace forces that dependence to be linear; so a diagonal L is the identity or it is nothing."""
    with pytest.raises(ValueError, match="not a magnetic field"):
        maxwell.require_gradient_tensor_admissible(_diagonal_varying, P, "L(r)")


def test_the_defect_cannot_escape_by_adding_a_negligible_off_diagonal():
    """The first version tested the tensor's SHAPE -- diagonal-and-varying -- so 1e-8 of off-diagonal, which
    is physically nothing, took the identical defect straight through. The condition is now what actually
    has to hold, so the size of the off-diagonal is not a way out."""
    def escape(q):
        L = _diagonal_varying(q)
        L[:, 0, 1] += 1e-8
        return L
    with pytest.raises(ValueError, match="not a magnetic field"):
        maxwell.require_gradient_tensor_admissible(escape, P, "L(r)")


def test_columns_that_are_not_gradients_of_a_harmonic_are_refused():
    """Column j of L is grad Phi_j for a HARMONIC Phi_j, which is two conditions: zero curl (it is a
    gradient of something) and zero divergence (that something is harmonic). A shape test checks neither.

    The second tensor here was asserted to be ADMISSIBLE by the first version of this test. It is not: its
    x column has divergence 0.45 and its y column has curl 0.45, so that column is not the gradient of any
    potential at all."""
    def not_a_gradient(q):
        L = np.tile(np.eye(3), (len(q), 1, 1))
        L[:, 0, 1] += 0.3 * q[:, 0]
        L[:, 2, 2] += 0.7 * q[:, 1] ** 2
        return L

    def the_old_admissible_example(q):
        L = _diagonal_varying(q)
        L[:, 1, 0] += 0.45 * q[:, 1]
        return L

    for t in (not_a_gradient, the_old_admissible_example):
        with pytest.raises(ValueError, match="not a magnetic field"):
            maxwell.require_gradient_tensor_admissible(t, P, "L")


def test_the_defect_is_caught_off_the_axis_that_hides_it():
    """A shape test evaluated where the varying term vanishes saw a clean identity. On a sagittal slice the
    #350 defect has |L_ii - 1| = 0 at every point, so it was accepted."""
    sagittal = np.array([[0.0, 0.03, 0.02], [0.0, -0.05, 0.04], [0.0, 0.01, -0.06]])
    with pytest.raises(ValueError, match="not a magnetic field"):
        maxwell.require_gradient_tensor_admissible(_diagonal_varying, sagittal, "L(r)")


def test_a_uniform_calibration_error_is_a_field_and_is_admitted():
    """B_z = 1.02 G x is a perfectly good harmonic field, so a per-axis gradient calibration error must be
    expressible. The shape test refused it as impossible -- a false positive on ordinary hardware."""
    maxwell.require_gradient_tensor_admissible(
        lambda q: np.tile(np.diag([1.0, 0.98, 1.03]), (len(q), 1, 1)), P, "a miscalibrated L")
    maxwell.require_gradient_tensor_admissible(lambda q: np.tile(np.eye(3), (len(q), 1, 1)), P, "I")


def test_the_shipped_swoop_tensor_is_admissible():
    """It is built from harmonic potentials, so it cannot be otherwise -- and this is the check that refused
    what it replaced."""
    s = ScannerLimits.of("swoop")
    div, curl = maxwell.require_gradient_tensor_admissible(
        s.gradient_tensor, maxwell.probe_points(0.5 * s.b0_validity_radius), "the Swoop L(r)")
    assert div < 1e-9 and curl < 1e-9
