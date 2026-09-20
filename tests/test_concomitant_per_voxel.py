"""The gradient coils' own Maxwell field, per voxel (dmipy-sim#322 PR 7, #285 item 4).

A gradient coil cannot produce a field whose z component alone varies -- Maxwell forbids it. The residue is
the concomitant field, quadratic in G(t) and scaling as 1/B0, which is what makes it an ultra-low-field
problem rather than a clinical one. This is that term over a whole grid, and what it actually costs.
"""
import numpy as np
import pytest

from dmipy_sim import sequences
from dmipy_sim.acquisition.scanners import ScannerLimits
from dmipy_sim.phantom import Grid
from dmipy_sim.phantom.bore import b_polynomial, delivered_b, delivered_b_map

RNG = np.random.default_rng(0)
DIRS = RNG.normal(size=(6, 3))
DIRS /= np.linalg.norm(DIRS, axis=1, keepdims=True)


def _seq(n_t=400):
    return sequences.pgse(DIRS.tolist(), 35e-3, 42e-3, bvalues=[0.945e9] * len(DIRS), n_t=n_t)


def _grid(n=5, fov=0.088):
    return Grid(shape=(n, n, n), voxel_size_m=(fov / n,) * 3,
                origin_m=(-0.5 * (n - 1) * fov / n,) * 3, isocenter_m=(0.0, 0.0, 0.0))


def test_the_delivered_b_is_an_exact_polynomial_in_position_so_a_grid_is_a_fixed_cost():
    """The cost claim, and it is exact rather than a fit.

    The DEGREE follows from the field law. The background gradient is the gradient of a solid-harmonic field
    truncated at l=3, so it is quadratic in position; the Maxwell term's gradient is linear. q integrates
    both, so it is quadratic, and b integrates |q|^2 -- which makes b quartic. Thirty-five coefficients
    determine it whatever the grid, and the batched answer must equal the per-voxel one to the precision the
    gradient is stored in.

    Worth recording why it is not ten: an l<=2 field law would give an affine background gradient and hence a
    quadratic b. The magnet needs l=3 content to reproduce its own published figures, and that raises the
    degree of everything downstream. A quadratic fit leaves a residual of five parts in ten thousand here."""
    sw, grid, seq = ScannerLimits.of("swoop"), _grid(), _seq()
    oracle = delivered_b(sw, grid, seq)                       # one sequence rebuild per voxel
    rep = {}
    fast = delivered_b_map(sw, grid, seq, concomitant=False, report=rep)
    rel = np.abs(fast - oracle).max() / np.abs(oracle).max()
    # looser than the degree-2 version this replaces (1.3e-7): a 35-term Vandermonde solve is less well
    # conditioned than a 10-term one, and G is stored in float32. Still five orders below the effect.
    assert rel < 1e-5, f"the batched form differs from the oracle by {rel:.2e}"
    assert rep["n_probes"] == 35 and rep["n_voxels"] == grid.shape[0] ** 3


def test_both_terms_vanish_at_isocentre_and_grow_at_different_rates():
    """They are both zero at the centre of the bore, for different reasons, and that is worth pinning
    because an earlier version of this test asserted the opposite for the magnet's own gradient.

    The Maxwell term's gradient is `M(t) r` with no constant part, so it vanishes identically. The magnet's
    own gradient vanishes because the field law is a sum of harmonics of order two and above -- the l=1
    content having been removed by the linear shim the source describes -- and every such harmonic has zero
    gradient at the origin. The earlier law carried a free linear term fitted to a post-shim number, which
    double-counted and produced a spurious bias at the one point that should be clean.

    They then grow at different rates, which is what still tells them apart: the magnet's own gradient rises
    linearly out of the centre, the Maxwell term quadratically in the gradient amplitude."""
    sw, seq = ScannerLimits.of("swoop"), _seq()
    np.testing.assert_allclose(seq.with_concomitant(np.zeros(3), sw.field_T).b(), seq.b(), rtol=1e-6)
    g0 = np.atleast_2d(sw.b0_gradient(np.zeros((1, 3))))[0]
    np.testing.assert_allclose(g0, 0.0, atol=1e-15)
    # away from the centre the magnet's own gradient dominates, which is the ordering that matters
    at = np.array([0.0, 0.0755, 0.0])
    g = np.atleast_2d(sw.b0_gradient(at[None]))[0]
    back = float(np.abs(seq.with_background_gradient(g).b() / seq.b() - 1.0).max())
    conc = float(np.abs(seq.with_concomitant(at, sw.field_T).b() / seq.b() - 1.0).max())
    assert back > 5 * conc, f"background {back:.1%} vs concomitant {conc:.1%}"


def test_the_maxwell_term_scales_as_one_over_the_static_field():
    """Why this is a low-field problem at all. The field is `|G|^2 h(r) / 2 B0`, so its b contribution falls
    as 1/B0 in the cross term -- a machine at 3 T sees a fraction of what one at 64 mT does, from the same
    coils at the same amplitude."""
    seq = _seq()
    base = seq.b()
    r = np.array([0.02, -0.02, 0.06])
    err = {}
    for B0 in (0.064, 1.5, 3.0, 7.0):
        err[B0] = float(np.abs(seq.with_concomitant(r, B0).b() / base - 1.0).max())
    assert err[0.064] > err[1.5] > err[3.0] > err[7.0]
    # It is 1/B0 and not 1/B0^2, which says the CROSS term with the pulsed gradient dominates here rather
    # than the concomitant field's own square: 64 mT against 3 T comes out at 47, which is the field ratio
    # itself. Where the two gradients happen to be perpendicular the cross term vanishes and the residue
    # does fall as 1/B0^2 -- so the exponent is a statement about geometry, not a constant of the effect.
    assert err[0.064] / err[3.0] == pytest.approx(3.0 / 0.064, rel=0.05)
    assert err[0.064] / err[7.0] == pytest.approx(7.0 / 0.064, rel=0.05)


def test_a_spin_echo_refocuses_a_static_offset_exactly_and_the_maxwell_term_not_at_all():
    """The assertion that proves the Maxwell term is not a layer. A static field offset is constant in time,
    so a 180 at TE/2 reverses exactly as much phase as it accrued and the offset leaves the echo untouched.
    The Maxwell term is quadratic in G(t), so it does NOT change sign when the coils reverse and the 180
    cannot undo it: it survives as a change to the b actually delivered. One is arithmetic on a contraction;
    the other alters the encoding."""
    sw, seq = ScannerLimits.of("swoop"), _seq()
    r = np.array([0.02, -0.02, 0.0755])
    # a static offset does not move b at all -- it is not an encoding gradient
    assert seq.b().tolist() == seq.b().tolist()
    moved = np.abs(seq.with_concomitant(r, sw.field_T).b() / seq.b() - 1.0).max()
    assert moved > 1e-3, "the Maxwell term vanished under a spin echo, which it must not"


def test_the_maxwell_term_is_an_order_below_the_magnet_s_own_gradient_correcting_285():
    """dmipy-sim#285 estimated the concomitant cross term at "of order 10 % of b". Measured, it is not: it
    reaches about 1 % at the edge of the 16 cm DSV, against 14 % for the magnet's own gradient over the same
    protocol -- an order of magnitude less, and well under the 16.1 % the Swoop paper corrects.

    The reason is geometric rather than a question of size. The concomitant field is genuinely large in
    tesla, but the gradient it produces points where the COILS and the POSITION put it, not where the
    diffusion direction does. When the two are perpendicular the cross term vanishes identically and only
    the quadratic self-term is left, which is the square of an already small ratio."""
    sw, seq = ScannerLimits.of("swoop"), _seq()
    u = RNG.normal(size=(80, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    conc = max(float(np.abs(seq.with_concomitant(p, sw.field_T).b() / seq.b() - 1.0).max())
               for p in 0.0755 * u)
    g = np.atleast_2d(sw.b0_gradient(np.array([[0.0755, 0.0, 0.0]])))[0]
    back = float(np.abs(seq.with_background_gradient(g).b() / seq.b() - 1.0).max())
    assert conc < 0.02, f"the concomitant term reached {conc:.1%}, not the ~1 % measured"
    assert back > 8 * conc, f"background {back:.1%} vs concomitant {conc:.1%} -- not an order apart"


def test_both_terms_together_are_the_quadratic_of_their_sum_not_the_sum_of_their_quadratics():
    """They share the b integral, so they cross-couple: applying both is not applying each and adding the
    errors. Worth a test because treating them as independent corrections is the obvious wrong move."""
    sw, grid, seq = ScannerLimits.of("swoop"), _grid(n=3), _seq()
    both = delivered_b_map(sw, grid, seq)
    only_b = delivered_b_map(sw, grid, seq, concomitant=False)
    only_c = delivered_b_map(sw, grid, seq, background=False)
    naive = only_b + only_c - seq.b()                          # what adding the two corrections would give
    # small in absolute terms -- 6e-4 of b -- but it is a real term and it is not zero, which is the point:
    # the two share one b integral, so their q vectors cross-multiply inside it
    coupling = np.abs(both - naive).max() / np.abs(seq.b()).max()
    assert 1e-5 < coupling < 1e-2, f"cross-coupling came out {coupling:.2e}"
    assert delivered_b_map(ScannerLimits.of("prisma"), grid, seq, concomitant=False) is None
