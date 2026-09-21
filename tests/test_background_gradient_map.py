"""The magnet's own gradient, per voxel, and the ADC error it produces (dmipy-sim#322 PR 6, #285 item 3).

A permanent magnet's field is not uniform, so its spatial gradient is on during every pulse and every dead
time -- a magnet does not switch off. That gradient encodes diffusion alongside the pulsed one, and the
cross term between them is an ADC error the Swoop paper measures at up to 16.1 % and corrects for. Here it
is reproduced from the catalogued field shape rather than asserted.
"""
import numpy as np
import pytest

from dmipy_sim import sequences
from dmipy_sim.acquisition.scanners import ScannerLimits
from dmipy_sim.phantom import Grid
from dmipy_sim.phantom.bore import background_gradient_map, delivered_gradient

DIRS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1]]


def _swoop_protocol():
    """The Swoop's own diffusion measurement: b = 945 s/mm^2 at delta 35 / Delta 42 ms."""
    return sequences.pgse(DIRS, 35e-3, 42e-3, bvalues=[0.945e9] * len(DIRS), n_t=400)


def _grid(n=5, fov=0.088):
    """Small enough that the CORNERS stay inside the law's 8 cm anchor: a cube of side `fov` reaches
    sqrt(3)/2 fov, so a 12 cm cube is 8.3 cm out at its corners and the law refuses it -- correctly."""
    return Grid(shape=(n, n, n), voxel_size_m=(fov / n,) * 3,
                origin_m=(-0.5 * (n - 1) * fov / n,) * 3, isocenter_m=(0.0, 0.0, 0.0))


def _delivered_b(scanner, grid, seq):
    """The b every voxel receives from the magnet's own gradient alone, by the reference route."""
    return np.stack([seq.with_gradient(G).b()
                     for G in delivered_gradient(scanner, grid, seq, nonlinearity=False, concomitant=False)])


def test_the_map_is_the_law_s_derivative_at_every_voxel():
    sw, grid = ScannerLimits.of("swoop"), _grid()
    at = grid.offset_m(grid.every_voxel).reshape(-1, 3)
    np.testing.assert_allclose(background_gradient_map(sw, grid)(at), sw.b0_gradient(at), rtol=1e-12)
    assert background_gradient_map(ScannerLimits.of("prisma"), grid) is None


def test_a_gradient_is_rotated_back_into_the_grid_s_frame_where_an_offset_is_not():
    """The one thing that is easy to get wrong. A position goes FORWARD into the bore so the law can be
    evaluated there; the gradient that comes back is a vector in the bore's frame and has to come BACK into
    the grid's, because that is the frame the sequence's G is written in. An offset is a scalar and needs
    only the first half -- so the asymmetry between the two map functions is real.

    A 90-degree roll about z sends the magnet's R/L asymmetry onto the grid's y, and skipping the return
    rotation leaves it on x: the two differ by a whole axis, not by a small amount."""
    sw, grid = ScannerLimits.of("swoop"), _grid()
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])    # grid axes -> scanner axes
    at = np.array([[0.02, 0.0, 0.03], [0.03, 0.0, 0.0]])
    turned = background_gradient_map(sw, grid, to_scanner=R)(at)
    plain = background_gradient_map(sw, grid)(at)
    assert np.linalg.norm(plain[0]) > 1e-6      # the probe must have a gradient to rotate at all
    assert np.linalg.norm(turned[0] - plain[0]) > 0.3 * np.linalg.norm(plain[0])
    # The length legitimately CHANGES: the roll moves the voxel to a different place in the bore, where
    # the law is steeper or flatter. That is the forward half doing its job, not an error.
    assert np.linalg.norm(turned[1]) != pytest.approx(np.linalg.norm(plain[1]), rel=1e-3)


def test_the_delivered_b_reproduces_the_paper_s_adc_error_at_eight_centimetres():
    """The headline number. The Swoop paper reports ADC errors up to 16.1 % from its background gradient and
    corrects them with the b = 0 image. Fitting an ADC against the PRESCRIBED b absorbs the whole
    discrepancy, so the delivered-to-prescribed ratio minus one IS that error -- and it comes out at
    +15.7 % on the high side of the bore at 8 cm, from the catalogued field shape alone."""
    sw = ScannerLimits.of("swoop")
    seq = _swoop_protocol()
    prescribed = seq.b()
    rng = np.random.default_rng(0)
    u = rng.normal(size=(400, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    worst = 0.0
    for pt in 0.0755 * u:          # the shell: a harmonic law need not be steepest on an axis
        g = np.atleast_2d(sw.b0_gradient(np.array([pt])))[0]
        worst = max(worst, float(np.abs(seq.with_background_gradient(g).b() / prescribed - 1.0).max()))
    assert 0.11 < worst < 0.17, f"the ADC error came out {worst:.1%}, not the paper's ~16 %"
    # the background gradient at that radius is the cited fraction of the pulsed one
    steep = max(float(np.linalg.norm(np.atleast_2d(sw.b0_gradient(np.array([pt])))[0]))
                for pt in 0.0755 * u)
    frac = steep / np.abs(seq.G).max()
    assert 0.04 < frac < 0.08, f"{frac:.1%} of the diffusion gradient, not the cited 'up to 7 %'"


def test_the_error_is_signed_per_direction_so_a_mean_hides_it():
    """The cross term is LINEAR in the background gradient, so it flips sign with the diffusion direction.
    On a symmetric direction set the mean error therefore largely cancels while every single measurement
    keeps its own -- which is why this has to be reported per direction. A test that averaged first would
    pass on a magnet that ruins every voxel."""
    sw, seq = ScannerLimits.of("swoop"), _swoop_protocol()
    g = np.atleast_2d(sw.b0_gradient(np.array([[0.0, 0.0755, 0.0]])))[0]
    err = seq.with_background_gradient(g).b() / seq.b() - 1.0
    assert err[0] * err[1] < 0                             # +x and -x, opposite in sign
    assert max(abs(err[0]), abs(err[1])) > 0.08            # and large
    assert abs(err[0] + err[1]) < 0.3 * abs(err[0])        # nearly cancelling in a mean


def test_a_voxel_grid_gets_one_b_per_voxel_per_measurement():
    sw, grid = ScannerLimits.of("swoop"), _grid(n=3, fov=0.10)
    seq = _swoop_protocol()
    b = _delivered_b(sw, grid, seq)
    assert b.shape == (grid.shape[0] * grid.shape[1] * grid.shape[2], seq.n_meas)
    assert np.all(np.isfinite(b)) and np.all(b > 0)
    # the spread across the volume is the effect, and it is large
    spread = np.ptp(b, axis=0) / seq.b()
    assert spread.max() > 0.1, f"a magnet that varies by {spread.max():.1%} across the FOV is not an effect"
    assert background_gradient_map(ScannerLimits.of("prisma"), grid) is None


# ── what the background does on its own (dmipy-sim#349 item 6) ───────────────────────────────────────
def test_the_background_s_own_b_survives_the_spin_echo_that_refocuses_its_moment():
    """The claim this corrects. A symmetric spin echo refocuses the background's zeroth MOMENT, and it is
    tempting to stop there and say only the cross term survives. Stejskal and Tanner 1965 say otherwise in
    one line -- with the pulsed gradient off, "only the term in g0^2 remains" -- and that term is
    ``gamma^2 g0^2 (2/3) tau^3``. So even the b = 0 image of this magnet is diffusion-weighted."""
    from dmipy_sim.constants import GAMMA
    g0 = 1.4e-3
    for TE in (0.084, 0.150):
        seq = sequences.pgse([[1, 0, 0]], 5e-3, 20e-3, gradient_strengths=[0.0], TE=TE, n_t=1200)
        got = float(seq.with_background_gradient([g0, 0, 0]).b()[0])
        closed = GAMMA ** 2 * g0 ** 2 * (2 / 3) * (TE / 2) ** 3
        assert got == pytest.approx(closed, rel=5e-3)        # the residual is the discrete time grid
        assert got > 0, "the background's own b vanished, which the 1965 paper says it does not"
    # and it is not negligible: about 7 s/mm^2 at the Swoop's echo time
    at84 = float(sequences.pgse([[1, 0, 0]], 5e-3, 20e-3, gradient_strengths=[0.0], TE=0.084, n_t=1200)
                 .with_background_gradient([g0, 0, 0]).b()[0])
    assert 6e6 < at84 < 8e6


def test_the_background_s_own_b_grows_as_the_cube_of_the_echo_time():
    from dmipy_sim.constants import GAMMA
    g0 = 1.4e-3
    b = {TE: float(sequences.pgse([[1, 0, 0]], 5e-3, 20e-3, gradient_strengths=[0.0], TE=TE, n_t=1500)
                   .with_background_gradient([g0, 0, 0]).b()[0]) for TE in (0.084, 0.168)}
    assert b[0.168] / b[0.084] == pytest.approx(8.0, rel=0.02)     # doubling TE is eight times the b


def test_a_train_is_not_one_long_spin_echo_and_the_difference_is_orders_of_magnitude():
    """A refocusing train re-refocuses the background at every pulse, so its self-term accrues per ECHO --
    `gamma^2 g0^2 esp^3 / 12` each -- rather than over the whole readout. Computing it as a single spin echo
    of the same duration overestimates by ``(T / esp)^2``, which for the Swoop's seventy-echo train is about
    five thousand. This is the trap the correction exists to mark."""
    from dmipy_sim.constants import GAMMA
    g0, esp = 1.4e-3, 10e-3
    for n in (4, 16):
        seq = sequences.cpmg(n, esp, gradient_strengths=[0.0], n_t_per_echo=60)
        got = float(seq.with_background_gradient([g0, 0, 0]).b()[0])
        per_echo = GAMMA ** 2 * g0 ** 2 * (esp ** 3 / 12) * n
        assert got == pytest.approx(per_echo, rel=0.02)
        one_long = GAMMA ** 2 * g0 ** 2 * (2 / 3) * (n * esp / 2) ** 3
        assert one_long / got == pytest.approx(n ** 2, rel=0.05)


def test_the_cross_term_is_signed_and_the_self_term_is_not():
    """The two behave differently and only one of them averages away. The cross term is linear in the
    background so it flips with the diffusion direction; the self term is quadratic, unsigned, and present
    in every measurement including the unweighted one."""
    sw, seq = ScannerLimits.of("swoop"), _swoop_protocol()
    g = np.atleast_2d(sw.b0_gradient(np.array([[0.0, 0.0755, 0.0]])))[0]
    err = seq.with_background_gradient(g).b() / seq.b() - 1.0
    assert err.min() < 0 < err.max()                              # signed, so it flips with direction
    unweighted = sequences.pgse([[1, 0, 0]], 5e-3, 20e-3, gradient_strengths=[0.0], TE=0.084, n_t=1200)
    assert float(unweighted.with_background_gradient(g).b()[0]) > 0     # unsigned, so it survives at b = 0
