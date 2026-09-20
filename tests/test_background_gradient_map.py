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
from dmipy_sim.phantom.bore import background_gradient_map, delivered_b

DIRS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1]]


def _swoop_protocol():
    """The Swoop's own diffusion measurement: b = 945 s/mm^2 at delta 35 / Delta 42 ms."""
    return sequences.pgse(DIRS, 35e-3, 42e-3, bvalues=[0.945e9] * len(DIRS), n_t=400)


def _grid(n=5, fov=0.088):
    """Small enough that the CORNERS stay inside the law's 8 cm anchor: a cube of side `fov` reaches
    sqrt(3)/2 fov, so a 12 cm cube is 8.3 cm out at its corners and the law refuses it -- correctly."""
    return Grid(shape=(n, n, n), voxel_size_m=(fov / n,) * 3,
                origin_m=(-0.5 * (n - 1) * fov / n,) * 3, isocenter_m=(0.0, 0.0, 0.0))


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
    b = delivered_b(sw, grid, seq)
    assert b.shape == (grid.shape[0] * grid.shape[1] * grid.shape[2], seq.n_meas)
    assert np.all(np.isfinite(b)) and np.all(b > 0)
    # the spread across the volume is the effect, and it is large
    spread = np.ptp(b, axis=0) / seq.b()
    assert spread.max() > 0.1, f"a magnet that varies by {spread.max():.1%} across the FOV is not an effect"
    assert delivered_b(ScannerLimits.of("prisma"), grid, seq) is None
