"""A grid built from an image's affine places every voxel where the affine does (dmipy-sim#562).

The affine of a NIfTI image maps a voxel index to the scanner coordinate of its centre, and the sign of each
axis is part of that map: the usual radiological image runs its first index along -x, and a left-handed
oblique image runs its third index against the rotation's. A grid that keeps only the voxel SIZE mirrors
those axes, and every field a machine imposes is then read on the wrong side of the head.
"""
import numpy as np
import pytest

from dmipy_sim import Prescription
from dmipy_sim.acquisition.scanners import ScannerLimits
from dmipy_sim.phantom import Grid
from dmipy_sim.phantom.bore import _bore, b0_offset_map, background_gradient_map
from dmipy_sim.spec.seeding import fill_per_voxel


def _corners(shape):
    n = np.asarray(shape) - 1
    return np.array([[i, j, k] for i in (0, n[0]) for j in (0, n[1]) for k in (0, n[2])], dtype=np.int64)


def _truth_m(A, ijk):
    """The NIfTI map, index -> scanner coordinate, in metres."""
    return (np.asarray(ijk, np.float64) @ A[:3, :3].T + A[:3, 3]) * 1e-3


def _random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else q @ np.diag([1.0, 1.0, -1.0])


def test_a_nifti_style_las_affine_places_voxel_9_0_0_at_minus_x():
    """The issue's reproduction: diag(-2, 2, 2) with an offset, i runs along -x."""
    A = np.diag([-2.0, 2.0, 2.0, 1.0]); A[:3, 3] = (90.0, -126.0, -72.0)
    g = Grid.from_affine(A, (91, 109, 91))
    ijk = np.array([[0, 0, 0], [0, 0, 9], [9, 0, 0]])
    np.testing.assert_allclose(g.positions_m(ijk), _truth_m(A, ijk), rtol=0, atol=1e-15)
    assert g.axes == "LAS" and g.step_m == pytest.approx((-2e-3, 2e-3, 2e-3))


@pytest.mark.parametrize("handed", ["right", "left"])
def test_every_corner_of_a_random_oblique_affine_sits_where_the_affine_puts_it(handed):
    """Two hundred random oblique affines of each handedness: rotation, voxel size, offset, shape and
    isocentre all drawn; ``into_bore(positions_m(ijk)) = A @ [ijk, 1] - isocentre`` at every corner."""
    rng = np.random.default_rng(562 if handed == "right" else 563)
    for _ in range(200):
        R = _random_rotation(rng)
        vs = rng.uniform(0.5, 3.0, 3)
        M = R @ np.diag(vs)
        if handed == "left":
            M = M @ np.diag(rng.permutation([-1.0, 1.0, 1.0]))          # any one axis reversed
        A = np.eye(4); A[:3, :3] = M; A[:3, 3] = rng.uniform(-150.0, 150.0, 3)
        shape = tuple(int(v) for v in rng.integers(1, 120, 3))
        iso = rng.uniform(-0.05, 0.05, 3)
        g, Rg = Grid.from_oblique_affine(A, shape, isocenter_m=iso)
        assert np.linalg.det(Rg) > 0 and g.axes == ("RAS" if handed == "right" else "RAI")
        _, into_bore = _bore(g, None)
        ijk = _corners(shape)
        np.testing.assert_allclose(into_bore(g.positions_m(ijk)), _truth_m(A, ijk) - iso, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(g.bin(g.positions_m(ijk))[0], ijk)


@pytest.mark.parametrize("signs", [(1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, -1), (-1, -1, -1)])
def test_every_corner_of_a_random_axis_aligned_affine_sits_where_the_affine_puts_it(signs):
    """RAS, LAS, LPS, RPI, LPI: the axis-aligned constructor keeps each axis' sign, and the oblique one agrees
    with it on the same affine."""
    rng = np.random.default_rng([s + 1 for s in signs])
    for _ in range(100):
        A = np.diag(np.r_[np.asarray(signs) * rng.uniform(0.5, 3.0, 3), 1.0]); A[:3, 3] = rng.uniform(-150, 150, 3)
        shape = tuple(int(v) for v in rng.integers(1, 120, 3))
        iso = rng.uniform(-0.05, 0.05, 3)
        ijk = _corners(shape)
        g = Grid.from_affine(A, shape, isocenter_m=iso)
        np.testing.assert_allclose(g.positions_m(ijk), _truth_m(A, ijk), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(g.bin(g.positions_m(ijk))[0], ijk)
        go, _R = Grid.from_oblique_affine(A, shape, isocenter_m=iso)
        np.testing.assert_allclose(_bore(go, None)[1](go.positions_m(ijk)), _truth_m(A, ijk) - iso, rtol=0, atol=1e-12)


def _head_affine(left_handed_oblique):
    """A 2 mm head grid off-centre in x, inside the Swoop's 8 cm anchor: radiological (LAS) axis-aligned, or
    left-handed and tilted by 15 degrees about z."""
    if not left_handed_oblique:
        A = np.diag([-2.0, 2.0, 2.0, 1.0]); A[:3, 3] = (30.0, -20.0, -16.0)
        return A
    th = np.radians(15.0)
    Rz = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    A = np.eye(4); A[:3, :3] = Rz @ np.diag([2.0, 2.0, -2.0]); A[:3, 3] = (-10.0, -20.0, 16.0)
    return A


@pytest.mark.parametrize("left_handed_oblique", [False, True], ids=["LAS", "left-handed-oblique"])
def test_a_left_handed_head_grid_reads_the_swoop_field_at_the_affine_s_points(left_handed_oblique):
    """The Swoop's field is odd along R/L (an l=3, m=1 term), so a grid mirrored along any axis reads it on the
    wrong side. The bore maps of a left-handed head grid equal the law evaluated at ``A @ ijk`` itself, the
    offset as a scalar and the gradient brought back into the grid's frame."""
    swoop = ScannerLimits.of("swoop")
    A = _head_affine(left_handed_oblique)
    shape = (16, 20, 16)
    if left_handed_oblique:
        g, R = Grid.from_oblique_affine(A, shape, isocenter_m=(0.0, 0.0, 0.0))
    else:
        g, R = Grid.from_affine(A, shape, isocenter_m=(0.0, 0.0, 0.0)), np.eye(3)
    ijk = g.every_voxel
    at = _truth_m(A, ijk)
    np.testing.assert_allclose(b0_offset_map(swoop, g)(g.positions_m(ijk)), swoop.b0_offset(at), rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(background_gradient_map(swoop, g)(g.positions_m(ijk)), swoop.b0_gradient(at) @ R,
                               rtol=1e-12, atol=1e-15)
    mirrored = at.copy(); mirrored[:, 0] = 2 * at[:, 0].mean() - at[:, 0]
    assert np.abs(swoop.b0_offset(mirrored) - swoop.b0_offset(at)).max() > 1e-6 * swoop.field_T, \
        "the field must be asymmetric over this grid, or a mirror would go unseen"


def test_a_grid_from_a_radiological_prescription_is_centred_on_its_isocenter():
    p = Prescription(isocenter_m=(0.01, -0.02, 0.0), voxel_size_m=(2e-3, 2e-3, 3e-3), matrix=(8, 6, 4), axes="LPS")
    g = Grid.from_prescription(p)
    np.testing.assert_allclose(g.positions_m(_corners(g.shape)).mean(axis=0), p.isocenter_m, rtol=0, atol=1e-15)
    assert g.centre_m == pytest.approx(p.isocenter_m, abs=1e-15)
    assert g.step_m == pytest.approx((-2e-3, -2e-3, 3e-3))


def test_a_grid_refuses_permuted_axes():
    with pytest.raises(ValueError, match="permuted"):
        Grid(shape=(2, 2, 2), voxel_size_m=(1e-3,) * 3, axes="ARS")


def test_seeds_land_in_their_own_voxel_and_a_rebin_keeps_the_field_of_view_on_a_reversed_grid():
    g = Grid(shape=(3, 4, 2), voxel_size_m=(1e-3, 2e-3, 1.5e-3), origin_m=(5e-3, 0.0, 1e-3), axes="LPI")
    P, v, *_ = fill_per_voxel(lambda p: np.ones(len(p), bool), g, np.full(g.n_voxels, 50), trials_max=1000, seed=1)
    ijk, inside = g.bin(P)
    assert inside.all() and np.array_equal(np.ravel_multi_index(ijk.T, g.shape), v)
    fine = g.with_voxel_size((0.5e-3, 1e-3, 0.75e-3))
    assert fine.corner_m == pytest.approx(g.corner_m) and fine.axes == g.axes
    np.testing.assert_array_equal(fine.bin(g.positions_m(g.every_voxel))[1], True)


@pytest.mark.parametrize("kind", ["las", "oblique_left"])
def test_centred_at_puts_the_grid_centre_at_the_offset_from_isocentre_in_the_scanner_frame(kind):
    """``Grid.centred_at(offset)`` moves only the isocentre: afterwards the grid's centre (midway between the first
    and last voxel centres, whichever way each axis runs) sits ``offset`` from it in the SCANNER's frame, and every
    voxel's bore displacement is its NIfTI position less the NIfTI centre plus ``offset``."""
    shape = (11, 8, 6)
    if kind == "las":
        A = np.diag([-2.0, 2.0, 2.5, 1.0]); A[:3, 3] = (90.0, -126.0, -72.0)
        g = Grid.from_affine(A, shape)
    else:
        R = _random_rotation(np.random.default_rng(7))
        A = np.eye(4); A[:3, :3] = R @ np.diag([2.0, 1.5, -2.5]); A[:3, 3] = (-90.0, -126.0, 72.0)
        g, _ = Grid.from_oblique_affine(A, shape)
    offset = np.array([0.05, -0.03, 0.04])
    placed = g.centred_at(offset)
    _, into = _bore(placed, None)
    np.testing.assert_allclose(into(np.asarray(placed.centre_m))[0], offset, atol=1e-12)
    ijk = _corners(shape)
    truth = _truth_m(A, ijk)
    centre_truth = 0.5 * (_truth_m(A, np.zeros((1, 3), int))[0] + _truth_m(A, (np.array(shape) - 1)[None])[0])
    np.testing.assert_allclose(into(placed.positions_m(ijk)), truth - centre_truth + offset, atol=1e-12)
