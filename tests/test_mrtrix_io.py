"""Reading MRtrix images: any storage layout comes back in canonical axis order with the NIfTI-style affine, and
the pieces a brain phantom needs from one -- an oblique grid, a rotated FOD -- hold (dmipy-sim#193)."""
import numpy as np
import pytest

from dmipy_sim.io.mrtrix import read_mif
from dmipy_sim.phantom import Grid
from dmipy_sim.replay import so3
from dmipy_sim.replay.fod import FOD


def _write_mif(path, data, transform, vox, layout, datatype="Float32LE", scaling=None):
    """A .mif by hand, in the given storage layout (sign and rank per canonical axis)."""
    n = data.ndim
    sign = [1 if tok[0] == "+" else -1 for tok in layout]
    rank = [int(tok[1:]) for tok in layout]
    arr = data
    for ax in range(n):
        if sign[ax] < 0:
            arr = np.flip(arr, axis=ax)
    order = sorted(range(n), key=lambda ax: rank[ax], reverse=True)       # slowest first
    stored = np.ascontiguousarray(arr.transpose(order))
    head = ["mrtrix image", "dim: " + ",".join(str(v) for v in data.shape), "vox: " + ",".join(str(v) for v in vox),
            "layout: " + ",".join(layout), f"datatype: {datatype}"]
    head += ["transform: " + ",".join(f"{v}" for v in row) for row in transform[:3]]
    if scaling:
        head.append("scaling: " + ",".join(str(v) for v in scaling))
    body = "\n".join(head) + "\nfile: . "
    off = len(body.encode()) + 12
    body += f"{off:>6}\nEND\n"
    b = body.encode()
    b += b" " * (off - len(b))
    if datatype == "Bit":
        payload = np.packbits(stored.astype(bool).ravel(), bitorder="little").tobytes()
    else:
        payload = stored.astype({"Float32LE": "<f4", "UInt8": "u1"}[datatype]).tobytes()
    path.write_bytes(b + payload)


_R = so3.rotation_of((0.0082, 0.0371, 0.9993)) @ so3.rotation_of((0.0, 0.0, 1.0), roll=0.024)     # a few degrees oblique, exact
TRANSFORM = np.column_stack([_R, [-120.84, -99.16, -33.46]])


@pytest.mark.parametrize("layout", [("+0", "+1", "+2", "+3"), ("-1", "-2", "-3", "+0"), ("-0", "-1", "-2", "+3"), ("+2", "+0", "+1", "+3")])
def test_any_layout_reads_back_in_canonical_order(tmp_path, layout):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(5, 4, 3, 6)).astype(np.float32)
    p = tmp_path / "x.mif"
    _write_mif(p, data, TRANSFORM, (2.5, 2.5, 2.5, 1), layout)
    im = read_mif(p)
    np.testing.assert_array_equal(im.data, data)
    np.testing.assert_allclose(im.affine[:3, :3], TRANSFORM[:3, :3] * 2.5); np.testing.assert_allclose(im.affine[:3, 3], TRANSFORM[:3, 3])
    assert im.header["layout"] == ",".join(layout) and im.vox[0] == 2.5


def test_bit_masks_and_scaling(tmp_path):
    rng = np.random.default_rng(1)
    m = rng.random((7, 5, 3)) > 0.6
    _write_mif(tmp_path / "m.mif", m, TRANSFORM, (2.5, 2.5, 2.5), ("-0", "-1", "-2"), datatype="Bit")
    im = read_mif(tmp_path / "m.mif")
    assert im.data.dtype == bool and np.array_equal(im.data, m)
    x = rng.integers(0, 255, size=(4, 4, 2)).astype(np.uint8)
    _write_mif(tmp_path / "s.mif", x, TRANSFORM, (1, 1, 1), ("+0", "+1", "+2"), datatype="UInt8", scaling=(-1.0, 0.5))
    np.testing.assert_allclose(read_mif(tmp_path / "s.mif").data, x * 0.5 - 1.0)
    with pytest.raises(ValueError, match="not an MRtrix image"):
        (tmp_path / "n.mif").write_bytes(b"nope"); read_mif(tmp_path / "n.mif")


def test_an_oblique_affine_gives_the_image_grid_and_the_rotation():
    A = np.eye(4); A[:3, :3] = TRANSFORM[:3, :3] * 2.5; A[:3, 3] = TRANSFORM[:3, 3]
    with pytest.raises(ValueError, match="oblique"):
        Grid.from_affine(A, (96, 96, 60))
    grid, R = Grid.from_oblique_affine(A, (96, 96, 60))
    assert grid.shape == (96, 96, 60) and grid.axes == "RAS"
    np.testing.assert_allclose(grid.voxel_size_m, [2.5e-3] * 3, rtol=1e-3)
    np.testing.assert_allclose(grid.origin_m, np.array([-120.84, -99.16, -33.46]) * 1e-3)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6); assert np.linalg.det(R) > 0
    np.testing.assert_allclose(R @ np.diag(grid.voxel_size_m) * 1e3, A[:3, :3], rtol=1e-3)
    A[0, 1] += 0.4                                                      # a shear
    with pytest.raises(ValueError, match="shear"):
        Grid.from_oblique_affine(A, (96, 96, 60))


def test_rotate_sh_moves_the_features_with_the_rotation():
    R = so3.rotation_of((0.3, 0.5, 0.81)) @ so3.rotation_of((0.0, 1.0, 0.0))
    mu = np.array([0.2, -0.4, 0.89]); mu /= np.linalg.norm(mu)
    c = FOD.watson(8.0, mu=mu, lmax=8).coeffs
    np.testing.assert_allclose(so3.rotate_sh(c, R), FOD.watson(8.0, mu=R @ mu, lmax=8).coeffs, atol=1e-12)
    two = np.stack([c, FOD.watson(3.0, mu=(0, 0, 1), lmax=8).coeffs])                   # a batch, one rotation
    np.testing.assert_allclose(so3.rotate_sh(two, R)[0], so3.rotate_sh(c, R))
    with pytest.raises(ValueError, match="compact"):
        so3.rotate_sh(c[:44], R)
