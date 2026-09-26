"""The voxel-wall geometry: the readers, the traversal, confinement, and the Manhattan surface.

The validation ladder of dmrai-lab/dmipy-sim#468 part 4. Every number in an assertion here was
measured first and the docstring states it; the tolerances are the Monte-Carlo floor or a stated
geometric bias, never a bracket that happens to pass.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_sim.geometry import LabelVolume
from dmipy_sim.geometry.base import Box1D
from dmipy_sim.io.label_volume import (LabelVolumeError, crop_labels, read_label_volume, read_mhd,
                                       read_nrrd, write_nrrd)

D = 2e-9


# ─────────────────────────────────────────────────────────────────── fixtures (built on first use)
def voxelised(predicate, half, h):
    """A binary volume, label 0 where ``predicate(x, y, z)`` holds, over the cube ``[-half, half]^3``."""
    n = int(round(2 * half / h))
    c = (np.arange(n) + 0.5) * h - half
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    return np.where(predicate(X, Y, Z), 0, 1).astype(np.uint8), np.full(3, -half)


def slab(h=0.5e-6, n_pore=20, pad=10):
    """A 1-D pore of ``n_pore`` voxels between two grain slabs; the analytic twin is ``Box1D(n_pore h)``."""
    lab = np.ones((n_pore + 2 * pad, 8, 8), np.uint8)
    lab[pad:pad + n_pore] = 0
    return LabelVolume(lab, h), n_pore * h


def unit_local_time(g, step_l, n, n_walkers=20_000, seed=0):
    """``(rate, floor, end positions)``: the boundary local time per unit time at ``rho / D = 1``.

    For an equilibrium ensemble the exact value is ``D S / V`` of the walking pool: the expected
    overshoot per step at a flat wall is ``rho_0 l^2 / 12`` per unit area, and ``l^2 = 6 D dt``, so
    the estimator ``-dlog_w = 2 sum d_perp`` accumulates ``D (S/V) T``. That identity is what makes
    ``rho / D`` mean the same thing here as at every other wall in the package.
    """
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    r0 = g.init_positions(n_walkers, k1)
    keys = jax.random.split(k2, n_walkers)

    def one(r, key):
        def body(c, _):
            r, k, acc = c
            k, s = jax.random.split(k)
            u = jax.random.normal(s, (3,), dtype=jnp.float32)
            u = u / jnp.linalg.norm(u)
            r2, dl = g.reflect_with_log_weight(r, u * jnp.float32(step_l), jnp.float32(1.0))
            return (r2, k, acc + dl), None
        (rf, _, acc), _ = jax.lax.scan(body, (r, key, jnp.float32(0.0)), None, length=n)
        return acc, rf
    acc, rf = jax.jit(jax.vmap(one))(r0, keys)
    acc = np.asarray(acc)
    T = n * step_l ** 2 / (6 * D)
    return -acc.mean() / T, acc.std() / np.sqrt(n_walkers) / T, np.asarray(rf)


def walk(g, step_l, n, n_walkers=20_000, seed=0):
    """End positions of a plain reflecting walk of ``n`` steps of length ``step_l``."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    r0 = g.init_positions(n_walkers, k1)
    keys = jax.random.split(k2, n_walkers)

    def one(r, key):
        def body(c, _):
            r, k = c
            k, s = jax.random.split(k)
            u = jax.random.normal(s, (3,), dtype=jnp.float32)
            u = u / jnp.linalg.norm(u)
            return (g.reflect(r, u * jnp.float32(step_l)), k), None
        (rf, _), _ = jax.lax.scan(body, (r, key), None, length=n)
        return rf
    return np.asarray(r0), np.asarray(jax.jit(jax.vmap(one))(r0, keys))


# ─────────────────────────────────────────────────────────────────── the readers
def test_a_header_without_a_voxel_size_is_refused_and_a_contradicting_one_too(tmp_path):
    """A walk needs the physical extent of a voxel; a header that states none and a caller who states
    none is refused by name, and a header that states one while the caller states another is refused
    rather than resolved by precedence."""
    p = tmp_path / "bare.nrrd"
    lab = np.zeros((4, 3, 2), np.uint8)
    body = np.ascontiguousarray(lab.transpose(2, 1, 0)).tobytes()
    p.write_bytes(b"NRRD0004\ntype: uchar\nencoding: raw\ndimension: 3\nsizes: 4 3 2\n\n" + body)
    with pytest.raises(LabelVolumeError, match="states no voxel size"):
        read_nrrd(p)
    v = read_nrrd(p, voxel_size=1e-6)
    assert np.allclose(v.voxel_size, 1e-6) and v.labels.shape == (4, 3, 2)

    q = tmp_path / "sized.nrrd"
    write_nrrd(q, lab, (1e-6, 2e-6, 3e-6), (1e-3, 0.0, 0.0))
    with pytest.raises(LabelVolumeError, match="one of the two is wrong"):
        read_nrrd(q, voxel_size=5e-6)


def test_nrrd_round_trip_and_the_unit_of_the_header(tmp_path):
    """``write_nrrd`` / ``read_nrrd`` are a fixed point in metres, and a header in micrometres reads as
    metres: the Imperial 2007 rocks state ``space units: "um" "um" "um"`` with a 10.002 um voxel."""
    lab = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2) % 2
    p = tmp_path / "v.nrrd"
    write_nrrd(p, lab, (1e-6, 2e-6, 3e-6), (1e-3, -2e-3, 0.0))
    v = read_label_volume(p)
    assert np.array_equal(v.labels, lab)
    assert np.allclose(v.voxel_size, [1e-6, 2e-6, 3e-6]) and np.allclose(v.origin, [1e-3, -2e-3, 0.0])

    d = tmp_path / "um.nhdr"
    (tmp_path / "um.raw").write_bytes(np.ascontiguousarray(lab.transpose(2, 1, 0)).tobytes())
    d.write_text("NRRD0004\ntype: uchar\nencoding: raw\ndimension: 3\nsizes: 4 3 2\n"
                 'space directions: (10.002,0,0) (0,10.002,0) (0,0,10.002)\n'
                 'space units: "um" "um" "um"\ndata file: um.raw\n')
    w = read_label_volume(d)
    assert np.allclose(w.voxel_size, 10.002e-6) and np.array_equal(w.labels, lab)


def test_a_payload_that_does_not_match_its_header_is_refused(tmp_path):
    """The released LV60B declares ``sizes: 450 450 450`` and ships 450x450x425 voxels. A reader that
    reshapes whatever it is given would walk a different rock, so the mismatch is refused with both
    byte counts."""
    p = tmp_path / "short.nhdr"
    (tmp_path / "short.raw").write_bytes(b"\x00" * (4 * 3 * 1))
    p.write_text("NRRD0004\ntype: uchar\nencoding: raw\ndimension: 3\nsizes: 4 3 2\n"
                 "spacings: 1e-6 1e-6 1e-6\ndata file: short.raw\n")
    with pytest.raises(LabelVolumeError, match="do not describe the same image"):
        read_nrrd(p)


def test_metaimage_is_read_in_millimetres(tmp_path):
    """MetaImage's ``ElementSpacing`` and ``Offset`` are in millimetres, the format's unit: the Imperial
    2015 rocks state ``ElementSpacing = 3.0035 3.0035 3.0035`` in um-sized voxels of a mm-scale core,
    so the number a reader must produce is 3.0035e-3 m and nothing else."""
    lab = (np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2) % 3)
    (tmp_path / "m.raw").write_bytes(np.ascontiguousarray(lab.transpose(2, 1, 0)).tobytes())
    p = tmp_path / "m.mhd"
    p.write_text("ObjectType = Image\nNDims = 3\nDimSize = 4 3 2\nElementType = MET_UCHAR\n"
                 "ElementSpacing = 0.01 0.02 0.03\nOffset = 1 2 3\nElementDataFile = m.raw\n")
    v = read_mhd(p)
    assert np.array_equal(v.labels, lab)
    assert np.allclose(v.voxel_size, [1e-5, 2e-5, 3e-5]) and np.allclose(v.origin, [1e-3, 2e-3, 3e-3])


def test_a_tiff_stack_is_one_image_whether_it_is_a_directory_or_a_multi_page_file(tmp_path):
    """The same volume in the two TIFF containers reads back as the same array.

    A directory of slices and the multi-page file of those slices are one image; reading them into
    two different index orders would make them two substrates, silently, and no shape check catches
    it on a non-cubic volume. Measured on a 4x6x10 volume: both come back equal to the source, and to
    each other; before, the directory branch returned `vol.transpose(1, 0, 2)`.
    """
    tifffile = pytest.importorskip("tifffile")
    vol = (np.arange(4 * 6 * 10, dtype=np.uint8).reshape(4, 6, 10) % 3)     # (nx, ny, nz), non-cubic
    tifffile.imwrite(tmp_path / "multi.tif", np.transpose(vol, (2, 1, 0)))  # (z, y, x)
    d = tmp_path / "slices"
    d.mkdir()
    for k in range(vol.shape[2]):
        tifffile.imwrite(d / f"s{k:03d}.tif", vol[:, :, k].T)               # (y, x)
    a = read_label_volume(tmp_path / "multi.tif", voxel_size=1e-6)
    b = read_label_volume(d, voxel_size=1e-6)
    assert np.array_equal(a.labels, vol) and np.array_equal(b.labels, vol)
    assert np.allclose(a.voxel_size, 1e-6) and np.allclose(b.voxel_size, 1e-6)


def test_a_tiff_stack_takes_its_voxel_size_from_the_imagej_tags_when_it_has_them(tmp_path):
    """A TIFF carries in-plane resolution at best and no slice spacing, so the voxel size comes from
    ``voxel_size=``; an ImageJ stack that states ``spacing`` and ``unit`` is read from the tags, in
    that unit -- 0.5 um in plane and 2 um between slices here."""
    tifffile = pytest.importorskip("tifffile")
    vol = np.zeros((4, 6, 3), np.uint8)
    vol[1, 2, 1] = 1
    tifffile.imwrite(tmp_path / "ij.tif", np.transpose(vol, (2, 1, 0)), imagej=True,
                     resolution=(1 / 0.5, 1 / 0.5), metadata={"spacing": 2.0, "unit": "um"})
    v = read_label_volume(tmp_path / "ij.tif")
    assert np.array_equal(v.labels, vol)
    assert np.allclose(v.voxel_size, [0.5e-6, 0.5e-6, 2e-6])
    with pytest.raises(LabelVolumeError, match="one of the two is wrong"):
        read_label_volume(tmp_path / "ij.tif", voxel_size=1e-6)


def test_nifti_is_read_in_the_units_its_header_declares(tmp_path):
    """NIfTI's zooms are in the unit of ``xyzt_units``, millimetres being the format's default, and the
    origin is the affine's translation in the same unit: a 0.3 mm isotropic volume at an offset of
    (1, 2, 3) mm reads as 3e-4 m and (1e-3, 2e-3, 3e-3) m."""
    nib = pytest.importorskip("nibabel")
    vol = (np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3) % 2)
    aff = np.diag([0.3, 0.3, 0.3, 1.0])
    aff[:3, 3] = (1.0, 2.0, 3.0)
    nib.save(nib.Nifti1Image(vol, aff), str(tmp_path / "v.nii"))
    v = read_label_volume(tmp_path / "v.nii")
    assert np.array_equal(v.labels, vol)
    assert np.allclose(v.voxel_size, 3e-4) and np.allclose(v.origin, [1e-3, 2e-3, 3e-3])


def test_hdf5_reads_element_size_um_slowest_axis_first(tmp_path):
    """The ilastik / ImageJ ``element_size_um`` attribute is in micrometres, slowest axis first, and an
    h5py dataset is written ``(z, y, x)``: ``[3, 2, 1]`` um is a voxel of (1, 2, 3) um in ``(i, j, k)``.
    A file with more than one 3-D dataset is refused unless one is named."""
    h5py = pytest.importorskip("h5py")
    vol = (np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3) % 2)
    with h5py.File(tmp_path / "v.h5", "w") as fh:
        d = fh.create_dataset("labels", data=np.transpose(vol, (2, 1, 0)))
        d.attrs["element_size_um"] = np.array([3.0, 2.0, 1.0])
    v = read_label_volume(tmp_path / "v.h5")
    assert np.array_equal(v.labels, vol)
    assert np.allclose(v.voxel_size, [1e-6, 2e-6, 3e-6])

    with h5py.File(tmp_path / "two.h5", "w") as fh:
        fh.create_dataset("a", data=np.zeros((2, 2, 2), np.uint8))
        fh.create_dataset("b", data=np.zeros((2, 2, 2), np.uint8))
    with pytest.raises(LabelVolumeError, match="holds 2 3-D datasets"):
        read_label_volume(tmp_path / "two.h5", voxel_size=1e-6)
    v2 = read_label_volume(tmp_path / "two.h5", dataset="b", voxel_size=1e-6)
    assert v2.labels.shape == (2, 2, 2)
    with pytest.raises(LabelVolumeError, match="names an HDF5 dataset"):
        read_label_volume(tmp_path / "v.h5", format="nrrd", dataset="labels")


def test_a_crop_moves_the_origin_and_an_impossible_one_is_refused():
    """The crop IS part of the substrate (a published measurement is made on a stated sub-volume), so
    its lower corner becomes the origin; a crop outside the image is refused."""
    from dmipy_sim.io.label_volume import LabelVolumeFile
    lab = np.arange(5 * 5 * 5, dtype=np.uint8).reshape(5, 5, 5) % 2
    v = LabelVolumeFile(lab, np.full(3, 2e-6), np.zeros(3))
    c = crop_labels(v, (1, 1, 1, 4, 4, 4))
    assert c.labels.shape == (3, 3, 3) and np.array_equal(c.labels, lab[1:4, 1:4, 1:4])
    assert np.allclose(c.origin, 2e-6)
    with pytest.raises(LabelVolumeError, match="not a non-empty sub-volume"):
        crop_labels(v, (0, 0, 0, 6, 5, 5))


# ─────────────────────────────────────────────────────────────────── the construction contract
def test_a_label_no_pool_names_is_refused():
    """Every label a walker can meet is a pool: a volume holding a label the map does not name would
    otherwise walk it as if it were the pool id ``-1``, which is the crop's exterior."""
    lab = np.zeros((4, 4, 4), np.uint8)
    lab[0, 0, 0] = 7
    with pytest.raises(ValueError, match=r"labels \[7\]"):
        LabelVolume(lab, 1e-6)
    with pytest.raises(ValueError, match="is not one of"):
        LabelVolume(np.zeros((4, 4, 4), np.uint8), 1e-6, pool="cell")
    with pytest.raises(ValueError, match="occupies no voxel"):
        LabelVolume(np.zeros((4, 4, 4), np.uint8), 1e-6, pools={0: "free", 1: "grain"}, pool="grain")
    with pytest.raises(ValueError, match="positive and finite"):
        LabelVolume(np.zeros((4, 4, 4), np.uint8), 0.0)


def test_length_scales_are_the_voxel_and_the_pore():
    """``min_feature`` is the voxel -- the smallest feature a segmentation can express -- and
    ``surface_pore`` the walking pool's measured ``V/S``. There is no ``lookup_cell``: the traversal
    visits every voxel the path enters, so no step can outrun a candidate gather."""
    g, d = slab()
    ls = g.length_scales
    assert ls.min_feature == pytest.approx(0.5e-6)
    assert ls.surface_pore == pytest.approx(d / 2)          # a slab's V/S is half its width
    assert ls.lookup_cell is None and ls.is_mesh_feature is False


def test_the_voxelised_surface_is_the_manhattan_surface():
    """A convex solid's voxel-face area is the sum of its three projected areas, twice -- so a
    voxelised sphere's is ``6 pi R^2``, exactly 3/2 of the sphere's own ``4 pi R^2``, and a voxelised
    circle's perimeter is ``8 R``, ``4/pi`` of ``2 pi R``. Both ratios are resolution-INDEPENDENT:
    refining the grid does not converge them onto the smooth surface, it only sharpens the projection.
    Measured on R = 5 um: sphere 1.4493 / 1.4962 / 1.5069 at h/R = 0.2 / 0.1 / 0.05; cylinder 1.2658
    at both h/R = 0.1 and 0.05 (4/pi = 1.27324).

    This is why a relaxation rate measured on a segmentation is ``rho`` times the VOXELISED S/V, and
    why Talabi's fitted ``rho`` is an effective value tied to his 10 um voxel (thesis §7.6.2).
    """
    R = 5e-6
    ratios = []
    for hr in (0.2, 0.1, 0.05):
        lab, org = voxelised(lambda x, y, z: x * x + y * y + z * z < R * R, 1.4 * R, R * hr)
        ratios.append(LabelVolume(lab, R * hr, origin=org).surface_to_volume() / (3 / R))
    assert ratios == pytest.approx([1.4493, 1.4962, 1.5069], abs=2e-3)
    assert ratios[-1] < 1.5 * 1.01                       # converging onto 3/2 from above

    h = R * 0.05
    nxy = int(round(2 * 1.4 * R / h))
    c = (np.arange(nxy) + 0.5) * h - 1.4 * R
    X, Y = np.meshgrid(c, c, indexing="ij")
    disc = np.where(X * X + Y * Y < R * R, 0, 1).astype(np.uint8)
    cyl = LabelVolume(np.repeat(disc[:, :, None], 6, axis=2), h,
                      origin=[-1.4 * R, -1.4 * R, 0.0], periodic=[False, False, True])
    assert cyl.surface_to_volume() / (2 / R) == pytest.approx(4 / np.pi, rel=6e-3)


def test_seeding_is_uniform_on_the_walking_pool():
    """``init_positions`` draws uniformly on the union of the pool's voxels: for a 10 um pore the mean
    and standard deviation of the confined coordinate are the uniform ones to the 1/sqrt(N) floor."""
    g, d = slab()
    r0 = np.asarray(g.init_positions(40_000, jax.random.PRNGKey(0)))
    assert np.all(np.asarray(g.classify_positions_exact(r0)) == 0)
    x = r0[:, 0] - g.origin[0] - 10 * 0.5e-6
    floor = d / np.sqrt(12 * 40_000)
    assert x.mean() == pytest.approx(d / 2, abs=4 * floor)
    assert x.std() == pytest.approx(d / np.sqrt(12), rel=0.02)


def test_classify_position_is_the_label_and_the_crop_exterior_is_minus_one():
    g, _ = slab()
    h = 0.5e-6
    inside = jnp.asarray([12.5 * h, 4 * h, 4 * h], jnp.float32)
    grain = jnp.asarray([2.5 * h, 4 * h, 4 * h], jnp.float32)
    outside = jnp.asarray([12.5 * h, -3 * h, 4 * h], jnp.float32)
    assert int(g.classify_position(inside)) == 0
    assert int(g.classify_position(grain)) == 1
    assert int(g.classify_position(outside)) == -1


# ─────────────────────────────────────────────────────────────────── confinement and local time
@pytest.mark.parametrize("frac", [0.5, 1.0, 2.0])
def test_a_voxelised_sphere_confines_every_walker_to_its_voxel_corners(frac):
    """No walker leaves the pool, at any step length, and none goes further than the corner of a
    surface voxel: ``max|r| - R <= h sqrt(3) / 2``, which for h = 1 um is 866 nm. Measured over 200
    steps at 20,000 walkers: 0 escapes at 0.5, 1 and 2 voxels per step, max|r| - R = 686 / 732 /
    713 nm. The traversal is what makes that true at 2 voxels per step, and the reject-escape guard is
    what makes it true when float32 puts a position within an ulp of a face (without it, 9 of 200,000
    walkers left the sphere over 200 steps at 0.575 voxels).
    """
    R, h = 5e-6, 1e-6
    lab, org = voxelised(lambda x, y, z: x * x + y * y + z * z < R * R, 1.4 * R, h)
    g = LabelVolume(lab, h, origin=org)
    _, rf = walk(g, frac * h, 200)
    assert int((np.asarray(g.classify_positions_exact(rf)) != 0).sum()) == 0
    assert np.linalg.norm(rf, axis=1).max() - R <= h * np.sqrt(3) / 2


@pytest.mark.parametrize("frac", [0.5, 1.0, 2.0])
def test_the_boundary_local_time_is_D_times_the_voxelised_surface_to_volume(frac):
    """``-E[dlog_w] / T`` at ``rho / D = 1`` is ``D (S/V)`` of the walking pool, at any step length:
    the overshoot estimator has no curvature bias on a flat face, so nothing has to be resolved but
    the pore. Measured on the voxelised R = 5 um sphere at 20,000 walkers x 200 steps:
    1.00260 / 1.00156 / 0.99972 of ``D S/V`` at 0.5 / 1 / 2 voxels per step, floors 4.3 / 2.3 / 1.1e-3.

    This identity is the reason the surface tier's sub-step rule here is ``(V/S) / 2`` rather than the
    ``(V/S) / 8`` a curved analytic wall needs.
    """
    R, h = 5e-6, 1e-6
    lab, org = voxelised(lambda x, y, z: x * x + y * y + z * z < R * R, 1.4 * R, h)
    g = LabelVolume(lab, h, origin=org)
    rate, floor, _ = unit_local_time(g, frac * h, 200)
    exact = D * g.surface_to_volume()
    assert rate / exact == pytest.approx(1.0, abs=max(4 * floor / exact, 0.01))


def test_a_voxelised_slab_is_Box1D_and_its_surface_rate_is_brownstein_tarr():
    """The voxelised slab and ``Box1D`` of the same width accumulate the same boundary local time, to
    the Monte-Carlo floor, so the two geometries' ``rho`` is one quantity. Measured at 200,000 walkers
    over 2 / 1 / 0.5 / 0.25 voxels per step: LabelVolume 1.00146 / 1.00197 / 1.00391 / 0.99795 and
    Box1D 1.00049 / 1.00107 / 0.99863 / 0.99540 of ``D S/V``, floor 2.9e-3; at the 20,000 walkers this
    test runs (floor 9.0e-3) LabelVolume 1.00467 / 1.00349 and Box1D 1.00651 / 1.00974 at 0.5 / 1.

    And ``D (S/V)`` IS the Brownstein-Tarr rate: at ``rho = 10 um/s`` on a 10 um slab the eigenvalue
    ``xi tan(xi d/2) = rho/D`` gives ``D xi^2 = 1.98344 1/s`` against ``rho S/V = 2.0 1/s``, so the
    fast-diffusion limit is the right reference to 0.83 %.
    """
    g, d = slab()
    box = Box1D(d)
    sv = g.surface_to_volume()
    for frac in (0.5, 1.0):
        a, fa, _ = unit_local_time(g, frac * 0.5e-6, int(200 / frac ** 2))
        b, fb, _ = unit_local_time(box, frac * 0.5e-6, int(200 / frac ** 2))
        tol = 4 * max(fa, fb) / (D * sv)
        assert a / (D * sv) == pytest.approx(1.0, abs=tol)
        assert b / (D * sv) == pytest.approx(1.0, abs=tol)
        assert (a - b) / (D * sv) == pytest.approx(0.0, abs=np.sqrt(2) * tol)     # the parity itself

    from scipy.optimize import brentq
    rho, a_half = 1e-5, d / 2
    z = brentq(lambda z: z * np.tan(z) - rho * a_half / D, 1e-12, np.pi / 2 - 1e-9)
    assert D * (z / a_half) ** 2 == pytest.approx(1.98344, abs=1e-4)
    assert rho * sv == pytest.approx(2.0, rel=1e-9)


def test_a_step_across_many_voxels_still_meets_every_face():
    """A step of any length is exact: the traversal walks the voxels the segment crosses, so a step of
    four voxels through a one-voxel-wide channel cannot pass through its wall. Measured: 0 of 20,000
    walkers leave a one-voxel channel over 100 steps of 1, 2 and 4 voxels.
    """
    h = 1e-6
    lab = np.ones((3, 3, 24), np.uint8)
    lab[1, 1, :] = 0                                        # a single-voxel channel along z
    g = LabelVolume(lab, h)
    for frac in (1.0, 2.0, 4.0):
        _, rf = walk(g, frac * h, 100, n_walkers=20_000)
        assert int((np.asarray(g.classify_positions_exact(rf)) != 0).sum()) == 0
        assert np.all((rf[:, 0] > h) & (rf[:, 0] < 2 * h) & (rf[:, 1] > h) & (rf[:, 1] < 2 * h))


def test_a_periodic_axis_wraps_the_grid_and_keeps_the_position_continuous():
    """A periodic axis repeats the grid and the returned position stays continuous, so the gradient
    phase is right: a walk along an open tube spreads past the box without meeting a wall, and its
    mean square displacement is free diffusion's to the Monte-Carlo floor.
    """
    h = 1e-6
    lab = np.ones((5, 5, 8), np.uint8)
    lab[1:4, 1:4, :] = 0
    g = LabelVolume(lab, h, periodic=[False, False, True])
    r0, rf = walk(g, h, 400, n_walkers=20_000)
    assert int((np.asarray(g.classify_positions_exact(rf)) != 0).sum()) == 0
    dz = rf[:, 2] - r0[:, 2]
    assert np.abs(dz).max() > 8 * h                        # the walk left the box on the periodic axis
    msd_z = (dz ** 2).mean()
    expect = 400 * h ** 2 / 3                              # one third of the step length squared per step
    # measured 1.0199 of free diffusion at 20,000 walkers, whose floor on a mean square is sqrt(2/N)
    assert msd_z == pytest.approx(expect, rel=4 * np.sqrt(2 / 20_000))


def permeable_walk(g, step_l, n, kappa, n_walkers=20_000, seed=0):
    """End positions of a walk whose faces are crossed under the Powles rule at ``kappa``."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    r0 = g.init_positions(n_walkers, k1)
    keys = jax.random.split(k2, n_walkers)
    kod = jnp.float32(kappa / D)

    def one(r, key):
        def body(c, _):
            r, k = c
            k, s, w = jax.random.split(k, 3)
            u = jax.random.normal(s, (3,), dtype=jnp.float32)
            u = u / jnp.linalg.norm(u)
            hit = g.interact(r, u * jnp.float32(step_l), kappa_over_D=kod, key=w)
            return (hit.r, k), None
        (rf, _), _ = jax.lax.scan(body, (r, key), None, length=n)
        return rf
    return np.asarray(r0), np.asarray(jax.jit(jax.vmap(one))(r0, keys))


def test_a_permeable_face_equilibrates_the_two_pools_by_volume():
    """One Powles trial per step at the first face met: a permeable interface between two pools that
    both hold water equilibrates to their volume ratio, which is detailed balance. Measured on a box
    that is half one pool and half the other, at kappa = 4e-4 m/s over 1500 steps of one voxel
    (0.125 s, six exchange times ``V / (kappa S)``): 0.49890 of 20,000 walkers in the far pool against
    the exact 0.5, floor 3.5e-3; at 1000 and 2500 steps 0.49585 and 0.50570.

    The per-hit transmission here is 0.1, far above the ``CROSSING_P_MAX`` a RATE measurement needs --
    the equilibrium partition is a statement about detailed balance and holds at any probability, which
    is what makes it the cheap check.
    """
    h = 1e-6
    lab = np.ones((16, 6, 6), np.uint8)
    lab[:8] = 0
    g = LabelVolume(lab, h, pools={0: "free", 1: "other"}, permeability=4e-4)
    r0, rf = permeable_walk(g, h, 1500, 4e-4)
    assert np.all(np.asarray(g.classify_positions_exact(r0)) == 0)
    frac = float((np.asarray(g.classify_positions_exact(rf)) == 1).mean())
    assert frac == pytest.approx(0.5, abs=4 / np.sqrt(20_000))


def test_an_impermeable_volume_refuses_a_crossing_request():
    """``interact`` with ``kappa != 0`` on a wall the geometry was not given a permeability for is a
    request the geometry cannot honour, so it is refused rather than reflected silently."""
    g, _ = slab()
    r = g.init_positions(1, jax.random.PRNGKey(0))[0]
    step = jnp.asarray([1e-8, 0.0, 0.0], jnp.float32)
    hit = g.interact(r, step)
    assert hit.r.shape == (3,) and not bool(hit.illegal)
    with pytest.raises(NotImplementedError, match="carries no side"):
        g.interact(r, step, side=jnp.int32(1))


def test_a_label_volume_walk_packs_and_replays_C0_C1_C2(tmp_path, monkeypatch):
    """`build_replay_pack` needs nothing new for a label volume: the positions, the compartment column
    and the boundary local time are the channels every other substrate records, so one walk of a
    segmented image replays at any rho and T2.

    Measured on a voxelised R = 5 um sphere at 4,000 walkers over 20 ms (PGSE delta 3 ms / Delta 12 ms
    at b = 1e9): the bare diffusion replay is 0.70889, the same pack under ``Tissue(rho=1e-5)`` is
    0.61221, and the fused walk with the same rho baked in gives 0.61001 -- a difference of 0.0022
    against the walk's own floor of 0.0158.
    """
    from dmipy_sim import build_replay_pack, pgse, simulate, simulate_trajectories
    from dmipy_sim.spec import Tissue
    monkeypatch.setenv("DMIPY_SIM_SURFACE_DIR", str(tmp_path))
    R, h = 5e-6, 1e-6
    lab, org = voxelised(lambda x, y, z: x * x + y * y + z * z < R * R, 8e-6, h)
    g = LabelVolume(lab, h, origin=org, surface_relaxivity_t2=1e-5)
    walk = simulate_trajectories(4000, D, g, T_max=0.02, dt_save=5e-4, seed=0, require_gpu=False)
    assert walk.boundary_local_time is not None and walk.compartment is not None
    pack = build_replay_pack(walk, id="test/label-volume", license="CC-BY-4.0", citation="test")
    assert pack.substrate.walls[0].surface.kind == "label_volume"
    assert [p.name for p in pack.substrate.pools] == ["free", "grain"]
    wf = pgse([[1, 0, 0]], 0.003, 0.012, bvalues=[1e9], n_t=80)
    bare = float(np.asarray(pack.replay(wf)).ravel()[0])
    with_rho = float(np.asarray(pack.replay(wf, tissue=Tissue(rho=1e-5))).ravel()[0])
    fused = float(np.asarray(simulate(4000, D, wf, g, seed=0, require_gpu=False)).ravel()[0])
    assert 0.0 < with_rho < bare <= 1.0                       # relaxivity only ever costs signal
    assert with_rho == pytest.approx(fused, abs=4 / np.sqrt(4000))


@pytest.mark.skipif(not os.environ.get("DMIPY_SIM_IMPERIAL2007_DIR"),
                    reason="set DMIPY_SIM_IMPERIAL2007_DIR to the Imperial 2007 images")
def test_the_imperial_rocks_read_and_measure_as_talabi_tabulated():
    """The released Imperial 2007 images, read and measured: LV60A's central 300^3 crop has porosity
    0.3688 against Talabi's Table 7-1 value of 0.377 and staircase S/V 59,821 1/m against his 57,670.
    """
    d = os.environ["DMIPY_SIM_IMPERIAL2007_DIR"]
    v = read_label_volume(os.path.join(d, "LV60A.nhdr"))
    assert v.labels.shape == (450, 450, 450)
    assert np.allclose(v.voxel_size, 10.002e-6)
    g = LabelVolume(crop_labels(v, (75, 75, 75, 375, 375, 375)).labels, v.voxel_size)
    assert g.porosity() == pytest.approx(0.3688, abs=5e-4)
    assert g.surface_to_volume() == pytest.approx(59821, rel=5e-3)
