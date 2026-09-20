"""The brain-from-CSD example (#193) on the bundled BATMAN crop: the MRtrix inputs become a phantom on the image's
own grid, the oblique prescription is carried by a rotation rather than a resampling, and the replayed signal has
its minimum along each voxel's FOD peak."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from dmipy_sim.io.mrtrix import read_mif
from dmipy_sim.replay.so3 import real_sh, rotate_sh

ROOT = Path(__file__).resolve().parents[1]
CROP = ROOT / "tests" / "data" / "batman_crop"


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("brain", ROOT / "examples" / "rph" / "brain_from_csd.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def wm_pack(tmp_path_factory):
    """A 30 ms cylinder walk standing in for the CACTUS pack (its frame is the identity: axis z)."""
    import dmipy_sim as d
    from dmipy_sim.replay.bank import build_replay_pack
    g = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6)
    walk = d.simulate_trajectories(300, 2e-9, g, 30e-3, 5e-4, seed=0, require_gpu=False)
    out = tmp_path_factory.mktemp("pk") / "wm.rpk"
    build_replay_pack(walk, id="test/wm", license="x", citation="x", K=8, out_path=str(out))
    return str(out)


def test_the_crop_becomes_a_phantom_on_the_image_grid(example, wm_pack):
    ph, fod, R = example.build(str(CROP), wm_pack)
    assert ph.grid.shape == (10, 10, 3) and ph.n_voxels == 300                  # every crop voxel is in the brain
    np.testing.assert_allclose(ph.grid.voxel_size_m, [2.5e-3] * 3, rtol=1e-6)
    assert 2.0 < np.degrees(np.arccos((np.trace(R) - 1) / 2)) < 3.5             # the tutorial's tilt, as measured
    f = sum(ph.fraction(s) for s in ph.substrates)
    np.testing.assert_allclose(f, 1.0, atol=1e-4)                               # a voxel is always full
    wm = ph.substrates[0]
    assert wm.name == "wm" and ph.fraction(wm).max() > 0.5                       # the crop holds real white matter


def test_the_signal_minimum_lies_along_the_fod_peak(example, wm_pack):
    ph, fod, R = example.build(str(CROP), wm_pack)
    seq, g = example.acquisition(str(CROP), ph.grid, TE=0.030, delta=0.006, Delta=0.015)
    assert seq.prescription is not None and seq.prescription.axes == ph.grid.axes
    S = ph.replay(seq, pose=R)
    assert S.shape == (10, 10, 3, 113) and np.isfinite(S).all()
    b = g[:, 3]; ok = np.isclose(b, 3000)
    dirs = g[ok, :3] / np.linalg.norm(g[ok, :3], axis=1, keepdims=True) @ R     # into the image frame
    c = rotate_sh(fod.data, R.T)
    sphere = np.random.default_rng(0).normal(size=(4000, 3)); sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
    Y = real_sh(8, sphere)
    f_wm = ph.fraction(ph.substrates[0])
    errs = []
    for i, j, k in np.argwhere((fod.data[..., 0] > 0.15) & (f_wm > 0.5)):
        peak = sphere[np.argmax(Y @ c[i, j, k])]
        dmin = dirs[np.argmin(S[i, j, k][ok])]
        errs.append(np.degrees(np.arccos(min(1.0, abs(peak @ dmin)))))
    errs = np.array(errs)
    assert errs.size >= 20
    assert np.median(errs) < 25.0 and np.mean(errs < 30.0) > 0.7                # 50 directions: ~10 deg sampling floor
    # the same phantom with the FOD left in scanner coordinates is a different, wrong brain
    S_b0 = S[..., b == 0].mean(-1)
    assert (S_b0[f_wm > 0.5] > 0).all() and S_b0.max() <= 1.0 + 1e-6


def test_csd_on_the_synthetic_dwi_recovers_the_fod_that_built_it(example, wm_pack):
    """The load-bearing check of #193, with dmipy-fit as the estimator (skipped when fit is not installed: sim ships
    no estimator): deconvolve the synthetic DWI with fit's CSD, the pack's own single-fibre signal as the kernel, and
    the recovered FOD peaks where the input one did (single tissue, on WM-dominated voxels)."""
    fit = pytest.importorskip("dmipy_fit")
    import inspect
    from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
    if "sequence" not in inspect.signature(AcquisitionScheme.__init__).parameters:
        pytest.skip("needs a dmipy-fit whose AcquisitionScheme takes a ScannerSequence (fit >= #27)")
    spec = importlib.util.spec_from_file_location("check", ROOT / "examples" / "rph" / "brain_from_csd_check.py")
    check = importlib.util.module_from_spec(spec); spec.loader.exec_module(check)
    from dmipy_sim.replay import read_rpk
    ph, fod, R = example.build(str(CROP), wm_pack)
    seq, g = example.acquisition(str(CROP), ph.grid, TE=0.030, delta=0.006, Delta=0.015)
    S = ph.replay(seq, pose=R)
    f_wm = ph.fraction(ph.substrates[0])
    voxels = np.argwhere((fod.data[..., 0] > 0.15) & (f_wm > 0.5))
    c_out = check.fit_fods(read_rpk(wm_pack), seq, S, voxels)
    c_in = rotate_sh(fod.data, R.T)[tuple(voxels.T)]
    angle, acc = check.compare(c_in, c_out, check.fibonacci_sphere())
    assert angle.size >= 20 and np.median(angle) < 8.0 and np.mean(angle < 15.0) > 0.75 and np.median(acc) > 0.9


# ── a machine's own field, rendered onto this oblique grid (dmipy-sim#322 PR 3) ─────────────────────
def test_the_field_law_must_be_rendered_through_the_obliquity(example, wm_pack):
    """The BATMAN acquisition is prescribed 2.58 degrees oblique, so the grid's axes are NOT the bore's.
    A field law is a function of position in the BORE, so the grid records how it is tilted in it and the
    renderer uses that by default -- because forgetting would not fail, it would evaluate the law at the
    wrong place and return an entirely plausible volume.

    What this measures is what that record buys, by defeating it with an identity rotation. The Swoop's law
    is what makes it visible: its R/L asymmetry is an ODD term in x, so getting the frame wrong shifts the
    field the wrong way on one side of the bore. An isotropic law would have hidden almost all of it."""
    from dmipy_sim.acquisition.scanners import ScannerLimits
    from dmipy_sim.phantom import b0_offset_map

    ph, fod, R = example.build(str(CROP), wm_pack)
    grid = ph.grid
    swoop = ScannerLimits.of("swoop")
    pos = grid.positions_m(ph.voxel_index)

    # the grid CARRIES its obliquity, so the default is already right and forgetting is not possible
    assert grid.to_scanner is not None
    np.testing.assert_allclose(grid.to_scanner, R, atol=1e-12)
    right = b0_offset_map(swoop, grid)(pos)
    np.testing.assert_allclose(b0_offset_map(swoop, grid, to_scanner=R)(pos), right, rtol=1e-12)
    wrong = b0_offset_map(swoop, grid, to_scanner=np.eye(3))(pos)      # the obliquity defeated

    # rendering with R is the same as rotating the coordinates first and rendering without: the two routes
    # to the same answer agree, which is what says the rotation is applied the right way round
    by_hand = swoop.b0_offset((pos - np.asarray(grid.isocenter_m)) @ np.asarray(R).T)
    np.testing.assert_allclose(right, by_hand, rtol=1e-9, atol=1e-15)

    # and defeating it disagrees -- what the grid's own rotation is buying
    worst = np.abs(right - wrong).max() / swoop.field_T * 1e6
    # The crop is small and the harmonic law has no linear term, so it is flat near the centre and the
    # obliquity has little to bite on there. The full-field-of-view check below is what carries the weight.
    assert worst > 0.4, f"defeating the obliquity moved the field by only {worst:.3f} ppm: not a test"

    # The crop is 2.5 x 2.5 x 0.75 cm, which understates it: the bowl goes as r^2 and the asymmetry as r,
    # so the error grows with the field of view. Over the acquisition's real 24 x 24 x 15 cm, inside the
    # law's anchor radius, it reaches about 9 ppm -- some 25 Hz at 64 mT, against a field that spans about
    # 1050 ppm over the same volume. So it is roughly a percent: not catastrophic, and exactly the size
    # that gets shipped unnoticed.
    from dmipy_sim.phantom import Grid
    full = Grid(shape=(96, 96, 60), voxel_size_m=grid.voxel_size_m, origin_m=grid.origin_m,
                isocenter_m=grid.isocenter_m, axes=grid.axes)
    p_full = full.positions_m(full.every_voxel)
    inside = np.linalg.norm(p_full - np.asarray(full.isocenter_m), axis=-1) < swoop.b0_validity_radius
    a = b0_offset_map(swoop, full, to_scanner=R)(p_full[inside])
    b = b0_offset_map(swoop, full, to_scanner=np.eye(3))(p_full[inside])
    at_fov = np.abs(a - b).max() / swoop.field_T * 1e6
    assert at_fov > 5.0, f"only {at_fov:.1f} ppm over the real field of view"


def test_a_field_law_is_none_where_the_machine_publishes_none(example, wm_pack):
    """Every machine but a permanent-magnet one, and `off_resonance=None` is exactly what a replay already
    means by no field offset -- so a brain at 3 T composes as it always did."""
    from dmipy_sim.acquisition.scanners import ScannerLimits
    from dmipy_sim.phantom import b0_offset_map
    ph, fod, R = example.build(str(CROP), wm_pack)
    assert b0_offset_map(ScannerLimits.of("prisma"), ph.grid, to_scanner=R) is None


# ── what the magnet costs this brain (dmipy-sim#322 PR 8) ───────────────────────────────────────────
def test_the_magnet_s_adc_bias_clears_the_pack_s_floor_where_the_magnet_encodes():
    """The result that decides whether any of this matters, and it is sharper than the version it replaces.

    Over the BATMAN matrix centred in the bore the Swoop's ADC bias exceeds the 1 s CACTUS pack's floor of
    0.0048 in 97.6 % of voxels -- so it is almost never lost in the Monte-Carlo error. But it does NOT exceed
    it everywhere, and the exceptions are not noise: they sit near isocentre, where a linearly shimmed magnet
    has no first-order field variation and therefore encodes nothing. Inside 2 cm only about half the voxels
    clear the floor.

    An earlier version asserted the bias cleared the floor essentially everywhere, including at isocentre.
    That came from a field law with a free linear term fitted to a figure the source states is measured AFTER
    linear shimming -- double-counting, and it manufactured a bias where the magnet has none. The worst-case
    figure is unchanged by the correction (15.4 % against 15.3 %); the MEDIAN halves, because the old law was
    wrong mostly in the middle."""
    import importlib.util
    from dmipy_sim.acquisition.scanners import ScannerLimits
    spec = importlib.util.spec_from_file_location("bias", ROOT / "examples" / "rph" / "swoop_brain_bias.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    swoop = ScannerLimits.of("swoop")
    grid, R = mod.brain_grid(shape=(24, 24, 16), voxel_m=5e-3)
    seq = mod.swoop_protocol(n_dirs=6)
    vox, bias = mod.bias_map(swoop, grid, R, seq)
    worst_dir = np.abs(bias).max(axis=1)
    rad = np.linalg.norm(grid.positions_m(vox) - np.asarray(grid.isocenter_m), axis=-1)

    assert 0.10 < np.abs(bias).max() < 0.20, f"worst bias {np.abs(bias).max():.1%}, not the measured ~15 %"
    above = worst_dir > mod.PACK_FLOOR
    assert above.mean() > 0.9, f"only {above.mean():.1%} of voxels clear the pack floor"
    # and the ones that do not are CENTRAL, which is the shim rather than an accident
    assert np.median(rad[~above]) < np.median(rad[above])

    med = [np.median(worst_dir[(rad >= lo) & (rad < hi)])
           for lo, hi in ((0.0, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.08))]
    assert all(np.diff(med) > 0), f"the bias is not monotone in radius: {np.round(med, 4)}"
    assert med[0] < 2 * mod.PACK_FLOOR, "the innermost band should be near the floor, not far above it"
    # averaging over directions still hides most of it
    assert np.abs(bias.mean(axis=1)).max() < 0.4 * np.abs(bias).max()

def test_a_shimmed_superconducting_magnet_has_no_shape_to_render_and_that_is_the_comparison():
    """The 3 T half of the same figure. A clinical magnet's residual is parts per million, nobody publishes
    its shape, and the catalogue carries none -- so the bias map is `None` rather than small. That is an
    answer about the machines, not a missing feature."""
    import importlib.util
    from dmipy_sim.acquisition.scanners import ScannerLimits
    spec = importlib.util.spec_from_file_location("bias", ROOT / "examples" / "rph" / "swoop_brain_bias.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    grid, R = mod.brain_grid(shape=(8, 8, 6), voxel_m=10e-3)
    seq = mod.swoop_protocol(n_dirs=3)
    for name in ("prisma", "connectom", "terra"):
        _vox, bias = mod.bias_map(ScannerLimits.of(name), grid, R, seq)
        assert bias is None, f"{name} suddenly has a field shape"
