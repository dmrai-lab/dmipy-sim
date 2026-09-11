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
    seq, g = example.acquisition(str(CROP), R, ph.grid, TE=0.030, delta=0.006, Delta=0.015)
    assert seq.prescription is not None and seq.prescription.axes == ph.grid.axes
    S = ph.replay(seq)
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
