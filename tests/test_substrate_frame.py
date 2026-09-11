"""A pack's intrinsic frame (RPK.md 4.2, ``walk_params.substrate_frame``) is honoured by every posed replay: a
walk stored with its bundle along x composes exactly like the same walk stored along z, once the pack says so
(dmipy-sim#193: the CACTUS packs' bundles run along x while nothing declared it)."""
import copy

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.phantom import Grid, ODF, PackSubstrate, Peaks, Phantom
from dmipy_sim.replay import ReplayPack, read_rpk, so3
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.compression import pack_position_arrays, read_position_coeffs
from dmipy_sim.replay.fod import FOD


@pytest.fixture(scope="module")
def packs(tmp_path_factory):
    """The same cylinder walk twice: as stored (axis z, no frame), and rotated so its axis is x with the frame
    declared -- plus the rotated walk WITHOUT a frame, the lie the test catches."""
    g = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6)
    walk = d.simulate_trajectories(600, 2e-9, g, 10e-3, 5e-4, seed=0, require_gpu=False)
    p = tmp_path_factory.mktemp("pk") / "z.rpk"
    build_replay_pack(walk, id="t/z", license="x", citation="x", K=8, out_path=str(p))
    z = read_rpk(str(p))
    Q = so3.rotation_of((1.0, 0.0, 0.0))                                  # z -> x
    C = read_position_coeffs(z.arrays, dtype=np.float64) @ Q.T           # every coefficient vector rotated
    arrays = dict(z.arrays); arrays.update(pack_position_arrays(C, np.float32))
    meta = copy.deepcopy(z.meta)
    meta.setdefault("walk_params", {})["substrate_frame"] = Q.tolist()          # d_stored = Q d_canonical, exactly
    x_declared = ReplayPack(arrays, meta)
    x_silent = ReplayPack(arrays, copy.deepcopy(z.meta))
    return z, x_declared, x_silent


def test_the_frame_is_read_from_the_pack(packs):
    z, x, _ = packs
    np.testing.assert_allclose(z.substrate_frame, np.eye(3)); np.testing.assert_allclose(z.frame_axis, [0, 0, 1])
    np.testing.assert_allclose(x.frame_axis, [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(x.substrate_frame @ x.substrate_frame.T, np.eye(3), atol=1e-12)


def test_a_declared_frame_makes_the_rotated_walk_replay_like_the_original(packs):
    z, x, x_silent = packs
    Q = np.asarray(x.meta["walk_params"]["substrate_frame"])
    seq = sequences.pgse(np.eye(3), 2e-3, 5e-3, gradient_strengths=[1.0] * 3, TE=10e-3, slew_rate=1000.0)   # b ~ 1200 s/mm^2 on a 2 ms lobe
    # a rotation of the canonical frame: exact, the roll included
    for o in (so3.rotation_of((0.3, 0.5, 0.81)), np.eye(3)):
        np.testing.assert_allclose(x.replay(seq, orientation=o), z.replay(seq, orientation=o), rtol=1e-5)
    # (a lab DIRECTION for the axis leaves each pack its own roll about it, and a single cylinder in a square
    # periodic cell is only 4-fold symmetric about its axis, so that route is not an equality to test here)
    # stored coordinates, no pose: the rotated walk as stored IS the original posed by Q
    np.testing.assert_allclose(x.replay(seq), z.replay(seq, orientation=Q), rtol=1e-5)
    # the composition: peaks and an ODF through the pose expansion
    grid = Grid(shape=(1, 1, 1), voxel_size_m=(1e-3,) * 3)
    n = np.array([[[[0.3, 0.5, 0.81]]]]) / np.linalg.norm([0.3, 0.5, 0.81])
    c = np.zeros((1, 1, 1, 45)); c[0, 0, 0] = FOD.watson(20.0, mu=n[0, 0, 0], lmax=8).coeffs
    out = {}
    for name, pk in (("z", z), ("x", x), ("silent", x_silent)):
        wm = PackSubstrate(pk, m0=1.0, name=name)
        out[name] = (Phantom.compose(grid, fractions={wm: np.ones((1, 1, 1))}, orientation=Peaks(n)).replay(seq)[0, 0, 0],
                     Phantom.compose(grid, fractions={wm: np.ones((1, 1, 1))}, orientation=ODF(c, basis="mrtrix3")).replay(seq)[0, 0, 0])
    for k in range(2):
        declared = np.abs(out["x"][k] - out["z"][k]).max()
        silent = np.abs(out["silent"][k] - out["z"][k]).max()
        np.testing.assert_allclose(out["x"][k], out["z"][k], rtol=5e-3)         # the declared frame: the same physics
        assert silent > 10 * declared and silent > 1e-2                         # undeclared: a different (wrong) tissue
