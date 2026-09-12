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


# ---------------------------------------------------------------------------- the frame declared by the substrate (#194)
def test_a_strand_list_declares_its_frame_from_its_own_strands(tmp_path):
    """Three strands along x and two along y: z of the frame is the larger bundle's axis, y the other's, and the
    realisation records both bundles -- never a PCA over positions (the bisector of a crossing)."""
    from dmipy_sim.io.strands import write_strands
    from dmipy_sim.spec import strands_spec
    from dmipy_sim.replay.bank import frame_of_spec
    r = 1e-6
    along_x = [np.array([[-5e-6, y, 0.0], [5e-6, y, 0.0]]) for y in (-3e-6, 0.0, 3e-6)]
    along_y = [np.array([[x, -5e-6, 0.0], [x, 5e-6, 0.0]]) for x in (-2e-6, 2e-6)]
    p = tmp_path / "cross.txt"
    write_strands(str(p), along_x + along_y, [r] * 5, 20e-6)
    spec = strands_spec(str(p), g_ratio=None)
    F = frame_of_spec(spec)
    assert abs(abs(F[:, 2] @ [1, 0, 0]) - 1) < 1e-9 and abs(abs(F[:, 1] @ [0, 1, 0]) - 1) < 1e-9
    assert np.linalg.det(F) > 0 and np.allclose(F @ F.T, np.eye(3))
    b = spec.realisation["bundles"]
    assert [x["n_strands"] for x in b] == [3, 2] and abs(abs(np.dot(b[1]["axis"], [0, 1, 0])) - 1) < 1e-9


def test_an_analytic_geometry_declares_its_axis_and_the_pack_carries_it():
    """A cylinder oriented along x: its spec's frame axis is x (not the default z), and a pack built from its walk
    declares ``walk_params.substrate_frame`` with that axis as column 3."""
    g = d.Cylinder(radius=2e-6, orientation=(1, 0, 0))
    assert np.allclose(g.spec.frame.axis, [1, 0, 0])
    walk = d.simulate_trajectories(200, 2e-9, g, 4e-3, 5e-4, seed=1, require_gpu=False)
    pk = build_replay_pack(walk, id="t/x", license="x", citation="x", K=4)
    F = np.asarray(pk.meta["walk_params"]["substrate_frame"])
    assert np.allclose(F[:, 2], [1, 0, 0]) and np.allclose(pk.frame_axis, [1, 0, 0])


def test_a_declared_frame_the_walk_contradicts_is_refused_at_build():
    """The bundle runs along x; a frame that says z is refused with the angle, a frame that says x is written,
    and a sphere walk (no dominant axis) declares nothing to contradict."""
    from dmipy_sim.replay.bank import frame_from_axis, check_frame_against_walk
    g = d.Cylinder(radius=1e-6, orientation=(1, 0, 0))                    # intra-axonal: free along x, restricted across
    walk = d.simulate_trajectories(400, 2e-9, g, 6e-3, 5e-4, seed=2, require_gpu=False)
    with pytest.raises(ValueError, match="principal displacement axis"):
        build_replay_pack(walk, id="t/wrong", license="x", citation="x", K=4, substrate_frame=np.eye(3))
    pk = build_replay_pack(walk, id="t/right", license="x", citation="x", K=4, substrate_frame=frame_from_axis((1, 0, 0)))
    assert np.allclose(pk.frame_axis, [1, 0, 0])
    rng = np.random.default_rng(0)
    iso = np.cumsum(rng.normal(size=(300, 20, 3)), axis=1) * 1e-7
    assert check_frame_against_walk(iso, np.eye(3)) == 0.0


@pytest.mark.skipif(not __import__("os").path.isdir("/home/rutger/dmrai-ws/CACTUS/prod/cactus_bundle_00000"),
                    reason="the CACTUS production run is not on this machine")
def test_the_cactus_run_declares_its_bundle_along_x():
    from dmipy_sim.spec import cactus_spec
    from dmipy_sim.replay.bank import frame_of_spec
    spec = cactus_spec("/home/rutger/dmrai-ws/CACTUS/prod/cactus_bundle_00000")
    F = frame_of_spec(spec)
    assert np.degrees(np.arccos(abs(F[:, 2] @ [1, 0, 0]))) < 2.0
    assert len(spec.realisation["bundles"]) == 1
