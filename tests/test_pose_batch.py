"""A batch of acquisitions on one pack is one pass over the walkers, and each response is the one the acquisition
gets alone (dmrai-lab/dmipy-sim#449 section 3)."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.replay import read_rpk
from dmipy_sim.replay.bank import build_replay_pack


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    g = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6)
    walk = d.simulate_trajectories(500, 2e-9, g, 10e-3, 5e-4, seed=2, require_gpu=False)
    p = tmp_path_factory.mktemp("pk") / "c.rpk"
    build_replay_pack(walk, id="t/c", license="x", citation="x", K=8, out_path=str(p))
    return read_rpk(str(p))


def _acquisitions():
    dirs = [[1, 0, 0], [0, 0, 1], [0.6, 0.8, 0.0]]
    a = sequences.pgse(dirs, 2e-3, 5e-3, gradient_strengths=[0.3, 0.3, 0.3], TE=10e-3)
    b = sequences.pgse(dirs, 2e-3, 5e-3, gradient_strengths=[0.2, 0.2, 0.2], TE=10e-3)      # another shell: other groups
    c = sequences.pgse([[0, 1, 0], [0, 0, 1], [0.0, 0.6, 0.8]], 2.5e-3, 5e-3, gradient_strengths=[0.3, 0.3, 0.3], TE=10e-3)
    return [a, b, c]


def test_a_batch_gives_each_acquisition_its_own_response(pack):
    acq = _acquisitions()
    batch = pack.pose_responses(acq, keep=(8, 0))
    for wf, resp in zip(acq, batch):
        one = pack.pose_response(wf, keep=(8, 0))
        assert resp.route == "closed" and one.route == "closed"
        assert (resp.lmax, resp.nmax) == (one.lmax, one.nmax)
        np.testing.assert_allclose(resp.coeffs, one.coeffs, rtol=1e-9, atol=1e-12)
        # the batch expands every member to the band of its worst member, so a member's tail bound can only tighten
        assert (np.asarray(resp.misfit) <= np.asarray(one.misfit) * (1 + 1e-9) + 1e-14).all()
        assert resp.n_bodies == one.n_bodies


def test_a_multi_axis_member_of_a_batch_takes_the_quadrature_alone(pack):
    a, b, _ = _acquisitions()
    tensor = sequences.pgse([[1, 0, 0]], 2e-3, 5e-3, gradient_strengths=[0.3], TE=10e-3)
    from dmipy_sim.acquisition.waveforms import rotate_waveform
    G = np.asarray(tensor.G).copy()
    G[:, :, 1] = 0.5 * G[:, :, 0] * np.linspace(0, 1, G.shape[1])[None, :]         # a second axis with another shape: rank 2
    tensor = tensor.replace(G=G) if hasattr(tensor, "replace") else tensor
    if not np.linalg.matrix_rank(np.asarray(tensor.G)[0]) > 1:
        pytest.skip("could not build a multi-axis measurement on this sequence type")
    batch = pack.pose_responses([a, tensor, b], keep=(6, 0))
    assert batch[0].route == "closed" and batch[2].route == "closed"
    assert batch[1].route == "quadrature"
    np.testing.assert_allclose(batch[1].coeffs, pack.pose_response(tensor, keep=(6, 0)).coeffs, rtol=1e-9, atol=1e-12)
