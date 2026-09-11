"""The pose expansion cached on disk (#200 item 5): the same pack under the same acquisition and knobs is expanded
once; anything that changes what the expansion depends on is a different entry; and the Bessel orders come from
one recurrence that matches scipy."""
import numpy as np
import pytest
from scipy.special import spherical_jn

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.phantom import Grid, PackSubstrate, Phantom, Watson
from dmipy_sim.replay import read_rpk
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.replay import PoseResponse, _spherical_jn_all


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    g = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6)
    walk = d.simulate_trajectories(300, 2e-9, g, 10e-3, 5e-4, seed=5, require_gpu=False)
    p = tmp_path_factory.mktemp("pk") / "c.rpk"
    build_replay_pack(walk, id="t/c", license="x", citation="x", K=8, out_path=str(p))
    return read_rpk(str(p))


def test_every_bessel_order_from_one_recurrence_matches_scipy():
    x = np.concatenate([[0.0, 1e-13, 1e-6, 0.1, 1.0], np.linspace(0.5, 60, 300)])
    J = _spherical_jn_all(40, x)
    ref = np.stack([spherical_jn(l, x) for l in range(41)])
    np.testing.assert_allclose(J, ref, atol=1e-13, rtol=1e-10)
    assert J.shape == (41, x.size) and J[0, 0] == 1.0 and J[1, 0] == 0.0


def test_the_cache_is_hit_only_by_the_same_expansion(pack, tmp_path, monkeypatch):
    seq = sequences.pgse([[1, 0, 0], [0, 0, 1]], 2e-3, 5e-3, gradient_strengths=[0.3, 0.3], TE=10e-3)
    cache = tmp_path / "pose"
    first = pack.pose_response(seq, keep=(8, 0), cache=cache)
    files = sorted(cache.glob("*.npz"))
    assert len(files) == 1 and first.route == "closed"
    # a hit: the computation is not run again and the numbers are the stored ones
    monkeypatch.setattr(pack, "_pose_coeffs_closed", lambda *a, **k: (_ for _ in ()).throw(AssertionError("recomputed")))
    second = pack.pose_response(seq, keep=(8, 0), cache=cache)
    np.testing.assert_array_equal(second.coeffs, first.coeffs)
    assert (second.lmax, second.nmax, second.n_bodies, second.route) == (first.lmax, first.nmax, first.n_bodies, first.route)
    monkeypatch.undo()
    # every dependency changes the key: the band, a knob, a direction, the method
    pack.pose_response(seq, keep=(6, 0), cache=cache)
    pack.pose_response(seq, keep=(8, 0), T2=[0.05, 0.05, 0.05], cache=cache)
    pack.pose_response(seq.with_gradient(np.asarray(seq.G)[:, :, [1, 0, 2]]), keep=(8, 0), cache=cache)
    assert len(sorted(cache.glob("*.npz"))) == 4
    # cache=True goes to the environment's directory
    monkeypatch.setenv("DMIPY_SIM_CACHE", str(tmp_path / "env"))
    pack.pose_response(seq, keep=(8, 0), cache=True)
    assert len(sorted((tmp_path / "env" / "pose").glob("*.npz"))) == 1
    # round trip of the object itself
    p = tmp_path / "one.npz"; first.save(p); back = PoseResponse.load(p)
    np.testing.assert_array_equal(back.coeffs, first.coeffs); assert back.route == "closed" and back.n_bodies == first.n_bodies


def test_a_phantom_replayed_twice_pays_the_expansion_once(pack, tmp_path):
    grid = Grid(shape=(2, 2, 1), voxel_size_m=(1e-3,) * 3)
    mu = np.zeros((2, 2, 1, 3)); mu[..., 2] = 1.0
    wm = PackSubstrate(pack, m0=1.0)
    ph = Phantom.compose(grid, fractions={wm: np.ones((2, 2, 1))}, orientation=Watson(mu=mu, kappa=8.0))
    seq = sequences.pgse([[1, 0, 0], [0, 0, 1]], 2e-3, 5e-3, gradient_strengths=[0.3, 0.3], TE=10e-3)
    S1 = ph.replay(seq, cache=tmp_path / "c")
    assert len(list((tmp_path / "c").glob("*.npz"))) == 1
    S2 = ph.replay(seq, cache=tmp_path / "c")
    np.testing.assert_array_equal(np.nan_to_num(S1), np.nan_to_num(S2))
    np.testing.assert_allclose(np.nan_to_num(S1), np.nan_to_num(ph.replay(seq)), atol=1e-12)
