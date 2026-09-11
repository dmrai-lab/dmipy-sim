"""``ReplayPack.prefix`` (#199): a walk re-encoded to a shorter echo time at the same bands per second, with its own
certificate; and the pack's temporal band as a frequency."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.replay import read_rpk
from dmipy_sim.replay.bank import build_replay_pack


@pytest.fixture(scope="module")
def parent(tmp_path_factory):
    g = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6)
    walk = d.simulate_trajectories(600, 2e-9, g, 20e-3, 2.5e-4, seed=7, require_gpu=False)      # 81 saves, 20 ms
    p = tmp_path_factory.mktemp("pk") / "parent.rpk"
    build_replay_pack(walk, id="test/parent", license="x", citation="x", K=16, out_path=str(p))
    return read_rpk(str(p))


def test_the_band_is_a_frequency(parent, tmp_path):
    assert parent.temporal_bandwidth_hz == pytest.approx(parent.K / (2 * 20e-3))
    assert parent.meta["compression"]["temporal_bandwidth_hz"] == pytest.approx(parent.temporal_bandwidth_hz)
    g = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6)
    walk = d.simulate_trajectories(100, 2e-9, g, 10e-3, 2.5e-4, seed=1, require_gpu=False)
    pk = build_replay_pack(walk, id="t", license="x", citation="x", temporal_bandwidth_hz=400.0)
    assert pk.K == 8 and pk.temporal_bandwidth_hz == pytest.approx(400.0)                         # K = 2 T f


def test_a_prefix_keeps_the_bands_per_second_and_replays_like_the_parent(parent, tmp_path):
    half = parent.prefix(10e-3, out_path=tmp_path / "half.rpk")
    assert half.n_t == 41 and half.K == 8 and half.temporal_bandwidth_hz == pytest.approx(parent.temporal_bandwidth_hz)
    assert half.meta["provenance"]["prefix"]["parent_digest"] == parent.digest and half.meta["provenance"]["prefix"]["TE_s"] == pytest.approx(10e-3)
    assert half.meta["fidelity"]["within_2x_floor"]
    np.testing.assert_array_equal(half.spin_weights, parent.spin_weights)
    np.testing.assert_allclose(half.substrate_frame, parent.substrate_frame)
    # the same acquisition, inside the prefix, on both packs: the same signal to the prefix's certified error
    seq = sequences.pgse([[1, 0, 0], [0, 0, 1]], 2e-3, 5e-3, gradient_strengths=[0.3, 0.3], TE=10e-3)
    S_parent = parent.replay(seq, complex_signal=True)
    S_half = half.replay(seq, complex_signal=True)
    np.testing.assert_allclose(S_half, S_parent, atol=3 * half.meta["fidelity"]["err_max"] + 1e-6)
    # with the compartment channel: relaxation replays on both
    np.testing.assert_allclose(half.replay(seq, T2=[0.02, 0.02, 0.02]), parent.replay(seq, T2=[0.02, 0.02, 0.02]),
                               atol=3 * half.meta["fidelity"]["err_max"] + 1e-6)
    # the start positions are the parent's, exactly
    np.testing.assert_allclose(half.r0, parent.r0, atol=1e-12)
    back = read_rpk(str(tmp_path / "half.rpk"))
    np.testing.assert_array_equal(back.position_coeffs, half.position_coeffs)


def test_a_prefix_outside_the_walk_or_below_the_floor_is_refused(parent):
    with pytest.raises(ValueError, match="not a prefix"):
        parent.prefix(30e-3)
    with pytest.raises(ValueError, match="not a prefix"):
        parent.prefix(1e-4)
    # forcing far too few bands fails the certificate rather than writing a pack
    with pytest.raises(ValueError, match="Monte-Carlo floor"):
        parent.prefix(10e-3, K=2, tol=0.01)


def test_a_short_acquisition_relaxes_to_its_own_echo_on_a_longer_walk(parent):
    """The TE-prefix property on the replay side: the walk beyond the echo is not part of the acquisition, so
    relaxation and surface terms stop there, whatever the pack's length."""
    seq = sequences.pgse([[1, 0, 0]], 2e-3, 5e-3, gradient_strengths=[0.3], TE=10e-3)
    S0 = parent.replay(seq, tissue=False)
    S = parent.replay(seq, T2=[0.02, 0.02, 0.02])
    assert S[0] / S0[0] == pytest.approx(np.exp(-10e-3 / 0.02), rel=2e-2)          # exp(-TE/T2), not exp(-T/T2)
