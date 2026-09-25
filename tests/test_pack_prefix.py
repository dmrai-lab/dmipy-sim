"""``ReplayPack.prefix`` (#199): a walk re-encoded to a shorter echo time at the same bands per second, with its own
certificate; and the pack's temporal band as a frequency."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.replay import read_rpk
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec.tissue import Tissue


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
    np.testing.assert_allclose(half.replay(seq, tissue=Tissue(T2=[0.02, 0.02, 0.02])), parent.replay(seq, tissue=Tissue(T2=[0.02, 0.02, 0.02])),
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
    # left to itself the band starts at the parent's bands per second and doubles until the certificate passes
    tight = parent.prefix(5e-3, tol=0.5)
    tried = tight.meta["provenance"]["prefix"]["K_tried"]
    assert tried[0] == 4 and tight.K == tried[-1] and tight.meta["fidelity"]["err_max"] <= 0.5 * tight.meta["fidelity"]["floor_max"]


def test_a_short_acquisition_relaxes_to_its_own_echo_on_a_longer_walk(parent):
    """The TE-prefix property on the replay side: the walk beyond the echo is not part of the acquisition, so
    relaxation and surface terms stop there, whatever the pack's length."""
    seq = sequences.pgse([[1, 0, 0]], 2e-3, 5e-3, gradient_strengths=[0.3], TE=10e-3)
    S0 = parent.replay(seq)
    S = parent.replay(seq, tissue=Tissue(T2=[0.02, 0.02, 0.02]))
    assert S[0] / S0[0] == pytest.approx(np.exp(-10e-3 / 0.02), rel=2e-2)          # exp(-TE/T2), not exp(-T/T2)


def test_a_prefix_of_a_field_pack_re_encodes_the_path_channel_and_certifies_it(tmp_path):
    """The path-field channel has no grid in the pack to re-sample from: the prefix re-encodes the parent's decoded
    series and certifies the result on the producer's battery (GRE, SE, the CPMG train it advertises)."""
    from tests.test_bank import _susc_master, _lean_env
    from dmipy_sim.replay.bank import susc_path_decode
    env = dict(_lean_env(), B0_list=[7.0], theta_deg=[0, 90])
    parent = build_replay_pack(_susc_master(), id="test/slab-susc", method="bridge_dst", envelope=env, K=64,
                               susc_path_K=32, license="CC-BY-4.0", citation="test")
    T = (parent.n_t - 1) * parent.dt
    half = parent.prefix(T / 2)
    pm = half.meta["compression"]["channels"]["susceptibility_path"]
    K_half = int(np.ceil(32 * (half.n_t - 1) / (parent.n_t - 1)))                # the same bands per second
    assert pm["K"] == K_half and pm["max_refocus_pulses"] == K_half // 2 and half.meta["replay_envelope"]["field"]
    assert half.meta["replay_envelope"]["acquisition"]["max_refocusing_pulses"] == K_half // 2
    f = half.meta["fidelity"]
    assert f["susc_path_pulses_certified"] == K_half // 2 and f["err_susc_path"] <= 2.0 * f["floor_susc_path"] + 1e-9
    assert f["err_max"] >= f["err_susc_path"] and f["within_2x_floor"]
    # the child's decoded series is the parent's decoded prefix, channel for channel, to the certified error
    b_parent, names = susc_path_decode(parent.arrays, parent.meta["compression"]["channels"]["susceptibility_path"])
    b_half, names_half = susc_path_decode(half.arrays, pm, n_w=half.n_walkers)
    assert names_half == names and b_half.shape == b_parent[:, :, :half.n_t].shape


def test_a_short_acquisition_dephases_in_the_field_to_its_own_echo(tmp_path):
    """The field route ends at the echo too: a spin echo or gradient echo shorter than the walk integrates the
    off-resonance under ITS gate, zero beyond its echo, so the parent and its prefix agree to the certificate."""
    from tests.test_bank import _susc_master, _lean_env
    from dmipy_sim.replay.bank import susc_path_decode, susc_path_field
    from dmipy_sim.constants import GAMMA
    env = dict(_lean_env(), B0_list=[7.0], theta_deg=[0, 90])
    parent = build_replay_pack(_susc_master(), id="test/slab-susc", method="bridge_dst", envelope=env, K=64,
                               susc_path_K=32, license="CC-BY-4.0", citation="test")
    T = (parent.n_t - 1) * parent.dt
    half = parent.prefix(T / 2)
    TE = (half.n_t - 1) * half.dt
    # a fine sequence grid: a sample is a block, so a sequence's last block spills half a sample past its echo,
    # which the longer parent integrates and the prefix cannot -- an O(dt_wf / TE) convention, not a gate
    se = sequences.pgse([[0, 0, 1]], 0.15 * TE, 0.5 * TE, gradient_strengths=[0.01], TE=TE, n_t=4000)
    gre = sequences.gre(TE, n_t=4000)
    for seq in (se, gre):
        S_p = parent.replay(seq, scanner=7.0, tissue=Tissue(chi_iso=1.06e-6), complex_signal=True)
        S_h = half.replay(seq, scanner=7.0, tissue=Tissue(chi_iso=1.06e-6), complex_signal=True)
        np.testing.assert_allclose(S_h, S_p, atol=3 * half.meta["fidelity"]["err_max"] + 5e-4)
    # the gradient echo by hand on the parent: the field over the first n_t' saves only, +1 throughout
    b, _ = susc_path_decode(parent.arrays, parent.meta["compression"]["channels"]["susceptibility_path"])
    dB = susc_path_field(b, (0.0, 0.0, 1.0), B0=7.0, chi_iso=1.06e-6)
    n_cut = half.n_t
    w = np.ones(n_cut); w[0] = w[-1] = 0.5                                        # the trapezoid over the prefix
    phi = GAMMA * parent.dt * (dB[:, :n_cut] * w[None, :]).sum(1)
    S_hand = np.exp(1j * phi).mean()
    S_p = parent.replay(gre, scanner=7.0, tissue=Tissue(chi_iso=1.06e-6), complex_signal=True)[0]
    assert abs(S_p - S_hand) < 5e-3
    phi_full = GAMMA * parent.dt * dB.sum(1)                                       # what the old gate integrated
    assert abs(S_p - np.exp(1j * phi_full).mean()) > 5e-2 or abs(S_hand - np.exp(1j * phi_full).mean()) < 5e-3


def test_a_prefix_is_one_window_whatever_the_parents_save_grid(tmp_path):
    """A parent declared as one window of its own duration, on a save grid that does not divide the storage rule's
    0.1 s, prefixed past 0.1 s: the prefix is one window of its own duration, not a refusal of the default windows
    (found by the paper's Swoop train on the 1 s grey-matter pack)."""
    g = d.Cylinder(radius=3e-6, orientation=(0.0, 0.0, 1.0))
    T, n_t = 0.3, 1302                                                  # dt = 0.3 / 1301: 0.1 s is not a whole number of saves
    walk = d.simulate_trajectories(200, 2e-9, g, T, T / (n_t - 1), seed=3, require_gpu=False)
    p = tmp_path / "parent.rpk"
    build_replay_pack(walk, id="test/one-window", license="x", citation="x", K=16, out_path=str(p), segment_T=T)
    parent = read_rpk(str(p))
    assert parent.n_segments == 1
    pre = parent.prefix(0.15, out_path=str(tmp_path / "prefix.rpk"))
    assert pre.n_segments == 1 and abs((pre.n_t - 1) * pre.dt - 0.15) < pre.dt


def test_a_pack_built_from_a_lazy_walk_is_the_pack_of_the_array(tmp_path):
    """The builder, its encoder, its certificate and its frame check read a LazyWalk per walker range and build the
    same pack, to the bit, as from the array."""
    from dmipy_sim.replay.compression import LazyWalk
    g = d.Cylinder(radius=3e-6, orientation=(0.0, 0.0, 1.0))
    walk = d.simulate_trajectories(300, 2e-9, g, 8e-3, 2.5e-4, seed=5, require_gpu=False)
    X = np.asarray(walk.positions)
    m = walk._bank_dict()
    a = build_replay_pack(dict(m, traj=X), id="t/array", license="x", citation="x", K=12, out_path=str(tmp_path / "a.rpk"))
    lazy = LazyWalk(lambda lo, hi: X[lo:hi], X.shape, chunk_bytes=1 << 14)
    b = build_replay_pack(dict(m, traj=lazy), id="t/lazy", license="x", citation="x", K=12, out_path=str(tmp_path / "b.rpk"))
    for k in ("pos_x", "pos_y", "pos_z"):
        assert np.array_equal(np.asarray(a.arrays[k]), np.asarray(b.arrays[k])), k
    assert a.meta["fidelity"]["err_max"] == pytest.approx(b.meta["fidelity"]["err_max"], rel=1e-9)
    assert a.meta["fidelity"]["floor_max"] == pytest.approx(b.meta["fidelity"]["floor_max"], rel=1e-9)
    with pytest.raises(TypeError, match="LazyWalk"):
        np.asarray(lazy)


def test_the_prefix_decoder_on_the_device_is_the_numpy_one_to_float32_rounding():
    from dmipy_sim.replay.compression import encode, _bridge_positions, read_position_coeffs
    from dmipy_sim.replay.compression import decode_prefix
    rng = np.random.default_rng(3)
    traj = np.cumsum(rng.normal(size=(400, 300, 3)).astype(np.float32) * np.float32(1e-7), axis=1)
    arrays, meta, _ = encode(traj, "bridge_dst", 24, device="numpy")
    C = read_position_coeffs(arrays, dtype=np.float64)
    for n_cut in (2, 150, 299, 300):
        ref = _bridge_positions(C, 300)[:, :n_cut, :]
        cpu = decode_prefix(C, 300, n_cut, device="numpy")
        dev = decode_prefix(C, 300, n_cut, device="jax", chunk_bytes=1 << 16)
        assert np.abs(cpu - ref).max() <= 1e-9 and np.abs(dev - ref).max() <= 2e-6 * np.abs(ref).max(), n_cut


def test_the_channels_decoded_at_the_prefix_are_the_whole_ones_cut():
    """The boundary bridge and the path series evaluated at the first saves only equal the whole decode cut."""
    from dmipy_sim.replay import compression as cx
    from dmipy_sim.replay.bank import susc_path_decode
    rng = np.random.default_rng(8)
    n_w, n_t = 300, 240
    dlog = -(rng.exponential(1e-3, size=(n_w, n_t)).astype(np.float32) * (rng.uniform(size=(n_w, n_t)) < 0.2))
    arrays, meta = cx.encode_boundary_bridge(dlog, K=12, device="numpy")
    whole = cx.decode_boundary_bridge(arrays, meta)
    for n_cut in (2, 100, 239, 240):
        part = cx.decode_boundary_bridge(arrays, meta, n_cut=n_cut)
        assert part.shape == (n_w, n_cut) and np.abs(part - whole[:, :n_cut]).max() <= 1e-6 * np.abs(whole).max()
    # the path series: a synthetic DCT-II coefficient block with the pack's reader convention
    K = 9; C = rng.normal(size=(n_w, 7, K))
    from scipy.fft import idct
    ref = idct(np.pad(C, ((0, 0), (0, 0), (0, n_t - K))), type=2, norm="ortho", axis=2)
    for n_cut in (1, 50, 239):
        k = np.arange(K)[:, None]; n = np.arange(n_cut)[None, :]
        D = np.sqrt(2.0 / n_t) * np.cos(np.pi * k * (2 * n + 1) / (2.0 * n_t)); D[0] = np.sqrt(1.0 / n_t)
        assert np.abs(np.einsum("wck,kn->wcn", C, D) - ref[:, :, :n_cut]).max() <= 1e-9


def test_a_prefix_inside_a_later_window_reads_across_the_windows(tmp_path):
    """A parent of three 0.1 s windows prefixed to 0.25 s: the lazy positions are the windows' decoded saves joined
    on their shared save, cut, and the prefix replays like the parent."""
    g = d.Cylinder(radius=3e-6, orientation=(0.0, 0.0, 1.0))
    dt = 0.1 / 200
    walk = d.simulate_trajectories(300, 2e-9, g, 0.3, dt, seed=9, require_gpu=False)
    p = tmp_path / "parent.rpk"
    build_replay_pack(walk, id="t/windows", license="x", citation="x", K=16, out_path=str(p))
    parent = read_rpk(str(p))
    assert parent.n_segments == 3
    pre = parent.prefix(0.25, out_path=str(tmp_path / "prefix.rpk"))
    assert pre.n_segments == 1 and abs((pre.n_t - 1) * pre.dt - 0.25) < dt
    seq = sequences.pgse([[1, 0, 0], [0, 0, 1]], 20e-3, 60e-3, gradient_strengths=[0.05, 0.05], TE=0.25)
    np.testing.assert_allclose(pre.replay(seq), parent.replay(seq), atol=3 * pre.meta["fidelity"]["floor_max"])
