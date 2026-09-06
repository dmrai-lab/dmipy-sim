"""The host-side replay reads a pack once and never holds a whole float64 trajectory.

A `ReplayPack` stacks its per-axis coefficient tensors into one block on first use and hands
that block to `K`, `n_walkers`, `n_coeffs` and the signal contraction; `gradient_phase` runs the
phase contraction over walker chunks, so its working memory is bounded by the chunk and not by
the stored walk.
"""
import tracemalloc

import numpy as np
import pytest

from dmipy_sim.replay._replay_kernel import GAMMA, gradient_phase
from dmipy_sim.replay.compression import pack_position_arrays
from dmipy_sim.replay.replay import ReplayPack


def _pack(n_w=50, K=8):
    rng = np.random.default_rng(0)
    arrays = pack_position_arrays(rng.normal(size=(n_w, K + 2, 3)) * 1e-6)
    meta = {"n_t": 200, "dt": 1e-5, "compression": {"method": "bridge_dst", "K": K}}
    return ReplayPack(arrays, meta)


def test_pack_coefficients_are_stacked_once():
    pack = _pack()
    first = pack.position_coeffs
    assert pack.position_coeffs is first
    assert (pack.n_walkers, pack.n_coeffs, pack.K) == (50, 10, 8)
    assert pack.position_coeffs is first                     # the accessors read the same block
    np.testing.assert_array_equal(first[:, :, 1], pack.arrays["pos_y"])


def test_chunked_phase_matches_one_contraction():
    rng = np.random.default_rng(1)
    n_meas, n_t, n_w = 3, 40, 37
    G = rng.normal(size=(n_meas, n_t, 3))
    traj = rng.normal(size=(n_w, n_t, 3)).astype(np.float32)
    dt = 2e-5
    ref = (GAMMA * dt) * (G.reshape(n_meas, -1) @ traj.astype(np.float64).reshape(n_w, -1).T)
    whole = gradient_phase(G, traj, dt)
    row_bytes = 8 * n_t * 3
    for chunk in (row_bytes, 5 * row_bytes, 10 ** 9):          # one walker, five, everything
        out = gradient_phase(G, traj, dt, chunk_bytes=chunk)
        assert out.shape == (n_meas, n_w)
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=0)
        np.testing.assert_allclose(out, whole, rtol=1e-12, atol=0)


def test_phase_working_memory_is_the_chunk_not_the_walk():
    rng = np.random.default_rng(2)
    n_w, n_t = 2000, 500                                      # 12 MB float32, 24 MB as float64
    traj = rng.normal(size=(n_w, n_t, 3)).astype(np.float32)
    G = rng.normal(size=(2, n_t, 3))
    chunk = 2 * 2 ** 20
    tracemalloc.start()
    gradient_phase(G, traj, 1e-5, chunk_bytes=chunk)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 3 * chunk, f"peak {peak / 2**20:.1f} MB for a {chunk / 2**20:.0f} MB chunk"


def test_replay_of_a_stored_walk_agrees_with_the_direct_sum():
    """The public entry point rides on the chunked contraction: the same signal either way."""
    from dmipy_sim.replay.trajectories import replay
    rng = np.random.default_rng(3)
    n_w, n_t = 300, 60
    dt = 1e-4
    traj = np.cumsum(rng.normal(scale=np.sqrt(2 * 2e-9 * dt), size=(n_w, n_t, 3)), axis=1)
    G = np.zeros((2, n_t, 3))
    G[0, :20, 0] = 0.05
    G[0, 40:, 0] = -0.05
    G[1, :20, 2] = 0.08
    G[1, 40:, 2] = -0.08
    phi, _, sig = replay(traj, dt, G, dt, return_walker_signals=True)
    phi_ref = (GAMMA * dt) * np.einsum("mti,wti->mw", G, traj)
    np.testing.assert_allclose(phi, phi_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sig, np.cos(phi_ref).mean(1), rtol=1e-10, atol=1e-12)
