"""The compressed producer and its replay: simulate_trajectories(compress=K) emits a master of the pack's own
coefficients (two exact endpoints and K sine bands of the bridge per axis, the cumulative local time in the same
form) instead of the raw trajectory, and replay() reads that master in coefficient space, never reconstructing
positions. Same seed as the raw walk, so the only difference is the band truncation -- checked below the MC floor.
"""
import numpy as np
import pytest

from dmipy_sim import simulate_trajectories, Cylinder, Box1D
from dmipy_sim.replay.trajectories import replay
from dmipy_sim.acquisition.waveforms import set_b
from dmipy_sim.sequences import pgse

N = 4000
MC = 1.0 / np.sqrt(N)          # Monte-Carlo floor


def _grad_battery(n_t):
    G = [np.asarray(set_b(pgse([bv], 0.02, 0.038, gradient_strengths=0.2, n_t=n_t), b).G[0])
         for b in (1e9, 2e9) for bv in ([1, 0, 0], [0, 0, 1])]
    return np.stack(G, 0)


def test_compressed_gradient_replay_matches_raw():
    D, R = 2e-9, 5e-6
    kw = dict(T_max=0.05, dt_save=1e-4, seed=1, require_gpu=False)
    raw = simulate_trajectories(N, D, Cylinder(radius=R, orientation=[0, 0, 1.]), **kw)
    mst = simulate_trajectories(N, D, Cylinder(radius=R, orientation=[0, 0, 1.]),
                                compress=32, **kw)
    assert isinstance(mst, dict) and mst["compressed"]
    traj, dt = np.asarray(raw.positions), raw.dt
    G = _grad_battery(traj.shape[1])
    S_raw = np.asarray(replay(traj, dt, G, dt))
    S_cmp = np.asarray(replay(mst, mst["dt_traj"], G, dt))
    assert np.abs(S_cmp - S_raw).max() < 2 * MC
    # host memory: modes are far smaller than the raw trajectory
    assert mst["pos_modes"].nbytes < 0.25 * traj.astype(np.float16).nbytes


def test_compressed_surface_replay_matches_raw():
    D, rho, R = 2e-9, 1e-6, 2e-6
    kw = dict(T_max=0.4, dt_save=2e-3, seed=7, require_gpu=False)
    raw = simulate_trajectories(N, D, Box1D(length=R), **kw)
    mst = simulate_trajectories(N, D, Box1D(length=R), compress=8, **kw)
    traj, dt, dlog = np.asarray(raw.positions), raw.dt, np.asarray(raw.boundary_local_time)
    G0 = np.zeros((1, traj.shape[1], 3))              # b0: pure surface-relaxivity decay
    S_raw = float(np.asarray(replay(traj, dt, G0, dt, surface_relaxivity=rho, D=D,
                                    dlog_boundary_unit=dlog))[0])
    S_cmp = float(np.asarray(replay(mst, mst["dt_traj"], G0, dt,
                                    surface_relaxivity=rho, D=D))[0])
    # ungated surface uses the stored endpoint B(T) -> exact (not just within the MC floor)
    assert abs(S_cmp - S_raw) < 1e-4


def test_the_compressed_master_is_the_codec_of_the_raw_walk():
    """compress=K encodes each batch on the device with the pack's own codec: the master equals
    encode_bridge_dst / encode_boundary_bridge of the raw walk of the same seed, computed on the host in float64,
    to float32 rounding -- one codec, no TF32."""
    from dmipy_sim.replay.compression import encode_bridge_dst, encode_boundary_bridge, read_position_coeffs
    D, R, K = 2e-9, 2e-6, 16
    kw = dict(T_max=0.02, dt_save=1e-4, seed=3, require_gpu=False)
    raw = simulate_trajectories(512, D, Box1D(length=R), **kw)
    mst = simulate_trajectories(512, D, Box1D(length=R), compress=K, **kw)
    arrays, _, _ = encode_bridge_dst(np.asarray(raw.positions, np.float32), K, device="numpy")
    C = read_position_coeffs(arrays, dtype=np.float64)
    scale = np.abs(C).max()
    assert np.abs(mst["pos_modes"] - C).max() <= 1e-5 * scale
    b, _ = encode_boundary_bridge(np.asarray(raw.boundary_local_time, np.float32), K, device="numpy")
    B_T = np.abs(b["blt_endpoint"]).max()
    assert np.abs(mst["blt_endpoint"] - b["blt_endpoint"]).max() <= 1e-5 * B_T
    assert np.abs(mst["blt_start"] - b["blt_start"]).max() <= 1e-5 * B_T
    assert np.abs(mst["blt_modes"] - b["blt_bridge_dst"]).max() <= 1e-5 * B_T


def test_compressed_susceptibility_is_rejected():
    D, R = 2e-9, 5e-6
    mst = simulate_trajectories(N, D, Cylinder(radius=R, orientation=[0, 0, 1.]),
                                T_max=0.02, dt_save=1e-4, seed=1, require_gpu=False,
                                compress=16)
    G = np.zeros((1, mst["n_t"], 3))
    with pytest.raises(NotImplementedError):
        replay(mst, mst["dt_traj"], G, mst["dt_traj"], susceptibility=object())
