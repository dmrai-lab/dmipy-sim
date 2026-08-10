"""End-to-end IR-basis compression: producer streaming (piece 1) -> replay routing (piece 2).

simulate_trajectories(compress=K) emits a compressed master (DCT position modes + boundary
endpoint/modes) instead of the raw trajectory; replay() dispatches that master through
mode-space phase + boundary/relaxation weights, never reconstructing positions. Same seed as
the raw walk, so the only difference is DCT truncation -- checked below the MC floor.
"""
import numpy as np
import pytest

from dmipy_sim import simulate_trajectories, Cylinder, Box1D
from dmipy_sim.trajectories import replay
from dmipy_sim.waveforms import pgse, set_b

N = 4000
MC = 1.0 / np.sqrt(N)          # Monte-Carlo floor


def _grad_battery(n_t):
    G = [np.asarray(set_b(pgse(delta=0.02, DELTA=0.038, G_magnitude=0.2,
                               bvecs=[bv], n_t=n_t), b).G[0])
         for b in (1e9, 2e9) for bv in ([1, 0, 0], [0, 0, 1])]
    return np.stack(G, 0)


def test_compressed_gradient_replay_matches_raw():
    D, R = 2e-9, 5e-6
    kw = dict(T_max=0.05, dt_save=1e-4, seed=1, require_gpu=False)
    raw = simulate_trajectories(N, D, Cylinder(radius=R, orientation=[0, 0, 1.]), **kw)
    mst = simulate_trajectories(N, D, Cylinder(radius=R, orientation=[0, 0, 1.]),
                                compress=32, **kw)
    assert isinstance(mst, dict) and mst["compressed"]
    traj, dt = np.asarray(raw[0]), raw[1]
    G = _grad_battery(traj.shape[1])
    S_raw = np.asarray(replay(traj, dt, G, dt))
    S_cmp = np.asarray(replay(mst, mst["dt_traj"], G, dt))
    assert np.abs(S_cmp - S_raw).max() < 2 * MC
    # host memory: modes are far smaller than the raw trajectory
    assert mst["pos_modes"].nbytes < 0.25 * traj.astype(np.float16).nbytes


def test_compressed_surface_replay_matches_raw():
    D, rho, R = 2e-9, 1e-6, 2e-6
    kw = dict(T_max=0.4, dt_save=2e-3, seed=7, save_relaxation_data=True, require_gpu=False)
    raw = simulate_trajectories(N, D, Box1D(length=R), **kw)
    mst = simulate_trajectories(N, D, Box1D(length=R), compress=8, **kw)
    traj, dt, dlog = np.asarray(raw[0]), raw[1], np.asarray(raw[4])
    G0 = np.zeros((1, traj.shape[1], 3))              # b0: pure surface-relaxivity decay
    S_raw = float(np.asarray(replay(traj, dt, G0, dt, surface_relaxivity=rho, D=D,
                                    dlog_boundary_unit=dlog))[0])
    S_cmp = float(np.asarray(replay(mst, mst["dt_traj"], G0, dt,
                                    surface_relaxivity=rho, D=D))[0])
    # ungated surface uses the stored endpoint B(T) -> exact (not just within the MC floor)
    assert abs(S_cmp - S_raw) < 1e-4


def test_compressed_susceptibility_is_rejected():
    D, R = 2e-9, 5e-6
    mst = simulate_trajectories(N, D, Cylinder(radius=R, orientation=[0, 0, 1.]),
                                T_max=0.02, dt_save=1e-4, seed=1, require_gpu=False,
                                compress=16)
    G = np.zeros((1, mst["n_t"], 3))
    with pytest.raises(NotImplementedError):
        replay(mst, mst["dt_traj"], G, mst["dt_traj"], susceptibility=object())
