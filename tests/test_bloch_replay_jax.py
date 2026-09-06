"""`replay_bloch_jax` is `replay_bloch` with every feature, to float32 rounding.

The two propagate one operator per step: RF sub-rotations, free precession, transverse wall
attenuation, relaxation. The numpy loop is the reference; the JAX scan takes the same arguments
and must agree on every path a sequence can exercise -- finite and shaped pulses with an
off-resonance carrier, slice-select, per-walker B1+, the MT bound-pool blend, susceptibility
phase, surface relaxivity, per-compartment relaxation, weights and per-walker echo readout.
"""
import numpy as np
import pytest

from dmipy_sim.trajectories import replay_bloch, replay_bloch_jax

DT = 1e-4
N_T, N_W = 120, 64


def _walk(seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(scale=np.sqrt(2 * 2e-9 * DT), size=(N_W, N_T, 3))
    return np.cumsum(steps, axis=1).astype(np.float32)


def _gradient(n_meas=2):
    G = np.zeros((n_meas, N_T, 3))
    G[0, 10:40, 0] = 0.03
    G[0, 70:100, 0] = 0.03                       # physical same-sign lobes: the 180 refocuses
    if n_meas > 1:
        G[1, 10:40, 2] = 0.05
        G[1, 70:100, 2] = 0.05
    return G


def _rf(finite=False):
    dur = 8 * DT if finite else 0.0
    return [{'t_s': 5 * DT, 'flip_deg': 90.0, 'axis_deg': 90.0, 'duration_s': dur,
             'offset_hz': 40.0 if finite else 0.0,
             'b1_envelope': [0.2, 1.0, 0.2] if finite else None},
            {'t_s': 55 * DT, 'flip_deg': 180.0, 'axis_deg': 0.0, 'duration_s': dur}]


def _close(a, b, tol=2e-5):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.abs(a - b).max() < tol, np.abs(a - b).max()


def test_hard_pulses_relaxation_and_echo_readout():
    traj, G = _walk(), _gradient()
    kw = dict(T2=40e-3, T1=0.8)
    _close(replay_bloch_jax(traj, DT, G, DT, _rf(), **kw), replay_bloch(traj, DT, G, DT, _rf(), **kw))
    echoes = [60, 110, 119]
    _close(replay_bloch_jax(traj, DT, G, DT, _rf(), echo_steps=echoes, **kw),
           replay_bloch(traj, DT, G, DT, _rf(), echo_steps=echoes, **kw))
    per_j = replay_bloch_jax(traj, DT, G, DT, _rf(), echo_steps=echoes, echo_per_walker=True, **kw)
    per_n = replay_bloch(traj, DT, G, DT, _rf(), echo_steps=echoes, echo_per_walker=True, **kw)
    assert per_j.shape == (2, 3, N_W)
    _close(per_j, per_n)
    Mj, sj = replay_bloch_jax(traj, DT, G, DT, _rf(), return_walker_signals=True, **kw)
    Mn, sn = replay_bloch(traj, DT, G, DT, _rf(), return_walker_signals=True, **kw)
    assert Mj.shape == (3, N_W)
    _close(Mj, Mn)
    _close(sj, sn)


def test_finite_shaped_pulses_with_carrier_slice_and_b1_scale():
    traj, G = _walk(1), _gradient()
    rng = np.random.default_rng(1)
    kw = dict(T2=40e-3, b1_scale=rng.uniform(0.8, 1.2, N_W), slice_offsets=rng.uniform(-2e-3, 2e-3, N_W),
              slice_gradient=5e-3)
    j = replay_bloch_jax(traj, DT, G, DT, _rf(finite=True), **kw)
    n = replay_bloch(traj, DT, G, DT, _rf(finite=True), **kw)
    _close(j, n)
    plain = replay_bloch(traj, DT, G, DT, _rf(finite=True), T2=40e-3)
    assert np.abs(n - plain).max() > 1e-3           # the features acted


def test_mt_blend_susceptibility_surface_compartments_and_weights():
    traj, G = _walk(2), _gradient()
    rng = np.random.default_rng(2)
    comp = (rng.uniform(size=(N_W, N_T)) < 0.4).astype(int)
    kw = dict(comp_traj=comp, T2_per_comp=[30e-3, 80e-3], T1_per_comp=[0.6, 1.2],
              bound_frac=rng.uniform(0, 0.3, (N_W, N_T)), T2_bound=12e-6, T1_bound=1.0,
              off_resonance_bound=250.0,
              extra_phase_per_step=rng.normal(scale=0.02, size=(N_W, N_T)),
              dlog_boundary_unit=-rng.exponential(2e-7, (N_W, N_T)), surface_relaxivity=5e-6, D=2e-9,
              weights=rng.uniform(0.5, 1.5, N_W))
    j = replay_bloch_jax(traj, DT, G, DT, _rf(), **kw)
    n = replay_bloch(traj, DT, G, DT, _rf(), **kw)
    _close(j, n)
    plain = replay_bloch(traj, DT, G, DT, _rf(), T2=40e-3)
    assert np.abs(n - plain).max() > 1e-3


def test_both_engines_refuse_an_incomplete_mt_or_surface_spec():
    traj, G = _walk(), _gradient(1)
    for fn in (replay_bloch, replay_bloch_jax):
        with pytest.raises(ValueError, match="T2_bound"):
            fn(traj, DT, G, DT, _rf(), bound_frac=np.zeros((N_W, N_T)))
        with pytest.raises(ValueError, match="D required"):
            fn(traj, DT, G, DT, _rf(), dlog_boundary_unit=np.zeros((N_W, N_T)), surface_relaxivity=1e-6)
        with pytest.raises(ValueError, match="comp_traj"):
            fn(traj, DT, G, DT, _rf(), T2_per_comp=[0.1, 0.2])


def test_measurement_batching_is_invisible_in_the_result():
    """A budget that fits one, two or all six phase tables gives the same signals and echoes."""
    traj, G = _walk(3), np.concatenate([_gradient()] * 3)
    one_table = 4 * N_T * N_W
    ref = replay_bloch(traj, DT, G, DT, _rf(), T2=40e-3, echo_steps=[60, 119], echo_per_walker=True)
    for budget in (1, 2 * one_table, 6 * one_table):
        out = replay_bloch_jax(traj, DT, G, DT, _rf(), T2=40e-3, echo_steps=[60, 119], echo_per_walker=True,
                               phase_table_bytes=budget)
        _close(out, ref)
