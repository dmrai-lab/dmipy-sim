"""Phase-2 replay: vector-Bloch, susceptibility, and magnetization-transfer replay.

Validates the replay operators AGAINST the public forward engine
(``bloch.simulate_bloch``) and the analytic two-pool oracle (``mt.mt_z_spectrum``):

* (a) vector-Bloch replay vs the fused ``simulate_bloch`` forward on a spin echo with a
      gradient — emergent refocusing, noise-floor parity;
* (b) susceptibility replay vs ``simulate_bloch(susceptibility=<provider>)`` on a GRE
      (dephasing) and an SE (refocused) — same provider both paths;
* (c) MT replay Z-spectrum vs the analytic ``mt.mt_z_spectrum`` oracle.

Kept CPU-feasible (small N, coarse grids, short walks).  The whole module is auto-marked
``slow`` in ``conftest.py`` (heavy MC); the pure-function unit checks at the end are cheap.
"""
import numpy as np
from types import SimpleNamespace

import pytest

import dmipy_sim as d
from dmipy_sim import (simulate_bloch, simulate_trajectories, simulate_mt_trajectories,
                       replay_bloch, replay_bloch_jax,
                       finite_180_longitudinal_dwell, pathway_sign_se,
                       SusceptibilitySources, mt)
from dmipy_sim.trajectories import replay


# ── (a) vector-Bloch replay vs the forward engine: emergent SE refocusing ────────
def test_bloch_replay_matches_forward_spin_echo():
    """replay_bloch replays a spin echo (with a gradient) off one walk and
    matches the fused simulate_bloch forward to the noise floor.  Refocusing is
    emergent (the 180 conjugates the accumulated gradient phase)."""
    D, N, seed = 1.5e-9, 4000, 7
    TE, dt = 15e-3, 0.15e-3                     # dt chosen so the walk has sub_steps=1
    n_t = int(round(TE / dt)) + 1
    G = np.zeros((1, n_t, 3)); G[0, :, 0] = 0.02          # constant gradient (T/m)
    wf = SimpleNamespace(G=G, dt=dt)
    rf = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': 90.0, 'duration_s': 0.0},
          {'t_s': TE / 2, 'flip_deg': 180.0, 'axis_deg': 0.0, 'duration_s': 0.0}]
    geom, T2 = d.Sphere(radius=8e-6), 120e-3

    fwd = simulate_bloch(N, D, wf, geom, rf, T2=T2, seed=seed, require_gpu=False)
    traj, dt_tr, subs, _ = simulate_trajectories(N, D, geom, TE, dt, seed=seed,
                                                 require_gpu=False)
    assert subs == 1                                       # bit-identical walk to forward
    rep = replay_bloch(traj, dt_tr, G, dt, rf, T2=T2)
    assert abs(np.real(fwd[0]) - np.real(rep[0])) < 5e-3


# ── (b) susceptibility replay vs the forward engine (GRE dephases, SE refocuses) ──
def test_susceptibility_replay_gre_and_se():
    """Same susceptibility provider through the forward Bloch walk and the replay: the
    GRE dephases, the SE refocuses it, and replay reproduces both."""
    D, N, seed = 1.5e-9, 4000, 7
    TE, dt = 15e-3, 0.15e-3
    n_t = int(round(TE / dt)) + 1
    geom, T2 = d.Sphere(radius=8e-6), 120e-3
    traj, dt_tr, subs, _ = simulate_trajectories(N, D, geom, TE, dt, seed=seed,
                                                 require_gpu=False)
    assert subs == 1
    prov = SusceptibilitySources(centers=[[0, 0, 0]], radii=[3e-6],
                                 delta_chi=8e-6, B0=3.0)
    Z = np.zeros((1, n_t, 3)); wf0 = SimpleNamespace(G=Z, dt=dt)
    rf_gre = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': 90.0, 'duration_s': 0.0}]
    rf_se = rf_gre + [{'t_s': TE / 2, 'flip_deg': 180.0, 'axis_deg': 0.0,
                       'duration_s': 0.0}]

    f_gre = simulate_bloch(N, D, wf0, geom, rf_gre, T2=T2, seed=seed,
                           susceptibility=prov, require_gpu=False)
    r_gre = replay_bloch(traj, dt_tr, Z, dt, rf_gre, T2=T2, susceptibility=prov)
    f_se = simulate_bloch(N, D, wf0, geom, rf_se, T2=T2, seed=seed,
                          susceptibility=prov, require_gpu=False)
    r_se = replay_bloch(traj, dt_tr, Z, dt, rf_se, T2=T2, susceptibility=prov)

    # forward vs replay parity (same provider, bit-identical walk)
    assert abs(abs(f_gre[0]) - abs(r_gre[0])) < 8e-3
    assert abs(abs(f_se[0]) - abs(r_se[0])) < 8e-3
    # the static susceptibility field dephases the GRE; the SE 180 refocuses it
    assert abs(r_se[0]) > abs(r_gre[0]) + 0.02
    assert abs(f_se[0]) > abs(f_gre[0]) + 0.02


# ── (b') scalar-path susceptibility replay: eps_P refocuses the static field ─────
def test_scalar_susceptibility_replay_eps_p_refocus():
    """replay replays a susceptibility provider: with eps_P=None
    (FID / GRE) the static field dephases the signal, and the SE pathway sign eps_P
    (flip at TE/2) refocuses it."""
    D, N, seed = 1.5e-9, 4000, 7
    TE, dt = 15e-3, 0.15e-3
    n_t = int(round(TE / dt)) + 1
    geom = d.Sphere(radius=8e-6)
    traj, dt_tr, _, _ = simulate_trajectories(N, D, geom, TE, dt, seed=seed,
                                              require_gpu=False)
    prov = SusceptibilitySources(centers=[[0, 0, 0]], radii=[3e-6],
                                 delta_chi=8e-6, B0=3.0)
    Z = np.zeros((1, n_t, 3)); chi = np.ones((1, n_t))
    gre = replay(traj, dt_tr, Z, dt, chi_perp=chi,
                        susceptibility=prov, eps_P=None)
    eps = pathway_sign_se(n_t, dt, TE)[None, :]
    se = replay(traj, dt_tr, Z, dt, chi_perp=chi,
                       susceptibility=prov, eps_P=eps)
    assert se[0] > gre[0] + 0.1                    # SE refocuses the static dephasing
    assert abs(gre[0]) < 0.05                       # GRE strongly dephased


# ── (c) MT replay Z-spectrum vs the analytic two-pool oracle ─────────────────────
def test_mt_replay_zspectrum_matches_oracle():
    """Emergent MT Z-spectrum by replay (simulate_mt_trajectories + replay_bloch
    bound-pool blend) vs the analytic mt.mt_z_spectrum oracle.  RF rotates the bound spins
    too, so the saturation transfer is emergent; the broad short-T2b pool produces the dip."""
    D, N, seed, R = 1.5e-9, 2000, 3, 5e-6
    kappa_MT, dwell = 3.3e-5, 5.5e-3
    T2, T1, T2b, T1b = 80e-3, 1.0, 1e-5, 1.0
    w1_hz, t_sat, dt = 80.0, 10e-3, 40e-6
    offsets = np.array([-2000., -800., -250., 0., 250., 800., 2000.])
    geom = d.Sphere(radius=R)
    S_V = 3.0 / R
    k_f, k_r = kappa_MT * S_V, 1.0 / dwell

    oracle = mt.mt_z_spectrum(offsets, w1_hz=w1_hz, t_sat=t_sat, T1a=T1, T2a=T2,
                              T1b=T1b, T2b=T2b, k_f=k_f, k_r=k_r)
    traj, dt_tr, subs, _, bfrac, _ = simulate_mt_trajectories(
        N, D, geom, t_sat, dt, kappa_MT, dwell, seed=seed, require_gpu=False)
    # the equilibrated occupancy should track f_b = k_f/(k_f+k_r)
    assert abs(float(np.asarray(bfrac, float).mean()) - k_f / (k_f + k_r)) < 0.03

    Z = np.zeros((1, traj.shape[1], 3))
    flip = 360.0 * w1_hz * t_sat
    rep = np.empty_like(offsets)
    for i, off in enumerate(offsets):
        rf = [{'t_s': t_sat / 2, 'flip_deg': flip, 'axis_deg': 0.0,
               'duration_s': t_sat, 'offset_hz': float(off)}]
        Ml, _ = replay_bloch(traj, dt_tr, Z, dt, rf, T2=T2, T1=T1,
                                     bound_frac=bfrac, T2_bound=T2b, T1_bound=T1b,
                                     return_walker_signals=True)
        rep[i] = float(np.mean(Ml[2]))

    assert np.sqrt(np.mean((rep - oracle) ** 2)) < 0.03
    assert np.max(np.abs(rep - oracle)) < 0.06
    # MT contrast: an on-resonance saturation dip well below the far wings
    assert rep[3] < rep[0] - 0.3


# ── packed-myelin MT: simulate_trajectories returns the 7th bound_frac channel ───
def test_simulate_trajectories_packed_myelin_mt_channel():
    """kappa_MT == 0 keeps the 6-tuple bit-for-bit; kappa_MT > 0 adds a 7th per-save
    bound_frac channel in [0, 1] (RNG-preserved packed-myelin walk)."""
    geom_kw = dict(inner_radii=np.full(3, 2e-6), g_ratios=np.full(3, 0.7),
                   centers=np.array([[0., 0.], [12e-6, 0.], [-12e-6, 0.]]),
                   cell_size=40e-6, N_max=3, D_intra=2e-9, D_myelin=0.1e-9, D_extra=2e-9)
    D, N, T, dt, seed = 2e-9, 400, 8e-3, 0.4e-3, 1

    base = simulate_trajectories(N, D, d.PackedMyelinatedCylinders(**geom_kw), T, dt,
                                 seed=seed, save_relaxation_data=True, require_gpu=False)
    off = simulate_trajectories(N, D, d.PackedMyelinatedCylinders(**geom_kw), T, dt,
                                seed=seed, save_relaxation_data=True, require_gpu=False,
                                kappa_MT=0.0)
    assert len(base) == 6 and len(off) == 6
    assert np.array_equal(base[0], off[0])                      # positions bit-identical
    assert np.array_equal(base[4], off[4])                      # dlog bit-identical

    on = simulate_trajectories(N, D, d.PackedMyelinatedCylinders(**geom_kw), T, dt,
                               seed=seed, save_relaxation_data=True, require_gpu=False,
                               kappa_MT=5e-5, dwell_time=3e-3)
    assert len(on) == 7
    bf = np.asarray(on[6], dtype=float)
    assert bf.shape == base[0].shape[:2]                        # (n_walkers, n_t)
    assert bf.min() >= 0.0 and bf.max() <= 1.0 and bf.mean() > 0.0


# ── JAX vectorisation parity (hard pulses) ───────────────────────────────────────
def test_bloch_replay_jax_matches_numpy():
    """replay_bloch_jax reproduces the numpy primitive for hard pulses."""
    D, N, seed = 1.5e-9, 2000, 5
    TE, dt = 12e-3, 0.15e-3
    n_t = int(round(TE / dt)) + 1
    G = np.zeros((1, n_t, 3)); G[0, :, 0] = 0.015
    geom, T2 = d.Sphere(radius=8e-6), 100e-3
    traj, dt_tr, _, _ = simulate_trajectories(N, D, geom, TE, dt, seed=seed,
                                              require_gpu=False)
    rf = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': 90.0, 'duration_s': 0.0},
          {'t_s': TE / 2, 'flip_deg': 180.0, 'axis_deg': 0.0, 'duration_s': 0.0}]
    num = replay_bloch(traj, dt_tr, G, dt, rf, T2=T2)
    jx = replay_bloch_jax(traj, dt_tr, G, dt, rf, T2=T2)
    assert abs(np.real(num[0]) - np.real(jx[0])) < 2e-3


# ── pure-function unit checks (cheap) ────────────────────────────────────────────
def test_finite_180_longitudinal_dwell():
    """Coherent on-resonance spin never visits z; a uniformly dephased ensemble
    averages to tau_180/4."""
    tau = 2e-3
    assert finite_180_longitudinal_dwell(0.0, tau) == pytest.approx(0.0)
    phi = np.linspace(0, 2 * np.pi, 100_000, endpoint=False)
    assert finite_180_longitudinal_dwell(phi, tau).mean() == pytest.approx(tau / 4, rel=1e-3)


def test_pathway_sign_se():
    """SE pathway sign flips +1 -> -1 at TE/2."""
    n_t, dt, TE = 101, 1e-4, 10e-3
    eps = pathway_sign_se(n_t, dt, TE)
    t = np.arange(n_t) * dt
    assert np.all(eps[t < TE / 2] == 1.0)
    assert np.all(eps[t >= TE / 2] == -1.0)
