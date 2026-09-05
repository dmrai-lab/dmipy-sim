"""One sub-step dispatch for every driver.

`physics.resolve_sub_steps` is the maximum over the criteria that apply to a walk, and every
driver -- the fused scan, the trajectory producer, the vector-Bloch engine, the MT walker, the
packed-myelin kernel -- reports the count it took from it. The slow test at the end is the
consequence: fused and replay agree on a surface-relaxivity substrate at the auto count, where
before they were run at different resolutions.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.physics import (resolve_sub_steps, walk_sub_steps, surface_sub_steps,
                               permeable_sub_steps, collision_sub_steps, mt_sub_steps,
                               make_step_fn, make_packed_myelin_step_fn)

D = 2e-9
DT = 1e-4


def _sphere_rho():
    return d.Sphere(radius=2e-6, surface_relaxivity_t2=1e-6)


def _mesh_rho():
    from dmipy_sim import mesh_shapes
    V, F = mesh_shapes.icosphere(2e-6, subdivisions=2)
    return d.Mesh(V, F, feature_radius=0.4e-6, intra={"surface_relaxivity_t2": 1e-6})


def _packed_myelin():
    c, L, _ = d.pack_cylinders([1e-6] * 4, target_vf=0.3, seed=0)
    return d.PackedMyelinatedCylinders([1e-6] * 4, 0.7, c, L, N_max=8, rho_inner=1e-6)


# ── the composition rule ──────────────────────────────────────────────────────────────────
def test_resolve_is_the_maximum_of_the_applicable_criteria():
    g = _sphere_rho()
    assert resolve_sub_steps(g, D, DT) == walk_sub_steps(g, D, DT)
    assert resolve_sub_steps(g, D, DT, surface=True) == max(walk_sub_steps(g, D, DT),
                                                             surface_sub_steps(g, D, DT))
    assert resolve_sub_steps(g, D, DT, surface=True) > resolve_sub_steps(g, D, DT)

    k = d.Cylinder(radius=2e-6, orientation=(0, 0, 1), permeability=1e-5)
    assert resolve_sub_steps(k, D, DT) == permeable_sub_steps(k, D, DT)

    m = _mesh_rho()
    assert resolve_sub_steps(m, D, DT, surface=True) == collision_sub_steps(m, D, DT)
    assert surface_sub_steps(m, D, DT) == 1, "a feature radius is not a pore"

    assert resolve_sub_steps(g, D, DT, mt_dwell_time=1e-3) == max(
        walk_sub_steps(g, D, DT), mt_sub_steps(g, D, DT, 1e-3))

    assert resolve_sub_steps(d.FreeDiffusion(), D, DT, surface=True) == 1
    assert resolve_sub_steps(g, D, DT, surface=True, override=3) == 3


def test_disabling_the_surface_criterion_keeps_the_reflection_criterion():
    g = _sphere_rho()
    g.surface_substep_frac = 0.0
    assert resolve_sub_steps(g, D, DT, surface=True) == walk_sub_steps(g, D, DT) > 1


# ── every driver reports the dispatcher's count ──────────────────────────────────────────
@pytest.mark.parametrize("make", [_sphere_rho, lambda: d.Box1D(4e-6, surface_relaxivity_t2=1e-6),
                                  lambda: d.Cylinder(2e-6, (0, 0, 1), permeability=1e-5),
                                  lambda: d.Sphere(2e-6), _mesh_rho])
def test_fused_and_bloch_kernels_take_the_dispatched_count(make):
    from dmipy_sim.bloch import _make_bloch_step_fn
    g = make()
    rho = g.surface_relaxivity_t2 or 0.0
    want = resolve_sub_steps(g, D, DT, surface=rho > 0.0)
    step_fn, _ = make_step_fn(g, D, DT)
    assert step_fn.n_sub == want
    bloch_fn = _make_bloch_step_fn(g, D, DT, T2=None, T1=None, M0=1.0, off_resonance_hz=0.0, rho=rho)
    assert bloch_fn.n_sub == want
    # the override is honoured on every branch, including the permeable one (pinned finer than the
    # auto count, so the collision-lookup guard has nothing to say)
    assert make_step_fn(g, D, DT, sub_steps=want + 1)[0].n_sub == want + 1
    assert _make_bloch_step_fn(g, D, DT, T2=None, T1=None, M0=1.0, off_resonance_hz=0.0,
                               rho=rho, sub_steps=want + 1).n_sub == want + 1


def test_packed_myelin_kernel_takes_the_dispatched_count():
    g = _packed_myelin()
    D_ref = float(max(np.max(np.asarray(g._D_intra_jax)), np.max(np.asarray(g._D_extra_jax))))
    assert make_packed_myelin_step_fn(g, DT).n_sub == resolve_sub_steps(g, D_ref, DT, surface=True)
    g0 = d.PackedMyelinatedCylinders([1e-6] * 4, 0.7, g._centers_np[:4], g._L_float, N_max=8)
    assert make_packed_myelin_step_fn(g0, DT).n_sub == resolve_sub_steps(g0, D_ref, DT) > 1


def test_trajectory_producers_take_the_dispatched_count():
    g = _sphere_rho()
    T_max, dt_save = 5 * DT, DT
    _, _, n_plain, _ = d.simulate_trajectories(64, D, g, T_max, dt_save, seed=0, require_gpu=False)
    assert n_plain == resolve_sub_steps(g, D, dt_save)
    out = d.simulate_trajectories(64, D, g, T_max, dt_save, seed=0, save_relaxation_data=True,
                                  require_gpu=False)
    assert out[2] == resolve_sub_steps(g, D, dt_save, surface=True)
    assert out[2] > n_plain, "recording the boundary local time resolves the surface criterion"

    mt = d.simulate_mt_trajectories(64, D, d.Sphere(2e-6), T_max, dt_save, kappa_MT=1e-5,
                                    dwell_time=1e-3, seed=0, require_gpu=False,
                                    equilibrate_binding="off")
    assert mt[2] == resolve_sub_steps(d.Sphere(2e-6), D, dt_save, surface=True, mt_dwell_time=1e-3)


# ── the consequence: fused == replay at the auto count ───────────────────────────────────
@pytest.mark.slow
@pytest.mark.parametrize("make", [_sphere_rho, lambda: d.Box1D(4e-6, surface_relaxivity_t2=1e-6)])
def test_fused_and_replay_agree_at_the_auto_sub_step_count(make):
    g = make()
    wf = d.set_b(d.pgse(delta=4e-3, DELTA=12e-3, G_magnitude=0.1, bvecs=[[1, 0, 0]], n_t=160,
                        slew_rate=np.inf), 1e9)
    assert resolve_sub_steps(g, D, wf.dt, surface=True) > 1, "the test must exercise sub-stepping"
    N = 40_000
    s_fused = np.asarray(d.simulate(N, D, wf, g, seed=1, engine="fused", require_gpu=False)).ravel()
    s_replay = np.asarray(d.simulate(N, D, wf, g, seed=1, engine="replay", require_gpu=False)).ravel()
    tol = max(0.01, 3.0 / np.sqrt(N))
    assert abs(s_fused[0] - s_replay[0]) < tol, f"fused {s_fused[0]:.4f} vs replay {s_replay[0]:.4f}"
