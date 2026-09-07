"""One gradient phase, one gate, one MT walk.

Every route from a stored walk to a signal integrates `gamma dt sum_t G(t) . r(t)` through
`dmipy_sim.replay._replay_kernel`. Synthetic trajectories with a closed-form phase -- a stationary
walker, a constant velocity, a single sine mode -- go through each route and must return that
phase to 1e-8 of the waveform's first moment (the pack routes to the float32 their coefficients
are stored in). The
spin-echo gate has one implementation shared by the bank and the SH responder, and the MT
trajectory producer at zero binding is the plain producer to the bit.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.replay import trajectories as T
from dmipy_sim.replay import compression as cx
from dmipy_sim.replay._replay_kernel import (effective_gradient, waveform_moments, gradient_phase, phase_increments,
                                      se_gate)
from dmipy_sim.constants import GAMMA
from dmipy_sim.replay.replay import compile_scheme, replay_signal, replay_signal_jax

N_T, DT = 200, 1e-4
T_TOTAL = (N_T - 1) * DT
G0 = 0.05


def _bipolar(n_t=N_T, G=G0, axis=0):
    """A refocused bipolar pair along `axis`: +G for the first quarter, -G for the third."""
    g = np.zeros((1, n_t, 3))
    q = n_t // 4
    g[0, :q, axis] = G
    g[0, 2 * q:3 * q, axis] = -G
    return g


def _synthetic_trajectories(n_t=N_T, dt=DT):
    """(name, traj (n_w, n_t, 3), closed-form phase (n_w,)) for the bipolar waveform of `_bipolar`: the exact
    integral of the sample-and-hold waveform against the piecewise-linear path through the saves,
    ``gamma dt sum_k g_k (r_k + r_{k+1}) / 2`` (the last sample sits at T and integrates to nothing)."""
    t = np.arange(n_t) * dt
    g = _bipolar(n_t)[0, :, 0]
    mid = lambda r: (r[:, :-1] + r[:, 1:]) / 2.0                     # the path's mean over each save interval
    out = []
    x0 = np.array([1e-6, -2e-6, 3e-6])                               # stationary: gamma x0 int g = 0 (refocused)
    traj = np.zeros((3, n_t, 3)); traj[:, :, 0] = x0[:, None]
    out.append(("stationary", traj, GAMMA * dt * x0 * g[:-1].sum()))
    v = np.array([1e-3, -5e-4])                                      # constant velocity: gamma v int g(t) t dt
    traj = np.zeros((2, n_t, 3)); traj[:, :, 0] = v[:, None] * t[None, :]
    out.append(("constant velocity", traj, GAMMA * dt * (g[None, :-1] * mid(traj[:, :, 0])).sum(1)))
    A = np.array([2e-6, 7e-7])                                       # one sine mode through the saves
    mode = np.sin(np.pi * t / T_TOTAL)
    traj = np.zeros((2, n_t, 3)); traj[:, :, 0] = A[:, None] * mode[None, :]
    out.append(("sine mode", traj, GAMMA * dt * (g[None, :-1] * mid(traj[:, :, 0])).sum(1)))
    return out


def _phase_scale():
    """gamma dt * first moment of |G|: the size of a phase a micron of displacement makes."""
    g = _bipolar()[0, :, 0]
    return GAMMA * DT * np.abs(g).sum() * 1e-6


# ── every route returns the closed-form phase ─────────────────────────────────────────────
@pytest.mark.parametrize("case", range(3), ids=["stationary", "constant-velocity", "sine-mode"])
def test_every_route_returns_the_closed_form_phase(case):
    name, traj, phi_exact = _synthetic_trajectories()[case]
    G = _bipolar()
    tol = 1e-8 * _phase_scale()
    routes = {}
    # the kernel itself, and the numpy replay (return_walker_signals gives phi)
    routes["kernel"] = gradient_phase(effective_gradient(G, DT, N_T, DT), traj, DT)[0]
    routes["replay"] = T.replay(traj, DT, G, DT, return_walker_signals=True)[0][0]
    # the pre-pulse azimuth with the cutoff at the end IS the full phase
    routes["pre_pulse"] = T.pre_pulse_gradient_phase(traj, DT, G, DT, N_T)[0]
    # per-step increments summed
    routes["increments"] = phase_increments(effective_gradient(G, DT, N_T, DT)[0], traj, DT).sum(0)
    # the mode-space replay of a losslessly encoded walk (K = n_t - 2)
    arrays, meta, _ = cx.encode_bridge_dst(traj, K=N_T - 2)
    routes["mode space"] = cx.mode_space_phi(arrays, meta, G, DT)[:, 0]
    W = compile_scheme(G, DT, meta["K"], n_t=N_T)
    C = cx.read_position_coeffs(arrays, dtype=np.float64)
    routes["compiled scheme"] = (C.reshape(C.shape[0], -1) @ W)[:, 0]
    # a pack stores its coefficients in float32, so the two pack routes carry that storage precision
    tol_pack = 1e-6 * max(_phase_scale(), np.abs(phi_exact).max())
    for route, phi in routes.items():
        err = np.abs(np.asarray(phi) - phi_exact).max()
        t = tol_pack if route in ("mode space", "compiled scheme") else tol
        assert err < t, f"{name} through {route}: |phi - exact| = {err:.3e} rad (tol {t:.1e})"


def test_jax_routes_agree_with_numpy():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from dmipy_sim.replay._replay_kernel import gradient_phase_jax, effective_gradient_jax, phase_increments_jax
    _, traj, phi_exact = _synthetic_trajectories()[2]
    G = _bipolar()
    # float32 arithmetic: a relative 1e-5 of the phase scale
    tol = 1e-5 * max(_phase_scale(), np.abs(phi_exact).max())
    phi_j = np.asarray(gradient_phase_jax(effective_gradient_jax(jnp.asarray(G), DT, N_T, DT),
                                          jnp.asarray(traj), DT))[0]
    assert np.abs(phi_j - phi_exact).max() < tol
    s_j = np.asarray(T.replay_jax(jnp.asarray(G), jnp.asarray(traj, dtype=jnp.float32), DT, DT))
    assert np.abs(s_j - np.mean(np.cos(phi_exact))).max() < tol
    inc = np.asarray(phase_increments_jax(effective_gradient_jax(jnp.asarray(G), DT, N_T, DT)[0], traj, DT)).sum(0)
    assert np.abs(inc - phi_exact).max() < tol
    # the vector-Bloch replays: a 90 at t = 0 and no relaxation leave Mxy = exp(+-i phi) (the sign
    # is the rotation convention; the magnitude of the phase is the physics)
    rf = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': 90.0, 'duration_s': 0.0}]
    M_final, s_mean = T.replay_bloch(traj, DT, G, DT, rf, return_walker_signals=True)   # (3, n_w)
    phi_bloch = np.angle(M_final[0] + 1j * M_final[1])
    assert np.abs(np.abs(phi_bloch) - np.abs(phi_exact)).max() < 1e-6
    sign = np.sign(np.sum(phi_bloch * phi_exact))
    s_jx = T.replay_bloch_jax(traj, DT, G, DT, rf)
    assert np.abs(s_jx - np.mean(np.exp(1j * sign * phi_exact))).max() < 1e-4
    assert np.abs(s_jx - s_mean).max() < 1e-4


def test_the_waveform_enters_through_its_exact_moments():
    """Same grid: the trapezoid weights, the last sample (at T) integrating to nothing. Other grid: the exact
    moments of the sample-and-hold waveform over the save intervals -- no interpolation of G, the integral
    conserved, zero outside, a waveform running past T refused."""
    G = _bipolar()
    W = effective_gradient(G, DT, N_T, DT)
    g = G[0, :, 0]
    np.testing.assert_allclose(W[0, 1:-1, 0], (g[1:-1] + g[:-2]) / 2)
    assert W[0, 0, 0] == pytest.approx(g[0] / 2) and W[0, -1, 0] == pytest.approx(g[-2] / 2)
    assert W.sum() * DT == pytest.approx(g[:-1].sum() * DT)                               # int G conserved
    with pytest.raises(ValueError, match="beyond the pack"):
        effective_gradient(G, DT, N_T // 2, DT)                                            # lobes past T_max
    padded = effective_gradient(G, DT, 2 * N_T, DT)
    assert padded.shape == (1, 2 * N_T, 3) and np.all(padded[:, N_T + 1:] == 0)
    fine = effective_gradient(G, DT, 4 * (N_T - 1) + 1, DT / 4)
    assert fine.shape == (1, 4 * (N_T - 1) + 1, 3)
    assert fine[0, 2:4 * (N_T // 4) - 1, 0] == pytest.approx(G0)                          # inside a lobe: constant
    assert fine.sum() * DT / 4 == pytest.approx(g[:-1].sum() * DT)
    # the moments against a brute-force quadrature of a waveform on a foreign grid with edges between saves
    rng = np.random.default_rng(3)
    Gf = rng.normal(size=(2, 37, 3)); dt_wf = 1e-5; n_t, dt_p = 20, 2.3e-5
    A0, A1 = waveform_moments(Gf, dt_wf, n_t, dt_p)
    tt = np.linspace(0, (n_t - 1) * dt_p, 400_001); wid = np.minimum(37 - 1, (tt / dt_wf).astype(int))
    Gf_t = np.where((tt < 37 * dt_wf)[None, :, None], Gf[:, wid], 0.0)             # zero past the waveform's end
    for k in range(n_t - 1):
        m = (tt >= k * dt_p) & (tt < (k + 1) * dt_p)
        np.testing.assert_allclose(A0[:, k], np.trapezoid(Gf_t[:, m], tt[m], axis=1), rtol=5e-3, atol=1e-8)
        np.testing.assert_allclose(A1[:, k], np.trapezoid(Gf_t[:, m] * (tt[m] - k * dt_p)[None, :, None], tt[m], axis=1), rtol=5e-3, atol=1e-12)


def test_compiled_scheme_and_mode_space_read_one_projection():
    G = np.concatenate([_bipolar(axis=0), _bipolar(axis=2)], axis=0)
    K = 12
    W = compile_scheme(G, DT, K, n_t=N_T)
    P = cx.bridge_projection(effective_gradient(G, DT, N_T, DT), N_T, K)
    np.testing.assert_allclose(W, (GAMMA * DT * P).reshape(2, -1).T, rtol=0, atol=0)
    arrays, meta, _ = cx.encode_bridge_dst(_synthetic_trajectories()[2][1], K=K)
    C = cx.read_position_coeffs(arrays, dtype=np.float64)
    np.testing.assert_allclose(cx.mode_space_phi(arrays, meta, G, DT), C.reshape(C.shape[0], -1) @ W,
                               rtol=1e-12)


def test_replay_signal_jax_takes_the_host_surface_logweight():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from dmipy_sim.replay.replay import surface_logweight
    traj = _synthetic_trajectories()[2][1]
    arrays, meta, _ = cx.encode_bridge_dst(traj, K=16)
    n_w = traj.shape[0]
    dlog = -np.abs(np.random.default_rng(0).normal(0, 1e-6, (n_w, N_T)))
    a2, cm = cx.encode_boundary_bridge(dlog, K=16)
    arrays = {**arrays, **a2, "spin_weights": np.ones(n_w)}
    W = compile_scheme(_bipolar(), DT, 16, n_t=N_T)
    slw = surface_logweight(arrays, 5e3, cm)
    E_np = replay_signal(arrays, W, rho_over_D=5e3)
    E_jx = np.abs(np.asarray(replay_signal_jax(cx.read_position_coeffs(arrays, dtype=np.float32),
                                               arrays["spin_weights"], W, surface_logw=slw)))
    np.testing.assert_allclose(E_jx, E_np, rtol=1e-5)
    assert E_np[0] < replay_signal(arrays, W)[0]                      # the knob bites
    with pytest.raises(TypeError):
        replay_signal_jax(cx.read_position_coeffs(arrays), arrays["spin_weights"], W, blt_dct=dlog)


# ── one spin-echo gate ────────────────────────────────────────────────────────────────────
def test_one_spin_echo_gate():
    from dmipy_sim.replay import bank
    assert bank.se_gate is se_gate and not hasattr(bank, "_se_gate")
    s = se_gate(N_T, DT, T_TOTAL / 2)                                  # a 180 between two saves
    assert s.sum() == pytest.approx(0.0, abs=1e-12)                    # a static field refocuses exactly
    assert (s[1: N_T // 4] == 1).all() and (s[-N_T // 4:-1] == -1).all()
    g = se_gate(N_T, DT, None)
    assert (g[1:-1] == 1).all() and g[0] == pytest.approx(0.5) and g[-1] == pytest.approx(0.5)   # the path integral's own end weights
    for tr in (0.3 * T_TOTAL, 0.5 * T_TOTAL, 0.61 * T_TOTAL):          # exact for ANY 180 time
        assert se_gate(N_T, DT, tr).sum() * DT == pytest.approx(2 * tr - T_TOTAL, abs=1e-12)


# ── the MT producer at zero binding is the plain producer ─────────────────────────────────
def test_mt_walk_at_zero_binding_is_the_plain_walk_to_the_bit():
    g = d.Sphere(2e-6, surface_relaxivity_t2=1e-6)
    kw = dict(seed=3, require_gpu=False)
    plain = d.simulate_trajectories(200, 2e-9, g, 2e-3, 5e-4, **kw)
    mt = d.simulate_mt_trajectories(200, 2e-9, g, 2e-3, 5e-4, kappa_MT=0.0, dwell_time=0.0,
                                    equilibrate_binding="off", **kw)
    assert mt.positions.dtype == np.float32 and mt.bound_frac.dtype == np.float32
    assert mt.sub_steps == plain.sub_steps, "the two producers must take the same sub-step count"
    np.testing.assert_array_equal(mt.positions, plain.positions)
    np.testing.assert_array_equal(mt.boundary_local_time, plain.boundary_local_time)
    assert (mt.bound_frac == 0).all()
    mt16 = d.simulate_mt_trajectories(50, 2e-9, g, 1e-3, 5e-4, kappa_MT=0.0, dwell_time=0.0,
                                      equilibrate_binding="off", storage_dtype=np.float16, **kw)
    assert mt16.positions.dtype == np.float16
