"""One gradient phase, one gate, one MT walk.

Every route from a stored walk to a signal integrates `gamma dt sum_t G(t) . r(t)` through
`dmipy_sim.replay._replay_kernel`. Synthetic trajectories with a closed-form phase -- a stationary
walker, a constant velocity, a single sine mode -- go through each route and must return that
phase to 1e-8 of the waveform's first moment; the pack routes return the phase of the path they store (the
chord plus the bands' series, which at block order 0 holds the sine mode at ``sinc(pi / 2N)`` of its
amplitude) to the float32 their coefficients are stored in. The
field gate has one implementation, the kernel's, which every field route reads, and the MT
trajectory producer at zero binding is the plain producer to the bit.
"""
import dmipy_replay.replay.compression as _r_compression
import numpy as np
from dmipy_sim import RFEvent
import pytest

import dmipy_sim as d
from dmipy_sim.replay import trajectories as T
from dmipy_sim.replay import compression as cx
from dmipy_replay.replay._replay_kernel import effective_gradient, waveform_moments, gradient_phase, field_gate
from dmipy_replay.constants import GAMMA
from dmipy_replay.replay.replay import compile_scheme, replay_signal
from dmipy_sim.replay.replay import replay_signal_jax

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
    """(name, traj (n_w, n_t, 3), closed-form phase (n_w,), closed-form phase of the stored path (n_w,)) for the
    bipolar waveform of `_bipolar`: the exact integral of the sample-and-hold waveform against the piecewise-linear
    path through the saves, ``gamma dt sum_k g_k (r_k + r_{k+1}) / 2`` (the last sample sits at T and integrates to
    nothing), and against the path a pack stores of those saves (the same for the chord; the sine mode at
    ``sinc(pi / 2N)`` of its amplitude)."""
    t = np.arange(n_t) * dt
    g = _bipolar(n_t)[0, :, 0]
    mid = lambda r: (r[:, :-1] + r[:, 1:]) / 2.0                     # the path's mean over each save interval
    out = []
    x0 = np.array([1e-6, -2e-6, 3e-6])                               # stationary: gamma x0 int g = 0 (refocused)
    traj = np.zeros((3, n_t, 3)); traj[:, :, 0] = x0[:, None]
    phi = GAMMA * dt * x0 * g[:-1].sum()
    out.append(("stationary", traj, phi, phi))
    v = np.array([1e-3, -5e-4])                                      # constant velocity: gamma v int g(t) t dt
    traj = np.zeros((2, n_t, 3)); traj[:, :, 0] = v[:, None] * t[None, :]
    phi = GAMMA * dt * (g[None, :-1] * mid(traj[:, :, 0])).sum(1)
    out.append(("constant velocity", traj, phi, phi))
    A = np.array([2e-6, 7e-7])                                       # one sine mode through the saves
    mode = np.sin(np.pi * t / T_TOTAL)
    traj = np.zeros((2, n_t, 3)); traj[:, :, 0] = A[:, None] * mode[None, :]
    N = n_t - 1
    T_ = N * dt
    stored = GAMMA * A * np.sinc(1.0 / (2.0 * N)) * (g[:-1] * (T_ / np.pi) * (np.cos(np.pi * t[:-1] / T_) - np.cos(np.pi * t[1:] / T_))).sum()
    out.append(("sine mode", traj, GAMMA * dt * (g[None, :-1] * mid(traj[:, :, 0])).sum(1), stored))
    return out


def _order0_bands(X, K, h):
    """The bands of a bridge kept at its saves (``N + 1`` instants ``h`` apart, in axis 1) at block order 0: the
    orthonormal DST-I of the saved bridge times ``sqrt(h) sinc(pi k / 2N)`` (the cosine projection of the increments
    between saves at their midpoints). ``(n_w, K)`` for a scalar bridge, ``(n_w, K, 3)`` for a path."""
    from scipy.fft import dst
    N = X.shape[1] - 1
    tau = (np.arange(N + 1) / N).reshape((1, -1) + (1,) * (X.ndim - 2))
    u = X - (X[:, :1] + (X[:, -1:] - X[:, :1]) * tau)
    b = dst(u[:, 1:-1], type=1, norm="ortho", axis=1)[:, :K]
    f = np.sqrt(h) * np.sinc(np.arange(1, K + 1) / (2.0 * N))
    return b * (f[None, :, None] if b.ndim == 3 else f[None, :])


def _stored_path(traj, K, dt):
    """``(arrays, meta)`` of the pack a walk kept at its saves would hold: ``[x(0), x(T) - x(0), beta_1..K]`` per axis
    at float32, the window ``T = N dt``."""
    C = np.concatenate([traj[:, :1], (traj[:, -1] - traj[:, 0])[:, None], _order0_bands(traj, K, dt)], axis=1)
    return cx.pack_position_arrays(C), dict(method=_r_compression.POSITION_METHOD, K=K, T=(traj.shape[1] - 1) * dt)


def _phase_scale():
    """gamma dt * first moment of |G|: the size of a phase a micron of displacement makes."""
    g = _bipolar()[0, :, 0]
    return GAMMA * DT * np.abs(g).sum() * 1e-6


# ── every route returns the closed-form phase ─────────────────────────────────────────────
@pytest.mark.parametrize("case", range(3), ids=["stationary", "constant-velocity", "sine-mode"])
def test_every_route_returns_the_closed_form_phase(case):
    name, traj, phi_exact, phi_stored = _synthetic_trajectories()[case]
    G = _bipolar()
    tol = 1e-8 * _phase_scale()
    routes = {}
    # the kernel itself, and the numpy replay (return_walker_signals gives phi)
    routes["kernel"] = gradient_phase(effective_gradient(G, DT, N_T, DT), traj, DT)[0]
    routes["replay"] = T.replay(traj, DT, G, DT, return_walker_signals=True)[0][0]
    # the mode-space replay of the walk kept at its saves (K = n_t - 2)
    arrays, meta = _stored_path(traj, N_T - 2, DT)
    routes["mode space"] = cx.mode_space_phi(arrays, meta, G, DT)[:, 0]
    W = compile_scheme(G, DT, meta["K"])
    C = _r_compression.read_position_coeffs(arrays, dtype=np.float64)
    routes["compiled scheme"] = (C.reshape(C.shape[0], -1) @ W)[:, 0]
    # a pack stores its coefficients in float32, so the two pack routes carry that storage precision
    tol_pack = 1e-6 * max(_phase_scale(), np.abs(phi_exact).max())
    for route, phi in routes.items():
        pack_route = route in ("mode space", "compiled scheme")
        err = np.abs(np.asarray(phi) - (phi_stored if pack_route else phi_exact)).max()
        t = tol_pack if pack_route else tol
        assert err < t, f"{name} through {route}: |phi - exact| = {err:.3e} rad (tol {t:.1e})"


def test_the_vector_bloch_replays_return_the_closed_form_phase():
    pytest.importorskip("jax")
    _, traj, phi_exact, _ = _synthetic_trajectories()[2]
    G = _bipolar()
    # the vector-Bloch replays: a 90 at t = 0 and no relaxation leave Mxy = exp(+-i phi) (the sign
    # is the rotation convention; the magnitude of the phase is the physics)
    rf = [RFEvent(0.0, 90.0, axis_deg=90.0, duration_s=0.0)]
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
    W = compile_scheme(G, DT, K)
    B = _r_compression.band_projection(G, DT, T_TOTAL, K)
    np.testing.assert_allclose(W, (GAMMA * B).reshape(2, -1).T, rtol=0, atol=0)
    arrays, meta = _stored_path(_synthetic_trajectories()[2][1], K, DT)
    C = _r_compression.read_position_coeffs(arrays, dtype=np.float64)
    np.testing.assert_allclose(cx.mode_space_phi(arrays, meta, G, DT), C.reshape(C.shape[0], -1) @ W,
                               rtol=1e-12)


def test_replay_signal_jax_takes_the_host_surface_logweight():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from dmipy_replay.replay.replay import surface_logweight
    traj = _synthetic_trajectories()[2][1]
    arrays, meta = _stored_path(traj, 16, DT)
    n_w = traj.shape[0]
    dlog = -np.abs(np.random.default_rng(0).normal(0, 1e-6, (n_w, N_T)))
    B = np.cumsum(dlog, axis=1)                                       # the cumulative contact at the block ends
    a2, cm = cx.encode_boundary_bands(np.zeros(n_w), B[:, -1], _order0_bands(B, 16, DT), T=meta["T"])
    arrays = {**arrays, **a2, "spin_weights": np.ones(n_w)}
    W = compile_scheme(_bipolar(), DT, 16)
    slw = surface_logweight(arrays, 5e3, cm)
    E_np = replay_signal(arrays, W, rho2_over_D=5e3)
    E_jx = np.abs(np.asarray(replay_signal_jax(_r_compression.read_position_coeffs(arrays, dtype=np.float32),
                                               arrays["spin_weights"], W, surface_logw=slw)))
    np.testing.assert_allclose(E_jx, E_np, rtol=1e-5)
    assert E_np[0] < replay_signal(arrays, W)[0]                      # the knob bites


# ── one field gate ────────────────────────────────────────────────────────────────────────
def test_one_field_gate():
    """The bank holds no gate of its own; the kernel's field gate is the acquisition's effective gate on the save grid,
    so a spin echo refocuses a static field exactly with its 180 between two saves, and a gradient echo carries the
    path integral's own end weights."""
    from dmipy_sim.replay import bank
    assert not any(hasattr(bank, n) for n in ("se_gate", "_se_gate", "field_gate"))
    n_wf = 4 * (N_T - 1) + 1                                           # the waveform's grid: the 180 between two saves
    se = d.pgse([[1, 0, 0]], 2e-3, 1e-2, bvalues=[1e8], TE=T_TOTAL, n_t=n_wf, slew_rate=np.inf)
    s = field_gate(se, N_T, DT)
    assert s.sum() == pytest.approx(0.0, abs=1e-12)                    # a static field refocuses exactly
    np.testing.assert_allclose(s[1: N_T // 4], 1.0, atol=1e-12); np.testing.assert_allclose(s[-N_T // 4:-1], -1.0, atol=1e-12)
    g = field_gate(d.gre(T_TOTAL, n_t=n_wf), N_T, DT)
    np.testing.assert_allclose(g[1:-1], 1.0, atol=1e-12)
    assert g[0] == pytest.approx(0.5) and g[-1] == pytest.approx(0.5)


# ── the MT producer at zero binding is the plain producer ─────────────────────────────────
def test_mt_walk_at_zero_binding_is_the_plain_walk_to_the_bit():
    g = d.Sphere(2e-6, surface_relaxivity_t2=1e-6)
    kw = dict(seed=3, require_gpu=False)
    plain = d.simulate_trajectories(200, 2e-9, g, 2e-3, 5e-4, f_band=0.5 / 5e-4, raw_subset=200, **kw)
    mt = d.simulate_mt_trajectories(200, 2e-9, g, 2e-3, 5e-4, kappa_MT=0.0, dwell_time=0.0,
                                    equilibrate_binding="off", raw_subset=200, f_band=0.5 / 5e-4, **kw)
    assert mt.position_coeffs.dtype == np.float32 and mt.raw_positions.dtype == np.float32
    assert mt.sub_steps == plain.sub_steps, "the two producers must take the same sub-step count"
    np.testing.assert_array_equal(mt.position_coeffs, plain.position_coeffs)
    np.testing.assert_array_equal(mt.raw_positions, plain.raw_positions)
    np.testing.assert_array_equal(mt.boundary_bands, plain.boundary_bands)
    np.testing.assert_array_equal(mt.boundary_endpoint, plain.boundary_endpoint)
    np.testing.assert_array_equal(mt.raw_boundary, plain.raw_boundary)
    assert not mt.has_binding and mt.bound_start is None and mt.bound_events is None and mt.raw_bound is None


# ------------------------------------------------------------ one host kernel, any component subset (#200 item 9)
def test_the_compiled_scheme_is_generic_in_its_components_and_the_host_kernel_is_one_function(x64):
    """A consumer that reads two of the three stored axes compiles the waveform's matching components and gets
    exactly the rows of the full compile; ``replay_coefficients`` is what ``replay_signal`` evaluates; the
    traced batch twin equals the looped twin. Needs ``jax_enable_x64`` for the device/host parity check below
    (the ``x64`` fixture scopes it to this test and restores the flag on teardown)."""
    import dmipy_sim as d
    from dmipy_sim.replay import (compile_scheme, replay_signal, replay_coefficients, replay_signal_jax,
                                  replay_batch_jax, surface_logweight)
    from dmipy_sim.replay.bank import build_replay_pack
    from dmipy_replay.replay.compression import read_position_coeffs
    g = d.Cylinder(radius=2e-6, orientation=(0, 0, 1), surface_relaxivity_t2=0.0)
    walk = d.simulate_trajectories(300, 2e-9, g, 8e-3, 4e-4, seed=3, require_gpu=False)
    pk = build_replay_pack(walk, id="t/k", license="x", citation="x", K=6, blt_temporal_K=4)
    seq = d.set_b(d.pgse([[1, 0, 0], [0, 1, 1], [1, 1, 1]], 2e-3, 5e-3, gradient_strengths=0.1, n_t=pk.n_t, slew_rate=np.inf), [1e9] * 3)
    G = np.asarray(seq.G_eff)
    W3 = compile_scheme(G, seq.dt, pk.K, T=pk.T_window)
    W2 = compile_scheme(G[..., [0, 2]], seq.dt, pk.K, T=pk.T_window)
    Kw = pk.K + 2
    np.testing.assert_allclose(W2, W3.reshape(Kw, 3, -1)[:, [0, 2], :].reshape(2 * Kw, -1), atol=1e-12 * np.abs(W3).max())
    C = read_position_coeffs(pk.arrays, dtype=np.float64); w = np.asarray(pk.spin_weights, float)
    np.testing.assert_allclose(replay_coefficients(C, w, W3, complex_signal=True), replay_signal(pk, W3, complex_signal=True), rtol=1e-13)
    C2 = read_position_coeffs(pk.arrays, axes=[0, 2], dtype=np.float64)
    assert np.isfinite(replay_coefficients(C2, w, W2)).all() and C2.shape[2] == 2
    with pytest.raises(ValueError, match="rows"):
        replay_coefficients(C2, w, W3)
    cm = pk.meta["compression"]["channels"]["boundary_local_time"]
    slw = surface_logweight(pk.arrays, 5e3, cm)
    np.testing.assert_allclose(replay_coefficients(C, w, W3, surface_logw=slw), replay_signal(pk, W3, rho2_over_D=5e3), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(replay_signal_jax(C, w, W3, surface_logw=slw)),
                               replay_coefficients(C, w, W3, surface_logw=slw, complex_signal=True), atol=1e-10)
    Wb = np.stack([W3, 0.5 * W3, W2.repeat(1, axis=0)[:0] if False else W3 * 0.1])
    np.testing.assert_allclose(np.asarray(replay_batch_jax(C, w, Wb)), np.stack([replay_coefficients(C, w, Wk) for Wk in Wb]), atol=1e-10)
