"""IR-basis compression: codec correctness + replay fidelity.

Fast tests use synthetic arrays (exact algebraic identities). One slow test walks a real
substrate and checks the boundary-DCT codec reproduces the surface-relaxivity T2 -- the
property that lets the replay path store compressed modes instead of raw trajectories.
"""
import numpy as np
import pytest

from dmipy_sim import compression as cx
from dmipy_sim.constants import GAMMA


# --------------------------------------------------------------- position codec
def test_temporal_dct_full_K_is_exact_roundtrip():
    rng = np.random.default_rng(0)
    X = np.cumsum(rng.standard_normal((50, 200, 3)) * 1e-7, axis=1)   # smooth-ish paths
    arrays, meta, _ = cx.encode_temporal_dct(X, K=200)
    Xr = cx.decode(arrays, meta)
    assert np.allclose(Xr, X, atol=1e-6)


def test_mode_space_phi_equals_raw_at_full_K():
    """phi in the DCT basis == raw gamma*dt*sum_t G.r exactly (Parseval, orthonormal DCT)."""
    rng = np.random.default_rng(1)
    N, n_t, M = 40, 128, 6
    X = np.cumsum(rng.standard_normal((N, n_t, 3)) * 1e-7, axis=1)
    G = rng.standard_normal((M, n_t, 3)) * 0.1
    dt = 1e-4
    phi_raw = (GAMMA * dt) * np.einsum("mtd,ntd->nm", G, X)
    arrays, meta, _ = cx.encode_temporal_dct(X, K=n_t)
    phi_mode = cx.mode_space_phi(arrays, meta, G, dt)
    assert np.allclose(phi_mode, phi_raw, rtol=1e-6, atol=1e-6)


def test_mode_space_signal_matches_raw_reduction():
    rng = np.random.default_rng(2)
    N, n_t, M = 60, 100, 5
    X = np.cumsum(rng.standard_normal((N, n_t, 3)) * 1e-7, axis=1)
    G = rng.standard_normal((M, n_t, 3)) * 0.1
    dt = 1e-4
    S_raw = cx._replay_complex_np(X, dt, G)
    arrays, meta, _ = cx.encode_temporal_dct(X, K=n_t)
    S_mode = cx.mode_space_signal(arrays, meta, G, dt)
    assert np.allclose(S_mode, S_raw, atol=1e-9)


# --------------------------------------------------------------- boundary codec
def test_boundary_dct_full_K_recovers_per_save_ell():
    rng = np.random.default_rng(3)
    ell = -np.abs(rng.standard_normal((30, 150))) * 1e-6     # <=0 per-save local time
    arrays, meta = cx.encode_boundary_dct(ell, K=150)
    ell_r = cx.decode_boundary_dct(arrays, meta)
    assert np.allclose(ell_r, ell, atol=1e-9)


def test_boundary_dct_preserves_cumulative_at_low_K():
    """The cumulative B(t)=cumsum(ell) is smooth (an integral of a wall-contact density) ->
    few modes reproduce it -- that is what the surface log-weight sum_t ell depends on."""
    rng = np.random.default_rng(4)
    n_t = 400
    t = np.linspace(0.0, 1.0, n_t)[None, :]
    # smooth nonneg per-save contact density: a low-frequency positive profile per walker
    amp = rng.uniform(0.5, 1.5, (25, 1))
    ph = rng.uniform(0, 2 * np.pi, (25, 1))
    dens = amp * (1.0 + 0.4 * np.sin(2 * np.pi * t + ph)) * 1e-9
    ell = -dens
    B = np.cumsum(ell, axis=1)
    arrays, meta = cx.encode_boundary_dct(ell, K=16)
    B_r = np.cumsum(cx.decode_boundary_dct(arrays, meta), axis=1)
    # endpoint is stored exactly; the residual modes must reproduce B(t) at EVERY t (every TE
    # truncation of the surface weight), not just the ends
    rel = np.abs(B_r - B).max() / np.abs(B[:, -1]).max()
    assert rel < 5e-3, rel   # smoothness sanity; the rigorous per-TE claim is the slow test


# --------------------------------------------------------------- channel codecs
def test_compartment_rle_lossless_integer():
    rng = np.random.default_rng(5)
    comp = (rng.random((20, 300)) < 0.02).cumsum(1) % 3      # piecewise-constant-ish
    arrays, meta = cx.encode_compartment(comp.astype(np.int16))
    dec = cx.decode_compartment(arrays, meta)
    assert np.array_equal(dec.astype(np.int16), comp.astype(np.int16))


def test_bound_fraction_roundtrip_within_quant():
    rng = np.random.default_rng(6)
    b = np.clip(rng.random((15, 200)), 0, 1)
    arrays, meta = cx.encode_bound_fraction(b, Q=256)
    dec = cx.decode_bound_fraction(arrays, meta)
    assert np.abs(dec - b).max() < 1.0 / 255 + 1e-6


# --------------------------------------------------------------- real-walk physics
@pytest.mark.slow
def test_boundary_dct_replays_surface_signal_below_mc_floor():
    """A real surface-relaxivity walk: the detrend boundary-DCT codec at K=8 reproduces the
    surface survival E(TE) at EVERY echo-time truncation to below the Monte-Carlo floor
    (1/sqrt(N)) -- the property that lets replay store K+1 modes instead of the raw dlog."""
    from dmipy_sim import simulate_trajectories, Box1D
    D, rho, N, R = 2e-9, 1e-6, 8000, 2e-6
    res = simulate_trajectories(N, D, Box1D(length=R), T_max=0.6, dt_save=3e-3,
                                seed=7, save_relaxation_data=True, require_gpu=False)
    dlog = np.asarray(res[4]).astype(np.float64)
    n_t = dlog.shape[1]; ts = np.arange(n_t) * 0.6 / (n_t - 1)

    def survival(dl):                       # E(TE) for every truncation TE
        return np.exp((rho / D) * np.cumsum(dl, axis=1)).mean(0)

    E_raw = survival(dlog)
    arrays, meta = cx.encode_boundary_dct(dlog, K=8)
    E_codec = survival(cx.decode_boundary_dct(arrays, meta))
    mc_floor = 1.0 / np.sqrt(N)
    assert np.abs(E_codec - E_raw).max() < mc_floor, np.abs(E_codec - E_raw).max()

    # fitted T2 matches Brownstein-Tarr theory (sanity that the walk is right)
    m = (ts > 0.05) & (E_raw > 1e-3)
    T2_raw = -1.0 / np.polyfit(ts[m], np.log(E_raw[m]), 1)[0]
    assert abs(T2_raw - R / (2 * rho)) / (R / (2 * rho)) < 0.05

    # storage: K+1 floats/walker vs raw dlog
    comp = arrays["blt_dct_coeffs"].nbytes + arrays["blt_endpoint"].nbytes
    assert comp < 0.2 * dlog.astype(np.float16).nbytes
