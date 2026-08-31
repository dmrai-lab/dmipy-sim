"""The bridge+DST position codec: exactness, the moment columns, and the duality with DCT."""
import numpy as np
import numpy.testing as npt
import pytest
from scipy.fft import dct, idct, dst, idst

from dmipy_sim import compression as cx
from dmipy_sim.constants import GAMMA
from dmipy_sim.replay import compile_scheme

N_W, N_T, N_M = 60, 128, 8


@pytest.fixture(scope="module")
def walk():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal((N_W, N_T, 3)) * 1e-7, axis=1)
    return x + rng.uniform(-2e-5, 2e-5, (N_W, 1, 3))


def _deliverable(n_meas=N_M, n_t=N_T, seed=1):
    """Slew-limited waveforms that vanish at both ends, as a real gradient system must."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_t) / (n_t - 1)
    G = []
    for _ in range(n_meas):
        p = sum(rng.standard_normal() * np.sin(np.pi * k * t) for k in range(1, 7))
        d = rng.standard_normal(3); d /= np.linalg.norm(d)
        G.append(0.05 * p[:, None] * d[None, :])
    return np.asarray(G)


def test_representation_is_exactly_rank_preserving(walk):
    """Two endpoints plus n_t-2 interior bands is exactly n_t coefficients -- no overhead."""
    assert cx.rank_of("bridge_dst", N_T) == N_T - 2
    a, m, _ = cx.encode_bridge_dst(walk, N_T - 2)
    n_stored = np.asarray(a["pos_x"]).shape[1]
    assert n_stored == N_T, f"{n_stored} coefficients per axis for {N_T} samples"
    assert cx.is_lossless_at("bridge_dst", N_T - 2, N_T)


def test_full_rank_roundtrip_is_exact(walk):
    a, m, _ = cx.encode_bridge_dst(walk, N_T - 2)
    npt.assert_allclose(cx.decode(a, m), walk, atol=1e-11)


def test_endpoints_are_exact_at_every_truncation(walk):
    """The property temporal_dct cannot offer: a stored walk is continuable from where it ended."""
    for K in (4, 8, 16, 32):
        a, m, _ = cx.encode_bridge_dst(walk, K)
        r = cx.decode(a, m)
        npt.assert_allclose(r[:, 0, :], walk[:, 0, :], atol=1e-11)
        npt.assert_allclose(r[:, -1, :], walk[:, -1, :], atol=1e-11)
        # the DCT codec's endpoint error is the same size as its interior error
        ad, md, _ = cx.encode_temporal_dct(walk, K)
        rd = cx.decode(ad, md)
        assert np.abs(rd[:, -1, :] - walk[:, -1, :]).max() > 1e-9


def test_mode_space_phase_equals_the_raw_sum(walk):
    G, dt = _deliverable(), 1e-4
    raw = (GAMMA * dt) * np.einsum("mtd,ntd->nm", G, walk)
    a, m, _ = cx.encode_bridge_dst(walk, N_T - 2)
    # coefficients are stored float32, as they are for every position codec
    npt.assert_allclose(cx.mode_space_phi(a, m, G, dt), raw, rtol=1e-4, atol=1e-6)


def test_moment_nulled_waveform_annihilates_the_first_two_columns(walk):
    """Refocusing is M0 = 0 and velocity compensation M1 = 0, so both columns vanish."""
    n_t = N_T
    t = np.arange(n_t) / (n_t - 1)
    rng = np.random.default_rng(3)
    S = np.stack([np.sin(np.pi * k * t) for k in range(1, 9)], axis=1)
    A = np.stack([np.ones(n_t), t])                     # the two moment functionals
    c = rng.standard_normal(S.shape[1])
    p = S @ c
    p = p - A.T @ np.linalg.solve(A @ A.T, A @ p)       # null M0 and M1 exactly
    G = 0.05 * p[:, None] * np.array([0.6, 0.5, 0.62])[None, :]
    M0, M1 = cx.bridge_moment_rows(G[None], n_t)
    assert np.abs(M0).max() < 1e-10 and np.abs(M1).max() < 1e-10

    W = compile_scheme(G[None], 1e-4, 16, method="bridge_dst", n_t=n_t)
    # the moment rows must vanish against the scale of the bands they sit beside
    assert np.abs(W[:6]).max() < 1e-8 * np.abs(W[6:]).max()


def test_compile_scheme_shape_and_agreement(walk):
    G, dt, K = _deliverable(), 1e-4, 16
    a, m, _ = cx.encode_bridge_dst(walk, K)
    W = compile_scheme(G, dt, K, method="bridge_dst", n_t=N_T)
    assert W.shape == (3 * (K + 2), N_M)
    C = cx.read_position_coeffs(a, dtype=np.float64)
    npt.assert_allclose(C.reshape(N_W, -1) @ W, cx.mode_space_phi(a, m, G, dt), rtol=1e-9)


def test_difference_operator_maps_cosine_onto_sine():
    """diff(c_k) = -2 sin(pi k / 2N) s_{k-1}: why the two codecs truncate alike."""
    N = 256
    for k in (1, 5, 17):
        d = idct(np.eye(1, N, k)[0], type=2, norm="ortho")
        lam = dst(np.diff(d), type=1, norm="ortho")[k - 1]
        npt.assert_allclose(lam, -2 * np.sin(np.pi * k / (2 * N)), rtol=1e-9)


def test_sine_basis_is_the_bridge_karhunen_loeve_basis():
    """cov = min(m,n) - mn/N has DST-I eigenvectors; that is why the residual belongs there."""
    N = 64
    i = np.arange(1, N); j = i[:, None]
    cov = (np.minimum(j, i) - j * i / N).astype(float)
    S = np.stack([idst(np.eye(1, N - 1, k)[0], type=1, norm="ortho") for k in range(N - 1)]).T
    M = S.T @ cov @ S
    assert np.abs(M - np.diag(np.diag(M))).max() < 1e-10
    npt.assert_allclose(np.diag(M),
                        1.0 / (4 * np.sin(np.pi * np.arange(1, N) / (2 * N)) ** 2),
                        rtol=1e-10)


def test_accuracy_matches_temporal_dct_on_deliverable_waveforms(walk):
    """Equal-budget comparison: the duality says neither should win."""
    G, dt = _deliverable(), 1e-4
    ref = np.exp(1j * (GAMMA * dt) * np.einsum("mtd,ntd->nm", G, walk)).mean(0)
    for K in (8, 16, 32):
        ab, mb, _ = cx.encode_bridge_dst(walk, K - 2)          # same coefficient budget
        ad, md, _ = cx.encode_temporal_dct(walk, K)
        eb = np.abs(np.exp(1j * cx.mode_space_phi(ab, mb, G, dt)).mean(0) - ref).max()
        ed = np.abs(np.exp(1j * cx.mode_space_phi(ad, md, G, dt)).mean(0) - ref).max()
        assert eb < 5 * max(ed, 1e-12), f"K={K}: bridge {eb:.2e} vs dct {ed:.2e}"
