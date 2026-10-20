"""The bridge position codec: both ends exact, the moment columns, the order-0 reading of saves, and the duality with
DCT."""
import re

import numpy as np
import numpy.testing as npt
import pytest
from scipy.fft import dct, idct, dst, idst

from dmipy_sim.replay import compression as cx
from dmipy_sim.constants import GAMMA
from dmipy_sim.replay.replay import compile_scheme

N_W, N_T, N_M = 60, 128, 8
DT = 1e-4


@pytest.fixture(scope="module")
def walk():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal((N_W, N_T, 3)) * 1e-7, axis=1)
    return x + rng.uniform(-2e-5, 2e-5, (N_W, 1, 3))


def _dct_truncate(x, K):
    """Reference band truncation, computed here so no retired codec has to stay in the library."""
    c = dct(x, type=2, norm="ortho", axis=1).copy()
    c[:, K:] = 0
    return idct(c, type=2, norm="ortho", axis=1)


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


def test_the_saves_band_count_is_their_count(walk):
    """Two endpoints plus n_t - 2 interior bands is exactly n_t coefficients -- no overhead -- and a walk kept at its
    saves holds no more bands than that."""
    a, m, _ = cx.encode_bridge_dst(walk, N_T + 10, DT)
    n_stored = np.asarray(a["pos_x"]).shape[1]
    assert n_stored == N_T and m["K"] == N_T - 2, f"{n_stored} coefficients per axis for {N_T} samples"


def test_the_saves_read_back_at_the_saves_are_the_order_zero_series(walk):
    """Decoded at the saves, the full band count is the inverse DST-I of the saved bridge's DST-I times
    ``sinc(pi k / 2N)``: the series of the order-0 reading, not an interpolation of the saves."""
    a, m, _ = cx.encode_bridge_dst(walk, N_T - 2, DT)
    tau = np.arange(N_T) / (N_T - 1.0)
    chord = walk[:, :1] + (walk[:, -1:] - walk[:, :1]) * tau[None, :, None]
    u = walk - chord
    k = np.arange(1, N_T - 1)
    b = dst(u[:, 1:-1], type=1, norm="ortho", axis=1) * np.sinc(k / (2.0 * (N_T - 1)))[None, :, None]
    ref = chord.copy(); ref[:, 1:-1] += idst(b, type=1, norm="ortho", axis=1)
    npt.assert_allclose(cx.decode(a, m, n_t=N_T), ref, rtol=0, atol=1e-11)


def test_endpoints_are_exact_at_every_truncation(walk):
    """The property a plain band codec (e.g. a bare DCT truncation) cannot offer: a stored walk is
    continuable from where it ended."""
    for K in (4, 8, 16, 32):
        a, m, _ = cx.encode_bridge_dst(walk, K, DT)
        r = cx.decode(a, m, n_t=N_T)
        npt.assert_allclose(r[:, 0, :], walk[:, 0, :], atol=1e-11)
        npt.assert_allclose(r[:, -1, :], walk[:, -1, :], atol=1e-11)
        # a band codec's endpoint error is the size of its interior error, not smaller
        rd = _dct_truncate(walk, K)
        assert np.abs(rd[:, -1, :] - walk[:, -1, :]).max() > 1e-9


def test_moment_nulled_waveform_annihilates_the_first_two_columns(walk):
    """Refocusing is M0 = int G dt = 0 and velocity compensation M1 = int G t dt = 0, so both columns vanish. The
    moments are those of the sample-and-hold waveform (the last sample sits at T and integrates to nothing), and
    the waveform is nulled inside the sine span so it still vanishes at both ends."""
    from dmipy_sim.replay._replay_kernel import effective_gradient
    n_t = N_T
    t = np.arange(n_t) / (n_t - 1)
    rng = np.random.default_rng(3)
    S = np.stack([np.sin(np.pi * k * t) for k in range(1, 9)], axis=1)
    mid = t[:-1] + 0.5 / (n_t - 1)                                      # each sample's interval midpoint
    F = np.stack([S[:-1].sum(0), (S[:-1] * mid[:, None]).sum(0)])       # int S dt, int S t dt  (up to dt)
    c = rng.standard_normal(S.shape[1])
    c = c - F.T @ np.linalg.solve(F @ F.T, F @ c)                       # null both moments within the span
    p = S @ c
    G = 0.05 * p[:, None] * np.array([0.6, 0.5, 0.62])[None, :]
    P = cx.band_projection(G[None], 1e-4, (n_t - 1) * 1e-4, 16)[0]                # (K + 2, 3): M0, M1, the bands
    assert np.abs(P[:2]).max() < 1e-10 * np.abs(P[2:]).max()

    W = compile_scheme(G[None], 1e-4, 16, method="bridge_dst")
    # the moment rows must vanish against the scale of the bands they sit beside
    assert np.abs(W[:6]).max() < 1e-8 * np.abs(W[6:]).max()


def test_compile_scheme_shape_and_agreement(walk):
    G, dt, K = _deliverable(), 1e-4, 16
    a, m, _ = cx.encode_bridge_dst(walk, K, dt)
    W = compile_scheme(G, dt, K, method="bridge_dst")
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


def test_accuracy_matches_a_band_truncation_at_equal_budget(walk):
    """The duality says neither basis should win; this pins that it does not lose either."""
    G, dt = _deliverable(), 1e-4
    from dmipy_sim.replay._replay_kernel import effective_gradient
    Geff = effective_gradient(G, dt, N_T, dt)
    phi = lambda r: (GAMMA * dt) * np.einsum("mtd,ntd->nm", Geff, r)
    ref = np.exp(1j * phi(walk)).mean(0)
    for K in (8, 16, 32):
        ab, mb, _ = cx.encode_bridge_dst(walk, K - 2, dt)      # same coefficient budget
        eb = np.abs(np.exp(1j * cx.mode_space_phi(ab, mb, G, dt)).mean(0) - ref).max()
        ed = np.abs(np.exp(1j * phi(_dct_truncate(walk, K))).mean(0) - ref).max()
        assert eb < 5 * max(ed, 1e-12), f"K={K}: bridge {eb:.2e} vs bands {ed:.2e}"


# --- the ways this could go quietly wrong --------------------------------------------------

def test_unknown_codecs_are_refused_not_decoded(walk):
    """A pack declaring any codec but the one this build reads stores different quantities under the same
    tensor names, so it is refused generically, by name -- naming the one codec it reads."""
    a, m, _ = cx.encode_bridge_dst(walk, 16, DT)
    for bad in ("not_a_codec", "unknown", "some_future_codec", "bridge_dst_v2"):
        with pytest.raises(ValueError, match=re.escape(cx.POSITION_METHOD)):
            cx.decode(a, dict(m, method=bad), n_t=N_T)
        with pytest.raises(ValueError, match=re.escape(cx.POSITION_METHOD)):
            cx.mode_space_phi(a, dict(m, method=bad), _deliverable(), 1e-4)


def test_missing_method_is_refused_rather_than_assumed(walk):
    a, m, _ = cx.encode_bridge_dst(walk, 16, DT)
    with pytest.raises(ValueError, match="declares no position codec"):
        cx.mode_space_phi(a, {k: v for k, v in m.items() if k != "method"},
                          _deliverable(), 1e-4)


def test_stored_width_is_not_K(walk):
    """K+2 multiplied cleanly against a K+2-band scheme and meant nothing; it must not."""
    from dmipy_sim.replay.replay import ReplayPack, replay_signal
    K = 16
    a, m, _ = cx.encode_bridge_dst(walk, K, DT)
    pk = ReplayPack(a, {"compression": {"method": "bridge_dst", "K": K}})
    assert pk.K == K and pk.n_coeffs == K + 2
    G = _deliverable()
    good = compile_scheme(G, 1e-4, pk.K)
    replay_signal(pk, good)                                   # the correct call
    wrong = compile_scheme(G, 1e-4, pk.n_coeffs)              # passing the stored width
    with pytest.raises(ValueError, match="compiled scheme has"):
        replay_signal(pk, wrong)


def test_declared_and_stored_K_cannot_drift(walk):
    from dmipy_sim.replay.replay import ReplayPack
    a, m, _ = cx.encode_bridge_dst(walk, 16, DT)
    pk = ReplayPack(a, {"compression": {"method": "bridge_dst", "K": 8}})
    with pytest.raises(ValueError, match="declares K=8 but stores"):
        pk.K


def test_dct_coeffs_accessor_is_gone(walk):
    from dmipy_sim.replay.replay import ReplayPack
    a, m, _ = cx.encode_bridge_dst(walk, 16, DT)
    pk = ReplayPack(a, {"compression": {"method": "bridge_dst", "K": 16}})
    with pytest.raises(AttributeError, match="dct_coeffs is gone"):
        pk.dct_coeffs
