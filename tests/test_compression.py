"""IR-basis compression: codec correctness + replay fidelity.

Fast tests use synthetic arrays (exact algebraic identities). One slow test walks a real
substrate and checks the boundary-DCT codec reproduces the surface-relaxivity T2 -- the
property that lets the replay path store compressed modes instead of raw trajectories.
"""
import re

import numpy as np
import numpy.testing as npt
import pytest

from dmipy_sim.replay import compression as cx
from dmipy_sim.constants import GAMMA


# --------------------------------------------------------------- position codec
def _sinc_dst(u, K, h):
    """``sqrt(h) sinc(pi k / 2N)`` times the orthonormal DST-I of a bridge ``u`` kept at ``N + 1`` instants ``h``
    apart: the bands its order-0 reading gives."""
    from scipy.fft import dst
    N = u.shape[1] - 1
    k = np.arange(1, K + 1)
    f = np.sqrt(h) * np.sinc(k / (2.0 * N))
    b = dst(u[:, 1:-1], type=1, norm="ortho", axis=1)[:, :K]
    return b * (f[None, :, None] if b.ndim == 3 else f[None, :])


def test_the_position_codec_is_the_order_zero_reading_of_the_saves():
    """Both ends exact; the bands ``sqrt(h) sinc(pi k / 2N)`` times the DST-I of the saved bridge (the cosine
    projection of the increments at the saves' midpoints); decoded at the saves, the series of those bands."""
    rng = np.random.default_rng(0)
    X = np.cumsum(rng.standard_normal((50, 200, 3)) * 1e-7, axis=1)
    h = 1e-4
    arrays, meta, _ = cx.encode_bridge_dst(X, 198, h)
    C = cx.read_position_coeffs(arrays, dtype=np.float64)
    tau = np.arange(200) / 199.0
    u = X - (X[:, :1] + (X[:, -1:] - X[:, :1]) * tau[None, :, None])
    npt.assert_allclose(C[:, 0], X[:, 0], rtol=1e-6)
    npt.assert_allclose(C[:, 1], X[:, -1] - X[:, 0], rtol=1e-5, atol=1e-12)
    assert np.abs(C[:, 2:] - _sinc_dst(u, 198, h)).max() <= 1e-6 * np.abs(C[:, 2:]).max()
    S, t = cx.saves_basis(200, 0, 200, 198, T=meta["T"])
    npt.assert_allclose(cx.decode(arrays, meta, n_t=200), C[:, None, 0] + C[:, None, 1] * t[None, :, None]
                        + np.einsum("wkc,kn->wnc", C[:, 2:], S), rtol=1e-12, atol=1e-18)


def test_mode_space_phi_is_the_held_waveform_against_the_stored_series():
    """phi in the stored basis == the integral of the held waveform against the chord plus the bands' series."""
    rng = np.random.default_rng(1)
    N, n_t, M, K = 40, 128, 6, 32
    X = np.cumsum(rng.standard_normal((N, n_t, 3)) * 1e-7, axis=1)
    G = rng.standard_normal((M, n_t, 3)) * 0.1
    dt = 1e-4
    T = (n_t - 1) * dt
    arrays, meta, _ = cx.encode_bridge_dst(X, K, dt)
    C = cx.read_position_coeffs(arrays, dtype=np.float64)
    sub = 64
    t = (np.arange((n_t - 1) * sub) + 0.5) * dt / sub
    k = np.arange(1, K + 1)
    r = (C[:, None, 0, :] + C[:, None, 1, :] * (t / T)[None, :, None]
         + np.einsum("wkc,kt->wtc", C[:, 2:, :], np.sqrt(2.0 / T) * np.sin(np.pi * np.outer(k, t) / T)))
    g = G[:, np.minimum((t / dt).astype(int), n_t - 1), :]
    phi_ref = GAMMA * (dt / sub) * np.einsum("mtc,wtc->wm", g, r)
    phi_mode = cx.mode_space_phi(arrays, meta, G, dt)
    assert np.abs(phi_mode - phi_ref).max() <= 1e-5 * max(np.abs(phi_ref).max(), 1e-12)
    S_mode = cx.mode_space_signal(arrays, meta, G, dt)
    npt.assert_allclose(S_mode, np.exp(1j * phi_mode).mean(0), atol=1e-12)


# --------------------------------------------------------------- boundary codec
def test_the_boundary_bridge_is_the_order_zero_reading_of_the_cumulative_contact():
    """The cumulative local time's two ends exact; its bands ``sqrt(h) sinc`` times the DST-I of its bridge; the
    contact decoded at the saves the increments of that series."""
    rng = np.random.default_rng(3)
    ell = -np.abs(rng.standard_normal((30, 150))) * 1e-6     # <=0 per-save local time
    h = 1e-4
    arrays, meta = cx.encode_boundary_bridge(ell, 148, h, dtype=np.float64)
    B = np.cumsum(ell, axis=1)
    tau = np.arange(150) / 149.0
    u = B - (B[:, :1] + (B[:, -1:] - B[:, :1]) * tau[None, :])
    npt.assert_allclose(arrays["blt_endpoint"], B[:, -1], rtol=1e-6)
    assert np.abs(arrays["blt_bridge_dst"] - _sinc_dst(u, 148, h)).max() <= 1e-9 * np.abs(u).max()
    S, t = cx.saves_basis(150, 0, 150, 148, T=meta["T"])
    B_series = arrays["blt_start"][:, None] + (arrays["blt_endpoint"] - arrays["blt_start"])[:, None] * t[None, :] \
        + arrays["blt_bridge_dst"] @ S
    ell_r = cx.decode_boundary_bridge(arrays, meta, n_t=150)
    npt.assert_allclose(np.cumsum(ell_r.astype(np.float64), axis=1), B_series,
                        rtol=1e-4, atol=1e-12)


def test_boundary_bridge_preserves_cumulative_at_low_K():
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
    arrays, meta = cx.encode_boundary_bridge(ell, 16, 1e-4)
    B_r = np.cumsum(cx.decode_boundary_bridge(arrays, meta, n_t=n_t), axis=1)
    # endpoint is stored exactly; the residual modes must reproduce B(t) at EVERY t (every TE
    # truncation of the surface weight), not just the ends
    rel = np.abs(B_r - B).max() / np.abs(B[:, -1]).max()
    assert rel < 5e-3, rel   # smoothness sanity; the rigorous per-TE claim is the slow test


# --------------------------------------------------------------- channel codecs
def test_compartment_rle_lossless_integer():
    rng = np.random.default_rng(5)
    comp = (rng.random((20, 300)) < 0.02).cumsum(1) % 3      # piecewise-constant-ish
    arrays, meta = cx.encode_occupancy({"comp": comp.astype(np.int16)})
    dec = cx.decode_occupancy(arrays, meta)["comp"]
    assert np.array_equal(dec.astype(np.int16), comp.astype(np.int16))


def test_bound_fraction_roundtrip_within_quant():
    rng = np.random.default_rng(6)
    b = np.clip(rng.random((15, 200)), 0, 1)
    arrays, meta = cx.encode_occupancy({"comp": np.zeros_like(b, np.int16), "bound": b}, Q=256)
    dec = cx.decode_occupancy(arrays, meta)["bound"]
    assert np.abs(dec - b).max() < 1.0 / 255 + 1e-6


# --------------------------------------------------------------- real-walk physics
@pytest.mark.slow
def test_boundary_bridge_replays_surface_signal_below_mc_floor():
    """A real surface-relaxivity walk: the boundary bridge codec at K=8 reproduces the
    surface survival E(TE) at EVERY echo-time truncation to below the Monte-Carlo floor
    (1/sqrt(N)) -- the property that lets replay store K+1 modes instead of the raw dlog."""
    from dmipy_sim import simulate_trajectories, Box1D
    D, rho2, N, R = 2e-9, 1e-6, 8000, 2e-6
    res = simulate_trajectories(N, D, Box1D(length=R), T_max=0.6, dt_save=3e-3, window_T=0.6, K=8,
                                seed=7, require_gpu=False, raw_subset=N).saves()
    dlog = np.asarray(res.boundary_local_time).astype(np.float64)
    n_t = dlog.shape[1]; ts = np.arange(n_t) * 0.6 / (n_t - 1)

    def survival(dl):                       # E(TE) for every truncation TE
        return np.exp((rho2 / D) * np.cumsum(dl, axis=1)).mean(0)

    E_raw = survival(dlog)
    arrays, meta = cx.encode_boundary_bridge(dlog, 8, 3e-3)
    E_codec = survival(cx.decode_boundary_bridge(arrays, meta, n_t=n_t))
    mc_floor = 1.0 / np.sqrt(N)
    assert np.abs(E_codec - E_raw).max() < mc_floor, np.abs(E_codec - E_raw).max()

    # fitted T2 matches Brownstein-Tarr theory (sanity that the walk is right)
    m = (ts > 0.05) & (E_raw > 1e-3)
    T2_raw = -1.0 / np.polyfit(ts[m], np.log(E_raw[m]), 1)[0]
    assert abs(T2_raw - R / (2 * rho2)) / (R / (2 * rho2)) < 0.05

    # storage: K+1 floats/walker vs raw dlog
    comp = (arrays["blt_bridge_dst"].nbytes + arrays["blt_start"].nbytes
            + arrays["blt_endpoint"].nbytes)
    assert comp < 0.2 * dlog.astype(np.float16).nbytes


def test_boundary_bridge_endpoints_are_exact_at_every_K_so_segments_chain():
    """C2's basis is chosen for exactness, not error: a truncated DCT-II residual does not vanish
    at t=T, so the reconstruction misses the stored total by percent at every K, and chaining
    segment endpoints inherits that drift. The sine form is identically zero at both ends."""
    from scipy.fft import dct, idct
    rng = np.random.default_rng(11)
    ell = np.abs(rng.standard_normal((200, 258))) * 1e-6
    B = np.cumsum(ell, axis=1)
    nt = B.shape[1]

    ep, ramp = B[:, -1], np.linspace(0.0, 1.0, nt)[None, :]
    for K in (4, 8, 16, 64):
        # f64 storage so the comparison is about the BASIS, not the container's rounding
        arrays, meta = cx.encode_boundary_bridge(ell, K, 1e-4, dtype=np.float64)
        Br = np.cumsum(cx.decode_boundary_bridge(arrays, meta, n_t=nt), axis=1)
        # exact up to the f32 endpoint container (both endpoints are always stored f32)
        e_bridge = np.abs(Br[:, -1] - ep).max() / ep.mean()
        npt.assert_allclose(Br[:, 0], B[:, 0], rtol=1e-5, atol=1e-18)

        # the retired cosine form on the same residual, same K, same endpoint float
        C = dct(B - ep[:, None] * ramp, type=2, norm="ortho", axis=1)[:, :K]
        B_dct = idct(C, type=2, norm="ortho", axis=1, n=nt) + ep[:, None] * ramp
        e_dct = np.abs(B_dct[:, -1] - ep).max() / ep.mean()

        assert e_bridge < 1e-5                       # zero basis error; only f32 rounding left
        assert e_dct > 100 * e_bridge                # the cosine form is nowhere near


def test_boundary_bridge_rejects_a_stale_cosine_pack_loudly():
    """The array name changed with the basis, so a stale pack cannot be silently decoded as if its
    cosine bands were sine bands (which would return a plausible but wrong attenuation): a pack
    storing its residual under the old name has no ``blt_bridge_dst`` key for this reader to find."""
    rng = np.random.default_rng(12)
    arrays, meta = cx.encode_boundary_bridge(np.abs(rng.standard_normal((10, 64))) * 1e-6, 8, 1e-4)
    stale = {"blt_dct_coeffs": arrays["blt_bridge_dst"], "blt_endpoint": arrays["blt_endpoint"]}
    with pytest.raises(KeyError):
        cx.decode_boundary_bridge(stale, meta, n_t=64)


def test_require_position_method_refuses_any_codec_but_the_one_it_reads():
    """No retired-codec table: an unknown codec name, whatever it is, is refused generically, naming
    the one codec this build reads -- the refusal does not enumerate what used to be written."""
    assert cx.require_position_method(cx.POSITION_METHOD) == cx.POSITION_METHOD
    for method in ("not_a_codec", "unknown", "some_future_codec", "bridge_dst_v2"):
        with pytest.raises(ValueError, match=re.escape(cx.POSITION_METHOD)):
            cx.require_position_method(method)
    with pytest.raises(ValueError, match="no position codec"):
        cx.require_position_method(None)
    assert not hasattr(cx, "_RETIRED") and not hasattr(cx, "_RETIRED_BOUNDARY")


def test_c1_carries_the_mt_bound_pool_as_a_column_on_an_independent_axis():
    """MT needs no channel of its own: binding is an occupancy on an axis independent of the
    geometric compartment, and C1 already stores occupancy. A walker can be intra-axonal AND
    bound, so the two are separate columns whose product is the joint simplex."""
    rng = np.random.default_rng(21)
    n_w, n_t = 40, 300
    comp = (np.cumsum(rng.random((n_w, n_t)) < 0.01, axis=1) % 3).astype(np.int16)
    bound = np.repeat(rng.random((n_w, n_t // 10)) < 0.3, 10, axis=1).astype(np.float64)

    arrays, meta = cx.encode_occupancy({"comp": comp, "bound": bound})
    got = cx.decode_occupancy(arrays, meta)
    assert sorted(got) == ["bound", "comp"]
    npt.assert_array_equal(got["comp"], comp)                    # label axis is lossless
    assert np.abs(got["bound"] - bound).max() <= 1.0 / 255       # fraction axis to the quant step

    # the two axes are genuinely independent -- both states co-occur, which a single label
    # (or a single fraction) could not represent
    assert ((comp == 1) & (bound > 0.5)).any() and ((comp == 2) & (bound > 0.5)).any()

    # run lengths on the bound column ARE the dwell times, so C1 keeps the realised exchange
    # statistics rather than a summarising mean rate
    d = next(c for c in meta["columns"] if c["name"] == "bound")
    assert d["kind"] == "fraction" and arrays["bound_rle_lens"].max() >= 10

    # and C1 refuses to exist without its exclusive geometric axis
    with pytest.raises(ValueError, match="needs the 'comp' column"):
        cx.encode_occupancy({"bound": bound})


def test_c1_rejects_a_pre_columns_pack_loudly():
    """A pre-columns pack keeps `comp_rle_*` under its original name and with unchanged bytes, so
    nothing stops it being READ -- only its metadata says what the track means. That is exactly the
    case a bare KeyError would turn into a puzzle, so the refusal has to name the layout change."""
    rng = np.random.default_rng(31)
    comp = (np.cumsum(rng.random((12, 80)) < 0.05, axis=1) % 3).astype(np.int16)
    arrays, meta = cx.encode_occupancy({"comp": comp})

    # the metadata a pack written before the fold carries: one track, flags at the top level
    for stale in ({"channel": "compartment", "fractional": False, "n_t": 80},
                  {"channel": "compartment", "fractional": True, "Q": 256, "scale": 1.0,
                   "n_t": 80}):
        assert not cx.is_current_c1(stale)
        with pytest.raises(ValueError, match="predates the occupancy-column layout"):
            cx.decode_occupancy(arrays, stale)          # arrays are fine; the DECLARATION is stale

    assert cx.is_current_c1(meta)
    npt.assert_array_equal(cx.decode_occupancy(arrays, meta)["comp"], comp)

    # a bound column cannot sneak in under the stale shape either -- it has nowhere to be declared
    b = (rng.random((12, 80)) < 0.3).astype(np.float64)
    a2, m2 = cx.encode_occupancy({"comp": comp, "bound": b})
    assert "bound_rle_vals" in a2 and "bfrac_rle_vals" not in a2
    with pytest.raises(ValueError, match="predates the occupancy-column layout"):
        cx.decode_occupancy(a2, {"channel": "compartment", "fractional": False, "n_t": 80})
