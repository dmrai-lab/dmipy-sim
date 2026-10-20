"""The replay-side codecs on synthetic arrays (exact algebraic identities): the waveform's projection on the
stored bands, the C1 events columns and the relaxation on their runs, the contact bridge's gated weight; one slow
test walks a real substrate and checks the contact bands reproduce the surface-relaxivity T2 at every echo time
-- the property that lets the replay store K bands instead of the raw contact.
"""
import re

import numpy as np
import numpy.testing as npt
import pytest

from dmipy_sim.replay import compression as cx
from dmipy_sim.coefficient_walk import EventList, events_at_instants
from dmipy_sim.constants import GAMMA


# --------------------------------------------------------------- position bands
def _bands(rng, n_w, K, T):
    """A random bridge in the stored form ``[x0, dx, beta_1..K]`` per axis, its bands falling as 1 / k."""
    C = np.zeros((n_w, K + 2, 3))
    C[:, 0] = rng.normal(0, 1e-6, (n_w, 3)); C[:, 1] = rng.normal(0, 1e-6, (n_w, 3))
    C[:, 2:] = rng.normal(0, 1, (n_w, K, 3)) * (1e-7 * np.sqrt(T) / np.arange(1, K + 1))[None, :, None]
    return C


def test_mode_space_phi_is_the_held_waveform_against_the_stored_bridge():
    """phi in the stored basis == the integral of the held waveform against the chord plus the bands' series."""
    rng = np.random.default_rng(1)
    N, n_t, M, K = 40, 128, 6, 32
    dt = 1e-4
    T = (n_t - 1) * dt
    C = _bands(rng, N, K, T)
    arrays = cx.pack_position_arrays(C, dtype=np.float64)
    meta = dict(method=cx.POSITION_METHOD, K=K, T=T)
    G = rng.standard_normal((M, n_t, 3)) * 0.1
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
    # the coded phases of the certificate are the same reading, on the host and the device
    phi = cx.coded_phases(C, dt, G, n_t, device="numpy")
    npt.assert_allclose(phi, phi_mode, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(cx.coded_phases(C, dt, G, n_t, device="jax"), phi, atol=1e-5 * max(np.abs(phi).max(), 1.0))


def test_the_bridge_read_at_instants_is_its_series():
    """``decode_bridge_dst`` reads the stored bridge at ``n_t`` instants of the window: the chord plus the sine series."""
    rng = np.random.default_rng(2)
    n_w, K, n_t, T = 20, 12, 41, 0.02
    C = _bands(rng, n_w, K, T)
    arrays = cx.pack_position_arrays(C, dtype=np.float64)
    t = np.linspace(0.0, T, n_t)
    k = np.arange(1, K + 1)
    ref = C[:, None, 0] + C[:, None, 1] * (t / T)[None, :, None] + np.einsum("wkc,kt->wtc", C[:, 2:], np.sqrt(2.0 / T) * np.sin(np.pi * np.outer(k, t) / T))
    npt.assert_allclose(cx.decode_bridge_dst(arrays, dict(K=K, T=T), n_t=n_t), ref, rtol=1e-10, atol=1e-18)


# --------------------------------------------------------------- the contact bridge
def test_the_gated_surface_weight_is_the_bridge_against_the_gate():
    """``surface_logweight_bridge``: the gate held sample by sample against the contact's continuous bridge, each
    piece's cosine integral in closed form -- against a dense quadrature of the same bridge; ungated, the exact total."""
    rng = np.random.default_rng(3)
    n_w, K, T, n_t = 30, 16, 0.012, 121
    dt = T / (n_t - 1)
    end = -np.abs(rng.normal(0, 1e-5, n_w)); bands = rng.normal(0, 1, (n_w, K)) * (1e-7 / np.arange(1, K + 1))[None, :]
    arrays, meta = cx.encode_boundary_bands(np.zeros(n_w, np.float32), end, bands, T=T, dtype=np.float64)
    chi = np.ones(n_t); chi[0] = 0.0; chi[80:] = 0.0
    sub = 200
    t = (np.arange((n_t - 1) * sub) + 0.5) * dt / sub
    k = np.arange(1, K + 1)
    dB = end[:, None] / T + np.einsum("wk,kt->wt", bands, np.sqrt(2.0 / T) * (np.pi * k / T)[:, None] * np.cos(np.pi * np.outer(k, t) / T))
    g = chi[np.minimum((t / dt).astype(int), n_t - 1)]
    ref = 3.0 * (dt / sub) * (dB * g[None, :]).sum(1)
    got = cx.surface_logweight_bridge(arrays, meta, 3.0, chi, dt)
    assert np.abs(got - ref).max() <= 1e-6 * np.abs(ref).max()
    npt.assert_allclose(cx.surface_logweight_bridge(arrays, meta, 3.0), 3.0 * end, rtol=1e-6)           # float32 endpoints


@pytest.mark.slow
def test_contact_bands_replay_surface_signal_below_mc_floor():
    """A real surface-relaxivity walk: the contact bands accumulated in the walk at K=8 reproduce the surface
    survival E(TE) at EVERY echo-time truncation to below the Monte-Carlo floor (1/sqrt(N)) -- the property that
    lets the replay store K bands instead of the raw contact."""
    from dmipy_sim import simulate_trajectories, Box1D
    D, rho2, N, R = 2e-9, 1e-6, 8000, 2e-6
    w = simulate_trajectories(N, D, Box1D(length=R), T_max=0.6, block_h=3e-3, window_T=0.6, K=8, seed=7, require_gpu=False, raw_subset=N)
    dlog = np.asarray(w.raw_boundary).astype(np.float64)
    n_t = dlog.shape[1]; ts = np.arange(n_t) * 0.6 / (n_t - 1)
    E_raw = np.exp((rho2 / D) * np.cumsum(dlog, axis=1)).mean(0)
    arrays, meta = cx.encode_boundary_bands(np.zeros(N, np.float32), np.asarray(w.boundary_endpoint)[:, 0], np.asarray(w.boundary_bands)[:, 0], T=0.6, dtype=np.float32)
    E_codec = np.array([np.exp(cx.surface_logweight_bridge(arrays, meta, rho2 / D, (np.arange(n_t) < j + 1).astype(float), 3e-3)).mean()
                        for j in range(n_t)])
    mc_floor = 1.0 / np.sqrt(N)
    assert np.abs(E_codec - E_raw).max() < mc_floor, np.abs(E_codec - E_raw).max()
    # fitted T2 matches Brownstein-Tarr theory (sanity that the walk is right)
    m = (ts > 0.05) & (E_raw > 1e-3)
    T2_raw = -1.0 / np.polyfit(ts[m], np.log(E_raw[m]), 1)[0]
    assert abs(T2_raw - R / (2 * rho2)) / (R / (2 * rho2)) < 0.05
    # storage: K + 2 floats per walker against the raw contact
    comp = arrays["blt_bridge_dst"].nbytes + arrays["blt_start"].nbytes + arrays["blt_endpoint"].nbytes
    assert comp < 0.2 * dlog.astype(np.float16).nbytes


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


# --------------------------------------------------------------- C1: events
def _events(rng, n_w, T, rate, n_labels=3):
    """Random label changes per walker at rate ``rate`` over ``T``: (start labels, EventList)."""
    start = rng.integers(0, n_labels, n_w)
    ts, labs, counts = [], [], []
    for i in range(n_w):
        n = rng.poisson(rate * T)
        t = np.sort(rng.uniform(0, T, n)); lab = start[i]
        out = []
        for tt in t:
            lab = (lab + rng.integers(1, n_labels)) % n_labels; out.append(lab)
        ts.append(t); labs.append(out); counts.append(n)
    off = np.concatenate([[0], np.cumsum(counts)])
    return start, EventList(off, np.concatenate(ts), np.concatenate(labs).astype(np.int8) if off[-1] else np.zeros(0, np.int8))


def test_c1_events_round_trip_losslessly():
    rng = np.random.default_rng(5)
    start, ev = _events(rng, 20, 0.1, 30.0)
    arrays, meta = cx.encode_events_occupancy({"comp": (start, ev)})
    assert meta["columns"] == [{"name": "comp", "kind": "events"}] and cx.is_current_c1(meta) and cx.has_c1(arrays)
    assert arrays["comp_events_offsets"].shape == (21,) and arrays["comp_events_t"].shape[0] == ev.n_events + 20
    s2, ev2 = cx.decode_occupancy(arrays, meta)["comp"]
    npt.assert_array_equal(s2, start)
    assert ev2 == ev
    times = np.linspace(0, 0.1, 17)
    npt.assert_array_equal(cx.occupancy_at(arrays, meta["columns"][0], times), events_at_instants(ev, start, times))


def test_c1_carries_the_mt_bound_pool_as_a_column_on_an_independent_axis():
    """MT needs no channel of its own: binding is an occupancy on an axis independent of the geometric compartment,
    and C1 already stores occupancy. A walker can be intra-axonal AND bound, so the two are separate columns of
    events whose product is the joint state; the bind / unbind instants ARE the dwell times, so C1 keeps the realised
    exchange statistics rather than a summarising mean rate."""
    rng = np.random.default_rng(21)
    n_w, T = 40, 0.3
    comp0, comp = _events(rng, n_w, T, 10.0)
    bnd0, bnd = _events(rng, n_w, T, 40.0, n_labels=2)
    arrays, meta = cx.encode_events_occupancy({"comp": (comp0, comp), "bound": (bnd0, bnd)})
    got = cx.decode_occupancy(arrays, meta)
    assert sorted(got) == ["bound", "comp"]
    npt.assert_array_equal(got["comp"][0], comp0); assert got["comp"][1] == comp
    npt.assert_array_equal(got["bound"][0], bnd0); assert got["bound"][1] == bnd
    times = np.linspace(0, T, 31)
    c, b = (cx.occupancy_at(arrays, col, times) for col in meta["columns"])
    assert ((c == 1) & (b == 1)).any() and ((c == 2) & (b == 1)).any()         # both states co-occur: two axes
    dwell = np.diff(arrays["bound_events_t"][arrays["bound_events_offsets"][0]:arrays["bound_events_offsets"][1]])
    assert dwell.size == 0 or dwell.min() > 0
    with pytest.raises(ValueError, match="needs the 'comp' column"):
        cx.encode_events_occupancy({"bound": (bnd0, bnd)})
    with pytest.raises(ValueError, match="one of"):
        cx.encode_events_occupancy({"comp": (comp0, comp), "bfrac": (bnd0, bnd)})


def test_relaxation_on_the_runs_is_exact_at_the_event_instants():
    """``relaxation_logweight_runs`` on an events column equals the dense rate x time sum of a fine sampling of the
    same walk, to the sampling's own resolution, and exactly for a static column."""
    rng = np.random.default_rng(6)
    n_w, T = 25, 0.05
    start, ev = _events(rng, n_w, T, 60.0)
    arrays, meta = cx.encode_events_occupancy({"comp": (start, ev)})
    T2, T1 = [0.08, 0.03, 0.01], [1.0, 1.2, 0.3]
    n_g = 201; dt_g = T / (n_g - 1)
    chi = np.ones(n_g); chi[0] = 0.0; chi[150:] = 0.0                               # transverse for 3/4 of the window
    act = np.ones(n_g); act[-1] = 0.0                                                # the readout, held over nothing
    gate, on = cx.CumulativeGate(chi, dt_g), cx.CumulativeGate(act, dt_g)
    got = cx.relaxation_logweight_runs(arrays, meta["columns"][0], T2, T1, gate, on, t1=T)
    fine = 20000; tf = (np.arange(fine) + 0.5) * T / fine
    lab = events_at_instants(ev, start, tf)
    chi_f = chi[np.minimum((tf / dt_g).astype(int), n_g - 2)]                         # the held gate on the fine grid
    r2, r1 = 1.0 / np.asarray(T2), 1.0 / np.asarray(T1)
    dense = -(T / fine) * (r2[lab] * chi_f[None, :] + r1[lab] * (1.0 - chi_f)[None, :]).sum(1)
    assert np.abs(got - dense).max() < 2e-3 * np.abs(dense).max()                      # the fine grid's resolution
    a1, m1 = cx.encode_static_occupancy(start)
    npt.assert_allclose(cx.relaxation_logweight_runs(a1, m1["columns"][0], T2, T1, gate, on, t1=T),
                        cx.relaxation_logweight(np.repeat(start[:, None], n_g, axis=1), T2, T1, dt_g, chi, act), rtol=1e-12)


def test_c1_rejects_a_pre_columns_or_rle_pack_loudly():
    """A pack written before the events layout keeps its arrays under their original names, so nothing stops it
    being READ -- only its metadata says what the track means; the refusal names the layout change."""
    rng = np.random.default_rng(31)
    start, ev = _events(rng, 12, 0.02, 50.0)
    arrays, meta = cx.encode_events_occupancy({"comp": (start, ev)})
    for stale in ({"channel": "compartment", "fractional": False, "n_t": 80},
                  {"channel": "compartment", "fractional": True, "Q": 256, "scale": 1.0, "n_t": 80}):
        assert not cx.is_current_c1(stale)
        with pytest.raises(ValueError, match="predates the occupancy-column layout"):
            cx.decode_occupancy(arrays, stale)
    with pytest.raises(ValueError, match="stored as 'label'"):
        cx.occupancy_events({"comp_rle_vals": np.zeros(3, np.int8)}, {"name": "comp", "kind": "label"})
    assert cx.is_current_c1(meta)
