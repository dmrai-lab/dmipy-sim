"""The walk's band accumulation (``_walk_geometry``, dmrai-lab/dmipy-sim#741): the walk accumulates the bands
a pack stores while it runs, on the continuous basis, and returns a :class:`CoefficientWalk`;
:func:`build_replay_pack` takes it and certifies it walk-free.

The measurements live once, in ``tools/coefficient_walk_measure.py`` (the PR's tables); these tests run them small:
the order-0 accumulation against the pack of the same walk's block ends and the exact ``sqrt(h) sinc`` relation to
their DST-I on five engine classes, the quadrature against a walk accumulated per step stated absolutely on a free
walk and a 0.8 um slab, the matrix method on a sphere and a cylinder, and the walk's host memory over blocks spanning
16x (the quadrature and the memory are minutes of walking on a CPU: the slow tier).
"""
import dmipy_replay.replay.compression as _r_compression
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine.core import _walk_geometry
from dmipy_replay.constants import GAMMA
from dmipy_sim.replay import compression as cx
from dmipy_sim.replay.bank import build_replay_pack

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("coefficient_walk_measure", ROOT / "tools" / "coefficient_walk_measure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _dst1_to_continuous(K, n_t, T):
    """Per band the factor taking the orthonormal DST-I band of a bridge sampled at ``n_t`` block ends to its continuous
    band at block order 0 (the ``m = 0`` exactness relation): summation by parts gives the DST-I band ``sqrt(2 / N) /
    (2 sin(pi k / 2N)) c_k`` with ``c_k`` the cosine projection of the increments between block ends at their midpoints
    (``N = n_t - 1``), exactly; the continuous band of the same ``c_k`` is :func:`cosine_band_scale`. ``sqrt(h)`` times
    ``sinc(k / 2N)``. The in-test reference, never a reader: no pack is on the block ends."""
    N = int(n_t) - 1
    k = np.arange(1, int(K) + 1, dtype=np.float64)
    return _r_compression.cosine_band_scale(K, T) * 2.0 * np.sin(np.pi * k / (2.0 * N)) / np.sqrt(2.0 / N)


# ------------------------------------------------------------------------------------------------ the basis itself
def test_the_order_zero_table_is_the_scaled_dst1_and_the_variances_their_limit():
    """The block table's order-0 projection of a path's increments, turned into bands (``cosine_band_scale``), IS
    ``sqrt(h) sinc(pi k / 2N)`` times the orthonormal DST-I of the bridge at the block ends (:func:`_dst1_to_continuous`);
    the free bridge's DST-I variances, read in the continuous normalisation, tend to ``2 D T^2 / (pi k)^2``; and the
    bands carry together the closed-form bridge variance."""
    from scipy.fft import dst
    rng = np.random.default_rng(0)
    N, K, T = 50, 20, 0.01
    h = T / N
    x = np.cumsum(rng.normal(size=(4, N + 1, 3)), axis=1)
    c = np.einsum("bk,wbd->wkd", cx.block_table(1.0, N, K, 0)[:, :, 0], np.diff(x, axis=1))
    beta = _r_compression.cosine_band_scale(K, T)[None, :, None] * c[:, 1:]
    tau = np.arange(N + 1) / N
    u = x - (x[:, :1] + (x[:, -1:] - x[:, :1]) * tau[None, :, None])
    ref = dst(u[:, 1:-1], type=1, norm="ortho", axis=1)[:, :K]
    k = np.arange(1, K + 1)
    np.testing.assert_allclose(_dst1_to_continuous(K, N + 1, T), np.sqrt(h) * np.sinc(k / (2.0 * N)), rtol=1e-12)
    assert np.abs(beta - _dst1_to_continuous(K, N + 1, T)[None, :, None] * ref).max() <= 1e-12 * np.abs(beta).max()
    assert np.abs(c[:, 0] - (x[:, -1] - x[:, 0])).max() <= 1e-12 * np.abs(x).max()          # k = 0: the displacement
    D, n = 2e-9, 20000
    dst_var = D * (T / n) / (2.0 * np.sin(np.pi * np.arange(1, 9) / (2 * n)) ** 2)            # the DST-I's free variances
    ratio = dst_var * _dst1_to_continuous(8, n + 1, T) ** 2 / _r_compression.band_variances(8, T, D)
    assert np.abs(ratio - 1).max() < 1e-6
    G = rng.normal(size=(2, 38, 3))
    Wb = _r_compression.band_projection(G, T / 37, T, 20000)[:, 2:]
    np.testing.assert_allclose((Wb ** 2 * _r_compression.band_variances(20000, T, D)[None, :, None]).sum((1, 2)),
                               _r_compression.bridge_phase_variance(G, T / 37, T, D), rtol=1e-6)


def test_the_projection_is_the_integral_of_the_held_waveform():
    """``W_k = int G s_k dt`` for the sample-and-hold waveform, against a midpoint rule fine enough to see it; ``M0``
    and ``M1`` likewise; a window starting at ``t0`` reads the waveform shifted."""
    rng = np.random.default_rng(1)
    T, n = 0.01, 37
    G = rng.normal(size=(2, n + 1, 3))
    W = _r_compression.band_projection(G, T / n, T, 8)
    nf = 10000 * n                                                                          # pieces aligned with the samples
    t = (np.arange(nf) + 0.5) * T / nf
    g = G[:, np.minimum((t / (T / n)).astype(int), n), :]
    for k in (1, 5, 8):
        num = (g * (np.sqrt(2 / T) * np.sin(np.pi * k * t / T))[None, :, None]).sum(1) * (T / nf)
        assert np.abs(num - W[:, 1 + k]).max() <= 2e-4 * np.abs(W[:, 1 + k]).max()
    np.testing.assert_allclose(W[:, 0], (g.sum(1) * T / nf), rtol=1e-4, atol=1e-12)
    Gs = np.concatenate([np.zeros((2, 10, 3)), G], axis=1)                                   # the same waveform 10 samples later
    np.testing.assert_allclose(_r_compression.band_projection(Gs, T / n, T, 8, t0=10 * T / n), W, rtol=1e-9, atol=1e-15)


# -------------------------------------------------------------------------------------------- exactness at m = 0
@pytest.mark.parametrize("name", ["sphere", "cylinder", "packed cell", "mesh", "myelinated cylinder"])
def test_the_order_zero_accumulation_is_the_pack_of_the_block_ends(name):
    """Same walk: its accumulation at ``block_order=0`` against its block ends (every walker kept) -- the start and the
    displacement exact, the bands and the contact's bands ``sqrt(h) sinc(pi k / 2N)`` times the DST-I of the block ends
    (#741) to float32 rounding, the occupancy read off the events at the block ends the raw one, and the measured
    certificate (the battery replayed on the block ends against the bands) within the floor."""
    if name == "mesh":
        pytest.importorskip("trimesh")
    # the mesh walks at its resolved count: three steps per block outrun its collision cell, and a step longer than the
    # cell passes walls its gather does not hold
    row = _tool().exactness([name], n=200, T=0.008, block_h=2e-4, K=16, sub_steps=None if name == "mesh" else 3)[name]
    assert row["start_m"] == 0.0 and row["displacement_rel"] <= 1e-6
    assert row["bands_vs_block_ends_rel"] <= 2e-6, row
    assert row.get("contact_bands_vs_block_ends_rel", 0.0) <= 5e-6 and row.get("contact_endpoint_rel", 0.0) <= 2e-6, row
    assert row.get("occupancy_equal", True), row
    assert row["replay_err"] <= 2.0 * row["replay_floor"], row                   # the measured certificate on the block ends


# ----------------------------------------------------------------------------------------------- the quadrature
@pytest.mark.slow
@pytest.mark.parametrize("which", ["free", "slab 0.8 um"])
def test_the_quadrature_is_bounded_absolutely(which):
    """The phase variance the block accumulation adds per measurement of the battery, against the same walk accumulated
    per step, is within ``rel(m, h) 2 b D_free`` on a free walk and on a 0.8 um slab; on the free walk it IS the
    table's prediction (the measurement the table predicts largest, within the variance estimate's spread); on the slab
    the error RELATIVE to a mode's own variance is several times the table, which is why the statement is absolute."""
    geom = d.FreeDiffusion() if which == "free" else d.Box1D(length=0.8e-6)
    rows = _quiet(_tool().quadrature, geom, T=0.01, dt_sim=1e-6, K=48, cases=((1, 2e-5), (2, 5e-5), (3, 1e-4)), n=1500)
    for r in rows:
        assert r["within_bound"], r
        if which == "free" and r["predicted_largest"] > 1e-10:                        # above the float32 phase rounding
            assert 0.75 <= r["ratio_at_largest_prediction"] <= 1.3, r
    if which != "free":
        assert all(r["relative_vs_table_top_modes"] > 2.0 for r in rows), rows


# ---------------------------------------------------------------------------------------------------- the oracle
def test_sphere_and_cylinder_agree_with_the_matrix_method():
    """A sphere and a cylinder replayed from their accumulated packs under a cosine OGSE that the pack's band only just
    holds agree with the matrix method within the replay's own Monte-Carlo floor."""
    res = _quiet(_tool().oracle, n=6000, K=80)                                       # the band's own block and order
    for name, r in res.items():
        assert r["band_needed"] <= r["K"] and r["block"]["m"] == 3, (name, r)
        for diff, se in zip(r["diff"], r["se"]):
            assert abs(diff) <= 3.0 * se + 2e-3, (name, r)


# ------------------------------------------------------------------------------------------------- the contact tier
def test_the_contact_tier_is_admitted_at_the_signal():
    """The surface-relaxation rows at the envelope's largest rho2 on a reflecting sphere: the accumulated pack against
    the dense reading of the same walk's block ends (the codec: the whole contact at TE = T, the contact gated at the
    acquisition's end at TE = T / 2) and against the fused engine (another walk), each within the replay's floor."""
    res = _quiet(_tool().surface, n=4000)
    assert res["qv_contact_mean"] > 0 and res["quadrature_local_time"] is not None
    assert res["err_surface"] <= 2.0 * res["floor_surface"] + 1e-9
    for row in res["rows"]:
        for a_s, a_f, se in zip(row["acc_minus_raw"], row["acc_minus_fused"], row["se"]):
            assert abs(a_s) <= 2.0 * se, row
            assert abs(a_f) <= 3.0 * np.sqrt(2.0) * se + 2e-3, row


# ------------------------------------------------------------------------------------------------------- memory
@pytest.mark.slow
def test_the_walks_host_bytes_do_not_grow_with_the_block_count():
    """Three blocks spanning 16x at a fixed K, each walk in a process of its own: the accumulated walk's peak host bytes
    stay within a few MB of each other, while the same walk keeping every walker's block ends grows by them."""
    rows = _tool().memory(n=40000, T=0.004, dt_sim=3.125e-6, K=16, block_hs=(2e-4, 5e-5, 1.25e-5))
    acc = [r["host_peak_over_base"] for r in rows if r["mode"] == "accumulated"]
    saves = [r for r in rows if r["mode"] == "raw_subset"]
    grown = saves[-1]["walk_host_bytes"] - saves[0]["walk_host_bytes"]
    assert grown > 5e6
    assert max(acc) - min(acc) < 0.25 * grown, (acc, [r["host_peak_over_base"] for r in saves])
    assert saves[-1]["host_peak_over_base"] - saves[0]["host_peak_over_base"] > 0.5 * grown, rows


def test_the_walk_takes_the_bands_own_block_and_the_storage_band():
    """The walk alone: the storage band (K = 256 per 100 ms), the block order 3 and the block ``h = 0.8 T / (pi K)``
    in a whole number of blocks of one step count each, recorded in the walk; ``K``, a given ``dt_save`` and
    ``block_order`` set them instead; a walk longer than a storage window is accumulated window by window."""
    from dmipy_sim.engine.accumulate import block_length, BLOCK_ORDER, default_K
    g = d.Cylinder(radius=3e-6, orientation=(0, 0, 1))
    T, K = 0.01, 32
    assert default_K(0.1) == 256 and default_K(T) == 26
    plain = _quiet(_walk_geometry, 20, 2e-9, g, T, require_gpu=False)
    assert plain.K == 26 and plain.n_windows == 1
    cw = _quiet(_walk_geometry, 50, 2e-9, g, T, K=K, require_gpu=False)
    n = int(round(T * np.pi * K / (0.8 * T)))
    assert BLOCK_ORDER == 3 and cw.m == 3 and cw.block["rule"] == "derived"
    assert cw.n_blocks == n and cw.h == pytest.approx(T / n) == pytest.approx(block_length(T, K))
    assert np.pi * K / T * cw.h == pytest.approx(0.8, rel=0.01)
    assert cx.quadrature_rel(3, cw.h, np.pi * K / T) == pytest.approx(1.3e-7, rel=0.05)
    free = _quiet(_walk_geometry, 2000, 2e-9, d.FreeDiffusion(), T, K=K, require_gpu=False)
    qv = free.block["quadratic_variation"]                                                # sum (dx)^2 = 2 D T per axis
    assert np.allclose(qv["positions_over_2DT"], 1.0, rtol=0.05), qv
    assert free.qv_positions.shape == (2000, 1, 3) and cw.qv_boundary.shape == (50, 1)
    given = _quiet(_walk_geometry, 50, 2e-9, g, T, 2e-4, K=K, block_order=1, require_gpu=False)
    assert (given.m, given.h, given.block["rule"]) == (1, pytest.approx(2e-4), "given")
    long = _quiet(_walk_geometry, 20, 2e-9, g, 0.2, 2e-3, K=8, require_gpu=False, sub_steps=2)
    assert long.n_windows == 2 and long.position_coeffs.shape == (20, 2, 10, 3) and long.n_blocks == 100
    with pytest.raises(ValueError, match="whole number"):
        _walk_geometry(10, 2e-9, g, 0.15, require_gpu=False)


def test_walk_cost_states_the_accumulators():
    from dmipy_sim.engine.core import walk_cost
    g = d.Cylinder(radius=3e-6, orientation=(0, 0, 1))
    costs = [walk_cost(1000, 2e-9, g, 0.01, h, K=32) for h in (4e-4, 1e-4, 2.5e-5)]
    assert all(c.accumulator_bytes_per_walker > 4 * 33 * 4 for c in costs)
    assert len({c.bytes_per_walker for c in costs}) == 1                                    # whatever the block count
    assert costs[0].f_band_hz == pytest.approx(32 / (2 * 0.01))                             # the band K / (2 T)
    assert [c.n_blocks for c in costs] == [25, 100, 400] and costs[0].K == 32


# ------------------------------------------------------------------------------------------------- the pack of it
@pytest.fixture(scope="module")
def walks():
    g = d.Cylinder(radius=2.5e-6, orientation=(0, 0, 1))
    kw = dict(T_max=0.01, block_h=1e-4, seed=4, require_gpu=False, sub_steps=4)
    return dict(c=_quiet(_walk_geometry, 600, 2e-9, g, K=24, raw_subset=200, **kw),
                z=_quiet(_walk_geometry, 600, 2e-9, g, K=24, block_order=0, **kw))


def test_a_coefficient_walk_packs_walk_free_and_measured_only_on_its_raw_subset(walks):
    cw = walks["c"]
    assert cw.position_coeffs.shape == (600, 1, 26, 3) and cw.raw_positions.shape == (200, 101, 3)
    np.testing.assert_array_equal(cw.raw_positions[:, 0], cw.position_coeffs[:200, 0, 0])   # the subset is the walk's own
    np.testing.assert_allclose(cw.raw_positions[:, -1], cw.position_coeffs[:200, 0, 0] + cw.position_coeffs[:200, 0, 1],
                               rtol=0, atol=1e-12)
    np.testing.assert_array_equal(cw.position_coeffs[:, :, :2], walks["z"].position_coeffs[:, :, :2])
    assert cw.h == pytest.approx(1e-4)
    pk = build_replay_pack(cw, id="t/derived", license="x", citation="x")
    fid = pk.fidelity
    assert fid["certified"] == "measured" and fid["measured_by"] == "walk_free_terms" and "basis" not in pk.meta["compression"]
    assert pk.meta["walk_params"]["block"]["m"] == 3
    for k in ("n_t", "dt_traj"):
        assert k not in pk.meta["walk_params"] and k not in pk.meta["compression"]
    assert pk.n_windows == 1 and "n_t" not in pk.meta["walk_params"]["windows"]
    assert fid["err_quadrature"] > 0 and fid["err_max"] <= 2 * fid["floor_max"], fid
    pm = build_replay_pack(cw, id="t/measured", license="x", citation="x", fidelity="measured")
    assert pm.fidelity["certified"] == "measured" and pm.fidelity["measured_on"] == 200
    with pytest.raises(ValueError, match="raw_subset"):
        build_replay_pack(walks["z"], id="t/x", license="x", citation="x", fidelity="measured")
    with pytest.raises(ValueError, match="K >= 48"):
        build_replay_pack(cw, id="t/x", license="x", citation="x", K=48)


def test_err_max_carries_every_term_and_the_float32_rounding(walks):
    """RPK.md 9.5.5: a float32 accumulation carries its measured rounding as ``fidelity.terms.rounding``, ``r sqrt(2
    b_max D)`` with ``r`` the walk's relative ``k = 0`` residual, and ``err_max`` is at least the sum of the terms."""
    from dmipy_sim.replay import compression as cx
    from dmipy_sim.replay.bank import _battery_b_max
    cw = walks["c"]
    pk = build_replay_pack(cw, id="t/rounding", license="x", citation="x")
    f, t = pk.fidelity, pk.fidelity["terms"]
    assert cw.block["precision"] == "float32" and "rounding" in t
    want = cw.block["k0_residual_relative"] * np.sqrt(2.0 * _battery_b_max(cx.default_envelope()) * f["bounds_diffusivity"])
    assert t["rounding"] == pytest.approx(want, rel=1e-5) and t["rounding"] > 0          # the header keeps 6 digits
    assert f["err_max"] >= t["truncation"] + sum(t["quadrature"].values()) + t["rounding"]


def test_both_orders_replay_one_walk_alike_and_their_views_follow(walks):
    """The same walk accumulated at order 3 and at order 0 replays alike within the two certificates; the views keep
    it: a faster diffusivity (the bands renormalised by ``a^(-1/2)``) and a prefix (re-encoded from the bands at the
    block ends)."""
    from dmipy_replay.spec.tissue import Tissue
    pc = build_replay_pack(walks["c"], id="t/c", license="x", citation="x")
    pz = build_replay_pack(walks["z"], id="t/z", license="x", citation="x")
    seq = d.pgse([[1, 0, 0], [0, 0, 1]], 1e-3, 3e-3, bvalues=[1e9, 1e9], TE=0.006, slew_rate=np.inf)
    tol = pc.fidelity["err_max"] + pz.fidelity["err_max"] + 1e-3
    assert np.abs(pc.replay(seq) - pz.replay(seq)).max() <= tol
    t = Tissue(rho2=2e-5, T2={"extra": 0.05, "intra": 0.04})
    assert np.abs(pc.replay(seq, tissue=t) - pz.replay(seq, tissue=t)).max() <= tol + 1e-3
    short = d.pgse([[1, 0, 0]], 0.5e-3, 1.5e-3, bvalues=[1e9], TE=0.003, slew_rate=np.inf)
    assert abs(float(pc.at_diffusivity(4e-9).replay(short)[0]) - float(pz.at_diffusivity(4e-9).replay(short)[0])) <= tol
    pre = pc.prefix(0.006)
    assert pre.n_t == 61 and abs(float(pre.replay(seq)[0]) - float(pc.replay(seq)[0])) <= 2 * tol


def test_a_walk_of_several_windows_packs_window_by_window():
    g = d.Cylinder(radius=3e-6, orientation=(0, 0, 1), permeability=1e-5)
    cw = _quiet(_walk_geometry, 200, 2e-9, g, 0.2, 2e-3, K=8, require_gpu=False, sub_steps=2)
    pk = build_replay_pack(cw, id="t/w", license="x", citation="x")
    assert pk.n_windows == 2 and pk.windows["T"] == pytest.approx(0.1) and pk.n_t == 101
    assert pk.meta["walk_params"]["block"]["n_blocks"] == 100 and pk.fidelity["combined"] == "bound_over_windows"
    seq = d.pgse([[1, 0, 0]], 5e-3, 0.12, bvalues=[3e8], TE=0.15, slew_rate=np.inf)
    one = build_replay_pack(cw.window(0), id="t/w0", license="x", citation="x")
    assert one.n_windows == 1 and np.isfinite(pk.replay(seq)).all()


def test_what_a_walk_does_not_carry_is_refused_by_name():
    from dmipy_sim.engine.backends import WalkBatch

    class Other:
        """A backend that returns a walk of saves."""
        name = "other"

        def refuses(self, request):
            return None

        def walk_batch(self, request, r0, keys):
            n, n_t = r0.shape[0], request.n_blocks + 1
            zero = np.zeros((n,), np.int32)
            return WalkBatch(np.repeat(np.asarray(r0, np.float32)[:, None], n_t, axis=1), None, None, zero, (None, None, zero))
    g = d.Cylinder(radius=3e-6, orientation=(0, 0, 1))
    with pytest.raises(ValueError, match="returned a WalkBatch for the coefficient walk.*converts nothing"):
        _walk_geometry(10, 2e-9, g, 0.002, 1e-4, require_gpu=False, K=8, backend=Other())
    with pytest.raises(ValueError, match="hold 19 bands"):
        _walk_geometry(10, 2e-9, g, 0.002, 1e-4, require_gpu=False, K=30, block_order=0)
    cw = _quiet(_walk_geometry, 50, 2e-9, g, 0.2, 2e-3, require_gpu=False, K=8, sub_steps=2)
    with pytest.raises(ValueError, match="window_T"):
        build_replay_pack(cw, id="t/x", license="x", citation="x", window_T=0.05)
    assert cw.raw_positions is None                                    # no raw subset asked for, none kept


def test_the_contact_is_the_gated_integral():
    """Ungated, the contact is the exact total ``e - a``; gated by a step at ``T / 2`` it is the bridge's
    ``int chi dB`` = ``B(T / 2) - B(0)`` read from the series."""
    rng = np.random.default_rng(3)
    T, n_t, K = 0.01, 101, 12
    a = np.zeros(5); e = -rng.uniform(1e-7, 1e-6, 5); bands = rng.normal(size=(5, K)) * 1e-8
    arrays, meta = cx.encode_boundary_bands(a, e, bands, T=T)
    assert "n_t" not in meta
    np.testing.assert_allclose(_r_compression.surface_logweight_bridge(arrays, meta, 2.0), 2.0 * (e - a), rtol=1e-6)
    gate = np.r_[np.ones(50), np.zeros(51)]                                                    # on over [0, T/2]
    B_half = _r_compression.local_time_at(arrays, meta, [T / 2])[:, 0]
    S, tau = _r_compression.bridge_basis([T / 2], K, T)
    np.testing.assert_allclose(B_half, a + (e - a) * tau[0] + (bands @ S)[:, 0], rtol=1e-6, atol=1e-14)
    np.testing.assert_allclose(_r_compression.surface_logweight_bridge(arrays, meta, 1.0, gate, T / (n_t - 1)), B_half - a,
                               rtol=1e-5, atol=1e-14)
