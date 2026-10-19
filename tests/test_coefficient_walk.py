"""The coefficient walk (``simulate_trajectories(accumulate=K)``, dmrai-lab/dmipy-sim#741): a walk that accumulates the
bands a pack stores while it runs, on the continuous basis (``block_order >= 1``) or, at ``block_order=0``, exactly
the DST-I of its block ends; :func:`build_replay_pack` takes it and certifies it walk-free.

The measurements live once, in ``tools/coefficient_walk_measure.py`` (the PR's tables); these tests run them small:
the order-0 pack against the pack of the same walk stored at its saves on five engine classes, the quadrature against
a walk accumulated per step stated absolutely on a free walk and a 0.8 um slab, the matrix method on a sphere and a
cylinder, and the walk's host memory over save intervals spanning 16x (the quadrature and the memory are minutes of
walking on a CPU: the slow tier).
"""
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.constants import GAMMA
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


# ------------------------------------------------------------------------------------------------ the basis itself
def test_the_order_zero_table_is_the_dst1_and_the_continuous_variances_its_limit():
    """The block table's order-0 projection of a path's increments, scaled by ``cosine_band_scale("dst1")``, IS the
    orthonormal DST-I of the bridge at the block ends; the free bridge's DST-I variances, read in the continuous
    normalisation, tend to ``2 D T^2 / (pi k)^2``; and the continuous bands carry together the closed-form bridge
    variance."""
    from scipy.fft import dst
    rng = np.random.default_rng(0)
    N, K = 50, 20
    x = np.cumsum(rng.normal(size=(4, N + 1, 3)), axis=1)
    c = np.einsum("bk,wbd->wkd", cx.block_table(1.0, N, K, 0)[:, :, 0], np.diff(x, axis=1))
    beta = cx.cosine_band_scale("dst1", K, N + 1)[None, :, None] * c[:, 1:]
    tau = np.arange(N + 1) / N
    u = x - (x[:, :1] + (x[:, -1:] - x[:, :1]) * tau[None, :, None])
    ref = dst(u[:, 1:-1], type=1, norm="ortho", axis=1)[:, :K]
    assert np.abs(beta - ref).max() <= 1e-12 * np.abs(ref).max()
    assert np.abs(c[:, 0] - (x[:, -1] - x[:, 0])).max() <= 1e-12 * np.abs(x).max()          # k = 0: the displacement
    T, D, n_t = 0.01, 2e-9, 20001
    dt = T / (n_t - 1)
    ratio = cx.band_variances("dst1", 8, n_t, dt, D) * dt / cx.band_variances("continuous", 8, n_t, dt, D)
    assert np.abs(ratio - 1).max() < 1e-6
    G = rng.normal(size=(2, 38, 3))
    Wb = cx.continuous_projection(G, T / 37, T, 20000)[:, 2:]
    np.testing.assert_allclose((Wb ** 2 * cx.band_variances("continuous", 20000, 2, T, D)[None, :, None]).sum((1, 2)),
                               cx.bridge_phase_variance(G, T / 37, T, D), rtol=1e-6)


def test_the_continuous_projection_is_the_integral_of_the_held_waveform():
    """``W_k = int G s_k dt`` for the sample-and-hold waveform, against a midpoint rule fine enough to see it; ``M0``
    and ``M1`` likewise; a window starting at ``t0`` reads the waveform shifted."""
    rng = np.random.default_rng(1)
    T, n = 0.01, 37
    G = rng.normal(size=(2, n + 1, 3))
    W = cx.continuous_projection(G, T / n, T, 8)
    nf = 10000 * n                                                                          # pieces aligned with the samples
    t = (np.arange(nf) + 0.5) * T / nf
    g = G[:, np.minimum((t / (T / n)).astype(int), n), :]
    for k in (1, 5, 8):
        num = (g * (np.sqrt(2 / T) * np.sin(np.pi * k * t / T))[None, :, None]).sum(1) * (T / nf)
        assert np.abs(num - W[:, 1 + k]).max() <= 2e-4 * np.abs(W[:, 1 + k]).max()
    np.testing.assert_allclose(W[:, 0], (g.sum(1) * T / nf), rtol=1e-4, atol=1e-12)
    Gs = np.concatenate([np.zeros((2, 10, 3)), G], axis=1)                                   # the same waveform 10 samples later
    np.testing.assert_allclose(cx.continuous_projection(Gs, T / n, T, 8, t0=10 * T / n), W, rtol=1e-9, atol=1e-15)


# -------------------------------------------------------------------------------------------- exactness at m = 0
@pytest.mark.parametrize("name", ["sphere", "cylinder", "packed cell", "mesh", "myelinated cylinder"])
def test_the_order_zero_pack_is_the_pack_of_the_walk_at_its_saves(name):
    """Same seed, same starts, same steps: the pack of the walk accumulated at ``block_order=0`` equals the pack of the
    walk stored at its saves to float32 rounding -- start and displacement exact, the bands, the contact and the
    replay within the float32 accumulation, the occupancy identical (#741)."""
    if name == "mesh":
        pytest.importorskip("trimesh")
    row = _tool().exactness([name], n=200, T=0.008, dt_save=2e-4, K=16, sub_steps=3)[name]
    assert row["basis"] == "dst1"
    assert row["start_m"] == 0.0 and row["displacement_rel"] <= 1e-6
    assert row["bands_rel"] <= 2e-6, row
    assert row["replay_abs"] <= 1e-6, row
    assert row.get("contact_bands_rel", 0.0) <= 5e-6 and row.get("contact_endpoint_rel", 0.0) <= 2e-6, row
    assert row.get("occupancy_equal", True), row


# ----------------------------------------------------------------------------------------------- the quadrature
@pytest.mark.slow
@pytest.mark.parametrize("which", ["free", "slab 0.8 um"])
def test_the_quadrature_is_bounded_absolutely(which):
    """The phase variance the block accumulation adds per measurement of the battery, against the same walk accumulated
    per step, is within ``rel(m, h) 2 b D_free`` on a free walk and on a 0.8 um slab; on the free walk it IS the
    table's prediction (the measurement the table predicts largest, within the variance estimate's spread); on the slab
    the error RELATIVE to a mode's own variance is several times the table, which is why the statement is absolute."""
    geom = d.FreeDiffusion() if which == "free" else d.Box1D(length=0.8e-6)
    rows = _quiet(_tool().quadrature, geom, T=0.01, dt_sim=1e-6, K=48, cases=((0, 2e-5), (1, 2e-5), (1, 4e-5)), n=1500)
    for r in rows:
        assert r["within_bound"], r
        if which == "free":
            assert 0.75 <= r["ratio_at_largest_prediction"] <= 1.3, r
    if which != "free":
        assert all(r["relative_vs_table_top_modes"] > 2.0 for r in rows), rows


# ---------------------------------------------------------------------------------------------------- the oracle
def test_sphere_and_cylinder_agree_with_the_matrix_method():
    """A sphere and a cylinder replayed from their accumulated packs (the continuous basis) under a cosine OGSE that the
    pack's band only just holds agree with the matrix method within the replay's own Monte-Carlo floor."""
    res = _quiet(_tool().oracle, n=6000, K=80)
    for name, r in res.items():
        assert r["basis"] == "continuous" and r["band_needed"] <= r["K"], (name, r)
        for diff, se in zip(r["diff"], r["se"]):
            assert abs(diff) <= 3.0 * se + 2e-3, (name, r)


# ------------------------------------------------------------------------------------------------------- memory
@pytest.mark.slow
def test_the_walks_host_bytes_do_not_grow_with_the_save_interval():
    """Three save intervals spanning 16x at a fixed K, each walk in a process of its own: the accumulated walk's peak
    host bytes stay within a few MB of each other, while the walk stored at its saves grows by its positions."""
    rows = _tool().memory(n=8000, T=0.004, dt_sim=3.125e-6, K=16, dt_saves=(2e-4, 5e-5, 1.25e-5))
    acc = [r["host_peak_over_base"] for r in rows if r["mode"] == "accumulated"]
    saves = [r for r in rows if r["mode"] == "saves"]
    grown = saves[-1]["walk_host_bytes"] - saves[0]["walk_host_bytes"]
    assert grown > 5e6
    assert max(acc) - min(acc) < 0.25 * grown, (acc, [r["host_peak_over_base"] for r in saves])
    assert saves[-1]["host_peak_over_base"] - saves[0]["host_peak_over_base"] > 0.5 * grown, rows


def test_walk_cost_states_the_accumulators():
    from dmipy_sim.engine.core import walk_cost
    g = d.Cylinder(radius=3e-6, orientation=(0, 0, 1))
    costs = [walk_cost(1000, 2e-9, g, 0.01, h, accumulate=32) for h in (4e-4, 1e-4, 2.5e-5)]
    assert all(c.accumulator_bytes_per_walker and c.accumulator_bytes_per_walker > 4 * 33 * 4 for c in costs)
    assert len({c.bytes_per_walker for c in costs}) == 1
    assert walk_cost(1000, 2e-9, g, 0.01, 1e-4).accumulator_bytes_per_walker is None
    assert costs[0].f_band_hz == pytest.approx(32 / (2 * 0.01))                             # the band K / (2 T)


# ------------------------------------------------------------------------------------------------- the pack of it
@pytest.fixture(scope="module")
def walks():
    g = d.Cylinder(radius=2.5e-6, orientation=(0, 0, 1))
    kw = dict(T_max=0.01, dt_save=1e-4, seed=4, require_gpu=False, sub_steps=4)
    return dict(c=_quiet(d.simulate_trajectories, 600, 2e-9, g, accumulate=24, raw_subset=200, **kw),
                z=_quiet(d.simulate_trajectories, 600, 2e-9, g, accumulate=24, block_order=0, **kw),
                raw=_quiet(d.simulate_trajectories, 600, 2e-9, g, **kw))


def test_a_coefficient_walk_packs_walk_free_and_measured_only_on_its_raw_subset(walks):
    cw = walks["c"]
    assert cw.basis == "continuous" and cw.position_coeffs.shape == (600, 26, 3) and cw.raw_positions.shape == (200, 101, 3)
    np.testing.assert_array_equal(cw.raw_positions, walks["raw"].positions[:200])          # the subset is the walk's own
    np.testing.assert_array_equal(cw.position_coeffs[:, :2], walks["z"].position_coeffs[:, :2])
    pk = build_replay_pack(cw, id="t/derived", license="x", citation="x")
    fid = pk.fidelity
    assert pk.meta["compression"]["basis"] == "continuous" and fid["certified"] == "derived"
    assert pk.meta["walk_params"]["block"]["m"] == 1
    assert fid["err_quadrature"] > 0 and fid["err_max"] <= 2 * fid["floor_max"], fid
    pm = build_replay_pack(cw, id="t/measured", license="x", citation="x", fidelity="measured")
    assert pm.fidelity["certified"] == "measured" and pm.fidelity["measured_on"] == 200
    with pytest.raises(ValueError, match="raw_subset"):
        build_replay_pack(walks["z"], id="t/x", license="x", citation="x", fidelity="measured")
    with pytest.raises(ValueError, match="accumulate"):
        build_replay_pack(cw, id="t/x", license="x", citation="x", K=48)


def test_both_bases_replay_one_walk_alike_and_their_views_follow(walks):
    """The same walk on the continuous basis and on the DST-I of its block ends replays alike within the two
    certificates; the views keep it: a faster diffusivity (the continuous bands renormalised by ``a^(-1/2)``) and a
    prefix (re-encoded from the bands at the saves)."""
    from dmipy_sim.spec.tissue import Tissue
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


def test_what_a_coefficient_walk_does_not_carry_is_refused_by_name():
    class Other:
        name = "other"

        def refuses(self, request):
            return None

        def walk_batch(self, request, r0, keys):
            raise AssertionError("not reached")
    g = d.Cylinder(radius=3e-6, orientation=(0, 0, 1))
    with pytest.raises(ValueError, match="coefficient accumulation"):
        d.simulate_trajectories(10, 2e-9, g, 0.002, 1e-4, require_gpu=False, accumulate=8, backend=Other())
    with pytest.raises(NotImplementedError, match="field"):
        d.simulate_trajectories(10, 2e-9, g, 0.002, 1e-4, require_gpu=False, accumulate=8, field_basis=object())
    with pytest.raises(ValueError, match="DST-I"):
        d.simulate_trajectories(10, 2e-9, g, 0.002, 1e-4, require_gpu=False, accumulate=30, block_order=0)
    cw = _quiet(d.simulate_trajectories, 50, 2e-9, g, 0.2, 2e-3, require_gpu=False, accumulate=8, sub_steps=2)
    with pytest.raises(ValueError, match="one window"):
        build_replay_pack(cw, id="t/x", license="x", citation="x")


def test_the_continuous_contact_is_the_gated_integral():
    """Ungated, the continuous contact is the exact total ``e - a``; gated by a step at ``T / 2`` it is the bridge's
    ``int chi dB`` = ``B(T / 2) - B(0)`` read from the series."""
    rng = np.random.default_rng(3)
    T, n_t, K = 0.01, 101, 12
    a = np.zeros(5); e = -rng.uniform(1e-7, 1e-6, 5); bands = rng.normal(size=(5, K)) * 1e-8
    arrays, meta = cx.encode_boundary_bands(a, e, bands, n_t=n_t, basis="continuous", T=T)
    np.testing.assert_allclose(cx.surface_logweight_continuous(arrays, meta, 2.0), 2.0 * (e - a), rtol=1e-6)
    gate = np.r_[np.ones(50), np.zeros(51)]                                                    # on over [0, T/2]
    S, tau = cx.saves_basis(n_t, 50, 51, K, basis="continuous", T=T)
    B_half = a + (e - a) * tau[0] + (bands @ S)[:, 0]
    np.testing.assert_allclose(cx.surface_logweight_continuous(arrays, meta, 1.0, gate, T / (n_t - 1)), B_half - a,
                               rtol=1e-5, atol=1e-14)
    with pytest.raises(ValueError, match="continuous"):
        cx.surface_logweight_bridge(arrays, meta, 1.0, np.ones(n_t))
