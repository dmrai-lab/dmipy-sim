"""A foreign engine's saves as a C0-only pack (``dmipy_sim.foreign.walk_from_positions``): the bands of the
piecewise-linear interpolant, the save grid's interpolation term against a fine-grid walk, a dmipy-sim cylinder walk
re-encoded from its own saves, and the refusals."""
import warnings

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_replay.constants import GAMMA
from dmipy_replay.spec.tissue import Tissue
from dmipy_sim.foreign import (band_limit, interpolation_variance, interpolation_variance_bound, walk_from_positions)
from dmipy_sim.replay.bank import build_replay_pack

D0 = 2e-9
PRODUCER = dict(engine="synthetic", version="0")


def _spec():
    return d.Cylinder(radius=2.5e-6, orientation=(0, 0, 1)).spec


def _free(n, N, h, seed):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=(n, N, 3)) * np.sqrt(2 * D0 * h), axis=1)
    return np.concatenate([np.zeros((n, 1, 3)), x], axis=1)


def _walk(x, h, K, **kw):
    return walk_from_positions(x, dt_save=h, K=K, pool="intra", spec=_spec(), D=D0, producer=PRODUCER, **kw)


def test_the_bands_are_the_exact_projection_of_the_piecewise_linear_path():
    """beta_k against the integral of the interpolant on a fine resample (the trapezoid rule is exact on a fine
    enough grid for a sine times a piecewise-linear function; here 64 points per interval, error O(1e-6))."""
    N, h, K = 24, 1e-4, 10
    x = _free(5, N, h, 1)
    w = _walk(x, h, K)
    T = N * h
    t = np.linspace(0, T, N * 64 + 1)
    L = np.stack([np.interp(t, np.arange(N + 1) * h, x[i, :, c]) for i in range(5) for c in range(3)]).reshape(5, 3, -1)
    chord = x[:, :1, :].transpose(0, 2, 1) + (x[:, -1:, :] - x[:, :1, :]).transpose(0, 2, 1) * (t / T)[None, None, :]
    k = np.arange(1, K + 1)
    s = np.sqrt(2 / T) * np.sin(np.pi * k[:, None] * t[None, :] / T)
    want = np.trapezoid((L - chord)[:, :, None, :] * s[None, None], t, axis=-1)          # (5, 3, K)
    got = w.position_coeffs[:, 0, 2:, :].astype(np.float64).transpose(0, 2, 1)
    assert np.abs(got - want).max() <= 2e-4 * np.abs(want).max()
    np.testing.assert_array_equal(w.position_coeffs[:, 0, 0], x[:, 0].astype(np.float32))
    np.testing.assert_allclose(w.position_coeffs[:, 0, 1], x[:, -1] - x[:, 0], atol=1e-11)
    assert (w.m, w.n_windows, w.K, w.h) == (0, 1, K, h) and w.comp_events is None and w.boundary_bands is None and not w.has_field
    assert (w.compartment == 1).all() and band_limit(N + 1) == N - 1


def test_the_interpolation_term_is_bounded_by_D_h2_over_6():
    """A free walk sampled finely (the truth, 16x finer) against the same walk kept at every 16th sample: the bands'
    difference has the closed-form variance, below D h^2 / 6 in every band, and the walker count is large enough that
    the empirical variance sits within 3 % of the formula."""
    n, M, N, h = 6000, 16, 40, 1e-4
    xf = _free(n, N * M, h / M, 2)
    K = N - 1
    fine = _walk(xf, h / M, K)                                         # the 640-interval walk: its own error is (1/16)^2 of the coarse's
    coarse = _walk(xf[:, ::M], h, K)
    e = (coarse.position_coeffs[:, 0, 2:, :] - fine.position_coeffs[:, 0, 2:, :]).astype(np.float64)
    var = (e ** 2).mean(axis=(0, 2))                                   # per band, over walkers and axes
    k = np.arange(1, K + 1)
    exact = interpolation_variance(D0, h, K, N) - interpolation_variance(D0, h / M, K, N * M) * 0   # coarse error; fine's is 1/256 of it
    assert np.all(var <= interpolation_variance_bound(D0, h) * 1.03)
    assert abs(var.mean() / exact.mean() - 1) < 0.03, (var.mean(), exact.mean())
    assert np.all(np.diff(exact) < 0) and exact[0] == pytest.approx(interpolation_variance_bound(D0, h), rel=1e-3)
    # the band the saves resolve: the interpolated and true band variances add to the free bridge's
    true = 2 * D0 * (N * h) ** 2 / (np.pi * k) ** 2
    mine = (coarse.position_coeffs[:, 0, 2:, :].astype(np.float64) ** 2).mean(axis=(0, 2))
    assert np.abs((mine + exact) / true - 1).max() < 0.1


def test_a_cylinder_walk_re_encoded_from_its_saves_replays_within_its_certificate():
    g = d.Cylinder(radius=2.5e-6, orientation=(0, 0, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nat = d.simulate_trajectories(400, D0, g, 0.01, K=24, block_h=1e-4, seed=3, require_gpu=False, sub_steps=4, raw_subset=400)
    x = nat.raw_positions[:, ::2].astype(np.float64)                                   # saves every 0.2 ms
    h = 2e-4
    fw = walk_from_positions(x, dt_save=h, K=20, pool="intra", spec=nat.spec, D=D0, producer=PRODUCER)
    env = dict(bvals=[0.0, 1e9, 3e9], dirs=[[1, 0, 0], [0, 0, 1]], ogse_periods=[2], shortd_b=1e9,
               shortd_deltas_frac=[0.05], B0_list=[], theta_deg=[0], delta_frac=0.2, Delta_frac=0.5, rho2_list=[1e-5])
    pf = build_replay_pack(fw, id="t/foreign", license="x", citation="x", envelope=env)
    pn = build_replay_pack(nat, id="t/native", license="x", citation="x", envelope=env, K=20)
    fid = pf.fidelity
    assert fid["source"]["engine"] == "synthetic" and fid["source"]["limited_by"] == "save grid" and fid["source"]["dt_save"] == h
    assert fid["terms"]["quadrature"]["positions"] == pytest.approx(fid["err_quadrature"], rel=1e-5) and fid["err_quadrature"] > 0
    assert pf.meta["walk_params"]["producer"]["version"] == "0" and pf.meta["compression"]["channels"].get("boundary_local_time") is None
    assert "compartment" in pf.meta["compression"]["channels"]
    bound = fid["err_max"]
    for TE, delta, Delta, G in ((None, 2e-3, 5e-3, 0.08), (None, 1e-3, 3e-3, 0.15)):
        wf = d.set_b(d.pgse([[1, 0, 0], [0, 0, 1]], delta, Delta, gradient_strengths=G, n_t=pf.n_t, slew_rate=np.inf), [1e9, 1e9])
        a, b = np.asarray(pf.replay(wf, tol=None)), np.asarray(pn.replay(wf, tol=None))
        assert np.abs(a - b).max() <= bound, (np.abs(a - b).max(), bound)


@pytest.fixture(scope="module")
def good():
    return _free(20, 30, 1e-4, 5)


def test_refusals_by_name(good):
    h = 1e-4
    with pytest.raises(ValueError, match="uniform grid"):
        _walk(good, h, 8, times=np.cumsum(np.r_[0, np.full(29, h)] * np.linspace(1, 1.2, 30)))
    with pytest.raises(ValueError, match="uniform grid"):
        _walk(good, h, 8, times=h + np.arange(30) * h)                                  # does not start at t = 0
    with pytest.raises(ValueError, match="above the band limit.*at most K = 29"):
        _walk(good, h, 30)
    _walk(good, h, 29)
    with pytest.raises(ValueError, match="producer names the engine"):
        walk_from_positions(good, dt_save=h, K=8, pool="intra", spec=_spec(), D=D0, producer=None)
    with pytest.raises(ValueError, match="producer names the engine"):
        walk_from_positions(good, dt_save=h, K=8, pool="intra", spec=_spec(), D=D0, producer=dict(engine="x"))
    with pytest.raises(ValueError, match="not a pool of the substrate"):
        walk_from_positions(good, dt_save=h, K=8, pool="myelin", spec=_spec(), D=D0, producer=PRODUCER)
    with pytest.raises(TypeError, match="SubstrateSpec"):
        walk_from_positions(good, dt_save=h, K=8, pool="intra", spec=_spec().to_dict(), D=D0, producer=PRODUCER)
    wrapped = good.copy(); wrapped[:, 15:] += 5e-5                                      # a periodic wrap: a jump of 50 um
    with pytest.raises(ValueError, match="wrapped at a periodic boundary"):
        _walk(wrapped, h, 8)


def test_the_reader_refuses_the_channels_the_pack_does_not_carry(good):
    pk = build_replay_pack(_walk(good, 1e-4, 8), id="t/c0", license="x", citation="x")
    ch = pk.meta["compression"]["channels"]
    assert set(ch) == {"compartment"} and pk.meta["replay_envelope"]["surface_relaxivity"] is False and pk.meta["replay_envelope"]["field"] is False
    wf = d.set_b(d.pgse([[1, 0, 0]], 5e-4, 1.5e-3, gradient_strengths=0.05, n_t=pk.n_t, slew_rate=np.inf), [5e8])
    assert np.isfinite(pk.replay(wf, tol=None)).all()
    with pytest.raises(Exception, match="(?i)rho2|boundary_local_time|contact|C2"):
        pk.replay(wf, tissue=Tissue(rho2=1e-5), tol=None)
    with pytest.raises(Exception, match="(?i)chi|susceptib|absent"):
        pk.replay(wf, tissue=Tissue(chi_iso=1e-7), scanner=3.0, tol=None)
