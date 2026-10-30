"""The gradient side of a certificate stated for a scanner class.

A waveform the class plays keeps every axis within ``G_max`` over the window ``T``. Its band weights ``W_k = int G s_k
dt`` on the orthonormal sine basis then obey Bessel's inequality, ``sum_k W_k^2 <= int G_axis^2 dt <= G_max^2 T`` per
axis, so ``gamma^2 sum_{k, axis} W_k^2 <= 3 gamma^2 G_max^2 T`` (``compression.class_phase_weight``) whatever its shape,
direction, timing or b; and a refocused axis has ``|q(t)| <= gamma G_max min(t, T - t)``, so ``b <= gamma^2 G_max^2 T^3
/ 4`` (``compression.class_b_max``). A pack certified for a class reads its position terms against these, and a segment
layout its widths against the class's levels.
"""
import dmipy_replay.replay.compression as _r_compression
import dmipy_replay.replay.time_segments as _r_time_segments
import warnings

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_replay.constants import GAMMA
from dmipy_sim.replay import compression as cx
from dmipy_sim.replay import time_segments as ts
from dmipy_sim.replay.bank import build_replay_pack

G_MAX, SLEW, T, N_T = 0.5, 600.0, 0.1, 1001
DT = T / (N_T - 1)


def _slew_limited(target, rng_slew=SLEW):
    """``target`` (n_t, 3) followed as closely as the slew allows, clipped at G_max per axis."""
    out = np.zeros_like(target)
    step = rng_slew * DT
    for i in range(1, target.shape[0]):
        out[i] = np.clip(out[i - 1] + np.clip(target[i] - out[i - 1], -step, step), -G_MAX, G_MAX)
    return out


def _refocused(G):
    """``G`` with each axis's net moment removed by a last lobe within G_max, or None when it cannot be."""
    m = G.sum(axis=0) * DT
    G = G.copy()
    tail = int(0.2 * N_T)
    need = -m / (tail * DT)
    if np.any(np.abs(G[-tail:] + need) > G_MAX):
        return None
    G[-tail:] += need
    return G


def _feasible(rng):
    """Random feasible waveforms of many kinds: slew-limited random levels, bang-bang, diagonal PGSE, OGSE, PGSTE."""
    t = np.arange(N_T) * DT
    out = []
    for _ in range(40):
        out.append(_slew_limited(rng.uniform(-G_MAX, G_MAX, (N_T, 3)) * (rng.random((1, 3)) < 0.8)))
        sw = np.sign(np.sin(2 * np.pi * rng.uniform(1, 40) * t[:, None] + rng.uniform(0, 2 * np.pi, 3)))
        out.append(_slew_limited(G_MAX * sw))
    for delta, Delta in ((0.01, 0.03), (0.02, 0.05), (0.04, 0.05)):
        lobe = ((t < delta).astype(float) - ((t >= Delta) & (t < Delta + delta)).astype(float))[:, None]
        out.append(_slew_limited(G_MAX * lobe * np.ones((1, 3))))                    # diagonal: |G| = sqrt(3) G_max
    for n in (1, 3, 10, 40):
        out.append(G_MAX * np.cos(2 * np.pi * n * t / T)[:, None] * np.array([[1.0, -1.0, 1.0]]))
    lobe, TM = 0.01, 0.05                                                              # PGSTE: lobe, mixing, lobe
    st = ((t < lobe).astype(float) - ((t >= lobe + TM) & (t < 2 * lobe + TM)).astype(float))[:, None]
    out.append(_slew_limited(G_MAX * st * np.array([[1.0, 0.0, 1.0]])))
    return out


def test_no_waveform_of_the_class_exceeds_the_bessel_bound_and_bang_bang_attains_it():
    rng = np.random.default_rng(0)
    bound = cx.class_phase_weight(G_MAX, T)
    K = N_T - 1
    worst = 0.0
    for G in _feasible(rng):
        assert np.abs(G).max() <= G_MAX * (1 + 1e-12)
        W = _r_compression.band_projection(G[None, :-1], DT, T, K)[0, 2:, :]
        S = float(GAMMA ** 2 * (W ** 2).sum())
        assert S <= bound * (1 + 1e-9), S / bound
        worst = max(worst, S)
    # bang-bang on every axis at G_max, the extremal waveform: Bessel's inequality is an equality in the limit of all bands
    bb = G_MAX * np.sign(np.sin(2 * np.pi * 3 * np.arange(N_T)[:, None] * DT / T + np.array([[0.3, 1.1, 2.0]])))
    W = _r_compression.band_projection(bb[None, :-1], DT, T, K)[0, 2:, :]
    assert float(GAMMA ** 2 * (W ** 2).sum()) >= 0.95 * bound
    assert worst >= 0.5 * bound                                         # the slew-limited ones come within 2x of it


def test_no_refocused_waveform_of_the_class_exceeds_the_b_bound():
    rng = np.random.default_rng(1)
    bound = cx.class_b_max(G_MAX, T)
    n = 0
    for G in _feasible(rng):
        G = _refocused(G)
        if G is None:
            continue
        q = GAMMA * np.cumsum(G, axis=0) * DT
        assert float((q ** 2).sum() * DT) <= bound * (1 + 1e-9)
        n += 1
    assert n >= 20
    t = np.arange(N_T) * DT                                             # the extremal: +G_max then -G_max on every axis
    G = G_MAX * np.where(t < T / 2, 1.0, -1.0)[:, None] * np.ones((1, 3))
    q = GAMMA * np.cumsum(G, axis=0) * DT
    assert float((q ** 2).sum() * DT) == pytest.approx(bound, rel=1e-2)


def test_the_envelope_of_a_class_states_it_and_its_phase_weight_is_the_class_bound():
    env = cx.certified_envelope(dict(cx.default_envelope(), scanner="connectome_2"))
    assert env["scanner"] == "connectome_2" and env["G_max"] == 0.5 and env["slew_max"] == 600.0
    assert cx._phase_weight(64, N_T, DT, cx._env_key(env)) == pytest.approx(cx.class_phase_weight(0.5, T))
    G, meta = cx.acquisition_battery(N_T, DT, env)
    top = [m for m in meta if m.get("class_amplitude")]
    assert top and all(np.abs(G[i]).max() <= 0.5 * (1 + 1e-9) for i in range(len(meta)))
    assert max(np.abs(G[i]).max() for i, m in enumerate(meta) if m.get("class_amplitude")) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="no scanner"):
        cx.certified_envelope(dict(cx.default_envelope(), G_max=0.5))


@pytest.fixture(scope="module")
def walk():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return d.simulate_trajectories(600, 2e-9, d.Cylinder(radius=4e-6, orientation=[0, 0, 1]), 0.02, 2.5e-4, seed=2,
                                       require_gpu=False, K=32, window_T=0.02)


def test_a_pack_certified_for_a_class_reads_its_position_terms_against_it(walk):
    kw = dict(id="t/class", license="x", citation="x", K=32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cls = build_replay_pack(walk, envelope=dict(cx.default_envelope(), scanner="connectome_2"), **kw)
        bat = build_replay_pack(walk, **kw)
    acq = cls.replay_envelope["acquisition"]
    assert acq["bound"] == "scanner_class" and acq["scanner"] == "connectome_2" and acq["G_max"] == 0.5
    T_w = float(cls.T_window)
    assert acq["b_max"] == pytest.approx(cx.class_b_max(0.5, T_w))
    f = cls.fidelity
    assert f["gradient_bound"]["G_max_per_axis"] == 0.5
    k = np.arange(1, 33)
    rel = cx.quadrature_rel(int(f["quadrature"]["m"]), float(f["quadrature"]["h"]), np.pi * k / T_w)
    qv = np.asarray(cls.meta["walk_params"]["qv"]["positions"], np.float64)
    expect = 0.5 * GAMMA ** 2 * 0.25 * T_w * float((rel * T_w / (np.pi * k) ** 2).max()) * float(qv.sum())
    assert f["err_quadrature"] == pytest.approx(expect, rel=1e-3)
    assert "bound" not in bat.replay_envelope["acquisition"] and "gradient_bound" not in bat.fidelity


def test_a_layout_of_a_class_derives_its_widths_from_the_class_levels():
    seq = d.pgse([[1, 0, 0]], 4e-3, 12e-3, bvalues=[1e9], TE=0.03, n_t=301, slew_rate=np.inf)
    p = _r_time_segments.Partition.of("pgse", seq)
    idx, V = p.class_vertices(G_MAX)
    d_ = p.durations[idx]
    assert np.allclose(V @ d_, 0.0) and np.abs(V).max() == pytest.approx(G_MAX)
    reach = p.reach(G_MAX)
    assert reach[idx].max() == pytest.approx(np.sqrt(3.0) * G_MAX)
    # the bound over the class's levels is the largest over random feasible levels, attained at a vertex
    rng = np.random.default_rng(3)
    A = rng.normal(size=(3, idx.size, idx.size)); A = np.einsum("cij,ckj->cik", A, A)
    env = dict(G_max=G_MAX, b_max=None)
    top = ts.level_bound(p, A, env)
    for _ in range(2000):
        l = rng.uniform(-G_MAX, G_MAX, (3, idx.size))
        l -= np.outer(l @ d_, d_) / (d_ @ d_)                              # onto the refocused plane
        if np.abs(l).max() > G_MAX:
            continue
        assert float(np.einsum("cs,cst,ct->", l, A, l)) <= top * (1 + 1e-9)
    assert top == pytest.approx(sum(float(np.einsum("vs,st,vt->v", V, A[c], V).max()) for c in range(3)))


@pytest.fixture(scope="module")
def disco_timing_layout(tmp_path_factory):
    """A pack certified for connectome_2 over a window that holds DiSCo's b = 3091 s/mm^2 shell (ActiveAx delta 7.6 ms,
    Delta 45.9 ms, TE 53.5 ms), its segment layout of that shell's partition, and the shell's sequence."""
    from dmipy_sim.phantom import Grid
    tmp = tmp_path_factory.mktemp("class_layout")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = d.simulate_trajectories(300, 2e-9, d.Cylinder(radius=4e-6, orientation=[0, 0, 1]), 0.06, 5e-4, seed=4,
                                    require_gpu=False, K=64, window_T=0.06)
        pk = build_replay_pack(w, id="t/class-layout", license="x", citation="x", K=64,
                               envelope=dict(cx.default_envelope(), scanner="connectome_2"))
    seq = d.pgse([[1, 0, 0]], 7.6e-3, 45.9e-3, bvalues=[3.091e9], TE=0.0535, n_t=1071, slew_rate=np.inf)
    grid = Grid(shape=(1, 1, 1), voxel_size_m=(1e-4,) * 3, origin_m=(0.0,) * 3, attach="substrate")    # one voxel about the axis
    ts.segment_rows(pk, {"b3091": seq}, str(tmp / "rows"), grid=grid, floor_target=0.05)
    ts.write_time_segment_layout([str(tmp / "rows")], str(tmp / "layout"))
    return ts.TimeSegmentLayout(str(tmp / "layout")), seq


def test_a_layout_of_a_class_reads_any_level_the_class_plays_and_refuses_one_beyond_it(disco_timing_layout):
    """No b refuses a read of a class's layout: DiSCo's b = 3091 s/mm^2 shell (131 mT/m on x) and the class's largest
    level on every axis at once (b far above any battery's) are served, a level above G_max on one axis is refused."""
    lay, seq = disco_timing_layout
    env = lay.manifest["source"]["envelope"]
    assert env["bound"] == "scanner_class" and env["b_max"] is None and env["G_max"] == 0.5
    S, _ = lay.image(seq)
    assert np.isfinite(np.asarray(S)).any()
    G = np.asarray(seq.G, np.float64)
    top = G / np.abs(G).max() * 0.5
    top = np.concatenate([top[..., :1]] * 3, axis=-1)                    # 0.5 T/m on x, y and z: |G| = 0.87 T/m
    lay.image(seq.with_gradient(top))
    with pytest.raises(ValueError, match="on one axis"):
        lay.image(seq.with_gradient(top * 1.01))
