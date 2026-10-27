"""The paper families' re-walk driver (``tools/family_rewalk.py``): its walker rule holds the toggle-proof bound
at the floor it is given and is the cheapest set of counts that does, and the envelope it certifies under is
the family's own declaration over the campaign's."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fr():
    spec = importlib.util.spec_from_file_location("family_rewalk", ROOT / "tools" / "family_rewalk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_equal_costs_seed_in_proportion_to_the_water_and_hold_the_floor(fr):
    shares = {"intra": 0.3, "myelin": 0.2, "extra": 0.5}
    n = fr.toggle_proof_counts(shares, 5e-3)
    assert n == {"intra": 12000, "myelin": 8000, "extra": 20000}
    assert fr.toggle_proof_bound(shares, n) <= 5e-3 * (1 + 1e-12)
    assert fr.toggle_proof_counts({"water": 1.0}, 1e-2) == {"water": 10000}


def test_the_cost_weighted_counts_are_the_cheapest_that_hold_the_floor(fr):
    shares, costs, f = {"a": 0.4, "b": 0.6}, {"a": 9.0, "b": 1.0}, 1e-2
    n = fr.toggle_proof_counts(shares, f, costs)
    assert fr.toggle_proof_bound(shares, n) <= f * (1 + 1e-9)
    cost = lambda m: sum(m[p] * costs[p] for p in m)
    for t in np.linspace(0.2, 0.8, 13):                     # every other split on the bound's own surface costs more
        na = shares["a"] ** 2 / (t * f ** 2)
        nb = shares["b"] ** 2 / ((1 - t) * f ** 2)
        assert cost(n) <= cost({"a": na, "b": nb}) * (1 + 1e-3)


def test_a_floor_or_share_that_is_not_positive_is_refused(fr):
    with pytest.raises(ValueError):
        fr.toggle_proof_counts({"water": 1.0}, 0.0)
    with pytest.raises(ValueError):
        fr.toggle_proof_counts({"water": 0.0}, 1e-2)
    with pytest.raises(ValueError):
        fr.toggle_proof_counts({"a": 1.0}, 1e-2, {"a": 0.0})


def test_a_family_declaration_is_stated_over_the_campaign_envelope(fr):
    env = fr.family_envelope({"rho2_list": [1e-5], "bvals": [0.0, 1e9]}, {"B0_list": [3.0]}, max_refocus_pulses=8)
    assert env["scanner"] == fr.ENVELOPE["scanner"]
    assert env["rho2_list"] == [1e-5] and env["bvals"] == [0.0, 1e9]
    assert env["B0_list"] == [3.0] and env["max_refocus_pulses"] == 8
    assert "ogse_periods" in env                             # the library's battery shapes underneath


# ------------------------------------------------------------------ the pilot's variance bound
G_MAX = 0.5                       # T/m per axis: the Connectome 2.0 class
D0 = 2e-9
T_WIN = 0.01


def _pilot(diameter_m, n=800, seed=3):
    """A sphere's pilot: the pore confined on every axis, so every feasible waveform is in the bound's scope."""
    import dmipy_sim as d
    w = d.simulate_trajectories(n, D0, d.Sphere(radius=0.5 * diameter_m), T_max=T_WIN, K=26, seed=seed, tiers="all",
                                require_gpu=False, window_T=T_WIN, raw_subset=n)
    return w, float(w.block["h"])


N_WF = 2000                       # samples of a test waveform over the window (sample-and-hold)


def _refocused(G):
    """``G`` (n_wf, 3) with zero mean per axis (``int G dt = 0`` for the sample-and-hold), scaled back within
    ``G_MAX``."""
    G = G - G.mean(axis=0)
    top = np.abs(G).max()
    return G * (G_MAX / top) if top > G_MAX else G


def _waveforms(rng, adversary):
    """Feasible refocused waveforms: random slew-limited, PGSE along x and the diagonal, cosine OGSE, and the sign
    pattern of the bound's own maximiser (the adversary)."""
    t = (np.arange(N_WF) + 0.5) * T_WIN / N_WF
    out = []
    for _ in range(20):
        steps = rng.uniform(-1, 1, (N_WF, 3)) * (0.05 * G_MAX)
        out.append(_refocused(np.clip(np.cumsum(steps, axis=0), -G_MAX, G_MAX)))
    for axes in ([1, 0, 0], [1, 1, 1]):
        for d_frac in (0.1, 0.3, 0.45):
            lobe = ((t >= 0.02 * T_WIN) & (t < (0.02 + d_frac) * T_WIN)).astype(float)
            lobe2 = ((t >= 0.52 * T_WIN) & (t < (0.52 + d_frac) * T_WIN)).astype(float)
            out.append(_refocused(G_MAX * (lobe - lobe2)[:, None] * np.asarray(axes, float)[None, :]))
    for periods in (1, 2, 4, 8, 16):
        out.append(_refocused(G_MAX * np.cos(2 * np.pi * periods * t / T_WIN)[:, None] * np.ones(3)[None, :]))
    out.append(_refocused(G_MAX * np.sign(adversary)))
    return out


def _adversary(fr, C, K):
    """The bound's maximiser as a function of time, ``sum_j alpha_j P g_j`` with ``alpha = A^-1/2 u``, ``u`` the top
    eigenvector of ``A^1/2 S A^1/2`` over the three axes."""
    A = fr.projected_gram(T_WIN, K)
    ev, U = np.linalg.eigh(A)
    R, Ri = (U * np.sqrt(ev)) @ U.T, (U / np.sqrt(ev)) @ U.T
    Y = np.einsum("jl,wla->wja", R, C[:, 1:, :] - C[:, 1:, :].mean(0))
    n = Y.shape[0]
    u = np.linalg.eigh(Y.reshape(n, -1).T @ Y.reshape(n, -1))[1][:, -1].reshape(K + 1, 3)
    alpha = Ri @ u
    t = (np.arange(N_WF) + 0.5) * T_WIN / N_WF
    k = np.arange(1, K + 1)
    g = np.concatenate([(t / T_WIN)[None, :], np.sqrt(2 / T_WIN) * np.sin(np.pi * np.outer(k, t) / T_WIN)])
    g = g - g.mean(axis=1, keepdims=True)
    return np.einsum("ja,jt->ta", alpha, g)


def _var(Z):
    Z = np.asarray(Z)
    return float(np.mean(np.abs(Z - Z.mean()) ** 2) * len(Z) / (len(Z) - 1))


def test_the_projected_gram_is_the_quadrature_of_its_functions(fr):
    K, n = 12, 200_000
    t = (np.arange(n) + 0.5) * T_WIN / n
    k = np.arange(1, K + 1)
    g = np.concatenate([(t / T_WIN)[None, :], np.sqrt(2 / T_WIN) * np.sin(np.pi * np.outer(k, t) / T_WIN)])
    g = g - g.mean(axis=1, keepdims=True)
    assert np.allclose(fr.projected_gram(T_WIN, K), g @ g.T * (T_WIN / n), atol=1e-9 * T_WIN)


def test_no_feasible_waveform_or_toggle_exceeds_the_pilot_bound(fr):
    from dmipy_sim.constants import GAMMA
    from dmipy_sim.replay.compression import band_projection
    walk, _ = _pilot(2e-6)
    C = np.asarray(walk.position_coeffs, np.float64)[:, 0]
    K = C.shape[1] - 2
    ph = fr.phase_variance_bound(walk.position_coeffs, T_WIN, G_max=G_MAX)
    V0, _ = fr.toggle_variance_bound(ph["var_phi"], None, D=D0, rho2_max=0.0)
    V3, _ = fr.toggle_variance_bound(ph["var_phi"], walk.raw_boundary, D=D0, rho2_max=3e-6)
    assert 0.0 < ph["var_phi"] < 1.0 and V0 <= V3 <= 1.0
    L_end = np.asarray(walk.raw_boundary, np.float64)[:, 1:].sum(1)
    worst = 0.0
    for G in _waveforms(np.random.default_rng(0), _adversary(fr, C, K)):
        assert np.abs(G).max() <= G_MAX * (1 + 1e-12)
        W = band_projection(G[None], T_WIN / N_WF, T_WIN, K)[0]                 # (K + 2, 3): the replay's weights
        assert np.abs(W[0]).max() < 1e-12 * G_MAX * T_WIN
        phi = GAMMA * np.einsum("wja,ja->w", C, W)                            # the phase the pack serves
        e = np.exp(1j * phi)
        assert _var(e) <= V0 + 1e-12
        worst = max(worst, float(np.var(phi, ddof=1)))
        for rho in (1e-6, 3e-6):
            assert _var(np.exp((rho / D0) * L_end) * e) <= V3 + 1e-12
    assert worst <= ph["var_phi"]
    assert worst >= 0.2 * ph["var_phi"]                    # the adversary reaches the bound's order: it is not slack


def test_a_large_pore_takes_the_whole_toggle_proof_count_and_a_small_one_fewer(fr):
    """A pore wider than the class's dephasing length needs the whole a / f^2; a smaller one, fewer; a free axis is
    never confined, so a plane's bound over all three axes is 1 whatever its width."""
    walk, _ = _pilot(20e-6, n=400)
    ph = fr.phase_variance_bound(walk.position_coeffs, T_WIN, G_max=G_MAX)
    V, _ = fr.toggle_variance_bound(ph["var_phi"], walk.raw_boundary, D=D0, rho2_max=3e-6)
    assert V == pytest.approx(1.0)
    assert fr.pilot_count(V, 5e-3, 4000) == 40000
    small, _ = _pilot(1e-6, n=400)
    ph2 = fr.phase_variance_bound(small.position_coeffs, T_WIN, G_max=G_MAX)
    V2, _ = fr.toggle_variance_bound(ph2["var_phi"], small.raw_boundary, D=D0, rho2_max=3e-6)
    assert V2 < 0.1 and fr.pilot_count(V2, 5e-3, 4000) < 40000
    import dmipy_sim as d
    plane = d.simulate_trajectories(200, D0, d.Box1D(length=1e-6), T_max=T_WIN, K=26, seed=5, tiers="all",
                                    require_gpu=False, window_T=T_WIN, raw_subset=200)
    free = fr.phase_variance_bound(plane.position_coeffs, T_WIN, G_max=G_MAX)
    normal = fr.phase_variance_bound(plane.position_coeffs, T_WIN, G_max=G_MAX, axes=(0,))
    assert free["var_phi"] > 1.0 > normal["var_phi"]


def test_a_pore_at_n_min_is_walked_once_and_a_larger_one_afresh(fr, tmp_path, monkeypatch):
    """``plan --pilot n_min`` packs a pore whose count is ``n_min`` from its pilot (walked once, at the pack's seed,
    its receipt saying so); a pore that needs more is walked by ``walk --plan`` at its count from a seed of its own;
    a walk resumed walks nothing."""
    import json
    import dmipy_sim as d
    from dmipy_sim.replay.reference import pack_seed
    calls = []
    real = d.simulate_trajectories

    def counted(n, *a, seed=42, **kw):
        calls.append((int(n), int(seed)))
        return real(n, *a, seed=seed, **kw)

    monkeypatch.setattr(d, "simulate_trajectories", counted)
    common = ["--floor", "0.05", "--T", str(T_WIN), "--diameters", "1", "20", "--backend", "jax", "--out", str(tmp_path)]
    fr.main(["plan", "pores", "--pilot", "200", "--n-min", "200"] + common)
    plan = tmp_path / "plan_pores.json"
    fr.main(["walk", "pores", "--plan", str(plan), "--allow-jax"] + common)
    rows = {r["name"]: r for r in json.loads(plan.read_text())["rows"]}
    for name, row in rows.items():
        rid = row["id"]
        receipt = json.loads((tmp_path / "receipts" / f"{name}.json").read_text())
        mine = [c for c in calls if c[1] in (pack_seed(rid), pack_seed(rid + fr.AFTER_PILOT))]
        if name.endswith("d01.00um"):
            assert row["pilot"]["pilot_is_pack"] and row["n_walkers"] == 200
            assert mine == [(200, pack_seed(rid))]                         # walked once: the pilot is the pack
            assert receipt["walked_as"] == "pilot" and receipt["n_walkers"] == 200
        else:
            assert not row["pilot"]["pilot_is_pack"] and row["n_walkers"] == 400
            assert mine == [(200, pack_seed(rid)), (400, pack_seed(rid + fr.AFTER_PILOT))]
            assert receipt["walked_as"] == "walk" and receipt["n_walkers"] == 400
        assert row["pilot"]["certified_axes"] == list(fr.Pores.CONFINED[name.split("-")[0]])
    n_calls = len(calls)
    fr.main(["walk", "pores", "--plan", str(plan), "--allow-jax"] + common)
    fr.main(["plan", "pores", "--pilot", "200", "--n-min", "200"] + common)
    assert len(calls) == n_calls                                           # resumed: nothing is walked again
