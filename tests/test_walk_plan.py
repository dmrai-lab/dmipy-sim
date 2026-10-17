"""``walk_plan``: a trajectory walk states its sub-steps, walker-steps, binding rule and measured wall time before it
runs (dmrai-lab/dmipy-sim#733). It is the resolution ``simulate_trajectories`` walks, not a copy of it."""
import json
import logging
import time

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import run as R
from dmipy_sim.engine import backends as B

D = 2e-9
T, DT, N = 4e-3, 1e-3, 64


def _pack():
    r = np.array([1.0e-6, 0.8e-6, 0.9e-6, 1.1e-6])
    L = float(np.sqrt(np.pi * np.sum(r ** 2) / 0.5))
    c, _, _ = d.pack_cylinders(r + 0.05e-6, L=L, seed=3, max_attempts=20_000)
    return d.PackedCylinders(r, c, L, orientation=(0, 0, 1))


def _mesh():
    from dmipy_sim.geometry import mesh_shapes
    V, F = mesh_shapes.icosphere(2e-6, subdivisions=2)
    return d.Mesh(V, F, feature_radius=0.4e-6)


@pytest.mark.parametrize("build", [lambda: d.Sphere(radius=2e-6), _pack, _mesh], ids=["sphere", "pack", "mesh"])
def test_plan_is_what_the_walk_reports(build):
    g = build()
    p = d.walk_plan(g, D, T, DT, N)
    w = d.simulate_trajectories(N, D, g, T, DT)
    assert p["sub_steps"] == w.sub_steps
    assert p["walker_steps"] == w.work["walker_steps"] == N * (p["n_saves"] - 1) * w.sub_steps
    assert p["n_saves"] == w.positions.shape[1]
    assert p["sub_step_rules"] == w.work["step_rule"]["rules"]
    assert p["sub_steps"] == max(p["sub_step_rules"].values())
    assert p["sub_step_rules"][p["binding_rule"]] == p["sub_steps"]
    assert p["backend"] == "jax" == w.work["backend"]


def test_plan_follows_the_walks_keywords():
    g = d.Sphere(radius=2e-6)
    pinned = d.walk_plan(g, D, T, DT, N, sub_steps=7)
    assert pinned["sub_steps"] == 7 and pinned["binding_rule"] == "override"
    free = d.walk_plan(g, D, T, DT, N, tiers=())
    assert "surface local time" not in free["sub_step_rules"]
    assert "surface local time" in d.walk_plan(g, D, T, DT, N)["sub_step_rules"]
    assert d.walk_plan(g, D, T, DT, N, f_band=1.0)["sub_steps"] <= d.walk_plan(g, D, T, DT, N)["sub_steps"]
    assert d.walk_plan(g, D, T, DT, N, walker_batch_size=20)["walker_batches"] == 4
    assert d.walk_plan(g, D, T, DT, N, walker_batch_size=None)["walker_batches"] == 1
    with pytest.raises(TypeError, match="does not take"):
        d.walk_plan(g, D, T, DT, N, sub_step=3)


def test_plan_refuses_what_the_walk_refuses():
    _, _, c = d.pack_myelinated_cylinders([1e-6] * 4, 0.7, 0.5, seed=0)
    L = float(np.sqrt(np.pi * 4 * (1e-6 / 0.7) ** 2 / 0.5))
    g = d.PackedMyelinatedCylinders([1e-6] * 4, 0.7, c, L, N_max=8, rho_inner=1e-6)
    with pytest.raises(NotImplementedError, match="requires the recorded channels"):
        d.walk_plan(g, D, T, DT, N, tiers=())
    with pytest.raises(NotImplementedError, match="requires the recorded channels"):
        d.simulate_trajectories(N, D, g, T, DT, tiers=())
    with pytest.raises(ValueError, match="no backend 'nowhere' is installed"):
        d.walk_plan(d.Sphere(radius=2e-6), D, T, DT, N, backend="nowhere")


class _Fake:
    """An installed backend: accepts every request unless ``reason`` is set."""
    def __init__(self, name, reason=None):
        self.name, self.reason = name, reason

    def refuses(self, request):
        return self.reason

    def walk_batch(self, request, r0, keys, *, out=None):
        raise AssertionError("never walked")


class _EP:
    def __init__(self, obj):
        self.obj = obj

    def load(self):
        return self.obj


def _install(monkeypatch, **backends):
    monkeypatch.setattr(B, "installed", lambda: {k: _EP(v) for k, v in backends.items()})


def test_plan_names_the_backends_that_accept_it(monkeypatch):
    _install(monkeypatch, fast=_Fake("fast"), picky=_Fake("picky", "no spheres"))
    p = d.walk_plan(d.Sphere(radius=2e-6), D, T, DT, N)
    assert p["backends_accepting"] == ["fast"]
    assert p["backends_refusing"] == {"picky": "backend 'picky' refuses this walk of Sphere: no spheres"}
    # the paths only the JAX kernels implement are refused for every other backend
    q = d.walk_plan(d.Sphere(radius=2e-6), D, T, DT, N, enforce_compartment=True)
    assert q["backends_accepting"] == [] and "only the jax kernels" in q["backends_refusing"]["fast"]


def test_the_walk_warns_when_an_installed_backend_accepts_it(monkeypatch, caplog, tmp_path):
    _install(monkeypatch, fast=_Fake("fast"))
    monkeypatch.setenv("DMIPY_SIM_RUN_DIR", str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="dmipy_sim"):
        w = d.simulate_trajectories(N, D, d.Sphere(radius=2e-6), T, DT)
    rows = w.run._events
    warn = [r for r in rows if r["kind"] == "warning"]
    assert len(warn) == 1
    plan = [r for r in rows if r["kind"] == "plan"][0]
    assert "fast" in warn[0]["message"] and f"{plan['walker_steps']:.3g}" in warn[0]["message"]
    assert plan["binding_rule"] in warn[0]["message"]
    assert any(r.levelno == logging.WARNING and "fast" in r.getMessage() for r in caplog.records)


def test_the_walk_is_silent_when_no_other_backend_accepts_it(monkeypatch, caplog):
    for installed in ({}, {"picky": _Fake("picky", "no")}):
        _install(monkeypatch, **installed)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="dmipy_sim"):
            w = d.simulate_trajectories(N, D, d.Sphere(radius=2e-6), T, DT)
        assert not [r for r in w.run._events if r["kind"] == "warning"]
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING and "walker-steps" in r.getMessage()]


def test_the_plan_is_in_the_runs_events(tmp_path):
    g = d.Sphere(radius=2e-6)
    rd = tmp_path / "rec"
    # a persisted record: the walk joins the outer run, whose events.jsonl is on disk
    with R.Run("outer", run_dir=str(rd)):
        w = d.simulate_trajectories(N, D, g, T, DT)
    rows = [json.loads(l) for l in open(rd / "events.jsonl")]
    plan = [r for r in rows if r["kind"] == "plan"]
    assert len(plan) == 1
    p = d.walk_plan(g, D, T, DT, N)
    for k in ("sub_steps", "walker_steps", "binding_rule", "n_saves", "backend", "sub_step_rules"):
        assert plan[0][k] == p[k]


def test_measure_projects_the_walk():
    g = d.Sphere(radius=2e-6)
    n, t = 2000, 8e-3
    p = d.walk_plan(g, D, t, 1e-3, n, walker_batch_size=1000, measure=True)
    assert p["seconds_per_walker_step"] > 0 and p["projected_s"] > 0
    assert p["measured_batch"] == 1000 and p["measured_saves"] == 3
    assert p["projected_s"] == pytest.approx(p["walker_batches"] * p["measured_s"] / 3 * (p["n_saves"] - 1))
    # within 2x of a measured walk of the same shape (compile excluded)
    d.simulate_trajectories(n, D, g, t, 1e-3, walker_batch_size=1000)
    t0 = time.perf_counter()
    d.simulate_trajectories(n, D, g, t, 1e-3, walker_batch_size=1000)
    real = time.perf_counter() - t0
    assert real / 2 < p["projected_s"] < real * 2


def test_scaling_times_the_pilot_at_several_batches(monkeypatch):
    from dmipy_sim.engine import core
    g = d.Sphere(radius=2e-6)
    monkeypatch.setattr(core, "_free_device_bytes", lambda: None)
    p = d.walk_plan(g, D, 8e-3, 1e-3, 400, walker_batch_size=200, measure="scaling")
    s = p["scaling"]
    assert sorted(s["walker_steps_per_s"]) == [100, 200, 400, 800]
    assert s["plateau_walker_steps_per_s"] == max(s["walker_steps_per_s"].values())
    assert s["batch_fraction_of_plateau"] == pytest.approx(s["walker_steps_per_s"][200] / s["plateau_walker_steps_per_s"])
    assert s["max_batch_memory"] is None and "unavailable" in s["memory_cap"]
    want = "under-filled" if s["seconds"][400] < core.UNDERFILLED_DOUBLING * s["seconds"][200] else "saturated"
    assert s["verdict"] == want
    # a device that holds 1.5 batches: 2B does not fit
    per = core._device_bytes_per_walker(core._resolve_walk(g, D, 8e-3, 1e-3))
    monkeypatch.setattr(core, "_free_device_bytes", lambda: int(1.5 * 200 * per))
    q = d.walk_plan(g, D, 8e-3, 1e-3, 400, walker_batch_size=200, measure="scaling")["scaling"]
    assert sorted(q["walker_steps_per_s"]) == [100, 200] and q["verdict"] == "memory-bound"
    assert q["max_batch_memory"] == 300
