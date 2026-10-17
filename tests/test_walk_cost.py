"""``walk_cost``: a trajectory walk states its sub-steps, walker-steps, binding rule and measured wall time before it
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


def cost(g, T_max=T, dt=DT, n=N, **kw):
    """``walk_cost`` on the test's save grid and no scanner class (the save grid's Nyquist band)."""
    kw.setdefault("scanner", None)
    return d.walk_cost(g, D, n, T_max=T_max, dt_save=dt, **kw)


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
def test_cost_is_what_the_walk_reports(build):
    g = build()
    p = cost(g)
    w = d.simulate_trajectories(N, D, g, T, DT)
    assert p.sub_steps == w.sub_steps
    assert p.walker_steps == w.work["walker_steps"] == N * (p.n_saves - 1) * w.sub_steps
    assert p.n_saves == w.positions.shape[1]
    assert p.sub_step_rules == w.work["step_rule"]["rules"]
    assert p.sub_steps == max(p.sub_step_rules.values())
    assert p.sub_step_rules[p.binding_rule] == p.sub_steps
    assert p.backend == "jax" == w.work["backend"]


def test_cost_follows_the_walks_keywords():
    g = d.Sphere(radius=2e-6)
    pinned = cost(g, sub_steps=7)
    assert pinned.sub_steps == 7 and pinned.binding_rule == "override"
    free = cost(g, tiers=())
    assert "surface local time" not in free.sub_step_rules
    assert "surface local time" in cost(g).sub_step_rules
    assert cost(g, scanner="prisma").sub_steps <= cost(g).sub_steps
    assert cost(g, walker_batch_size=20).walker_batches == 4
    assert cost(g, walker_batch_size=None).walker_batches == 1
    with pytest.raises(TypeError, match="does not take"):
        cost(g, sub_step=3)


def test_cost_refuses_what_the_walk_refuses():
    _, _, c = d.pack_myelinated_cylinders([1e-6] * 4, 0.7, 0.5, seed=0)
    L = float(np.sqrt(np.pi * 4 * (1e-6 / 0.7) ** 2 / 0.5))
    g = d.PackedMyelinatedCylinders([1e-6] * 4, 0.7, c, L, N_max=8, rho_inner=1e-6)
    with pytest.raises(NotImplementedError, match="requires the recorded channels"):
        cost(g, tiers=())
    with pytest.raises(NotImplementedError, match="requires the recorded channels"):
        d.simulate_trajectories(N, D, g, T, DT, tiers=())
    with pytest.raises(ValueError, match="no backend 'nowhere' is installed"):
        cost(d.Sphere(radius=2e-6), backend="nowhere")


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


def test_cost_names_the_backends_that_accept_it(monkeypatch):
    _install(monkeypatch, fast=_Fake("fast"), picky=_Fake("picky", "no spheres"))
    p = cost(d.Sphere(radius=2e-6))
    assert p.backends_accepting == ["fast"]
    assert p.backends_refusing == {"picky": "backend 'picky' refuses this walk of Sphere: no spheres"}
    # the paths only the JAX kernels implement are refused for every other backend
    q = cost(d.Sphere(radius=2e-6), enforce_compartment=True)
    assert q.backends_accepting == [] and "only the jax kernels" in q.backends_refusing["fast"]


def test_the_walk_warns_when_an_installed_backend_accepts_it(monkeypatch, caplog, tmp_path):
    _install(monkeypatch, fast=_Fake("fast"))
    monkeypatch.setenv("DMIPY_SIM_RUN_DIR", str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="dmipy_sim"):
        w = d.simulate_trajectories(N, D, d.Sphere(radius=2e-6), T, DT)
    rows = w.run._events
    warn = [r for r in rows if r["kind"] == "warning"]
    assert len(warn) == 1
    c = [r for r in rows if r["kind"] == "cost"][0]
    assert "fast" in warn[0]["message"] and f"{c['walker_steps']:.3g}" in warn[0]["message"]
    assert c["binding_rule"] in warn[0]["message"]
    assert any(r.levelno == logging.WARNING and "fast" in r.getMessage() for r in caplog.records)


def test_the_walk_is_silent_when_no_other_backend_accepts_it(monkeypatch, caplog):
    for installed in ({}, {"picky": _Fake("picky", "no")}):
        _install(monkeypatch, **installed)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="dmipy_sim"):
            w = d.simulate_trajectories(N, D, d.Sphere(radius=2e-6), T, DT)
        assert not [r for r in w.run._events if r["kind"] == "warning"]
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING and "walker-steps" in r.getMessage()]


def test_the_cost_is_in_the_runs_events(tmp_path):
    g = d.Sphere(radius=2e-6)
    rd = tmp_path / "rec"
    # a persisted record: the walk joins the outer run, whose events.jsonl is on disk
    with R.Run("outer", run_dir=str(rd)):
        w = d.simulate_trajectories(N, D, g, T, DT)
    rows = [json.loads(l) for l in open(rd / "events.jsonl")]
    c = [r for r in rows if r["kind"] == "cost"]
    assert len(c) == 1
    p = cost(g)
    for k in ("sub_steps", "walker_steps", "binding_rule", "n_saves", "backend", "sub_step_rules"):
        assert c[0][k] == getattr(p, k)


def test_measure_projects_the_walk():
    g = d.Sphere(radius=2e-6)
    n, t = 2000, 8e-3
    p = cost(g, t, 1e-3, n, walker_batch_size=1000, measure=True)
    m = p.measured
    assert m.seconds_per_walker_step > 0 and m.projected_s > 0
    assert m.batch == 1000 and m.saves == 1
    assert m.projected_s == pytest.approx(p.walker_batches * m.seconds / m.saves * (p.n_saves - 1))
    assert cost(g, t, 1e-3, n).measured is None and cost(g, t, 1e-3, n).scaling is None
    # within 2x of a measured walk of the same shape (compile excluded)
    d.simulate_trajectories(n, D, g, t, 1e-3, walker_batch_size=1000)
    t0 = time.perf_counter()
    d.simulate_trajectories(n, D, g, t, 1e-3, walker_batch_size=1000)
    real = time.perf_counter() - t0
    assert real / 2 < m.projected_s < real * 2


def test_scaling_times_the_pilot_at_several_batches(monkeypatch):
    from dmipy_sim.engine import core
    g = d.Sphere(radius=2e-6)
    monkeypatch.setattr(core, "_free_device_bytes", lambda: None)
    p = cost(g, 8e-3, 1e-3, 400, walker_batch_size=200, measure="scaling")
    s = p.scaling
    assert sorted(s.walker_steps_per_s) == [100, 200, 400, 800]
    assert s.plateau_walker_steps_per_s == max(s.walker_steps_per_s.values())
    assert s.batch_fraction_of_plateau == pytest.approx(s.walker_steps_per_s[200] / s.plateau_walker_steps_per_s)
    assert s.max_batch_memory is None and "unavailable" in s.memory_cap
    want = "under-filled" if s.seconds[400] < core.UNDERFILLED_DOUBLING * s.seconds[200] else "saturated"
    assert s.verdict == want
    # a device that holds 1.5 batches: 2B does not fit
    per = core._device_bytes_per_walker(core._resolve_walk(g, D, 8e-3, 1e-3))
    monkeypatch.setattr(core, "_free_device_bytes", lambda: int(1.5 * 200 * per))
    q = cost(g, 8e-3, 1e-3, 400, walker_batch_size=200, measure="scaling").scaling
    assert sorted(q.walker_steps_per_s) == [100, 200] and q.verdict == "memory-bound"
    assert q.max_batch_memory == 300


def test_a_cost_is_frozen_and_its_row_is_its_fields():
    import dataclasses
    p = cost(d.Sphere(radius=2e-6))
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.sub_steps = 1
    assert p.dt_save_rule == "given" and p.scanner is None


def test_dt_save_none_is_the_scanners_save_interval():
    from dmipy_sim.acquisition.scanners import save_interval
    g = d.Sphere(radius=2e-6)
    c = d.walk_cost(g, D, N, T_max=T, scanner="prisma")
    assert c.dt_save == save_interval(T, N, "prisma", D=D, field=False)
    assert c.dt_save_rule == "save_interval(prisma)" and c.scanner == "prisma"
    given = d.walk_cost(g, D, N, T_max=T, scanner="prisma", dt_save=DT)
    assert given.dt_save == DT and given.dt_save_rule == "given"
    # a stronger gradient needs a finer save grid: more saves, more bytes per walker
    strong = d.walk_cost(g, D, N, T_max=T, scanner="connectome_2")
    assert strong.dt_save < c.dt_save and strong.n_saves > c.n_saves and strong.bytes_per_walker > c.bytes_per_walker
    with pytest.raises(ValueError, match="name a scanner"):
        d.walk_cost(g, D, N, T_max=T, scanner=None)


def test_the_scanners_band_reaches_the_sub_step_rules():
    from dmipy_sim.replay.bank import class_band
    from dmipy_sim.engine.physics import sub_step_rules
    g = d.Sphere(radius=20e-6)                  # a pore so wide that the band, not the pore, sizes the step
    c = d.walk_cost(g, D, N, T_max=T, scanner="prisma", dt_save=DT)
    assert c.f_band_hz == class_band("prisma") != 0.5 / DT
    assert c.sub_step_rules == sub_step_rules(g, D, DT, surface=True, f_band=class_band("prisma"))
    assert c.sub_step_rules != cost(g).sub_step_rules and cost(g).f_band_hz == 0.5 / DT


def test_the_sweep_stops_after_the_rate_falls_and_recommends_the_fastest():
    from dmipy_sim.engine import core
    # walkers per second: 4096 -> 4096, 8192 -> 7447, 16384 -> 11916, 32768 -> 7282 (falls: the sweep stops there)
    t = {4096: 1.0, 8192: 1.1, 16384: 1.375, 32768: 4.5, 65536: 9.0}
    times, why = core._sweep(lambda b: t[b], 10 ** 6, "memory")
    assert sorted(times) == [4096, 8192, 16384, 32768] and why == "rate fell"
    assert core._recommended(times) == 16384
    # the rate rises to the cap, which is timed itself: the sweep ends at the limit, recommending the cap
    times, why = core._sweep(lambda b: b ** 0.5, 20000, "n_walkers")
    assert sorted(times) == [4096, 8192, 16384, 20000] and why == "n_walkers" and core._recommended(times) == 20000
    # under one rung of walkers: the cap alone
    times, why = core._sweep(lambda b: 1.0, 100, "n_walkers")
    assert list(times) == [100] and why == "n_walkers"


def test_the_verdict_is_the_doubling_at_the_recommended_batch(monkeypatch):
    from dmipy_sim.engine import core
    g = d.Sphere(radius=2e-6)
    res = core._resolve_walk(g, D, 2e-3, 1e-3)
    # nearly free doubling -> under-filled; a proportional one -> saturated; 2B beyond the device -> memory-bound
    assert core._scaling({100: 1.0, 200: 1.1}, 100, res, 1, None, "given").verdict == "under-filled"
    assert core._scaling({100: 1.0, 200: 1.2}, 100, res, 1, None, "given").verdict == "saturated"
    assert core._scaling({100: 1.0}, 100, res, 1, 150, "given").verdict == "memory-bound"
    assert core._scaling({100: 1.0}, 100, res, 1, None, "given").verdict == "unresolved"


def test_no_batch_given_is_n_walkers_capped_by_memory(monkeypatch):
    from dmipy_sim.engine import core
    g = d.Sphere(radius=2e-6)
    monkeypatch.setattr(core, "_free_device_bytes", lambda: None)
    c = cost(g)
    assert c.walker_batch_size == N and c.walker_batches == 1 and c.walker_batch_rule == "n_walkers capped by memory"
    per = c.bytes_per_walker
    monkeypatch.setattr(core, "_free_device_bytes", lambda: 20 * per)
    c = cost(g)
    assert c.walker_batch_size == 20 and c.walker_batches == 4 and c.walker_batch_rule == "n_walkers capped by memory"
    assert cost(g, walker_batch_size=30).walker_batch_rule == "given"


def test_scaling_finds_the_batch_within_the_memory_cap(monkeypatch):
    from dmipy_sim.engine import core
    g = d.Sphere(radius=2e-6)
    per = core._device_bytes_per_walker(core._resolve_walk(g, D, 2e-3, 1e-3))
    monkeypatch.setattr(core, "_free_device_bytes", lambda: int(9000.5 * per))
    c = cost(g, 2e-3, 1e-3, 20000, measure="scaling")
    s = c.scaling
    assert s.max_batch_memory == 9000
    assert set(s.seconds) <= {4096, 8192, 9000, 18000}          # the sweep stays within the cap; 2B is timed only if it fits
    assert max(b for b in s.seconds if b <= 9000) <= 9000 and 18000 not in s.seconds
    assert c.walker_batch_size == s.recommended_batch and c.walker_batch_size <= 9000
    assert s.recommended_batch == max(s.walker_steps_per_s, key=s.walker_steps_per_s.get)
    assert c.walker_batch_rule == "recommended by the sweep"
    assert c.measured.batch == c.walker_batch_size and c.measured.saves == 1
    assert c.walker_batches == -(-20000 // c.walker_batch_size)
