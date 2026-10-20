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
    """``walk_cost`` of the walk ``simulate_trajectories(n, D, g, T_max, dt, **kw)``."""
    return d.walk_cost(n, D, g, T_max, dt, **kw)


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
    assert p.walker_steps == w.work["walker_steps"] == N * p.n_blocks * w.sub_steps
    assert p.n_blocks == w.n_blocks and p.K == w.K and p.block_s == pytest.approx(w.h)
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
    assert cost(g, f_band=100.0).sub_steps <= cost(g).sub_steps
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


def test_an_installed_backend_that_accepts_the_walk_is_named_and_the_walk_warns(monkeypatch, caplog):
    """An installed backend that does not refuse the request is named among the accepting (the coefficient walk is a
    backend's to walk, dmipy-sim#747); one that refuses is named with its reason; and a walk on the jax kernels that
    an installed backend accepts warns, naming it."""
    _install(monkeypatch, fast=_Fake("fast"), picky=_Fake("picky", "no spheres"))
    p = cost(d.Sphere(radius=2e-6))
    assert p.backends_accepting == ["fast"]
    assert set(p.backends_refusing) == {"picky"} and "no spheres" in p.backends_refusing["picky"]
    with caplog.at_level(logging.WARNING, logger="dmipy_sim"):
        w = d.simulate_trajectories(N, D, d.Sphere(radius=2e-6), T, DT)
    warned = [r for r in w.run._events if r["kind"] == "warning"]
    assert len(warned) == 1 and warned[0]["backends"] == ["fast"]
    assert any("backend='fast'" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


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
    for k in ("sub_steps", "walker_steps", "binding_rule", "n_blocks", "K", "backend", "sub_step_rules"):
        assert c[0][k] == getattr(p, k)


def test_measure_projects_the_walk():
    g = d.Sphere(radius=2e-6)
    n, t = 2000, 8e-3
    p = cost(g, t, 1e-3, n, walker_batch_size=1000, measure=True)
    m = p.measured
    assert m.seconds_per_walker_step > 0 and m.projected_s > 0
    assert m.batch == 1000 and m.pilot_blocks[0] == 1 and 1 < m.pilot_blocks[1] <= p.n_blocks
    assert m.projected_s == pytest.approx(p.walker_batches * (m.fixed_seconds_per_call + m.seconds_per_block * p.n_blocks))
    assert cost(g, t, 1e-3, n).measured is None and cost(g, t, 1e-3, n).profile is None
    # within 2x of a measured walk of the same shape (compile excluded)
    d.simulate_trajectories(n, D, g, t, 1e-3, walker_batch_size=1000)
    t0 = time.perf_counter()
    d.simulate_trajectories(n, D, g, t, 1e-3, walker_batch_size=1000)
    real = time.perf_counter() - t0
    assert real / 2 < m.projected_s < real * 2


def test_profile_times_the_pilot_at_several_batches(monkeypatch):
    from dmipy_sim.engine import core
    g = d.Sphere(radius=2e-6)
    monkeypatch.setattr(core, "_free_device_bytes", lambda: None)
    p = cost(g, 8e-3, 1e-3, 400, walker_batch_size=200, measure="profile")
    s = p.profile
    assert sorted(s.walker_steps_per_s) == [100, 200, 400, 800]
    assert s.plateau_walker_steps_per_s == max(s.walker_steps_per_s.values())
    assert s.batch_fraction_of_plateau == pytest.approx(s.walker_steps_per_s[200] / s.plateau_walker_steps_per_s)
    assert s.max_batch_memory is None and "unavailable" in s.memory_cap
    want = "under-filled" if s.seconds_per_block[400] < core.UNDERFILLED_DOUBLING * s.seconds_per_block[200] else "saturated"
    assert s.verdict == want
    # a device that holds 1.5 batches: 2B does not fit
    per = core._device_bytes_per_walker(core._resolve_walk(g, D, 8e-3, 1e-3))
    monkeypatch.setattr(core, "_free_device_bytes", lambda: int(1.5 * 200 * per))
    q = cost(g, 8e-3, 1e-3, 400, walker_batch_size=200, measure="profile").profile
    assert sorted(q.walker_steps_per_s) == [100, 200] and q.verdict == "memory-bound"
    assert q.max_batch_memory == 300


def test_a_cost_is_frozen_and_its_row_is_its_fields():
    import dataclasses
    p = cost(d.Sphere(radius=2e-6))
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.sub_steps = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.profile = None


def test_the_cost_takes_the_walks_arguments():
    import inspect
    sig, walk = inspect.signature(d.walk_cost).parameters, inspect.signature(d.simulate_trajectories).parameters
    assert list(sig)[:5] == list(walk)[:5] == ["n_walkers", "diffusivity", "geometry", "T_max", "dt_save"]
    assert sig["walker_batch_size"].default == walk["walker_batch_size"].default
    kw = dict(n_walkers=N, diffusivity=D, geometry=d.Sphere(radius=2e-6), T_max=T, dt_save=DT, tiers=())
    assert d.walk_cost(**kw).sub_steps == d.simulate_trajectories(**kw).sub_steps


def test_the_band_is_the_walks():
    from dmipy_sim.engine.physics import sub_step_rules
    g = d.Sphere(radius=20e-6)                  # a pore so wide that the band, not the pore, sizes the step
    c0 = cost(g)
    assert c0.f_band_hz == pytest.approx(c0.K / (2 * T))   # the walk's own band, K / (2 T)
    c = cost(g, f_band=1280.0)
    assert c.f_band_hz == 1280.0 and c.sub_step_rules == sub_step_rules(g, D, DT, surface=True, f_band=1280.0)
    assert c.sub_steps == d.simulate_trajectories(N, D, g, T, DT, f_band=1280.0).sub_steps


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
    assert core._profile({100: 1.0, 200: 1.1}, 100, res, None, "given").verdict == "under-filled"
    assert core._profile({100: 1.0, 200: 1.2}, 100, res, None, "given").verdict == "saturated"
    assert core._profile({100: 1.0}, 100, res, 150, "given").verdict == "memory-bound"
    assert core._profile({100: 1.0}, 100, res, None, "given").verdict == "unresolved"


def test_no_batch_given_is_n_walkers_capped_by_memory(monkeypatch):
    from dmipy_sim.engine import core
    g = d.Sphere(radius=2e-6)
    monkeypatch.setattr(core, "_free_device_bytes", lambda: None)
    assert cost(g).walker_batch_rule == "given"                      # the walk's own default batch
    c = cost(g, walker_batch_size=None)
    assert c.walker_batch_size == N and c.walker_batches == 1 and c.walker_batch_rule == "n_walkers capped by memory"
    per = c.bytes_per_walker
    monkeypatch.setattr(core, "_free_device_bytes", lambda: 20 * per)
    c = cost(g, walker_batch_size=None)
    assert c.walker_batch_size == 20 and c.walker_batches == 4 and c.walker_batch_rule == "n_walkers capped by memory"
    assert cost(g, walker_batch_size=30).walker_batch_rule == "given"


def test_profile_finds_the_batch_within_the_memory_cap(monkeypatch):
    from dmipy_sim.engine import core
    g = d.Sphere(radius=2e-6)
    per = core._device_bytes_per_walker(core._resolve_walk(g, D, 2e-3, 1e-3))
    monkeypatch.setattr(core, "_free_device_bytes", lambda: int(9000.5 * per))
    c = cost(g, 2e-3, 1e-3, 20000, walker_batch_size=None, measure="profile")
    s = c.profile
    assert s.max_batch_memory == 9000
    assert set(s.seconds_per_block) <= {4096, 8192, 9000, 18000}          # the sweep stays within the cap; 2B is timed only if it fits
    assert 18000 not in s.seconds_per_block
    assert c.walker_batch_size == s.recommended_batch and c.walker_batch_size <= 9000
    assert s.recommended_batch == max(s.walker_steps_per_s, key=s.walker_steps_per_s.get)
    assert c.walker_batch_rule == "recommended by the sweep"
    assert c.measured.batch == c.walker_batch_size
    assert c.walker_batches == -(-20000 // c.walker_batch_size)


def test_the_fixed_part_of_a_call_is_not_multiplied_by_the_blocks():
    from dmipy_sim.engine import core
    # a call costs 5 s whatever its blocks, plus b * blocks / 1e4 s, but b = 32768 is slower per walker (the true optimum is 16384)
    per_walker = {4096: 1.0, 8192: 0.6, 16384: 0.4, 32768: 0.5}
    calls = []

    def seconds(b, saves):
        calls.append((b, saves))
        return 5.0 + b * per_walker[b] * saves / 1e4

    ps, fixed, (n1, n2) = core._fit_blocks(seconds, 16384, 1111)
    assert ps == pytest.approx(16384 * 0.4 / 1e4) and fixed == pytest.approx(5.0) and n1 == 1
    assert n2 > 2 and n2 < 1111                       # doubled until the blocks outweighed the fixed part
    t_n2, t_1 = 5.0 + ps * n2, 5.0 + ps
    assert t_n2 >= 2 * t_1 and (n2 == 2 or 5.0 + ps * (n2 // 2) < 2 * t_1)
    # the projection is fixed + blocks * per_block, not 1111 * t(1)
    assert fixed + ps * 1111 < 0.5 * 1111 * (5.0 + ps)
    # the sweep rates by seconds per block: the fastest batch is 16384 and the sweep stops after the slower 32768
    per, why = core._sweep(lambda b: core._fit_blocks(seconds, b, 1111)[0], 32768, "n_walkers")
    assert sorted(per) == [4096, 8192, 16384, 32768] and core._recommended(per) == 16384 and why == "rate fell"
    # a walk of one block has no fixed part to separate
    assert core._fit_blocks(seconds, 4096, 1) == (seconds(4096, 1), 0.0, (1, 1))
