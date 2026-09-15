"""The record of a run (`dmipy_sim.run`): a producer opens it, a caller never does; it lives in memory and persists
itself once it outlives the sampling interval (or at once with run_dir=); a nested producer joins the outer run;
the event log survives a kill line by line; a pack records the runs that made it."""
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

from dmipy_sim import run as R


def test_a_short_run_touches_no_disk_and_a_persisted_one_is_complete(tmp_path, monkeypatch):
    monkeypatch.setenv("DMIPY_SIM_RUN_DIR", str(tmp_path / "runs"))
    with R.Run("quick", params=dict(n=3)) as r:
        r.progress(1, 3); r.progress(3, 3)
    assert r.dir is None and not (tmp_path / "runs").exists() and r.status == "ok"       # over before the interval
    d = tmp_path / "one"
    with R.Run("walk", params=dict(n_walkers=10, geometry=object()), run_dir=str(d)) as r:
        r.phase("seeding", pool="extra"); r.phase("walk", n_walkers=10)
        for k in range(1, 11):
            r.progress(k, 10)                                     # throttled: the last is always written
        r.artifact(str(d / "x.rpk"))
    man = json.load(open(d / "manifest.json")); rows = R.read_events(str(d)); summ = json.load(open(d / "summary.json"))
    assert man["producer"] == "walk" and man["params"]["n_walkers"] == 10 and man["memory_ceiling_bytes"] > 0 and "started" in man
    kinds = [x["kind"] for x in rows]
    assert kinds[0] == "start" and kinds[-1] == "end" and rows[-1]["status"] == "ok" and "resource" in kinds
    prog = [x for x in rows if x["kind"] == "progress"]
    assert prog and prog[-1]["done"] == 10 and prog[-1]["total"] == 10 and prog[-1]["phase"] == "walk"
    assert [x["name"] for x in rows if x["kind"] == "phase"] == ["seeding", "walk"]
    assert summ["status"] == "ok" and set(summ["phases_s"]) == {"seeding", "walk"} and summ["peak_rss_bytes"] > 0
    rep = R.report(str(d))
    assert rep["status"] == "ok" and rep["last_progress"]["done"] == 10 and "walk" in rep["phases_s"]
    with pytest.raises(ValueError):
        with R.Run("broken", run_dir=str(tmp_path / "two")):
            raise ValueError("boom")
    rows = R.read_events(str(tmp_path / "two"))
    assert rows[-1]["kind"] == "end" and rows[-1]["status"] == "error" and "boom" in rows[-1]["traceback"]
    assert R.report(str(tmp_path / "two"))["status"] == "error"
    assert {x["producer"] for x in R.list_runs(str(tmp_path))} >= {"walk", "broken"}     # a root of run directories


def test_a_nested_producer_joins_the_outer_run(tmp_path):
    with R.Run("outer", run_dir=str(tmp_path / "o")) as outer:
        outer.phase("walk extra")
        with R.Run("inner", params=dict(n=2)) as inner:
            assert inner is outer and R.current() is outer          # one record
            inner.progress(2, 2)
        assert R.current() is outer
    rows = R.read_events(str(tmp_path / "o"))
    assert [x["name"] for x in rows if x["kind"] == "phase"] == ["walk extra"]         # no phase of its own
    assert [x["producer"] for x in rows if x["kind"] == "join"] == ["inner"]
    assert [x for x in rows if x["kind"] == "progress"][-1]["phase"] == "walk extra"
    assert R.current() is None


def test_a_killed_run_leaves_a_readable_record(tmp_path):
    d = tmp_path / "k"
    code = f"""
import time, os
os.environ["DMIPY_SIM_SAMPLE_S"] = "0.2"; os.environ["DMIPY_SIM_PROGRESS_S"] = "0.1"
from dmipy_sim import run as R
with R.Run("victim", run_dir={str(d)!r}) as r:
    r.phase("walk")
    for k in range(10 ** 9):
        r.progress(k, 10 ** 9); time.sleep(0.01)
"""
    p = subprocess.Popen([sys.executable, "-c", code], env=dict(os.environ, JAX_PLATFORMS="cpu"))
    deadline = time.time() + 30
    while time.time() < deadline and not (d / "events.jsonl").exists():
        time.sleep(0.1)
    time.sleep(1.5)
    os.kill(p.pid, signal.SIGKILL); p.wait(); time.sleep(1.0)                          # past two of its 0.2 s intervals
    raw = open(d / "events.jsonl").read().splitlines()
    rows = R.read_events(str(d))
    assert len(rows) >= len(raw) - 1 and all(json.loads(l) for l in raw[:-1])          # every line but maybe the last
    kinds = {x["kind"] for x in rows}
    assert "resource" in kinds and "progress" in kinds and "end" not in kinds
    rep = R.report(str(d))
    assert rep["status"] in ("stale", "killed") and rep["last_progress"]["phase"] == "walk"
    assert not (d / "summary.json").exists()


def test_a_pack_records_the_runs_that_made_it(tmp_path, monkeypatch):
    from dmipy_sim import Cylinder, simulate_trajectories
    from dmipy_sim.replay.bank import build_replay_pack
    monkeypatch.setenv("DMIPY_SIM_RUN_DIR", str(tmp_path / "runs"))
    w = simulate_trajectories(200, 2e-9, Cylinder(radius=3e-6, orientation=(0, 0, 1)), T_max=2e-3, dt_save=1e-4, seed=0, require_gpu=False)
    assert w.run is not None and w.run.status == "ok" and w.run.summary["producer"] == "simulate_trajectories"
    pk = build_replay_pack(w, id="t/run", license="x", citation="x", K=8, device="numpy", out_path=str(tmp_path / "r.rpk"))
    prov = pk.meta["provenance"]["run"]
    assert prov["walk"]["producer"] == "simulate_trajectories" and prov["walk"]["status"] == "ok" and prov["walk"]["wall_s"] > 0
    assert prov["pack"]["id"].endswith(f"-build_replay_pack-{os.getpid()}") and prov["pack"]["host"]
    json.dumps(pk.meta)                                                                # JSON-ready


def test_the_cli_lists_and_reports(tmp_path):
    with R.Run("walk", run_dir=str(tmp_path / "runs" / "a")) as r:
        r.phase("walk"); r.progress(5, 10); r.progress(10, 10)
    out = subprocess.run([sys.executable, "-m", "dmipy_sim.run", str(tmp_path / "runs" / "a")], capture_output=True, text=True,
                         env=dict(os.environ, JAX_PLATFORMS="cpu"))
    assert out.returncode == 0 and "status ok" in out.stdout and "walk" in out.stdout, out.stdout + out.stderr
    out = subprocess.run([sys.executable, "-m", "dmipy_sim.run", "list"], capture_output=True, text=True,
                         env=dict(os.environ, JAX_PLATFORMS="cpu", DMIPY_SIM_RUN_DIR=str(tmp_path / "runs")))
    assert out.returncode == 0 and "ok" in out.stdout, out.stdout + out.stderr


def test_every_long_producer_opens_a_run():
    """The rule: a producer that can run for hours is a defect without a record."""
    import inspect
    from dmipy_sim.engine import core, adaptive, bloch, mt_walk
    from dmipy_sim.spec import walk
    from dmipy_sim.replay import bank
    for f in (core.simulate, core.simulate_cpmg, core.simulate_trajectories, adaptive.simulate_trajectories_adaptive,
              bloch.simulate_bloch, mt_walk.simulate_mt_trajectories, walk.walk_spec, bank.build_replay_pack, bank.merge_packs):
        assert "with Run(" in inspect.getsource(f), f.__name__
    # and ONE batch loop: no producer counts its own batches
    for mod in (core, adaptive, bloch, mt_walk):
        assert "n_batches" not in inspect.getsource(mod), mod.__name__


def test_the_budget_guard_stops_a_run_before_the_host_does(tmp_path, monkeypatch):
    """A ceiling just above the current footprint: a walk-like loop that grows by a batch's worth per batch is
    stopped at a batch boundary with the projection in its record, and the pieces it spooled are intact."""
    import gc
    monkeypatch.setenv("DMIPY_SIM_MEMORY_CEILING_BYTES", str(R._rss() + 400 * 2 ** 20))
    monkeypatch.setattr(R, "BUDGET_FRACTION", 0.9)
    keep = []
    with pytest.raises(R.ResourceBudgetError, match="memory budget"):
        with R.Run("hog", run_dir=str(tmp_path / "h")) as r:
            for b, (s, e) in enumerate(r.batches(10, 1)):
                keep.append(np.ones(150 * 2 ** 20 // 8)); keep[-1][:] = b     # 150 MB per batch, touched
                r.spool(f"batch-{b:04d}", dict(x=np.arange(3)), dict(batch=b))
    rows = R.read_events(str(tmp_path / "h"))
    assert rows[-1]["kind"] == "end" and rows[-1]["status"] == "error" and "ResourceBudgetError" in rows[-1]["error"]
    warn = [x for x in rows if x["kind"] == "warning"]
    assert warn and warn[-1]["projected"] > warn[-1]["ceiling_bytes"] * 0.9
    spooled = [x for x in rows if x["kind"] == "spool"]
    assert 1 <= len(spooled) < 10 and all(os.path.isfile(x["path"]) for x in spooled)
    del keep; gc.collect()
