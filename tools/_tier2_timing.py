"""TEMPORARY (removed before the pull request is ready): time tests on a hosted CPU runner, one file per job.

    python tools/_tier2_timing.py tests/geometry/test_mesh_mc.py

Environment: ``TIMING_MARK`` (``-m`` expression, default ``gpu``), ``TIMING_K`` (``-k`` expression),
``TIMING_BUDGET`` (seconds per test, default 900: a watchdog records the test as past the budget and the file's remaining
tests run in a fresh process). Each test prints one ``TIMING_ROW {json}`` line: its setup, call and teardown seconds."""
import json
import os
import platform
import subprocess
import sys
import tempfile
import textwrap

PLUGIN = textwrap.dedent('''
    import os, json, threading, time, pytest
    RES = os.environ["TIMING_RES"]
    BUDGET = float(os.environ.get("TIMING_BUDGET", "900"))
    _t = {}
    def _log(rec):
        with open(RES, "a") as f: f.write(json.dumps(rec) + "\\n")
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(item, nextitem):
        def boom():
            _log({"id": item.nodeid, "st": "OVER_BUDGET", "total": BUDGET})
            os._exit(99)
        timer = threading.Timer(BUDGET, boom); timer.daemon = True; timer.start()
        _t[item.nodeid] = {"st": "passed", "reason": "", "setup": 0.0, "call": 0.0, "teardown": 0.0}
        yield
        timer.cancel()
        r = _t.pop(item.nodeid)
        _log({"id": item.nodeid, "st": r["st"].upper(), "setup": round(r["setup"], 2), "call": round(r["call"], 2),
              "teardown": round(r["teardown"], 2), "total": round(r["setup"] + r["call"] + r["teardown"], 2), "reason": r["reason"]})
    def pytest_runtest_logreport(report):
        r = _t.get(report.nodeid)
        if r is None: return
        r[report.when] = report.duration
        if report.failed: r["st"] = "failed"; r["reason"] = str(report.longrepr)[-400:]
        elif report.skipped and r["st"] == "passed": r["st"] = "skipped"; r["reason"] = str(report.longrepr[-1])[:200]
''')

def time_file(f, mark, kexpr, budget):
    res = os.path.abspath("timing-" + f.replace("/", "_") + ".jsonl")
    plug = tempfile.mkdtemp()
    open(os.path.join(plug, "timing_plugin.py"), "w").write(PLUGIN)
    sel = ["-m", mark] + (["-k", kexpr] if kexpr else [])
    collected = subprocess.run([sys.executable, "-m", "pytest", f, *sel, "--collect-only", "-q", "-o", "addopts=", "-p", "no:cacheprovider"],
                               capture_output=True, text=True, env={**os.environ, "JAX_PLATFORMS": "cpu"}).stdout.splitlines()
    want = [l.strip() for l in collected if "::" in l]
    cpu = ""
    try:
        cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception:
        pass
    print(f"TIMING_ENV {json.dumps(dict(file=f, nproc=os.cpu_count(), cpu=cpu, python=platform.python_version(), tests=len(want)))}", flush=True)
    for _ in range(80):
        done = set(json.loads(l)["id"] for l in open(res)) if os.path.exists(res) else set()
        if all(w in done for w in want):
            break
        cmd = [sys.executable, "-m", "pytest", f, *sel, "-p", "timing_plugin", "-p", "no:cacheprovider", "-o", "addopts=", "-q", "--tb=line", "-rs"]
        for d in done:
            cmd += ["--deselect", d]
        env = {**os.environ, "PYTHONPATH": plug + os.pathsep + os.environ.get("PYTHONPATH", ""), "TIMING_RES": res, "TIMING_BUDGET": str(budget), "JAX_PLATFORMS": "cpu"}
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 99:
            break
    rows = [json.loads(l) for l in open(res)] if os.path.exists(res) else []
    for r in rows:
        print("TIMING_ROW " + json.dumps(r), flush=True)


arg = sys.argv[1]
if arg.startswith("["):
    for e in json.loads(arg):
        time_file(e["file"], e.get("mark", "gpu"), e.get("k") or None, e.get("budget", "900"))
else:
    time_file(arg, os.environ.get("TIMING_MARK", "gpu"), os.environ.get("TIMING_K"), os.environ.get("TIMING_BUDGET", "900"))
