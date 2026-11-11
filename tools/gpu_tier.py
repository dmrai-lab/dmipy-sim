"""The GPU tier (tier 2): the tests marked ``gpu``, run on one bounded Modal function, under a monthly budget.

    python tools/gpu_tier.py budget            # exit 0 if this month's recorded spend is under the limit, 3 if not
    python tools/gpu_tier.py run [--gpu T4]    # one container, a hard timeout, the card-seconds recorded

``run`` builds an image from this checkout (CUDA JAX + the package + pytest), runs ``pytest -m gpu`` inside a single
Modal function whose ``timeout`` is the cost ceiling (``TIMEOUT_S`` of card time), and writes ``gpu-ledger.txt`` with
one line ``<UTC date> <card> <seconds> <usd> <basis>`` (``seconds`` are the card seconds the function measured; ``basis``
says what they are: ``card``, ``upper-bound`` when the function never returned, ``none`` when no container started)
for the workflow to upload as an artifact and print in the job summary.
``budget`` sums the ledger artifacts of the workflow's runs in the current month through ``gh api`` and refuses past
``LIMIT_USD``. With no Modal credentials the tier states so and exits 0 (``skip``); the workflow reports it as a notice.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
import subprocess
import sys
import re
import threading
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT_USD = 25.0
TIMEOUT_S = 15 * 60                                   # a run costs at most this much card time
PER_TEST_S = 120                                       # a tier-2 test is failed once past this, at its next return to Python
BUILD_ALLOWANCE_S = 10 * 60                            # the client waits at most this much longer, for the image build
USD_PER_S = {"T4": 0.000164, "L4": 0.000222}          # Modal's list price per second of the card
STALL_S = 300                                          # a run whose pytest prints nothing for this long is killed and reported
START_S = 300                                          # no container this long after the call is spawned: cancelled and reported
POLL_S = 15
LEDGER = "gpu-ledger.txt"
WORKFLOW = "gpu-tier.yml"


def have_credentials():
    return bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")) or (Path.home() / ".modal.toml").exists()


def month_spend(repo, month):
    """The recorded spend of ``month`` (``YYYY-MM``): the sum of the ledger lines of this workflow's artifacts."""
    out = subprocess.run(["gh", "api", "--paginate", f"repos/{repo}/actions/artifacts?per_page=100", "-q",
                          ".artifacts[] | select(.name|startswith(\"gpu-ledger\")) | select(.expired|not) | [.id,.created_at]|@tsv"],
                         capture_output=True, text=True, check=True).stdout
    total = 0.0
    for line in out.splitlines():
        aid, created = line.split("\t")
        if not created.startswith(month):
            continue
        z = subprocess.run(["gh", "api", f"repos/{repo}/actions/artifacts/{aid}/zip"], capture_output=True, check=True).stdout
        for name in zipfile.ZipFile(io.BytesIO(z)).namelist():
            for row in zipfile.ZipFile(io.BytesIO(z)).read(name).decode().splitlines():
                total += float(row.split()[3])
    return total


def modal_month_to_date(month):
    """Modal's own bill for ``month`` so far (every app of the workspace), from its billing report; information only:
    returns the sentence to print, and says so when the report cannot be read."""
    first = _dt.date.fromisoformat(month + "-01")
    end = _dt.datetime.now(_dt.timezone.utc).date() + _dt.timedelta(days=1)
    try:
        out = subprocess.run(["modal", "billing", "report", "--start", str(first), "--end", str(end), "--json"],
                             capture_output=True, text=True, timeout=120)
        if out.returncode:
            return f"modal billing report unavailable: {out.stderr.strip()[:200]}"
        return f"modal workspace: ${sum(float(r['cost']) for r in json.loads(out.stdout)):.2f} billed in {month} to date"
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        return f"modal billing report unavailable: {type(exc).__name__}"


def cmd_budget(a):
    repo = os.environ.get("GITHUB_REPOSITORY", "dmrai-lab/dmipy-sim")
    month = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m")
    spent = month_spend(repo, month)
    print(f"gpu tier: ${spent:.2f} recorded in {month} of ${LIMIT_USD:.2f}")
    print(modal_month_to_date(month))
    return 0 if spent < LIMIT_USD else 3


def run_streamed(command, cwd, stall_s):
    """Run ``command`` with its output streamed to this process's stdout; kill it when it prints nothing for
    ``stall_s`` seconds. Returns ``(returncode, stalled)``; a stalled run has returncode 1."""
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                               start_new_session=True)
    last = [time.monotonic()]

    def pump():
        for line in process.stdout:
            last[0] = time.monotonic()
            print(line, end="", flush=True)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    stalled = False
    while process.poll() is None:
        time.sleep(min(1.0, stall_s / 5))
        if time.monotonic() - last[0] >= stall_s:
            stalled = True
            os.killpg(process.pid, 9)
            process.wait()
    reader.join(timeout=10)
    return (1 if stalled else process.returncode), stalled


def wait_for_result(call, beacon, *, start_s, total_s, poll_s, timeout_error, clock=time.monotonic):
    """Wait for ``call``'s result. ``beacon`` (a mapping the function writes ``started`` into when its container is up)
    ends the wait early: no ``started`` within ``start_s`` of the spawn means the card was never allocated, so the
    call is cancelled at once instead of at ``total_s``. Returns ``(result, card_seconds, basis)``; result is a dict
    with ``rc`` and ``tail``."""
    spawned = clock()
    while True:
        try:
            res = call.get(timeout=poll_s)
            return res, res["seconds"], "card"
        except timeout_error:
            pass
        waited = clock() - spawned
        started = beacon.get("started")
        if started is None and waited >= start_s:
            call.cancel()
            return dict(rc=1, tail=f"no container started within {start_s} s of the call: cancelled, no card time used"), 0.0, "none"
        if waited >= total_s:
            call.cancel()
            held = 0.0 if started is None else max(0.0, time.time() - started)
            return (dict(rc=1, tail=f"no result within {total_s} s: the call was cancelled"), held,
                    "none" if started is None else "upper-bound")


def pinned_stack():
    """The dmipy-analytic and dmipy-replay requirements at the commits ``.github/workflows/tests.yml`` pins
    (``DMIPY_ANALYTIC_REF``, ``DMIPY_REPLAY_REF``): the one place the stack's commits are stated, so the GPU tier tests
    the same stack as the CPU tier. Refused when tests.yml states either ref no more."""
    wf = (ROOT / ".github" / "workflows" / "tests.yml").read_text()
    refs = {k: re.search(rf"^\s*{k}:\s*([0-9a-f]{{40}})\s*$", wf, re.M) for k in ("DMIPY_ANALYTIC_REF", "DMIPY_REPLAY_REF")}
    missing = [k for k, m in refs.items() if m is None]
    if missing:
        raise SystemExit(f"gpu tier: tests.yml states no {', '.join(missing)}")
    return [f"dmipy-analytic[special,jax] @ git+https://github.com/dmrai-lab/dmipy-analytic@{refs['DMIPY_ANALYTIC_REF'].group(1)}",
            f"dmipy-replay[torch] @ git+https://github.com/dmrai-lab/dmipy-replay@{refs['DMIPY_REPLAY_REF'].group(1)}"]


try:
    import modal
except ImportError:                                   # `budget` needs no Modal
    modal = None

if modal is not None:
    app = modal.App("dmipy-sim-gpu-tier")
    image = (modal.Image.debian_slim(python_version="3.12")
             .apt_install("git")
             .pip_install("jax[cuda12]", "torch", "pytest", "pytest-timeout")      # torch: the pose-series tests on a CUDA device
             # the commits tests.yml pins, never the released packages; read where the app is launched, since the
             # container imports its own copy of this file, beside no workflows
             .pip_install(*(pinned_stack() if modal.is_local() else ()))
             .add_local_dir(str(ROOT), "/repo", copy=True, ignore=[".git", "**/__pycache__", ".jax_cache", "docs"])
             .run_commands('pip install -e "/repo[dev,interop]"'))

    @app.function(image=image, gpu="T4", timeout=TIMEOUT_S, retries=0)
    def gpu_tests(beacon):
        t0 = time.time()
        beacon["started"] = t0
        # streamed (not captured), each test on its own line with its outcome: a run cut by the timeout still shows which
        # tests finished and which one was running. PER_TEST_S fires when a test returns to Python; a single device call
        # can outlast it, so the function timeout is the hard bound of the run
        rc, stalled = run_streamed([sys.executable, "-u", "-m", "pytest", "-m", "gpu", "-v", "--tb=short", "-p", "no:cacheprovider",
                                    f"--timeout={PER_TEST_S}", "--durations=0", "tests"], "/repo", STALL_S)
        tail = f"pytest printed nothing for {STALL_S} s: killed" if stalled else "(pytest output streamed above)"
        return dict(rc=rc, tail=tail, seconds=time.time() - t0)


def cmd_run(a):
    if not have_credentials():
        print("no Modal credentials: GPU tier skipped")
        return 0
    t0 = time.time()
    res, card_s, basis = dict(rc=1, tail="the run did not start"), 0.0, "none"
    try:
        with modal.enable_output(), modal.Dict.ephemeral() as beacon, app.run():      # leaving the blocks stops the app, on every path
            call = gpu_tests.with_options(gpu=a.gpu).spawn(beacon)
            try:
                res, card_s, basis = wait_for_result(call, beacon, start_s=START_S, total_s=TIMEOUT_S + BUILD_ALLOWANCE_S,
                                                     poll_s=POLL_S, timeout_error=TimeoutError)
            except modal.exception.FunctionTimeoutError:  # the function's own timeout: the run used its whole budget
                res, card_s, basis = dict(rc=1, tail=f"the run hit its {TIMEOUT_S} s timeout; the tests that finished are listed above"), TIMEOUT_S, "card"
    finally:
        usd = min(card_s, TIMEOUT_S) * USD_PER_S[a.gpu]
        line = f"{_dt.datetime.now(_dt.timezone.utc):%Y-%m-%d} {a.gpu} {card_s:.0f} {usd:.4f} {basis}"
        Path(LEDGER).write_text(line + "\n")           # written on every path: the monthly budget sums these lines
        print(f"ledger: {line} (client wall time {time.time() - t0:.0f} s, including the image build and the wait for a card)")
    print(res["tail"])
    return res["rc"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("budget").set_defaults(f=cmd_budget)
    r = sub.add_parser("run"); r.add_argument("--gpu", default="T4", choices=sorted(USD_PER_S)); r.set_defaults(f=cmd_run)
    a = ap.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
