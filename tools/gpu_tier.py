"""The GPU tier (tier 2): the tests marked ``gpu``, run on one bounded Modal function, under a monthly budget.

    python tools/gpu_tier.py budget            # exit 0 if this month's recorded spend is under the limit, 3 if not
    python tools/gpu_tier.py run [--gpu T4]    # one container, a hard timeout, the card-seconds recorded

``run`` builds an image from this checkout (CUDA JAX + the package + pytest), runs ``pytest -m gpu`` inside a single
Modal function whose ``timeout`` is the cost ceiling (``TIMEOUT_S`` of card time), and writes ``gpu-ledger.txt`` with
one line ``<UTC date> <card> <seconds> <usd>`` for the workflow to upload as an artifact and print in the job summary.
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
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT_USD = 25.0
TIMEOUT_S = 15 * 60                                   # a run costs at most this much card time
PER_TEST_S = 120                                       # one tier-2 test may take at most this long
BUILD_ALLOWANCE_S = 10 * 60                            # the client waits at most this much longer, for the image build
USD_PER_S = {"T4": 0.000164, "L4": 0.000222}          # Modal's list price per second of the card
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


def cmd_budget(a):
    repo = os.environ.get("GITHUB_REPOSITORY", "dmrai-lab/dmipy-sim")
    month = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m")
    spent = month_spend(repo, month)
    print(f"gpu tier: ${spent:.2f} recorded in {month} of ${LIMIT_USD:.2f}")
    return 0 if spent < LIMIT_USD else 3


try:
    import modal
except ImportError:                                   # `budget` needs no Modal
    modal = None

def replay_ref():
    """The dmipy-replay commit the CPU suite runs against: ``DMIPY_REPLAY_REF`` in ``.github/workflows/tests.yml``, the one
    place it is stated, so both tiers test the same reader. Refused by name when the workflow states none."""
    import re
    m = re.search(r"^\s*DMIPY_REPLAY_REF:\s*([0-9a-f]{7,40})\s*$", (ROOT / ".github/workflows/tests.yml").read_text(), re.M)
    if not m:
        raise SystemExit("tests.yml states no DMIPY_REPLAY_REF: the GPU tier installs the reader the CPU suite tests")
    return m.group(1)


if modal is not None:
    # the checkout exists on the client only; inside the container this module is imported again from /root, where the
    # image is already built, so the pin is read where the image is defined and nowhere else
    REPLAY_REF = replay_ref() if modal.is_local() else None
    app = modal.App("dmipy-sim-gpu-tier")
    image = (modal.Image.debian_slim(python_version="3.12")
             .apt_install("git")
             .pip_install("jax[cuda12]", "torch", "pytest", "pytest-timeout")      # torch: the pose-series tests on a CUDA device
             .run_commands(f'pip install "dmipy-replay @ git+https://github.com/dmrai-lab/dmipy-replay@{REPLAY_REF}"')
             .add_local_dir(str(ROOT), "/repo", copy=True, ignore=[".git", "**/__pycache__", ".jax_cache", "docs"])
             .run_commands('pip install -e "/repo[dev,interop]"'))

    @app.function(image=image, gpu="T4", timeout=TIMEOUT_S, retries=0)
    def gpu_tests():
        t0 = time.time()
        # streamed (not captured), each test on its own line with its outcome: a run cut by the timeout still shows which
        # tests finished and which one was running; one test is held to PER_TEST_S so it cannot spend the whole budget
        p = subprocess.run([sys.executable, "-m", "pytest", "-m", "gpu", "-v", "--tb=short", "-p", "no:cacheprovider",
                            f"--timeout={PER_TEST_S}", "--durations=0", "tests"], cwd="/repo")
        return dict(rc=p.returncode, tail="(pytest output streamed above)", seconds=time.time() - t0)


def cmd_run(a):
    if not have_credentials():
        print("no Modal credentials: GPU tier skipped")
        return 0
    t0 = time.time()
    res = dict(rc=1, tail=f"no result within {TIMEOUT_S + BUILD_ALLOWANCE_S} s: the call was cancelled and the app stopped")
    try:
        with modal.enable_output(), app.run():                 # leaving this block stops the app, on every path
            call = gpu_tests.with_options(gpu=a.gpu).spawn()
            try:
                res = call.get(timeout=TIMEOUT_S + BUILD_ALLOWANCE_S)
            except TimeoutError:                       # e.g. a container that fails at startup and is restarted
                call.cancel()
            except modal.exception.FunctionTimeoutError:  # the function's own timeout: the run used its whole budget
                res = dict(rc=1, tail=f"the run hit its {TIMEOUT_S} s timeout; the tests that finished are listed above")
    finally:
        seconds = time.time() - t0                    # wall time on the client: an upper bound of the card time
        usd = min(seconds, TIMEOUT_S) * USD_PER_S[a.gpu]
        line = f"{_dt.datetime.now(_dt.timezone.utc):%Y-%m-%d} {a.gpu} {seconds:.0f} {usd:.4f}"
        Path(LEDGER).write_text(line + "\n")           # written on every path: the monthly budget sums these lines
        print("ledger:", line)
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
