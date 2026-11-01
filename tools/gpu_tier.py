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


def cmd_run(a):
    if not have_credentials():
        print("no Modal credentials: GPU tier skipped")
        return 0
    import modal
    app = modal.App("dmipy-sim-gpu-tier")
    image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("jax[cuda12]", "pytest", "pytest-timeout")
             .add_local_dir(str(ROOT), "/repo", copy=True, ignore=[".git", "**/__pycache__", ".jax_cache", "docs"])
             .run_commands('pip install -e "/repo[dev,interop]"'))

    @app.function(image=image, gpu=a.gpu, timeout=TIMEOUT_S, retries=0)
    def gpu_tests():
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", "pytest", "-m", "gpu", "-q", "--tb=short", "-p", "no:cacheprovider",
                            "--timeout=600", "--durations=20", "tests"], cwd="/repo", capture_output=True, text=True)
        return dict(rc=p.returncode, tail=p.stdout[-6000:], seconds=time.time() - t0)

    t0 = time.time()
    try:
        with modal.enable_output(), app.run():
            res = gpu_tests.remote()
    finally:
        seconds = time.time() - t0                    # card time is billed to the container's end, the timeout bounds it
    usd = min(seconds, TIMEOUT_S) * USD_PER_S[a.gpu]
    line = f"{_dt.datetime.now(_dt.timezone.utc):%Y-%m-%d} {a.gpu} {seconds:.0f} {usd:.4f}"
    Path(LEDGER).write_text(line + "\n")
    print(res["tail"])
    print("ledger:", line)
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
