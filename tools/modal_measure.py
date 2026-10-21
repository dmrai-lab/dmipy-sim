"""The measurements of ``tools/coefficient_walk_measure.py`` on a cloud T4 (Modal; one container per run, 32 GB of host
memory), for the admissions and the timing tables (dmrai-lab/dmipy-sim#742, #760) and the C0 throughput against
another checkout of the package.

    modal run tools/modal_measure.py --what "timing multiphysics" --n 20000 --t-max 0.1 --k-bands 256 --out t4.json
    MEASURE_COMPARE_DIR=/path/to/other/checkout modal run tools/modal_measure.py --what timing --n 20000 --out t4.json

The container holds this checkout at ``/root/sim`` (and ``MEASURE_COMPARE_DIR``'s at ``/root/other``), a CUDA JAX
from PyPI, and runs the tool with ``JAX_PLATFORMS=cuda``; with a comparison checkout it also times the same
positions-only walk on both (the C0 throughput, which must not regress). Nothing is uploaded anywhere but Modal's own container."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("dmipy-sim-measure")
IGNORE = ["**/.git", "**/__pycache__", "**/.jax_cache", "**/.pytest_cache", "**/*.rpk", "**/*.safetensors", "**/.venv", "**/docs", "**/examples", "**/*.npy"]
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("jax[cuda12]>=0.6.2", "numpy>=2.0", "scipy>=1.10", "safetensors>=0.4", "threadpoolctl", "trimesh")
         .add_local_dir(str(ROOT), remote_path="/root/sim", ignore=IGNORE))
COMPARE = os.environ.get("MEASURE_COMPARE_DIR")                 # another checkout of the package, for the C0 throughput
if COMPARE:
    image = image.add_local_dir(COMPARE, remote_path="/root/other", ignore=IGNORE)

C0_SNIPPET = """
import os, sys, time, json, warnings
warnings.simplefilter("ignore")
import jax, numpy as np
import dmipy_sim as d
n, T, K = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
g = d.Cylinder(radius=2e-6, orientation=(0, 0, 1))
ts = []
for i in range(3):
    t0 = time.time(); w = d.simulate_trajectories(n, 2e-9, g, T, seed=1, K=K, tiers=(), window_T=T, require_gpu=True); ts.append(time.time() - t0)
print(json.dumps(dict(device=str(jax.devices()[0]), seconds=min(ts[1:]), compile_s=ts[0] - min(ts[1:]), n=n, T=T, K=K,
                      sub_steps=int(w.sub_steps), h=float(w.h), tree=os.environ.get("PYTHONPATH"))))
"""


def _run(what, n, T, K, compare_dir, n_raw=0, bits=0, field=""):
    env = dict(os.environ, JAX_PLATFORMS="cuda", PYTHONPATH="/root/sim")
    out = {}
    if what:
        args = ["python", "/root/sim/tools/coefficient_walk_measure.py", "--what", *what.split(), "--gpu", "--json", "/root/out.json"]
        if n:
            args += ["--n", str(n)]
        if T:
            args += ["--T", str(T)]
        if K:
            args += ["--K", str(K)]
        if n_raw:
            args += ["--n-raw", str(n_raw)]
        if bits:
            args += ["--bits", str(bits)]
        if field:
            args += ["--field", field]
        t0 = time.time()
        p = subprocess.run(args, capture_output=True, text=True, env=env)
        out["tool"] = dict(returncode=p.returncode, seconds=time.time() - t0, stdout_tail=p.stdout[-4000:], stderr_tail=p.stderr[-4000:])
        if p.returncode == 0:
            out["results"] = json.loads(Path("/root/out.json").read_text())
    if compare_dir:
        for name, tree in (("this", "/root/sim"), ("other", "/root/other")):
            p = subprocess.run(["python", "-c", C0_SNIPPET, str(n or 20000), str(T or 0.1), str(K or 256)], capture_output=True, text=True,
                               env=dict(env, PYTHONPATH=tree))
            out[f"C0_{name}"] = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else dict(error=p.stderr[-3000:])
    return out


@app.function(image=image, gpu="T4", timeout=3 * 3600, memory=32768)
def measure(what: str, n: int, t_max: float, k_bands: int, compare: bool, n_raw: int = 0, bits: int = 0, field: str = "") -> dict:
    return _run(what, n, t_max, k_bands, compare, n_raw, bits, field)


@app.local_entrypoint()
def main(what: str = "timing", n: int = 0, t_max: float = 0.0, k_bands: int = 0, out: str = "t4.json", n_raw: int = 0, bits: int = 0,
         field: str = ""):
    t0 = time.time()
    res = measure.remote(what, n, t_max, k_bands, bool(COMPARE), n_raw, bits, field)
    res["launch_to_result_s"] = round(time.time() - t0, 1)
    Path(out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "results"}, indent=1, default=str)[:6000])
