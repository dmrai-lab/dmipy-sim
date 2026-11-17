"""The host check of the pinned recorded-bits job: is this runner's compiled arithmetic the recording host's?

Recorded bits (the all-myelinated walks' hashes in ``tests/geometry/test_mixed_myelinated_plain_cell.py``, the record's
bit identity in ``tests/test_trial_record.py``) are bits of one JAX build and of the kernels the compiler emits for one
host's CPU. This script walks a small fixed all-myelinated cell (16 walkers, 0.5 ms) through the same concentric wall
kernel and compares the hash of its coefficients with the value recorded on the recording host. A mismatch fails the job
with the CPU named, before any recorded-bits test runs, so a runner whose kernels differ is reported as that and not as a
hash mismatch of the code under test (a change to the wall kernel's arithmetic moves this value too: then compare on
the recording host).

Run: ``JAX_PLATFORMS=cpu python tools/recorded_bits_canary.py``; exit 0 on the recording host's kernels.
"""
import hashlib
import platform
import sys

import numpy as np

#: The hash of the canary walk's ``position_coeffs`` on the recording host (jax / jaxlib 0.10.2, CPU).
RECORDED = "22ccc499baa1797363d54e57d2b5dce6e6ae3c09fcd5a101f88cb9d4815b9eeb"


def _cpu():
    try:
        with open("/proc/cpuinfo") as f:
            text = f.read()
        model = next((l.split(":", 1)[1].strip() for l in text.splitlines() if l.startswith("model name")), "?")
        return f"{model} (avx512f: {'avx512f' in text})"
    except OSError:
        return platform.processor() or "?"


def canary():
    from dmipy_sim.compartments import Compartments, Pool
    from dmipy_sim.engine.core import _walk_geometry
    from dmipy_sim.geometry import PackedMyelinatedCylinders
    D, R = 2e-9, 2e-6
    g = np.array([0.7, 0.8, 0.7, 0.75])
    outer = np.array([R, 0.8 * R, R, 0.9 * R])
    centres = np.array([[0.0, 0.0], [2.2, 0.0], [0.0, 2.2], [2.2, 2.2]]) * R
    cell = PackedMyelinatedCylinders(outer * g, g, centres, 4.4 * R, N_max=4, kappa_inner=4e-5, kappa_outer=4e-5,
                                     compartments=Compartments(intra=Pool(D=D), myelin=Pool(D=0.0), extra=Pool(D=D)))
    w = _walk_geometry(16, D, cell, 5e-4, 5e-5, tiers="all", seed=3, require_gpu=False, K=4)
    return hashlib.sha256(np.ascontiguousarray(w.position_coeffs).tobytes()).hexdigest()


if __name__ == "__main__":
    got = canary()
    if got != RECORDED:
        print(f"this runner's kernels are not the recording host's: the canary walk hashes to {got}, the recording host's "
              f"to {RECORDED}; CPU {_cpu()}. Re-run the job to draw another runner.")
        sys.exit(1)
    print(f"recording host's kernels: canary {got}; CPU {_cpu()}")
