"""Engine modules report through `logging`, never `print`, and warning categories say what they are.

Progress, sub-step and codec diagnostics go to the ``dmipy_sim.*`` loggers (silent by default,
``logging.basicConfig(level=logging.INFO)`` shows them); a physics-regime warning is a
``UserWarning``, an environment / GPU / OOM warning a ``RuntimeWarning``.
"""
import logging
import pathlib
import re

import numpy as np
import pytest

import dmipy_sim as d

PKG = pathlib.Path(d.__file__).parent
ENGINE_MODULES = ["core.py", "mt_walk.py", "bank.py", "mesh_bundle.py", "mesh_axon.py", "compression.py",
                  "physics.py", "trajectories.py", "bloch.py", "replay.py", "mt.py", "gpu.py", "_gpu_config.py"]
GPU_MODULES = ["gpu.py", "_gpu_config.py"]


def _statements(src, token):
    """Source of every statement beginning with ``token`` (parentheses balanced)."""
    out = []
    for m in re.finditer(rf"^\s*{re.escape(token)}", src, re.M):
        depth, i = 0, m.start()
        while i < len(src):
            depth += (src[i] == "(") - (src[i] == ")")
            if depth == 0 and src[i] == ")":
                break
            i += 1
        out.append(src[m.start():i + 1])
    return out


def test_engine_modules_do_not_print():
    offenders = {m: len(_statements((PKG / m).read_text(), "print(")) for m in ENGINE_MODULES}
    assert not any(offenders.values()), {k: v for k, v in offenders.items() if v}


def test_gpu_and_environment_warnings_are_runtime_warnings():
    for m in GPU_MODULES:
        for stmt in _statements((PKG / m).read_text(), "warnings.warn("):
            assert "RuntimeWarning" in stmt, (m, stmt[:120])


def test_progress_goes_to_the_logger(caplog):
    wf = d.set_b(d.pgse(delta=3e-3, DELTA=8e-3, G_magnitude=0.1, bvecs=[[1, 0, 0]], n_t=30, slew_rate=np.inf), 5e8)
    with caplog.at_level(logging.INFO, logger="dmipy_sim"):
        d.simulate_trajectories(64, 2e-9, d.Sphere(3e-6), 1e-3, 5e-4, seed=0, require_gpu=False)
    assert any("sub_steps=" in r.getMessage() for r in caplog.records)
    assert all(r.name.startswith("dmipy_sim") for r in caplog.records)
