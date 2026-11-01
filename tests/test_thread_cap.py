"""``import dmipy_sim`` applies the CPU thread cap (dmipy-replay's ``dmipy_replay._blas``, dmrai-lab/dmipy-sim#700) before
any submodule imports JAX: the cap of ``DMIPY_REPLAY_CPU_THREADS`` is what BLAS runs at once the package is imported.
The cap itself (its default, its refusals, the faulty build's floor) is dmipy-replay's to test."""
import os
import subprocess
import sys

import pytest

threadpoolctl = pytest.importorskip("threadpoolctl")

_PROBE = ("import dmipy_sim, threadpoolctl\n"
          "print(sorted({p['num_threads'] for p in threadpoolctl.threadpool_info() if p['user_api']=='blas'}))")


def test_importing_the_package_applies_the_environment_cap():
    env = {k: v for k, v in os.environ.items() if k not in ("DMIPY_REPLAY_CPU_THREADS", "DMIPY_SIM_CPU_THREADS")}
    env.update(JAX_PLATFORMS="cpu", DMIPY_REPLAY_CPU_THREADS="2")
    r = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[2]"
