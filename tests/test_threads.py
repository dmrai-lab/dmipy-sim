"""dmipy-sim owns its CPU thread count (dmrai-lab/dmipy-sim#700): BLAS is capped at
:func:`dmipy_sim._threads.cpu_threads` as soon as the package is imported, ``DMIPY_SIM_CPU_THREADS``
is honoured (read fresh at import time, so this needs a subprocess), a bad value is refused by
name, and XLA's CPU intra-op pool is set through ``XLA_FLAGS`` in the one window before jax's
backend initialises."""
import os
import subprocess
import sys

import pytest
import threadpoolctl

import dmipy_sim as d  # noqa: F401  -- triggers the package-import-time cap
from dmipy_sim import _threads


def test_blas_pools_are_capped_after_import():
    n = _threads.cpu_threads()
    pools = [p for p in threadpoolctl.threadpool_info() if p.get("user_api") == "blas"]
    assert pools, "no BLAS pool loaded -- nothing to assert the cap against"
    assert all(int(p["num_threads"]) <= n for p in pools)


@pytest.mark.parametrize("bad", ["0", "-1", "3.5", "abc", " ", "1e3"])
def test_a_bad_value_is_refused_by_name(monkeypatch, bad):
    monkeypatch.setenv("DMIPY_SIM_CPU_THREADS", bad)
    with pytest.raises(ValueError, match="DMIPY_SIM_CPU_THREADS"):
        _threads.cpu_threads()


def test_default_is_min_8_cpu_count(monkeypatch):
    monkeypatch.delenv("DMIPY_SIM_CPU_THREADS", raising=False)
    assert _threads.cpu_threads() == min(8, os.cpu_count() or 8)


def _run(env_extra, code):
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.update(env_extra)
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_env_var_is_honoured_fresh_in_a_subprocess():
    code = (
        "import dmipy_sim, threadpoolctl\n"
        "pools = [p for p in threadpoolctl.threadpool_info() if p.get('user_api') == 'blas']\n"
        "assert pools and all(int(p['num_threads']) == 3 for p in pools), pools\n"
        "print('OK')\n"
    )
    r = _run({"DMIPY_SIM_CPU_THREADS": "3"}, code)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_xla_flags_carries_the_intra_op_setting_when_imported_before_jax():
    code = (
        "import os\n"
        "import dmipy_sim\n"  # imports jax itself, AFTER _cap_cpu_threads() already set XLA_FLAGS
        "flags = os.environ.get('XLA_FLAGS', '')\n"
        "assert 'intra_op_parallelism_threads=4' in flags, flags\n"
        "print('OK')\n"
    )
    r = _run({"DMIPY_SIM_CPU_THREADS": "4"}, code)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_xla_flags_is_not_touched_when_jax_already_imported():
    code = (
        "import jax\n"
        "import warnings\n"
        "import os\n"
        "before = os.environ.get('XLA_FLAGS', '')\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    warnings.simplefilter('always')\n"
        "    import dmipy_sim\n"
        "    assert any('XLA' in str(x.message) for x in w), [str(x.message) for x in w]\n"
        "assert os.environ.get('XLA_FLAGS', '') == before\n"
        "print('OK')\n"
    )
    r = _run({"DMIPY_SIM_CPU_THREADS": "4"}, code)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_cap_cpu_threads_is_idempotent():
    _threads.cap_cpu_threads()
    _threads.cap_cpu_threads()  # must not raise, must not rebuild from scratch each time
