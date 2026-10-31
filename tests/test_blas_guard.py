"""The CPU thread cap (dmipy_sim._blas, dmrai-lab/dmipy-sim#700) and its floor on the one BLAS build that hangs
(#564): the cap is applied at import and readable through threadpoolctl, an explicit setting wins over the
environment variable and the default, a bad value is refused by name, and the hanging build never runs above its
floor whatever was asked."""
import os
import subprocess
import sys

import pytest
import threadpoolctl

from dmipy_sim import _blas


def _pool(**kw):
    p = dict(user_api="blas", internal_api="openblas", version="0.3.28", architecture="neoversev1", num_threads=64)
    p.update(kw)
    return p


def _blas_threads():
    return {int(p["num_threads"]) for p in threadpoolctl.threadpool_info() if p["user_api"] == "blas"}


def _python(code, **env):
    e = {k: v for k, v in os.environ.items() if k != _blas.ENV_VAR}
    e.update(JAX_PLATFORMS="cpu", **env)
    r = subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


_PROBE = ("import dmipy_sim, threadpoolctl\n"
          "print(sorted({p['num_threads'] for p in threadpoolctl.threadpool_info() if p['user_api']=='blas'}))")


def test_import_applies_the_environment_cap():
    assert _python(_PROBE, **{_blas.ENV_VAR: "2"}) == "[2]"


def test_unset_environment_gives_the_stated_default():
    out = eval(_python(_PROBE))
    assert max(out) <= _blas.DEFAULT_CPU_THREADS


def test_an_explicit_setting_wins_over_the_environment():
    code = ("import dmipy_sim, threadpoolctl\n"
            "assert dmipy_sim.cap_cpu_threads(1) == 1\n"
            "print({p['num_threads'] for p in threadpoolctl.threadpool_info() if p['user_api']=='blas'})")
    assert _python(code, **{_blas.ENV_VAR: "3"}) == "{1}"


def test_the_cap_is_applied_and_readable_in_process():
    try:
        assert _blas.cap_cpu_threads(1) == 1
        assert _blas_threads() == {1}
    finally:
        _blas.cap_cpu_threads(_blas.cpu_threads())


@pytest.mark.parametrize("bad", [0, -1, 2.5, "3", True])
def test_a_bad_argument_is_refused_by_name(bad):
    with pytest.raises(ValueError, match="n="):
        _blas.cpu_threads(bad)


@pytest.mark.parametrize("bad", ["0", "-1", "3.5", "abc", " ", "1e3"])
def test_a_bad_environment_value_is_refused_by_name(monkeypatch, bad):
    monkeypatch.setenv(_blas.ENV_VAR, bad)
    with pytest.raises(ValueError, match=_blas.ENV_VAR):
        _blas.cpu_threads()


@pytest.mark.parametrize("pool", [_pool(version="0.3.27"), _pool(architecture="Haswell"), _pool(internal_api="mkl"),
                                  _pool(num_threads=8), _pool(user_api="openmp")])
def test_an_unaffected_build_gets_exactly_what_was_asked(monkeypatch, pool):
    calls = []
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [pool])
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", lambda **k: calls.append(k))
    _blas._cap_blas(72)
    assert calls == [dict(limits=72, user_api="blas")]


@pytest.mark.parametrize("asked,held", [(72, _blas.SAFE_THREADS), (3, 3)])
def test_the_hanging_build_never_exceeds_its_floor(monkeypatch, asked, held):
    calls = []
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [_pool()])
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", lambda **k: calls.append(k))
    _blas._cap_blas(asked)
    assert calls == [dict(limits=held, user_api="blas")]
