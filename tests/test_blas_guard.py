"""The floor :data:`dmipy_sim._blas.SAFE_THREADS` puts under :func:`dmipy_sim._threads.cap_cpu_threads`
on the one OpenBLAS build whose multithreaded drivers hang (dmrai-lab/dmipy-sim#564): BLAS never
runs more than that build's floor, whatever ``DMIPY_SIM_CPU_THREADS`` asked for; every other build
gets exactly what was asked. The pose expansion's SVD gives the same numbers at any thread count."""
import numpy as np
import pytest
import threadpoolctl

import dmipy_sim as d
from dmipy_sim import _blas, _threads, sequences
from dmipy_sim.replay import read_rpk
from dmipy_sim.replay.bank import build_replay_pack


def _pool(**kw):
    p = dict(user_api="blas", internal_api="openblas", version="0.3.28", architecture="neoversev1", num_threads=64)
    p.update(kw)
    return p


@pytest.mark.parametrize("pool", [_pool(version="0.3.27"), _pool(architecture="Haswell"), _pool(internal_api="mkl"),
                                  _pool(num_threads=8), _pool(user_api="openmp")])
def test_an_unaffected_build_gets_exactly_what_was_asked(monkeypatch, pool):
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [pool])
    calls = []
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", lambda **k: calls.append(k) or None)
    _threads._cap_blas(72)
    assert calls == [dict(limits=72, user_api="blas")]


@pytest.mark.parametrize("requested,expect", [(72, _blas.SAFE_THREADS), (3, 3)])
def test_the_hanging_build_never_exceeds_the_floor(monkeypatch, requested, expect):
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [_pool()])
    calls = []
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", lambda **k: calls.append(k) or None)
    _threads._cap_blas(requested)
    assert calls == [dict(limits=expect, user_api="blas")]


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    g = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6)
    walk = d.simulate_trajectories(500, 2e-9, g, 10e-3, 5e-4, seed=2, require_gpu=False)
    p = tmp_path_factory.mktemp("pk") / "c.rpk"
    build_replay_pack(walk, id="t/c", license="x", citation="x", K=8, out_path=str(p))
    return read_rpk(str(p))


def test_the_residual_bound_is_the_same_numbers_at_any_thread_count(pack):
    """The residual's SVD in the closed form's bound gives the same expansion and misfit, to the
    bit, at one BLAS thread or many -- the process-wide cap changes nothing about the physics."""
    played = sequences.pgse([[1, 0, 0], [0.0, 0.6, 0.8]], 2e-3, 5e-3, gradient_strengths=[0.3, 0.3], TE=10e-3,
                            slew_rate=200.0).with_concomitant(np.array([0.03, 0.05, 0.06]), 0.5)
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        one = pack.pose_response(played)
    with threadpoolctl.threadpool_limits(limits=max(2, _threads.cpu_threads()), user_api="blas"):
        many = pack.pose_response(played)
    assert one.route == "closed" and one.misfit.max() > 1e-3  # the bound is in play
    np.testing.assert_array_equal(one.coeffs, many.coeffs)
    np.testing.assert_array_equal(one.misfit, many.misfit)
