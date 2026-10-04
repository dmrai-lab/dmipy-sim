"""The thread cap on the LAPACK calls a known BLAS build hangs on (dmipy_sim._blas, dmrai-lab/dmipy-sim#564):
applied only on that build, inert everywhere else, and the expansion's numbers the same either way."""
import contextlib

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import _blas, sequences
from dmipy_sim.replay import read_rpk
from dmipy_sim.replay.bank import build_replay_pack


def _pool(**kw):
    p = dict(user_api="blas", internal_api="openblas", version="0.3.28", architecture="neoversev1", num_threads=64)
    p.update(kw)
    return p


@pytest.mark.parametrize("pool", [_pool(version="0.3.27"), _pool(architecture="Haswell"), _pool(internal_api="mkl"),
                                  _pool(num_threads=8), _pool(user_api="openmp")])
def test_an_unaffected_build_is_left_alone(monkeypatch, pool):
    import threadpoolctl

    def no_limits(*a, **k):
        raise AssertionError("the guard capped an unaffected build")

    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [pool])
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", no_limits)
    assert isinstance(_blas.lapack_threads(), contextlib.nullcontext)


def test_the_hanging_build_is_capped(monkeypatch):
    import threadpoolctl
    calls = []
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [_pool()])
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", lambda **k: calls.append(k) or contextlib.nullcontext())
    with _blas.lapack_threads():
        pass
    assert calls == [dict(limits=_blas.SAFE_THREADS, user_api="blas")]


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    g = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6)
    walk = d.simulate_trajectories(500, 2e-9, g, 10e-3, 5e-4, seed=2, require_gpu=False)
    p = tmp_path_factory.mktemp("pk") / "c.rpk"
    build_replay_pack(walk, id="t/c", license="x", citation="x", K=8, out_path=str(p))
    return read_rpk(str(p))


def test_the_residual_bound_is_the_same_numbers_guarded_or_not(pack, monkeypatch):
    """The residual's SVD in the closed form's bound, with the guard forced on (the real thread cap applied) and
    off: the same expansion and the same misfit, to the bit."""
    played = sequences.pgse([[1, 0, 0], [0.0, 0.6, 0.8]], 2e-3, 5e-3, gradient_strengths=[0.3, 0.3], TE=10e-3,
                            slew_rate=200.0).with_concomitant(np.array([0.03, 0.05, 0.06]), 0.5)
    monkeypatch.setattr(_blas, "_affected", lambda: [])
    off = pack.pose_response(played)
    monkeypatch.setattr(_blas, "_affected", lambda: [_pool()])
    on = pack.pose_response(played)
    assert off.route == "closed" and off.misfit.max() > 1e-3                 # the bound is in play
    np.testing.assert_array_equal(on.coeffs, off.coeffs)
    np.testing.assert_array_equal(on.misfit, off.misfit)
