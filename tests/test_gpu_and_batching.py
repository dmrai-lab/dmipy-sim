"""GPU guard, device-memory cleanup, and walker batching.

The guard and free_gpu_memory are unit-tested (the guard with a monkeypatched
"no GPU"); batching is checked against the free-diffusion analytic E=exp(-bD),
since the per-walker mean recombines exactly as a size-weighted mean.
"""
import numpy as np
import pytest

from dmipy_sim import gpu
from dmipy_sim.gpu import gpu_available, check_gpu, free_gpu_memory, list_gpu_processes


def test_gpu_available_is_bool():
    assert isinstance(gpu_available(), bool)


def test_check_gpu_raises_when_required_and_absent(monkeypatch):
    monkeypatch.setattr(gpu, "gpu_available", lambda: False)
    with pytest.raises(RuntimeError, match="no GPU"):
        check_gpu(n_walkers=100_000, require_gpu=True, what="unit")


def test_check_gpu_warns_for_large_cpu_run(monkeypatch):
    monkeypatch.setattr(gpu, "gpu_available", lambda: False)
    with pytest.warns(RuntimeWarning, match="run on CPU"):
        check_gpu(n_walkers=gpu.LARGE_RUN_WALKERS, require_gpu=None, what="unit")


def test_check_gpu_silent_small_run_and_optout(monkeypatch, recwarn):
    monkeypatch.setattr(gpu, "gpu_available", lambda: False)
    check_gpu(n_walkers=10, require_gpu=None)          # small -> no warning
    check_gpu(n_walkers=10**7, require_gpu=False)       # explicit opt-out -> silent
    assert len(recwarn) == 0


def test_free_gpu_memory_safe_and_aggressive():
    import jax.numpy as jnp
    x = jnp.ones((128, 128))          # a live array
    assert free_gpu_memory() == 0     # safe default deletes nothing
    assert float(x.sum()) == 128 * 128  # still usable
    n = free_gpu_memory(aggressive=True)  # nuclear: deletes live arrays
    assert isinstance(n, int) and n >= 1


def test_list_gpu_processes_returns_list():
    procs = list_gpu_processes()
    assert isinstance(procs, list)
    for p in procs:
        assert {"pid", "used_mib", "name"} <= set(p)


@pytest.mark.slow
def test_walker_batching_matches_single_shot_and_analytic():
    from dmipy_sim import simulate, pgse, set_b, FreeDiffusion

    D = 2.0e-9
    bvecs = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    wf = set_b(pgse(delta=0.01, DELTA=0.03, G_magnitude=1.0, bvecs=bvecs, n_t=200),
               np.array([1.0, 1.0e9]))
    geom = FreeDiffusion()
    NW = 60_000

    single = np.abs(np.real(np.asarray(
        simulate(n_walkers=NW, diffusivity=D, waveform=wf, geometry=geom, seed=0))))
    batched, pos = simulate(n_walkers=NW, diffusivity=D, waveform=wf, geometry=geom,
                            seed=0, walker_batch_size=20_000, return_positions=True)
    batched = np.abs(np.real(np.asarray(batched)))

    E_analytic = np.exp(-1.0e9 * D)
    tol = max(0.02, 3.0 / np.sqrt(NW))
    assert pos.shape == (NW, 3)                      # concatenated across 3 chunks
    assert abs(single[1] - E_analytic) < tol
    assert abs(batched[1] - E_analytic) < tol
    assert abs(batched[1] - single[1]) < tol         # batched ~ single-shot (diff seeds)
    # b=0 normalisation preserved through batching
    assert abs(batched[0] - 1.0) < 1e-3
