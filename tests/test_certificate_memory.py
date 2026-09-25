"""The certificate never holds a float64 copy of the walk.

`measure_fidelity` needs the phases of every walker, a few hundred numbers each, and reads the
positions in walker chunks to get them; the split-half floor of a raw walk reads them once. At
250,000 walkers and 3,246 saves the float32 walk is 9.7 GB, a float64 copy of it 19 GB (#438).
"""
import tracemalloc

import numpy as np

from dmipy_sim.replay import bank
from dmipy_sim.replay.compression import measure_fidelity


def _walk(n_w=3000, n_t=400, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(size=(n_w, n_t, 3)).astype(np.float32) * np.float32(1e-7)
    return np.cumsum(steps, axis=1, dtype=np.float32), 1e-5


def _peak(fn):
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_certificate_peak_is_bounded_by_the_chunk_not_the_walk():
    traj, dt = _walk()
    dec = traj + np.float32(1e-9)
    chunk = traj.nbytes // 8
    peak = _peak(lambda: measure_fidelity(traj, dt, dec, chunk_bytes=chunk))
    assert peak < traj.nbytes, f"peak {peak / 1e6:.1f} MB for a {traj.nbytes / 1e6:.1f} MB float32 walk"


def test_chunking_does_not_change_the_verdict():
    traj, dt = _walk(n_w=600, n_t=120)
    dec = traj + np.float32(1e-9)
    whole = measure_fidelity(traj, dt, dec)
    chunked = measure_fidelity(traj, dt, dec, chunk_bytes=traj.nbytes // 7)
    assert np.isclose(whole["err_max"], chunked["err_max"], rtol=1e-12, atol=1e-15)
    assert np.isclose(whole["floor_max"], chunked["floor_max"], rtol=1e-12, atol=1e-15)


def test_raw_floor_reads_the_walk_once_and_in_place():
    traj, dt = _walk()
    m = dict(traj=traj, dt_traj=dt)
    peak = _peak(lambda: bank._measure_floor(m, None, chunk_bytes=traj.nbytes // 8))
    assert peak < traj.nbytes, f"peak {peak / 1e6:.1f} MB for a {traj.nbytes / 1e6:.1f} MB float32 walk"
    ref = measure_fidelity(traj, dt, traj.copy())
    assert np.isclose(bank._measure_floor(m, None), ref["floor_max"], rtol=1e-12, atol=1e-15)
