"""The SO(3) coupling tables are constants and persist between processes (dmrai-lab/dmipy-sim#449 item 4)."""
import os

import numpy as np

from dmipy_sim.replay import so3


def test_the_tables_come_back_from_disk_equal_and_a_disabled_cache_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("DMIPY_SIM_CACHE", str(tmp_path))
    so3.coupling.cache_clear()
    first = so3.coupling(5, 3)
    files = sorted(os.listdir(tmp_path / "so3"))
    assert files == [f"coupling-v{so3.COUPLING_CACHE_VERSION}-5-3.npz"]
    so3.coupling.cache_clear()
    again = so3.coupling(5, 3)                                             # from the file
    assert again.keys() == first.keys()
    for L in first:
        assert np.array_equal(again[L], first[L])
    monkeypatch.setenv("DMIPY_SIM_CACHE", "0")
    so3.coupling.cache_clear()
    so3.coupling(4, 2)
    assert sorted(os.listdir(tmp_path / "so3")) == files                   # nothing written when disabled
