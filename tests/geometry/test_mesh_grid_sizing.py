"""The mesh grid is sized by the triangles, the sub-step follows, and the memory is knowable.

The collision rule bounds a sub-step at 0.9 cell, so the cell trades candidates per gather
(`(cell/edge)^2`) against sub-steps per waveform step (`(step/cell)^2`) at constant cost; only
the device memory of the gather changes. The default therefore holds a few triangles per cell and
the walk is unchanged in physics (`test_mesh_acceleration_invariance` pins the signal against the
grid); this file pins the sizing and the footprint it used to run into.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.geometry import mesh_shapes

D = 2e-9


def test_default_cell_holds_a_few_triangles_not_hundreds():
    V, F = mesh_shapes.icosphere(1e-6, subdivisions=4)              # 5120 triangles
    m = d.Mesh(V, F, feature_radius=1e-6)
    step = 1e-6 / 6
    assert step <= m.cell_size <= 4 * step
    assert m.cell_size == pytest.approx(float(np.clip(3 * m.edge_p90, step, 4 * step)))
    assert m.C < 120, f"{m.C} candidates per cell"
    rep = m.quality_report(verbose=False)
    assert rep["candidates_per_cell"] == m.C
    assert rep["gather_bytes_per_walker"] == 27 * m.C * 21 * 4
    assert m.memory_estimate(4000) < 2e9


def test_an_explicit_cell_size_is_honoured():
    V, F = mesh_shapes.icosphere(1e-6, subdivisions=2)
    m = d.Mesh(V, F, feature_radius=1e-6, cell_size=0.5e-6)
    assert m.cell_size == 0.5e-6
