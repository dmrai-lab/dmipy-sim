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


@pytest.mark.slow
def test_a_twenty_thousand_triangle_sphere_walks_at_default_arguments():
    """The case that asked 18.5 GiB and died: default `feature_radius` (half the box), 20 480
    triangles, 4000 walkers. The cell now sits at the step rather than four steps, so the gather
    is a few tens of triangles and the walk runs."""
    V, F = mesh_shapes.icosphere(5e-6, subdivisions=5)
    assert F.shape[0] == 20480
    m = d.Mesh(V, F)
    assert m.C < 200 and m.memory_estimate(4000) < 4e9, (m.C, m.memory_estimate(4000))
    wf = d.set_b(d.pgse([[1, 0, 0]], 3e-3, 6e-3, gradient_strengths=0.1, n_t=60, slew_rate=np.inf), 5e8)
    s = np.asarray(d.simulate(4000, D, wf, m, seed=0, require_gpu=False)).ravel()
    assert 0.0 < s[0] < 1.0 and np.isfinite(s[0])
