"""The spec's raster of a periodic cell is one period of its field (dmipy-sim#797): ``field_grid_of_spec`` spans the
domain box exactly and its field is the box's periodic convolution, so the grid states the domain's periodic axes
and a walker that has left the cell reads its image's field -- the same channels at ``r`` and at ``r`` displaced by
any whole number of cell periods, by the walk's own reader and by the host's. A grid that clamped there would read
its edge instead, which nothing downstream (the raster's shell certificate, the read rule's pilot, the band) sees."""
import os

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

UM = 1e-6


def _smoke_cactus(root, side=6.0):
    """Two capped myelinated tubes along z overrunning a periodic cell of ``side`` um (inner radius 1, outer 1.4), as a
    CACTUS run directory."""
    sim = os.path.join(root, "meshes", "simulations")
    os.makedirs(sim, exist_ok=True)
    for i, (cx, cy) in enumerate(((-0.25 * side, -0.2 * side), (0.25 * side, 0.2 * side))):
        for kind, r in (("inner", 1.0), ("outer", 1.4)):
            m = trimesh.creation.cylinder(radius=r, height=1.6 * side, sections=48)
            m.apply_translation([cx, cy, 0.0])
            m.export(os.path.join(sim, f"strand_{i:05d}_{kind}_erode_0.ply"))
    return root


def test_the_spec_raster_of_a_periodic_cell_reads_the_same_field_one_period_away(tmp_path):
    import jax
    import jax.numpy as jnp
    from dmipy_sim.spec.producers import cactus_spec
    from dmipy_sim.spec.walk import field_grid_of_spec
    import dataclasses
    from dmipy_replay.spec.substrate import Domain
    spec = cactus_spec(_smoke_cactus(str(tmp_path)), side_um=6.0)         # a CACTUS voxel reflects; this cell is periodic
    spec = dataclasses.replace(spec, domain=Domain(spec.domain.box_min, spec.domain.box_max, ["periodic"] * 3))
    fg = field_grid_of_spec(spec)
    lo, hi = np.asarray(spec.domain.box_min, float), np.asarray(spec.domain.box_max, float)
    assert fg.periodic == (True, True, True)
    np.testing.assert_allclose(fg.origin, lo, atol=1e-15)
    np.testing.assert_allclose(np.asarray(fg.basis["shape"]) * np.asarray(fg.basis["voxel_size"]), hi - lo, rtol=1e-12)
    L = hi - lo
    rng = np.random.default_rng(0)
    p = lo + rng.random((4000, 3)) * L
    shifts = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 2], [1, -1, 1]], float) * L
    arrays, read, _ = fg.device_read()
    at = jax.jit(jax.vmap(lambda r: read(arrays, r)))
    base_dev = np.asarray(at(jnp.asarray(p, jnp.float32)), np.float64)
    base_host = fg.channels(p)
    scale = np.abs(base_host).max()
    for s in shifts:
        q = p + s
        # the walk's float32 reader: the shifted index rounds at a few ulp of the cell's node count
        np.testing.assert_allclose(np.asarray(at(jnp.asarray(q, jnp.float32)), np.float64), base_dev, rtol=0, atol=1e-4 * scale)
        np.testing.assert_allclose(fg.channels(q), base_host, rtol=0, atol=1e-9 * scale)
    # what the grid read before it stated its axes: the clamped edge, another field for most displaced walkers
    clamped = fg._replace(periodic=(False, False, False)).channels(p + shifts[0])
    assert np.abs(clamped - base_host).max() > 0.1 * scale
