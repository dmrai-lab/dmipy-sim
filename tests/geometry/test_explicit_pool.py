"""The pool a geometry seeds is declared on the geometry, not defaulted inside the seeding call.

`Mesh(pool=)` and `CurvedMyelinatedCylinder(pool=)` say which pool `init_positions` fills when a driver
is given no `r0`; the call can still name a pool explicitly; the old `intra=` / `shell=` flags warn.
"""
import jax
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.geometry import mesh_shapes
from dmipy_sim.fields.susceptibility_field import mesh_contains


def _mesh(pool="intra"):
    V, F = mesh_shapes.icosphere(2e-6, subdivisions=2)
    return d.Mesh(V, F, feature_radius=1e-6, pool=pool), V, F


def _inside(V, F, pts):
    return mesh_contains(np.asarray(V, float), np.asarray(F, np.int64), np.asarray(pts, float))


def test_the_constructor_pool_is_what_a_driver_seeds():
    m_in, V, F = _mesh("intra")
    m_out, _, _ = _mesh("extra")
    assert (m_in.pool, m_out.pool) == ("intra", "extra")
    k = jax.random.PRNGKey(0)
    assert _inside(V, F, m_in.init_positions(300, k)).all()
    assert not _inside(V, F, m_out.init_positions(300, k)).any()
    # a driver with no r0 walks the declared pool
    w = d.simulate_trajectories(200, 2e-9, m_out, 1e-3, 5e-4, seed=0, require_gpu=False)
    assert (w.compartment[:, 0] == 0).all()
    with pytest.raises(ValueError, match="pool must be"):
        d.Mesh(V, F, pool="myelin")


def test_the_call_can_name_a_pool_and_the_old_flag_warns():
    m, V, F = _mesh("intra")
    k = jax.random.PRNGKey(1)
    assert not _inside(V, F, m.init_positions(200, k, pool="extra")).any()
    with pytest.warns(DeprecationWarning, match="pool="):
        old = m.init_positions(200, k, intra=False)
    np.testing.assert_array_equal(np.asarray(old), np.asarray(m.init_positions(200, k, pool="extra")))
    with pytest.raises(ValueError, match="not both"):
        with pytest.warns(DeprecationWarning):
            m.init_positions(10, k, pool="extra", intra=True)


def test_curved_cylinder_shells_are_pools_too():
    z = np.linspace(-20e-6, 20e-6, 9)
    cl = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    g = d.CurvedMyelinatedCylinder(cl, 1e-6, 2e-6, pool="myelin")
    k = jax.random.PRNGKey(2)
    r = np.linalg.norm(np.asarray(g.init_positions(500, k))[:, :2], axis=1)
    assert (r >= 1e-6).all() and (r <= 2e-6).all()
    with pytest.warns(DeprecationWarning, match="pool="):
        old = g.init_positions(500, k, shell="myelin")
    np.testing.assert_array_equal(np.asarray(old), np.asarray(g.init_positions(500, k)))
    with pytest.raises(ValueError, match="pool must be"):
        d.CurvedMyelinatedCylinder(cl, 1e-6, 2e-6, pool="csf")
