"""The helpers that existed twice now exist once, and the dead knobs are gone.

Each merge here replaced a copy with the original; the tests pin that the survivor gives the
copy's answer, and that the retired research knobs cannot be set by accident any more.
"""
import warnings

import numpy as np
import pytest

import dmipy_sim as d


def test_one_axis_to_z_rotation():
    from dmipy_sim.geometry.base import _rotation_to_z
    from dmipy_sim.fields.susceptibility import _axis_to_z_rotation
    rng = np.random.default_rng(0)
    for axis in list(rng.normal(size=(20, 3))) + [np.array([0, 0, 1.0]), np.array([0, 0, -1.0]),
                                                  np.array([1e-9, 0, 1.0])]:
        R = _axis_to_z_rotation(axis)
        np.testing.assert_allclose(R, _rotation_to_z(axis), atol=1e-12)
        np.testing.assert_allclose(R @ (axis / np.linalg.norm(axis)), [0, 0, 1], atol=1e-9)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_one_ensemble_signal():
    import jax.numpy as jnp
    from dmipy_sim.engine.core import _ensemble_signal
    rng = np.random.default_rng(1)
    w = jnp.asarray(rng.uniform(0.3, 1.0, 50), jnp.float32)
    phi = jnp.asarray(rng.normal(size=(50, 4)), jnp.float32)
    lw = jnp.asarray(-rng.uniform(0, 1, 50), jnp.float32)
    ref_w = np.sum(np.asarray(w)[:, None] * np.exp(np.asarray(lw))[:, None] * np.cos(np.asarray(phi)), 0) / np.sum(np.asarray(w))
    ref_np = np.sum(np.asarray(w)[:, None] * np.cos(np.asarray(phi)), 0) / np.sum(np.asarray(w))
    np.testing.assert_allclose(np.asarray(_ensemble_signal(w, phi, lw)), ref_w, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(_ensemble_signal(w, phi)), ref_np, rtol=1e-5)


def test_one_rf_increment():
    from dmipy_sim.viz import pedagogy
    from dmipy_sim.replay import trajectories
    assert pedagogy._rf_increment is trajectories._rf_increment


def test_retired_knobs_are_gone():
    from dmipy_sim.geometry import mesh_shapes
    V, F = mesh_shapes.icosphere(2e-6, subdivisions=1)
    m = d.Mesh(V, F, feature_radius=1e-6)
    assert not hasattr(m, "net_cross_check") and not hasattr(m, "_net_side_changed")
    assert not hasattr(d.geometry.base, "_is_inside_batch")
    sh = d.geometry.PermeableShell(2e-6, 4e-6, 1e-5)
    assert not hasattr(sh, "_dperp_mode")


def test_seed_in_cell_uses_the_exact_containment_test():
    from dmipy_sim.geometry import mesh_shapes
    from dmipy_sim.fields.susceptibility_field import mesh_contains
    V, F = mesh_shapes.icosphere(3e-6, subdivisions=2)
    m = d.Mesh(V, F, feature_radius=1e-6)
    pts = d.seed_in_cell(m, 500, seed=0)
    assert pts.shape == (500, 3)
    assert mesh_contains(np.asarray(V, float), np.asarray(F, np.int64), pts).all()


@pytest.mark.parametrize("flat", ["geometries", "mesh", "mesh_shapes", "curved_tube", "_boundary", "core", "physics",
                                  "bank", "trajectories", "waveforms", "pulse_sequence", "susceptibility", "pedagogy"])
def test_no_flat_path_shims(flat):
    """The old flat module paths are gone, not deprecated: only the packaged paths exist."""
    import importlib
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"dmipy_sim.{flat}")
