"""Save 0 of a persistent walk is t = 0: the initial position, no wall contact yet, the initial pool and bound state.
Before, the first stored sample was already one save interval into the walk, so the path ran one interval ahead of the
waveform clock and the initial position was never stored."""
import numpy as np

import jax

import dmipy_sim as d
from dmipy_sim.engine.mt_walk import simulate_mt_trajectories
from dmipy_sim.spec import geometry_from_spec
from dmipy_sim.substrate import Substrate

D0 = 2e-9


def test_the_first_save_is_the_start_for_every_producer():
    rng = np.random.default_rng(0)
    # a single-pool geometry (positions + contact + occupancy through the relaxation kernel)
    g = d.Cylinder(3e-6, (0, 0, 1))
    r0 = np.asarray(g.init_positions(50, jax.random.PRNGKey(3)))
    w = d.simulate_trajectories(50, D0, g, 1e-3, 2.5e-4, seed=0, r0=r0, require_gpu=False)
    np.testing.assert_allclose(np.asarray(w.positions)[:, 0], r0, atol=1e-12)
    assert np.all(np.asarray(w.boundary_local_time)[:, 0] == 0) and w.n_t == 5
    assert np.all(np.asarray(w.compartment)[:, 0] == np.asarray(w.compartment)[:, 0].astype(int))
    # a packed myelinated cell (the three-pool kernel)
    spec = Substrate.canonical(field_T=3.0).request(n_fibres=3, seed=1)
    pm = geometry_from_spec(spec)
    r0 = np.asarray(pm.init_positions(60, jax.random.PRNGKey(4)))
    w = d.simulate_trajectories(60, D0, pm, 1e-3, 2.5e-4, seed=1, r0=r0, require_gpu=False)
    np.testing.assert_allclose(np.asarray(w.positions)[:, 0], r0, atol=1e-12)
    assert np.all(np.asarray(w.boundary_local_time)[:, 0] == 0)
    # the MT producer: the initial bound state is the burn-in's equilibrium occupancy per walker
    wm = simulate_mt_trajectories(60, D0, d.Cylinder(3e-6, (0, 0, 1)), 1e-3, 2.5e-4, 1e-4, 2e-4, seed=2, require_gpu=False)
    assert set(np.unique(np.asarray(wm.bound_frac)[:, 0])) <= {0.0, 1.0} and np.all(np.asarray(wm.boundary_local_time)[:, 0] == 0)
    assert np.asarray(wm.positions).shape[1] == 5
