"""The permeable walk's step (dmipy-sim#292): a zero permeability is an impermeable wall, and a permeable one
steps at a count that scales with kappa -- the crossing rule -- not at a fixed fraction of the smallest tube."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine.physics import (resolve_sub_steps, walk_sub_steps, crossing_sub_steps, permeable_sub_steps,
                                      surface_sub_steps, CROSSING_P_MAX)

D, DT = 2e-9, 2e-4                                          # the reporter's diffusivity and save interval


def _packing(kappa, rho=1.2e-6, n=60, seed=17):
    rng = np.random.default_rng(seed)
    radii = np.clip(rng.gamma((0.8 / 0.5) ** 2, 0.5 ** 2 / 0.8, n), 1.0, 3.0) * 1e-6
    centers, L = d.pack_cylinders(radii, target_vf=0.5, seed=seed)[:2]
    return d.PackedCylinders(radii, centers, L, surface_relaxivity_t2=rho, permeability=kappa)


@pytest.mark.parametrize("make", [lambda k: d.Cylinder(radius=1e-6, orientation=(0, 0, 1), permeability=k),
                                  lambda k: d.Sphere(radius=1e-6, permeability=k),
                                  lambda k: _packing(k)], ids=["cylinder", "sphere", "packing"])
def test_a_zero_permeability_is_an_impermeable_wall(make):
    """``permeability=0.0`` stores None: the reflect path, the impermeable step rule and the compartment guard
    apply, exactly as with ``permeability=None``; a negative value is refused."""
    assert make(0.0).permeability is None and make(None).permeability is None
    assert make(1e-5).permeability == 1e-5
    assert resolve_sub_steps(make(0.0), D, DT, surface=True) == resolve_sub_steps(make(None), D, DT, surface=True)
    with pytest.raises(ValueError, match="non-negative"):
        make(-1e-6)


def test_the_crossing_rule_scales_with_kappa():
    """On the reporter's packing (radii floored at 1 um, rho 1.2 um/s, 0.2 ms saves) the impermeable count is the
    surface rule's; at 10 um/s the crossing rule asks for nothing finer, at 30 um/s it asks for the count that
    keeps ``p`` at ``CROSSING_P_MAX``, a sixth of the fixed R/25 count of 1500."""
    n0 = resolve_sub_steps(_packing(None), D, DT, surface=True)
    assert n0 == surface_sub_steps(_packing(None), D, DT) == 154
    n10 = resolve_sub_steps(_packing(10e-6), D, DT, surface=True)
    n30 = resolve_sub_steps(_packing(30e-6), D, DT, surface=True)
    assert n10 == n0
    assert n30 == crossing_sub_steps(_packing(30e-6), D, DT) > n0
    step = np.sqrt(6 * D * DT / n30)
    assert 2 * 30e-6 / D * step <= CROSSING_P_MAX + 1e-12
    assert n30 == int(np.ceil(DT / ((CROSSING_P_MAX * D / (2 * 30e-6)) ** 2 / (6 * D)))) < 1500 / 2
    assert permeable_sub_steps(_packing(30e-6), D, DT) == max(walk_sub_steps(_packing(30e-6), D, DT), n30)


def test_a_zero_permeability_walks_as_no_permeability():
    """The same seed, the same walk: positions identical to the bit, no label change either way."""
    from dmipy_sim import simulate_trajectories
    g0, gn = _packing(0.0, n=12), _packing(None, n=12)
    w0 = simulate_trajectories(64, D, g0, 1e-3, DT, seed=3, walker_batch_size=64, storage_dtype=np.float32)
    wn = simulate_trajectories(64, D, gn, 1e-3, DT, seed=3, walker_batch_size=64, storage_dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(w0.positions), np.asarray(wn.positions))
    if w0.compartment is not None:
        assert (np.asarray(w0.compartment) == np.asarray(w0.compartment)[:, :1]).all()
