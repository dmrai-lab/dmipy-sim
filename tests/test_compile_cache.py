"""One compiled program per geometry and configuration, not one per call.

`simulate` (fused engine) and `simulate_trajectories` (the replay producer) keep their jitted
batch functions on the geometry object (`geometry._batch_cache`), keyed on its scalar state and
the configuration the closure bakes in. The waveform samples, positions, keys and labels are
arguments, so a sweep over b, seed or direction runs the same executable; a knob changed on the
geometry after a walk builds a new one; and the cached run is the fresh run to the bit.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import core

D = 2e-9


def _wf(b, n_t=100, direction=(1, 0, 0)):
    return d.set_b(d.pgse(delta=3e-3, DELTA=8e-3, G_magnitude=0.1, bvecs=[list(direction)], n_t=n_t,
                          slew_rate=np.inf), b)


def _entries(geometry):
    return vars(geometry).get(core._BATCH_CACHE_ATTR, {})


def test_a_sweep_over_b_and_seed_reuses_one_program():
    g = d.Sphere(3e-6)
    assert not _entries(g)
    s1 = d.simulate(1000, D, _wf(5e8), g, seed=0, require_gpu=False, engine="fused")
    s2 = d.simulate(1000, D, _wf(1.5e9), g, seed=7, require_gpu=False, engine="fused")
    s3 = d.simulate(1000, D, _wf(1.5e9, direction=(0, 0, 1)), g, seed=7, require_gpu=False, engine="fused")
    entries = _entries(g)
    assert len(entries) == 1, list(entries)
    fn = next(iter(entries.values()))
    assert fn._cache_size() == 1, "three calls at one shape must compile once"
    assert float(s1[0]) > float(s2[0])                      # b acted through the argument
    assert abs(float(s3[0]) - float(s2[0])) < 0.05           # a sphere: the direction cannot matter


def test_a_new_shape_or_configuration_is_a_new_program_and_a_knob_change_too():
    g = d.Sphere(3e-6)
    d.simulate(1000, D, _wf(1e9), g, seed=0, require_gpu=False, engine="fused")
    d.simulate(1000, D, _wf(1e9, n_t=150), g, seed=0, require_gpu=False, engine="fused")          # new n_t
    d.simulate(1000, D, _wf(1e9), g, seed=0, T2=0.05, require_gpu=False, engine="fused")          # T2 baked in
    d.simulate(1000, D, _wf(1e9), g, seed=0, return_compartments="final", require_gpu=False, engine="fused")
    n = len(_entries(g))
    assert n == 4, n
    d.simulate(500, D, _wf(1e9), g, seed=0, require_gpu=False, engine="fused")                     # new N: same entry
    assert len(_entries(g)) == 4
    g.surface_substep_frac = 0.0                                                   # a knob
    d.simulate(1000, D, _wf(1e9), g, seed=0, require_gpu=False, engine="fused")
    assert len(_entries(g)) == 5


def test_the_cached_run_is_the_fresh_run_to_the_bit():
    wf = _wf(1e9)
    g = d.Cylinder(2e-6, (0, 0, 1), surface_relaxivity_t2=1e-6)
    d.simulate(800, D, _wf(3e8), g, seed=1, require_gpu=False, engine="fused")                      # warm the cache
    cached = np.asarray(d.simulate(800, D, wf, g, seed=2, T2=0.1, require_gpu=False, engine="fused"))
    fresh = np.asarray(d.simulate(800, D, wf, d.Cylinder(2e-6, (0, 0, 1), surface_relaxivity_t2=1e-6),
                                  seed=2, T2=0.1, require_gpu=False, engine="fused"))
    np.testing.assert_array_equal(cached, fresh)


def test_every_engine_path_caches():
    from dmipy_sim.geometry import mesh_shapes
    V, F = mesh_shapes.icosphere(2e-6, subdivisions=2)
    m = d.Mesh(V, F, feature_radius=1e-6)
    mc = d.MyelinatedCylinder(2e-6, 3e-6, (0, 0, 1), D, D, water_fractions=(1.0, 0.0, 0.0))
    L = float(np.sqrt(np.pi * 3 * (1e-6 / 0.7) ** 2 / 0.5))
    _, _, c = d.pack_myelinated_cylinders([1e-6] * 3, 0.7, None, cell_size=L, seed=0)
    pm = d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4)
    for g, diff in ((m, D), (mc, None), (pm, None)):
        d.simulate(500, diff, _wf(5e8), g, seed=0, require_gpu=False, engine="fused")
        d.simulate(500, diff, _wf(1e9), g, seed=3, require_gpu=False, engine="fused")
        entries = _entries(g)
        assert len(entries) == 1 and next(iter(entries.values()))._cache_size() == 1, type(g).__name__
    cp = d.cpmg(3, 10e-3, 0.0, [[0, 0, 1]], n_t_per_echo=20)
    sp = d.Sphere(4e-6)
    d.simulate_cpmg(300, D, cp, sp, T2=0.05, seed=0, require_gpu=False)
    d.simulate_cpmg(300, D, cp, sp, T2=0.05, seed=1, require_gpu=False)
    assert any(k[0][0] == "cpmg" for k in _entries(sp))


def test_the_cache_dies_with_the_geometry():
    """The cached closure references the geometry; kept on the object that is a cycle, and the
    collector frees both. A module-level table keyed by the geometry would have pinned it."""
    import gc, weakref
    g = d.Sphere(3e-6)
    d.simulate(500, D, _wf(1e9), g, seed=0, require_gpu=False, engine="fused")
    assert _entries(g)
    ref = weakref.ref(g)
    del g
    gc.collect()
    assert ref() is None


def test_the_replay_producer_caches_too():
    g = d.Sphere(3e-6, surface_relaxivity_t2=1e-6)
    for seed in (0, 1):
        d.simulate_trajectories(400, D, g, 2e-3, 5e-4, seed=seed, require_gpu=False)
        d.simulate_trajectories(400, D, g, 2e-3, 5e-4, seed=seed, save_relaxation_data=True,
                                require_gpu=False)
    keys = [k[0][0] for k in _entries(g)]
    assert set(keys) == {"traj", "traj_relax"}, keys
    # the relaxation walk and the plain walk each compiled once (the producer also builds the
    # plain batch function it does not call on a relaxation run; that one never compiles)
    sizes = sorted(fn._cache_size() for fn in _entries(g).values())
    assert sizes == [0, 1, 1] or sizes == [1, 1], sizes
    # the replay engine of simulate() rides on the producer: a b-sweep compiles nothing new
    n0 = len(_entries(g))
    for b in (5e8, 1e9, 2e9):
        d.simulate(400, D, _wf(b, n_t=5), g, seed=0, require_gpu=False, engine="replay")
    assert len(_entries(g)) == n0 + 1 or len(_entries(g)) == n0 + 2
