"""A pack follows the spec's pools and never drops a walker silently (dmipy-sim#301): a packed cell walked with its
default seeding packs every walker at its pool's water, and replays to the raw-trajectory replay of the same walk;
a walk with walkers in a pool the spec says holds no water is refused."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.trajectories import replay


def _walk(geom, n, seed=0, r0=None):
    return d.simulate_trajectories(n, 2e-9, geom, 4e-3, 4e-5, seed=seed, walker_batch_size=n, require_gpu=False, r0=r0)


def _unit_weights(pack):
    """Every walker at weight 1: stored as such, or not stored at all (the implicit default)."""
    w = pack.arrays.get("spin_weights")
    return w is None or np.all(np.asarray(w) == 1.0)


def test_a_packed_cell_packs_both_pools_and_replays_to_the_trajectories():
    radii = np.array([1.5e-6, 1.0e-6])
    geom = d.PackedCylinders(radii, [[-2.5e-6, 0.0], [2.5e-6, 0.0]], 10e-6)
    walk = _walk(geom, 400)
    pool0 = np.asarray(walk.compartment)[:, 0]
    assert 0 < np.mean(pool0 == 1) < 1, "the default seeding covers both pools"
    pack = build_replay_pack(walk, id="test/packed", license="CC-BY-4.0", citation="test", K=16)
    assert _unit_weights(pack)
    seq = sequences.pgse([[1, 0, 0], [0, 0, 1]], 0.5e-3, 1.5e-3, bvalues=[2e9, 2e9], TE=4e-3, slew_rate=np.inf)
    f_intra = float(np.mean(pool0 == 1))
    mixed = (f_intra * pack.replay(seq, compartment=1, complex_signal=True)
             + (1 - f_intra) * pack.replay(seq, compartment=0, complex_signal=True))
    assert np.allclose(pack.replay(seq, complex_signal=True), mixed, atol=1e-6), "every walker counts once, at its pool's water"
    raw = np.abs(np.asarray(replay(np.asarray(walk.positions), float(walk.dt), np.asarray(seq.G_eff), float(seq.dt))))
    assert np.allclose(pack.replay(seq), raw, atol=2e-2), (pack.replay(seq), raw)


def test_walkers_in_a_dry_pool_are_refused():
    """An isolated cylinder's spec gives the outside no water; a walk seeded out there cannot be packed as if it
    were not there."""
    geom = d.Cylinder(1e-6, (0, 0, 1))
    r0 = np.column_stack([np.full(50, 2.5e-6), np.zeros(50), np.zeros(50)]).astype(np.float32)
    walk = _walk(geom, 50, r0=r0)
    assert np.all(np.asarray(walk.compartment)[:, 0] == 0)
    with pytest.raises(ValueError, match="holds no water"):
        build_replay_pack(walk, id="test/dry", license="CC-BY-4.0", citation="test", K=8)
    pack = build_replay_pack(walk, id="test/dry", license="CC-BY-4.0", citation="test", K=8, weights=np.ones(50))
    assert _unit_weights(pack)
