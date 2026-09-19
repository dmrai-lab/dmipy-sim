"""A pack merged from a distributed fill carries the list of its shards' seeds, and every operation that
derives a pack from it carries that list on rather than coercing it to one number."""
from __future__ import annotations

import numpy as np

from dmipy_sim import Cylinder, simulate_trajectories
from dmipy_sim.replay.bank import build_replay_pack, merge_packs, seed_value


def _pack(seed, id):
    walk = simulate_trajectories(200, 2e-9, Cylinder(4e-6, (0, 0, 1)), T_max=0.02, dt_save=2e-4,
                                 seed=seed, require_gpu=False)
    return build_replay_pack(walk, id=id, license="CC-BY-4.0", citation="test", K=16)


def test_seed_value_keeps_a_list_and_coerces_a_scalar():
    assert seed_value(3) == 3
    assert seed_value(np.int64(3)) == 3
    assert seed_value([1, 2, 3]) == [1, 2, 3]
    assert seed_value(np.array([1, 2])) == [1, 2]
    assert seed_value(None) == 0


def test_a_merged_pack_can_be_prefixed():
    """The defect this locks: ``prefix`` read the merged pack's seed with ``int()`` and raised on the list,
    so a pack built by a fill could not be cut to a shorter echo time at all."""
    merged = merge_packs([_pack(0, "test/a"), _pack(1, "test/b")], id="test/merged")
    assert merged.meta["walk_params"]["seed"] == [0, 1]

    short = merged.prefix(0.01)
    assert short.n_t == int(round(0.01 / merged.dt)) + 1
    assert short.meta["walk_params"]["seed"] == [0, 1]
    assert short.n_walkers == merged.n_walkers
