"""A distributed fill: the packs of disjoint voxel blocks of one walk merge into one pack.

Each device seeds and walks its block of the voxel grid and packs it with the per-voxel certificate; `merge_packs`
concatenates the walker-indexed arrays, unions the certificates, and refuses two shards that hold the same voxel.
The merged pack replays exactly as the shards combined by their walker weights.
"""
from __future__ import annotations

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.io.strands import write_tck
from dmipy_sim.phantom import Grid
from dmipy_sim.replay.bank import build_replay_pack, merge_packs, voxel_fidelity_volumes
from dmipy_sim.spec import disco_spec, walk_spec, StratifiedByVoxel


@pytest.fixture(scope="module")
def shards(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("shards")
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp / "t.tck"), str(tmp / "d.txt")
    write_tck(tck, cls_, coordinate_unit_m=25e-6); np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    spec = disco_spec(tck, dia, side_m=20e-6)
    grid = Grid(shape=(2, 2, 2), voxel_size_m=(10e-6,) * 3, origin_m=(5e-6,) * 3)
    packs = []
    for b in (0, 1):
        want = np.zeros(grid.shape, np.int64); want[b] = 12                    # the block: one x-slab of the grid
        w = walk_spec(spec, T_max=8e-4, dt_save=2e-4, seed=7 + b, require_gpu=False, field=False,
                      seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}))
        packs.append(build_replay_pack(w, id=f"t/shard-{b}", license="x", citation="x", K=3, voxel_grid=grid,
                                       out_path=str(tmp / f"shard-{b}.rpk")))
    return packs, grid, tmp


def test_the_merged_pack_replays_as_the_weight_combined_shards(shards):
    (a, b), grid, tmp = shards
    m = merge_packs([a, b], id="t/merged", out_path=str(tmp / "merged.rpk"))
    assert m.n_walkers == a.n_walkers + b.n_walkers and m.meta["walk_params"]["seed"] == [7, 8]
    seq = d.set_b(d.pgse([[1, 0, 0], [0, 0, 1]], 0.2e-3, 0.5e-3, gradient_strengths=0.1, n_t=m.n_t, slew_rate=np.inf), [1e9, 1e9])
    wa, wb = np.asarray(a.spin_weights).sum(), np.asarray(b.spin_weights).sum()
    def combined(**kw):                                                        # the shards' complex signals, by weight
        return np.abs((wa * a.replay(seq, complex_signal=True, **kw) + wb * b.replay(seq, complex_signal=True, **kw)) / (wa + wb))
    expect = combined(tissue=False)
    np.testing.assert_allclose(m.replay(seq, tissue=False), expect, rtol=1e-6)
    np.testing.assert_allclose(m.replay(seq, tissue=False, rho=1e-5, D=1.7e-9), combined(tissue=False, rho=1e-5, D=1.7e-9), rtol=1e-6)
    from dmipy_sim.replay import read_rpk
    back = read_rpk(str(tmp / "merged.rpk"))
    np.testing.assert_allclose(back.replay(seq, tissue=False), expect, rtol=1e-6)
    # the certificate: the union of the two blocks, each voxel from its shard
    g, floors, counts = voxel_fidelity_volumes(m)
    _, fa, ca = voxel_fidelity_volumes(a); _, fb, cb = voxel_fidelity_volumes(b)
    for name in counts:
        np.testing.assert_array_equal(counts[name], ca[name] + cb[name])
        np.testing.assert_allclose(floors[name], np.where(ca[name] > 0, fa[name], fb[name]))
    pv = m.meta["fidelity"]["per_voxel"]
    assert pv["shards"] == 2 and pv["n_voxels"] == 8 and m.meta["provenance"]["shards"][1]["id"] == "t/shard-1"
    assert not m.meta["compression"]["precision_tiers"]["usable"]                 # ordered by shard, not shuffled


def test_two_shards_of_the_same_block_are_refused(shards):
    (a, b), grid, tmp = shards
    with pytest.raises(ValueError, match="same voxel"):
        merge_packs([a, a], id="t/dup")
