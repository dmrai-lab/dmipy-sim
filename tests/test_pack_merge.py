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
def spec_grid(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("shards")
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp / "t.tck"), str(tmp / "d.txt")
    write_tck(tck, cls_, coordinate_unit_m=25e-6); np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    spec = disco_spec(tck, dia, side_m=20e-6)
    grid = Grid(shape=(2, 2, 2), voxel_size_m=(10e-6,) * 3, origin_m=(5e-6,) * 3)
    return spec, grid, tmp


@pytest.fixture(scope="module")
def shards(spec_grid):
    spec, grid, tmp = spec_grid
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



def test_passes_of_one_block_merge_with_the_union_weights(spec_grid):
    """A first pass of 4 walkers per voxel and pool and a top-up of 12, seeded apart, walked on the same block:
    the merged shard's walkers of a (voxel, pool) all carry one weight, the voxel's fraction over the union's
    count (each pass alone summed to that fraction with its own count), the certificate counts the union, and
    the replay is the union's weighted signal, not the two passes weighed alike."""
    spec, grid, tmp = spec_grid
    packs, walks = [], []
    for p_, per in ((1, 4), (2, 12)):
        want = np.zeros(grid.shape, np.int64); want[0] = per
        w = walk_spec(spec, T_max=8e-4, dt_save=2e-4, seed=100 * p_, require_gpu=False, field=False,
                      seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}))
        walks.append(w)
        packs.append(build_replay_pack(w, id=f"t/pass-{p_}", license="x", citation="x", K=3, voxel_grid=grid, out_path=str(tmp / f"pass-{p_}.rpk")))
    m = merge_packs(packs, id="t/union", out_path=str(tmp / "union.rpk"), overlap="recertify")
    from dmipy_sim.replay import compression as _cx
    start = _cx.read_position_coeffs(m.arrays, dtype=np.float64)[:, 0, :]                          # the walkers' start positions
    vox = grid.bin(start)[0]; pool = np.asarray(_cx.decode_occupancy(m.arrays, m.meta["compression"]["channels"]["compartment"])["comp"])[:, 0]
    cells = [tuple(v) + (int(p_),) for v, p_ in zip(vox.tolist(), pool)]
    shard = np.repeat([0, 1], [pk.n_walkers for pk in packs])
    wm = np.asarray(m.spin_weights, np.float64); ws = np.concatenate([np.asarray(pk.spin_weights, np.float64) for pk in packs])
    for c in set(cells):
        sel = np.array([k_ == c for k_ in cells])
        np.testing.assert_allclose(wm[sel], wm[sel][0], rtol=1e-6)                                     # one weight per (voxel, pool)
        own = [ws[sel & (shard == s_)].sum() for s_ in (0, 1) if (sel & (shard == s_)).any()]        # each pass's census of the fraction
        np.testing.assert_allclose(wm[sel].sum(), np.mean(own), rtol=1e-6)                             # the union takes their mean
    g, floors, counts = voxel_fidelity_volumes(m)
    for name in counts:
        assert counts[name][0].sum() == 16 * 4 and counts[name][1].sum() == 0
    seq = d.set_b(d.pgse([[1, 0, 0]], 0.2e-3, 0.5e-3, gradient_strengths=0.1, n_t=m.n_t, slew_rate=np.inf), [1e9])
    E = np.concatenate([pk.walker_signals(seq, tissue=False)[2] for pk in packs])                     # per-walker complex signals
    np.testing.assert_allclose(m.replay(seq, tissue=False), np.abs((wm[:, None] * E).sum(0) / wm.sum()), rtol=1e-6)
    alike = np.abs((ws[:, None] * E).sum(0) / ws.sum())
    assert not np.allclose(m.replay(seq, tissue=False), alike, rtol=1e-6)
