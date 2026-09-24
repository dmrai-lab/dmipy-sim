"""Seeds drawn ahead of the walk: `draw_seeds(spec, StratifiedByVoxel, seed)` returns a `DrawnSeeds` that `walk_spec`
takes in place of the seeding it was drawn from, with the same walk to the bit -- what a producer draws for its
next block on the CPU while the device walks this one (dmipy-sim#258)."""
import numpy as np
import pytest

from dmipy_sim.io.strands import write_tck
from dmipy_sim.phantom import Grid
from dmipy_sim.spec import DrawnSeeds, StratifiedByVoxel, disco_spec, draw_seeds, walk_spec


def test_a_walk_from_drawn_seeds_is_the_walk_from_the_seeding(spec_grid):
    spec, grid, _ = spec_grid
    want = np.zeros(grid.shape, np.int64); want[0] = 6
    seeding = StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want})
    drawn = draw_seeds(spec, seeding, 11)
    assert isinstance(drawn, DrawnSeeds) and set(drawn.positions) == {"extra", "intra"} and drawn.seed == 11 and drawn.drawn_from is seeding
    assert drawn.n_walkers == sum(len(P) for P in drawn.positions.values()) > 0
    kw = dict(T_max=8e-4, dt_save=2e-4, seed=11, require_gpu=False, field=False)
    a = walk_spec(spec, seeding=seeding, **kw); b = walk_spec(spec, seeding=drawn, **kw)
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.compartment, b.compartment)
    np.testing.assert_array_equal(np.asarray(a.weights), np.asarray(b.weights))
    assert a.positions.shape[0] == drawn.n_walkers


def test_drawn_seeds_are_checked(spec_grid):
    spec, grid, _ = spec_grid
    with pytest.raises(ValueError, match="different pools"):
        DrawnSeeds(positions={"extra": np.zeros((2, 3))}, weights={"intra": np.ones(2)}, grid=grid, seed=0)
    with pytest.raises(ValueError, match="positions must be"):
        DrawnSeeds(positions={"extra": np.zeros((2, 2))}, weights={"extra": np.ones(2)}, grid=grid, seed=0)
    with pytest.raises(TypeError, match="StratifiedByVoxel"):
        draw_seeds(spec, "extra", 0)


def test_a_walk_context_is_kept_across_walks(spec_grid):
    """A walk with a `WalkContext` equals the walk without one; the same context serves a second walk with another
    seeding; a context of another spec, or one given beside `field_far`, is refused."""
    from dmipy_sim.spec import WalkContext
    spec, grid, _ = spec_grid
    want = np.zeros(grid.shape, np.int64); want[0] = 5
    seeding = StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want})
    kw = dict(T_max=8e-4, dt_save=2e-4, require_gpu=False, field=False)
    ctx = WalkContext(spec)
    assert ctx.key == WalkContext.key_of(spec, None) and ctx.field_basis() is None
    a = walk_spec(spec, seeding=seeding, seed=3, **kw); b = walk_spec(spec, seeding=seeding, seed=3, context=ctx, **kw)
    np.testing.assert_array_equal(a.positions, b.positions); np.testing.assert_array_equal(np.asarray(a.weights), np.asarray(b.weights))
    drawn = draw_seeds(spec, seeding, 4, context=ctx)
    c = walk_spec(spec, seeding=drawn, seed=4, context=ctx, **kw); d = walk_spec(spec, seeding=seeding, seed=4, **kw)
    np.testing.assert_array_equal(c.positions, d.positions)
    assert len(ctx.tests._bounds) > 0 and all(len(b_._geom) > 0 for b_ in ctx.tests._bounds.values())   # the geometries, kept
    with pytest.raises(TypeError, match="not both"):
        walk_spec(spec, seeding=seeding, seed=3, context=ctx, field_far="x.npy", **kw)
    with pytest.raises(TypeError, match="WalkContext"):
        walk_spec(spec, seeding=seeding, seed=3, context="ctx", **kw)


def test_a_pool_the_seeding_wants_nowhere_is_not_walked(spec_grid, tmp_path):
    """A round of a pass may hold none of a sparse pool: the seeding's zero map draws that pool empty, the walk
    carries the other pools only, its pack certifies them, and a merge with a shard that holds the pool counts
    the union; a pool wanted somewhere that no draw lands in, or a seeding with nothing in it, is refused."""
    from dmipy_sim.replay.bank import build_replay_pack, merge_packs, voxel_fidelity_volumes
    from dmipy_sim.spec import SpecError
    spec, grid, _ = spec_grid
    want = np.zeros(grid.shape, np.int64); want[0] = 6
    none = np.zeros(grid.shape, np.int64)
    kw = dict(T_max=8e-4, dt_save=2e-4, require_gpu=False, field=False)
    drawn = draw_seeds(spec, StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": none}), 5)
    assert len(drawn.positions["intra"]) == 0 and drawn.n_walkers == len(drawn.positions["extra"]) > 0
    w = walk_spec(spec, seeding=drawn, seed=5, **kw)
    intra = next(p.id for p in spec.pools if p.name == "intra")
    assert w.positions.shape[0] == drawn.n_walkers and intra not in set(np.unique(w.compartment).tolist())
    both = walk_spec(spec, seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}), seed=6, **kw)
    a = build_replay_pack(w, id="t/none", license="x", citation="x", K=3, voxel_grid=grid, out_path=str(tmp_path / "none.rpk"))
    b = build_replay_pack(both, id="t/both", license="x", citation="x", K=3, voxel_grid=grid, out_path=str(tmp_path / "both.rpk"))
    _, _, ca = voxel_fidelity_volumes(a); _, _, cb = voxel_fidelity_volumes(b)
    assert ca["extra"].sum() == drawn.n_walkers and ca.get("intra", np.zeros(1)).sum() == 0 and cb["intra"].sum() > 0
    m = merge_packs([a, b], id="t/union", overlap="recertify", out_path=str(tmp_path / "union.rpk"))
    _, _, cm = voxel_fidelity_volumes(m)
    assert m.n_walkers == a.n_walkers + b.n_walkers and cm["intra"].sum() == cb["intra"].sum()
    assert cm["extra"].sum() == ca["extra"].sum() + cb["extra"].sum()
    with pytest.raises(SpecError, match="nothing to walk"):
        walk_spec(spec, seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": none, "intra": none}), seed=5, **kw)
    wide = Grid(shape=(3, 2, 2), voxel_size_m=(10e-6,) * 3, origin_m=(5e-6,) * 3)   # its third column is past the box
    far = np.zeros(wide.shape, np.int64); far[2, 0, 0] = 4
    with pytest.raises(SpecError, match="no seed landed"):
        draw_seeds(spec, StratifiedByVoxel(grid=wide, walkers_per_voxel={"extra": np.zeros(wide.shape, np.int64), "intra": far}), 5)
