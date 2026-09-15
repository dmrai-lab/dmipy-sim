"""Seeds drawn ahead of the walk: `draw_seeds(spec, StratifiedByVoxel, seed)` returns a `DrawnSeeds` that `walk_spec`
takes in place of the seeding it was drawn from, with the same walk to the bit -- what a producer draws for its
next block on the CPU while the device walks this one (dmipy-sim#258)."""
import numpy as np
import pytest

from dmipy_sim.io.strands import write_tck
from dmipy_sim.phantom import Grid
from dmipy_sim.spec import DrawnSeeds, StratifiedByVoxel, disco_spec, draw_seeds, walk_spec


@pytest.fixture(scope="module")
def spec_grid(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("drawn")
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp / "t.tck"), str(tmp / "d.txt")
    write_tck(tck, cls_, coordinate_unit_m=25e-6); np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    return disco_spec(tck, dia, side_m=20e-6), Grid(shape=(2, 2, 2), voxel_size_m=(10e-6,) * 3, origin_m=(5e-6,) * 3)


def test_a_walk_from_drawn_seeds_is_the_walk_from_the_seeding(spec_grid):
    spec, grid = spec_grid
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
    spec, grid = spec_grid
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
    spec, grid = spec_grid
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
