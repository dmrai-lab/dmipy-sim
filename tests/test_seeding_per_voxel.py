"""The stratified intra draw is per voxel, not a box-wide rejection.

`fill_swept_by_voxel` draws in each voxel from the segments whose tube meets it, by their clipped volume, and reads
the pool's volume fraction there from the clipped volume and the acceptance; `fill_per_voxel` draws uniform points
in each wanted voxel and keeps them by the pool's membership, so a block of the grid is seeded at the cost of its
own voxels (a box-wide volume-uniform draw put 46 voxels of 64,000 within reach of one draw in 1400).
"""
from __future__ import annotations

import numpy as np

import dmipy_sim as d
from dmipy_sim.phantom import Grid
from dmipy_sim.spec.seeding import fill_swept_by_voxel, fill_per_voxel

R = 1.0e-6


def _strands():
    rng = np.random.default_rng(3)
    cls = []
    for k in range(12):
        p0 = rng.uniform(-8e-6, 8e-6, 3); u = rng.normal(size=3); u /= np.linalg.norm(u)
        cls.append(np.stack([p0 - 14e-6 * u, p0 + 0.3e-6 * rng.normal(size=3), p0 + 14e-6 * u]))
    return cls, np.full(12, R)


def test_the_per_voxel_draw_fills_every_voxel_the_tubes_meet_with_points_inside_both():
    cls, rr = _strands()
    A = np.vstack([c[:-1] for c in cls]); B = np.vstack([c[1:] for c in cls]); r = np.repeat(rr, 2)
    grid = Grid(shape=(4, 4, 4), voxel_size_m=(5e-6,) * 3, origin_m=(-7.5e-6,) * 3)
    want = np.full(grid.n_voxels, 25)
    P, v, f, n = fill_swept_by_voxel(A, B, r, grid, want, seed=0)
    g = d.PackedCurvedCylinders(cls, rr, interior=True)
    assert g.inside_any(P).all()
    ijk, inside = grid.bin(P)
    assert inside.all() and (np.ravel_multi_index(tuple(ijk.T), grid.shape) == v).all()
    cnt = np.bincount(v, minlength=grid.n_voxels)
    full = cnt == 25
    assert full.sum() > 20 and (f[full] > 0).all() and (f[cnt == 0] == 0).all()
    assert (f[(cnt > 0) & ~full] < 1e-3).all()                                # only a sliver a tube barely touches is short
    assert n < 40 * want.sum()


def test_the_per_voxel_census_is_the_box_wide_one():
    """The volume fraction per voxel read from the clipped volume and the acceptance equals the census of a
    box-wide volume-uniform draw, voxel by voxel, to the two draws' noise."""
    cls, rr = _strands()
    A = np.vstack([c[:-1] for c in cls]); B = np.vstack([c[1:] for c in cls]); r = np.repeat(rr, 2)
    grid = Grid(shape=(4, 4, 4), voxel_size_m=(5e-6,) * 3, origin_m=(-7.5e-6,) * 3)
    want = np.full(grid.n_voxels, 10)
    _, _, f_new, _ = fill_swept_by_voxel(A, B, r, grid, want, seed=0, census_draws=20_000)
    g = d.PackedCurvedCylinders(cls, rr, interior=True)
    rng = np.random.default_rng(0); n_drawn = 2_000_000
    Q = g.sample_inside(n_drawn, rng)                                          # volume-uniform in the tubes
    ijk, ins = grid.bin(Q)
    trials = np.bincount(np.ravel_multi_index(tuple(np.clip(ijk, 0, 3).T), grid.shape)[ins], minlength=grid.n_voxels)
    V_pool = float((np.pi * r ** 2 * np.linalg.norm(B - A, axis=1)).sum())
    f_old = trials / n_drawn * V_pool / float(np.prod(grid.voxel_size_m))
    m = trials > 2000
    assert m.sum() > 10
    np.testing.assert_allclose(f_new[m], f_old[m], rtol=0.1)                  # the box-wide census's 2 % noise, and the new's 0.7 %
    np.testing.assert_allclose(f_new.sum(), f_old.sum(), rtol=0.02)
    # both count segment volumes (overlaps twice, no joint spheres), the strand family's convention; a membership
    # count of uniform points differs by up to 25 % in a voxel a joint or an overlap sits in


def test_the_rejection_draw_is_per_voxel_and_a_block_costs_its_own_voxels():
    """`fill_per_voxel` draws IN each wanted voxel and keeps by membership: a voxel that wants nothing costs
    nothing, every wanted voxel the pool reaches holds its count, the census is the membership fraction read on
    at least `census_draws` draws, and the draws scale with the block, not the grid."""
    cls, rr = _strands()
    g = d.PackedCurvedCylinders(cls, rr, interior=True)
    pred = lambda P: ~g.inside_any(P)                                           # the extra pool: outside every tube
    grid = Grid(shape=(20, 20, 20), voxel_size_m=(1e-6,) * 3, origin_m=(-9.5e-6,) * 3)
    want = np.zeros(grid.n_voxels, np.int64); block = np.zeros(grid.shape, bool); block[8:11, 8:11, :] = True
    want[block.reshape(-1)] = 30
    P, v, f, trials, n = fill_per_voxel(pred, grid, want, trials_max=5000, census_draws=400, seed=0)
    assert (trials[~block.reshape(-1)] == 0).all() and n <= 60 * want.sum()    # a block of 90 voxels costs its own draws
    ijk, ins = grid.bin(P); assert ins.all() and (np.ravel_multi_index(tuple(ijk.T), grid.shape) == v).all()
    assert pred(P).all()                                                        # every seed is in the pool
    cnt = np.bincount(v, minlength=grid.n_voxels)
    reached = (f > 0.05) & block.reshape(-1)
    assert reached.sum() > 40 and (cnt[reached] == 30).all()                    # every reachable voxel holds its count
    assert (trials[block.reshape(-1)] >= 400).all()                             # the census is read on the asked draws
    rng = np.random.default_rng(1); vids = np.flatnonzero(block.reshape(-1))
    corner = np.asarray(grid.corner_m); vs = np.asarray(grid.voxel_size_m)
    for vid in vids[:12]:                                                       # the census is the membership fraction
        ijk_ = np.array(np.unravel_index(vid, grid.shape)); Q = corner + (ijk_ + rng.uniform(0, 1, (20000, 3))) * vs
        assert abs(f[vid] - pred(Q).mean()) < 0.06


def test_every_seed_reads_as_inside_at_millimetre_coordinates():
    """Seeds at 1 mm: the disc is drawn to the wall less the representable nudge, so the geometry's own float32
    classification reads every seed as inside. Drawn to the wall itself, one seed in 40k on the DiSCo strands sat
    within rounding of it, read as outside, and was refused by the adaptive walk as a pool change."""
    import jax
    R = 0.7182e-6; off = 1e-3
    cl = [np.array([[-30e-6, 0, 0], [30e-6, 0, 0]]) + off, np.array([[-30e-6, 3 * R, 0], [30e-6, 3 * R, 0]]) + off]
    A = np.vstack([c[:-1] for c in cl]); B = np.vstack([c[1:] for c in cl]); r = np.array([R, R])
    grid = Grid(shape=(4, 1, 1), voxel_size_m=(15e-6, 10e-6, 10e-6), origin_m=(off - 22.5e-6, off + 1.5 * R, off))
    P, v, f, n = fill_swept_by_voxel(A, B, r, grid, np.full(grid.n_voxels, 50_000), seed=0)
    g = d.PackedCurvedCylinders(cl, r, interior=True)
    lab = np.asarray(g.classify_positions_exact(P.astype(np.float32)))
    assert len(P) == 200_000 and (lab > 0).all(), f"{(lab == 0).sum()} of {len(P)} seeds read as outside"
    d_wall, _ = g.wall_scales(P.astype(np.float32))
    assert d_wall.min() > 0.0                                                    # strictly inside, to float32
