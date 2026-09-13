"""The stratified intra draw is per voxel, not a box-wide rejection.

`fill_per_voxel` drew the pool's own volume-uniform sampler over the whole substrate and kept what landed in a
voxel that still wanted walkers: on the DiSCo strands 100 million draws for 123k seeds, longer than the walk.
`fill_swept_by_voxel` draws in each voxel from the segments whose tube meets it, by their clipped volume, and reads
the pool's volume fraction there from the clipped volume and the acceptance.
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
    """The volume fraction per voxel read from the clipped volume and the acceptance equals the census of the old
    box-wide volume-uniform draw, voxel by voxel, to the two draws' noise."""
    cls, rr = _strands()
    A = np.vstack([c[:-1] for c in cls]); B = np.vstack([c[1:] for c in cls]); r = np.repeat(rr, 2)
    grid = Grid(shape=(4, 4, 4), voxel_size_m=(5e-6,) * 3, origin_m=(-7.5e-6,) * 3)
    want = np.full(grid.n_voxels, 10)
    _, _, f_new, _ = fill_swept_by_voxel(A, B, r, grid, want, seed=0, census_draws=20_000)
    g = d.PackedCurvedCylinders(cls, rr, interior=True)
    lo, hi = np.full(3, -10e-6), np.full(3, 10e-6)
    def draw(n, rng):
        Q = g.sample_inside(n, rng); return Q, np.all((Q >= lo) & (Q <= hi), axis=1)
    def bin_index(Q):
        ijk, ins = grid.bin(Q)
        return np.where(ins, np.ravel_multi_index(tuple(np.clip(ijk, 0, 3).T), grid.shape), -1)
    _, _, _, trials, n_drawn = fill_per_voxel(draw, bin_index, grid.n_voxels, want, trials_max=10 ** 9, draws_max=2_000_000, seed=0)
    V_pool = float((np.pi * r ** 2 * np.linalg.norm(B - A, axis=1)).sum())
    f_old = trials / n_drawn * V_pool / float(np.prod(grid.voxel_size_m))
    m = trials > 2000
    assert m.sum() > 10
    np.testing.assert_allclose(f_new[m], f_old[m], rtol=0.1)                  # the old census's 2 % noise, and the new's 0.7 %
    np.testing.assert_allclose(f_new.sum(), f_old.sum(), rtol=0.02)
    # both count segment volumes (overlaps twice, no joint spheres), the strand family's convention; a membership
    # count of uniform points differs by up to 25 % in a voxel a joint or an overlap sits in
