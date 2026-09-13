"""How a walk is seeded when the pack is meant to be cut into voxels (``Phantom.partition``): the same number
of walkers in every occupied voxel of a grid, per pool, with the weight ``f_pool,v / n_pool,v`` that keeps a
weighted per-voxel mean volume-correct. Uniform seeding gives Poisson counts per voxel -- a 5e5-walker DiSCo walk
puts ~8 walkers in each of 64,000 voxels -- so the per-voxel floor, the quantity a partition is certified on,
is set by the emptiest voxel; stratification puts the walkers where the certificate needs them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..phantom.grid import Grid


@dataclass(frozen=True)
class StratifiedByVoxel:
    """Seed ``walkers_per_voxel`` walkers of each seeded pool in every voxel of ``grid`` the pool occupies.

    ``walkers_per_voxel``: one count for every pool, or ``{pool name: count}``; a count is an int or an array of
    the grid's shape (a per-voxel budget, what :func:`plan_seeding` returns from a pilot's per-voxel floor).
    ``trials_per_voxel_max`` bounds the rejection sampling per voxel: a voxel holding a sliver of a pool is
    left with what it got, its walkers weighted by the sliver's measured volume fraction, rather than drawn
    for ever. The grid is in substrate coordinates (``attach="substrate"``).
    """
    grid: Grid
    walkers_per_voxel: object
    trials_per_voxel_max: int = 200_000

    def __post_init__(self):
        if not isinstance(self.grid, Grid):
            raise TypeError("grid must be a dmipy_sim.phantom.Grid")
        if self.grid.attach != "substrate":
            raise ValueError("a seeding grid is welded to the substrate: Grid(..., attach='substrate')")
        if int(self.trials_per_voxel_max) < 1:
            raise ValueError("trials_per_voxel_max must be at least 1")

    def count_for(self, pool_name):
        """``(n_voxels,)`` int: the walkers wanted per voxel of the grid for this pool (flattened C order)."""
        w = self.walkers_per_voxel
        if isinstance(w, dict):
            if pool_name not in w:
                raise ValueError(f"walkers_per_voxel names no count for pool {pool_name!r}: {sorted(w)}")
            w = w[pool_name]
        arr = np.asarray(w)
        if arr.ndim == 0:
            n = int(arr)
            if n < 1:
                raise ValueError("walkers_per_voxel must be at least 1")
            return np.full(self.grid.n_voxels, n, np.int64)
        arr = self.grid.check_volume(arr, "walkers_per_voxel")
        if (arr < 0).any():
            raise ValueError("a per-voxel walker count cannot be negative")
        return np.rint(arr).astype(np.int64).reshape(-1)

    def to_dict(self):
        w = self.walkers_per_voxel
        if isinstance(w, dict):
            w = {k: (int(v) if np.ndim(v) == 0 else np.asarray(v).tolist()) for k, v in w.items()}
        else:
            w = int(w) if np.ndim(w) == 0 else np.asarray(w).tolist()
        return dict(rule="stratified_by_voxel", grid=self.grid.to_meta(), walkers_per_voxel=w,
                    trials_per_voxel_max=int(self.trials_per_voxel_max))


def fill_per_voxel(draw, bin_index, n_voxels, want, *, trials_max, draws_max=None, batch=1_000_000, seed=0):
    """Draw points with ``draw(n, rng) -> (points (n, 3), accepted (n,) bool)`` -- the pool's own sampler, which
    may reject -- until every voxel with any accepted draw holds ``want[v]`` points or has spent ``trials_max``
    draws, or ``draws_max`` points have been drawn in all (default 200 x the points wanted: a sliver of a pool
    that a volume-uniform draw reaches once in a million is left with what it got). Returns ``(points, voxel, f, trials)``: the kept points and their voxels, the pool's volume fraction
    per voxel measured from the draws (accepted / drawn, the census), and the draws per voxel."""
    rng = np.random.default_rng(seed)
    want = np.asarray(want, np.int64)
    have = np.zeros(n_voxels, np.int64); trials = np.zeros(n_voxels, np.int64); acc = np.zeros(n_voxels, np.int64)
    kept_p, kept_v = [], []; n_drawn = 0
    draws_max = int(draws_max) if draws_max is not None else 200 * int(want.sum())
    log = logging.getLogger("dmipy_sim")
    while True:
        P, ok = draw(int(batch), rng); n_drawn += len(P)
        v = bin_index(P)
        inside = v >= 0
        trials += np.bincount(v[inside], minlength=n_voxels)
        ok = ok & inside
        P, v = P[ok], v[ok]
        acc += np.bincount(v, minlength=n_voxels)
        order = np.argsort(v, kind="stable"); v = v[order]; P = P[order]
        starts = np.searchsorted(v, np.arange(n_voxels))
        rank = np.arange(len(v)) - starts[v] + have[v]
        take = rank < want[v]
        kept_p.append(P[take]); kept_v.append(v[take])
        have += np.bincount(v[take], minlength=n_voxels)
        short = (acc > 0) & (have < want) & (trials < trials_max)
        log.info("fill_per_voxel: %d drawn, %d voxels occupied, %d short, %d kept", n_drawn, int((acc > 0).sum()),
                 int(short.sum()), int(have.sum()))
        if not short.any() or n_drawn >= draws_max:
            break
    P = np.concatenate(kept_p) if kept_p else np.zeros((0, 3)); v = np.concatenate(kept_v) if kept_v else np.zeros(0, np.int64)
    f = np.where(trials > 0, acc / np.maximum(trials, 1), 0.0)
    return P, v, f, trials, n_drawn


def plan_seeding(per_voxel_floor, walkers_per_voxel_pilot, *, target_floor, grid, minimum=2):
    """The per-voxel walker counts that bring a pilot's per-voxel Monte-Carlo floor down to ``target_floor``:
    ``n = n_pilot (floor / target)^2`` (the floor falls as one over the square root of the count), at least
    ``minimum`` where the pilot saw the pool. Inputs are dicts ``{pool name: array of the grid's shape}``;
    returns a :class:`StratifiedByVoxel` on ``grid``."""
    counts = {}
    for name, fl in per_voxel_floor.items():
        fl = grid.check_volume(fl, f"per_voxel_floor[{name!r}]")
        n0 = grid.check_volume(walkers_per_voxel_pilot[name], f"walkers_per_voxel_pilot[{name!r}]")
        seen = n0 > 0
        n = np.zeros(grid.shape)
        n[seen] = np.maximum(minimum, np.ceil(n0[seen] * (fl[seen] / float(target_floor)) ** 2))
        counts[name] = n
    return StratifiedByVoxel(grid=grid, walkers_per_voxel=counts)
