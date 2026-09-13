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


def _clip_segments_to_voxels(A, B, r, grid):
    """Every (segment, voxel) pair whose tube can meet the voxel, with the centerline's parameter interval inside
    the voxel grown by the radius: ``(seg, vox, t0, t1)``. A segment's bounding box grown by ``r`` gives the voxel
    range; Liang-Barsky against each of those voxels grown by ``r`` gives the interval."""
    corner = np.asarray(grid.corner_m, float); vs = np.asarray(grid.voxel_size_m, float); sh = np.asarray(grid.shape)
    lo = np.minimum(A, B) - r[:, None]; hi = np.maximum(A, B) + r[:, None]
    ilo = np.clip(np.floor((lo - corner) / vs).astype(np.int64), 0, sh - 1)
    ihi = np.clip(np.floor((hi - corner) / vs).astype(np.int64), 0, sh - 1)
    span = ihi - ilo + 1; n_pairs = np.prod(span, axis=1)
    keep = np.all(hi >= corner, axis=1) & np.all(lo <= corner + sh * vs, axis=1)
    seg = np.repeat(np.arange(len(A)), np.where(keep, n_pairs, 0))
    # the voxel offsets within each segment's range, enumerated
    off = np.concatenate([np.stack(np.meshgrid(*[np.arange(s) for s in span[k]], indexing="ij"), -1).reshape(-1, 3)
                          for k in np.flatnonzero(keep)]) if keep.any() else np.zeros((0, 3), np.int64)
    ijk = ilo[seg] + off
    vlo = corner + ijk * vs - r[seg][:, None]; vhi = corner + (ijk + 1) * vs + r[seg][:, None]
    D = B[seg] - A[seg]; A_ = A[seg]
    t0 = np.zeros(len(seg)); t1 = np.ones(len(seg))
    for ax in range(3):
        d = D[:, ax]; a = A_[:, ax]
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (vlo[:, ax] - a) / d; tb = (vhi[:, ax] - a) / d
        par = np.abs(d) < 1e-30
        tin = np.where(par, -np.inf, np.minimum(ta, tb)); tout = np.where(par, np.inf, np.maximum(ta, tb))
        outside = par & ((a < vlo[:, ax]) | (a > vhi[:, ax]))
        t0 = np.where(outside, 1.0, np.maximum(t0, tin)); t1 = np.where(outside, 0.0, np.minimum(t1, tout))
    ok = t1 > t0
    vox = np.ravel_multi_index(tuple(ijk[ok].T), grid.shape)
    return seg[ok], vox, t0[ok], t1[ok]


def fill_swept_by_voxel(A, B, r, grid, want, *, seed=0, census_draws=200, rounds_max=20):
    """``want[v]`` points uniform inside the swept polylines (segments ``A -> B`` of radius ``r``) AND inside voxel
    ``v``, for every voxel of ``grid``, drawn per voxel: a segment whose tube meets the voxel, by its clipped
    volume, a point uniform in that clipped piece, kept when it lies in the voxel (the disc pokes out of it near
    a face; nine in ten are kept). The pool's per-voxel volume fraction ``f`` is the clipped volume times the
    acceptance over the voxel's volume, the unbiased estimate of ``tube AND voxel``, read on at least
    ``census_draws`` draws per voxel (2 % at 200). Returns ``(points, voxel, f, n_drawn)``; a voxel no tube meets
    gets nothing, and ``f = 0`` there; a sliver a tube barely touches (a few draws in a thousand land in the
    voxel) may be left short after ``rounds_max`` rounds. Overlapping tubes count twice, as everywhere in the
    strand family."""
    rng = np.random.default_rng(seed)
    A = np.asarray(A, float); B = np.asarray(B, float); r = np.asarray(r, float)
    want = np.asarray(want, np.int64).reshape(-1)
    seg, vox, t0, t1 = _clip_segments_to_voxels(A, B, r, grid)
    L = np.linalg.norm(B - A, axis=1)
    w_pair = np.pi * r[seg] ** 2 * L[seg] * (t1 - t0)                 # the clipped piece's volume
    W_vox = np.bincount(vox, weights=w_pair, minlength=grid.n_voxels)
    order = np.argsort(vox, kind="stable"); seg, vox, t0, t1, w_pair = seg[order], vox[order], t0[order], t1[order], w_pair[order]
    starts = np.searchsorted(vox, np.arange(grid.n_voxels + 1))
    have = np.zeros(grid.n_voxels, np.int64); drawn = np.zeros(grid.n_voxels, np.int64); acc = np.zeros(grid.n_voxels, np.int64)
    kept_p, kept_v = [], []
    V_vox = float(np.prod(grid.voxel_size_m))
    corner = np.asarray(grid.corner_m, float); vs = np.asarray(grid.voxel_size_m, float)
    active = (want > 0) & (W_vox > 0)
    for _ in range(int(rounds_max)):
        short = active & ((have < want) | (drawn < int(census_draws)))
        if not short.any():
            break
        vids = np.flatnonzero(short)
        rate = np.where(drawn[vids] > 0, acc[vids] / np.maximum(drawn[vids], 1), 0.9)
        n_try = np.ceil(1.3 * (want[vids] - have[vids]) / np.maximum(rate, 0.05)).astype(np.int64) + 4
        n_try = np.maximum(n_try, int(census_draws) - drawn[vids])
        v_all = np.repeat(vids, n_try)
        # a pair within the voxel by its clipped volume: inverse-cdf on the voxel's own run of pairs
        u = rng.uniform(0.0, 1.0, len(v_all))
        cum = np.cumsum(w_pair); base = np.where(starts[v_all] > 0, cum[starts[v_all] - 1], 0.0)
        target = base + u * W_vox[v_all]
        p = np.clip(np.searchsorted(cum, target, side="right"), starts[v_all], starts[v_all + 1] - 1)
        k = seg[p]; t = t0[p] + (t1[p] - t0[p]) * rng.uniform(0.0, 1.0, len(p))
        C = A[k] + (B[k] - A[k]) * t[:, None]
        T = (B[k] - A[k]) / np.maximum(L[k], 1e-30)[:, None]
        ref = np.tile([0.0, 0.0, 1.0], (len(k), 1)); ref[np.abs((T * ref).sum(1)) > 0.9] = [1.0, 0.0, 0.0]
        e1 = np.cross(T, ref); e1 /= np.linalg.norm(e1, axis=1, keepdims=True); e2 = np.cross(T, e1)
        rad = r[k] * np.sqrt(rng.uniform(0.0, 1.0, len(k))); th = rng.uniform(0.0, 2 * np.pi, len(k))
        P = C + rad[:, None] * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
        ijk = np.floor((P - corner) / vs).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < np.asarray(grid.shape)), axis=1)
        v_hit = np.where(inside, np.ravel_multi_index(tuple(np.clip(ijk, 0, np.asarray(grid.shape) - 1).T), grid.shape), -1)
        ok = v_hit == v_all
        drawn += np.bincount(v_all, minlength=grid.n_voxels); acc += np.bincount(v_all[ok], minlength=grid.n_voxels)
        Pk, vk = P[ok], v_all[ok]
        o = np.argsort(vk, kind="stable"); vk, Pk = vk[o], Pk[o]
        st = np.searchsorted(vk, np.arange(grid.n_voxels))
        rank = np.arange(len(vk)) - st[vk] + have[vk]
        take = rank < want[vk]
        kept_p.append(Pk[take]); kept_v.append(vk[take]); have += np.bincount(vk[take], minlength=grid.n_voxels)
    f = np.where(drawn > 0, W_vox * acc / np.maximum(drawn, 1) / V_vox, 0.0)
    P = np.concatenate(kept_p) if kept_p else np.zeros((0, 3)); v = np.concatenate(kept_v) if kept_v else np.zeros(0, np.int64)
    return P, v, f, int(drawn.sum())


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
