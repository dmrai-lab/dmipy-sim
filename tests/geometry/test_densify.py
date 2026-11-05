"""The densification path (`dmipy_sim.geometry.densify`): one compression of a periodic packing from a sparse
random-sequential-addition start, snapshotted at packing fractions.

Where the path jams is held to the literature's jamming points of the same protocol (soft-particle compression and
energy minimisation, O'Hern et al. 2002 and 2003, Xu et al. 2005, Desmond & Weeks 2009 and 2014); every snapshot holds
its gap in float64 and in the float32 the walk keeps; the snapshots of a path are the same objects ever more
compressed and do not depend on which others were asked for; 10^4 objects run within a laptop's minutes."""
import time

import numpy as np
import pytest
from scipy.special import gammainc, gammaincinv

from dmipy_sim.geometry.densify import DENSIFY_LIMIT, DENSIFY_START, densify_packing, jamming_fraction
from dmipy_sim.geometry.packing import periodic_min_gap

R = 1e-6
GAP = 1e-3 * R


def _quantile_radii(cv, n, dim, seed=0, water=0.01):
    """The Gamma(CV) law by quantile, truncated at its 1 % water floor (computed here, independently of the producer:
    the floor of the r^dim-weighted law Gamma(k + dim))."""
    u = np.random.default_rng((seed, 1)).random(n)
    if cv == 0:
        return np.full(n, R)
    k = 1.0 / cv ** 2
    p_floor = gammainc(k, gammaincinv(k + dim, water))
    return R * gammaincinv(k, p_floor + u * (1.0 - p_floor)) / k


def _bidisperse(n):
    """O'Hern et al.'s system: a 50:50 mixture of diameters 1 and 1.4."""
    return np.where(np.arange(n) % 2 == 0, 1.0, 1.4) * R


def _fraction(radii, L, dim):
    c = np.pi if dim == 2 else 4.0 * np.pi / 3.0
    return float(c * np.sum(radii ** dim) / L ** dim)


# ── where the path jams ───────────────────────────────────────────────────────────────────────────────────────────
# phi* in the thermodynamic limit: 0.842 for the bidisperse disks and 0.648 for the bidisperse spheres (O'Hern, Langer,
# Liu & Nagel, Phys. Rev. Lett. 88, 075507, 2002: "We find phi* = 0.842 in 2D and phi* = 0.648 in 3D"), 0.639 +- 0.001
# for monodisperse spheres (O'Hern, Silbert, Liu & Nagel, Phys. Rev. E 68, 011306, 2003, Eq. 5). A finite cell jams
# lower by delta_0 L^(-1/nu), delta_0 = 0.12, nu = 0.71, L = N^(1/d) (ibid., Eq. 4) -- 0.006 for 256 disks and 0.010
# for 216 spheres -- and one realisation scatters about its peak by a width that narrows as N^-0.55 (2002); the
# tolerance is that shift plus 0.015.
LITERATURE = [
    ("bidisperse disks", 2, _bidisperse, 256, 0.842),
    ("monodisperse spheres", 3, lambda n: np.full(n, R), 216, 0.639),
]


@pytest.mark.parametrize("name,dim,radii_of,n,phi_star", LITERATURE, ids=[c[0] for c in LITERATURE])
def test_the_path_jams_at_the_literatures_jamming_point(name, dim, radii_of, n, phi_star):
    shift = 0.12 * n ** (-1.0 / (0.71 * dim))
    phi_j = jamming_fraction(radii_of(n), dim, seed=0, min_gap=1e-6 * R)
    assert abs(phi_j - (phi_star - shift)) <= 0.015, (name, phi_j, phi_star, shift)


def test_spheres_jam_where_desmond_and_weeks_put_the_polydisperse_jamming_point():
    """Desmond & Weeks (Phys. Rev. E 90, 022204, 2014), fitted over polydispersity delta <= 0.4 with the same
    protocol: phi_RCP = 0.634 + 0.0658 delta + 0.0857 S delta^2 (Eq. 4), delta and S the radii's coefficient of
    variation and skewness. The Gamma law at CV 0.45 truncated at its water floor has delta 0.36 and S 1.2: 0.672."""
    r = _quantile_radii(0.45, 216, 3)
    delta = r.std() / r.mean()
    skew = float(np.mean((r - r.mean()) ** 3) / r.var() ** 1.5)
    predicted = 0.634 + 0.0658 * delta + 0.0857 * skew * delta ** 2
    shift = 0.12 * 216 ** (-1.0 / (0.71 * 3))
    phi_j = jamming_fraction(r, 3, seed=0, min_gap=1e-6 * R)
    assert abs(phi_j - (predicted - shift)) <= 0.015, (phi_j, predicted, shift)


# ── no overlap, to rounding ───────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dim,cv,fractions", [(2, 0.65, (0.3, 0.8)), (3, 0.45, (0.2, 0.6))])
def test_every_snapshot_holds_its_gap_in_float64_and_in_the_walks_float32(dim, cv, fractions):
    r = _quantile_radii(cv, 300, dim)
    for (centers, L), f in zip(densify_packing(r, fractions, dim, seed=1, min_gap=GAP), fractions):
        assert _fraction(r, L, dim) == pytest.approx(f, rel=1e-12)
        assert np.all(np.abs(centers) <= L / 2)
        assert periodic_min_gap(centers, r, L) >= GAP
        c32 = centers.astype(np.float32).astype(np.float64)       # what PackedCylinders / PackedSpheres hold
        r32 = r.astype(np.float32).astype(np.float64)
        assert periodic_min_gap(c32, r32, float(np.float32(L))) > 0.5 * GAP


# ── one path: the same objects ever more compressed ───────────────────────────────────────────────────────────────
def test_a_snapshot_does_not_depend_on_which_others_were_asked_for():
    r = _quantile_radii(0.25, 200, 2)
    grid = densify_packing(r, (0.2, 0.5, 0.7), 2, seed=3, min_gap=GAP)
    alone = densify_packing(r, (0.5,), 2, seed=3, min_gap=GAP)
    branched = densify_packing(r, (0.7, 0.3337, 0.5), 2, seed=3, min_gap=GAP)    # 0.3337 is between two stops
    assert np.array_equal(alone[0][0], grid[1][0]) and alone[0][1] == grid[1][1]
    assert np.array_equal(branched[0][0], grid[2][0]) and np.array_equal(branched[2][0], grid[1][0])
    assert _fraction(r, branched[1][1], 2) == pytest.approx(0.3337, rel=1e-12)


def test_consecutive_snapshots_are_small_compressions_of_one_another():
    """Between grid nodes 0.1 apart an object moves a fraction of a radius (in the cell's own coordinates, which the
    affine compression leaves fixed), where the same index in an independent path sits a quarter of the cell away."""
    r = _quantile_radii(0.45, 300, 3)
    fractions = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    path = densify_packing(r, fractions, 3, seed=2, min_gap=GAP)
    rbar = r.mean()
    for (c0, L0), (c1, L1) in zip(path[:-1], path[1:]):
        ds = c1 / L1 - c0 / L0
        ds -= np.round(ds)
        moved = np.sqrt(np.sum(ds * ds, axis=1)) * L1 / rbar
        assert np.median(moved) < 0.75 and moved.max() < 3.0, (np.median(moved), moved.max())
    other = densify_packing(r, (0.6,), 3, seed=5, min_gap=GAP)[0]
    ds = other[0] / other[1] - path[-1][0] / path[-1][1]
    ds -= np.round(ds)
    assert np.median(np.sqrt(np.sum(ds * ds, axis=1))) > 0.25


def test_a_path_is_a_deterministic_function_of_its_seed():
    r = _quantile_radii(0.45, 150, 2)
    a = densify_packing(r, (0.6,), 2, seed=7, min_gap=GAP)[0]
    b = densify_packing(r, (0.6,), 2, seed=7, min_gap=GAP)[0]
    c = densify_packing(r, (0.6,), 2, seed=8, min_gap=GAP)[0]
    assert np.array_equal(a[0], b[0]) and not np.array_equal(a[0], c[0])


# ── refusals, before anything is placed ───────────────────────────────────────────────────────────────────────────
def test_a_fraction_beyond_the_reachable_limit_or_below_the_start_is_refused_by_name():
    r = _quantile_radii(0.25, 5000, 2)
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match="DENSIFY_LIMIT"):
        densify_packing(r, (0.5, DENSIFY_LIMIT[2] + 0.005), 2, seed=0, min_gap=GAP)
    with pytest.raises(ValueError, match="DENSIFY_LIMIT"):         # the gap's contact radii count
        densify_packing(r, (DENSIFY_LIMIT[2] - 0.01,), 2, seed=0, min_gap=0.05 * R)
    with pytest.raises(ValueError, match="DENSIFY_START"):
        densify_packing(r, (DENSIFY_START / 2,), 2, seed=0, min_gap=GAP)
    with pytest.raises(ValueError, match="min_gap must be positive"):
        densify_packing(r, (0.5,), 2, seed=0, min_gap=0.0)
    with pytest.raises(ValueError, match="FLOAT32_GAP"):          # 5,000 disks at 0.1: a cell of ~410 radii
        densify_packing(r, (0.1, 0.5), 2, seed=0, min_gap=1e-5 * R)
    assert time.perf_counter() - t0 < 1.0


# ── 10^4 objects, a packed family's size, on a laptop ──────────────────────────────────────────────────────────────
def test_ten_thousand_cylinders_densify_to_the_grids_top_within_laptop_minutes():
    """One path of 10^4 cylinders (Gamma CV 0.45 by quantile) from 0.05 to the grid's 0.8, every node snapshotted:
    56 s on one laptop core. Through ``packed_request`` with every grid node a spec, cylinders and spheres at CV 0 /
    0.25 / 0.45 / 0.65 took 45-89 s each beside other jobs, at a peak resident set of 0.3-0.45 GB."""
    r = _quantile_radii(0.45, 10_000, 2)
    fractions = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    t0 = time.perf_counter()
    path = densify_packing(r, fractions, 2, seed=0, min_gap=GAP)
    seconds = time.perf_counter() - t0
    assert seconds < 600.0, seconds
    for (centers, L), f in zip(path, fractions):
        assert _fraction(r, L, 2) == pytest.approx(f, rel=1e-12)
    assert periodic_min_gap(path[-1][0], r, path[-1][1]) >= GAP
