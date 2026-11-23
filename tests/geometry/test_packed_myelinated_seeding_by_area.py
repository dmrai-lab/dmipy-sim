"""A packed myelinated cell seeds its pools by area, for any number of cylinders.

Every walker's start is an independent draw of the area law over the seeded pools: the pool counts of a placement block
are one multinomial draw, each start is uniform in its pool, and the block's rows are put in a uniformly random order.
Rounding each pool's expected count to a whole number per block gives every pool below half a walker a block none, and
the whole remainder to the outside; the cell below (2,000 equal small lumens, 0.1 walkers a block each) is that case.

The cell: 2,000 axons of outer radius 0.1 um and g 0.7 on a 0.5 um lattice in a 25 um periodic cell, lumens and the
outside seeded, the sheaths not. Labels: outside 0, lumen k is k + 1, sheath k is 2000 + k + 1. Float64 areas give the
outside share ``P_OUT`` and each lumen's ``P_LUMEN``. The placement is 256 blocks of ``PLACEMENT_BLOCK`` walkers.
"""
import jax
import numpy as np
import pytest

from dmipy_sim import Compartments, Pool
from dmipy_sim.engine.physics import seed_walkers
from dmipy_sim.geometry import PackedMyelinatedCylinders
from dmipy_sim.geometry.base import PLACEMENT_BLOCK, placement

N_AXONS, R_OUTER, G, L, BLOCKS = 2000, 0.1e-6, 0.7, 25e-6, 256
_LUMEN_AREA = np.pi * (R_OUTER * G) ** 2
_OUTSIDE_AREA = L * L - N_AXONS * np.pi * R_OUTER ** 2
P_OUT = _OUTSIDE_AREA / (_OUTSIDE_AREA + N_AXONS * _LUMEN_AREA)
P_LUMEN = _LUMEN_AREA / (_OUTSIDE_AREA + N_AXONS * _LUMEN_AREA)

LUMEN_COUNT_BOUNDS = (6, 57)
"""Two-sided Bonferroni bounds on one lumen's count over all walkers: the ``1e-3 / 4000`` and ``1 - 1e-3 / 4000``
quantiles of Binomial(256 * 4096, ``P_LUMEN``), mean 27.2. A union bound over the 2,000 lumens holds whatever their
dependence, at family level 1e-3."""

DISPERSION_BOUNDS = (188.0, 337.1)
"""The two-sided 1e-3 interval of a chi-squared variable of 256 degrees of freedom."""


def _cell(n=N_AXONS):
    centres = np.stack(np.meshgrid(np.arange(40), np.arange(50), indexing="ij"), -1).reshape(-1, 2) * 0.5e-6 \
        - [9.75e-6, 12.25e-6]
    return PackedMyelinatedCylinders(
        np.full(n, R_OUTER * G), G, centres[:n], L, N_max=n,
        compartments=Compartments(intra=Pool(D=2e-9), extra=Pool(D=2e-9), myelin=Pool(D=0.0)),
        seed_pools=("extra", "intra"))


@pytest.fixture(scope="module")
def cell():
    return _cell()


@pytest.fixture(scope="module")
def labels(cell):
    return np.asarray(placement(cell, BLOCKS * PLACEMENT_BLOCK, jax.random.PRNGKey(7))[1])


def test_the_outside_share_is_the_area_share(labels):
    """The outside share is within 4 binomial sd of its area share (the rounded rule gives 1.0)."""
    n = labels.size
    z = (np.mean(labels == 0) - P_OUT) / np.sqrt(P_OUT * (1 - P_OUT) / n)
    assert abs(z) <= 4.0, z


def test_every_lumen_is_seeded_within_its_binomial_bounds(labels):
    """Each of the 2,000 lumens, each holding 0.1 walkers a block in expectation, holds a count within the family bounds."""
    counts = np.bincount(labels, minlength=2 * N_AXONS + 1)[1:N_AXONS + 1]
    assert counts.min() >= LUMEN_COUNT_BOUNDS[0] and counts.max() <= LUMEN_COUNT_BOUNDS[1], (counts.min(), counts.max())


def test_blocks_are_independent_draws_not_fixed_counts(labels):
    """The per-block outside counts are as dispersed as independent draws: a fixed-count rule (rounding, systematic
    rounding) gives a dispersion near 0."""
    n_out = (labels.reshape(BLOCKS, PLACEMENT_BLOCK) == 0).sum(1)
    x = float(np.sum((n_out - PLACEMENT_BLOCK * P_OUT) ** 2) / (PLACEMENT_BLOCK * P_OUT * (1 - P_OUT)))
    assert DISPERSION_BOUNDS[0] <= x <= DISPERSION_BOUNDS[1], x


def test_no_walker_starts_in_an_unseeded_sheath(labels):
    assert not np.any(labels > N_AXONS)


def test_a_walker_is_its_seed_and_index():
    """Starts and labels of any range are the whole's rows, across placement blocks (n = 2 blocks + 700)."""
    g = _cell()
    n = 2 * PLACEMENT_BLOCK + 700
    key = jax.random.PRNGKey(11)
    r0, lab = (np.asarray(a) for a in placement(g, n, key))
    for a, b in ((0, 1), (5, 300), (4093, 4106), (3096, 8197), (1000, 8892)):
        r, lb = placement(g, b - a, key, first_walker=a)
        np.testing.assert_array_equal(np.asarray(r), r0[a:b])
        np.testing.assert_array_equal(np.asarray(lb), lab[a:b])
    _, r_whole, keys_whole, lab_whole = seed_walkers(g, 30 + 4081, 11)
    _, r, keys, lb = seed_walkers(g, 30, 11, first_walker=4081)
    np.testing.assert_array_equal(np.asarray(r), np.asarray(r_whole)[4081:])
    np.testing.assert_array_equal(np.asarray(keys), np.asarray(keys_whole)[4081:])
    np.testing.assert_array_equal(np.asarray(lb), np.asarray(lab_whole)[4081:])


def test_a_small_cell_seeds_its_class_shares_by_area():
    """A 150-axon cell over 16 blocks (65,536 walkers): the outside and the lumen shares are within 4 binomial sd."""
    g = _cell(150)
    n = 16 * PLACEMENT_BLOCK
    lab = np.asarray(placement(g, n, jax.random.PRNGKey(3))[1])
    area_l = np.pi * (R_OUTER * G) ** 2
    out = L * L - 150 * np.pi * R_OUTER ** 2
    for share, expected in ((np.mean(lab == 0), out / (out + 150 * area_l)),
                            (np.mean((lab >= 1) & (lab <= 150)), 150 * area_l / (out + 150 * area_l))):
        assert abs(share - expected) <= 4 * np.sqrt(expected * (1 - expected) / n), (share, expected)
    assert not np.any(lab > 150)
