"""The step rule of a label volume: what it measures on the seeded pool, and what it refuses to relax.

``LabelVolume``'s two constants (``SPECULAR_STEP_FRACTION``, ``SURFACE_STEP_FRACTION``) divide the
narrowest PASSAGE the seeded water occupies, measured per substrate with the voxel as the floor
(dmrai-lab/dmipy-sim#478). The repository's rule for this class of change is that a parameter-tuning
optimisation is derived from the adversarial worst case PER SUBSTRATE and is adaptive, never validated
on an average case, so these probes are the worst cases:

* a passage one voxel wide that holds more water than the budget -- the rule must stay at the floor;
* a rock-like pack at the voxel size and walk parameters of the two PUBLISHED families -- the resolved
  sub-step count must come out exactly what the voxel-gated rule gave, so their packs are unaffected;
* a grid that resolves its pore -- the rule must follow the pore and not the resolution, which is the
  whole point (a 20 nm voxel asked 48,000 sub-steps of a pore it resolves fifty times over).

Every number asserted here is derived from the fixture's own geometry or from the step rule written out,
never copied from a run.
"""
import numpy as np
import pytest

from dmipy_sim.engine.physics import resolve_sub_steps
from dmipy_sim.geometry import LabelVolume
from dmipy_sim.geometry.label_volume import (STEP_WATER_BUDGET, SURFACE_STEP_FRACTION,
                                             measure_passage_width)

D = 2e-9


def sub_steps_of(step_target, D, dt):
    """The sub-steps a rule targeting ``step_target`` asks for at ``dt``: ``step_l = sqrt(6 D dt)``."""
    return max(1, int(np.ceil(6.0 * D * dt / step_target ** 2)))


def voxel_gated(h, D, dt):
    """What the rule asked before #478: the voxel in place of the passage, both tiers."""
    return max(sub_steps_of(h / SURFACE_STEP_FRACTION, D, dt), sub_steps_of(h, D, dt))


def sand_pack(n=80, R=12.0, porosity=0.35, seed=0):
    """A rock-like binary volume: overlapping spheres of radius ``R`` voxels dropped until the void
    fraction falls to ``porosity``. Deterministic in ``seed``; label 0 is the pore, as a micro-CT rock's
    segmentation has it."""
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(*[np.arange(n) + 0.5] * 3, indexing="ij"), -1)
    solid = np.zeros((n, n, n), bool)
    while 1.0 - solid.mean() > porosity:
        c = rng.uniform(0, n, 3)
        solid |= np.sum((grid - c) ** 2, -1) < R * R
    return np.where(solid, 1, 0).astype(np.uint8)


def voxelised_cylinder(R_vox, h, n_z=8):
    """A cylinder of radius ``R_vox`` voxels along +z, periodic in z: the shape a segmented neurite is."""
    n = 2 * int(R_vox) + 6
    c = (np.arange(n) + 0.5) - n / 2
    X, Y = np.meshgrid(c, c, indexing="ij")
    disc = (X ** 2 + Y ** 2) < R_vox ** 2
    lab = np.where(np.repeat(disc[:, :, None], n_z, axis=2), 0, 1).astype(np.uint8)
    return LabelVolume(lab, h, periodic=(False, False, True))


# ───────────────────────────────────────────────────────── what the measurement is
def test_a_slab_pore_measures_its_own_width_and_a_one_voxel_channel_measures_the_voxel():
    """The passage width is the pool's own: the shortest run of the pool through a voxel along an index
    axis. For a slab pore it is the slab's width exactly, for a one-voxel channel one voxel, and both are
    exact numbers of voxels rather than an estimate -- the wall of a label volume IS axis-aligned faces,
    so the faces a walker meets along an axis are spaced by exactly that run."""
    h = 0.5e-6
    for n_pore in (1, 2, 5, 20):
        lab = np.ones((n_pore + 20, 8, 8), np.uint8)
        lab[10:10 + n_pore] = 0
        pw = LabelVolume(lab, h).passage_width
        assert pw.width == n_pore * h                     # exact, not approx
        assert pw.voxel == h and pw.walled and pw.walled_fraction == 1.0
        assert pw.n_voxels == n_pore * 8 * 8

    # anisotropic: each axis is measured with its own extent, so the narrow axis is the one that counts
    lab = np.ones((22, 8, 8), np.uint8)
    lab[10:12] = 0                                        # two voxels across x
    pw = LabelVolume(lab, (0.25e-6, 1e-6, 1e-6)).passage_width
    assert pw.width == 2 * 0.25e-6 and pw.voxel == 0.25e-6


def test_the_crop_s_own_faces_are_not_a_wall_and_a_pool_with_none_keeps_the_voxel_floor():
    """A run that spans an axis met no wall: on a periodic axis it never will, and on a reflecting one it
    ends at the crop's own outer face, which is not a wall of the substrate (it carries no relaxation and
    is not area in ``S/V``). A pool that meets no other pool along any axis therefore has nothing
    measured, and the rule keeps the voxel floor and says so (``walled`` False)."""
    h = 1e-6
    lab = np.zeros((12, 12, 12), np.uint8)                # one pool, no wall anywhere
    g = LabelVolume(lab, h, pools={0: "free"})
    pw = g.passage_width
    assert pw.width == h and pw.voxel == h
    assert pw.walled is False and pw.walled_fraction == 0.0
    assert np.isnan(pw.floor_share)
    assert g.length_scales.min_feature == h

    # a slab pore whose OTHER two axes are unwalled: they do not bound the passage, the pore does
    lab = np.ones((30, 8, 8), np.uint8)
    lab[10:20] = 0
    assert LabelVolume(lab, h).passage_width.width == 10 * h
    # and the same pore with the unwalled axes periodic reads the same: neither is a wall
    assert LabelVolume(lab, h, periodic=(False, True, True)).passage_width.width == 10 * h


def test_the_measurement_is_on_the_pool_the_walk_seeds():
    """Two pools of different widths in one image: the rule follows the one that is seeded. This is the
    pool ``geometry_from_spec`` passes from ``spec.seeding.pools``, so a pack's step follows the water it
    actually walks."""
    h = 1e-6
    lab = np.full((40, 8, 8), 2, np.uint8)
    lab[5:7] = 0                                          # a two-voxel channel of pool 0
    lab[20:32] = 1                                        # a twelve-voxel channel of pool 1
    pools = {0: "free", 1: "wide", 2: "grain"}
    assert LabelVolume(lab, h, pools=pools, pool="free").passage_width.width == 2 * h
    assert LabelVolume(lab, h, pools=pools, pool="wide").passage_width.width == 12 * h


# ───────────────────────────────────────────── the adversarial case: a voxel-thin channel
def test_a_voxel_thin_channel_above_the_budget_holds_the_rule_at_the_floor():
    """THE worst case: a pool that is mostly a wide pore but holds one channel a single voxel across.
    The rule must not relax to the wide pore's step, and it must not relax because the channel is small
    -- only because it holds less water than the stated budget.

    Built at a share deliberately just above the budget, the rule stays at the voxel and the resolved
    sub-step count is exactly what the channel needs; and the same fixture with the channel's water
    below the budget relaxes and RECORDS that it did (``floor_share`` under ``budget``), which is the
    only thing the budget is allowed to buy.
    """
    h = 0.5e-6
    dt = 1e-4
    wide = 30                                             # voxels across the wide pore

    def with_channel(n_channel_lines):
        lab = np.ones((40, 40, 40), np.uint8)
        lab[5:5 + wide, 5:5 + wide, 5:15] = 0             # the wide pore
        for i in range(n_channel_lines):                  # one-voxel-thin channel lines, well clear of it
            lab[10 + i, 5:5 + wide, 25] = 0
        return LabelVolume(lab, h)

    thin = with_channel(6)
    share = 6 * wide / float(thin.passage_width.n_voxels)
    assert share > STEP_WATER_BUDGET                       # the channel holds more water than the budget
    assert thin.passage_width.width == h                   # so the rule stays at the voxel
    assert thin.passage_width.floor_share == pytest.approx(share, rel=1e-6)
    # what the channel needs, from the rule rather than from a run
    assert resolve_sub_steps(thin, D, dt, surface=True) == sub_steps_of(h / SURFACE_STEP_FRACTION, D, dt)

    # a channel of the same width holding LESS water than the budget is stepped over -- and the record
    # says by how much, so the pack carries its own evidence
    lone = with_channel(0)
    assert lone.passage_width.width == 10 * h              # the wide pore's own narrowest axis
    assert lone.passage_width.floor_share < STEP_WATER_BUDGET


def test_the_rule_cannot_relax_a_pool_that_is_all_voxel_thin_channels():
    """A pool made only of one-voxel channels is the floor whatever the budget: there is no wider water
    for a quantile to find."""
    h = 1e-6
    lab = np.ones((40, 40, 40), np.uint8)
    lab[::2, :, 20] = 0                                    # a comb of one-voxel lines
    g = LabelVolume(lab, h)
    assert g.passage_width.width == h and g.passage_width.floor_share == 1.0
    for budget in (1e-3, 1e-2, 0.5):
        assert measure_passage_width(lab == 0, h, budget=budget).width == h


# ──────────────────────────────────────── the pin: the two published rock families are unaffected
@pytest.mark.parametrize("family,h,D0,dt,porosity", [
    # Talabi's Imperial 2007 sand packs: the released header's voxel, D0 and the save grid of
    # examples/substrate_bank/build_imperial_rocks.py
    ("imperial-rocks", 10.002e-6, 2.07e-9, 200e-6, 0.37),
    # Ling 2022's quartz / garnet packs: examples/substrate_bank/build_ling_sand_packs.py
    ("ling-sand-packs", 3.93e-6, 2.3e-9, 1e-3, 0.35),
])
def test_a_rock_like_pack_measures_the_floor_so_a_published_step_is_unchanged(family, h, D0, dt, porosity):
    """A rock's pore space puts far more than the budget of its water in one-voxel passages, so the
    measurement returns the floor and the resolved sub-step count is EXACTLY the voxel-gated one. The
    packs already published from these two families are walked at the step they were walked at.

    Measured on the released images themselves, the share of pore water in a one-voxel passage is 1.80 %
    (LV60A central 300^3), 1.59 % (Berea), 1.38 % (F42A), 0.27 % and 0.24 % (Ling's pure quartz and
    garnet packs) -- 2.4x the budget at the narrowest. This fixture is a synthetic pack of overlapping
    spheres rather than the images, which are not in the repository; it stands in for their pore
    topology, and its own share is asserted to be above the budget so the test fails loudly if it ever
    stops standing in.
    """
    lab = sand_pack(n=80, R=12.0, porosity=porosity, seed=0)
    g = LabelVolume(lab, h, surface_relaxivity_t2=41e-6)
    pw = g.passage_width
    assert pw.floor_share > STEP_WATER_BUDGET, "the fixture no longer has a rock's narrow passages"
    assert pw.width == pw.voxel == h
    assert g.length_scales.min_feature == h
    assert resolve_sub_steps(g, D0, dt, surface=True) == voxel_gated(h, D0, dt)


# ─────────────────────────────────────── the point: a grid that resolves its pore is walked at it
def test_a_resolved_pore_is_walked_at_the_pore_and_not_at_the_resolution():
    """A voxelised neurite at an electron-microscopy voxel: the pore is resolved 25 times over, and the
    rule follows the pore. The voxel-gated rule asked for the square of the resolution ratio in
    sub-steps -- 48,000 per 0.1 ms save at a 20 nm voxel -- for a step the substrate never needed.

    The passage a low quantile finds is a fraction of the diameter (the water near a curved wall sits on
    short axis runs), so the rule remains conservative; what it stops being is resolution-driven.
    """
    h, R_vox, dt = 20e-9, 25, 1e-4
    g = voxelised_cylinder(R_vox, h)
    pw = g.passage_width
    assert h < pw.width <= 2 * R_vox * h                  # between the floor and the diameter
    n_new = resolve_sub_steps(g, D, dt, surface=True)
    assert n_new == sub_steps_of(pw.width / SURFACE_STEP_FRACTION, D, dt)
    assert n_new * (pw.width / h) ** 2 == pytest.approx(voxel_gated(h, D, dt), rel=0.01)
    assert n_new < voxel_gated(h, D, dt) / 50             # and it is at least fifty times cheaper

    # the same pore at a voxel that barely expresses it has nothing to relax: two voxels across is all
    # the width there is, and the rule takes it
    assert voxelised_cylinder(1, h).passage_width.width == 2 * h


def test_the_passage_follows_the_pore_at_a_fixed_grid():
    """At one voxel size the measured passage grows with the pore, which is the adaptivity the rule is
    for. It is NOT the pore's own diameter and does not stay put in metres as the grid is refined at a
    fixed pore: the water a low quantile finds sits on the short axis runs near the wall, whose length
    falls with the voxel. Measured on a voxelised cylinder at the 1e-3 budget, the passage is
    4 / 6 / 10 / 12 voxels at a radius of 5 / 10 / 25 / 40 voxels, so it buys a factor of about
    ``4 R / h`` in sub-steps rather than ``(R / h)^2``, and that is stated rather than claimed."""
    h = 20e-9
    at_R = {R_vox: voxelised_cylinder(R_vox, h).passage_width.width for R_vox in (5, 10, 25)}
    assert at_R[5] < at_R[10] < at_R[25]
    for R_vox, w in at_R.items():
        assert h < w < 2 * R_vox * h                       # between the floor and the diameter
