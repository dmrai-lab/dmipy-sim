"""The step rule of a label volume: what it measures on the seeded pool, and what it refuses to relax.

``LabelVolume``'s two constants (``SPECULAR_STEP_FRACTION``, ``SURFACE_STEP_FRACTION``) divide the
narrowest PASSAGE the seeded water occupies, measured per substrate with the voxel as the floor
(dmrai-lab/dmipy-sim#478). The repository's rule for this class of change is that a parameter-tuning
optimisation is derived from the adversarial worst case PER SUBSTRATE and is adaptive, never validated
on an average case, so these probes are the worst cases:

* a passage one voxel wide that holds more water than the budget -- the rule must stay at the floor,
  and a set holding EXACTLY the budget counts as held;
* a set of one-voxel fissures holding a quarter of the budget of the WATER and ten times it of the wall
  AREA -- the rule must stay at the floor, because what a label-volume pack certifies at the wall is
  area-weighted;
* a PERMEABLE wall, whose estimator is one Powles decision per step and so counts faces per step rather
  than dividing a length -- the rule must keep the voxel there, and the crossing rate it gets must be
  the analytic one where the relaxed step's is not;
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
                                             measure_passage_width, step_width)

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
    assert np.isnan(pw.floor_water_share) and np.isnan(pw.floor_area_share)
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
    below the budget relaxes and RECORDS what it stepped over (``water_share``, ``area_share``), which is
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
    assert thin.passage_width.floor_water_share == pytest.approx(share, rel=1e-6)
    # what the channel needs, from the rule rather than from a run
    assert resolve_sub_steps(thin, D, dt, surface=True) == sub_steps_of(h / SURFACE_STEP_FRACTION, D, dt)

    # a channel of the same width holding LESS water than the budget is stepped over -- and the record
    # says by how much, so the pack carries its own evidence
    lone = with_channel(0)
    assert lone.passage_width.width == 10 * h              # the wide pore's own narrowest axis
    assert lone.passage_width.floor_water_share < STEP_WATER_BUDGET
    assert lone.passage_width.water_share < STEP_WATER_BUDGET
    assert lone.passage_width.area_share < STEP_WATER_BUDGET


def test_the_rule_cannot_relax_a_pool_that_is_all_voxel_thin_channels():
    """A pool made only of one-voxel channels is the floor whatever the budget: there is no wider water
    for a quantile to find."""
    h = 1e-6
    lab = np.ones((40, 40, 40), np.uint8)
    lab[::2, :, 20] = 0                                    # a comb of one-voxel lines
    g = LabelVolume(lab, h)
    assert g.passage_width.width == h and g.passage_width.floor_water_share == 1.0
    for budget in (0.0, 1e-3, 1e-2, 0.5):        # 0 resolves everything, which is this pool's own width
        assert measure_passage_width(lab == 0, h, budget=budget).width == h
    # and a budget of zero on a pool of MANY widths is its narrowest, not something finer than it
    lab = np.ones((40, 40, 40), np.uint8)
    lab[10:30, 10:30, 10:30] = 0
    lab[35, 10:30, 20] = 0
    assert measure_passage_width(lab == 0, h, budget=0.0).width == h


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
    assert pw.floor_water_share > STEP_WATER_BUDGET, "the fixture no longer has a rock's narrow passages"
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
    # the whole difference is the square of the resolution ratio, which is what "resolution-driven" meant
    assert n_new * (pw.width / h) ** 2 == pytest.approx(voxel_gated(h, D, dt), rel=0.01)
    assert pw.width >= 2 * R_vox * h / 8                  # the pore is resolved, so the passage is not the floor

    # the same pore at a voxel that barely expresses it has nothing to relax: two voxels across is all
    # the width there is, and the rule takes it
    assert voxelised_cylinder(1, h).passage_width.width == 2 * h


def test_the_passage_follows_the_pore_at_a_fixed_grid():
    """At one voxel size the measured passage grows with the pore, which is the adaptivity the rule is
    for. It is NOT the pore's own diameter and does not stay put in metres as the grid is refined at a
    fixed pore: the water a low quantile finds sits on the short axis runs near the wall, whose length
    falls with the voxel, so the rule buys less than the square of the resolution ratio. Asserted here is
    only what the rule guarantees -- the passage grows with the pore and lies between the floor and the
    diameter."""
    h = 20e-9
    at_R = {R_vox: voxelised_cylinder(R_vox, h).passage_width.width for R_vox in (5, 10, 25)}
    assert at_R[5] < at_R[10] < at_R[25]
    for R_vox, w in at_R.items():
        assert h < w < 2 * R_vox * h                       # between the floor and the diameter


# ──────────────────── the area weighting: what a pack certifies at the wall is not water-weighted
def fissured(n_fissures, wide=60, n=80, nz=64, h=1e-6):
    """A resolved slab pore of ``wide`` voxels plus ``n_fissures`` one-voxel-thin fissures of the same
    pool, each isolated from it. A fissure is nearly all wall, so it holds ~40x more of the pool's wall
    FACES than of its water -- the case where a water-only budget and an area-weighted one disagree."""
    lab = np.ones((n, wide + 8, nz), np.uint8)
    lab[4:n - 4, 4:4 + wide, 4:nz - 4] = 0
    for k in range(n_fissures):
        lab[6 + 2 * k, 4:4 + wide, 1] = 0
    return LabelVolume(lab, h)


def test_the_wall_area_share_holds_the_floor_where_the_water_share_alone_would_relax():
    """What a label-volume pack certifies at the wall -- the boundary local time, and any surface
    relaxation replayed from it -- accrues per FACE, not per unit volume, so the budget is spent in both
    weightings and the rule relaxes only where both are under it.

    One one-voxel fissure beside a 60-voxel pore holds a quarter of the budget of the water and ten times
    the budget of the wall faces, an amplification of about 40 (a fissure voxel carries two wall faces
    where a bulk voxel carries none). A water-only budget would step over it and take the 60-voxel
    passage; the area share holds the floor.
    """
    pw = fissured(1).passage_width
    assert pw.floor_water_share < STEP_WATER_BUDGET            # the water alone would license relaxing
    assert pw.floor_area_share > STEP_WATER_BUDGET             # the wall area does not
    assert pw.floor_area_share / pw.floor_water_share > 10     # the amplification, measured on the fixture
    assert pw.width == pw.voxel                                # so the rule stays at the floor
    # and what it did step over is under the budget in BOTH weightings, by construction
    assert pw.water_share < STEP_WATER_BUDGET and pw.area_share < STEP_WATER_BUDGET
    # with no fissure at all there is nothing to hold it and the pore's own width is taken
    assert fissured(0).passage_width.width > 10 * pw.voxel


def test_a_feature_set_holding_exactly_the_budget_is_resolved_and_not_stepped_over():
    """The predicate is strictly "under the budget", so a set holding exactly ``budget`` is held at the
    floor and only a budget above it steps over. Probed by running the measurement at the fixture's own
    share as the budget -- the boundary a quantile taken by index crosses on the wrong side."""
    g = fissured(2)
    pw = g.passage_width
    mask, vox = g._pool_grid == g.pool_index, g.voxel_size
    share = max(pw.floor_water_share, pw.floor_area_share)     # whichever weighting binds first
    assert measure_passage_width(mask, vox, budget=share).width == pw.voxel
    assert measure_passage_width(mask, vox, budget=share * (1 + 1e-9)).width > pw.voxel


# ───────────────────────────── a permeable wall: one Powles decision per step, so the voxel stays
def permeable_stack(kappa, h=1e-6, n_pore=1, n_wall=1, reps=16, nz=8):
    """A periodic stack of ``n_pore``-voxel pores of the walking pool separated by ``n_wall`` voxels of
    another, every face permeable at ``kappa``: at equilibrium the crossing rate per walker per unit time
    is the analytic ``kappa * S/V``, which is what a step that meets more than one face per step misses."""
    per = n_pore + n_wall
    col = np.ones(per * reps, np.uint8)
    for k in range(reps):
        col[k * per:k * per + n_pore] = 0
    lab = np.broadcast_to(col[:, None, None], (len(col), nz, nz)).copy()
    return LabelVolume(lab, h, periodic=True, pools={0: "free", 1: "other"}, permeability=kappa)


def crossing_rate(g, step_l, n_steps, kappa, n_walkers=4000, seed=0):
    """Granted crossings per walker per unit time at a fixed step length, on the geometry's own kernel."""
    import jax
    import jax.numpy as jnp
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    r0 = g.init_positions(n_walkers, k1)
    keys = jax.random.split(k2, n_walkers)
    kod = jnp.float32(kappa / D)

    def one(r, key):
        def body(c, _):
            r, k, n = c
            k, s, p = jax.random.split(k, 3)
            u = jax.random.normal(s, (3,), dtype=jnp.float32)
            u = u / jnp.linalg.norm(u)
            hit = g.interact(r, u * jnp.float32(step_l), kappa_over_D=kod, rho_over_D=0.0, key=p)
            return (hit.r, k, n + hit.crossed.astype(jnp.int32)), None
        (rf, _, n), _ = jax.lax.scan(body, (r, key, jnp.int32(0)), None, length=n_steps)
        return n
    n = jax.jit(jax.vmap(one))(r0, keys)
    return float(np.asarray(n).mean()) / (n_steps * (step_l ** 2 / (6.0 * D)))


def test_a_permeable_wall_is_stepped_against_the_voxel_and_not_the_passage():
    """``_wall`` grants ONE Powles trial per step, at the first face met, so what the step has to bound is
    the number of faces it may meet -- about ``step / (2 * local passage)`` -- and not a length. The
    passage is the wrong length for that: it licenses ``passage / voxel`` faces per step in the
    sub-budget one-voxel water it is allowed to step over. So a permeable label volume keeps the voxel,
    and ``step_width`` is the one place that choice is made."""
    h, dt = 0.5e-6, 1e-4
    lab = np.ones((40, 8, 8), np.uint8)
    lab[10:30] = 0                                             # a 20-voxel passage
    reflecting = LabelVolume(lab, h, surface_relaxivity_t2=10e-6)
    permeable = LabelVolume(lab, h, permeability=1e-5)
    assert reflecting.passage_width.width == permeable.passage_width.width == 20 * h   # same substrate
    assert step_width(permeable.passage_width, permeable=True)[0] == h
    assert step_width(reflecting.passage_width, permeable=False)[0] == 20 * h
    assert permeable.length_scales.min_feature == h
    assert reflecting.length_scales.min_feature == 20 * h
    # and the resolved count follows: the permeable walk is stepped as the voxel rule stepped it
    assert resolve_sub_steps(permeable, D, dt, surface=True) >= voxel_gated(h, D, dt)
    assert resolve_sub_steps(reflecting, D, dt, surface=True) == sub_steps_of(20 * h / SURFACE_STEP_FRACTION, D, dt)
    assert "Powles decision per step" in permeable.passage_width.record(permeable=True)["step_rule"]


def test_the_relaxed_step_undercounts_the_crossing_rate_and_the_rule_s_step_does_not():
    """The measurement the voxel rule rests on, on the worst case for it: a one-voxel pore, where every
    step meets a face. ``kappa`` is set at ``crossing_sub_steps``' own per-hit limit so that rule is not
    the one binding, and the step is varied alone.

    At the voxel -- half a face per step -- the crossing rate is the analytic ``kappa S/V``. At ten
    passages per step, which the passage rule would license for the same water inside a resolved
    substrate, it is a small fraction of it: one decision per step cannot grant the ten crossings the
    step swept past. This test fails under the relaxed rule, which is the point.
    """
    h = 1e-6
    for step_l, lo, hi in ((h, 0.85, 1.10), (20 * h, 0.0, 0.30)):
        kappa = 3e-3 * D / (2 * step_l)                        # CROSSING_P_MAX at this step
        g = permeable_stack(kappa, h)
        ratio = crossing_rate(g, step_l, 100, kappa) / (kappa * g.surface_to_volume())
        assert lo <= ratio <= hi, f"step {step_l / h:g} voxels: crossing rate / analytic = {ratio:.4f}"


# ──────────────────────────────────── an anisotropic grid: one voxel of the coarse axis is not the floor
def test_an_anisotropic_grid_measures_each_axis_with_its_own_extent():
    """The floor is one voxel of the FINEST axis, and a single voxel along a coarser axis is a wider
    passage and reads as one -- so ``width == voxel`` and ``floor_water_share >= budget`` say the same
    thing, which is what the producer's own prose is keyed on."""
    vox = (0.25e-6, 1e-6, 1e-6)
    lab = np.ones((22, 8, 8), np.uint8)
    lab[10:12] = 0                                             # two voxels across the finest axis
    pw = LabelVolume(lab, vox).passage_width
    assert pw.voxel == 0.25e-6 and pw.width == 2 * 0.25e-6     # 0.5 um, not one coarse voxel
    assert pw.floor_water_share == 0.0                         # nothing is one FINE voxel wide
    lab = np.ones((22, 8, 8), np.uint8)
    lab[10] = 0                                                # one voxel across the finest axis
    pw = LabelVolume(lab, vox).passage_width
    assert pw.width == pw.voxel and pw.floor_water_share == 1.0


def test_the_area_weight_is_the_wall_the_pack_certifies():
    """The weight the area share is taken with is the geometry's own wall: the face census
    ``_wall_faces`` sums to exactly the ``surface_area`` the pack's boundary local time accrues on, so the
    budget is spent against the same surface a consumer replays ``rho`` against, not a proxy for it."""
    from dmipy_sim.geometry.label_volume import _wall_faces
    h = 1e-6
    lab = np.ones((20, 20, 20), np.uint8)
    lab[4:16, 4:16, 4:16] = 0
    lab[9, 4:16, 18] = 0                                       # plus an isolated one-voxel fissure
    for periodic in (False, True):
        g = LabelVolume(lab, h, periodic=periodic)
        faces = _wall_faces(g._pool_grid == g.pool_index, g.periodic)
        assert faces.sum() * h ** 2 == pytest.approx(g.surface_area(), rel=1e-12)
