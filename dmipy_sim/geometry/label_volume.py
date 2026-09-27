"""Diffusion through a **segmented label volume**: the wall is the set of faces between voxels of
different pools.

A label volume already IS a wall. Every face shared by two voxels of different labels is a piece of
axis-aligned plane, so the surface needs no isosurface, no smoothing, no decimation and no repair:
the grid is both the geometry and its own spatial index. A step is resolved by walking the voxels the
segment crosses (the Amanatides-Woo grid traversal), stopping at the first face whose far voxel is
another pool, and reflecting specularly off it -- which for an axis-aligned face is the negation of
one component -- or crossing it under the Powles rule. The traversal visits every voxel the path
enters, so it cannot outrun a lookup and no collision sub-step rule applies; the step length is
bounded only by what the surface-relaxation estimator needs.

The surface is the **Manhattan** surface of the segmentation, not a smooth surface through it. A
voxelised sphere's face area is 3/2 of the sphere's own area (the sum of |n_x| + |n_y| + |n_z| over
the surface, averaged over orientations, is 3/2), so a relaxation rate measured here is the rate the
*voxelised* S/V gives. That is a property of the substrate, not an error in the walk: it is what a
random walk on a segmented image computes, and it is why a reference measurement made on the same
voxels is a like-for-like comparison. A consumer who wants the smooth-surface rate needs a smoothed
surface, which is a different substrate.

The step rules divide the **narrowest passage the seeded water occupies**, measured on the seeded
pool itself (:func:`measure_passage_width`) with the voxel as the floor, and not the voxel: a
segmentation's voxel is its resolution, so a rule gated on it asks 48,000 sub-steps of a 20 nm
electron-microscopy volume for a pore the grid resolves fifty times over (#478). A PERMEABLE wall is
the exception and keeps the voxel, because its estimator counts wall faces per step rather than a
length; :func:`step_width` is the one place that choice is made, with the measurements behind both.

Coordinates are metres. Voxel ``(i, j, k)`` spans ``origin + (i, j, k) * voxel_size`` to
``origin + (i + 1, j + 1, k + 1) * voxel_size``. An axis is ``periodic`` (the grid repeats, the
returned position stays continuous so the gradient phase is right) or not, and a non-periodic axis's
outer face is a reflecting boundary of the crop -- an artificial face, so it carries no surface
relaxation, exactly as a mesh's voxel faces do.

The spec spelling is ``surface.kind = "label_volume"`` (replay-pack-spec/SUBSTRATE.md); the producer
is :func:`dmipy_sim.spec.label_volume_spec`.
"""
from __future__ import annotations

import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._boundary import WallHit, representable_nudge, transmit_probability
from .base import Geometry, LengthScales, permeability_of

log = logging.getLogger(__name__)

#: The reflection rule of this family: ``step_l <= step_width / SPECULAR_STEP_FRACTION``, one step
#: length per :func:`step_width` -- the narrowest passage the seeded water occupies
#: (:func:`measure_passage_width`) at a reflecting wall, the VOXEL at a permeable one.
#:
#: Reflection off an axis-aligned face is exact at any step and the traversal misses no face, so this
#: bounds nothing about the collision. What it bounds is how many wall faces one step may meet, since
#: the faces a walker can meet along an axis are spaced by exactly the passage -- and the estimator that
#: cares is the PERMEABLE one, because ``_wall`` grants one Powles trial per step: the numbers are in
#: :func:`step_width`, and they are why a permeable label volume divides the voxel and the budget of
#: :data:`STEP_WATER_BUDGET` does not apply to it. Reflecting, measured on the voxelised R = 5 um sphere
#: (h = 1 um, 20,000 walkers x 200 steps): 0 escapes and a boundary local time of 1.0026 / 1.0016 /
#: 0.9997 of ``D S/V`` at 0.5 / 1 / 2 voxels per step, flat within the Monte-Carlo floor, and flat to
#: 0.3 % out to 25 passages per step on the three one-voxel worst cases (:func:`step_width`).
SPECULAR_STEP_FRACTION = 1.0

#: The surface tier's rule, ``step_l <= step_width / SURFACE_STEP_FRACTION``: a quarter of the narrowest
#: PASSAGE the seeded water occupies (:func:`measure_passage_width`), floored at the voxel -- or a quarter
#: of the voxel where :func:`step_width` says the voxel.
#:
#: The 4 is NOT in the estimator, and measurement says so twice over. A voxel face is exactly flat, so
#: the overshoot has no curvature bias and the per-bounce sum telescopes: on a voxelised 10 um slab at
#: 200,000 walkers, ``-E[dlog_w] / T`` at ``rho / D = 1`` is 1.0015 / 1.0020 / 1.0039 / 0.9980 of
#: ``D S/V`` at 2 / 1 / 0.5 / 0.25 voxels per step (floor 2.9e-3) and ``Box1D`` of the same width gives
#: 1.0005 / 1.0011 / 0.9986 / 0.9954; and the boundary local time stays within 0.3 % of ``D S/V`` out to
#: 25 passages per step on a one-voxel slab, tube and pocket (the table in :func:`step_width`). What the
#: quarter buys is the pore's SPATIAL sampling, which is exactly the quantity the rule now divides.
#:
#: What needs the finer step is a REAL substrate, whose passages are not all the size of its mean.
#: ``V/S`` is that mean: on the Imperial LV60A sand pack it is 1.67 voxels while its narrowest passages
#: are one voxel, so gating on ``(V/S) / 2`` licensed a 0.84-voxel step. Measured there, at 200,000
#: walkers on one sampling grid with the sub-step count pinned so that only the WALK's step changes, the
#: T2 decay is still changing at that step: its single-exponential fit (which does not go through a
#: Laplace inversion, so it reads the decay and not the estimator) is 563.2 / 562.2 / 560.6 / 560.3 ms
#: at 0.836 / 0.591 / 0.418 / 0.241 voxels -- moving by 0.18 %, then 0.28 %, then 0.05 %. It has stopped
#: by a quarter of a voxel and not before. Past that limit nothing moves: at the grid the reproduction
#: uses, the log-mean T2 is 487.6 / 487.0 / 486.1 ms and the fit 558.8 / 557.0 / 556.7 ms at
#: 0.249 / 0.176 / 0.125 voxels, 0.31 % and 0.38 % over a four-fold refinement, inside the walkers'
#: own 0.22 % floor.
#:
#: That calibration is on a substrate whose narrowest passages ARE one voxel (measured: 1.8 % of
#: LV60A's pore water sits in a one-voxel passage), so the 4 divides a passage, not a resolution. The
#: two readings differ by two orders of magnitude the moment a segmentation resolves its pores -- a
#: 20 nm electron-microscopy volume of a 1 um neurite asks 48,000 sub-steps per 0.1 ms save of the
#: voxel and 481 of the passage -- and it is the passage that was measured (#478).
SURFACE_STEP_FRACTION = 4.0

#: The largest share of a voxel the representable nudge may be. The nudge is ``1e-4`` of the voxel by
#: design, raised to eight float32 ulps at the box's extent where that is larger
#: (:func:`~dmipy_sim.geometry._boundary.representable_nudge`), and a volume far from the origin drives it
#: up until "just inside this voxel" is no longer a float32 position: the Ling 2022 sand packs are
#: distributed at the micro-CT stage's absolute coordinates, near ``-717 mm`` at a 3.93 um voxel, where
#: the ulp is 1/65 of a voxel and the nudge becomes 0.12 of one. MEASURED there: 739 walker-steps of
#: 8,000 walkers x 500 saves were refused for ending in another pool without a granted crossing, against
#: 6 for the same volume translated to the origin -- a factor of 123. So a nudge above this fraction is
#: a substrate that cannot be walked where it sits, and it is refused with the translation that fixes it.
NUDGE_FRACTION_MAX = 1e-2

#: The share of the seeded pool the step rules may leave unresolved, counted BOTH as water and as wall
#: area: the quantile of the passage-width distribution they divide (:func:`measure_passage_width`).
#:
#: What the share bounds, exactly. A passage the step does not resolve is mis-stepped for as long as a
#: walker is in it, and seeding is uniform over the pool by rejection, so occupancy equals volume share:
#: the water share is the share of walker-TIME spent where the step is wrong. It is not a share of
#: walkers -- over a long walk in a connected pool essentially every walker visits every passage. For an
#: observable weighted by wall area rather than by water the relevant share is the area one, and the two
#: come apart badly: a set of one-voxel fissures beside a 60-voxel pore holds 7.8e-4 of the water and
#: 3.2e-2 of the wall faces, a 41x amplification. So the quantile is taken over both weightings and the
#: rule relaxes only where BOTH are under the budget.
#:
#: Why 1e-3 and not something looser. It is below the Monte-Carlo floor of the walker counts these packs
#: are built at (2.2e-3 at 200,000 walkers, dmrai-lab/dmipy-sim#470), so a passage the rule steps over
#: cannot be read in the pack that steps over it. From above it is held by the substrate the constants
#: were calibrated on and the packs already published from it and from Ling's, all of which must keep the
#: voxel floor: the share of pore water in a ONE-VOXEL passage is 1.80 % on Talabi's LV60A (central
#: 300^3, 10.002 um), 1.59 % on Berea, 1.38 % on F42A, and 0.27 % / 0.24 % on Ling's pure quartz /
#: garnet packs (450^3, 3.935 um). The narrowest margin is 2.4x, and the 5e-3 quantile of both Ling packs
#: is already two voxels: this budget cannot be loosened without moving a published pack.
STEP_WATER_BUDGET = 1e-3

#: ``uint16`` sentinel for "no wall along any axis" in the passage-width grid: the run spanned its axis.
_NO_WALL = np.uint16(np.iinfo(np.uint16).max)


class PassageWidth(NamedTuple):
    """What a label volume's step rules divide, and the measurement it came from.

    Attributes
    ----------
    width : float
        The passage width the rules of a REFLECTING wall divide (m): the widest passage such that the
        water and the wall area in anything narrower are both strictly under ``budget``, never below
        ``voxel``. :func:`step_width` is what a geometry actually divides, which for a permeable wall is
        the voxel instead.
    voxel : float
        The floor (m), ``min(voxel_size)``: the narrowest passage a segmentation can express.
    budget : float
        The share of the seeded pool's walled water, AND of its wall area, allowed to sit in something
        narrower than ``width`` (:data:`STEP_WATER_BUDGET`).
    water_share, area_share : float
        What the rule actually stepped over at ``width``: the share of the walled water, and of the wall
        faces, in a passage narrower than it. Both are strictly under ``budget`` by construction.
    floor_water_share, floor_area_share : float
        The share of the walled water, and of the wall faces, whose passage IS the floor -- how much
        margin the budget has on this substrate. ``width == voxel`` exactly when one of them reaches the
        budget.
    median : float
        The median passage width (m) over the walled water: what a rule built on a mean would divide.
    walled : bool
        Whether the seeded pool meets another pool along any axis at all. A pool that meets none is
        confined only by the crop's own faces, which are not a wall of the substrate, so it keeps the
        voxel floor and says so here.
    walled_fraction : float
        The share of the seeded pool with a wall along at least one axis. Every share above is over those
        voxels alone, so unwalled water cannot hide a narrow passage in the quantile.
    periodic : tuple of 3 bool
        Which axes repeat, because that is an input that changes the answer: a periodic axis's spanning
        run is a walk that never meets a wall.
    n_voxels : int
        The seeded pool's voxels the measurement was made on.
    """
    width: float
    voxel: float
    budget: float
    water_share: float
    area_share: float
    floor_water_share: float
    floor_area_share: float
    median: float
    walled: bool
    walled_fraction: float
    periodic: tuple
    n_voxels: int

    def record(self, *, permeable=False):
        """The measurement as a JSON-able dict for a spec's ``realisation`` -- so a pack's provenance says
        which feature set its step, which rule chose it and what it stepped over."""
        width, rule = step_width(self, permeable=permeable)
        finite = lambda x: None if not np.isfinite(x) else float(x)
        return {"step_width_m": float(width), "step_rule": rule,
                "passage_width_m": float(self.width), "voxel_m": float(self.voxel),
                "budget": float(self.budget), "water_share": finite(self.water_share),
                "area_share": finite(self.area_share),
                "floor_water_share": finite(self.floor_water_share),
                "floor_area_share": finite(self.floor_area_share),
                "median_passage_width_m": finite(self.median), "walled": bool(self.walled),
                "walled_fraction": float(self.walled_fraction),
                "periodic": [bool(p) for p in self.periodic], "n_voxels": int(self.n_voxels),
                "measured": "the narrowest passage the seeded water occupies: per voxel the shortest run "
                            "of the seeded pool through it along an index axis, at the quantile of the "
                            "walled water AND of the wall area given by `budget`, floored at the voxel "
                            "(geometry.label_volume.measure_passage_width)"}


def step_width(passage, *, permeable):
    """``(width, rule)``: the length a label volume's step rules divide, and why it is that one.

    A REFLECTING wall divides the measured passage. Its estimator does not need the voxel: the faces are
    exactly flat, so the per-bounce overshoot sum telescopes and the boundary local time comes out
    step-INDEPENDENT. Measured at 4,000 walkers on the three worst cases a segmentation can hold, as
    ``rate / (D S/V)`` against step / passage of 0.25 / 1 / 2 / 5 / 10 / 25: a one-voxel slab gives
    0.9999 / 0.9996 / 0.9996 / 0.9989 / 0.9985 / 1.0015, a one-voxel tube 0.9992 / 0.9987 / 0.9990 /
    0.9988 / 0.9984 / 1.0002, and a one-voxel pocket 1.0032 / 1.0030 / 1.0031 / 1.0030 / 1.0030 /
    1.0030 -- flat to 0.3 % over a hundredfold range of step, with no trend. What the step buys a
    reflecting walk is the SPATIAL SAMPLING of the pore, which is what the passage measures.

    A PERMEABLE wall divides the voxel, and the budget does not apply to it. ``_wall`` grants ONE Powles
    trial per step, at the first face met, so what the step has to keep bounded is the number of wall
    faces a step meets -- about ``step / (2 * local passage)`` -- and not a length. Measured on a
    one-voxel slab at ``crossing_sub_steps``' own per-hit limit (20,000 walkers x 200 steps), the
    crossing rate over the analytic ``kappa S/V`` is 0.961 / 0.853 / 0.476 / 0.223 / 0.112 at 0.5 / 1 /
    2.5 / 6 / 12.5 hits per step. The voxel keeps it under one hit per step in one-voxel water, which is
    where a crossing is most likely to be missed; the passage would license ``passage / voxel`` hits
    there -- ten on a 20 nm EM tube -- and undercount exchange four- to ninefold. ``crossing_sub_steps``
    bounds the per-hit probability, which is a different failure, so nothing else catches this.
    """
    if permeable:
        return passage.voxel, ("the voxel: a permeable wall takes ONE Powles decision per step, so the "
                               "step is bounded by the faces it may meet in the narrowest water a "
                               "walker can cross, not by the passage the seeded water occupies")
    return passage.width, (f"the narrowest passage the seeded water occupies, at a {passage.budget:g} "
                           f"budget of the walled water and of the wall area, floored at the voxel")


def _axis_runs(mask, axis, periodic):
    """Per voxel, the length in voxels of the maximal run of ``mask`` along ``axis`` through it.

    A run that spans the axis comes back as the axis length: on a periodic axis it is a walk that never
    meets a wall, and on a reflecting one it is bounded by the crop's own faces, which are not a wall of
    the substrate -- :func:`passage_widths` reads both as no wall along that axis.
    """
    m = np.moveaxis(np.asarray(mask, bool), axis, -1)
    n = m.shape[-1]
    if periodic:
        m = np.concatenate([m, m, m], axis=-1)      # a run may wrap; three copies hold any run of length n
    span = m.shape[-1]
    idx = np.arange(span, dtype=np.int32)
    lo = np.maximum.accumulate(np.where(m, np.int32(-1), idx), axis=-1)      # the last non-pool voxel before
    hi = np.minimum.accumulate(np.where(m, np.int32(span), idx)[..., ::-1], axis=-1)[..., ::-1]
    run = hi - lo - 1
    if periodic:
        run = np.minimum(run[..., n:2 * n], n)
    return np.moveaxis(run, -1, axis)


def _passage_voxels(mask, vox, periodic, chunk_bytes=64 << 20):
    """The passage width at every voxel in units of ``min(vox)``, ``uint16``, ``_NO_WALL`` where no axis
    has a wall -- the grid :func:`passage_widths` and :func:`measure_passage_width` are built on.

    ``uint16`` in voxel units rather than float metres, and one axis at a time in blocks of the other
    two, because this runs on whole segmentations: the resident cost is two bytes per voxel plus the
    mask's one plus a bounded block. Measured peak VmHWM over :func:`measure_passage_width` on a 450^3
    mask: 1.04 GiB in 2.9 s, and 1.06 GiB in 7.4 s with all three axes periodic, against 3.03 and
    6.23 GiB for the dense float version this replaced -- about 11 bytes per voxel including the
    interpreter, so a 1000^3 electron-microscopy volume, the class this rule is for, costs about 11 GB
    rather than 33-68. A run wider than 65534 voxels saturates, which shortens the passage and so can
    only make the step finer.
    """
    m = np.asarray(mask, bool)
    if m.ndim != 3:
        raise ValueError(f"a passage width is measured on a 3-D mask, got shape {m.shape}")
    per = np.broadcast_to(np.asarray(periodic, bool).ravel(), (3,))
    scale = np.asarray(vox, np.float64) / float(np.min(vox))
    out = np.full(m.shape, _NO_WALL, np.uint16)
    for ax in range(3):
        lead = 1 if ax == 0 else 0                                  # block over an axis the runs do not span
        rows = max(1, int(chunk_bytes // (16 * max(1, m.size // m.shape[lead]))))
        for a in range(0, m.shape[lead], rows):
            sl = (slice(None),) * lead + (slice(a, a + rows),)
            run = _axis_runs(m[sl], ax, bool(per[ax]))
            val = np.minimum(np.floor(run * scale[ax]), float(_NO_WALL) - 1.0).astype(np.uint16)
            np.minimum(out[sl], np.where(run >= m.shape[ax], _NO_WALL, val), out=out[sl])
    return out


def _wall_faces(mask, periodic):
    """Per voxel, how many of its six faces are shared with another pool, ``uint8``.

    The weight of a voxel in every quantity a label-volume pack certifies at the wall: the boundary local
    time, and so a surface relaxation, accrues per face and not per unit volume. The crop's own outer
    faces are not counted, exactly as they are not counted in ``S/V``.
    """
    m = np.asarray(mask, bool)
    per = np.broadcast_to(np.asarray(periodic, bool).ravel(), (3,))
    faces = np.zeros(m.shape, np.uint8)
    for ax in range(3):
        a = np.moveaxis(m, ax, 0)
        f = np.moveaxis(faces, ax, 0)
        wall = a[:-1] != a[1:]
        f[:-1] += wall & a[:-1]
        f[1:] += wall & a[1:]
        if per[ax]:
            wrap = a[-1] != a[0]
            f[-1] += wrap & a[-1]
            f[0] += wrap & a[0]
    return faces


def passage_widths(mask, voxel_size, *, periodic=(False, False, False)):
    """``(nx, ny, nz)`` float32 of the passage width at every voxel of ``mask``: the shortest run of the
    pool through it along an index axis, times that axis's voxel extent; ``inf`` where no axis has a wall.

    The wall of a label volume is made of axis-aligned faces, so the faces a walker can meet along an
    axis are spaced by exactly that run -- which is why the passage is measured along the axes and not as
    a distance to the nearest wall. The distance route is the standard porous-media pore size (local
    thickness: the largest inscribed ball, scipy's EDT plus ball openings) and it measures a real
    quantity, but not this one, and it cannot be substituted here: measured, its 1e-3 quantile is TWO
    voxels on every released rock (LV60A central 300^3 and 120^3, Berea, F42A; 8 and 16 voxels on
    voxelised spheres at h/R = 0.1 and 0.05, so it does separate a rock from a smooth wall), and two
    voxels would take Berea from 2 sub-steps to 1 and Ling's packs from 15 to 4 -- moving packs that are
    already published, at a step whose reflecting estimator was never measured to need the change.
    The axis run puts those rocks at one voxel and leaves them where they are.

    An oblique passage is read along the axes too, so a sheet at 45 deg to the grid reads up to
    ``sqrt(3)`` times its perpendicular thickness. That is the quantity the walk sees: its faces are the
    axis-aligned ones the run measures, and their spacing along an axis is what a walker rattles between.
    """
    vox = np.broadcast_to(np.asarray(voxel_size, np.float64).ravel(), (3,))
    v = _passage_voxels(mask, vox, periodic)
    return np.where(v == _NO_WALL, np.float32(np.inf), v.astype(np.float32) * np.float32(np.min(vox)))


def measure_passage_width(mask, voxel_size, *, periodic=(False, False, False), budget=STEP_WATER_BUDGET):
    """The narrowest passage the water in ``mask`` occupies, as a :class:`PassageWidth`.

    The seeded pool's own passage-width distribution (:func:`passage_widths`), cut at the widest value
    whose tail below it holds strictly less than ``budget`` of the walled water AND strictly less than
    ``budget`` of the wall faces (:func:`_wall_faces`), floored at the voxel. Strictly less, so a feature
    set holding exactly the budget is resolved and not stepped over.

    This is the worst case the walk has to resolve rather than the average one -- ``V/S`` is the average,
    it is measured and reported by :meth:`LabelVolume.surface_to_volume`, and it is not a step rule -- and
    it is measured per substrate, so a grid that resolves its pores is walked at its pores and a
    segmentation whose passages are one voxel wide is walked at the voxel.
    """
    m = np.asarray(mask, bool)
    vox = np.broadcast_to(np.asarray(voxel_size, np.float64).ravel(), (3,))
    per = tuple(bool(p) for p in np.broadcast_to(np.asarray(periodic, bool).ravel(), (3,)))
    floor = float(np.min(vox))
    n = int(m.sum())
    if n == 0:
        raise ValueError("a passage width is measured on the water: this mask holds no voxel of the "
                         "seeded pool")
    water, faces = _passage_census(m, vox, per)
    n_walled = float(water[1:-1].sum())
    kw = dict(voxel=floor, budget=float(budget), periodic=per, n_voxels=n,
              walled_fraction=n_walled / n)
    if n_walled == 0.0:
        return PassageWidth(width=floor, water_share=0.0, area_share=0.0, floor_water_share=float("nan"),
                            floor_area_share=float("nan"), median=float("inf"), walled=False, **kw)
    w = water[1:-1] / n_walled                                       # per width in voxels, 1 .. _NO_WALL-1
    a = faces[1:-1] / max(faces[1:-1].sum(), 1.0)
    below_w = np.concatenate([[0.0], np.cumsum(w)[:-1]])             # the share strictly NARROWER
    below_a = np.concatenate([[0.0], np.cumsum(a)[:-1]])
    ok = (below_w < float(budget)) & (below_a < float(budget))       # monotone in the width
    i = int(np.flatnonzero(ok & (w > 0.0))[-1])                      # the widest admissible OBSERVED value
    median = 1 + int(np.searchsorted(np.cumsum(w), 0.5))
    return PassageWidth(width=floor * (i + 1), water_share=float(below_w[i]),
                        area_share=float(below_a[i]), floor_water_share=float(w[0]),
                        floor_area_share=float(a[0]), median=floor * median, walled=True, **kw)


def _passage_census(mask, vox, periodic, chunk_voxels=1 << 22):
    """``(water, faces)``: how many voxels of ``mask``, and how many of its wall faces, sit at each passage
    width in voxels -- two histograms of ``_NO_WALL + 1`` bins, bin ``_NO_WALL`` being the unwalled water.

    Counted in blocks so that nothing the size of the grid is ever held per voxel beyond the three grids
    the measurement needs (the mask's byte, the width's two, the face count's one). The dense route --
    one float array of passage widths, ``np.unique`` over the pool and an int64 inverse -- is what made
    the measurement cost 33 bytes per voxel; a 1000^3 electron-microscopy volume, the class this rule is
    for, cannot afford that.
    """
    m = np.asarray(mask, bool)
    width = _passage_voxels(m, vox, periodic)
    area = _wall_faces(m, periodic)
    nb = int(_NO_WALL) + 1
    water = np.zeros(nb, np.float64)
    faces = np.zeros(nb, np.float64)
    rows = max(1, int(chunk_voxels // max(1, m.size // m.shape[0])))
    for a0 in range(0, m.shape[0], rows):
        sl = slice(a0, a0 + rows)
        sel = m[sl]
        v = width[sl][sel].astype(np.int32)
        if not v.size:
            continue
        water += np.bincount(v, minlength=nb)
        faces += np.bincount(v, weights=area[sl][sel].astype(np.float64), minlength=nb)
    return water, faces


class LabelVolume(Geometry):
    """Diffusion in one pool of a segmented 3-D image (metres).

    Parameters
    ----------
    labels : (nx, ny, nz) array of small non-negative integers
        The pool label of each voxel.
    voxel_size : float or (3,)
        The voxel's extent along each index axis, in metres.
    origin : (3,), optional
        The lower corner of voxel ``(0, 0, 0)``. Default the coordinate origin.
    periodic : bool or (3,) of bool, optional
        Whether the grid repeats along each axis. A non-periodic axis's outer face reflects.
    pools : mapping ``{label value: pool name}``, optional
        Which pool each label is. Insertion order is the pool id, so the first entry is pool 0, the
        free pool, and the ids are dense -- the ``.rpk`` convention every channel is indexed by.
        Default ``{0: "free", 1: "grain"}``, the segmentation convention of a micro-CT rock (void 0).
        A label present in ``labels`` and absent from the map is refused.
    pool : str, optional
        The pool the walk occupies: which pool :meth:`init_positions` seeds. Default the first.
    surface_relaxivity_t2 : float, optional
        Transverse surface relaxivity rho (m/s) at every face between two pools.
    permeability : float, optional
        Membrane permeability kappa (m/s) at every face between two pools, one Powles trial per step
        at the first face met. ``None`` (the default) is a reflecting wall.
    """

    supports_permeability = True
    reflection_step_fraction = SPECULAR_STEP_FRACTION
    surface_substep_frac = SURFACE_STEP_FRACTION

    def __init__(self, labels, voxel_size, *, origin=(0.0, 0.0, 0.0), periodic=False,
                 pools=None, pool=None, surface_relaxivity_t2=None, permeability=None):
        lab = np.asarray(labels)
        if lab.ndim != 3:
            raise ValueError(f"a label volume is 3-D, got shape {lab.shape}")
        if lab.dtype != np.uint8:
            if lab.min() < 0 or lab.max() > 255:
                raise ValueError(f"labels are 0..255, got [{lab.min()}, {lab.max()}]")
            lab = lab.astype(np.uint8)
        self.labels = np.ascontiguousarray(lab)
        self.dims = np.asarray(self.labels.shape, np.int64)

        vox = np.broadcast_to(np.asarray(voxel_size, np.float64).ravel(), (3,)).astype(np.float64)
        if np.any(vox <= 0) or not np.all(np.isfinite(vox)):
            raise ValueError(f"voxel_size must be positive and finite on every axis, got {list(vox)}")
        self.voxel_size = vox
        self.origin = np.broadcast_to(np.asarray(origin, np.float64).ravel(), (3,)).astype(np.float64)
        self.periodic = np.broadcast_to(np.asarray(periodic, bool).ravel(), (3,)).astype(bool)

        self.pools = dict(pools if pools is not None else {0: "free", 1: "grain"})
        if not self.pools:
            raise ValueError("a label volume needs at least one pool in `pools`")
        names = list(self.pools.values())
        if len(set(names)) != len(names):
            raise ValueError(f"pool names must be unique, got {names}")
        present = set(int(v) for v in np.unique(self.labels))
        missing = sorted(present - set(int(k) for k in self.pools))
        if missing:
            raise ValueError(f"the volume holds labels {missing} that `pools` does not name; every label a walker "
                             f"can meet must be a pool (pools = {self.pools})")
        self.pool = str(pool) if pool is not None else names[0]
        if self.pool not in names:
            raise ValueError(f"pool {self.pool!r} is not one of {names}")
        self.pool_index = names.index(self.pool)

        self.surface_relaxivity_t2 = None if surface_relaxivity_t2 is None else float(surface_relaxivity_t2)
        self.permeability = permeability_of(permeability)

        # label -> pool id, -1 for a label no pool claims (never reached: the constructor refused it)
        pool_of = np.full(256, -1, np.int8)
        for i, (value, name) in enumerate(self.pools.items()):
            pool_of[int(value)] = i
        self._pool_grid = np.take(pool_of, self.labels)             # (nx, ny, nz) int8 pool ids

        self.box_min = self.origin.copy()
        self.box_max = self.origin + self.dims * self.voxel_size
        extent = float(np.max(np.abs(np.concatenate([self.box_min, self.box_max]))) + self.voxel_size.max())
        self._nudge = float(representable_nudge(1e-4 * float(self.voxel_size.min()), extent))
        frac = self._nudge / float(self.voxel_size.min())
        if frac > NUDGE_FRACTION_MAX:
            raise ValueError(
                f"this volume cannot be walked where it sits: a float32 coordinate at "
                f"{extent:.4g} m has an ulp of {np.spacing(np.float32(extent)):.3g} m, so the smallest "
                f"nudge that moves a walker off a voxel face is {self._nudge:.3g} m = {frac:.3g} of the "
                f"{self.voxel_size.min():.4g} m voxel, above the {NUDGE_FRACTION_MAX:g} this geometry "
                f"allows (it is designed to be 1e-4). The origin is {list(self.origin)} m -- a scanner's "
                f"absolute stage coordinates, most likely. Translate the volume: the substrate's position "
                f"is nothing, and origin=-0.5 * shape * voxel_size centres it."
            )

        self._passage = None                                        # measured on first use (`passage_width`)
        self._porosity = float(np.mean(self._pool_grid == self.pool_index))
        if self._porosity <= 0.0:
            raise ValueError(f"pool {self.pool!r} occupies no voxel of this volume")
        self._faces, self._area = self._measure_interfaces()
        area = self.surface_area()
        self._v_over_s = self.volume() / area if area > 0 else float("inf")

        # A step crosses at most as many grid planes as the grid has, plus one reflection each; four
        # times that is a bound no step of a diffusion walk can reach, and reaching it refuses the
        # step (`WallHit.illegal`) instead of leaving part of the path untested.
        self._max_events = int(4 * self.dims.sum() + 16)

        self.LAB = jnp.asarray(self._pool_grid, jnp.int8)
        self.VOX = jnp.asarray(self.voxel_size, jnp.float32)
        self.ORG = jnp.asarray(self.origin, jnp.float32)
        self.DIMS = jnp.asarray(self.dims, jnp.int32)
        self.PER = jnp.asarray(self.periodic, bool)

    # ------------------------------------------------------------------ measured properties
    def _measure_interfaces(self):
        """The face census of the grid: ``{(pool_a, pool_b): faces}`` over every neighbouring pair of
        voxels of different pools, ``a < b``.

        Only neighbouring pairs INSIDE the grid are counted, plus the wrap pair on a periodic axis.
        A non-periodic axis's outer faces are the crop's own boundary, not an interface between two
        pools of the substrate, so they are neither a wall in the spec nor area in ``S/V``.
        """
        g = self._pool_grid
        per_pair = {}
        for ax in range(3):
            a = np.swapaxes(g, 0, ax)
            slabs = [(a[:-1], a[1:])]
            if self.periodic[ax]:
                slabs.append((a[-1:], a[:1]))
            for left, right in slabs:
                lo = np.minimum(left, right)
                hi = np.maximum(left, right)
                sel = lo != hi
                if not sel.any():
                    continue
                code = lo[sel].astype(np.int32) * 256 + hi[sel].astype(np.int32)
                vals, counts = np.unique(code, return_counts=True)
                for v, c in zip(vals.tolist(), counts.tolist()):
                    key = (v // 256, v % 256, ax)
                    per_pair[key] = per_pair.get(key, 0) + int(c)
        h = self.voxel_size
        face_area = (h[1] * h[2], h[0] * h[2], h[0] * h[1])
        faces, area = {}, {}
        for (i, j, ax), c in per_pair.items():
            faces[(i, j)] = faces.get((i, j), 0) + c
            area[(i, j)] = area.get((i, j), 0.0) + c * face_area[ax]
        return faces, area

    def interfaces(self):
        """``{(pool_a, pool_b): faces}``, ``a < b``: which pools of this volume share a wall, and how
        many voxel faces it is made of."""
        return dict(self._faces)

    def porosity(self):
        """The walking pool's volume fraction of the volume."""
        return self._porosity

    def volume(self):
        """The walking pool's volume (m^3)."""
        return self._porosity * float(np.prod(self.dims)) * float(np.prod(self.voxel_size))

    def surface_area(self):
        """The walking pool's wall area (m^2): its faces with other pools, the Manhattan surface."""
        return float(sum(a for pair, a in self._area.items() if self.pool_index in pair))

    def surface_to_volume(self):
        """``S/V`` (1/m) of the walking pool against the voxelised surface."""
        return self.surface_area() / self.volume()

    @property
    def passage_width(self):
        """The narrowest passage the walking pool's water occupies, as a :class:`PassageWidth`.

        Measured once on the pool this geometry seeds -- which for a geometry built from a spec is the
        pool the spec seeds -- and kept, because the step rules ask for it once per driver. The cost is
        blocked passes over the grid: 4.2 s and 1.04 GiB peak VmHWM for Ling's 450^3 sand pack
        (:func:`_passage_voxels` has the memory numbers and what they replaced).
        """
        if self._passage is None:
            self._passage = measure_passage_width(self._pool_grid == self.pool_index, self.voxel_size,
                                                  periodic=self.periodic)
            pw = self._passage
            width, rule = step_width(pw, permeable=self.permeability is not None)
            if not pw.walled:
                log.info("LabelVolume: pool %r meets no other pool along any axis, so nothing bounds its "
                         "passages and the step rules keep the voxel floor, %.4g um", self.pool,
                         width * 1e6)
            else:
                log.info("LabelVolume: pool %r is stepped against %.4g um -- %s. Its narrowest passage at "
                         "the %g budget is %.4g um (%.3g of its walled water and %.3g of its wall faces "
                         "are in a one-voxel passage); voxel %.4g um, median passage %.4g um, mean V/S "
                         "%.4g um", self.pool, width * 1e6, rule, pw.budget, pw.width * 1e6,
                         pw.floor_water_share, pw.floor_area_share, pw.voxel * 1e6, pw.median * 1e6,
                         self._v_over_s * 1e6)
        return self._passage

    @property
    def length_scales(self):
        """What the step rules divide: :func:`step_width` of this pool's measured passage.

        ``min_feature`` is the narrowest passage the seeded water occupies (:attr:`passage_width`,
        floored at the voxel) for a reflecting wall, and the VOXEL when this geometry carries a
        permeability -- the two rest on different measurements and :func:`step_width` states both. It is
        the worst case the walk has to resolve ON THIS SUBSTRATE, which is what the reflection and
        surface constants of this family were measured against (both were calibrated on a sand pack whose
        narrowest passages ARE one voxel; see :data:`SURFACE_STEP_FRACTION`). The voxel alone is a
        RESOLUTION, and dividing it asks 48,000 sub-steps of a 20 nm electron-microscopy volume for a
        pore its grid resolves fifty times over (#478). ``surface_pore`` is left unset so the
        surface-relaxivity rule divides the same length rather than ``V/S``, which is a MEAN and on a
        real rock is larger than the passages that set the bias. ``V/S`` is measured and reported by
        :meth:`surface_to_volume`; it is a property of the substrate, not a step rule.
        """
        return LengthScales(min_feature=float(step_width(self.passage_width,
                                                         permeable=self.permeability is not None)[0]))

    # ------------------------------------------------------------------ labelling
    def _wrap(self, r):
        """``r`` folded into the grid's box on the periodic axes, untouched on the others.

        The walk keeps a CONTINUOUS position, so the gradient phase is right, and the geometry is
        queried at the wrapped one -- which also keeps ``(r - origin) / voxel_size`` inside the box's
        own float32 resolution however far a periodic walk has drifted.
        """
        L = self.DIMS.astype(jnp.float32) * self.VOX
        return jnp.where(self.PER, self.ORG + jnp.mod(r - self.ORG, L), r)

    def _index(self, r):
        """The voxel index of ``r``, unwrapped: a periodic walker one step outside the box indexes one
        voxel outside it, and :meth:`_pool_at_index` wraps that when it reads the grid."""
        return jnp.floor((r - self.ORG) / self.VOX).astype(jnp.int32)

    def _pool_at_index(self, i):
        """The pool id at a voxel index: wrapped on the periodic axes, ``-1`` beyond a non-periodic face."""
        iw = jnp.where(self.PER, jnp.mod(i, self.DIMS), i)
        inside = jnp.all((iw >= 0) & (iw < self.DIMS))
        c = jnp.clip(iw, 0, self.DIMS - 1)
        return jnp.where(inside, self.LAB[c[0], c[1], c[2]].astype(jnp.int32), jnp.int32(-1))

    def _into_voxel(self, r, i):
        """``r`` moved strictly inside voxel ``i``, by at most the nudge.

        The traversal decides which voxel the walker is in by integer steps; ``floor((r - origin) /
        voxel_size)`` decides it again, in float32, every time the walk asks what pool a position is.
        Within one ulp of a face the two disagree, because ``(r - origin)`` rounds onto the plane, and
        a step that starts from such a position starts in the wrong voxel and leaves its pool without
        meeting a face. A position a nudge clear of every face of its own voxel is one the two agree
        on: the nudge is at least eight float32 ulps at the box's extent
        (:func:`~dmipy_sim.geometry._boundary.representable_nudge`), so no rounding in the index can
        reach a face, and it is 100 pm against a micrometre voxel.
        """
        lo = self.ORG + i.astype(jnp.float32) * self.VOX
        n = jnp.float32(self._nudge)
        return jnp.clip(r, lo + n, lo + self.VOX - n)

    def classify_position(self, r):
        """The pool id of the voxel containing ``r``: ``-1`` beyond a non-periodic face."""
        return self._pool_at_index(self._index(self._wrap(r)))

    def classify_positions_exact(self, pts, chunk=1_000_000):
        """Pool ids of host-side points, by indexing the grid on the host -- no device gather per point."""
        p = np.asarray(pts, np.float64).reshape(-1, 3)
        i = np.floor((p - self.origin) / self.voxel_size).astype(np.int64)
        i = np.where(self.periodic, np.mod(i, self.dims), i)
        inside = np.all((i >= 0) & (i < self.dims), axis=1)
        c = np.clip(i, 0, self.dims - 1)
        out = np.where(inside, self._pool_grid[c[:, 0], c[:, 1], c[:, 2]], -1)
        return jnp.asarray(out, jnp.int32)

    # ------------------------------------------------------------------ seeding
    def init_positions(self, n_walkers, key, oversample=4):
        """Uniform positions in the walking pool, exact.

        A point uniform on the union of the pool's voxels is a point uniform in the bounding box
        conditioned on landing in one of them, so rejection against the label grid is exact and
        costs one host gather per draw. The pool's voxel list is the other exact route and is not
        taken: a 450^3 rock's pore holds 33 million voxels, and their index list is larger than the
        image.
        """
        need, out = int(n_walkers), []
        lo, hi = self.box_min, self.box_max
        while need > 0:
            key, sub = jax.random.split(key)
            m = max(1024, int(need * oversample / max(self._porosity, 1e-6)))
            p = np.asarray(jax.random.uniform(sub, (m, 3), dtype=jnp.float32,
                                              minval=jnp.asarray(lo, jnp.float32),
                                              maxval=jnp.asarray(hi, jnp.float32)), np.float64)
            keep = p[np.asarray(self.classify_positions_exact(p)) == self.pool_index]
            if len(keep):
                i = np.floor((keep - self.origin) / self.voxel_size)
                lo = self.origin + i * self.voxel_size
                keep = np.clip(keep, lo + self._nudge, lo + self.voxel_size - self._nudge)
                out.append(keep[:need])
                need -= len(out[-1])
        return jnp.asarray(np.concatenate(out, axis=0), jnp.float32)

    # ------------------------------------------------------------------ the wall
    def _wall(self, r, step, kappa_over_D, rho_over_D, perm_key):
        """One step through the grid: traverse, reflect off every face to another pool, cross the
        first one if the membrane grants it.

        Returns ``(r_new, dlog_w, crossed, illegal)``. The traversal carries the distance already
        travelled implicitly in ``rem``, restarting from the hit point at each reflection, so the
        path length is conserved exactly. ``dlog_w`` is ``-2 (rho/D) * sum d_perp`` with
        ``d_perp = (rem - t) |u . n|`` the perpendicular overshoot past the face -- the same
        Brownstein-Tarr estimator every wall here accumulates, so ``rho/D`` means one thing.
        """
        step_l = jnp.linalg.norm(step)
        u0 = jnp.where(step_l > 0, step / jnp.maximum(step_l, jnp.float32(1e-30)), jnp.zeros(3, jnp.float32))
        u_rand = jax.random.uniform(perm_key, dtype=jnp.float32)
        nudge = jnp.float32(self._nudge)
        big = jnp.float32(np.finfo(np.float32).max)
        kappa_over_D = jnp.float32(kappa_over_D)
        rho_over_D = jnp.float32(rho_over_D)

        def dda(p, u):
            """The traversal state at ``p`` along ``u``: voxel index, step sign, distance to the next
            plane on each axis and the spacing between planes along ``u``."""
            i = self._index(p)
            sgn = jnp.where(u >= 0, 1, -1).astype(jnp.int32)
            au = jnp.abs(u)
            t_delta = jnp.where(au > 0, self.VOX / jnp.maximum(au, jnp.float32(1e-30)), big)
            # the next plane along each axis, measured from p
            wall = self.ORG + (jnp.floor((p - self.ORG) / self.VOX) + jnp.where(u >= 0, 1.0, 0.0)) * self.VOX
            t_next = jnp.where(au > 0, (wall - p) / jnp.where(au > 0, u, jnp.float32(1.0)), big)
            return i, sgn, jnp.maximum(t_next, 0.0), t_delta

        r_w = self._wrap(r)
        i0, sgn0, tn0, td0 = dda(r_w, u0)
        init = (r_w, u0, step_l, i0, sgn0, tn0, td0,
                jnp.float32(0.0), jnp.bool_(False), jnp.bool_(False), jnp.bool_(False), jnp.int32(0))

        def cond(c):
            _, _, _, _, _, _, _, _, done, _, _, n = c
            return (~done) & (n < self._max_events)

        def body(c):
            p, u, rem, i, sgn, t_next, t_delta, dlog, done, decided, crossed, n = c
            ax = jnp.argmin(t_next)
            t = t_next[ax]
            arrive = t >= rem                                      # the step ends before the next plane

            j = i.at[ax].add(sgn[ax])
            pool_here = self._pool_at_index(i)
            pool_far = self._pool_at_index(j)
            same = pool_far == pool_here

            # --- a face between two pools
            d_perp = (rem - t) * jnp.abs(u[ax])
            real_wall = pool_far >= 0                              # not the crop's own outer face
            p_t = transmit_probability(kappa_over_D, d_perp)
            transmit = (~same) & (~decided) & real_wall & (u_rand < p_t)
            reflect = (~same) & (~transmit)

            # advance into the next voxel (same pool, or a granted crossing)
            adv = same | transmit
            i_adv = j
            tn_adv = t_next.at[ax].add(t_delta[ax])

            # reflect: restart the traversal at the face, one component negated, a nudge clear of it
            # ALONG THE FACE NORMAL. Nudging along the reflected direction moves the walker off the
            # plane by nudge * |u_ax|, which vanishes at grazing incidence and leaves the position
            # rounding onto the far side of the face it just bounced off.
            u_ref = u.at[ax].multiply(-1.0)
            p_hit = p + t * u
            p_ref = p_hit.at[ax].add(-sgn[ax].astype(jnp.float32) * nudge)
            rem_ref = rem - t - nudge
            i_r, sgn_r, tn_r, td_r = dda(p_ref, u_ref)

            p_new = jnp.where(reflect, p_ref, p)
            u_new = jnp.where(reflect, u_ref, u)
            rem_new = jnp.where(reflect, rem_ref, rem)
            i_new = jnp.where(reflect, i_r, i_adv)
            sgn_new = jnp.where(reflect, sgn_r, sgn)
            tn_new = jnp.where(reflect, tn_r, tn_adv)
            td_new = jnp.where(reflect, td_r, t_delta)
            dlog_new = dlog - jnp.where(reflect & real_wall, 2.0 * rho_over_D * d_perp, jnp.float32(0.0))

            # arrival wins over everything: the plane is beyond the end of the step
            p_end = p + rem * u
            return (jnp.where(arrive, p_end, p_new),
                    jnp.where(arrive, u, u_new),
                    jnp.where(arrive, jnp.float32(0.0), jnp.maximum(rem_new, 0.0)),
                    jnp.where(arrive, i, i_new),
                    jnp.where(arrive, sgn, sgn_new),
                    jnp.where(arrive, t_next, tn_new),
                    jnp.where(arrive, t_delta, td_new),
                    jnp.where(arrive, dlog, dlog_new),
                    arrive | (rem_new <= 0.0),
                    decided | jnp.where(arrive, False, (~same) & real_wall),
                    crossed | jnp.where(arrive, False, transmit),
                    n + 1)

        p_f, _, _, i_f, _, _, _, dlog_f, done_f, _, crossed_f, _ = jax.lax.while_loop(cond, body, init)
        # The step's end, a nudge clear of every face of the voxel the traversal left it in. On a
        # periodic axis the position is continuous and the query is wrapped, so only the DELTA is
        # added back; on the others the clamped value is returned as it is, because `r + (x - r)` is
        # not `x` in float32 and putting the walker back within an ulp of a face is the very failure
        # the clamp exists to prevent.
        clamped = self._into_voxel(p_f, i_f)
        p_out = jnp.where(self.PER, r + (clamped - r_w), clamped)
        # Reject-escape: a step that ends in a pool other than the one it started in, without a
        # granted crossing, never happened. It is a net, not a guarantee: it reads the pool at the
        # END of the step, and the failure the nudge prevents corrupts the pool read at the START of
        # the NEXT one, where this test is blind. With `_into_voxel` disabled and this guard on, 5 of
        # 200,000 walkers still ended in a grain voxel, up to 5.3 voxels out. The nudge is the fix;
        # this catches what no argument about float32 has been made for.
        escaped = (self._pool_at_index(i_f) != self._pool_at_index(i0)) & ~crossed_f
        # The budget is unreachable for a step of a diffusion walk; a step that hits it has part of
        # its path untested, so it does not happen at all either.
        bad = (~done_f) | escaped
        # `illegal` is BOTH refusals, not the budget alone: a refused step is the engine holding a
        # walker still, which `PersistentWalk.illegal_crossings` and its warning exist to report.
        return (jnp.where(bad, r, p_out), jnp.where(bad, jnp.float32(0.0), dlog_f),
                crossed_f & ~bad, bad)

    def reflect(self, r, step):
        """Impermeable wall interaction -- the ``kappa = 0`` case of :meth:`_wall`."""
        return self._wall(r, step, jnp.float32(0.0), jnp.float32(0.0), jax.random.PRNGKey(0))[0]

    def reflect_with_log_weight(self, r, step, rho_over_D):
        """Impermeable wall interaction that also accrues the boundary local time."""
        r_new, dlog, _, _ = self._wall(r, step, jnp.float32(0.0), rho_over_D, jax.random.PRNGKey(0))
        return r_new, dlog

    def permeate(self, r, step, kappa_over_D, rho_over_D, perm_key):
        """Wall interaction with a permeable face (one Powles trial per step, at the first face met)."""
        return self._wall(r, step, kappa_over_D, rho_over_D, perm_key)

    def interact(self, r, step, *, kappa_over_D=0.0, rho_over_D=0.0, key=None, side=None):
        """One wall interaction, as a :class:`WallHit`."""
        if side is not None:
            raise NotImplementedError("LabelVolume reads the pool from the grid; it carries no side. Omit `side`.")
        k = key if key is not None else jax.random.PRNGKey(0)
        return WallHit(*self._wall(r, step, kappa_over_D, rho_over_D, k))
