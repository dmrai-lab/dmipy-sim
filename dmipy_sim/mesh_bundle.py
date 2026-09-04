"""Three-pool master walk for an inner/outer surface-mesh fibre bundle (CACTUS, or any bundle).

A packed bundle is three compartments, and unlike an isolated axon (:mod:`dmipy_sim.mesh_axon`) all three
carry substrate information:

    intra   = inside the inner (axonal) surface        -- restricted, walked
    myelin  = the inner -> outer shell                 -- effectively stuck (D ~ 0), frozen, not walked
    extra   = the periodic box minus the outer surfaces -- hindered, walked

Walkers are split between the two diffusing pools in proportion to their volume fractions, stacked into one
master-walk dict (the ``bank._master_arrays`` schema), and weighted by volume fraction times proton density.
Compartment ids follow the packed-myelin convention: 0 extra, 1 intra, 2 myelin.

Substrate-agnostic by construction: it takes a loaded :class:`dmipy_sim.io.cactus.CactusBundle`, so anything
:mod:`dmipy_sim.io.mesh_substrate` can load works here, and a single-axon bundle
(``has_extra_substrate=False``) automatically drops the extra pool.
"""
from __future__ import annotations

import numpy as np

# comp ids (match the packed-myelin convention: 0=extra, 1=intra, 2=myelin)
EXTRA, INTRA, MYELIN = 0, 1, 2

# What the walk needs, and what it deliberately does NOT.
#
# Walker paths and boundary events depend only on (geometry, diffusivity, seed) plus the walk's own extent
# (T_max, dt_save) -- and, for emergent MT alone, on (kappa_MT, dwell_time), because a bound walker freezes.
# Everything else that shapes a signal is applied at REPLAY: the whole waveform, T2/T1 per compartment (C1),
# surface relaxivity rho (C2 -- the boundary channel is recorded at rho/D = 1 and scaled later), and the
# susceptibility strength/orientation/B0 (C3 -- the stored basis is geometry-only).
#
# So none of T1, T2, rho or chi is an input here. `nominal_*` arguments exist only to write a reference value
# into the pack for certification and provenance; they never touch the walk, and replay is free to ignore
# them. rho is not even set on the Mesh: the trajectory path passes rho/D = 1 itself (core.py), and setting a
# magnitude on the substrate as well is the documented way to double-count it.


# ---------------------------------------------------------------- orientation frame
def _principal_axis(T):
    """Dominant sign-free orientation of unit tangents ``T``: top eigenvector of the orientation tensor
    ``sum t_i t_i^T``, with a fixed sign convention (largest-|component| positive)."""
    _, V = np.linalg.eigh(T.T @ T)
    a = V[:, -1]
    return a * (1.0 if a[int(np.argmax(np.abs(a)))] >= 0 else -1.0)


def _population_axes(tangents, angle_tol_deg=30.0):
    """Greedy **axial** clustering of per-fibre tangents into bundle populations.

    Deliberately not a global PCA or average: for two crossing bundles that returns their bisector, which is
    no real bundle's axis. Pull out the dominant orientation, assign every fibre within ``angle_tol_deg`` of
    it (on ``|cos|``), refine that population's axis from its own members, repeat on the remainder. Returns
    ``(axes (k,3) unit, counts (k,))`` sorted by descending fibre count, so index 0 is the primary bundle.
    """
    T = np.asarray(tangents, float)
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    ct = np.cos(np.radians(angle_tol_deg))
    idx = np.arange(len(T)); axes = []; counts = []
    while len(idx):
        a = _principal_axis(T[idx])
        sel = idx[np.abs(T[idx] @ a) >= ct]
        if len(sel) == 0:                                  # tolerance too tight: take the closest single
            sel = idx[[int(np.argmax(np.abs(T[idx] @ a)))]]
        a = _principal_axis(T[sel])                        # refine from the cluster's own members
        axes.append(a); counts.append(int(len(sel)))
        idx = idx[~np.isin(idx, sel)]
    order = np.argsort(counts)[::-1]
    return np.array([axes[i] for i in order]), np.array([counts[i] for i in order])


def substrate_orientation(tangents, angle_tol_deg=30.0):
    """Substrate frame + orientation fingerprint from per-fibre tangents.

    Groups fibres into populations and anchors the frame to the LARGEST one (primary -> z, secondary -> +y),
    so a crossing substrate keeps a shared reference rather than one that drifts with the mix. Returns a dict
    with ``frame`` (3x3), ``n_populations``, ``counts``, ``fibre_orientations`` (each population axis in
    canonical coordinates, z = primary) and ``crossing_angle_deg`` (None for a single population), or None if
    no tangents are available.
    """
    if tangents is None or len(tangents) == 0:
        return None
    from .bank import frame_from_bundles
    axes, counts = _population_axes(tangents, angle_tol_deg)
    R = frame_from_bundles(axes, primary=0)                # axes sorted desc => primary = largest
    canon = (R.T @ axes.T).T
    crossing = None
    if len(axes) >= 2:
        crossing = round(float(np.degrees(np.arccos(min(1.0, abs(float(axes[0] @ axes[1])))))), 1)
    return dict(frame=R, n_populations=int(len(axes)), counts=[int(c) for c in counts],
                fibre_orientations=[[round(float(x), 4) for x in v] for v in canon],
                crossing_angle_deg=crossing)


# ---------------------------------------------------------------- geometry helpers
def _min_radius(V, F):
    """A cheap feature radius: half the median edge length of the mesh."""
    e = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
    return 0.5 * float(np.median(e))


def _exact_labels(mesh, pts):
    """Interior(0)/exterior(1) for host-side points, decided EXACTLY.

    Uses :meth:`Mesh.classify_positions_exact`, which takes the cell-gather answer where the gather can
    decide and falls back to ray-parity where it cannot. The raw classifier alone is not usable for seeding
    a bundle: it calls any point with an empty 27-cell gather *exterior*, and the gather reach scales with
    the TRIANGLE size rather than the lumen, so the deep interior of a wide fibre reads as outside it
    (measured 49.3% of intra-axonal volume on a real CACTUS bundle, 19.6% on a 366-strand axon set,
    dmipy-sim#33/#50). Seeding the extra pool on that answer fills it with intra-axonal walkers.
    """
    return np.asarray(mesh.classify_positions_exact(np.asarray(pts, float)))


def _exterior_seeds(mesh_out, box_min, box_max, n, seed):
    """Uniform box points exterior to every outer axon surface (the extra pool)."""
    rng = np.random.default_rng(seed)
    keep, got = [], 0
    while got < n:
        p = rng.uniform(box_min, box_max, (max(2 * (n - got), 8192), 3))
        ext = p[_exact_labels(mesh_out, p) == 1]
        keep.append(ext); got += len(ext)
    return np.vstack(keep)[:n]


def _shell_seeds(mesh_out, mesh_in, box_min, box_max, n, seed):
    """Uniform box points inside an outer surface but outside the matching inner one (the myelin pool)."""
    rng = np.random.default_rng(seed + 1)
    keep, got = [], 0
    while got < n:
        p = rng.uniform(box_min, box_max, (max(6 * (n - got), 8192), 3))
        shell = p[(_exact_labels(mesh_out, p) == 0) & (_exact_labels(mesh_in, p) == 1)]
        keep.append(shell); got += len(shell)
    return np.vstack(keep)[:n]


class BoxedMesh:
    """A :class:`dmipy_sim.mesh.Mesh` wrapped in REFLECTING voxel walls.

    CACTUS bundles are finite and non-periodic, so an extra-axonal walker would simply leave the box, and the
    engine's ``reject_escape`` net would then pin it against the boundary and bias the extra signal. This
    composes the axon mesh with specular reflection at the voxel faces -- the standard representative-volume
    boundary. Box reflection carries **no** surface log-weight (only the axon walls do), so the surface
    channel stays myelin-only.
    """

    def __init__(self, mesh, box_min, box_max):
        import jax.numpy as jnp
        self.mesh = mesh
        self._lo = jnp.asarray(box_min, jnp.float32)
        self._hi = jnp.asarray(box_max, jnp.float32)
        self.radius = mesh.radius                       # the sub-step auto-tune reads this
        self.cell_size = getattr(mesh, "cell_size", None)   # ...and collision_sub_steps reads this
        self.permeability = mesh.permeability
        self.surface_relaxivity_t2 = mesh.surface_relaxivity_t2
        self.reject_escape = True

    def _mirror(self, r):
        """Raw specular mirror of a position back into the box."""
        import jax.numpy as jnp
        r = jnp.where(r > self._hi, 2.0 * self._hi - r, r)
        r = jnp.where(r < self._lo, 2.0 * self._lo - r, r)
        return r

    def _box(self, r_in, r_out):
        """Mirror ``r_out`` back into the box, but never THROUGH a wall (dmipy-sim#61).

        The mirror is a reflection of space, not a displacement, so it is applied without any collision test.
        On a tortuous bundle that is a hole in the confinement: mirroring changes which fibre cross-section a
        point falls in, so it can place a walker inside a fibre with no wall ever being crossed. Measured on
        the 366-fibre CACTUS bundle, 47.4% of extra-axonal walkers ended up inside a fibre within 10 ms, and
        the leaked population clustered at the faces (median 0.933 um from the nearest, against 3.75 um for a
        uniform one).

        The mirror is therefore vetoed when it would change the compartment. That veto is trustworthy exactly
        where it has to be: a mirrored walker lands JUST inside a surface, and the cell-gather classifier is
        100% correct for depths below 0.2 um (falling to 16.4% at 0.4-0.8 um and 1.3% beyond, which is why the
        resident state cannot be policed the same way -- see #61). Vetoing leaves the walker outside the box
        for that step; it is pulled back on a later one, which is a far smaller error than tunnelling into the
        wrong compartment and being trapped there.

        This is a mitigation, not the design fix. Properly, the box faces should take part in the same bounce
        loop as the mesh triangles so a face reflection is clipped and tested like any other collision.
        """
        import jax.numpy as jnp
        mirrored = self._mirror(r_out)
        moved = jnp.any(mirrored != r_out)
        crossed = self.mesh.classify_position(mirrored) != self.mesh.classify_position(r_in)
        if getattr(self.mesh, "box_reflect", False):
            # The inner Mesh now reflects at the voxel faces INSIDE its bounce loop, which is the design fix
            # this mirror was a stand-in for (see Mesh.box_reflect). Mirroring on top would be redundant at
            # best -- a confined walker never leaves the box, so there is nothing to mirror -- and it is the
            # very operation that placed walkers inside fibres without crossing a wall. So: do nothing.
            return r_out
        return jnp.where(moved & crossed, r_out, mirrored)

    def init_positions(self, n_walkers, key, intra=True):
        return self.mesh.init_positions(n_walkers, key, intra=intra)

    def classify_position(self, r):
        return self.mesh.classify_position(r)

    def reflect(self, r, step):
        return self._box(r, self.mesh.reflect(r, step))

    def reflect_with_log_weight(self, r, step, rho_over_D):
        r1, dlog = self.mesh.reflect_with_log_weight(r, step, rho_over_D)
        return self._box(r, r1), dlog


# ---------------------------------------------------------------- C4: MT
def _box_clipped_area(V, F, box_min, box_max):
    """Surface area of ``(V, F)`` restricted to the box, via an UNCAPPED clip.

    Uncapped on purpose: the faces a clip introduces on the cut planes are not real surface, and counting
    them would inflate S. Uses :mod:`trimesh` rather than reimplementing polygon clipping.
    """
    import trimesh
    m = trimesh.Trimesh(np.asarray(V, float), np.asarray(F, np.int64), process=False)
    lo, hi = np.asarray(box_min, float), np.asarray(box_max, float)
    for k in range(3):
        for origin, normal in ((lo, np.eye(3)[k]), (hi, -np.eye(3)[k])):
            if len(m.faces) == 0:
                return 0.0
            m = trimesh.intersections.slice_mesh_plane(m, plane_normal=normal, plane_origin=origin,
                                                      cap=False)
    return float(m.area)


def _free_pool_geometry(bundle, *, containment="fast", n_probe=200_000, seed=99):
    """Per-pool ``(S/V, volume)`` for the free-water pools, with S and V over the SAME region.

    Both quantities are taken in-box. That is not pedantry: CACTUS fibres are finite and their end caps
    overrun the periodic cell, so only 94.7% of the inner surface and 93.0% of the outer lie inside the
    volume the walkers actually sample. Dividing the FULL mesh area by an in-box volume overstates the intra
    S/V by 15% -- and since ``k_f = kappa_MT * (S/V)``, that made the analytic bound fraction unreachable:
    the measured boundary local time came out at 0.851 of the mismatched prediction and 0.982 of this one.

    Volumes are sampled with the containment predicates rather than taken from the loader's mesh-volume
    fractions, which use a different reference again (cross-section times fibre extent, to strip that same
    overhang). Sampling keeps the denominator in the same units as the numerator and as the seeding.
    """
    inside_in, inside_out = containment_predicates(bundle, containment)
    lo, hi = np.asarray(bundle.box_min, float), np.asarray(bundle.box_max, float)
    V_box = float(np.prod(hi - lo))
    q = np.random.default_rng(seed).uniform(lo, hi, (int(n_probe), 3))
    pin, pout = inside_in(q), inside_out(q)
    V_i = float(pin.mean()) * V_box
    V_e = float((~pout).mean()) * V_box
    A_in = _box_clipped_area(*bundle.inner, lo, hi)
    A_out = _box_clipped_area(*bundle.outer, lo, hi)
    pools = {}
    if V_i > 0:
        pools["intra"] = (A_in / V_i, V_i)
    if V_e > 0 and getattr(bundle, "has_extra_substrate", True):
        pools["extra"] = (A_out / V_e, V_e)
    if not pools:
        raise ValueError("no free-water volume: both intra and extra fractions sampled as zero")
    return pools


def bundle_mt_params(bundle, kappa_MT, dwell_time):
    """Two-pool qMT parameters, **per compartment**.

    ``kappa_MT`` is a material surface reactivity (m/s), the same on every wall, so a pool's exchange rate is
    ``k_f = kappa_MT * (S/V)`` for ITS OWN surface-to-volume ratio. With impermeable walls the pools never
    exchange with each other -- an intra walker only ever meets the inner surface inside the intra volume, an
    extra walker only the outer surface inside the extra volume -- so there is no single ``f_bound`` for the
    substrate, and the asymmetry between the pools is a PREDICTION of the geometry rather than an input.

    This previously formed one bundle-average ``S/V = (A_in + A_out)/(V_intra + V_extra)``, which is the
    well-mixed answer and wrong here: measured against an emergent binding walk on an 8-strand CACTUS subset,
    the bundle-average predicted f_bound = 0.0042 where the intra pool's own S/V predicted 0.0349 against a
    measured 0.0388 (11% agreement). The average was 8x low.

    S/V comes from :func:`_free_pool_geometry`, which takes area and volume over the same in-box region --
    see there for why the obvious full-area version is 15% out on a bundle whose fibres overrun the cell.

    ``f_bound_voxel`` is the volume-weighted mean, which is the thing comparable to a qMT measurement of a
    voxel. Note the weighting counts only the free pools: a frozen myelin-water pool has D = 0, never strikes
    a wall and so never binds.
    """
    from .mt import forward_rate, bound_fraction
    pools = _free_pool_geometry(bundle)
    sv = {k: v[0] for k, v in pools.items()}
    vol = {k: v[1] for k, v in pools.items()}
    tot = sum(vol.values())
    f_bound = {k: float(bound_fraction(kappa_MT, dwell_time, sv[k])) for k in pools}
    k_fwd = {k: float(forward_rate(kappa_MT, sv[k])) for k in pools}
    return dict(model="two_pool_qMT", kappa_MT=float(kappa_MT), dwell_time=float(dwell_time),
                S_over_V=sv, f_bound=f_bound, k_forward=k_fwd,
                f_bound_voxel=float(sum(f_bound[k] * vol[k] for k in pools) / tot),
                volume_weights={k: float(vol[k] / tot) for k in pools})


def kappa_MT_for_voxel_f_bound(bundle, f_bound_voxel, dwell_time, *, tol=1e-10):
    """Solve for the single ``kappa_MT`` whose volume-weighted free-pool bound fraction hits a target.

    One reactivity has to serve both pools, so a target can only be met on the voxel average; each pool then
    lands where its own S/V puts it. Monotone in kappa_MT, so bisected in log space.
    """
    from .mt import bound_fraction
    pools = _free_pool_geometry(bundle)
    tot = sum(v[1] for v in pools.values())

    def voxel(kappa):
        return sum(bound_fraction(kappa, dwell_time, sv) * vol for sv, vol in pools.values()) / tot

    lo, hi = 1e-12, 1.0
    if voxel(hi) < f_bound_voxel:
        raise ValueError(f"target f_bound {f_bound_voxel} unreachable for this geometry even at kappa_MT=1")
    while hi / lo > 1.0 + tol:
        mid = np.sqrt(lo * hi)
        if voxel(mid) < f_bound_voxel:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def wm_mt_parameters(bundle, *, field_T=3.0, f_bound=None, exchange_rate_R=None):
    """``(kappa_MT, dwell_time)`` for this substrate from the catalogued white-matter qMT observables.

    qMT reports the macromolecular pool fraction ``M0B`` and the fundamental exchange rate ``R``, not the
    pair the walk needs. The pseudo-first-order rates follow as ``k_f = R*M0B`` and ``k_r = R*M0A``
    (``M0A = 1 - M0B``), so ``dwell_time = 1/k_r`` is a material property shared by both pools, while
    ``kappa_MT`` is solved so the volume-weighted free-pool bound fraction equals ``M0B``.

    Defaults are Stanisz et al. 2005 white matter at 3 T (``M0B = 13.9 +/- 2.8 %``, ``R = 23 +/- 4 /s``),
    which are bovine and in vitro at 37 C -- a canonical reference, not in vivo human. Propagating those
    uncertainties moves ``kappa_MT`` by about +/-20% and ``dwell_time`` between roughly 43 and 61 ms.
    """
    from .substrate.biophysical_constants import canonical_white_matter
    p = canonical_white_matter(field_T)
    M0B = float(p["mt_bound_pool_fraction"] if f_bound is None else f_bound)
    R = float(p["mt_exchange_rate"] if exchange_rate_R is None else exchange_rate_R)
    if not (0.0 < M0B < 1.0):
        raise ValueError(f"bound pool fraction must be in (0,1), got {M0B}")
    dwell_time = 1.0 / (R * (1.0 - M0B))                     # 1/k_r
    return kappa_MT_for_voxel_f_bound(bundle, M0B, dwell_time), dwell_time


# ---------------------------------------------------------------- seeding
def _rejection_seeds(pred, box_min, box_max, n, seed, oversample=4):
    """Uniform points in the box accepted by ``pred(pts) -> bool[]``.

    Returns ``(pts[:n], acceptance_fraction)`` -- and that fraction IS the compartment's volume fraction of
    the box, measured on the same points that seeded it rather than taken from a separate mesh-volume
    estimate.
    """
    rng = np.random.default_rng(seed)
    keep, tried, acc = [], 0, 0
    while acc < n:
        q = rng.uniform(box_min, box_max, (max(oversample * (n - acc), 20000), 3))
        m = pred(q); tried += len(q); acc += int(m.sum())
        keep.append(q[m])
    return np.vstack(keep)[:n], float(acc) / float(tried)


def containment_predicates(bundle, mode="fast"):
    """``(inside_inner, inside_outer)`` predicates for host-side points.

    ``'exact'`` is ray-crossing parity (:func:`mesh_contains`) -- the reference, and the only sound choice
    for a thin object in a large box, where a nearest-surface test accepts points far outside a tortuous
    lumen. It costs about 21 ms PER POINT on a 366-fibre bundle without an embree backend, so seeding ~90k
    walkers with it takes hours: that is what it is, not a tuning knob.

    ``'fast'`` is the grid-based nearest-surface test (:func:`mesh_inside`), ~1000x quicker and effectively
    fixed-cost. Its documented failure is a FAR field, which a densely packed bundle does not have -- every
    point sits within about a micron of some surface. Measured on the 366-fibre CACTUS bundle at ICVF 0.61,
    it disagrees with parity on well under 1% of points. Validate with
    :func:`compare_containment` on any new substrate before trusting it: on a sparse one it will be wrong.
    """
    from .susceptibility_field import mesh_contains, mesh_inside
    fn = mesh_contains if mode == "exact" else mesh_inside
    if mode not in ("fast", "exact"):
        raise ValueError(f"containment must be 'fast' or 'exact', got {mode!r}")
    Vi, Fi = bundle.inner
    Vo, Fo = bundle.outer
    return (lambda q: np.asarray(fn(Vi, Fi, np.asarray(q, float))),
            lambda q: np.asarray(fn(Vo, Fo, np.asarray(q, float))))


def compare_containment(bundle, n=5000, seed=11):
    """Disagreement between the fast and exact containment tests, per surface.

    Run this before seeding a NEW substrate with ``containment='fast'``: the fast test's validity is a
    property of how densely the box is filled, not of the code.
    """
    from .susceptibility_field import mesh_contains, mesh_inside
    rng = np.random.default_rng(seed)
    q = rng.uniform(bundle.box_min, bundle.box_max, (int(n), 3))
    out = {}
    for name, (V, F) in (("inner", bundle.inner), ("outer", bundle.outer)):
        fast = np.asarray(mesh_inside(V, F, q)); exact = np.asarray(mesh_contains(V, F, q))
        out[name] = dict(disagreement=float((fast != exact).mean()),
                         false_inside=int((fast & ~exact).sum()),
                         false_outside=int((~fast & exact).sum()),
                         frac_fast=float(fast.mean()), frac_exact=float(exact.mean()))
    return out


# ---------------------------------------------------------------- the master walk
def mesh_bundle_master(bundle, *, n_walkers=30_000, params=None, T_max=0.04, dt_save=None, n_t=80,
                       seed=0, feature_radius_intra=None, feature_radius_extra=None,
                       require_gpu=None, walker_batch_size=50_000, field=False, field_res=0.2e-6,
                       include_extra=None, mt=False, kappa_MT=None, dwell_time=None,
                       myelin_water_proton_density=None, containment="fast", n_probe=200_000,
                       nominal_T2=None, nominal_T1=None, nominal_delta_chi_a=None, verbose=True):
    """Walk the pools of ``bundle`` and return a master-walk dict (``bank._master_arrays`` schema).

    **Uniform density.** ``n_walkers`` is the TOTAL spin count, seeded at uniform density through the box:
    the volume fractions are MEASURED on one common probe set and the budget split in proportion, so a
    walker represents the same tissue volume wherever it sits. Myelin's lower water content is then applied
    by THINNING -- keep a random ``myelin_water_proton_density`` fraction of the sheath spins and drop the
    rest -- so every surviving spin matches the others in both volume and water content and the ensemble is
    unweighted. An exact count is kept rather than a Bernoulli draw, which would add binomial noise to a
    quantity known exactly.

    This matters beyond tidiness: unequal per-walker weights reduce the effective sample size of the
    weighted mean, so an unweighted ensemble reaches a given Monte-Carlo floor with fewer walkers.

    Parameters
    ----------
    T_max, dt_save, n_t
        Walk length and save grid (``dt_save`` defaults to ``T_max / n_t``). These set the replay envelope:
        ``T_max`` is the longest replayable TE, ``dt_save`` the finest replayable waveform feature. Both are
        baked in; nothing else about the acquisition is.
    containment : {'fast', 'exact'}
        See :func:`containment_predicates`. ``'fast'`` is sound on a densely packed bundle and about 1000x
        quicker; check a new substrate with :func:`compare_containment` first.
    kappa_MT, dwell_time : float, optional
        Both None (the default) derives them from the catalogued white-matter qMT observables for this
        geometry -- see :func:`wm_mt_parameters`.
    mt : bool or {'parametric', 'emergent'}
        ``'parametric'`` stores :func:`bundle_mt_params` and leaves the MT parameters replayable;
        ``'emergent'`` walks with binding, which BAKES them in, and stores a per-walker ``bfrac``.
    nominal_T2, nominal_T1, nominal_delta_chi_a : optional
        Reference values for certification/provenance only -- each is a replay knob, so None costs nothing
        physical. ``nominal_T2`` lets ``build_replay_pack`` certify the C1 tier. Order (extra, intra, myelin).
    """
    from .geometry.mesh import Mesh
    from .core import simulate_trajectories
    from .substrate.biophysical_constants import canonical_white_matter

    p = dict(canonical_white_matter())
    if params:
        p.update(params)
    if dt_save is None:
        dt_save = T_max / n_t
    if include_extra is None:
        include_extra = bool(getattr(bundle, "has_extra_substrate", True))
    rho_m = float(p["myelin_water_proton_density"] if myelin_water_proton_density is None
                  else myelin_water_proton_density)
    mt_mode = None
    if mt:
        mt_mode = "parametric" if mt is True else str(mt)
        if mt_mode not in ("parametric", "emergent"):
            raise ValueError(f"mt must be False, 'parametric' or 'emergent', got {mt!r}")
        if kappa_MT is None or dwell_time is None:
            k_lit, d_lit = wm_mt_parameters(bundle)
            kappa_MT = k_lit if kappa_MT is None else kappa_MT
            dwell_time = d_lit if dwell_time is None else dwell_time
            if verbose:
                print(f"[bundle] MT from catalogued WM qMT (Stanisz 2005 @3T): "
                      f"kappa_MT={kappa_MT:.4e} m/s, dwell={dwell_time*1e3:.2f} ms", flush=True)

    Vi, Fi = bundle.inner
    Vo, Fo = bundle.outer
    fr_i = feature_radius_intra or _min_radius(Vi, Fi)
    fr_e = feature_radius_extra or _min_radius(Vo, Fo)
    inside_in, inside_out = containment_predicates(bundle, containment)

    # ---- split the spin budget by MEASURED volume fractions, on one common probe set ----
    rng0 = np.random.default_rng(seed + 99)
    probe = rng0.uniform(bundle.box_min, bundle.box_max, (int(n_probe), 3))
    pin, pout = inside_in(probe), inside_out(probe)
    f_i_pre = float(pin.mean())
    f_m_pre = float((pout & ~pin).mean())
    f_e_pre = float((~pout).mean()) if include_extra else 0.0
    tot = f_i_pre + f_m_pre + f_e_pre
    n_intra = max(1, int(round(n_walkers * f_i_pre / tot)))
    n_myelin_seeded = max(1, int(round(n_walkers * f_m_pre / tot)))
    n_extra = max(1, int(round(n_walkers * f_e_pre / tot))) if include_extra else 0
    if verbose:
        print(f"[bundle] uniform density ({containment} containment, {int(n_probe):,} probes): "
              f"f_intra={f_i_pre:.4f} f_myelin={f_m_pre:.4f} f_extra={f_e_pre:.4f} -> "
              f"n_intra={n_intra} n_myelin={n_myelin_seeded} n_extra={n_extra}", flush=True)
        print(f"[bundle] fr_intra={fr_i/1e-6:.3f}um fr_extra={fr_e/1e-6:.3f}um | mt={mt_mode} | "
              f"{bundle.summary()}", flush=True)

    mesh_in = Mesh(Vi, Fi, periodic=False, voxel_min=bundle.box_min, voxel_max=bundle.box_max,
                   feature_radius=fr_i)
    mesh_out = Mesh(Vo, Fo, periodic=False, voxel_min=bundle.box_min, voxel_max=bundle.box_max,
                    feature_radius=fr_e)

    def _walk(n, D, geom, sd, r0):
        """One pool, plain or binding walk -> (traj, dlog_b, bfrac|None), float32.

        float32 rather than float64: positions are stored float16 on device, so upcasting past float32
        inflates the master 4x for no information.
        """
        if mt_mode == "emergent":
            from .mt_walk import simulate_mt_trajectories
            o = simulate_mt_trajectories(n, D, geom, T_max, dt_save, kappa_MT, dwell_time, seed=sd,
                                         r0=r0, walker_batch_size=walker_batch_size,
                                         require_gpu=require_gpu)
            return (np.asarray(o[0], np.float32), np.asarray(o[5], np.float32),
                    np.asarray(o[4], np.float32))
        o = simulate_trajectories(n, D, geom, T_max=T_max, dt_save=dt_save, seed=sd, r0=r0,
                                  save_relaxation_data=True, require_gpu=require_gpu,
                                  walker_batch_size=walker_batch_size)
        return np.asarray(o[0], np.float32), np.asarray(o[4], np.float32), None

    # ---- intra: restricted inside the inner wall. Seeds passed explicitly, never left to the cell-gather
    # classifier, which calls a deep-interior point exterior wherever its 27-cell gather is empty (49.3% of
    # intra volume here) and would hug the seeds to the wall. ----
    r0_i, f_i = _rejection_seeds(inside_in, bundle.box_min, bundle.box_max, n_intra, seed)
    tr_i, dlog_i, bf_i = _walk(n_intra, p["D_intra"], mesh_in, seed, r0_i)
    n_t_actual = tr_i.shape[1]

    # ---- extra: hindered outside the outer wall, inside reflecting voxel walls ----
    if n_extra > 0:
        r0_e, f_e = _rejection_seeds(lambda q: ~inside_out(q), bundle.box_min, bundle.box_max,
                                     n_extra, seed + 7)
        tr_e, dlog_e, bf_e = _walk(n_extra, p["D_extra"],
                                   BoxedMesh(mesh_out, bundle.box_min, bundle.box_max), seed + 7, r0_e)
    else:
        f_e = 0.0
        tr_e = np.zeros((0, n_t_actual, 3), np.float32); dlog_e = np.zeros((0, n_t_actual), np.float32)
        bf_e = None

    # ---- myelin: frozen sheath water (D = 0), so its trajectory is its seed ----
    r0_m, f_m = _rejection_seeds(lambda q: inside_out(q) & ~inside_in(q),
                                 bundle.box_min, bundle.box_max, n_myelin_seeded, seed + 1)
    n_myelin = max(1, int(round(rho_m * n_myelin_seeded)))
    pick = np.random.default_rng(seed + 7).permutation(n_myelin_seeded)[:n_myelin]
    r0_m = r0_m[np.sort(pick)]
    if verbose:
        print(f"[bundle] myelin water content by thinning: kept {n_myelin}/{n_myelin_seeded} "
              f"(rho={rho_m}) -> unweighted ensemble", flush=True)
    tr_m = np.repeat(r0_m[:, None, :].astype(np.float32), n_t_actual, axis=1)
    dlog_m = np.zeros((n_myelin, n_t_actual), np.float32)

    # ---- stack (extra, intra, myelin) ----
    traj = np.concatenate([tr_e, tr_i, tr_m], axis=0)
    dlog_b = np.concatenate([dlog_e, dlog_i, dlog_m], axis=0)
    ids = np.concatenate([np.full(n_extra, EXTRA), np.full(n_intra, INTRA),
                          np.full(n_myelin, MYELIN)]).astype(np.int8)

    # Per-walker weight = (volume represented) x (proton density). Under the uniform-density design the
    # volume per walker is the same in every pool, and thinning has already applied the myelin density, so
    # all weights coincide -- the spread is reported precisely so a regression away from 1.000 is visible.
    vol_e = (f_e / n_extra) if n_extra > 0 else 0.0
    vol_i = f_i / n_intra
    vol_m = f_m / n_myelin_seeded          # volume per SEEDED sheath spin: what the survivors represent
    w = np.concatenate([np.full(n_extra, vol_e), np.full(n_intra, vol_i),
                        np.full(n_myelin, vol_m)]).astype(np.float64)
    if verbose:
        spread = float(w.max() / w.min()) if w.min() > 0 else float("nan")
        print(f"[bundle] measured fractions: intra={f_i:.4f} myelin={f_m:.4f} extra={f_e:.4f} | "
              f"weight spread max/min={spread:.4f} (1.0000 = unweighted)", flush=True)

    # ---- deterministic shuffle so any walker PREFIX is a valid sub-ensemble ----
    # Precision tiers read the first n rows of the walker-leading arrays. Stacked pool-by-pool those rows
    # are all extra-axonal, so a prefix read would silently return a substrate with no intra and no myelin.
    order = np.random.default_rng(int(seed) + 991).permutation(traj.shape[0])
    traj, dlog_b, ids, w = traj[order], dlog_b[order], ids[order], w[order]
    comp = np.repeat(ids[:, None], n_t_actual, axis=1)

    out = dict(
        traj=traj, dt_traj=float(dt_save), T_max=float(T_max),
        comp=comp, comp0=ids.copy(), w=w, dlog_b=dlog_b,
        R=np.eye(3), D_intra=float(p["D_intra"]),
        n_walkers=int(traj.shape[0]), seed=int(seed),
    )
    if nominal_T2 is not None:
        out["T2_per_comp"] = np.asarray(nominal_T2, float)
    if nominal_T1 is not None:
        out["T1_per_comp"] = np.asarray(nominal_T1, float)

    tang = getattr(bundle, "fibre_tangents", None)
    if tang is None or len(tang) == 0:
        tang = np.eye(3)[None, bundle.fibre_axis]
    ori = substrate_orientation(tang)
    if ori is not None:
        out["substrate_frame"] = ori["frame"].tolist()
        out["orientation"] = {k: v for k, v in ori.items() if k != "frame"}

    if field:
        from .susceptibility_field import mesh_field_basis
        basis, origin, vs = mesh_field_basis(bundle.inner, bundle.outer, bundle.box_min, bundle.box_max,
                                             res=field_res, include_aniso=True)
        out["susc_field_basis"] = {
            "iso_local": np.asarray(basis["iso_local"], np.float32),
            "iso_P": np.asarray(basis["iso_P"], np.float32),
            "aniso_G": (np.asarray(basis["aniso_G"], np.float32)
                        if basis.get("aniso_G") is not None else None),
            "shape": tuple(int(s) for s in basis["shape"]),
            "voxel_size": np.asarray(vs, float)}
        out["susc_grid_origin"] = np.asarray(origin, float)
        if nominal_delta_chi_a is not None:
            out["delta_chi_a"] = float(nominal_delta_chi_a)

    if mt_mode == "parametric":
        out["mt_params"] = bundle_mt_params(bundle, kappa_MT, dwell_time)
        if verbose:
            m = out["mt_params"]
            print(f"[bundle] MT (parametric, per compartment): voxel f_bound={m['f_bound_voxel']:.4f}; "
                  + ", ".join(f"{k}: f_b {m['f_bound'][k]:.4f} k_f {m['k_forward'][k]:.2f}/s"
                              for k in m["f_bound"]), flush=True)
    elif mt_mode == "emergent":
        bf = np.concatenate([
            bf_e if bf_e is not None else np.zeros((0, n_t_actual), np.float32),
            bf_i, np.zeros((n_myelin, n_t_actual), np.float32)])[order]
        out["bfrac"] = bf.astype(np.float32)
        out["mt_params"] = bundle_mt_params(bundle, kappa_MT, dwell_time)
        if verbose:
            occ = float(bf[ids != MYELIN].mean())
            print(f"[bundle] MT emergent: bound occupancy over the diffusing pools {occ:.4f} "
                  f"(per-pool analytic voxel {out['mt_params']['f_bound_voxel']:.4f})", flush=True)

    return out
