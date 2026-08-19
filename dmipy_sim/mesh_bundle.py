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

    def _box(self, r):
        import jax.numpy as jnp
        r = jnp.where(r > self._hi, 2.0 * self._hi - r, r)
        r = jnp.where(r < self._lo, 2.0 * self._lo - r, r)
        return r

    def init_positions(self, n_walkers, key, intra=True):
        return self.mesh.init_positions(n_walkers, key, intra=intra)

    def classify_position(self, r):
        return self.mesh.classify_position(r)

    def reflect(self, r, step):
        return self._box(self.mesh.reflect(r, step))

    def reflect_with_log_weight(self, r, step, rho_over_D):
        r1, dlog = self.mesh.reflect_with_log_weight(r, step, rho_over_D)
        return self._box(r1), dlog


# ---------------------------------------------------------------- C4: MT
def _free_pool_geometry(bundle):
    """Per-pool ``(S/V, volume)`` for the two free-water pools, from the meshes themselves."""
    import trimesh
    A_in = float(trimesh.Trimesh(*bundle.inner, process=False).area)     # vertices already in metres
    A_out = float(trimesh.Trimesh(*bundle.outer, process=False).area)
    V_box = float(np.prod(bundle.box_side))
    V_i, V_e = bundle.f_intra * V_box, bundle.f_extra * V_box
    pools = {"intra": (A_in / V_i, V_i)} if V_i > 0 else {}
    if V_e > 0:
        pools["extra"] = (A_out / V_e, V_e)
    if not pools:
        raise ValueError("no free-water volume: both intra and extra fractions are zero")
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


# ---------------------------------------------------------------- the master walk
def mesh_bundle_master(bundle, *, n_walkers=30_000, params=None, T_max=0.04, dt_save=None, n_t=80,
                       seed=0, feature_radius_intra=None, feature_radius_extra=None, n_myelin=256,
                       require_gpu=None, walker_batch_size=50_000, field=False, field_res=0.2e-6,
                       include_extra=None, mt=False, kappa_MT=None, dwell_time=None,
                       nominal_T2=None, nominal_T1=None, nominal_delta_chi_a=None, verbose=True):
    """Walk the pools of ``bundle`` and return a master-walk dict (``bank._master_arrays`` schema).

    Parameters
    ----------
    n_walkers : int
        Total walkers, including the frozen myelin pool; the diffusing remainder is split between intra and
        extra by volume fraction.
    T_max, dt_save, n_t : float, float, int
        Walk length and save grid; ``dt_save`` defaults to ``T_max / n_t``. Together these set the replay
        envelope: ``T_max`` is the longest replayable TE and ``dt_save`` the finest replayable waveform
        feature. Both are baked in -- everything else about the acquisition is not.
    include_extra : bool, optional
        Defaults to the substrate's own ``has_extra_substrate``: a packed bundle's extra-axonal space carries
        structure, an isolated axon's surroundings are free water and carry none.
    kappa_MT, dwell_time : float, optional
        Left None (the default), both are derived from the catalogued white-matter qMT observables for this
        substrate's own geometry -- see :func:`wm_mt_parameters`. There is no meaningful default for
        ``kappa_MT`` alone, since it is a reactivity whose effect depends entirely on the surface it acts on.
    mt : bool or {'parametric', 'emergent'}
        C4 tier. ``'parametric'`` (True) stores :func:`bundle_mt_params` and leaves ``kappa_MT``/
        ``dwell_time`` replayable. ``'emergent'`` instead walks the diffusing pools with
        :func:`dmipy_sim.mt_walk.simulate_mt_trajectories`, so walkers bind and freeze at the walls and a
        per-walker ``bfrac`` channel is stored -- physically complete, but it BAKES those two parameters into
        the pack, since a frozen walker's trajectory depends on them.
    field : bool
        C3 tier: store the static myelin susceptibility field basis (geometry only), which replay samples
        along the reconstructed trajectory for any B0/orientation/susceptibility.
    nominal_T2, nominal_T1, nominal_delta_chi_a : optional
        Reference values written into the pack for certification and provenance only -- each is a REPLAY
        knob, so leaving them None costs nothing physical. Supplying ``nominal_T2`` lets
        ``build_replay_pack`` certify the C1 tier, which is otherwise skipped for want of a value to
        certify against. Order is (extra, intra, myelin), matching the compartment ids.
    """
    from .mesh import Mesh
    from .core import simulate_trajectories
    from .substrate.biophysical_constants import canonical_white_matter

    p = dict(canonical_white_matter())
    if params:
        p.update(params)
    if dt_save is None:
        dt_save = T_max / n_t
    if include_extra is None:
        include_extra = bool(getattr(bundle, "has_extra_substrate", True))
    mt_mode = None
    if mt:
        mt_mode = "parametric" if mt is True else str(mt)
        if mt_mode not in ("parametric", "emergent"):
            raise ValueError(f"mt must be False, 'parametric' or 'emergent', got {mt!r}")
        # Default to the catalogued white-matter qMT observables converted for THIS geometry, rather than a
        # pair of bare constants: kappa_MT is only meaningful relative to a substrate's own S/V.
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

    f_i, f_e = bundle.f_intra, bundle.f_extra
    n_diff = max(1, n_walkers - n_myelin)
    if include_extra:
        n_intra = max(1, int(round(n_diff * f_i / (f_i + f_e))))
        n_extra = max(1, n_diff - n_intra)
    else:
        n_intra, n_extra = n_diff, 0
    if verbose:
        print(f"[bundle] pools: intra={n_intra} extra={n_extra} myelin={n_myelin} | "
              f"fr_intra={fr_i/1e-6:.3f}um fr_extra={fr_e/1e-6:.3f}um | mt={mt_mode} | "
              f"{bundle.summary()}", flush=True)

    # No surface_relaxivity_t2: the trajectory path records the boundary channel at rho/D = 1 on its own,
    # and setting a magnitude here as well double-counts it (see the module note).
    mesh_in = Mesh(Vi, Fi, periodic=False, voxel_min=bundle.box_min, voxel_max=bundle.box_max,
                   feature_radius=fr_i)
    mesh_out = Mesh(Vo, Fo, periodic=False, voxel_min=bundle.box_min, voxel_max=bundle.box_max,
                    feature_radius=fr_e)

    def _walk(n, D, geom, sd, r0=None):
        """One pool, through the plain or the binding walk. Returns (traj, dlog_b, bfrac|None)."""
        if mt_mode == "emergent":
            from .mt_walk import simulate_mt_trajectories
            o = simulate_mt_trajectories(n, D, geom, T_max, dt_save, kappa_MT, dwell_time,
                                         seed=sd, walker_batch_size=walker_batch_size,
                                         require_gpu=require_gpu)
            # (traj, dt, sub_steps, dt_sim, bound_frac, dlog_boundary_unit)
            return (np.asarray(o[0], np.float64), np.asarray(o[5], np.float64),
                    np.asarray(o[4], np.float64))
        kw = dict(save_relaxation_data=True, seed=sd, require_gpu=require_gpu,
                  walker_batch_size=walker_batch_size)
        if r0 is not None:
            kw["r0"] = r0
        o = simulate_trajectories(n, D, geom, T_max=T_max, dt_save=dt_save, **kw)
        return np.asarray(o[0], np.float64), np.asarray(o[4], np.float64), None

    # ---- intra pool: restricted inside the inner wall (Mesh seeds its own interior exactly) ----
    tr_i, dlog_i, bf_i = _walk(n_intra, p["D_intra"], mesh_in, seed)
    n_t_actual = tr_i.shape[1]

    # ---- extra pool: hindered outside the outer wall, inside reflecting voxel walls ----
    if n_extra > 0:
        r0_e = _exterior_seeds(mesh_out, bundle.box_min, bundle.box_max, n_extra, seed)
        tr_e, dlog_e, bf_e = _walk(n_extra, p["D_extra"],
                                   BoxedMesh(mesh_out, bundle.box_min, bundle.box_max),
                                   seed + 7, r0=r0_e)
    else:
        tr_e = np.zeros((0, n_t_actual, 3)); dlog_e = np.zeros((0, n_t_actual)); bf_e = None

    # ---- myelin pool: frozen shell water (D = 0), so its trajectory is its seed ----
    r0_m = _shell_seeds(mesh_out, mesh_in, bundle.box_min, bundle.box_max, n_myelin, seed)
    tr_m = np.repeat(r0_m[:, None, :], n_t_actual, axis=1)
    dlog_m = np.zeros((n_myelin, n_t_actual))

    # ---- stack in comp-id order: extra, intra, myelin ----
    traj = np.concatenate([tr_e, tr_i, tr_m], axis=0)
    dlog_b = np.concatenate([dlog_e, dlog_i, dlog_m], axis=0)
    ids = np.concatenate([np.full(n_extra, EXTRA), np.full(n_intra, INTRA),
                          np.full(n_myelin, MYELIN)]).astype(np.int8)
    comp = np.repeat(ids[:, None], n_t_actual, axis=1)

    rho_m = p["myelin_water_proton_density"]
    w = np.concatenate([
        np.full(n_extra, (f_e * 1.0) / n_extra) if n_extra > 0 else np.zeros(0),
        np.full(n_intra, (bundle.f_intra * 1.0) / n_intra),
        np.full(n_myelin, (bundle.f_myelin * rho_m) / n_myelin),
    ]).astype(np.float64)

    out = dict(
        traj=traj, dt_traj=float(dt_save), T_max=float(T_max),
        comp=comp, comp0=ids.copy(), w=w, dlog_b=dlog_b,
        R=np.eye(3), D_intra=float(p["D_intra"]),
        n_walkers=int(traj.shape[0]), seed=int(seed),
    )
    # Nominal (provenance/certification) relaxation, omitted entirely when not supplied.
    if nominal_T2 is not None:
        out["T2_per_comp"] = np.asarray(nominal_T2, float)
    if nominal_T1 is not None:
        out["T1_per_comp"] = np.asarray(nominal_T1, float)

    # ---- intrinsic orientation frame: per-population mean axes, not a global average ----
    tang = getattr(bundle, "fibre_tangents", None)
    if tang is None or len(tang) == 0:
        tang = np.eye(3)[None, bundle.fibre_axis]
    ori = substrate_orientation(tang)
    if ori is not None:
        out["substrate_frame"] = ori["frame"].tolist()
        out["orientation"] = {k: v for k, v in ori.items() if k != "frame"}
        if verbose:
            ca = ori["crossing_angle_deg"]
            print(f"[bundle] orientation: {ori['n_populations']} population(s), counts={ori['counts']}"
                  + (f", crossing angle {ca}deg" if ca is not None else ""), flush=True)

    # ---- C3 field: the static myelin susceptibility field basis (geometry only) ----
    if field:
        from .susceptibility_field import mesh_field_basis
        if verbose:
            print(f"[bundle] building susceptibility field basis (res={field_res/1e-6:.2f}um)...",
                  flush=True)
        basis, origin, vs = mesh_field_basis(bundle.inner, bundle.outer, bundle.box_min, bundle.box_max,
                                             res=field_res, include_aniso=True)
        if nominal_delta_chi_a is not None:
            out["delta_chi_a"] = float(nominal_delta_chi_a)
        out["susc_field_basis"] = {
            "iso_local": np.asarray(basis["iso_local"], np.float32),
            "iso_P": np.asarray(basis["iso_P"], np.float32),
            "aniso_G": (np.asarray(basis["aniso_G"], np.float32)
                        if basis.get("aniso_G") is not None else None),
            "shape": tuple(int(s) for s in basis["shape"]),
            "voxel_size": np.asarray(vs, float)}
        out["susc_grid_origin"] = np.asarray(origin, float)
        if verbose:
            print(f"[bundle] susc field-grid channel: shape={basis['shape']}", flush=True)

    # ---- C4 MT ----
    if mt_mode == "parametric":
        out["mt_params"] = bundle_mt_params(bundle, kappa_MT, dwell_time)
        if verbose:
            m = out["mt_params"]
            print(f"[bundle] MT two-pool (parametric, per compartment): voxel f_bound="
                  f"{m['f_bound_voxel']:.4f}; "
                  + ", ".join(f"{k}: f_b {m['f_bound'][k]:.4f} k_f {m['k_forward'][k]:.2f}/s "
                              f"S/V {m['S_over_V'][k]:.3e}/m" for k in m['f_bound']), flush=True)
    elif mt_mode == "emergent":
        # The frozen myelin pool never binds (it does not move, so it never strikes a wall).
        bf = np.concatenate([
            bf_e if bf_e is not None else np.zeros((0, n_t_actual)),
            bf_i,
            np.zeros((n_myelin, n_t_actual)),
        ])
        out["bfrac"] = bf.astype(np.float32)
        out["mt_params"] = bundle_mt_params(bundle, kappa_MT, dwell_time)
        if verbose:
            occ = float(bf[:len(bf) - n_myelin].mean()) if len(bf) > n_myelin else 0.0
            print(f"[bundle] MT emergent: mean bound occupancy over the diffusing pools {occ:.4f} "
                  f"(per-pool parametric prediction, voxel "
                  f"{out['mt_params']['f_bound_voxel']:.4f}: "
                  f"{ {k: round(v, 4) for k, v in out['mt_params']['f_bound'].items()} })", flush=True)

    return out
