"""Master-walk builder for a single myelinated axon mesh (intra + frozen myelin), with the static
susceptibility field-grid channel — the producer for a Winther-style susceptibility replay pack.

Walks the intra-axonal water inside the inner (axonal) surface, represents the myelin water as a
FROZEN pool (D=0) seeded in the sheath and weighted by its water content, builds the geometry-only
susceptibility field basis once (:func:`dmipy_sim.susceptibility_field.mesh_field_basis`), and returns
a master-walk dict for :func:`dmipy_sim.bank.build_replay_pack` (``field_store='grid'``). No extra-
axonal pool (an isolated axon's surroundings are free water carrying no substrate information) and no
magnetization transfer — the minimal physical axon: restricted intra + static internal gradients.

Cited defaults follow Winther et al. 2024 (doi:10.1038/s41598-024-79043-5): D0 = 0.6e-9 m^2/s (ex vivo),
myelin-water contrast chi_iso = +1.06e-6 (isotropic), myelin frozen with short T2, weighted by the
myelin-water proton density.
"""
from __future__ import annotations

import numpy as np

from .susceptibility_field import mesh_contains, mesh_field_basis

INTRA, MYELIN = 1, 2                      # compartment ids (0 = extra, unused here)


def _min_radius(V, F):
    """Cheap feature radius: half the median mesh edge length (sets the sub-step)."""
    e = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
    return 0.5 * float(np.median(e))


def _rejection_seeds(pred, box_min, box_max, n, seed, oversample=4):
    """Uniform points in the box accepted by ``pred(pts) -> bool[]`` (rejection sampling).
    Also returns the measured acceptance fraction = that compartment's volume fraction of the box."""
    rng = np.random.default_rng(seed)
    keep, tried, acc = [], 0, 0
    while acc < n:
        p = rng.uniform(box_min, box_max, (max(oversample * (n - acc), 20000), 3))
        m = pred(p); tried += len(p); acc += int(m.sum())
        keep.append(p[m])
    pts = np.vstack(keep)
    return pts[:n], float(acc) / float(tried)


def mesh_axon_master(bundle, *, n_walkers=30_000, n_myelin=None, n_t=1600, T_max=36e-3,
                     D0=0.6e-9, field_res=0.131e-6, feature_radius_intra=None, seed=0,
                     require_gpu=None, walker_batch_size=20_000,
                     chi_iso=1.06e-6, delta_chi_a=0.0, myelin_water_proton_density=0.4,
                     T2=(0.055, 0.05, 0.01), T1=(1.0, 1.2, 0.44),
                     include_aniso=False, mask_supersample=4, field_crop_margin=1.5e-6,
                     kspace_lowpass=None, water_content="thin", verbose=True):
    """Walk one axon and return a master-walk dict (bank ``_master_arrays`` schema) with the static
    field-grid susceptibility channel.

    ``n_walkers`` is the TOTAL spin count, seeded at **uniform density through the sampled volume**
    (intra-axonal + myelin): the per-pool counts follow the measured volume fractions, so a walker
    represents the same volume of tissue wherever it sits.

    Myelin's lower water content is then applied in one of two ways, ``water_content=``:

    * ``'thin'`` (default) --- randomly KEEP a fraction ``myelin_water_proton_density`` of the myelin
      spins and discard the rest. Every surviving spin then represents the same volume AND the same water
      content, so all weights are identical: the pack is a plain, unweighted spin ensemble that a consumer
      cannot mis-weight. It is also smaller, since the discarded spins are never stored.
    * ``'weight'`` --- keep every myelin spin and carry the water content in its weight. Uses all samples
      (marginally lower variance for a given simulated ensemble) at the cost of a weighted pack.

    Pass ``n_myelin`` explicitly to set pool sizes by hand instead; the weights then also absorb the
    resulting sampling-density mismatch, which stays unbiased but makes the weights bookkeeping rather
    than physics.

    ``field_res`` is the susceptibility field-grid voxel size (m)."""
    from .mesh import Mesh
    from .core import simulate_trajectories

    Vi, Fi = bundle.inner
    Vo, Fo = bundle.outer
    fr_i = feature_radius_intra or _min_radius(Vi, Fi)
    box_min, box_max = bundle.box_min, bundle.box_max
    if verbose:
        print(f"[mesh_axon] {bundle.summary()} | fr_intra={fr_i/1e-6:.3f}um "
              f"n_intra={n_walkers} n_myelin={n_myelin}", flush=True)

    # Containment via the EXACT parity test (mesh_contains), NOT the Mesh's cell-gather classify: for a
    # thin axon in a large box the gather is empty almost everywhere and returns an arbitrary side, which
    # would seed most walkers in free space (silently unrestricted) and corrupt the volume fractions.
    # Seeds are therefore passed explicitly via r0=.
    #
    # Nor the fast nearest-surface test (mesh_inside) on its own: over a padded box it accepts points far
    # outside a tortuous lumen (13.6% of its acceptances on axon06, median 10.2 um from a wall bounding a
    # <=1.24 um radius), because nearest-surface sidedness is only a near-field test. Those walkers are
    # unrestricted free water labelled intra, which biases both the intra signal and f_intra. mesh_contains
    # uses mesh_inside to propose candidates and then ray casts each one, so this stays affordable.
    inside_in = lambda p: mesh_contains(Vi, Fi, p)
    inside_out = lambda p: mesh_contains(Vo, Fo, p)

    # Uniform-density design: measure both volume fractions on a common point set FIRST, then split the
    # spin budget in proportion. A walker then represents the same tissue volume in either pool, and the
    # weights carry only proton density (1.0 vs myelin_water_proton_density) rather than doubling as a
    # correction for unequal sampling density.
    if n_myelin is None:
        rng0 = np.random.default_rng(seed + 99)
        probe = rng0.uniform(box_min, box_max, (200_000, 3))
        pin, pout = inside_in(probe), inside_out(probe)
        fi_pre = float(pin.mean()); fm_pre = float((pout & ~pin).mean())
        frac_i = fi_pre / max(fi_pre + fm_pre, 1e-12)
        n_intra_target = max(1, int(round(n_walkers * frac_i)))
        n_myelin = max(1, int(n_walkers - n_intra_target))
        if verbose:
            print(f"[mesh_axon] uniform density: f_intra={fi_pre:.4f} f_myelin={fm_pre:.4f} "
                  f"-> n_intra={n_intra_target} n_myelin={n_myelin} (total {n_walkers})", flush=True)
        n_walkers = n_intra_target

    # ---- intra pool: restricted inside the inner (axon) wall ----
    r0_i, f_i = _rejection_seeds(inside_in, box_min, box_max, n_walkers, seed)
    mesh_in = Mesh(Vi, Fi, periodic=False, voxel_min=box_min, voxel_max=box_max, feature_radius=fr_i)
    oi = simulate_trajectories(n_walkers, D0, mesh_in, T_max=T_max, dt_save=T_max / n_t,
                               save_relaxation_data=True, seed=seed, r0=r0_i,
                               require_gpu=require_gpu, walker_batch_size=walker_batch_size)
    tr_i = np.asarray(oi[0], np.float64); dt_traj = float(oi[1]); n_t_actual = tr_i.shape[1]
    # Boundary local time (C2). The membrane is impermeable, so there is no EXCHANGE tier -- but the
    # intra-axonal walkers still strike the myelin inner wall repeatedly, so the surface tier is real:
    # rho is a replay knob, not zero by construction. This channel is also what an analytic MT tier is
    # derived from (contact statistics / S:V at the myelin surface), so discarding it would silently
    # remove both capabilities from the pack.
    dlog_i = np.asarray(oi[4], np.float64)

    # ---- myelin pool: frozen shell water (D=0), seeded once in the sheath and held ----
    r0_m, f_shell = _rejection_seeds(lambda p: inside_out(p) & ~inside_in(p),
                                     box_min, box_max, n_myelin, seed + 1)
    rho_m = float(myelin_water_proton_density)
    n_myelin_seeded = n_myelin
    if water_content == "thin":
        # Myelin holds rho_m as much water per unit volume as axonal water: keep that fraction of the
        # uniformly-seeded sheath spins and drop the rest. The survivors then match the intra spins in
        # BOTH volume and water content, so every weight in the pack is identical.
        # Keep an EXACT count, chosen at random: Bernoulli thinning would add binomial noise to a
        # quantity we know exactly (measured 0.385 instead of 0.400 on a 541-spin pool).
        n_keep = max(1, int(round(rho_m * n_myelin)))
        pick = np.random.default_rng(seed + 7).permutation(n_myelin)[:n_keep]
        r0_m = r0_m[np.sort(pick)]
        n_myelin = n_keep
        if verbose:
            print(f"[mesh_axon] water content by thinning: kept {n_myelin}/{n_myelin_seeded} myelin spins "
                  f"(rho={rho_m}) -> unweighted ensemble", flush=True)
    elif water_content != "weight":
        raise ValueError(f"water_content must be 'thin' or 'weight', got {water_content!r}")
    tr_m = np.repeat(r0_m[:, None, :], n_t_actual, axis=1)
    if verbose:
        print(f"[mesh_axon] measured volume fractions: intra={f_i:.4f} myelin={f_shell:.4f} "
              f"(bundle mesh-volume estimate: intra={bundle.f_intra:.4f} myelin={bundle.f_myelin:.4f})",
              flush=True)

    # ---- stack (intra, myelin) ----
    traj = np.concatenate([tr_i, tr_m], axis=0)
    # frozen spins never contact a wall -> their boundary local time is legitimately zero
    dlog_b = np.concatenate([dlog_i, np.zeros((n_myelin, n_t_actual))], axis=0)
    ids = np.concatenate([np.full(len(tr_i), INTRA), np.full(n_myelin, MYELIN)]).astype(np.int8)
    comp = np.repeat(ids[:, None], n_t_actual, axis=1)
    # Per-walker weight = (tissue volume the walker represents) x (proton density of that tissue).
    # Under the uniform-density design the volume per walker is the same in both pools, so it cancels
    # and the weight reduces to proton density alone -- 1.0 for axonal water, rho for myelin water.
    f_m = float(f_shell)
    vol_i = f_i / max(1, len(tr_i))
    # Volume per SEEDED sheath spin (before thinning) -- that is the density the survivors represent.
    vol_m = f_m / max(1, n_myelin_seeded)
    w_m = vol_m * (1.0 if water_content == "thin" else rho_m)
    w = np.concatenate([np.full(len(tr_i), vol_i), np.full(n_myelin, w_m)]).astype(np.float64)
    if verbose:
        spread = w.max() / w.min() if w.min() > 0 else float("nan")
        print(f"[mesh_axon] weight per spin: intra={vol_i:.3e} myelin={w_m:.3e} "
              f"(max/min={spread:.3f}; 1.000 = unweighted ensemble)", flush=True)

    # ---- DETERMINISTIC SHUFFLE: make any walker PREFIX a valid sub-ensemble ----
    # Precision tiers are served by reading the first n rows of the walker-leading arrays (a single
    # contiguous byte range). Stacked intra-then-myelin, the first rows are ALL axonal water, so a
    # prefix read would silently return a myelin-free substrate. One permutation applied to every
    # walker-indexed array makes each prefix an unbiased random sample of the full ensemble.
    order = np.random.default_rng(int(seed) + 991).permutation(traj.shape[0])
    traj, dlog_b, comp, ids, w = traj[order], dlog_b[order], comp[order], ids[order], w[order]
    if verbose:
        pre = min(2000, len(ids))
        print(f"[mesh_axon] shuffled walkers (seed {int(seed)+991}); first {pre} rows are "
              f"{100.0*np.mean(ids[:pre] == MYELIN):.1f}% myelin vs "
              f"{100.0*np.mean(ids == MYELIN):.1f}% overall", flush=True)

    out = dict(
        traj=traj, dt_traj=dt_traj, T_max=float(T_max), comp=comp, comp0=ids.copy(), w=w,
        dlog_b=dlog_b,                       # C2: surface relaxivity + the basis for an analytic MT tier
        R=np.eye(3), D_intra=float(D0), walkers_shuffled=True,   # see the shuffle above
        T2_per_comp=np.asarray(T2, float), T1_per_comp=np.asarray(T1, float),
        n_walkers=int(traj.shape[0]), seed=int(seed),
        substrate_frame=np.eye(3),                       # axon aligned to +z (Winther): identity frame
        delta_chi_a=float(delta_chi_a), susc_chi_iso=float(chi_iso),
    )

    # ---- C3 static susceptibility field-grid channel ----
    if verbose:
        print(f"[mesh_axon] building susceptibility field basis (res={field_res/1e-6:.3f}um)...", flush=True)
    # kspace_lowpass=None by default, for two reasons established by the oracle calibration:
    #  (a) with a partial-volume source the window is unnecessary and marginally worse (lumen null
    #      0.098% -> 0.073% of chi*B0 when removed; sheath amplitude unchanged at 0.999x analytic);
    #  (b) the window is applied to iso_P but NOT to iso_local, which breaks the exact identity
    #      trace(iso_P) == 3*iso_local (machine precision at None, 6.2e-2 relative error at 0.5).
    #      With the identity intact, iso_P_zz need not be stored -- it is reconstructable as
    #      3*iso_local - iso_P_xx - iso_P_yy, saving 1 of 7 channels in the path/grid field tier.
    basis, origin, vs = mesh_field_basis(bundle.inner, bundle.outer, box_min, box_max,
                                         res=field_res, include_aniso=include_aniso,
                                         mask_supersample=mask_supersample,
                                         kspace_lowpass=kspace_lowpass)
    # The basis is SOLVED on the full padded box (domain-converged), but only the region the
    # walkers occupy needs STORING -- crop to the axon + margin so packs stay shareable.
    il, iP, aG = basis["iso_local"], basis["iso_P"], basis.get("aniso_G")
    if field_crop_margin:
        lo = np.maximum(0, np.floor((Vo.min(0) - field_crop_margin - origin) / vs).astype(int))
        hi = np.minimum(np.array(il.shape), np.ceil((Vo.max(0) + field_crop_margin - origin) / vs).astype(int) + 1)
        sl = tuple(slice(int(lo[a]), int(hi[a])) for a in range(3))
        il = il[sl]; iP = iP[(slice(None),) + sl]
        aG = None if aG is None else aG[(slice(None),) + sl]
        origin = origin + lo * vs
        if verbose:
            print(f"[mesh_axon] field grid cropped {basis['shape']} -> {il.shape} for storage", flush=True)
    out["susc_field_basis"] = {"iso_local": np.asarray(il, np.float32),
                               "iso_P": np.asarray(iP, np.float32),
                               "aniso_G": (None if aG is None else np.asarray(aG, np.float32)),
                               "shape": tuple(int(s) for s in il.shape),
                               "voxel_size": np.asarray(vs, float)}
    out["susc_grid_origin"] = np.asarray(origin, float)
    if verbose:
        print(f"[mesh_axon] susc field-grid: shape={basis['shape']}  comps={np.unique(comp)}", flush=True)
    return out
