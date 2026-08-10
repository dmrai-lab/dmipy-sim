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

from .susceptibility_field import mesh_field_basis, mesh_inside

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


def mesh_axon_master(bundle, *, n_walkers=30_000, n_myelin=3000, n_t=100, T_max=36e-3,
                     D0=0.6e-9, field_res=0.1e-6, feature_radius_intra=None, seed=0,
                     require_gpu=None, walker_batch_size=20_000,
                     chi_iso=1.06e-6, delta_chi_a=0.0, myelin_water_proton_density=0.4,
                     T2=(0.055, 0.05, 0.01), T1=(1.0, 1.2, 0.44),
                     include_aniso=True, mask_supersample=2, verbose=True):
    """Walk one axon and return a master-walk dict (bank ``_master_arrays`` schema) with the static
    field-grid susceptibility channel. ``n_walkers`` intra-axonal diffusers + ``n_myelin`` frozen
    myelin walkers; ``field_res`` is the susceptibility field-grid voxel size (m)."""
    from .mesh import Mesh
    from .core import simulate_trajectories

    Vi, Fi = bundle.inner
    Vo, Fo = bundle.outer
    fr_i = feature_radius_intra or _min_radius(Vi, Fi)
    box_min, box_max = bundle.box_min, bundle.box_max
    if verbose:
        print(f"[mesh_axon] {bundle.summary()} | fr_intra={fr_i/1e-6:.3f}um "
              f"n_intra={n_walkers} n_myelin={n_myelin}", flush=True)

    # Containment via the exact global nearest-surface test (mesh_inside), NOT the Mesh's
    # cell-gather classify: for a thin axon in a large box the gather is empty almost everywhere and
    # returns an arbitrary side, which would seed most walkers in free space (silently unrestricted)
    # and corrupt the volume fractions. Seeds are therefore passed explicitly via r0=.
    inside_in = lambda p: mesh_inside(Vi, Fi, p, clip_axis=2)
    inside_out = lambda p: mesh_inside(Vo, Fo, p, clip_axis=2)

    # ---- intra pool: restricted inside the inner (axon) wall ----
    r0_i, f_i = _rejection_seeds(inside_in, box_min, box_max, n_walkers, seed)
    mesh_in = Mesh(Vi, Fi, periodic=False, voxel_min=box_min, voxel_max=box_max, feature_radius=fr_i)
    oi = simulate_trajectories(n_walkers, D0, mesh_in, T_max=T_max, dt_save=T_max / n_t,
                               save_relaxation_data=True, seed=seed, r0=r0_i,
                               require_gpu=require_gpu, walker_batch_size=walker_batch_size)
    tr_i = np.asarray(oi[0], np.float64); dt_traj = float(oi[1]); n_t_actual = tr_i.shape[1]

    # ---- myelin pool: frozen shell water (D=0), seeded once in the sheath and held ----
    r0_m, f_shell = _rejection_seeds(lambda p: inside_out(p) & ~inside_in(p),
                                     box_min, box_max, n_myelin, seed + 1)
    tr_m = np.repeat(r0_m[:, None, :], n_t_actual, axis=1)
    if verbose:
        print(f"[mesh_axon] measured volume fractions: intra={f_i:.4f} myelin={f_shell:.4f} "
              f"(bundle mesh-volume estimate: intra={bundle.f_intra:.4f} myelin={bundle.f_myelin:.4f})",
              flush=True)

    # ---- stack (intra, myelin) ----
    traj = np.concatenate([tr_i, tr_m], axis=0)
    ids = np.concatenate([np.full(len(tr_i), INTRA), np.full(n_myelin, MYELIN)]).astype(np.int8)
    comp = np.repeat(ids[:, None], n_t_actual, axis=1)
    f_m = float(f_shell)
    w = np.concatenate([
        np.full(len(tr_i), f_i / max(1, len(tr_i))),
        np.full(n_myelin, f_m * float(myelin_water_proton_density) / max(1, n_myelin)),
    ]).astype(np.float64)

    out = dict(
        traj=traj, dt_traj=dt_traj, T_max=float(T_max), comp=comp, comp0=ids.copy(), w=w,
        R=np.eye(3), D_intra=float(D0),
        T2_per_comp=np.asarray(T2, float), T1_per_comp=np.asarray(T1, float),
        n_walkers=int(traj.shape[0]), seed=int(seed),
        substrate_frame=np.eye(3),                       # axon aligned to +z (Winther): identity frame
        delta_chi_a=float(delta_chi_a), susc_chi_iso=float(chi_iso),
    )

    # ---- C3 static susceptibility field-grid channel ----
    if verbose:
        print(f"[mesh_axon] building susceptibility field basis (res={field_res/1e-6:.3f}um)...", flush=True)
    basis, origin, vs = mesh_field_basis(bundle.inner, bundle.outer, box_min, box_max,
                                         res=field_res, include_aniso=include_aniso,
                                         mask_supersample=mask_supersample)
    out["susc_field_basis"] = {"iso_local": np.asarray(basis["iso_local"], np.float32),
                               "iso_P": np.asarray(basis["iso_P"], np.float32),
                               "aniso_G": (np.asarray(basis["aniso_G"], np.float32)
                                           if basis.get("aniso_G") is not None else None),
                               "shape": tuple(int(s) for s in basis["shape"]),
                               "voxel_size": np.asarray(vs, float)}
    out["susc_grid_origin"] = np.asarray(origin, float)
    if verbose:
        print(f"[mesh_axon] susc field-grid: shape={basis['shape']}  comps={np.unique(comp)}", flush=True)
    return out
