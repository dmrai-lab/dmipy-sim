"""Build canonical single-cylinder / single-sphere restricted-diffusion replay packs.

One ``.rpk`` per ``(shape, diameter, D0)``: walk an impermeable ``Cylinder(radius, axis=z)`` or
``Sphere(radius)`` to ``T_max``, recording C0 (gradient) + C1 (bulk T2/T1) + C2 (surface local-time)
channels, then freeze via :func:`dmipy_sim.bank.build_replay_pack`. Diameter is fixed at walk time
(the substrate); orientation (cylinder) and the acquisition are replay knobs. These are the reference
substrates behind dmipy-fit's ``C5MonteCarloReplayCylinder`` / ``S5MonteCarloReplaySphere``.

    python -m experiments.substrate_bank.build_canonical_restricted \
        --shape cylinder --diameter-um 10 --D0 2.0e-9 --T-max-ms 200 --walkers 100000 --out-dir /tmp/canon

The substrate geometry is stored canonically (cylinder axis = +z), so at replay a gradient along z is
"parallel" (free diffusion) and any in-plane gradient is "perpendicular" (restricted) — no frame needed.
"""
import os
import numpy as np

from dmipy_sim import simulate_trajectories, bank
from dmipy_sim.geometries import Cylinder, Sphere
from dmipy_sim import compression as cx


def walk_restricted_master(shape, diameter, D0, *, T_max=200e-3, n_t=2000, n_walkers=100_000,
                           seed=0, T2=0.08, T1=1.0, require_gpu=None, walker_batch_size=50_000,
                           verbose=True):
    """Walk one impermeable restricted pore and return a master dict for :func:`bank.build_replay_pack`
    (C0+C1+C2). ``shape`` is ``"cylinder"`` or ``"sphere"``; ``diameter`` and ``D0`` are SI (m, m²/s)."""
    radius = 0.5 * float(diameter)
    if shape == "cylinder":
        geom = Cylinder(radius=radius, orientation=[0.0, 0.0, 1.0])   # axis = +z (canonical)
    elif shape == "sphere":
        geom = Sphere(radius=radius)
    else:
        raise ValueError(f"shape must be 'cylinder' or 'sphere', got {shape!r}")
    dt_save = T_max / (n_t - 1)
    out = simulate_trajectories(n_walkers=int(n_walkers), diffusivity=float(D0), geometry=geom,
                                T_max=float(T_max), dt_save=float(dt_save), seed=int(seed),
                                save_relaxation_data=True, require_gpu=require_gpu,
                                walker_batch_size=int(walker_batch_size))
    traj, dt, sub_steps, dt_sim, dlog_b, comp = out
    traj = np.asarray(traj, np.float64)
    nw, nt = traj.shape[0], traj.shape[1]
    if verbose:
        print(f"[walk] {shape} d={diameter*1e6:.2f}um D0={D0:.2e} R={radius:.2e} "
              f"n_w={nw} n_t={nt} sub_steps={sub_steps} dt_sim={dt_sim:.2e}", flush=True)
    return dict(traj=traj, dt_traj=float(dt), T_max=float(T_max),
                comp=np.zeros((nw, nt), np.int8), comp0=np.zeros(nw, np.int64),
                w=np.ones(nw), T2_per_comp=np.array([float(T2)]), T1_per_comp=np.array([float(T1)]),
                dlog_b=np.asarray(dlog_b, np.float64), D_intra=float(D0),
                n_walkers=int(nw), seed=int(seed))


def restricted_envelope():
    """Lean gradient+relaxation fidelity battery for a single restricted pore (no field/MT tier).
    The lowrank codec compresses *positions* (acquisition-agnostic), so a small PGSE/OGSE/short-δ
    battery over ⊥/∥ directions certifies the reconstruction — vs the ~87-waveform default that makes
    ``measure_fidelity`` (host-numpy) the build bottleneck."""
    return dict(bvals=[0.0, 1.5e9, 3e9], dirs=[[0, 0, 1], [1, 0, 0]],
                ogse_periods=[2], shortd_b=1e9, shortd_deltas_frac=[0.05],
                B0_list=[], theta_deg=[0], delta_frac=0.2, Delta_frac=0.5)


def pack_id(shape, diameter, D0):
    """Stable dataset id: ``canonical/<regime>/<shape>/d<NN.NN>um`` (regime tag from D0)."""
    return f"canonical/D0-{D0*1e9:.2f}e-9/{shape}/d{diameter*1e6:06.2f}um"


def build_pack(shape, diameter, D0, *, out_dir=None, sigma_star=5e-3, K=128,
               method="bridge_dst",
               envelope=None, citation="Substrate Commons canonical restricted-shape reference dataset",
               license="CC-BY-4.0", verbose=True, **walk_kw):
    """Walk + freeze one canonical pack. Returns ``(ReplayPack, out_path_or_None)``.

    Codec is ``bridge_dst``: two exact endpoints per axis followed by ``K`` sine bands of the
    pinned residual (the Brownian bridge).  Walker-preserving and SVD-FREE -- which matters
    here, since ``numpy.linalg.svd`` is pathologically slow on this aarch64/OpenBLAS build (a
    (5000,1200) SVD does not finish in 2 min while an equivalent GEMM is <1 s).  ``K`` is the
    number of retained sine bands; the stored width per axis is ``K + 2``."""
    m = walk_restricted_master(shape, diameter, D0, verbose=verbose, **walk_kw)
    if envelope is None:
        envelope = restricted_envelope()
    pid = pack_id(shape, diameter, D0)
    out_path = None
    if out_dir is not None:
        out_path = os.path.join(out_dir, pid + ".rpk")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pack = bank.build_replay_pack(
        m, id=pid, method=method, envelope=envelope, K=int(K), sigma_star=sigma_star,
        surface_relaxivity=True, license=license, citation=citation, out_path=out_path,
        provenance=dict(shape=shape, diameter_m=float(diameter), diffusivity=float(D0),
                        geometry="single impermeable restricted pore", real_or_synthetic="synthetic"),
        verbose=verbose)
    if verbose:
        f = pack.fidelity
        print(f"[pack] {pid}  floor_max={f.get('floor_max'):.4g} err_max={f.get('err_max'):.4g} "
              f"meets_sigma*={f.get('meets_target')}  -> {out_path}", flush=True)
    return pack, out_path


def _parse_diams(spec):
    """'0.1:20:0.1' -> np.arange; or comma list '1,2,5,10' (µm)."""
    if ":" in spec:
        lo, hi, step = (float(x) for x in spec.split(":"))
        return np.round(np.arange(lo, hi + 0.5 * step, step), 6)
    return np.array([float(x) for x in spec.split(",")])


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Build canonical restricted cylinder/sphere replay packs.")
    ap.add_argument("--shape", choices=["cylinder", "sphere"], required=True)
    ap.add_argument("--diameter-um", default="10", help="µm: single, comma-list, or lo:hi:step")
    ap.add_argument("--D0", type=float, default=2.0e-9, help="intrinsic diffusivity m²/s")
    ap.add_argument("--T-max-ms", type=float, default=200.0)
    ap.add_argument("--n-t", type=int, default=2000)
    ap.add_argument("--walkers", type=int, default=100_000)
    ap.add_argument("--sigma-star", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--cpu", action="store_true", help="allow CPU (else require GPU)")
    a = ap.parse_args(argv)
    diams = _parse_diams(a.diameter_um) * 1e-6
    for d in diams:
        build_pack(a.shape, float(d), a.D0, out_dir=a.out_dir, sigma_star=a.sigma_star,
                   T_max=a.T_max_ms * 1e-3, n_t=a.n_t, n_walkers=a.walkers, seed=a.seed,
                   require_gpu=(None if a.cpu else True))


if __name__ == "__main__":
    main()
