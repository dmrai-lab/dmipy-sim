"""Prototype .rph: a circular white-matter phantom composed from ONE solved pack.

Mirrors the annulus of the dmipy Bloch studio -- a 30x30 grid, tissue between radii 8 and 13,
fibres running tangentially -- so every voxel is the same solved microstructure at a different
pose.  That is the whole point of a replay phantom: the expensive object is shared, and the
phantom carries only where the packs are, how they are oriented, and in what proportion.

Writes a `.rph` per RPH.md (replay-pack-spec), then replays it: one Lambda per measurement,
then a contraction per voxel.
"""
import json
import numpy as np

from dmipy_sim.replay import read_rpk
from dmipy_sim.gaunt import n_sh_coeffs
from dmipy_sim.sh_convolution import (PackResponder, apply_odf_coupled,
                                      compose_voxel, free_water_response,
                                      watson_odf_sh)

PACK = "/home/rutger/dmrai-ws/winther-data/hf_release_winther_g6/packs/axon06.rpk"
N, R_IN, R_OUT, LMAX, KAPPA = 30, 8.0, 13.0, 8, 12.0


def circular_wm(n=N, r_in=R_IN, r_out=R_OUT, lmax=LMAX, kappa=KAPPA, sub=8):
    """Annulus of tangentially-oriented fibres around a free-water core (RPH.md Sec. 3).

    Three substrates: 0 white matter (a pack), 1 free water (analytic, ``exp(-bD)``), 2 inert.
    The boundaries are supersampled, so voxels straddling the inner radius are a genuine
    WM / CSF partial volume and those at the outer radius are WM / inert.  Rows sum to one.

    Free water is analytic rather than a pack because its signal decays exponentially in b
    while a Monte-Carlo floor does not: at b = 3000 s/mm^2 a 4000-walker free-water pack has
    ~113x more noise than signal, and 1% relative accuracy would need of order 1e11 walkers.
    Inert is not air -- an air interface would dominate the field of its neighbours, which a
    voxel-by-voxel phantom has no mechanism for.
    """
    c = (n - 1) / 2.0
    off = (np.arange(sub) + 0.5) / sub - 0.5
    ox, oy = np.meshgrid(off, off, indexing="ij")
    idx, sid, frac, sh = [], [], [], []
    zero = np.zeros(n_sh_coeffs(lmax), np.float64)
    for j in range(n):
        for i in range(n):
            rr = np.hypot(i + ox - c, j + oy - c)
            f_wm = float(np.mean((rr >= r_in) & (rr <= r_out)))
            f_csf = float(np.mean(rr < r_in))               # ventricle-like core
            f_inert = 1.0 - f_wm - f_csf
            if f_wm == 0.0 and f_csf == 0.0:
                continue                                    # sparse: unoccupied voxels absent
            dx, dy = i - c, j - c
            r = np.hypot(dx, dy)
            mu = np.array([-dy / r, dx / r, 0.0]) if r > 1e-9 else np.array([1.0, 0.0, 0.0])
            idx.append((i, j, 0))
            sid.append((0, 1, 2))
            frac.append((f_wm, f_csf, max(f_inert, 0.0)))
            sh.append((watson_odf_sh(kappa, mu=mu, lmax=lmax), zero, zero))
    return (np.asarray(idx, np.int32), np.asarray(sid, np.int16),
            np.asarray(frac, np.float32), np.asarray(sh, np.float32))


def write_rph(path, pack_path, pack_id, arrays, m0, *, embed=True, lmax=LMAX, n=N):
    """Write a .rph. ``embed`` puts the pack's arrays under ``substrate0/`` so the phantom is a
    standalone artifact -- nothing to resolve, nothing to go missing."""
    from safetensors.numpy import save_file
    import hashlib
    h = hashlib.sha256(open(pack_path, "rb").read()).hexdigest()
    vi, sid, gf, sh = arrays
    tensors = {"voxel_index": vi, "substrate_id": sid,
               "geometric_fraction": gf, "odf_sh": sh}
    wm = {"id": pack_id, "kind": "pack", "m0": float(m0), "sha256": h,
          "embedded": bool(embed)}
    csf = {"id": "csf/free-water", "kind": "analytic", "m0": 1.0,
           "model": "free_water", "params": {"diffusivity": 3.0e-9}}
    inert = {"id": "background/inert", "kind": "inert", "m0": 0.0}
    if embed:
        pk = read_rpk(pack_path)
        for k, v in pk.arrays.items():
            tensors[f"substrate0/{k}"] = np.ascontiguousarray(v)
        wm["pack_meta"] = pk.meta
    else:
        wm["uri"] = pack_path
    meta = {
        "rph_schema_version": "0.2.0",
        "id": "phantoms/circular-wm/annulus-30",
        "grid": {"shape": [n, n, 1], "voxel_size_m": [1e-3, 1e-3, 1e-3], "frame": "phantom"},
        "orientation": {"mode": "odf_sh", "lmax": lmax, "basis": "real",
                        "convention": "orthonormal"},
        "substrates": [wm, csf, inert],
        "license": "CC-BY-4.0",
        "citation": "dmipy-sim replay phantom prototype",
    }
    save_file(tensors, str(path), metadata={"rph": json.dumps(meta)})
    return meta


def replay_rph(path, responder, g_dir, b0_dir, b_value, l_g=8, l_b=6):
    """One Lambda for the measurement, then a contraction per voxel (RPH.md Sec. 3)."""
    from safetensors import safe_open
    with safe_open(str(path), framework="numpy") as f:
        meta = json.loads(f.metadata()["rph"])
        vi, pid = f.get_tensor("voxel_index"), f.get_tensor("substrate_id")
        pf, sh = f.get_tensor("geometric_fraction"), f.get_tensor("odf_sh")
    lmax = int(meta["orientation"]["lmax"])
    m0 = np.array([s["m0"] for s in meta["substrates"]], float)
    bearing = np.array([s["kind"] != "inert" for s in meta["substrates"]])
    lam, resid, _ = responder.spectrum_at(g_dir, l_g=l_g, l_b=l_b)
    out = np.zeros(len(vi), np.complex128)
    spectra = []
    for sub in meta["substrates"]:
        if sub["kind"] == "analytic":
            spectra.append(free_water_response(b_value, sub["params"]["diffusivity"]))
        else:
            spectra.append(lam)
    for v in range(len(vi)):
        out[v] = compose_voxel(spectra, pid[v], pf[v], sh[v], m0, g_dir, b0_dir,
                               l_fod=lmax, l_g=l_g, l_b=l_b, signal_bearing=bearing)
    return vi, out, meta, resid


def amplitude_for_b(profile, dt, b_target):
    """Gradient amplitude [T/m] giving ``b_target`` [s/m^2] for a unit-amplitude profile."""
    from dmipy_sim.constants import GAMMA
    q = GAMMA * np.cumsum(np.asarray(profile, np.float64)) * dt
    return float(np.sqrt(b_target / (np.sum(q * q) * dt)))


if __name__ == "__main__":
    import time, os
    pk = read_rpk(PACK)
    pm = pk.meta["compression"]["channels"]["susceptibility_path"]
    n_t, dt = int(pm["n_t"]), pk.dt
    t = np.arange(n_t) * dt; T = n_t * dt
    prof = ((t < 0.2 * T).astype(float) - ((t >= 0.5 * T) & (t < 0.7 * T)).astype(float))
    b0 = np.array([0, 0, 1.0])

    arrays = circular_wm()
    here = os.path.dirname(os.path.abspath(__file__))
    for embed in (False, True):
        out = os.path.join(here, f"circular_wm{'_standalone' if embed else ''}.rph")
        write_rph(out, PACK, "winther/g6/axon06", arrays, m0=0.70, embed=embed)
        f = arrays[2]
        edge = int(((f[:, 0] > 0) & (f[:, 0] < 0.999)).sum())
        csfv = int((f[:, 1] > 0).sum())
        print(f"  {'embedded ' if embed else 'referenced'}: {os.path.getsize(out)/1e6:8.2f} MB"
              f"   ({len(arrays[0])} voxels, {edge} partial-volume WM, {csfv} with free "
              f"water, pack is {os.path.getsize(PACK)/1e6:.0f} MB)")
    out = os.path.join(here, "circular_wm_standalone.rph")

    backend = os.environ.get("RPH_BACKEND", "numpy")
    B_VALUE = 1e9                                     # s/m^2  == 1000 s/mm^2
    amp = amplitude_for_b(prof, dt, B_VALUE)
    print(f"  acquisition: b = {B_VALUE/1e6:.0f} s/mm^2 at TE = {T*1e3:.0f} ms "
          f"(amplitude {amp:.3f} T/m)")
    R = PackResponder(pk, prof, b0, amplitude=amp, B0=3.0, chi_iso=-9.4e-6,
                      chi_aniso=-1.0e-7, refocus_time=0.5 * T, n_theta=32, n_phi=64,
                      backend=backend)
    for nm, g in (("g || x", [1, 0, 0.]), ("g || y", [0, 1, 0.]), ("g at 55deg", [np.sin(.96), 0, np.cos(.96)])):
        t0 = time.time()
        vi, S, _, resid = replay_rph(out, R, np.asarray(g, float), b0, B_VALUE)
        print(f"  {nm:>11}: |S| {np.abs(S).min():.4f}..{np.abs(S).max():.4f}  "
              f"resid {resid:.1e}  {time.time()-t0:.2f} s for {len(vi)} voxels")
