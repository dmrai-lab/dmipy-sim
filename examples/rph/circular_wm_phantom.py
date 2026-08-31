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
                                      compose_voxel, watson_odf_sh)

PACK = "/home/rutger/dmrai-ws/winther-data/hf_release_winther_g6/packs/axon06.rpk"
N, R_IN, R_OUT, LMAX, KAPPA = 30, 8.0, 13.0, 8, 12.0


def circular_wm(n=N, r_in=R_IN, r_out=R_OUT, lmax=LMAX, kappa=KAPPA, sub=8):
    """Annulus of tangentially-oriented fibres; returns the arrays of RPH.md Sec. 3.

    The annulus edge is supersampled, so boundary voxels are genuinely partial-volume: the
    white-matter fraction is the covered area and the remainder is an INERT substrate --
    volume that carries no modelled physics.  Rows sum to one, as the spec requires.

    Inert is not air.  An air cavity would perturb the field of neighbouring voxels at a
    magnitude far above anything the packs carry, and a phantom has no mechanism for that, so
    this background stands for "outside the modelled object", not for a material.
    """
    c = (n - 1) / 2.0
    off = (np.arange(sub) + 0.5) / sub - 0.5                # subsample offsets within a voxel
    ox, oy = np.meshgrid(off, off, indexing="ij")
    idx, sh, frac, sid = [], [], [], []
    for j in range(n):
        for i in range(n):
            rr = np.hypot(i + ox - c, j + oy - c)
            f = float(np.mean((rr >= r_in) & (rr <= r_out)))
            if f == 0.0:
                continue                                    # sparse: unoccupied voxels absent
            dx, dy = i - c, j - c
            r = np.hypot(dx, dy)
            mu = np.array([-dy / r, dx / r, 0.0]) if r > 1e-9 else np.array([1.0, 0.0, 0.0])
            idx.append((i, j, 0))
            sid.append((0, 1))                              # 0 = white matter, 1 = inert
            frac.append((f, 1.0 - f))                       # sums to one by construction
            sh.append((watson_odf_sh(kappa, mu=mu, lmax=lmax),
                       np.zeros(n_sh_coeffs(lmax), np.float64)))
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
    wm = {"id": pack_id, "m0": float(m0), "sha256": h, "embedded": bool(embed)}
    inert = {"id": "background/inert", "m0": 0.0, "pack": None}
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
        "substrates": [wm, inert],
        "license": "CC-BY-4.0",
        "citation": "dmipy-sim replay phantom prototype",
    }
    save_file(tensors, str(path), metadata={"rph": json.dumps(meta)})
    return meta


def replay_rph(path, responder, g_dir, b0_dir, l_g=8, l_b=6):
    """One Lambda for the measurement, then a contraction per voxel (RPH.md Sec. 3)."""
    from safetensors import safe_open
    with safe_open(str(path), framework="numpy") as f:
        meta = json.loads(f.metadata()["rph"])
        vi, pid = f.get_tensor("voxel_index"), f.get_tensor("substrate_id")
        pf, sh = f.get_tensor("geometric_fraction"), f.get_tensor("odf_sh")
    lmax = int(meta["orientation"]["lmax"])
    m0 = np.array([s["m0"] for s in meta["substrates"]], float)
    bearing = np.array([s.get("pack", "-") is not None for s in meta["substrates"]])
    lam, resid, _ = responder.spectrum_at(g_dir, l_g=l_g, l_b=l_b)
    out = np.zeros(len(vi), np.complex128)
    spectra = [lam] * len(meta["substrates"])
    for v in range(len(vi)):
        out[v] = compose_voxel(spectra, pid[v], pf[v], sh[v], m0, g_dir, b0_dir,
                               l_fod=lmax, l_g=l_g, l_b=l_b, signal_bearing=bearing)
    return vi, out, meta, resid


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
        edge = int((arrays[2][:, 0] < 0.999).sum())
        print(f"  {'embedded ' if embed else 'referenced'}: {os.path.getsize(out)/1e6:8.2f} MB"
              f"   ({len(arrays[0])} voxels, {edge} partial-volume at the annulus edge, "
              f"pack is {os.path.getsize(PACK)/1e6:.0f} MB)")
    out = os.path.join(here, "circular_wm_standalone.rph")

    backend = os.environ.get("RPH_BACKEND", "numpy")
    R = PackResponder(pk, prof, b0, amplitude=0.05, B0=3.0, chi_iso=-9.4e-6,
                      chi_aniso=-1.0e-7, refocus_time=0.5 * T, n_theta=32, n_phi=64,
                      backend=backend)
    for nm, g in (("g || x", [1, 0, 0.]), ("g || y", [0, 1, 0.]), ("g at 55deg", [np.sin(.96), 0, np.cos(.96)])):
        t0 = time.time()
        vi, S, _, resid = replay_rph(out, R, np.asarray(g, float), b0)
        print(f"  {nm:>11}: |S| {np.abs(S).min():.4f}..{np.abs(S).max():.4f}  "
              f"resid {resid:.1e}  {time.time()-t0:.2f} s for {len(vi)} voxels")
