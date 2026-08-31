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
from dmipy_sim.sh_convolution import PackResponder, apply_odf_coupled, watson_odf_sh

PACK = "/home/rutger/dmrai-ws/winther-data/hf_release_winther_g6/packs/axon06.rpk"
N, R_IN, R_OUT, LMAX, KAPPA = 30, 8.0, 13.0, 8, 12.0


def circular_wm(n=N, r_in=R_IN, r_out=R_OUT, lmax=LMAX, kappa=KAPPA):
    """Annulus of tangentially-oriented fibres; returns the arrays of RPH.md Sec. 2."""
    c = (n - 1) / 2.0
    idx, sh, frac = [], [], []
    for j in range(n):
        for i in range(n):
            dx, dy = i - c, j - c
            r = np.hypot(dx, dy)
            if not (r_in <= r <= r_out):
                continue
            mu = np.array([-dy / r, dx / r, 0.0])          # tangential, as in the studio
            idx.append((i, j))
            sh.append(watson_odf_sh(kappa, mu=mu, lmax=lmax))
            frac.append(1.0)
    return (np.asarray(idx, np.int32),
            np.zeros((len(idx), 1), np.int16),             # one pack, cited by every voxel
            np.asarray(frac, np.float32)[:, None],
            np.asarray(sh, np.float32)[:, None, :])


def write_rph(path, pack_path, pack_id, arrays, lmax=LMAX, n=N):
    from safetensors.numpy import save_file
    import hashlib
    h = hashlib.sha256(open(pack_path, "rb").read()).hexdigest()
    vi, pid, pf, sh = arrays
    meta = {
        "rph_schema_version": "0.1.0",
        "id": "phantoms/circular-wm/annulus-30",
        "packs": [{"id": pack_id, "sha256": h, "uri": pack_path}],
        "grid": {"shape": [n, n], "voxel_size_m": [1e-3, 1e-3], "frame": "phantom"},
        "sh": {"basis": "real", "convention": "orthonormal", "lmax": lmax},
        "license": "CC-BY-4.0",
        "citation": "dmipy-sim replay phantom prototype",
    }
    save_file({"voxel_index": vi, "pack_id": pid, "pack_fraction": pf, "odf_sh": sh},
              str(path), metadata={"rph": json.dumps(meta)})
    return meta


def replay_rph(path, responder, g_dir, b0_dir, l_g=8, l_b=6):
    """One Lambda for the measurement, then a contraction per voxel (RPH.md Sec. 3)."""
    from safetensors import safe_open
    with safe_open(str(path), framework="numpy") as f:
        meta = json.loads(f.metadata()["rph"])
        vi, pid = f.get_tensor("voxel_index"), f.get_tensor("pack_id")
        pf, sh = f.get_tensor("pack_fraction"), f.get_tensor("odf_sh")
    lmax = int(meta["sh"]["lmax"])
    lam, resid, _ = responder.spectrum_at(g_dir, l_g=l_g, l_b=l_b)
    out = np.zeros(len(vi), np.complex128)
    for v in range(len(vi)):
        for p in range(pid.shape[1]):
            if pid[v, p] < 0 or pf[v, p] == 0:
                continue
            out[v] += pf[v, p] * apply_odf_coupled(lam, sh[v, p], g_dir, b0_dir,
                                                   l_fod=lmax, l_g=l_g, l_b=l_b)
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
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "circular_wm.rph")
    meta = write_rph(out, PACK, "winther/g6/axon06", arrays)
    print(f"wrote {out}  ({os.path.getsize(out)/1024:.0f} kB, {len(arrays[0])} voxels, "
          f"1 pack of {os.path.getsize(PACK)/1e6:.0f} MB)")

    backend = os.environ.get("RPH_BACKEND", "numpy")
    R = PackResponder(pk, prof, b0, amplitude=0.05, B0=3.0, chi_iso=-9.4e-6,
                      chi_aniso=-1.0e-7, refocus_time=0.5 * T, n_theta=32, n_phi=64,
                      backend=backend)
    for nm, g in (("g || x", [1, 0, 0.]), ("g || y", [0, 1, 0.]), ("g at 55deg", [np.sin(.96), 0, np.cos(.96)])):
        t0 = time.time()
        vi, S, _, resid = replay_rph(out, R, np.asarray(g, float), b0)
        print(f"  {nm:>11}: |S| {np.abs(S).min():.4f}..{np.abs(S).max():.4f}  "
              f"resid {resid:.1e}  {time.time()-t0:.2f} s for {len(vi)} voxels")
