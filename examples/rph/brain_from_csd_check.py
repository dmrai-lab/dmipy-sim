"""The round trip of brain_from_csd.py: deconvolve the synthetic DWI with the phantom's own white-matter response
and compare the recovered FOD with the one that built it, voxel by voxel.

A Tournier-style constrained spherical deconvolution (positivity on a sphere, Laplace-Beltrami smoothing), with the
kernel taken from the replay pack itself -- the pack replayed at the identity pose on the acquisition's directions
-- so nothing outside this package is needed. It is single-tissue: the GM and CSF terms of a voxel are the isotropic
contamination a single-fibre CSD always sees, which is why the comparison is made on white-matter-dominated voxels.

    python examples/rph/brain_from_csd_check.py <batman_dir> <out_dir> <wm_pack.rpk> [--TE 0.100 --delta 0.025 --Delta 0.055]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain_from_csd import acquisition  # noqa: E402

from dmipy_sim.io.mrtrix import read_mif
from dmipy_sim.phantom import Grid, PackSubstrate
from dmipy_sim.replay.so3 import n_sh_coeffs, real_sh, rotate_sh, sh_block


def fibonacci_sphere(n=724):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def kernel(pack, seq, g, lmax=8):
    """``A (n_meas, n_c)``: FOD harmonics -> signal, from the pack's response about its own axis on each shell."""
    E = np.real(pack.replay(seq, orientation=np.eye(3), complex_signal=True))
    dirs = np.asarray(seq.encoding.gradient_directions, np.float64)
    shells = np.round(g[:, 3] / 100.0) * 100.0
    ls = list(range(0, lmax + 1, 2))
    cosz = np.abs(dirs[:, 2])                                             # the angle to the axis (z at the identity pose)
    polar = np.stack([np.sqrt(np.clip(1 - cosz ** 2, 0, 1)), np.zeros_like(cosz), cosz], 1)
    Yz = np.stack([real_sh(lmax, polar)[:, sh_block(l).start + l] for l in ls], 1)   # Y_l0 per measurement, (n_meas, n_l)
    r = {}
    for sh in np.unique(shells):
        m = shells == sh
        if sh == 0:
            r[sh] = np.array([E[m].mean() / Yz[m, 0].mean()] + [0.0] * (len(ls) - 1))
        else:
            r[sh] = np.linalg.lstsq(Yz[m], E[m], rcond=None)[0]
    Y = real_sh(lmax, dirs)                                               # (n_meas, n_c)
    A = np.zeros_like(Y)
    for i in range(len(shells)):
        for li, l in enumerate(ls):
            A[i, sh_block(l)] = np.sqrt(4 * np.pi / (2 * l + 1)) * r[shells[i]][li] * Y[i, sh_block(l)]
    return A


def csd(A, S, sphere_Y, lmax=8, lambda_pos=1.0, lambda_lb=5e-4, tau=0.1, max_iter=50):
    """Tournier 2007 CSD for one voxel: positivity on the sphere, iteratively reweighted."""
    n4 = n_sh_coeffs(4)
    ls = np.concatenate([[l] * (2 * l + 1) for l in range(0, lmax + 1, 2)])
    Rlb = np.diag((ls * (ls + 1.0)) ** 2)
    AT_A = A.T @ A + lambda_lb * Rlb
    f = np.zeros(A.shape[1]); f[:n4] = np.linalg.pinv(A[:, :n4]) @ S
    thr = tau * sphere_Y[0, 0] * f[0]
    neg = sphere_Y @ f < thr
    for _ in range(max_iter):
        L = sphere_Y[neg]
        Q = AT_A + lambda_pos * (L.T @ L)
        f = np.linalg.solve(Q + 1e-8 * np.eye(Q.shape[0]), A.T @ S)
        new = sphere_Y @ f < thr
        if np.array_equal(new, neg):
            break
        neg = new
    return f


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("batman"); ap.add_argument("out"); ap.add_argument("wm_pack")
    ap.add_argument("--TE", type=float, default=0.100); ap.add_argument("--delta", type=float, default=0.025)
    ap.add_argument("--Delta", type=float, default=0.055); ap.add_argument("--afd", type=float, default=0.15)
    a = ap.parse_args(argv)
    import nibabel as nib
    fod = read_mif(os.path.join(a.batman, "wmfod_norm.mif"))
    grid, R = Grid.from_oblique_affine(fod.affine, fod.shape[:3])
    seq, g = acquisition(a.batman, R, grid, TE=a.TE, delta=a.delta, Delta=a.Delta)
    S = nib.load(os.path.join(a.out, "synthetic_dwi.nii.gz")).get_fdata()
    from dmipy_sim.replay.phantom import read_rph
    ph = read_rph(os.path.join(a.out, "batman_brain.rph"))
    f_wm = ph.to_volume(ph.fraction(0), fill=0.0)
    pack = PackSubstrate(a.wm_pack, m0=1.0).pack
    A = kernel(pack, seq, g)
    sphere = fibonacci_sphere(); Ys = real_sh(8, sphere)
    c_in = rotate_sh(fod.data, R.T)
    live = np.argwhere((fod.data[..., 0] > a.afd) & (f_wm > 0.5) & (S[..., 0] > 0))
    angles, accs = [], []
    for i, j, k in live:
        s = S[i, j, k]
        f = csd(A, s / s[g[:, 3] == 0].mean(), Ys)
        pin, pout = sphere[np.argmax(Ys @ c_in[i, j, k])], sphere[np.argmax(Ys @ f)]
        angles.append(np.degrees(np.arccos(min(1.0, abs(pin @ pout)))))
        u, v = c_in[i, j, k][1:], f[1:]                                   # angular correlation: the anisotropic part
        accs.append(float(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-30)))
    angles, accs = np.array(angles), np.array(accs)
    print(f"CSD round trip on {len(angles)} WM voxels (AFD > {a.afd}, f_wm > 0.5): peak angle median {np.median(angles):.1f} deg, "
          f"< 10 deg {np.mean(angles < 10) * 100:.0f} %, < 20 deg {np.mean(angles < 20) * 100:.0f} %; "
          f"angular correlation median {np.median(accs):.3f}", flush=True)
    np.savez(os.path.join(a.out, "csd_roundtrip.npz"), voxels=live, angle_deg=angles, acc=accs)
    return angles, accs


if __name__ == "__main__":
    main()
