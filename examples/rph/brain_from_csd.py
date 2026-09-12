"""A brain from an MRtrix CSD result: the B.A.T.M.A.N. tutorial subject as a 3-D replay phantom in the bore.

Inputs are what the MRtrix3 tutorial by Marlene Tahedl (osf.io/fkyht, CC BY 4.0) produces and ships in its
``Supplementary_Files``: the multi-shell multi-tissue CSD white-matter FOD (``wmfod_norm.mif``, lmax 8,
MRtrix3 basis, scanner coordinates), the five-tissue segmentation coregistered to the diffusion data
(``5tt_coreg.mif``, on the T1 grid), the brain mask, and the gradient table (``grad.b``, scanner space).
Nothing here re-runs MRtrix; the ``.mif`` files are read directly (:mod:`dmipy_sim.io.mrtrix`).

The phantom's voxels are the image's. The acquisition was prescribed oblique (a few degrees), so
:meth:`Grid.from_oblique_affine` hands back the image grid and the rotation ``R`` (image -> scanner), and
everything given in scanner coordinates -- the FOD's harmonics, the gradient directions, the field -- is rotated
by ``R.T`` into the image frame. Nothing is resampled except the 5TT fractions, which live on the 1 mm T1 grid
and are averaged into each 2.5 mm voxel (partial volume by construction).

    python examples/rph/brain_from_csd.py <batman_dir> <wm_pack.rpk> [--gm-pack gm.rpk] [--out out_dir]
                                          [--TE 0.100 --delta 0.025 --Delta 0.055] [--slab k0 k1] [--snr 30]
"""
import argparse
import os
import sys
import time

import numpy as np

from dmipy_sim import Prescription, sequences
from dmipy_sim.io.mrtrix import read_mif
from dmipy_sim.phantom import FreeWater, Grid, Inert, ODF, PackSubstrate, Phantom
from dmipy_sim.replay.so3 import rotate_sh

# 5TT columns (MRtrix): cortical GM, sub-cortical GM, WM, CSF, pathological tissue
GM_COLS, WM_COL, CSF_COL, PATH_COL = (0, 1), 2, 3, 4
from dmipy_sim.substrate.biophysical_constants import get_value
M0 = {"wm": get_value("proton_density_white_matter"), "gm": get_value("proton_density_grey_matter"),
      "csf": get_value("proton_density_csf")}                       # water content relative to CSF, cited in the table


def fractions_on(grid_img, target_affine, tt_img, sub=3):
    """The five tissue fractions averaged into every voxel of the target image: ``sub^3`` points per voxel mapped
    through both affines and sampled trilinearly on the T1 grid. Returns ``(nx, ny, nz, 5)``."""
    from scipy.ndimage import map_coordinates
    nx, ny, nz = grid_img
    off = (np.arange(sub) + 0.5) / sub - 0.5
    o = np.stack(np.meshgrid(off, off, off, indexing="ij"), -1).reshape(-1, 3)          # (sub^3, 3)
    ijk = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1).reshape(-1, 3)
    pts = (ijk[:, None, :] + o[None, :, :]).reshape(-1, 3)                               # (N sub^3, 3) target voxel coords
    xyz = pts @ target_affine[:3, :3].T + target_affine[:3, 3]                           # scanner mm
    inv = np.linalg.inv(tt_img.affine)
    src = xyz @ inv[:3, :3].T + inv[:3, 3]                                               # T1 voxel coords
    out = np.zeros((pts.shape[0], 5))
    for t in range(5):
        out[:, t] = map_coordinates(tt_img.data[..., t], src.T, order=1, mode="constant", cval=0.0)
    return out.reshape(nx, ny, nz, sub ** 3, 5).mean(axis=3)


def build(batman, wm_pack, gm_pack=None, slab=None):
    t0 = time.time()
    fod = read_mif(os.path.join(batman, "wmfod_norm.mif"))
    tt = read_mif(os.path.join(batman, "5tt_coreg.mif"))
    mask = read_mif(os.path.join(batman, "mask_den_unr_preproc_unb.mif")).data
    grid, R = Grid.from_oblique_affine(fod.affine, fod.shape[:3])
    print(f"grid {grid.shape} at {np.round(np.array(grid.voxel_size_m) * 1e3, 2)} mm, image axes tilted "
          f"{np.degrees(np.arccos((np.trace(R) - 1) / 2)):.2f} deg in the bore", flush=True)
    F = fractions_on(fod.shape[:3], fod.affine, tt)                                        # (nx, ny, nz, 5)
    F *= mask[..., None]
    c = rotate_sh(fod.data, R.T)                                                           # the FOD in the image frame
    f_wm = F[..., WM_COL] * (fod.data[..., 0] > 0)                                         # no FOD, no oriented WM
    f_gm = F[..., GM_COLS[0]] + F[..., GM_COLS[1]] + F[..., WM_COL] * (fod.data[..., 0] <= 0)
    f_csf = F[..., CSF_COL]
    if slab is not None:                                                                   # a few slices, for speed
        keep = np.zeros(grid.shape, bool); keep[:, :, slab[0]:slab[1]] = True
        f_wm, f_gm, f_csf = f_wm * keep, f_gm * keep, f_csf * keep
    wm = PackSubstrate(wm_pack, m0=M0["wm"], name="wm")
    gm = PackSubstrate(gm_pack, m0=M0["gm"], name="gm") if gm_pack else FreeWater(D_m2_s=0.8e-9, m0=M0["gm"], name="gm/stand-in")
    csf = FreeWater(D_m2_s=3.0e-9, m0=M0["csf"])
    orientation = {wm: ODF(c, basis="mrtrix3")}
    if gm_pack:
        from dmipy_sim.replay.fod import FOD
        iso = np.broadcast_to(FOD.isotropic(lmax=8).coeffs, grid.shape + (45,)).copy()
        orientation[gm] = ODF(iso, basis="mrtrix3")                                         # a cortex has no fibre axis
    ph = Phantom.compose(grid, fractions={wm: f_wm, gm: f_gm, csf: f_csf}, remainder=Inert(), orientation=orientation)
    print(f"{ph!r}\n  built in {time.time() - t0:.0f} s; WM {int((f_wm > 0).sum())}, GM {int((f_gm > 0).sum())}, "
          f"CSF {int((f_csf > 0).sum())} voxels", flush=True)
    return ph, fod, R


def acquisition(batman, R, grid, *, TE, delta, Delta):
    """The tutorial's own gradient table (scanner space, b in s/mm^2) as one PGSE prescribed on the image."""
    g = np.loadtxt(os.path.join(batman, "dwipreproc_grad.b"))
    dirs_s, b = g[:, :3], g[:, 3] * 1e6
    n = np.linalg.norm(dirs_s, axis=1)
    dirs_s = np.where(n[:, None] > 0, dirs_s / np.where(n[:, None] > 0, n[:, None], 1.0), [0.0, 0.0, 1.0])
    dirs_img = dirs_s @ R                                                                  # R^T g, row-wise
    seq = sequences.pgse(dirs_img, delta, Delta, bvalues=b, TE=TE)
    return seq.with_prescription(Prescription(isocenter_m=(0.0, 0.0, 0.0), axes=grid.axes, voxel_size_m=grid.voxel_size_m,
                                              matrix=grid.shape, origin_m=grid.origin_m)), g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("batman"); ap.add_argument("wm_pack"); ap.add_argument("--gm-pack", default=None)
    ap.add_argument("--out", default=None); ap.add_argument("--TE", type=float, default=0.100)
    ap.add_argument("--delta", type=float, default=0.025); ap.add_argument("--Delta", type=float, default=0.055)
    ap.add_argument("--slab", type=int, nargs=2, default=None); ap.add_argument("--snr", type=float, default=None)
    a = ap.parse_args(argv)
    out = a.out or os.path.dirname(os.path.abspath(__file__))
    ph, fod, R = build(a.batman, a.wm_pack, a.gm_pack, a.slab)
    seq, g = acquisition(a.batman, R, ph.grid, TE=a.TE, delta=a.delta, Delta=a.Delta)
    t0 = time.time()
    S = ph.replay(seq, b0_dir=tuple(R.T @ np.array([0.0, 0.0, 1.0])))
    print(f"  replayed {seq.n_meas} measurements in {time.time() - t0:.0f} s", flush=True)
    S = np.nan_to_num(S).astype(np.float32)
    if a.snr:                                                                              # SNR of the brain's mean b = 0
        from dmipy_sim.acquisition.noise import add_rician_noise
        b0 = S[..., g[:, 3] == 0]
        sigma = float(b0[b0 > 0].mean()) / a.snr
        S = add_rician_noise(S, sigma, seed=0).astype(np.float32)
    import nibabel as nib
    nib.save(nib.Nifti1Image(S, fod.affine), os.path.join(out, "synthetic_dwi.nii.gz"))
    np.savetxt(os.path.join(out, "synthetic_dwi.b"), g, fmt="%.6f")
    ph.write(os.path.join(out, "batman_brain.rph"), id="phantoms/batman-brain", license="CC-BY-4.0",
             citation="Tahedl M. B.A.T.M.A.N.: Basic and Advanced Tractography with MRtrix for All Neurophiles. OSF 2018, "
                      "doi:10.17605/OSF.IO/FKYHT (tutorial subject, CC BY 4.0); dmipy-sim replay phantom", embed=False)
    print(f"  wrote synthetic_dwi.nii.gz {S.shape}, synthetic_dwi.b, batman_brain.rph in {out}", flush=True)
    return ph, seq, S


if __name__ == "__main__":
    main()
