"""The round trip of brain_from_csd.py: deconvolve the synthetic DWI with dmipy-fit and compare the recovered FOD
with the one that built the phantom, voxel by voxel.

Needs ``dmipy-fit`` (``pip install dmipy-fit``): the estimator is fit's business. The white-matter kernel is the
pack's own single-fibre signal -- the pack replayed at the identity pose on the acquisition's directions -- handed to
fit's anisotropic tissue-response estimator exactly as a single-fibre voxel of real data would be; the FOD is then
fit's spherical-harmonics model with the CSD solver. Single tissue: the GM and CSF terms of a voxel are the isotropic
contamination a single-fibre CSD always sees, so the comparison is made on white-matter-dominated voxels.

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
from dmipy_sim.replay.so3 import real_sh, rotate_sh


def fibonacci_sphere(n=724):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def fit_fods(pack, seq, S, voxels):
    """FOD harmonics (fit's CSD, MRtrix3 basis) of the given voxels of a synthetic DWI, with the pack as kernel."""
    from dmipy_fit.core.acquisition_scheme import AcquisitionScheme
    from dmipy_fit.core.spherical_harmonics_framework import MultiCompartmentSphericalHarmonicsModel
    from dmipy_fit.signal_models.tissue_response_models import estimate_TR2_anisotropic_tissue_response_model
    scheme = AcquisitionScheme(seq)
    single_fibre = np.real(pack.replay(seq, orientation=np.eye(3), complex_signal=True))[None, :]
    _S0, tr2 = estimate_TR2_anisotropic_tissue_response_model(scheme, single_fibre)     # (S0, model)
    model = MultiCompartmentSphericalHarmonicsModel(models=[tr2], sh_order=8)
    data = np.stack([S[tuple(v)] for v in voxels])
    fitted = model.fit(scheme, data, solver="csd", verbose=False)
    return np.asarray(fitted.fod_sh())                                    # (n_voxels, 45), dipy tournier non-legacy = MRtrix3


def compare(c_in, c_out, sphere):
    Ys = real_sh(8, sphere)
    pin = sphere[np.argmax(Ys @ c_in.T, axis=0)]; pout = sphere[np.argmax(Ys @ c_out.T, axis=0)]
    angle = np.degrees(np.arccos(np.clip(np.abs(np.sum(pin * pout, 1)), 0, 1)))
    u, v = c_in[:, 1:], c_out[:, 1:]                                      # angular correlation of the anisotropic part
    acc = np.sum(u * v, 1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-30)
    return angle, acc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("batman"); ap.add_argument("out"); ap.add_argument("wm_pack")
    ap.add_argument("--TE", type=float, default=0.100); ap.add_argument("--delta", type=float, default=0.025)
    ap.add_argument("--Delta", type=float, default=0.055); ap.add_argument("--afd", type=float, default=0.15)
    a = ap.parse_args(argv)
    import nibabel as nib
    from dmipy_sim.replay.phantom import read_rph
    fod = read_mif(os.path.join(a.batman, "wmfod_norm.mif"))
    grid, R = Grid.from_oblique_affine(fod.affine, fod.shape[:3])
    seq, g = acquisition(a.batman, R, grid, TE=a.TE, delta=a.delta, Delta=a.Delta)
    S = nib.load(os.path.join(a.out, "synthetic_dwi.nii.gz")).get_fdata()
    ph = read_rph(os.path.join(a.out, "batman_brain.rph"))
    f_wm = ph.to_volume(ph.fraction(0), fill=0.0)
    voxels = np.argwhere((fod.data[..., 0] > a.afd) & (f_wm > 0.5) & (S[..., 0] > 0))
    c_out = fit_fods(PackSubstrate(a.wm_pack, m0=1.0).pack, seq, S, voxels)
    c_in = rotate_sh(fod.data, R.T)[tuple(voxels.T)]
    angle, acc = compare(c_in, c_out, fibonacci_sphere())
    print(f"CSD round trip (dmipy-fit) on {len(voxels)} WM voxels (AFD > {a.afd}, f_wm > 0.5): peak angle median "
          f"{np.median(angle):.1f} deg, < 10 deg {np.mean(angle < 10) * 100:.0f} %, < 20 deg {np.mean(angle < 20) * 100:.0f} %; "
          f"angular correlation median {np.median(acc):.3f}", flush=True)
    np.savez(os.path.join(a.out, "csd_roundtrip.npz"), voxels=voxels, angle_deg=angle, acc=acc, fod_sh=c_out)
    return angle, acc


if __name__ == "__main__":
    main()
