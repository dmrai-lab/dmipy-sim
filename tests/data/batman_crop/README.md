# BATMAN crop

A 10 x 10 x 3 voxel crop (voxels 40-49, 48-57, 30-32 of the 96 x 96 x 60 diffusion grid) of the derived images of the
B.A.T.M.A.N. tutorial subject -- Marlene Tahedl, *B.A.T.M.A.N.: Basic and Advanced Tractography with MRtrix for All
Neurophiles*, OSF 2018, https://osf.io/fkyht/ (CC BY 4.0) -- for `tests/test_brain_from_csd.py`:

* `wmfod_norm.mif`: the multi-shell multi-tissue CSD white-matter FOD (lmax 8, MRtrix3 basis), the crop's own affine;
* `5tt_coreg.mif`: the matching region of the five-tissue segmentation on the 1 mm T1 grid, plus a margin;
* `mask_den_unr_preproc_unb.mif`: the brain mask; `dwipreproc_grad.b`: the full gradient table.

Written by `dmipy_sim.io.mrtrix.write_mif` from the tutorial's `Supplementary_Files`; nothing was resampled.
