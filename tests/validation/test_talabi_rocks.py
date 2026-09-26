"""The Imperial 2007 micro-CT rocks against Talabi's random-walk NMR, on the same voxels.

Talabi (2008), *Pore Scale Simulation of NMR Response in Porous Media*, Imperial College London PhD
thesis (http://hdl.handle.net/10044/1/4261), ran a random walk for T2 on the released segmentations of
the Imperial 2007 collection (Dong & Blunt 2009; Figshare, CC BY 4.0 per record) with
``D0 = 2.07e-9 m^2/s``, ``T2B = 3.1 s``, 2,000,000 walkers and a central 300^3 crop, at a surface
relaxivity fitted to his own CPMG for the sand packs (41 um/s) and taken from the literature for Berea
(15 um/s). That makes it the one reference measurement for a label-volume substrate made on the very
voxels a consumer can download, so it is the reproduction this geometry is validated by.

The images are not in the repository. Point ``DMIPY_SIM_IMPERIAL2007_DIR`` at a directory holding
``LV60A.nhdr`` + ``LV60A.raw`` (and ``F42A``, ``Berea``) to run this; the producer script is
``examples/validation/talabi_micro_ct_rocks.py``.

Measured (200,000 walkers, dt 500 us, step 2.49 um, the 0-3 s window he plots, ~290 s per rock on 8
CPU threads):

| rock  | porosity ours / his | S/V (1/m) ours / his | T2lm ours | his simulated | his measured |
|-------|---------------------|----------------------|-----------|---------------|--------------|
| LV60A | 0.3688 / 0.377      | 59,821 / 57,670      | 487.6 ms  | 512 ms        | 496 ms       |
| F42A  | 0.3306 / 0.330      | 44,432 / 43,770      | 679.0 ms  | 677 ms        | 668 ms       |
| Berea | 0.1967 / 0.196      | 121,198 / 118,960    | 540.2 ms  | 584 ms        | -            |
"""
import os

import numpy as np
import pytest

DATA = os.environ.get("DMIPY_SIM_IMPERIAL2007_DIR")


def script():
    """The reproduction script, imported on first USE: a module-level import runs during COLLECTION,
    on every invocation of the suite, including the fast lane that deselects this module."""
    import importlib
    return importlib.import_module("examples.validation.talabi_micro_ct_rocks")


pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not DATA, reason="set DMIPY_SIM_IMPERIAL2007_DIR to the Imperial 2007 images")]

#: What this walk measured, per rock: porosity, staircase S/V (1/m) and the log-mean T2 (s) of the
#: inverted decay, at 200,000 walkers and dt = 500 us on the central 300^3 crop.
OURS = {
    "LV60A": dict(porosity=0.3688, s_over_v=59821, T2_lm=0.4876),
    "F42A": dict(porosity=0.3306, s_over_v=44432, T2_lm=0.6790),
    "Berea": dict(porosity=0.1967, s_over_v=121198, T2_lm=0.5402),
}


@pytest.mark.parametrize("rock", sorted(OURS))
def test_the_crop_measures_as_talabi_tabulated(rock):
    """Porosity and staircase S/V of the central 300^3 crop, against his Table 7-1 / Table 8-4.

    The two are the substrate, not the walk, so they are checked first and exactly: our own numbers to
    a part in 1e3, and his to 4 % in porosity and 4 % in S/V. The residual is the crop offset, which
    the thesis never records -- it says only "a central cubic section of voxel size 300^3" -- and
    LV60A is the one rock where a central cube does not land on his porosity (0.3688 against 0.377,
    -2.2 %), which no offset within a few voxels closes.
    """
    talabi = script()
    ref, ours = talabi.TALABI[rock], OURS[rock]
    path = os.path.join(DATA, f"{rock}.nhdr")
    from dmipy_sim.geometry import LabelVolume
    from dmipy_sim.io.label_volume import crop_labels, read_label_volume
    v = read_label_volume(path)
    crop = talabi.central_crop(v.labels.shape, ref.get("crop", talabi.CROP))
    g = LabelVolume(crop_labels(v, crop).labels, v.voxel_size)
    assert g.porosity() == pytest.approx(ours["porosity"], abs=1e-4)
    assert g.surface_to_volume() == pytest.approx(ours["s_over_v"], rel=1e-3)
    assert g.porosity() == pytest.approx(ref["porosity"], rel=0.04)
    assert g.surface_to_volume() == pytest.approx(ref["s_over_v"], rel=0.04)


@pytest.mark.parametrize("rock", sorted(OURS))
def test_the_T2_decay_reproduces_his_simulated_mean_T2(rock):
    """One walk per rock at his D0, T2B, relaxivity and crop, inverted to a T2 distribution.

    The log-mean T2 lands within 8 % of his simulated value on every rock (LV60A -4.8 %, F42A +0.3 %,
    Berea -7.5 %) and 1.7 % / 1.6 % of his *measured* T2lm on the two sand packs he measured. It is
    NOT the fast-diffusion value ``1 / (rho S/V + 1/T2B)`` -- 360 / 466 / 467 ms -- and must not be:
    ``rho (V/S) / D`` is 0.33 for LV60A, so a rock's pore-size distribution decays multi-exponentially
    and its log-mean weights the small pores far less than one rate does. That is why this rung needs
    the Laplace inversion his Table 7-2 was read from and not a single-exponential fit.

    The residual is dominated by two things this reproduction cannot remove: the crop offset (see the
    porosity test), and the difference between his discrete surface rule and this one -- he kills a
    walker at an attempted move into a grain voxel with probability ``2 rho s / (3 D)`` = 0.026 at his
    2 um step, this one weights the walker by ``exp(-2 (rho/D) d_perp)`` per specular reflection, whose
    accumulated local time is ``D (S/V) T`` exactly (tests/geometry/test_label_volume.py). The two
    agree in the continuum limit; at his step they need not agree to better than a few per cent.
    """
    talabi = script()
    ref, ours = talabi.TALABI[rock], OURS[rock]
    r = talabi.run_rock(rock, DATA, n_walkers=200_000, T_max=3.0, sample_ms=1.0, sub_echo=2,
                        walker_batch=25_000, seed=0)
    assert r["T2_lm"] == pytest.approx(ours["T2_lm"], rel=0.02)
    assert r["T2_lm"] == pytest.approx(ref["T2_sim"], rel=0.08)
    if ref["T2_exp"] is not None:
        assert r["T2_lm"] == pytest.approx(ref["T2_exp"], rel=0.10)
    assert r["T2_lm"] > r["T2_fd"]                    # multi-exponential, so above the single rate
    assert np.all(np.diff(r["S"]) <= 1e-6)            # a relaxation decay is monotone
