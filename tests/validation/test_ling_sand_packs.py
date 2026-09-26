"""Ling et al. 2022's quartz/garnet sand packs, against the CPMG measured on the same packs.

Ling, Hussaini, Elsayed, Connolly, El-Husseiny, Mahmoud, May & Johns (2022), *Model Synthetic Samples
for Validation of NMR Signal Simulations*, Transport in Porous Media 142(3):623-639,
https://doi.org/10.1007/s11242-022-01764-w (CC BY 4.0) measured a CPMG on five resin-set packs of
sieved quartz and garnet sand and then micro-CT imaged **those same packs**, releasing the images and
the raw echo trains together (figshare https://doi.org/10.6084/m9.figshare.17161730, CC BY 4.0). That
makes it the one label-volume reference where the measurement and the geometry are the same physical
object, and the second reproduction this geometry is validated by, beside Talabi's Imperial rocks.

Neither the images nor the workbooks are in the repository. Point ``DMIPY_SIM_LING2022_DIR`` at a
directory holding the extracted ``450CubicEqSized`` ``.am`` files, and ``DMIPY_SIM_LING2022_CPMG_DIR``
at a directory of ``<sample>.npz`` written by ``ling-sand-packs/extract_cpmg.py`` from
``CPMG data.zip``. The producer script is ``examples/validation/ling_sand_packs.py``.

What is asserted and what is not. Their surface relaxivities are **theirs** and fitted by them (12.5
and 98.5 um/s, SS3.2, with an independent maximal-ball estimate of 22 and 80 um/s in the same section);
the brine's bulk T2 is **ours**, because the paper states none; and their paper prints no T2 number at
all -- its comparison is Fig. 5, a figure -- so the measured number here is the log-mean of THEIR
released te = 100 us echo train, inverted by the same estimator on the same 1 ms grid as ours.
"""
import os

import numpy as np
import pytest

DATA = os.environ.get("DMIPY_SIM_LING2022_DIR")
CPMG = os.environ.get("DMIPY_SIM_LING2022_CPMG_DIR")


def script():
    """The reproduction script, imported on first USE: a module-level import runs during COLLECTION, on
    every invocation of the suite, including the fast lane that deselects this module."""
    import importlib
    return importlib.import_module("examples.validation.ling_sand_packs")


pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not DATA, reason="set DMIPY_SIM_LING2022_DIR to the Ling 2022 images")]

#: What this walk measured on the two PURE packs: 19,998 walkers, dt = 500 us x 8 sub-steps, the pack's
#: own released window, at Ling's relaxivity for that mineral. The same numbers the family's
#: ``records/`` carries and its gate reads.
#:
#: A mixture has two mineral surfaces and two of their relaxivities, and this geometry accumulates ONE
#: boundary local time over both walls, so no single-rho walk of a mixture is their simulation and none
#: is asserted here (dmrai-lab/dmipy-sim#491).
OURS = {
    "6_Q100": dict(window=3.5, porosity=0.38896, s_over_v=76606, T2_fd=0.77465,
                   T2_lm=0.85304, T2_lm_measured=0.72708, ratio_to_theirs=1.1733,
                   rho_V_over_S_over_D=0.071),
    "1_G100": dict(window=2.0, porosity=0.39893, s_over_v=71960, T2_fd=0.13475,
                   T2_lm=0.22118, T2_lm_measured=0.17845, ratio_to_theirs=1.2394,
                   rho_V_over_S_over_D=0.596),
}


@pytest.mark.parametrize("sample", sorted(OURS))
def test_the_released_sub_volume_measures_as_we_recorded_it(sample):
    """Porosity and staircase S/V of the released 450^3 sub-volume.

    The substrate, not the walk, so it is checked exactly -- to a part in 1e4 and 1e3 of our own
    recorded numbers. There is nothing of Ling's to check them against: the paper reports no porosity
    and no S/V per sample, and its Table 1 could not be read through any access path available, so these
    are ours and are stated as ours. The porosity of the five packs spans 0.382-0.399 and is NOT
    monotone in garnet content.
    """
    ling = script()
    from dmipy_sim.geometry import LabelVolume
    from dmipy_sim.io.label_volume import read_label_volume
    v = read_label_volume(ling.image_path(sample, DATA))
    h = np.asarray(v.voxel_size, float)
    g = LabelVolume(v.labels, h, origin=-0.5 * np.asarray(v.labels.shape) * h,
                    pools=dict(ling.LING[sample]["pools"]))
    assert g.porosity() == pytest.approx(OURS[sample]["porosity"], abs=1e-4)
    assert g.surface_to_volume() == pytest.approx(OURS[sample]["s_over_v"], rel=1e-3)
    # The header's voxel size, (BoundingBox span) / (n - 1), against the paper's stated 3.93 um.
    assert np.allclose(h, 3.93e-6, rtol=3e-3)


@pytest.mark.skipif(not CPMG, reason="set DMIPY_SIM_LING2022_CPMG_DIR to the released echo trains")
@pytest.mark.parametrize("sample", sorted(OURS))
def test_the_T2_decay_is_what_we_recorded_against_the_CPMG_of_the_same_pack(sample):
    """One walk per pure pack at Ling's relaxivity, inverted to a T2 distribution, beside the log-mean of
    the CPMG they measured on that same pack.

    This test locks the REPRODUCTION, not an agreement claim. It does not agree to within the walk's
    standard error and it is not made to: with their fitted relaxivity, our Manhattan S/V and our
    declared T2B = 3.0 s, the log-mean is **853 ms against their 727 ms (+17.3 %)** on the quartz pack
    and **221 ms against their 178 ms (+24 %)** on the garnet one, at standard errors of 0.4 % and
    0.8 %, so neither gap is noise.

    Bulk T2 cannot explain it -- the T2B that would close each gap is 1.88 s and 0.82 s, both far below
    any brine's -- and the two gaps are not independent: subtracting ``1/T2B`` from each rate leaves a
    surface term that is **1.248** and **1.263** times ours, so ONE factor of about 1.25 on their fitted
    relaxivity reconciles both packs, whose relaxivities differ by a factor of eight. The gap lives in
    the relaxivity or in the surface it was fitted against, not in the walk: their own §3.2 gives
    12.5 um/s from their T2 fit and 22 um/s from their maximal-ball S/V, a factor of 1.8, which is the
    size of the question. So the assertion is on OUR number and on THEIR number, each to its own
    precision, and on the ratio we measured; a change that moves either has to move this line and say why.

    What the walk must get right and a formula cannot: ``rho (V/S) / D`` is 0.071 for the quartz pack and
    0.596 for the garnet one, so ``1 / (rho S/V + 1/T2B)`` is within 10 % for the first and 39 % low for
    the second. The log-mean sits ABOVE the single rate on both, by far more on the garnet pack.
    """
    ling = script()
    ref = OURS[sample]
    r = ling.run_pack(sample, DATA, n_walkers=19_998, T_max=ref["window"], sample_ms=1.0,
                      walker_batch=25_000, seed=0, halves=6)
    m = ling.measured_log_mean(sample, CPMG, sample_ms=1.0)
    assert r["T2_lm"] == pytest.approx(ref["T2_lm"], rel=0.02)
    assert m["T2_lm"] == pytest.approx(ref["T2_lm_measured"], rel=1e-3)      # their data, not our walk
    assert r["T2_lm"] / m["T2_lm"] == pytest.approx(ref["ratio_to_theirs"], rel=0.03)
    assert r["T2_fd"] == pytest.approx(ref["T2_fd"], rel=1e-3)
    assert r["rho_V_over_S_over_D"] == pytest.approx(ref["rho_V_over_S_over_D"], rel=1e-2)
    assert r["T2_lm"] > r["T2_fd"]                        # multi-exponential, so above the single rate
    assert np.all(np.diff(r["S"]) <= 1e-6)                # a relaxation decay is monotone
    assert r["floor"] < 0.02                              # the log-mean's own standard error


@pytest.mark.skipif(not CPMG, reason="set DMIPY_SIM_LING2022_CPMG_DIR to the released echo trains")
def test_the_released_echo_trains_are_what_the_family_says_they_are():
    """The 32 echo spacings and the extent of the shortest train, read off the released data.

    The paper's SS2.2 says every train was acquired for 3500 ms. The released te = 100 us trains hold
    35,000 echoes to 3.5 s for the two quartz-rich packs and 20,000 echoes to 2.0 s for the three
    garnet-rich ones, and the paper states no echo count at all. Both facts are the family's, measured
    here rather than quoted.
    """
    ling = script()
    extent = {"6_Q100": (35000, 3.5), "5_Q75G25": (35000, 3.5),
              "4_Q50G50": (20000, 2.0), "2_Q25G75": (20000, 2.0), "1_G100": (20000, 2.0)}
    for sample, (n_echo, t_last) in extent.items():
        z = np.load(os.path.join(CPMG, f"{sample}.npz"))
        te = np.asarray(z["te"], float)
        assert te.size == 32 and np.isfinite(te).all()
        assert te[0] == pytest.approx(100e-6) and te[-1] == pytest.approx(20e-3)
        assert np.all(np.diff(te) > 0)
        m = ling.measured_log_mean(sample, CPMG)
        assert (m["n_echoes"], m["t_last"]) == (n_echo, pytest.approx(t_last))
        assert m["S_last_over_S0"] < 0.03                 # every train is down to a few per cent
