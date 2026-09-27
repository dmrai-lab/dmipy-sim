"""The Coronado-Leija 2024 reference family's DECLARATIONS (examples/substrate_bank/build_coronado_leija.py).

Nothing of the study's 21 GB is needed and nothing here reaches the network: the released files are replaced
by empty ones of the same names, the licence text by a cached copy, the Crossref resolver by a stub, and the
measurement of ``f`` by its own cache, keyed with the module's own :func:`cache_key`. What is asserted is what
the family SAYS -- the regions it reads from the authors' released limits, the grade its reference record
earns by rule, and the three refusals that stand in for the stages it cannot run: the walk's (dmipy-sim#478),
the spec producer's, and the pack's.

The sub-step counts the walk's refusal quotes are checked against
:func:`dmipy_sim.engine.physics.resolve_sub_steps` itself, because those numbers are the whole argument for
waiting: a refusal that quotes a stale cost is a refusal nobody can act on.
"""
import json
import os

import numpy as np
import pytest

from dmipy_sim.replay.reference import ReferenceRefusal, grade_of, grade_reason

B = pytest.importorskip("examples.substrate_bank.build_coronado_leija")

LICENCE = "SYNTHETIC CC BY 4.0 LEGALCODE\n\n" + ("This stands in for the text fetched from the host. " * 20)
NYU_LICENCE = "SYNTHETIC NYU NON-COMMERCIAL RESEARCH LICENCE\n\n" + ("Non-commercial research only. " * 20)


def _limits():
    """A 19 x 10 ROI-limits table of the authors' own shape: animal, hemisphere, counts, size, cc, cg.

    Column 3 (the sham #49 ipsilateral sample) carries the ``1 2`` corpus-callosum limits the authors write
    for a volume where that tract was not imaged, which their own ``> 10`` voxel rule excludes.
    """
    rows = np.zeros((19, len(B.SAMPLES)), int)
    for col, sample in enumerate(B.SAMPLES):
        animal = sample.split("_")[1]
        nx, ny, nz = 40 + col, 60 + col, 20 + col
        rows[:, col] = [int(animal), int(sample.endswith("ipsi")), 1000 + col, 10 + col, nx, ny, nz,
                        1, nx, 1, ny // 2, 1, nz,                      # cc: the lower half in y
                        1, nx, ny // 2 + 1, ny, 1, nz]                 # cg: the upper half in y
    rows[7:13, 3] = [1, 2, 1, 2, 1, 2]                                 # sham_49_ipsi has no corpus callosum
    return rows


def _write_limits(path, rows=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, _limits() if rows is None else rows, fmt="%d")
    return path


def _data_dir(tmp_path):
    """A directory of empty files at every path the family's source record names."""
    data = tmp_path / "data"
    for sample in B.SAMPLES:
        d = data / "em" / "White matter EM" / sample
        d.mkdir(parents=True, exist_ok=True)
        stem = sample.split("_", 1)[1]
        for res in ("LM", "HM"):
            for role in ("myelin", "myelinated_axons") + (("nucleus",) if res == "LM" else ()):
                (d / f"{res}_{stem}_{role}.mat").write_bytes(b"")
    (data / "em" / "White matter EM" / "README.docx").write_bytes(b"")
    for animal in B.ANIMALS:
        pre = data / "mri" / "Ex vivo diffusion" / "MRI_data" / "preproc" / animal
        pre.mkdir(parents=True, exist_ok=True)
        for n in ("ants.bval", "ants.bvec", "ants.txt", "ants.nii.gz", "dwi_mask.nii.gz", "ref_mask.nii.gz",
                  "ref_vol.nii.gz", "sigma.nii.gz"):
            (pre / n).write_bytes(b"")
        roi = data / "mri" / "Ex vivo diffusion" / "MRI_data" / "rois" / animal
        roi.mkdir(parents=True, exist_ok=True)
        for n in ("cc_contra.nii", "cc_ipsi.nii", "cg_contra.nii", "cg_ipsi.nii", "cccg_contra.nii",
                  "cccg_ipsi.nii"):
            (roi / n).write_bytes(b"")
    (data / "mri" / "Ex vivo diffusion" / "MRI_data" / "README.docx").write_bytes(b"")
    nyu = data / "nyu-analysis"
    (nyu / "EM_analysis").mkdir(parents=True, exist_ok=True)
    _write_limits(str(nyu / "EM_analysis" / "000_animal_exp_axons_cells_nx_ny_nz_cclims_cglims.txt"))
    (nyu / "EM_analysis" / "script03_compute_volume_fractions.m").write_bytes(b"")
    (nyu / "EM_analysis" / "RefineSeg.m").write_bytes(b"")
    (nyu / "LICENSE").write_text(NYU_LICENCE)
    return str(data)


def _seed_caches(work, data, rois, substrates):
    """The licence copy and the measurement of ``f``, so no stage fetches or reads 400 GB of gzip.

    The measurement is written under the module's OWN cache key, so the test cannot pass by writing a key the
    module would not look up.
    """
    os.makedirs(os.path.join(work, ".licence-cache"), exist_ok=True)
    with open(os.path.join(work, ".licence-cache", "cc-by-4.0-legalcode.txt"), "w") as fh:
        fh.write(LICENCE)
    cache = {}
    for i, s in enumerate(substrates):
        files = B.em_files(data, rois[s]["sample"])
        f = 0.30 + 0.01 * i
        n_water = 2_300_000
        cache[s] = dict(key=B.cache_key(files, rois[s]["crop"]),
                        value=dict(substrate=s, f_voxel_count=f, f_direct=f * 1.001, n_water=n_water,
                                   n_points=B.N_POINTS,
                                   se_direct=float(((1.0 - f) / (f * n_water)) ** 0.5),
                                   files={k: dict(path=os.path.basename(v), sha256="0" * 64)
                                          for k, v in sorted(files.items())}))
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, "f-measurements.json"), "w") as fh:
        json.dump(cache, fh)


def _resolver(doi, **kw):
    """Crossref and DataCite, stubbed: the one DOI this family resolves, with a fixed timestamp."""
    assert doi == B.PAPER_DOI, doi
    return dict(doi=doi, title=B.PAPER_TITLE, type="journal-article", container="Imaging Neuroscience",
                agency="crossref", resolved="2026-01-01T00:00:00Z")


def _family(tmp_path, substrates=None):
    data = _data_dir(tmp_path)
    work = str(tmp_path / "family")
    rois = B.roi_limits(os.path.join(data, "nyu-analysis", "EM_analysis",
                                     "000_animal_exp_axons_cells_nx_ny_nz_cclims_cglims.txt"))
    names = list(substrates) if substrates else sorted(rois)
    _seed_caches(work, data, rois, names)
    return B.family(data, work, substrates=names, resolver=_resolver), rois


# --------------------------------------------------------------------- the regions, as the authors wrote them
def test_the_released_limits_give_nineteen_regions_and_the_missing_corpus_callosum_excludes_itself(tmp_path):
    rois = B.roi_limits(_write_limits(str(tmp_path / "lims.txt")))
    assert len(rois) == 19, sorted(rois)
    assert "sham_49_ipsi_cg" in rois and "sham_49_ipsi_cc" not in rois
    assert sorted({r["sample"] for r in rois.values()}) == sorted(B.SAMPLES)
    assert {r["condition"] for r in rois.values()} == {"sham-operated", "traumatic brain injury"}


def test_the_crop_is_the_authors_one_based_inclusive_limits_read_half_open(tmp_path):
    rois = B.roi_limits(_write_limits(str(tmp_path / "lims.txt")))
    r = rois["sham_25_contra_cc"]
    nx, ny, nz = r["shape"]
    # their cc column is x 1..nx, y 1..ny//2, z 1..nz, one-based inclusive
    assert r["crop"] == (0, 0, 0, nx, ny // 2, nz)
    assert rois["sham_25_contra_cg"]["crop"] == (0, ny // 2, 0, nx, ny, nz)


def test_a_limits_file_of_another_shape_is_refused(tmp_path):
    with pytest.raises(ReferenceRefusal, match="19 rows by one column per sample"):
        B.roi_limits(_write_limits(str(tmp_path / "short.txt"), _limits()[:18]))


def test_a_column_whose_animal_does_not_match_the_sample_order_is_refused(tmp_path):
    rows = _limits()
    rows[0, 0] = 99
    with pytest.raises(ReferenceRefusal, match="the sample order says"):
        B.roi_limits(_write_limits(str(tmp_path / "wrong.txt"), rows))


# --------------------------------------------------------------------- the refusals that stand in for stages
def test_the_sub_step_counts_the_refusal_quotes_are_the_engines_own():
    """The cost of waiting for #478, recomputed with the rule that sets it.

    The declared table is what ``walk()``'s refusal quotes and what the PR's design note rests on; a stale
    number there is a refusal nobody can act on.
    """
    from dmipy_sim.engine.physics import MAX_SUB_STEPS, resolve_sub_steps
    from dmipy_sim.geometry.label_volume import LabelVolume
    lab = np.zeros((8, 8, 8), np.uint8)
    lab[2:6, 2:6, 2:6] = 1
    for (tier, dt), declared in sorted(B.SUB_STEPS.items()):
        vox = np.asarray(B.VOXEL_LM if tier == "lm" else B.VOXEL_HM)
        g = LabelVolume(lab, vox, pools={0: "extra", 1: "intra"}, pool="intra", origin=-4.0 * vox)
        if declared is None:
            with pytest.raises(ValueError, match=f"cap of {MAX_SUB_STEPS:,}"):
                resolve_sub_steps(g, B.D0, dt, surface=True)
        else:
            assert resolve_sub_steps(g, B.D0, dt, surface=True) == declared, (tier, dt)


def test_the_walk_refuses_by_name_with_the_number_that_makes_it_a_refusal(tmp_path):
    fam, _ = _family(tmp_path, ["sham_25_contra_cc"])
    with pytest.raises(ReferenceRefusal) as e:
        fam.build.walk(None, 10, n_t=2)
    assert "dmipy-sim#478" in str(e.value)
    assert f"{B.SUB_STEPS[('lm', 625e-6)]:,}" in str(e.value)
    assert f"{B.SUB_STEPS[('lm', B.DT_SAVE)]:,}" in str(e.value)


def test_the_chain_stops_at_the_spec_stage_and_the_design_stage_cannot_reach_its_pilot(tmp_path):
    """Two stages refuse, each in its own words, and neither is a note to a reader.

    The spec stage refuses first, because the producer cannot write this substrate; the design stage refuses
    after it, because a stage reads the record before it and there is none. The walk's own refusal -- the one
    that names #478 -- is what the design's pilot would meet if it got that far, which is why it is asserted
    directly (``test_the_walk_refuses_by_name_...``) rather than through a stage that cannot reach it.
    """
    fam, _ = _family(tmp_path, ["sham_25_contra_cc"])
    fam.stage("source")
    fam.stage("reference")
    with pytest.raises(ReferenceRefusal, match="label_volume_spec cannot write this substrate"):
        fam.stage("spec")
    with pytest.raises(ReferenceRefusal, match="spec.json is missing"):
        fam.stage("design")
    assert not fam.records.exists("spec") and not fam.records.exists("design")


def test_the_spec_producer_refuses_with_all_three_limits(tmp_path):
    fam, _ = _family(tmp_path, ["sham_25_contra_cc"])
    with pytest.raises(ReferenceRefusal) as e:
        fam.build.specs["sham_25_contra_cc"]()
    m = str(e.value)
    assert "released files" in m and "0..255" in m and "water_fraction 1.0 to exactly one pool" in m


@pytest.mark.parametrize("what", ["reproduce", "served_vs_channel", "snippet"])
def test_the_pack_and_card_callables_refuse_rather_than_estimate_from_a_pack_that_is_not_there(tmp_path, what):
    fam, _ = _family(tmp_path, ["sham_25_contra_cc"])
    fn = {"reproduce": lambda: fam.build.reproduce(None, None, None),
          "served_vs_channel": lambda: fam.build.served_vs_channel(None),
          "snippet": lambda: fam.publication.snippet("hf://nothing")}[what]
    with pytest.raises(ReferenceRefusal) as e:
        fn()
    assert "no pack" in str(e.value) or "no pack at" in str(e.value)


# --------------------------------------------------------------------- the records
def test_the_source_record_is_three_hosts_with_a_licence_text_and_a_digest_per_file(tmp_path):
    fam, rois = _family(tmp_path)
    rec = fam.stage("source")
    assert sorted(rec["sources"]) == ["fairdata-dmri", "fairdata-em", "nyu-analysis"]
    for key, s in rec["sources"].items():
        assert s["licence_copy"]["chars"] >= 400, key
        assert s["licence_stated"] and s["host_record"], key
        assert all(f["sha256"] and "bytes" in f for f in s["files"]), key
    em, mri, nyu = (rec["sources"][k] for k in ("fairdata-em", "fairdata-dmri", "nyu-analysis"))
    assert len(em["files"]) == 5 * len(B.SAMPLES) + 1          # LM triplet + HM pair per sample, and a README
    assert len(mri["files"]) == 14 * len(B.ANIMALS) + 1        # 8 preproc + 6 ROIs per brain, and a README
    assert len(nyu["files"]) == 4
    assert em["licence_id"] == mri["licence_id"] == "CC-BY-4.0"
    assert nyu["redistributes_bytes"] is False                 # a non-commercial licence is read, never shipped
    assert em["redistributes_bytes"] is False and mri["redistributes_bytes"] is False
    fam.records.check_chain("source")


def test_the_reference_record_earns_an_A_by_rule_because_no_parameter_of_ours_changes_the_geometry(tmp_path):
    fam, rois = _family(tmp_path)
    fam.stage("source")
    rec = fam.stage("reference")
    assert rec["published_kind"] == "data"                     # the released segmentation IS the measurement
    assert rec["sample_relation"] == "the same object"
    assert rec["same_released_geometry"] and rec["geometry_parameters_ours"] == []
    assert grade_of(rec) == "A"
    assert "released geometry" in grade_reason(rec)
    assert len(rec["quantities"]) == len(rois)
    whose = {p["name"]: p["whose"] for p in rec["parameters"]}
    assert whose["ROI limits"] == "theirs" and whose["their segmentation cleanup"] == "ours"
    for q in rec["quantities"]:
        assert q["name"] == "f_intra_axonal"
        assert q["published"]["uncertainty"] == 0.0 and q["published"]["uncertainty_is"]
        assert q["published"]["data_url"] and q["published"]["data_sha256"]
        assert q["direct"]["se_kind"] == "analytic_mean" and 0.0 < q["direct"]["se"] < 1.0
        assert q["direct"]["grid"]["n_points"] == B.N_POINTS
    fam.records.check_chain("reference")


def test_the_only_parameter_of_ours_that_touches_the_geometry_would_cost_the_A(tmp_path):
    """The grade is not a field: it follows from the parameters, and the record shows which way.

    Their ROI crop keeps the grade at ``A`` precisely because it is THEIRS. Marked ours, the same crop makes
    the substrate something other than the released geometry and the rule says ``B``.
    """
    fam, _ = _family(tmp_path, ["sham_25_contra_cc"])
    fam.stage("source")
    rec = fam.stage("reference")
    assert grade_of(rec) == "A"
    moved = dict(rec, geometry_parameters_ours=["ROI limits"], same_released_geometry=False)
    assert grade_of(moved) == "B"
    assert "ROI limits" in grade_reason(moved)
