"""The reference-pack protocol (dmipy_sim.replay.reference, dmipy-sim#482) and its card stage (#480), on a
synthetic family: a sphere large enough that its diffusion is free, whose reference quantity is therefore the
closed form ``exp(-b D)``.

Everything here is deterministic and runs on the CPU: one 300-walker walk, one pack, one pilot. What it covers
is the protocol's REFUSALS -- the whole point of the module -- the order rule, the record's write-once
behaviour, the grade rule, the gate's freedom from typed tolerances, and the card's executed snippet.
"""
import inspect
import json
import os
import re

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.replay import reference as R

D0 = 2e-9
R_SPHERE = 2e-4            # 200 um: sqrt(2 D T) is 4 um over the window, so the diffusion is free to < 1e-4
T_WIN = 4e-3
DT = 2.5e-4
N_T = 17
N_WALK = 300
B = 1e8
DELTA = 8e-4
BIGDELTA = 2.4e-3
SEED = 0
ENV = dict(bvals=[0.0, B], dirs=[[1, 0, 0]], ogse_periods=[2], shortd_b=B, shortd_deltas_frac=[0.05],
           delta_frac=0.2, Delta_frac=0.5, rho_list=[])
LICENCE_TEXT = "SYNTHETIC LICENCE\n\n" + ("This is the verbatim text of a licence that does not exist. " * 12)
GRID = dict(kind="none", solver="exp(-b D), the closed form of free diffusion")


def _seq():
    return d.pgse([[1, 0, 0]], DELTA, BIGDELTA, bvalues=[B], n_t=64, slew_rate=np.inf)


def waveforms():
    return (("pgse", _seq()),)


def _spec():
    return d.Sphere(radius=R_SPHERE).spec


def _spec_off():
    """A second free sphere, declared against a direct number that is 5 % off: the pack the gate must fail."""
    return d.Sphere(radius=0.75 * R_SPHERE).spec


def _walk(spec, n, *, n_t):
    from dmipy_sim.spec import geometry_from_spec
    return d.simulate_trajectories(int(n), D0, geometry_from_spec(spec), T_max=(n_t - 1) * DT, dt_save=DT,
                                   seed=SEED, tiers="all", require_gpu=False)


def _per_walker(pack):
    """The per-walker contribution to the ensemble mean: ``ew * E`` over ``sum(w)`` (replay's own reduction)."""
    w, ew, E = pack.walker_signals(_seq())
    return np.asarray(ew)[:, None] * np.asarray(E) / np.asarray(w).sum()


def _reproduce(pack, quantity, grid):
    """The replayed signal and the ANALYTIC standard error of its ensemble mean."""
    c = _per_walker(pack)[:, -1] * len(np.asarray(pack.walker_signals(_seq())[0]))
    value = float(abs(c.mean()))
    return dict(value=value, se=float(np.std(c.real, ddof=1) / np.sqrt(len(c)) / value),
                se_kind="analytic_mean", solver=grid["solver"],
                se_derivation="sd of the per-walker signal over sqrt(N), the standard error of the mean")


def _served(pack):
    a = float(np.asarray(pack.replay(_seq())).ravel()[-1])
    return abs(a - float(abs(_per_walker(pack)[:, -1].sum())))


def _snippet(uri):
    return (f"import numpy as np\n"
            f"import dmipy_sim as d\n"
            f"from dmipy_sim.replay import ReplayPack\n"
            f"pk = ReplayPack.load({uri!r})\n"
            f"seq = d.pgse([[1, 0, 0]], {DELTA!r}, {BIGDELTA!r}, bvalues=[{B!r}], n_t=64, slew_rate=np.inf)\n"
            f"print('S = %.5f' % float(np.asarray(pk.replay(seq)).ravel()[-1]))\n")


def _bad_snippet(uri):
    return "raise SystemExit('this snippet does not work')\n"


# --------------------------------------------------------------- the declaration
def _sources(tmp_path, *, licence_text=LICENCE_TEXT, host_record="synthetic v1", missing=False):
    p = tmp_path / "grain.raw"
    p.write_bytes(b"\x00\x01" * 64)
    return [R.Source(key="synthetic", url="https://example.invalid/record", host_record=host_record,
                     licence_id="SYNTHETIC-1.0", licence_url="https://example.invalid/licence",
                     licence_text=licence_text,
                     files=(R.SourceFile(path=str(tmp_path / ("absent.raw" if missing else "grain.raw")),
                                         cite_as="grain.raw", role="the synthetic input"),))]


def _quantity(direct_value, direct_se, *, grid=GRID, verbatim="S(b) = exp(-b D)", se_kind="analytic_mean",
              substrate="sphere", published=None):
    return R.ReferenceQuantity(
        substrate=substrate, name="signal_at_b",
        published=published or R.Published(value=float(np.exp(-B * D0)), unit="-", uncertainty=0.0,
                                           uncertainty_is="exact: it is a closed form",
                                           printed_in="the closed form of free diffusion", locator="exp(-b D)",
                                           verbatim=verbatim),
        direct=R.Direct(value=direct_value, unit="-", se=direct_se, se_kind=se_kind,
                        se_derivation="the closed form evaluated exactly, with the standard error the same "
                                      "walker count would give a Monte-Carlo estimate of it",
                        n_walkers=N_WALK, grid=grid, solver=grid["solver"],
                        source="the closed form; tests/replay/test_reference_family.py asserts it"))


def _reference(direct_value=None, direct_se=0.05, **kw):
    exact = float(np.exp(-B * D0))
    q = kw.pop("quantities", None) or (_quantity(direct_value or exact, direct_se),
                                       _quantity(1.25 * exact, 0.002, substrate="sphere-off"))
    base = dict(doi="10.1103/PhysRevE.80.036307", title="Pore-network extraction from "
                                                        "micro-computerized-tomography images",
                published_kind="analytic",
                sample="free diffusion in an unbounded medium", sample_relation="the same material",
                quantities=q,
                parameters=(R.FreeParameter(name="D", value=D0, unit="m^2/s", whose="ours",
                                            where="this test", how="chosen"),),
                description="a synthetic family", source_note="none", licence_note="synthetic")
    base.update(kw)
    return R.Reference(**base)


def _design(**kw):
    base = dict(window_s=T_WIN, dt_save_s=DT, save_grid_why="the test's own grid", K=8, envelope=ENV,
                waveforms=waveforms,
                tiers=(R.Tier(name="positions", floor_key="floor_max", err_key="err_max", target_floor=0.1),),
                memory_budget_bytes=200_000_000_000, pilot_n=N_WALK, safety=1.4,
                tolerance=R.Tolerance(terms=("quantity.direct.se", "quantity.replay_se",
                                             "quantity.published.uncertainty")),
                systematics=())
    base.update(kw)
    return R.Design(**base)


NO_REFERENCE = object()


def _family(tmp_path, *, sources=None, reference=NO_REFERENCE, design=None, build=None, publication=None,
            resolver=None):
    build = build or R.Build(specs={"sphere": _spec, "sphere-off": _spec_off},
                             pack_id={"sphere": "synthetic/sphere", "sphere-off": "synthetic/sphere-off"},
                             reproduce=_reproduce, served_vs_channel=_served, served_tier="positions",
                             walk=_walk)
    publication = publication or R.Publication(repo="owner/synthetic", licence="CC-BY-4.0",
                                               citation="a test family", snippet=_snippet,
                                               snippet_substrate="sphere", dry=True)
    return R.ReferenceFamily("synthetic", str(tmp_path), sources=sources or _sources(tmp_path),
                             reference=_reference() if reference is NO_REFERENCE else reference,
                             design=design or _design(), build=build, publication=publication,
                             resolver=resolver or _resolver)


def _resolver(doi, **kw):
    return dict(doi=doi, title="Pore-network extraction from micro-computerized-tomography images",
                type="journal-article", container="Physical Review E", resolved="2026-01-01T00:00:00Z")


# --------------------------------------------------------------- the happy path, once
@pytest.fixture(scope="module")
def ran(tmp_path_factory, monkeypatch_module):
    """The whole pipeline on the synthetic family, with the direct number taken from the closed form."""
    tmp = tmp_path_factory.mktemp("synthetic")
    fam = _family(tmp)
    monkeypatch_module.setattr(R, "code_commit", lambda **kw: "0" * 40)
    rec = fam.run()
    return fam, rec


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_every_stage_writes_its_record(ran):
    fam, rec = ran
    assert sorted(rec) == sorted(R.STAGES)
    for s in R.STAGES:
        assert os.path.exists(fam.records.path(s)), s
        assert rec[s]["stage"] == s
    # `sphere` reproduces its declared direct number; `sphere-off` is declared against one 25 % away, so the
    # gate fails THAT pack and only that pack -- the per-pack verdict is what the publish stage reads
    assert not rec["gate"]["passed"]
    assert rec["gate"]["per_substrate"]["sphere"]["passed"]
    assert not rec["gate"]["per_substrate"]["sphere-off"]["passed"]
    assert rec["gate"]["per_substrate"]["sphere-off"]["failures"] == ["sphere-off/reproduces-signal_at_b"]
    assert rec["spec"]["substrates"]["sphere"]["round_trips"]
    assert rec["walk"]["substrates"]["sphere"]["n_t"] == N_T
    assert rec["design"]["pilot"]["on_real_window"]
    assert rec["design"]["derived"]["n_walkers"] > 0


def test_a_stage_on_unchanged_inputs_returns_the_record_it_wrote(ran):
    fam, rec = ran
    before = (fam.records.digest("source"), os.path.getmtime(fam.records.path("source")))
    again = fam.stage("source")
    assert again == rec["source"]
    assert (fam.records.digest("source"), os.path.getmtime(fam.records.path("source"))) == before


def test_an_edited_record_is_not_a_measurement(ran):
    """The write-once check compares the BODY. It compared only the digest fields, so editing a number in
    `pack.json` and deleting `gate.json` re-ran the pack stage and got the edited number back unmeasured."""
    fam, rec = ran
    held = open(fam.records.path("pack")).read()
    gate = open(fam.records.path("gate")).read()
    try:
        forged = json.loads(held)
        forged["substrates"]["sphere"]["reproduced"]["signal_at_b"]["value"] *= 2.2
        with open(fam.records.path("pack"), "w") as fh:
            json.dump(forged, fh, indent=1, sort_keys=True)
        os.remove(fam.records.path("gate"))
        with pytest.raises(R.ReferenceRefusal, match="is not what this run measured"):
            fam.stage("pack")
    finally:
        for path, body in ((fam.records.path("pack"), held), (fam.records.path("gate"), gate)):
            with open(path, "w") as fh:
                fh.write(body)


def test_a_stage_whose_inputs_changed_is_refused(ran, tmp_path):
    fam, _ = ran
    other = _family(fam.dir, sources=_sources(tmp_path, host_record="synthetic v2"))
    with pytest.raises(R.ReferenceRefusal, match="is not what this run measured"):
        other.stage("source")


def test_a_record_changed_after_its_successor_read_it_is_refused(ran):
    fam, _ = ran
    body = open(fam.records.path("source")).read()
    try:
        with open(fam.records.path("source"), "a") as fh:
            fh.write("\n")
        with pytest.raises(R.ReferenceRefusal, match="has changed since the reference stage read it"):
            fam.stage("spec")
    finally:
        with open(fam.records.path("source"), "w") as fh:
            fh.write(body)


def test_the_gate_names_what_it_checked(ran):
    _, rec = ran
    names = {c["check"] for c in rec["gate"]["checks"]}
    assert "source/synthetic/licence" in names
    assert "source/synthetic/digests" in names
    assert "reference/doi" in names
    assert "sphere/served-equals-decoded" in names
    assert "sphere/reproduces-signal_at_b" in names
    assert "sphere/pgse-in-band" in names
    assert "sphere/spec-round-trips" in names
    assert rec["gate"]["per_substrate"]["sphere"]["passed"]


def test_the_card_runs_its_snippet_and_states_the_grade(ran):
    fam, rec = ran
    card = open(os.path.join(fam.dir, "README.md")).read()
    assert rec["card"]["snippet"]["stdout"].startswith("S = ")
    assert rec["card"]["snippet"]["seconds"] <= R.SNIPPET_CEILING_S
    assert rec["card"]["grade"] == "analytic"
    assert "**The comparison does not hold.**" in card
    assert "**Grade analytic.**" in card
    assert rec["card"]["snippet"]["stdout"] in card
    assert "previews/sphere.png" in card
    assert os.path.exists(os.path.join(fam.dir, "previews", "sphere.png"))
    assert "| `synthetic/sphere` |" in card


def test_publish_records_what_it_would_upload(ran):
    _, rec = ran
    p = rec["publish"]
    assert p["dry"] and p["withheld"] == ["sphere-off"]
    assert [u["substrate"] for u in p["uploaded"]] == ["sphere"]
    assert p["manifest_verdicts"]["packs/synthetic-sphere.rpk"]["gate"] == "pass"
    off = p["manifest_verdicts"]["packs/synthetic-sphere-off.rpk"]
    assert off["gate"] == "fail" and off["withheld"] and off["gate_failures"]
    assert "records/gate.json" in p["files"] and "README.md" in p["files"]
    assert "records/source.json" in p["files"] and "previews/sphere.png" in p["files"]
    assert "records/publish.json" not in p["files"]              # a record cannot publish itself


# --------------------------------------------------------------- the order rule
@pytest.mark.parametrize("stage", ["reference", "spec", "design", "walk", "pack", "gate", "card", "publish"])
def test_no_stage_runs_before_the_one_before_it(tmp_path, stage):
    fam = _family(tmp_path)
    with pytest.raises(R.ReferenceRefusal, match="is missing; run the .* stage first"):
        fam.stage(stage)


def test_a_name_that_is_not_a_stage_is_refused(tmp_path):
    with pytest.raises(R.ReferenceRefusal, match="is not a stage"):
        _family(tmp_path).stage("finalize")


# --------------------------------------------------------------- the source stage's refusals
def test_a_licence_title_is_not_a_licence_text(tmp_path):
    fam = _family(tmp_path, sources=_sources(tmp_path, licence_text="CC BY 4.0"))
    with pytest.raises(R.ReferenceRefusal, match="a licence TITLE is not a licence text"):
        fam.stage("source")


def test_a_source_without_a_host_record_is_refused(tmp_path):
    fam = _family(tmp_path, sources=_sources(tmp_path, host_record=""))
    with pytest.raises(R.ReferenceRefusal, match="no host_record"):
        fam.stage("source")


def test_a_file_that_cannot_be_digested_is_refused(tmp_path):
    fam = _family(tmp_path, sources=_sources(tmp_path, missing=True))
    with pytest.raises(R.ReferenceRefusal, match="without a digest per file is refused"):
        fam.stage("source")


# --------------------------------------------------------------- the reference stage's refusals
def _upto_source(tmp_path, **kw):
    os.makedirs(tmp_path, exist_ok=True)
    fam = _family(tmp_path, **kw)
    fam.stage("source")
    return fam


def test_a_reference_without_a_doi_is_refused(tmp_path):
    fam = _upto_source(tmp_path, reference=_reference(doi=""))
    with pytest.raises(R.ReferenceRefusal, match="no DOI"):
        fam.stage("reference")


def test_a_doi_whose_title_does_not_match_is_refused(tmp_path):
    fam = _upto_source(tmp_path, reference=_reference(title="Something else entirely"))
    with pytest.raises(R.ReferenceRefusal, match="do not describe one work"):
        fam.stage("reference")


def test_a_doi_that_does_not_resolve_is_refused(tmp_path):
    def dead(doi, **kw):
        raise R.ReferenceRefusal(f"reference: the DOI {doi!r} did not resolve through Crossref")
    fam = _upto_source(tmp_path, resolver=dead)
    with pytest.raises(R.ReferenceRefusal, match="did not resolve through Crossref"):
        fam.stage("reference")


def test_a_free_parameter_with_no_owner_is_refused(tmp_path):
    ref = _reference(parameters=(R.FreeParameter(name="D", value=D0, unit="m^2/s", whose="assumed",
                                                 where="x", how="y"),))
    fam = _upto_source(tmp_path, reference=ref)
    with pytest.raises(R.ReferenceRefusal, match="every free parameter is one of"):
        fam.stage("reference")


def test_a_reference_number_without_a_verbatim_quote_is_refused(tmp_path):
    ref = _reference(quantities=(_quantity(0.5, 0.05, verbatim="  "),))
    fam = _upto_source(tmp_path, reference=ref)
    with pytest.raises(R.ReferenceRefusal, match="quotes nothing from"):
        fam.stage("reference")


def test_a_resampled_error_bar_is_not_a_standard_error(tmp_path):
    ref = _reference(quantities=(_quantity(0.5, 0.05, se_kind="fold_spread"),))
    fam = _upto_source(tmp_path, reference=ref)
    with pytest.raises(R.ReferenceRefusal, match="never a fold spread or a split half"):
        fam.stage("reference")


def test_a_number_from_a_document_needs_that_documents_identifier(tmp_path):
    """The verified DOI was the 2009 paper; the number is printed in the thesis, whose identifier was an hdl
    handle in prose, never resolved or title-compared. A thesis HAS a DOI -- Talabi's is the DataCite
    10.25560/4261 -- so the document the number comes from is resolved too."""
    pub = R.Published(value=0.5, unit="-", uncertainty=0.0, uncertainty_is="none stated",
                      printed_in="Somebody (2008), A thesis", locator="Table 1", verbatim="a row")
    fam = _upto_source(tmp_path, reference=_reference(published_kind="number",
                                                      quantities=(_quantity(0.5, 0.05, published=pub),)))
    with pytest.raises(R.ReferenceRefusal, match="cites no identifier for it"):
        fam.stage("reference")


def test_a_documents_identifier_must_describe_that_document(tmp_path):
    pub = R.Published(value=0.5, unit="-", uncertainty=0.0, uncertainty_is="none stated",
                      printed_in="Somebody (2008), Quite another thesis", locator="Table 1", verbatim="a row",
                      document="10.0/x")
    fam = _upto_source(tmp_path, reference=_reference(published_kind="number",
                                                      quantities=(_quantity(0.5, 0.05, published=pub),)))
    with pytest.raises(R.ReferenceRefusal, match="do not describe one thing"):
        fam.stage("reference")


def test_released_data_is_a_reference_kind_and_needs_its_digest(tmp_path):
    """parity-fixtures, winther-g6-axons and disco-replay reproduce released DATA -- a signal, per-axon signals,
    a volume. The vocabulary could not say so, and the "verbatim line" rule would have made them invent one."""
    bare = R.Published(value=0.5, unit="-", uncertainty=0.0, uncertainty_is="none stated",
                       printed_in="their released DWI", locator="the b = 0 rows")
    fam = _upto_source(tmp_path / "bare", reference=_reference(published_kind="data",
                                                              quantities=(_quantity(0.5, 0.05, published=bare),)))
    with pytest.raises(R.ReferenceRefusal, match="records the data's URL and its sha256"):
        fam.stage("reference")
    good = R.Published(value=0.5, unit="-", uncertainty=0.0, uncertainty_is="none stated",
                       printed_in="their released DWI", locator="the b = 0 rows",
                       data_url="https://example.invalid/dwi.bfloat", data_sha256="ab" * 32)
    fam2 = _upto_source(tmp_path / "good", reference=_reference(published_kind="data",
                                                               quantities=(_quantity(0.5, 0.05, published=good),)))
    rec = fam2.stage("reference")
    assert R.grade_of(rec) == "A"                      # released data on the released geometry is an A
    assert "the released data itself" in R.grade_reason(rec)


def test_a_direct_measurement_without_a_grid_is_refused(tmp_path):
    q = _quantity(0.5, 0.05)
    q = R.ReferenceQuantity(substrate=q.substrate, name=q.name, published=q.published,
                            direct=R.Direct(value=0.5, unit="-", se=0.05, se_kind="analytic_mean",
                                            se_derivation="x", n_walkers=N_WALK, grid={}, solver="x",
                                            source="y"))
    fam = _upto_source(tmp_path, reference=_reference(quantities=(q,)))
    with pytest.raises(R.ReferenceRefusal, match="states no se derivation, grid or solver"):
        fam.stage("reference")


def test_a_quantity_about_no_substrate_of_the_family_is_refused(tmp_path):
    q = _quantity(0.5, 0.05)
    q = R.ReferenceQuantity(substrate="granite", name=q.name, published=q.published, direct=q.direct)
    fam = _upto_source(tmp_path, reference=_reference(quantities=(q,)))
    with pytest.raises(R.ReferenceRefusal, match="not one of this family's substrates"):
        fam.stage("reference")


def test_a_family_with_no_reference_records_none_and_may_publish(tmp_path):
    fam = _upto_source(tmp_path, reference=None)
    rec = fam.stage("reference")
    assert rec["absent"]
    assert R.grade_of(rec) == "none"


# --------------------------------------------------------------- the spec stage's refusals
def test_a_spec_citing_a_digest_no_source_holds_is_refused(tmp_path, monkeypatch):
    fam = _upto_source(tmp_path)
    fam.stage("reference")
    def forged():
        spec = _spec()
        from dataclasses import replace
        from dmipy_sim.spec import Surface, Wall
        w = spec.walls[0]
        return replace(spec, walls=[replace(w, surface=Surface(**{**w.surface.__dict__, "file": "x.ply",
                                                                 "format": "ply", "sha256": "ab" * 32}))])
    fam.build.specs["sphere"] = forged
    with pytest.raises(R.ReferenceRefusal, match="the source record does not hold"):
        fam.stage("spec")


# --------------------------------------------------------------- the design's envelope and the budget
def _upto_spec(tmp_path, **kw):
    fam = _family(tmp_path, **kw)
    for st in ("source", "reference", "spec"):
        fam.stage(st)
    return fam


def test_a_waveform_declared_twice_is_refused(tmp_path):
    """`waveforms()` used to de-duplicate silently, which hid that a family's `delta_frac` and its
    `shortd_deltas_frac` named the same delta twice."""
    fam = _upto_spec(tmp_path, design=_design(waveforms=lambda: (("pgse", _seq()), ("pgse", _seq()))))
    with pytest.raises(R.ReferenceRefusal, match="is declared twice"):
        fam.stage("design")


def test_a_waveform_beyond_the_envelope_the_pack_is_built_to_is_refused(tmp_path):
    """`design.envelope` and `design.waveforms` were two 'declared envelopes' nothing tied together."""
    hot = d.pgse([[1, 0, 0]], DELTA, BIGDELTA, bvalues=[10 * B], n_t=64, slew_rate=np.inf)
    fam = _upto_spec(tmp_path, design=_design(waveforms=lambda: (("pgse-hot", hot),)))
    with pytest.raises(R.ReferenceRefusal, match="the envelope the pack is BUILT to certifies"):
        fam.stage("design")


def test_the_walk_is_held_to_its_MEASURED_peak(tmp_path, monkeypatch):
    """Before the walk the budget is the design's own cap, so that check can only fire on a record the
    derivation did not write; after it, the walk's sampled peak is a measurement the projection can under-read,
    and the budget is a hard cap on THAT."""
    fam = _upto_spec(tmp_path)
    fam.stage("design")
    huge = int(fam.records.read("design")["memory_budget_bytes"]) * 4

    class _Big:
        peak = huge

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(R, "_PeakRSS", lambda *a, **k: _Big())
    with pytest.raises(R.ReferenceRefusal, match="the measurement, not the projection"):
        fam.stage("walk")


# --------------------------------------------------------------- the tolerance
def test_a_tolerance_term_that_is_a_literal_is_refused():
    with pytest.raises(R.ReferenceRefusal, match="is a literal"):
        R.tolerance_of(dict(terms=[0.005], combine="quadrature"), {})


def test_a_coverage_factor_is_not_a_tolerance_term():
    with pytest.raises(R.ReferenceRefusal, match="no coverage factor"):
        R.tolerance_of(dict(terms=["design.coverage_factor"], combine="quadrature"),
                       dict(design=dict(coverage_factor=2.0)))


def test_a_fold_spread_is_not_a_tolerance_term():
    with pytest.raises(R.ReferenceRefusal, match="no coverage factor and no resampled error bar"):
        R.tolerance_of(dict(terms=["quantity.fold_spread"], combine="quadrature"),
                       dict(quantity=dict(fold_spread=0.01)))


def test_a_term_that_does_not_resolve_is_refused():
    with pytest.raises(R.ReferenceRefusal, match="does not resolve"):
        R.tolerance_of(dict(terms=["quantity.direct.se"], combine="quadrature"), dict(quantity={}))


def test_no_terms_is_not_a_gate():
    with pytest.raises(R.ReferenceRefusal, match="a gate with no budget is not a gate"):
        R.tolerance_of(dict(terms=[], combine="quadrature"), {})


def test_terms_are_combined_in_quadrature_and_nothing_else():
    ctx = dict(a=dict(x=0.003), b=dict(y=0.004))
    terms, tol = R.tolerance_of(dict(terms=["a.x", "b.y"], combine="quadrature"), ctx)
    assert terms == {"a.x": 0.003, "b.y": 0.004}
    assert tol == pytest.approx(0.005)
    with pytest.raises(R.ReferenceRefusal, match="combined in quadrature"):
        R.tolerance_of(dict(terms=["a.x"], combine="sum"), ctx)


# --------------------------------------------------------------- the pass band
def test_a_one_sigma_criterion_is_not_a_criterion():
    """MEASURED: over four comparisons a one-standard-error rule fails a CORRECT family 78 % of the time. The
    band is derived from the comparison count and a DECLARED false-failure rate, before the gate runs."""
    band = R.pass_band(4, 42631, false_failure_rate=0.01)
    assert band["k"] == pytest.approx(3.0235, abs=1e-3)
    assert band["one_sigma_false_failure_rate"] > 0.75
    assert R.pass_band(1, 42631, false_failure_rate=0.01)["k"] < band["k"]     # fewer comparisons, tighter band
    assert R.pass_band(4, 42631, false_failure_rate=0.05)["k"] < band["k"]     # a looser rate, tighter band


def test_the_pass_band_is_the_one_multiplicity_quantile_in_the_repo():
    """`cross_engine_parity.multiplicity_thresholds` reads the same quantile, so the two cannot drift."""
    from examples.validation.cross_engine_parity import multiplicity_thresholds
    for m, nu in ((360, 99999), (100, 99999)):
        assert multiplicity_thresholds(m, nu)["k_family_wise"] == R.pass_band(m, nu, false_failure_rate=0.5)["k"]


@pytest.mark.parametrize("args,kw,match", [
    ((0, 10), dict(false_failure_rate=0.01), "must be positive"),
    ((4, 0), dict(false_failure_rate=0.01), "must be positive"),
    ((4, 10), dict(false_failure_rate=0.0), "not a probability"),
    ((4, 10), dict(false_failure_rate=1.0), "not a probability"),
])
def test_the_pass_band_refuses_what_is_not_a_band(args, kw, match):
    with pytest.raises(R.ReferenceRefusal, match=match):
        R.pass_band(*args, **kw)


def test_the_gate_reads_its_band_from_the_design_record(ran):
    fam, rec = ran
    band = rec["design"]["pass_band"]
    assert band["n_comparisons"] == 2 * len(rec["reference"]["quantities"])
    assert band["dof"] == rec["design"]["derived"]["n_walkers"] - 1
    assert band["false_failure_rate"] == 0.01
    c = next(c for c in rec["gate"]["checks"] if c["check"] == "sphere/reproduces-signal_at_b")
    assert c["k"] == band["k"]
    assert c["tolerance"] == pytest.approx(band["k"] * c["standard_uncertainty"])
    assert c["sigma"] == pytest.approx(c["relative"] / c["standard_uncertainty"])


def test_the_gate_contains_no_typed_tolerance():
    """Every threshold the gate applies comes from a record. A float literal in its body would be one that
    does not, which is how a pack 2.2x off its reference passed (#482)."""
    src = inspect.getsource(R._gate_checks)
    body = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert not re.findall(r"\d+\.\d|\d+[eE][-+]?\d", body), re.findall(r"\d+\.\d|\d+[eE][-+]?\d", body)


# --------------------------------------------------------------- the gate's verdicts
def test_the_gate_fails_a_reproduction_outside_the_tolerance(ran):
    fam, rec = ran
    pk = json.loads(json.dumps(rec["pack"]))
    q = pk["substrates"]["sphere"]["reproduced"]["signal_at_b"]
    q["value"] = q["value"] * 2.2
    checks = R._gate_checks(dict(rec, pack=pk))
    bad = [c for c in checks if c["check"] == "sphere/reproduces-signal_at_b"]
    assert bad and not bad[0]["passed"] and bad[0]["relative"] > bad[0]["tolerance"]
    passed, failures, per = R.gate_verdict(checks)
    assert not passed and not per["sphere"]["passed"]


def test_the_gate_fails_a_reproduction_on_another_grid(ran):
    fam, rec = ran
    pk = json.loads(json.dumps(rec["pack"]))
    pk["substrates"]["sphere"]["reproduced"]["signal_at_b"]["grid"] = dict(kind="coarser")
    checks = R._gate_checks(dict(rec, pack=pk))
    bad = [c for c in checks if c["check"].endswith("-on-the-recorded-grid")]
    assert bad and not bad[0]["passed"]


def test_the_gate_fails_a_waveform_out_of_band(ran):
    fam, rec = ran
    pk = json.loads(json.dumps(rec["pack"]))
    pk["substrates"]["sphere"]["waveforms"][0].update(in_band=False)
    checks = R._gate_checks(dict(rec, pack=pk))
    assert not [c for c in checks if c["check"] == "sphere/pgse-in-band"][0]["passed"]


def test_the_gate_fails_a_dropped_waveform(ran):
    fam, rec = ran
    pk = json.loads(json.dumps(rec["pack"]))
    pk["substrates"]["sphere"]["waveforms"] = []
    checks = R._gate_checks(dict(rec, pack=pk))
    assert not [c for c in checks if c["check"].endswith("envelope-is-the-declared-one")][0]["passed"]


def test_the_gate_fails_served_beyond_the_codec(ran):
    fam, rec = ran
    pk = json.loads(json.dumps(rec["pack"]))
    sub = pk["substrates"]["sphere"]
    sub["served_vs_channel"] = sub["served_vs_channel_bound"] * 10 + 1e-9
    checks = R._gate_checks(dict(rec, pack=pk))
    assert not [c for c in checks if c["check"].endswith("served-equals-decoded")][0]["passed"]


def test_a_tier_the_design_promised_must_hold(ran):
    """`trade` excused EVERY tier, so the design's own prediction could be falsified unseen: F42A's contact
    floor came out 0.00589 where the pilot's 1/sqrt(N) scaling predicted 0.00488, under a trade note naming
    contact as the tier that holds."""
    fam, rec = ran
    pk = json.loads(json.dumps(rec["pack"]))
    des = json.loads(json.dumps(rec["design"]))
    des["derived"]["tiers_that_hold"] = ["positions"]
    des["trade"] = "the budget allows fewer walkers than the floor asks"
    pk["substrates"]["sphere"]["tiers"]["positions"]["meets_target"] = False
    c = [c for c in R._gate_checks(dict(rec, pack=pk, design=des))
         if c["check"] == "sphere/tier-positions-holds-as-designed"]
    assert c and not c[0]["passed"] and "falsified" in c[0]["detail"]
    des["derived"]["tiers_that_hold"] = []               # the design never promised it: the trade excuses it
    c = [c for c in R._gate_checks(dict(rec, pack=pk, design=des))
         if c["check"] == "sphere/tier-positions-target-or-trade"]
    assert c and c[0]["passed"]


def test_a_tier_below_target_needs_the_design_records_trade(ran):
    fam, rec = ran
    pk = json.loads(json.dumps(rec["pack"]))
    pk["substrates"]["sphere"]["tiers"]["positions"]["meets_target"] = False
    des = json.loads(json.dumps(rec["design"]))
    des["derived"]["tiers_that_hold"] = []
    des["trade"] = None
    assert not [c for c in R._gate_checks(dict(rec, pack=pk, design=des))
                if c["check"].endswith("tier-positions-target-or-trade")][0]["passed"]
    des["trade"] = "the budget allows fewer walkers than the floor asks"
    assert [c for c in R._gate_checks(dict(rec, pack=pk, design=des))
            if c["check"].endswith("tier-positions-target-or-trade")][0]["passed"]


def test_the_gate_compares_us_with_them_and_not_only_with_ourselves(ran):
    """A `reproduces-` check compares two routes of one engine; the family exists to reproduce a PUBLISHED
    number, and the gate must say how far it is. `ling-sand-packs` sat 17.7 % from its measurement under a card
    saying every check passed."""
    fam, rec = ran
    names = {c["check"] for c in rec["gate"]["checks"]}
    assert "sphere/published-signal_at_b" in names
    c = next(c for c in rec["gate"]["checks"] if c["check"] == "sphere/published-signal_at_b")
    published = next(q for q in rec["reference"]["quantities"] if q["substrate"] == "sphere")["published"]
    got = rec["pack"]["substrates"]["sphere"]["reproduced"]["signal_at_b"]["value"]
    assert c["relative"] == pytest.approx(abs(got - published["value"]) / abs(published["value"]))
    assert c["uncertainty_stated"] is False              # the closed form states none, so it cannot fail


def test_a_published_disagreement_reaches_paragraph_one_even_when_it_cannot_fail(ran):
    """A publication with no stated uncertainty cannot be failed against; the disagreement is recorded as a
    number and the card's first paragraph states it, which is the whole of the Ling lesson."""
    fam, rec = ran
    gate = json.loads(json.dumps(rec["gate"]))
    c = next(c for c in gate["checks"] if c["check"] == "sphere/published-signal_at_b")
    c.update(relative=c["tolerance"] * 10 + 1.0, uncertainty_stated=False, passed=True)
    bad = R._disagreements(gate)
    assert [x["check"] for x in bad if x["check"].startswith("sphere/published-")] == \
        ["sphere/published-signal_at_b"]
    card = R._render_card("synthetic", "owner/synthetic", fam.read_all("pack"), gate,
                          fam.records.read("card")["previews"], "print(1)", dict(stdout="1", seconds=0.1),
                          "packs/x.rpk")
    opening = card.split("## The packs")[0]
    assert "`sphere`: " in opening and "cannot be failed on and is recorded instead" in opening
    only_published = json.loads(json.dumps(gate))
    for x in only_published["checks"]:                      # with nothing FAILING, the wording is the milder one
        x["passed"] = True if "/reproduces-" in x["check"] else x["passed"]
    only_published["failures"] = []
    card2 = R._render_card("synthetic", "owner/synthetic", fam.read_all("pack"), only_published,
                           fam.records.read("card")["previews"], "print(1)", dict(stdout="1", seconds=0.1),
                           "packs/x.rpk")
    assert "disagree with the published number" in card2.split("## The packs")[0]


def test_a_published_value_with_a_stated_uncertainty_can_fail(ran):
    fam, rec = ran
    pk = json.loads(json.dumps(rec["pack"]))
    ref = json.loads(json.dumps(rec["reference"]))
    for q in ref["quantities"]:
        q["published"]["uncertainty"] = 1e-4                         # they state one, tightly
        q["published"]["uncertainty_is"] = "stated in the paper"
    pk["substrates"]["sphere"]["reproduced"]["signal_at_b"]["value"] *= 1.5
    c = next(c for c in R._gate_checks(dict(rec, pack=pk, reference=ref))
             if c["check"] == "sphere/published-signal_at_b")
    assert c["uncertainty_stated"] and not c["passed"]


# --------------------------------------------------------------- the grade rule
def test_the_released_geometry_is_derived_from_the_parameters(tmp_path):
    """`same_released_geometry` was a hand-set boolean the family's own record contradicted: imperial-rocks
    marked its crop `ours` -- Talabi's 0.377 porosity is unreachable over all 3,442,951 crops -- and still
    claimed A. A parameter marked `ours` that changes the geometry now makes it B, and the card says which."""
    geo = R.FreeParameter(name="crop", value="central 300^3", unit="voxels", whose="ours",
                          changes_geometry=True, where="theirs states the size, not the offset", how="ours")
    theirs = R.FreeParameter(name="crop", value="theirs", unit="voxels", whose="theirs",
                             changes_geometry=True, where="stated", how="theirs")
    for params, same, grade in ((theirs, True, "A"), (geo, False, "B")):
        here = tmp_path / ("same" if same else "ours")
        here.mkdir()
        doc = R.Published(value=0.5, unit="-", uncertainty=0.0, uncertainty_is="none stated",
                          printed_in="Pore-network extraction from micro-computerized-tomography images",
                          locator="Table 1", verbatim="a row", document="10.1103/PhysRevE.80.036307")
        fam = _upto_source(here,
                           reference=_reference(published_kind="number",
                                                quantities=(_quantity(0.5, 0.05, published=doc),),
                                                parameters=(params, R.FreeParameter(
                                                    name="D", value=D0, unit="m^2/s", whose="ours",
                                                    where="x", how="y"))))
        rec = fam.stage("reference")
        assert rec["same_released_geometry"] is same
        assert R.grade_of(rec) == grade
        assert rec["geometry_parameters_ours"] == ([] if same else ["crop"])
    assert "NOT on the released geometry" in R.grade_reason(rec)
    assert "same_released_geometry" not in {f.name for f in R.Reference.__dataclass_fields__.values()}


@pytest.mark.parametrize("record,grade", [
    (dict(published_kind="number", same_released_geometry=True), "A"),
    (dict(published_kind="number", same_released_geometry=False), "B"),
    (dict(published_kind="figure", same_released_geometry=True), "B"),
    (dict(published_kind="figure", same_released_geometry=False), "B"),
    (dict(published_kind="analytic", same_released_geometry=True), "analytic"),
    (dict(published_kind="none", same_released_geometry=True), "none"),
    (dict(absent="no published quantity"), "none"),
    (None, "none"),
])
def test_the_grade_follows_from_the_reference_record(record, grade):
    assert R.grade_of(record) == grade


def test_a_reference_record_cannot_state_its_own_grade(tmp_path):
    """The grade is not a field: it is computed. `ling-sand-packs` wrote "A" for a paper with no number."""
    assert "grade" not in {f.name for f in R.Reference.__dataclass_fields__.values()}
    assert "grade" not in {f.name for f in R.Published.__dataclass_fields__.values()}


# --------------------------------------------------------------- the card stage
def test_a_card_whose_snippet_fails_is_not_written(tmp_path, monkeypatch):
    fam, _ = _prepared(tmp_path, monkeypatch, snippet=_bad_snippet)
    with pytest.raises(R.ReferenceRefusal, match="does not produce"):
        fam.stage("card")
    assert not os.path.exists(os.path.join(fam.dir, "README.md"))


def test_a_snippet_that_does_not_finish_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "SNIPPET_CEILING_S", 1.0)
    with pytest.raises(R.ReferenceRefusal, match="did not finish in"):
        R._run_snippet("import time\ntime.sleep(30)\n", str(tmp_path))


def test_a_snippet_is_run_against_the_code_that_built_the_card(tmp_path):
    out = R._run_snippet("import dmipy_sim, os\nprint(os.path.dirname(os.path.dirname(dmipy_sim.__file__)))\n",
                         str(tmp_path))
    assert out["stdout"] == os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(R.__file__))))


# --------------------------------------------------------------- the publish stage
def test_publish_refuses_when_the_gate_passed_no_pack(tmp_path, monkeypatch):
    fam, rec = _prepared(tmp_path, monkeypatch)
    gate = json.loads(open(fam.records.path("gate")).read())
    gate["per_substrate"]["sphere"]["passed"] = False
    gate["failures"] = ["sphere/reproduces-signal_at_b: it does not"]
    with open(fam.records.path("gate"), "w") as fh:
        json.dump(gate, fh, indent=1, sort_keys=True)
    fam.stage("card")                                  # the card is written, and states the failure
    assert "**FAIL**" in open(os.path.join(fam.dir, "README.md")).read()
    with pytest.raises(R.ReferenceRefusal, match="the gate passed no pack"):
        fam.stage("publish")


def test_publish_refuses_a_record_changed_after_the_gate(tmp_path, monkeypatch):
    fam, rec = _prepared(tmp_path, monkeypatch)
    fam.stage("card")
    with open(fam.records.path("design"), "a") as fh:
        fh.write("\n")
    with pytest.raises(R.ReferenceRefusal, match="changed since"):
        fam.stage("publish")


def test_the_manifest_carries_the_gate_verdict_per_pack(tmp_path, monkeypatch):
    """A withheld pack is not deleted -- its bytes are the maintainer's -- but it must not sit in the manifest
    with nothing to say the gate failed it."""
    from dmipy_sim.fill import FakeHub
    from dmipy_sim.replay import publish as pub
    hub = FakeHub(str(tmp_path / "hub"))
    pubn = R.Publication(repo="owner/synthetic", licence="CC-BY-4.0", citation="a test family",
                         snippet=_snippet, snippet_substrate="sphere", dry=False)
    fam = _family(tmp_path, publication=pubn)
    fam._hub = hub
    monkeypatch.setattr(R, "code_commit", lambda **k: "0" * 40)
    for st in ("source", "reference", "spec", "design", "walk", "pack", "gate", "card"):
        fam.stage(st)
    rec = fam.read_all("pack")
    off = rec["pack"]["substrates"]["sphere-off"]["pack"]
    local = os.path.join(fam.dir, off["path_local"])
    pub.publish(local, "owner/synthetic", path="packs/synthetic-sphere-off.rpk", hub=hub)  # an earlier run's
    fam.stage("publish")
    rows = {r["path"]: r for r in json.load(open(os.path.join(hub.root, "manifest.json")))["packs"]}
    assert rows["packs/synthetic-sphere.rpk"]["gate"] == "pass"
    assert rows["packs/synthetic-sphere.rpk"]["withheld"] is False
    assert rows["packs/synthetic-sphere-off.rpk"]["gate"] == "fail"
    assert rows["packs/synthetic-sphere-off.rpk"]["withheld"] is True
    assert rows["packs/synthetic-sphere-off.rpk"]["gate_failures"] == ["sphere-off/reproduces-signal_at_b"]
    assert rows["packs/synthetic-sphere-off.rpk"]["gate_record"] == "records/gate.json"
    assert rows["packs/synthetic-sphere-off.rpk"]["sha256"] == off["sha256"]      # the bytes are untouched


def _prepared(tmp_path, monkeypatch, **kw):
    """The synthetic family run as far as the gate, in its own directory."""
    pub = R.Publication(repo="owner/synthetic", licence="CC-BY-4.0", citation="a test family",
                        snippet=kw.pop("snippet", _snippet), snippet_substrate="sphere", dry=True)
    fam = _family(tmp_path, publication=pub, **kw)
    monkeypatch.setattr(R, "code_commit", lambda **k: "0" * 40)
    for s in ("source", "reference", "spec", "design", "walk", "pack", "gate"):
        fam.stage(s)
    return fam, fam.read_all("pack")


# --------------------------------------------------------------- the organisation page
def _md_tables(text):
    """Every GFM table in ``text`` as ``(header cells, separator cells, row cell counts)``."""
    out, lines = [], text.splitlines()
    for i, line in enumerate(lines[:-1]):
        if line.startswith("|") and set(lines[i + 1].replace("|", "").replace(" ", "")) == {"-"}:
            head = [c for c in line.strip().strip("|").split("|")]
            sep = [c for c in lines[i + 1].strip().strip("|").split("|")]
            rows = []
            for r in lines[i + 2:]:
                if not r.startswith("|"):
                    break
                rows.append(len(r.strip().strip("|").split("|")))
            out.append((len(head), len(sep), rows))
    return out


def test_every_card_table_is_a_table(ran):
    """GFM drops a column whose separator is short and refuses a table whose header has an unescaped pipe: the
    packs table had 12 header cells against an 11-cell separator (the `commit` column vanished) and the
    reproduction header carried a literal `|replayed - direct|`."""
    fam, rec = ran
    card = open(os.path.join(fam.dir, "README.md")).read()
    tables = _md_tables(card)
    assert len(tables) >= 4
    for head, sep, rows in tables:
        assert head == sep, (head, sep)
        assert all(r == head for r in rows), (head, rows)


def test_the_organisation_page_refuses_a_family_with_no_reference_record():
    rows = [dict(name="a", repo="O/a", packs=1, bytes=10,
                 reference=dict(published_kind="figure", same_released_geometry=False, description="x",
                                source_note="y", licence_note="z", sample="s",
                                sample_relation="the same material"),
                 packs_rows=[dict(path="packs/withheld.rpk", gate="fail", withheld=True),
                             dict(path="packs/good.rpk", gate="pass", withheld=False)]),
            dict(name="b", repo="O/b", packs=2, bytes=20, reference=None, packs_rows=[])]
    with pytest.raises(R.ReferenceRefusal, match="publish no records/reference.json"):
        R.organisation_card(rows)
    page = R.organisation_card(rows[:1], example=R.organisation_example(rows[:1]))
    assert "| **B** |" in page and "[a](https://huggingface.co/datasets/O/a)" in page
    # the snippet must read a pack the index holds and the gate passed, not the one it withheld
    assert "hf://O/a/packs/good.rpk" in page
    assert "withheld.rpk" not in page


def test_the_organisation_snippet_never_names_a_withheld_pack():
    rows = [dict(name="a", repo="O/a", packs=2, bytes=10, reference={},
                 packs_rows=[dict(path="packs/withheld.rpk", gate="fail", withheld=True),
                             dict(path="packs/good.rpk", gate="pass", withheld=False)])]
    assert R.organisation_example(rows) == "O/a/packs/good.rpk"
    assert R.organisation_example([dict(name="b", repo="O/b", packs_rows=[])]) is None


# --------------------------------------------------------------- the estimator's analytic error bar
def test_the_log_mean_gradient_is_the_estimators_own_derivative():
    """The delta method needs the log-mean's derivative with respect to the decay; a fold spread does not, and
    is seed-dependent, which is why the protocol will not take one (#482)."""
    from examples.validation.talabi_micro_ct_rocks import log_mean_T2, log_mean_gradient, t2_distribution
    rng = np.random.default_rng(0)
    t = np.arange(1, 400) * 1e-3
    S = 0.6 * np.exp(-t / 0.4) + 0.4 * np.exp(-t / 1.2)
    grid = np.logspace(-3, 1, 60)
    lm, g = log_mean_gradient(t, S, grid)
    assert lm == pytest.approx(log_mean_T2(grid, t2_distribution(t, S, grid)))
    for eps in (1e-6, 1e-5):
        v = rng.normal(size=S.shape); v /= np.linalg.norm(v)
        fd = (log_mean_T2(grid, t2_distribution(t, S + eps * v, grid))
              - log_mean_T2(grid, t2_distribution(t, S - eps * v, grid))) / (2 * eps)
        assert g @ v == pytest.approx(fd, rel=1e-5)
