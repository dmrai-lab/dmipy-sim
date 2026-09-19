"""The scanner catalogue is one source with typed views (#173 piece 1).

Every class of the band-limit certificate and every Pulseq preset is the cited JSON's number for that machine;
one scanner resolves from any of its names; a slew regime is a choice, not a second catalogue; and what the
catalogue does not know is ``None`` and listed, never a number standing in for one.
"""
import inspect
import json

import numpy as np
import pytest

import dmipy_sim
from dmipy_sim.acquisition import scanner_constants as scc
from dmipy_sim.acquisition.scanners import ScannerLimits, SCANNERS, scanner_limits
from dmipy_sim.constants import DEFAULT_SLEW_RATE
from dmipy_sim.sequences.pulseq import PULSEQ_SYSTEMS


def test_every_certificate_class_is_the_catalogue_number_for_that_machine():
    for cls_name, key in scc.SCANNER_CONSTANTS["classes"].items():
        _, entry, _ = scc.resolve(key)
        assert SCANNERS[cls_name] == (scc.leaf_si(entry, "gradient", "max_amplitude"),
                                      scc.leaf_si(entry, "gradient", "max_slew_rate")), cls_name
    assert set(SCANNERS) == set(scc.SCANNER_CONSTANTS["classes"])


def test_every_pulseq_preset_is_the_catalogue_number_for_that_machine():
    for name, opts in PULSEQ_SYSTEMS.items():
        lim = ScannerLimits.of(name)
        assert opts["max_grad"] == pytest.approx(lim.G_max * 1e3) and opts["max_slew"] == lim.slew_max, name
        assert opts["grad_unit"] == "mT/m" and opts["slew_unit"] == "T/m/s"
    # the two numbers the old hand-written preset table disagreed with the cited catalogue on
    assert PULSEQ_SYSTEMS["ge_premier"]["max_grad"] == 80.0          # was 70
    assert PULSEQ_SYSTEMS["philips_ingenia"]["max_grad"] == 45.0     # was 80


def test_one_scanner_resolves_from_every_name_it_has():
    a = ScannerLimits.of("prisma"); b = ScannerLimits.of("siemens_prisma"); c = ScannerLimits.of("siemens_magnetom_prisma_3T")
    assert a == b == c and a.name == "siemens_magnetom_prisma_3T" and a.kind == "scanner"
    assert ScannerLimits.of("PRISMA") == a
    assert ScannerLimits.of((0.08, 200.0)).gradient_limits == a.gradient_limits
    assert ScannerLimits.of(a) is a
    with pytest.raises(ValueError, match="unknown scanner"):
        scanner_limits("siemens")


def test_connectom_is_two_machines_and_a_regime_not_three_numbers():
    """The class table once read (0.30, 600): Connectom-1's amplitude with Connectome-2.0's slew, a machine that
    does not exist. Now each is its own class, and the diffusion regime is the derated slew a diffusion sequence
    is actually bound by."""
    c1 = ScannerLimits.of("connectom")
    assert c1.gradient_limits == (0.30, 200.0) and c1.name == "siemens_magnetom_connectom_3T"
    assert ScannerLimits.of("connectom", regime="diffusion").slew_max == 62.5
    assert ScannerLimits.of("connectome_2").gradient_limits == (0.50, 600.0)
    # a machine with no derated figure keeps its hardware slew in the diffusion regime
    assert ScannerLimits.of("prisma", regime="diffusion").slew_max == 200.0


def test_magnus_is_the_prototype_and_signa_magnus_the_product():
    assert ScannerLimits.of("magnus").gradient_limits == (0.20, 500.0)
    assert ScannerLimits.of("signa_magnus").gradient_limits == (0.30, 750.0)
    assert ScannerLimits.of("magnus").name != ScannerLimits.of("signa_magnus").name


def test_default_slew_is_the_prisma_class():
    assert DEFAULT_SLEW_RATE == SCANNERS["prisma"][1]


def test_what_the_catalogue_does_not_know_is_none_and_listed():
    p = ScannerLimits.of("prisma")
    assert p.grad_raster == pytest.approx(1e-5) and p.rf_raster == pytest.approx(1e-6)
    assert p.rf_dead_time is None and p.rf_ringdown_time is None          # NEEDS VERIFICATION leaves
    assert "rf_dead_time" not in p.pulseq_dict()                          # so the Opts omits them, not zeroes them
    with pytest.raises(ValueError, match="no verified gradient"):
        ScannerLimits.of("pulseq_example_system")
    unverified = {m for m, _, _ in scc.needs_verification()}
    assert {"pulseq_clinical_typical", "pulseq_preclinical_bruker", "pulseq_example_system"} <= unverified
    for k in ("clinical_typical", "preclinical_bruker"):
        assert ScannerLimits.of(k).kind == "envelope", k


def test_the_preclinical_classes_are_cited_bruker_hardware():
    """The three preclinical certificate classes were unsourced envelope points (750 / 6000, 1500 / 10000,
    3000 / 20000, 'NEEDS VERIFICATION'); they are Bruker's BGA-9S, Micro2.5 and Micro5 (#220), and no
    certificate can be published against a number no hardware has."""
    for cls, key, lim in (("bruker_bga_s", "bruker_bga_9s", (0.760, 6840.0)),
                          ("micro_insert", "bruker_micro2_5", (1.500, 15000.0)),
                          ("extreme_insert", "bruker_micro5", (3.000, 37500.0))):
        lm = ScannerLimits.of(cls)
        assert lm.kind == "scanner" and lm.name == key and lm.gradient_limits == pytest.approx(lim), cls
    assert ScannerLimits.of("bga_12s_hp").gradient_limits == pytest.approx((0.660, 4570.0))
    assert ScannerLimits.of("bga_6s").gradient_limits == pytest.approx((1.000, 11250.0))
    assert not [k for k in scc.SCANNER_CONSTANTS["envelopes"] if k.startswith("certificate_")]
    # the microscopy probes' slew rates are derived from a stated rise time, and say so
    for key in ("bruker_micro2_5", "bruker_micro5"):
        leaf = scc.get_limit(key, "gradient", "max_slew_rate")
        assert leaf["confidence"] == "derived" and "rise time" in leaf["context"] and "lower bound" in leaf["context"]
        assert scc.get_limit(key, "gradient", "max_amplitude")["confidence"] == "cited"
    assert not [m for m, _, _ in scc.needs_verification() if m.startswith("bruker_")]
    for key in ("bruker_almanac_2011", "bruker_micro2_5_web", "bruker_micro5_web", "ramos_llorden2026_connectome2"):
        assert scc.get_citation(key)["doi_or_url"]


def test_safe_model_is_the_representative_example_with_normalised_weights():
    coeffs = ScannerLimits.of("prisma").safe_model
    assert coeffs is not None and [c["axis"] for c in coeffs] == ["x", "y", "z"]
    for c in coeffs:                                   # Hebrank 2000: a1 + a2 + a3 = 1 per axis
        assert c["a1"] + c["a2"] + c["a3"] == pytest.approx(1.0)
    assert ScannerLimits.of("connectom").safe_model == coeffs      # shared data, not a per-coil calibration


def test_public_names():
    assert dmipy_sim.ScannerLimits is ScannerLimits and dmipy_sim.SCANNERS is SCANNERS


pypulseq = pytest.importorskip("pypulseq")


def test_make_system_reads_the_catalogue_and_refuses_the_unknown():
    from dmipy_sim.sequences.pulseq import make_system
    o = make_system("siemens_prisma")
    assert o.grad_raster_time == pytest.approx(1e-5) and o.rf_raster_time == pytest.approx(1e-6)
    assert o.rf_dead_time == 0.0                                          # pypulseq's own default, not ours
    assert make_system("connectom").max_slew == pytest.approx(200.0 * 267.513e6 / (2 * np.pi))   # Hz/m/s
    with pytest.raises(ValueError, match="unknown scanner"):
        make_system("siemens")

def test_the_low_field_class_is_the_cited_hyperfine_swoop():
    """A 64 mT portable magnet is the other end of the catalogue from a Connectom, and the experiment that
    needs it (dmipy-sim#285) must read its numbers here rather than spell them out. The schema carries ONE
    amplitude and slew per model, so the entry is the WEAKEST axis -- what a deliverability check must use --
    with all three in its context."""
    lm = ScannerLimits.of("low_field")
    assert lm.kind == "scanner" and lm.name == "hyperfine_swoop_64mT"
    assert lm.field_T == pytest.approx(0.064)
    assert lm.gradient_limits == pytest.approx((0.0244, 22.0))
    for alias in ("swoop", "hyperfine_swoop", "hyperfine_swoop_64mT"):
        assert ScannerLimits.of(alias).name == "hyperfine_swoop_64mT"
    amp = scc.get_limit("hyperfine_swoop_64mT", "gradient", "max_amplitude")
    assert amp["confidence"] == "cited" and "24.9 / 24.4 / 25.7" in amp["context"]
    slew = scc.get_limit("hyperfine_swoop_64mT", "gradient", "max_slew_rate")
    assert "23 / 22 / 67" in slew["context"]
    for key in ("gholam2025_swoop", "ohalloran2022_dwi_64mt"):
        assert scc.get_citation(key)["doi_or_url"]
    # the magnet's own background gradient is a catalogued number, not a constant in an experiment script
    assert scc.get_limit("hyperfine_swoop_64mT", "homogeneity", "background_gradient")["value"] == pytest.approx(1.4)


# ── the catalogue conforms to its own schema (dmipy-sim#322 PR 0) ───────────────────────────────────
def test_the_schema_ships_and_is_the_spec_repos():
    """`ACQUISITION.md` 8 is the normative text and `scanner_catalogue.schema.json` beside the data is the
    type schema, as `SUBSTRATE.md` and `substrate.schema.json` are for a substrate. Before this the scanner
    catalogue was the one cited-data asset in the ecosystem with no machine-readable schema and no
    conformance check, and three drifts had accumulated in the prose unnoticed."""
    schema = json.loads(scc.SCHEMA_PATH.read_text())
    assert schema["title"].startswith("Scanner limit catalogue")
    assert schema["$schema"].endswith("2020-12/schema")
    assert set(schema["required"]) == {"_schema", "citations", "scanners", "classes", "aliases"}
    assert set(schema["$defs"]["leaf"]["required"]) == set(scc.SCANNER_CONSTANTS["_schema"]["entry_fields"])


def test_the_catalogue_conforms_to_it():
    """Every leaf carries the seven fields with a context and a citation that resolves, every confidence is
    one the file declares, and every short name points at something. This is the guard that was missing."""
    assert scc.conformance_problems() == []


@pytest.mark.parametrize("break_it, expect", [
    (lambda c: c["scanners"]["siemens_magnetom_prisma_3T"]["gradient"]["max_amplitude"].pop("context"), "no context"),
    (lambda c: c["scanners"]["siemens_magnetom_prisma_3T"]["gradient"]["max_amplitude"].update(source_key="nobody"),
     "not in citations"),
    (lambda c: c["scanners"]["siemens_magnetom_prisma_3T"]["gradient"]["max_amplitude"].update(confidence="vibes"),
     "not one of"),
    (lambda c: c["aliases"].update(ghost="a_machine_that_does_not_exist"), "neither a scanner nor an envelope"),
])
def test_the_conformance_check_has_teeth(break_it, expect):
    """A check that passes on a broken catalogue guards nothing, so each rule is shown failing."""
    import copy
    c = copy.deepcopy(scc.SCANNER_CONSTANTS)
    break_it(c)
    problems = scc.conformance_problems(c)
    assert problems and any(expect in p for p in problems), f"{expect!r} not caught; got {problems}"


def test_an_alias_may_name_an_envelope_and_a_citation_may_be_prose_only():
    """Two rules found by checking rather than by reading, and both would make a naive schema reject a
    conforming file: `clinical_typical` resolves into `envelopes`, not `scanners`, and several citations are
    referenced only from an entry's prose `notes` -- a source for the machine rather than for one number."""
    cat = scc.SCANNER_CONSTANTS
    assert cat["aliases"]["clinical_typical"] in cat["envelopes"]
    assert cat["aliases"]["clinical_typical"] not in cat["scanners"]
    cited_by_a_leaf = {leaf.get("source_key")
                       for table in ("scanners", "envelopes")
                       for entry in cat[table].values()
                       for group in entry.values() if isinstance(group, dict)
                       for leaf in group.values() if isinstance(leaf, dict)}
    prose_only = set(cat["citations"]) - cited_by_a_leaf - {cat["safety"].get("source_key")}
    assert prose_only, "no prose-only citation left: the rule is untested"
    assert scc.conformance_problems() == []       # and they are not an error


# ── the magnet's field SHAPE, not just a figure (dmipy-sim#322 PR 2) ────────────────────────────────
def test_the_field_law_is_solved_against_both_cited_figures_at_once():
    """The catalogue held two numbers about the Swoop's field -- 1100 ppm over a 16 cm DSV and up to 1.4
    mT/m at 8 cm -- and nothing in Python read either. They determine the two coefficients BETWEEN them:
    a purely linear field matching the DSV figure would need 0.44 mT/m rather than 1.4, and a purely
    isotropic one reaches only 80 % of the DSV figure. The field needs both a bowl and a tilt, and with
    both it reproduces each cited number exactly."""
    s = ScannerLimits.of("swoop")
    R, B0 = s.b0_validity_radius, s.field_T
    x = np.linspace(-R, R, 20001)
    ppm = 1e6 * (s.b0_asymmetry_rl * x + s.b0_quadratic * x ** 2)
    assert ppm.max() - ppm.min() == pytest.approx(1100.0, rel=1e-3)                 # the DSV figure
    grad = B0 * np.abs(s.b0_asymmetry_rl + 2 * s.b0_quadratic * x).max()
    assert grad == pytest.approx(1.4e-3, rel=1e-3)                                  # the gradient figure
    for leaf in ("b0_quadratic", "b0_asymmetry_rl"):
        assert scc.get_limit("hyperfine_swoop_64mT", "homogeneity", leaf)["confidence"] == "derived"


def test_the_field_law_carries_the_magnets_RL_asymmetry():
    """'The magnet is asymmetric in RL' is an ODD term in x, which a bowl cannot express: a single yoke sits
    on one side, so the field is not mirror-symmetric about isocentre. It is not a small correction -- the
    Swoop's field differs by 716 ppm between +8 and -8 cm, 1054 against 338 -- so an isotropic law would be
    visibly wrong on one side of the bore and not the other."""
    s = ScannerLimits.of("swoop")
    R = s.b0_validity_radius
    plus, minus = (s.b0_offset([[v, 0, 0]])[0] / s.field_T * 1e6 for v in (R, -R))
    assert plus == pytest.approx(1054.0, rel=0.01) and minus == pytest.approx(338.0, rel=0.01)
    assert plus - minus == pytest.approx(716.0, rel=0.01)
    # the odd term acts along R/L only, so a displacement along the bore keeps the bowl alone
    along_z = s.b0_offset([[0, 0, R]])[0] / s.field_T * 1e6
    assert along_z == pytest.approx(1e6 * s.b0_quadratic * R ** 2, rel=1e-9)
    assert s.b0_offset([[0, R, 0]])[0] == pytest.approx(s.b0_offset([[0, 0, R]])[0], rel=1e-9)


def test_the_field_law_is_none_for_a_machine_that_does_not_publish_one():
    """Which is every machine but one. A shimmed superconducting magnet's residual is parts per million and
    its shape is not published; a permanent magnet's is parts per thousand and is. `None` keeps meaning
    'the catalogue does not know' rather than standing in for zero."""
    for name in ("prisma", "connectom", "magnus"):
        s = ScannerLimits.of(name)
        assert s.b0_quadratic is None and s.b0_offset([[0, 0, 0.05]]) is None


def test_the_field_offset_is_zero_at_isocentre_and_refused_beyond_its_anchor():
    s = ScannerLimits.of("swoop")
    assert s.b0_offset([[0.0, 0.0, 0.0]])[0] == pytest.approx(0.0, abs=1e-15)
    at8 = s.b0_offset([[0.0, 0.0, 0.08]])[0]
    # along the bore only the bowl acts, so half the radius is a quarter the offset
    assert s.b0_offset([[0.0, 0.0, 0.04]])[0] == pytest.approx(at8 / 4, rel=1e-9)
    # across it the odd term dominates near isocentre: 2.9 kHz at +8 cm against 0.9 at -8
    assert 42.577e6 * s.b0_offset([[0.08, 0, 0]])[0] == pytest.approx(2872.0, rel=0.01)
    assert 42.577e6 * s.b0_offset([[-0.08, 0, 0]])[0] == pytest.approx(921.0, rel=0.01)
    with pytest.raises(ValueError, match="anchored at 8 cm"):
        s.b0_offset([[0.0, 0.0, 0.12]])


def test_every_leafs_unit_is_one_the_SI_view_knows():
    """An unlisted unit converts by 1.0 and says nothing, which is the quietest failure mode in this file --
    it cost a factor of a million when `ppm/m` was first added here. The conformance check now catches it,
    and catching it turned up four leaves already in the catalogue (a coil diameter in cm, amplifier powers
    in kW and MW) whose units were equally unknown."""
    for table in ("scanners", "envelopes"):
        for name, entry in scc.SCANNER_CONSTANTS[table].items():
            for group, leaves in entry.items():
                if not isinstance(leaves, dict):
                    continue
                for leaf_name, leaf in leaves.items():
                    if isinstance(leaf, dict) and leaf.get("value") is not None and leaf.get("unit"):
                        assert leaf["unit"] in scc._TO_SI, f"{table}.{name}.{group}.{leaf_name}"


def test_the_new_units_convert_and_the_group_is_scanned_for_verification():
    """`ppm` was not in the SI table, and an unknown unit converts silently by 1.0 -- the quietest failure
    mode in this file. `needs_verification` also only scanned gradient and rf, so a homogeneity leaf was
    invisible to it."""
    assert scc._TO_SI["ppm"] == scc._TO_SI["ppm/m"] == scc._TO_SI["ppm/m^2"] == 1e-6
    raw = scc.get_limit("hyperfine_swoop_64mT", "homogeneity", "b0_quadratic")["value"]
    assert scc.get_limit("hyperfine_swoop_64mT", "homogeneity", "b0_quadratic", si=True) == raw * 1e-6
    assert "homogeneity" in inspect.getsource(scc.needs_verification)


# ── the machine's transmit profile (dmipy-sim#322 PR 5) ─────────────────────────────────────────────
def test_the_transmit_profile_is_two_separate_things_that_multiply():
    """`b1_axial_falloff` is SPATIAL: a coil's field weakens toward its ends, so the scale falls as
    1 - a z^2 along the bore and is FLAT across it. That anisotropy is the measurement's, not a
    simplification -- 'little appreciable inhomogeneity in the transverse plane ... 15%-20% variation in the
    superior-inferior direction' over a 144 mm sphere. `b1_calibration_offset` is SYSTEMATIC and applies at
    isocentre too, the machine's own transmit calibration sitting off nominal."""
    s = ScannerLimits.of("swoop")
    assert s.b1_calibration_offset == pytest.approx(190.0 / 180.0, rel=1e-3)   # the 180 null seen at 190
    at_iso = float(s.b1_scale([[0, 0, 0]]))
    assert at_iso == pytest.approx(s.b1_calibration_offset, rel=1e-9)          # systematic, so present at r=0
    # flat transversally: the whole variation is along the bore
    assert float(s.b1_scale([[0.072, 0, 0]])) == pytest.approx(at_iso, rel=1e-9)
    assert float(s.b1_scale([[0, 0.072, 0]])) == pytest.approx(at_iso, rel=1e-9)
    # and 15-20 % of fall-off across the sphere it was measured over
    drop = 1.0 - float(s.b1_scale([[0, 0, 0.072]])) / at_iso
    assert 0.15 <= drop <= 0.20


def test_the_transmit_profile_is_a_machine_property_only_at_low_field():
    """It may live in a catalogue of MACHINES at all because at 2.7 MHz the RF wavelength in tissue is
    metres, so the profile is the coil's geometry and not the subject's: across 47-100 mT a 40-tissue head
    model moves the pattern by 3 % while tissue permittivity changes by about 30 %. The leaf records that
    number and the reason, and the claim is not extended upward -- no other machine carries one."""
    leaf = scc.get_limit("hyperfine_swoop_64mT", "rf", "b1_load_independent")
    assert leaf["value"] == pytest.approx(0.03) and "must NOT be extended" in leaf["context"]
    for name in ("prisma", "connectom", "magnus"):
        assert ScannerLimits.of(name).b1_axial_falloff is None
