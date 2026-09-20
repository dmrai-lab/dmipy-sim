"""The scanner catalogue is one source with typed views (#173 piece 1).

Every class of the band-limit certificate and every Pulseq preset is the cited JSON's number for that machine;
one scanner resolves from any of its names; a slew regime is a choice, not a second catalogue; and what the
catalogue does not know is ``None`` and listed, never a number standing in for one.
"""
import inspect
import json

from dataclasses import replace

import numpy as np
import pytest

import dmipy_sim
from dmipy_sim.acquisition import scanner_constants as scc
from dmipy_sim.acquisition.scanners import GAMMA_BAR, ScannerLimits, SCANNERS, scanner_limits
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
    at_iso = float(s.b1_scale(np.zeros((1, 3)))[0])
    assert at_iso == pytest.approx(s.b1_calibration_offset, rel=1e-9)          # systematic, so present at r=0
    # NOT flat transversally, though the measurement is quoted that way: Laplace forbids it. What the
    # measurement and the physics agree on is that the variation is much SMALLER across than along, and here
    # it is exactly half -- see the harmonic tests below for why that half is not a free choice.
    across = float(s.b1_scale(np.array([[0.072, 0, 0]]))[0]) / at_iso - 1.0
    along = 1.0 - float(s.b1_scale(np.array([[0, 0, 0.072]]))[0]) / at_iso
    assert 0.4 * along < across < 0.6 * along
    # and 15-20 % of fall-off across the sphere it was measured over
    drop = 1.0 - float(s.b1_scale(np.array([[0, 0, 0.072]]))[0]) / at_iso
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


# ── a magnet that drifts with its own temperature (dmipy-sim#285 item 5) ─────────────────────────────
def test_the_drift_coefficient_is_the_machine_s_measured_one_not_its_material_s():
    """The distinction is worth a factor of two, so it is worth a test. Bulk NdFeB remanence falls about
    0.1 %/K, which at 64 mT would be 2725 Hz/K; the Swoop's own measured slope -- 244 phantom scans across
    17 sites, regressed with site as a random effect -- is -1400 Hz/K, half that. A yoked magnet is not all
    permanent-magnet material, and reaching for the materials constant would overstate everything
    downstream. (What this magnet is MADE of is not public in any case: the FDA filings say only 'permanent
    magnet'.)"""
    sw = ScannerLimits.of("swoop")
    assert sw.f0_temperature_slope == pytest.approx(-1400.0)
    assert sw.b0_drift_hz(1.0) == pytest.approx(-1400.0, abs=1.0)
    assert sw.b0_drift(2.0) == pytest.approx(2 * sw.b0_drift(1.0))        # linear, by construction
    # the published Hz/K and the derived fraction are the same number, which is what makes them two leaves
    assert sw.b0_temperature_coefficient * (GAMMA_BAR * sw.field_T) == pytest.approx(sw.f0_temperature_slope)
    assert abs(sw.b0_temperature_coefficient) < 0.6e-3, "that is the materials constant, not the measurement"
    # two kelvin to equal the magnet's whole spatial spread, not one
    spread_ppm = scc.get_limit("hyperfine_swoop_64mT", "homogeneity", "b0_homogeneity")["value"]
    assert 1.8 < spread_ppm / (abs(sw.b0_temperature_coefficient) * 1e6) < 2.5
    leaf = scc.get_limit("hyperfine_swoop_64mT", "thermal", "b0_temperature_coefficient")
    assert "say only 'permanent magnet'" in leaf["context"] and leaf["confidence"] == "derived"
    for name in ("prisma", "connectom", "terra"):
        assert ScannerLimits.of(name).b0_temperature_coefficient is None
        assert ScannerLimits.of(name).b0_drift(1.0) is None


def test_recentring_is_what_makes_the_drift_small_and_it_is_catalogued():
    """A coefficient of 1400 Hz/K would matter enormously if it accumulated. It does not, for two reasons
    the catalogue records: the pre-scan calibration removes the between-session ambient term entirely, and
    within a scan the protocol re-centres f0 after every two DWIs. So what reaches the data is the drift
    accrued within that interval and never the total. The leaf says plainly that no within-scan drift RATE
    is published for this machine -- the residual quoted there is carried over from another magnet's
    warming rate and is labelled as such."""
    sw = ScannerLimits.of("swoop")
    assert sw.f0_recentering_interval == pytest.approx(339.0)             # 2 x 212 shots at TR 800 ms
    leaf = scc.get_limit("hyperfine_swoop_64mT", "thermal", "f0_recentering_interval")
    assert leaf["confidence"] == "cited" and "NO within-scan drift RATE is published" in leaf["context"]
    # the span the magnet is uncontrolled over is the reason f0 is a calibration, not an assumption
    assert scc.get_limit("hyperfine_swoop_64mT", "thermal", "operating_temperature_span")["value"] == 15.0
    assert abs(sw.b0_drift_hz(15.0)) > 20e3                               # twenty kilohertz across the range


# ── the magnet's own encoding gradient (dmipy-sim#322 PR 6) ──────────────────────────────────────────
def test_the_background_gradient_is_the_field_law_differentiated_not_a_second_number():
    """The catalogue publishes a background gradient AND a field shape, and they must be the same magnet.
    `b0_gradient` is the shape's derivative in closed form, so it agrees with a finite difference of
    `b0_offset` to machine precision -- and it lands on the independently cited 1.4 mT/m at 8 cm, which is
    the check that the shape was solved correctly in the first place."""
    s = ScannerLimits.of("swoop")
    h = 1e-6
    for p in ([0.0, 0, 0], [0.05, 0, 0], [-0.05, 0, 0], [0, 0.06, 0], [0.03, -0.02, 0.05]):
        p = np.asarray(p, float)
        num = np.array([(s.b0_offset((p + h * e)[None])[0] - s.b0_offset((p - h * e)[None])[0]) / (2 * h)
                        for e in np.eye(3)])
        np.testing.assert_allclose(s.b0_gradient(p[None])[0], num, rtol=1e-6, atol=1e-12)
    # the cited anchor: 1.4 mT/m at 8 cm, on the side the odd term adds to
    steep = np.linalg.norm(s.b0_gradient(np.array([[0.0799, 0, 0]]))[0])
    assert steep == pytest.approx(1.4e-3, rel=2e-3)
    cited = scc.get_limit("hyperfine_swoop_64mT", "homogeneity", "background_gradient")
    assert steep * 1e3 == pytest.approx(cited["value"], rel=2e-3)


def test_the_background_gradient_does_not_vanish_at_isocentre_because_the_magnet_is_asymmetric():
    """A bowl differentiates to zero at its centre; an odd term does not. So a single-yoke magnet encodes
    diffusion even at isocentre, and anything that assumed "no error at the origin" is wrong about THIS
    magnet. The residue is the odd coefficient exactly, which is what makes it checkable rather than a
    surprise."""
    s = ScannerLimits.of("swoop")
    at_iso = s.b0_gradient(np.zeros((1, 3)))[0]
    assert at_iso[0] == pytest.approx(s.field_T * s.b0_asymmetry_rl)     # the odd term, alone
    np.testing.assert_allclose(at_iso[1:], 0.0, atol=1e-18)              # the bowl contributes nothing
    assert np.linalg.norm(at_iso) == pytest.approx(2.86e-4, rel=1e-2)    # 0.29 mT/m
    # and the two sides of the bore differ, which is the asymmetry showing up in the derivative
    hi = np.linalg.norm(s.b0_gradient(np.array([[0.0799, 0, 0]]))[0])
    lo = np.linalg.norm(s.b0_gradient(np.array([[-0.0799, 0, 0]]))[0])
    assert hi > 1.6 * lo, f"the derivative lost the asymmetry: {hi*1e3:.3f} vs {lo*1e3:.3f} mT/m"


def test_a_derivative_is_refused_beyond_the_law_s_anchor_and_absent_without_one():
    s = ScannerLimits.of("swoop")
    with pytest.raises(ValueError, match="extrapolates worse"):
        s.b0_gradient(np.array([[0.12, 0, 0]]))
    for name in ("prisma", "connectom", "terra"):
        assert ScannerLimits.of(name).b0_gradient(np.array([[0, 0, 0.05]])) is None


# ── which axis is the field, and which is the coil (dmipy-sim#349 item 0) ────────────────────────────
def test_every_machine_declares_where_its_field_points():
    """A missing axis and an axis that happens to be head-foot are not the same claim. Every cylindrical
    magnet's B0 runs along the bore, which is the patient's head-foot direction, and that is stated rather
    than assumed -- because one machine here does something else."""
    for name in ("prisma", "connectom", "terra", "swoop"):
        assert ScannerLimits.of(name).b0_axis is not None, f"{name} declares no field direction"
    assert ScannerLimits.of("prisma").b0_axis == (0.0, 0.0, 1.0)          # S: along the bore
    assert ScannerLimits.of("swoop").b0_axis == (0.0, 1.0, 0.0)           # A: across the patient


def test_the_swoop_s_field_is_not_along_its_bore_which_is_the_whole_reason_for_a_magnet_frame():
    """The fact that makes the frame necessary. On a cylindrical magnet the field, the bore and the transmit
    coil all coincide, so nothing distinguishes them and a simulator can call them all 'z' forever. A
    bi-planar magnet puts B0 across the patient and the coil along them -- and the two 'z's are ninety
    degrees apart."""
    sw = ScannerLimits.of("swoop")
    b0, b1 = np.asarray(sw.b0_axis), np.asarray(sw.b1_axis)
    assert abs(float(b0 @ b1)) < 1e-12                                   # perpendicular, as a solenoid must be
    assert abs(float(b0 @ np.asarray(ScannerLimits.of("prisma").b0_axis))) < 1e-12   # and not the bore's axis
    leaf = scc.get_limit("hyperfine_swoop_64mT", "frame", "b0_axis")
    assert leaf["confidence"] == "cited" and "vertical" in leaf["context"]


def test_a_coil_along_the_field_is_refused_because_it_would_not_excite():
    """Physics, not bookkeeping: only the component of B1 perpendicular to B0 excites, so a machine whose
    coil axis lies along its field would produce no signal. The confusion that produces such an entry is
    exactly the kind that never announces itself -- both are plausible unit vectors and every number
    downstream stays finite."""
    sw = ScannerLimits.of("swoop")
    with pytest.raises(ValueError, match="not perpendicular"):
        replace(sw, b1_axis=sw.b0_axis)._check_frame()
    with pytest.raises(ValueError, match="axis letter"):
        ScannerLimits.of("swoop").__class__ and _axis_check()


def _axis_check():
    from dmipy_sim.acquisition.scanners import _axis
    return _axis("Q", "b0_axis", "made-up")


def test_the_magnet_frame_puts_the_field_on_z_by_construction():
    """Every piece of field physics -- the concomitant expansion, the EPG states, off-resonance, the
    susceptibility contraction -- is written for B0 along +z and none of them says so in a way a caller can
    check. This is where that assumption becomes a value."""
    for name in ("prisma", "swoop"):
        s = ScannerLimits.of(name)
        R = s.magnet_frame()
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)        # a rotation
        assert np.linalg.det(R) == pytest.approx(1.0)                     # a PROPER one
        np.testing.assert_allclose(R @ np.asarray(s.b0_axis), [0, 0, 1], atol=1e-12)
    # on a cylindrical magnet it is the identity, which is why the assumption survived unexamined
    np.testing.assert_allclose(ScannerLimits.of("prisma").magnet_frame(), np.eye(3), atol=1e-12)
    assert not np.allclose(ScannerLimits.of("swoop").magnet_frame(), np.eye(3))


def test_the_transmit_fall_off_is_projected_onto_the_coil_not_read_off_an_index():
    """The discriminating test. For a coil along head-foot, projecting and indexing `[..., 2]` give the same
    number, so a frame-conflated implementation passes anything built from this machine alone. Move the coil
    to another axis and they part company -- which is what says the fall-off follows the COIL rather than a
    coordinate that happens to coincide with it."""
    sw = ScannerLimits.of("swoop")
    d = np.array([[0.02, -0.01, 0.05], [0.0, 0.06, 0.0], [0.03, 0.0, 0.07]])
    a, cal = sw.b1_axial_falloff, sw.b1_calibration_offset
    r2 = np.sum(d * d, axis=1)

    def law(along2):
        return cal * (1.0 - a * along2 + 0.5 * a * (r2 - along2))

    np.testing.assert_allclose(sw.b1_scale(d), law(d[:, 2] ** 2), rtol=1e-12)   # coil along S
    turned = replace(sw, b1_axis=(1.0, 0.0, 0.0), b0_axis=(0.0, 1.0, 0.0))      # coil moved to R/L
    assert not np.allclose(turned.b1_scale(d), sw.b1_scale(d))                  # ... and they part company
    np.testing.assert_allclose(turned.b1_scale(d), law(d[:, 0] ** 2), rtol=1e-12)


def test_a_fall_off_with_no_axis_to_fall_off_along_is_refused():
    """A quadrature birdcage drives two orthogonal transverse components and so has no single B1 axis; a
    fall-off law is meaningless for one. The catalogue says which kind of coil each machine has, and asking
    for the law without the axis is refused rather than defaulted."""
    assert ScannerLimits.of("prisma").b1_axis is None
    bad = replace(ScannerLimits.of("swoop"), b1_axis=None)
    with pytest.raises(ValueError, match="no b1_axis"):
        bad.b1_scale(np.zeros((2, 3)))


# ── the transmit profile is a field too (dmipy-sim#349 item 3) ───────────────────────────────────────
def test_the_transmit_scale_is_harmonic_where_a_flat_one_would_not_be():
    """At 2.7 MHz the coil bore is quasi-static, so the transmit field obeys div B = 0 and curl B = 0 there
    and its axial component is harmonic. A profile that falls along the coil and is FLAT across it is not a
    solution of those equations -- it is a statement about a field that cannot exist, in the same way an
    isotropic r^2 static field cannot."""
    s = ScannerLimits.of("swoop")
    h, a = 1e-4, s.b1_axial_falloff
    for p in ([0.0, 0.0, 0.05], [0.04, 0.02, 0.0], [0.03, -0.03, 0.06]):
        p = np.asarray([p])
        lap = sum(float(s.b1_scale(p + h * e)[0]) - 2 * float(s.b1_scale(p)[0]) + float(s.b1_scale(p - h * e)[0])
                  for e in np.eye(3)) / h ** 2
        assert abs(lap) < 1e-6, f"the transmit scale is not harmonic at {p}: laplacian {lap:.2e}"
        # the flat version, for contrast: its laplacian is -2a, nowhere near zero
        flat = lambda q: s.b1_calibration_offset * (1.0 - a * q[0, 2] ** 2)
        lap_flat = sum(flat(p + h * e) - 2 * flat(p) + flat(p - h * e) for e in np.eye(3)) / h ** 2
        assert abs(lap_flat) > 1e-6


def test_the_transverse_rise_is_exactly_half_the_axial_fall_and_costs_no_parameter():
    """The paraxial expansion of a quasi-static field is ``B(s) - (rho^2/4) B''(s)``, so asserting the axial
    behaviour DETERMINES the transverse behaviour. There is nothing to fit: the field must rise off-axis at
    exactly half the rate it falls along the axis, and a catalogue entry for it would be a second copy of a
    number already there."""
    s = ScannerLimits.of("swoop")
    for t in (0.03, 0.05, 0.07):
        fall = 1.0 - float(s.b1_scale(np.array([[0, 0, t]]))[0]) / s.b1_calibration_offset
        rise = float(s.b1_scale(np.array([[t, 0, 0]]))[0]) / s.b1_calibration_offset - 1.0
        assert rise / fall == pytest.approx(0.5, rel=1e-9)


def test_the_axial_profile_is_untouched_so_the_published_measurement_still_holds():
    """The correction adds a transverse term; it does not move the axis. What was measured and catalogued was
    the fall-off ALONG the coil, and that is unchanged -- so this fixes a field without disturbing a
    number."""
    s = ScannerLimits.of("swoop")
    for t in (0.0, 0.04, 0.072):
        on_axis = float(s.b1_scale(np.array([[0, 0, t]]))[0])
        assert on_axis == pytest.approx(s.b1_calibration_offset * (1.0 - s.b1_axial_falloff * t ** 2), rel=1e-12)
