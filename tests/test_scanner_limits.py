"""The scanner catalogue is one source with typed views (#173 piece 1).

Every class of the band-limit certificate and every Pulseq preset is the cited JSON's number for that machine;
one scanner resolves from any of its names; a slew regime is a choice, not a second catalogue; and what the
catalogue does not know is ``None`` and listed, never a number standing in for one.
"""
import inspect
import json

from dataclasses import replace

import re

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
    # 23 mT/m, not the literature's 24.4: the FDA filing's weakest axis is 23 and the two specs conflict,
    # so the only figure safe under either is the smaller. See per_axis_amplitude_* for the conflict.
    assert lm.gradient_limits == pytest.approx((0.023, 22.0))
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
def test_the_field_law_is_a_sum_of_solid_harmonics_and_nothing_else():
    """The constraint that decides the form. A magnet's field in the imaging volume solves Laplace's
    equation, so it is a sum of solid harmonics; anything else is not a field. The previous description used
    an isotropic r^2 bowl, whose entire harmonic content is a constant with ZERO gradient -- so every
    background gradient taken from it was an artefact of the basis, and it put a strict minimum of the field
    at isocentre, which the maximum principle forbids."""
    s = ScannerLimits.of("swoop")
    h = 2e-4
    for p in ([0.012, -0.019, 0.021], [-0.03, 0.01, 0.04], [0.0, 0.0, 0.0]):
        p = np.asarray([p])
        lap = sum(float(s.b0_offset(p + h * e)) - 2 * float(s.b0_offset(p)) + float(s.b0_offset(p - h * e))
                  for e in np.eye(3)) / h ** 2
        assert abs(lap) < 1e-9, f"the field is not harmonic at {p}: laplacian {lap:.2e}"
        # and the gradient of a harmonic field is divergence-free -- the bowl's was not
        div = sum((s.b0_gradient(p + h * e)[0][i] - s.b0_gradient(p - h * e)[0][i]) / (2 * h)
                  for i, e in enumerate(np.eye(3)))
        assert abs(div) < 1e-9, f"the gradient field has a source at {p}: div {div:.2e}"


def test_the_field_law_reproduces_both_cited_figures_at_once():
    """Two coefficients against two published numbers, and it is the ORDERS that are inferred rather than
    fitted: for a pure zonal harmonic of order l the steepest gradient at a given peak-to-peak over a ball is
    fixed at 0.50 / 1.33 / 1.51 / 2.80 times (p-p)/R for l = 1 / 2 / 3 / 4. The cited 1.4 mT/m at 1100 ppm
    needs 1.59, which no expansion through l=2 can reach -- so this magnet must carry l>=3 content."""
    s = ScannerLimits.of("swoop")
    R = s.b0_validity_radius
    rng = np.random.default_rng(0)
    u = rng.normal(size=(120000, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    P = u * (0.999 * R * rng.random(120000) ** (1 / 3))[:, None]
    ppm = s.b0_offset(P) / s.field_T * 1e6
    grad = np.linalg.norm(s.b0_gradient(P), axis=1) * 1e3
    assert np.ptp(ppm) == pytest.approx(1100.0, rel=0.03)
    assert grad.max() == pytest.approx(1.4, rel=0.03)
    # the ratio that forces l >= 3
    needed = 1.4e-3 / (1100e-6 * s.field_T / R)
    assert needed > 1.34, f"ratio {needed:.2f} would be reachable at l=2; the inference would not hold"


def test_the_field_law_carries_the_magnets_RL_asymmetry_at_order_three():
    """The asymmetry is real and it cannot be the linear term. The published homogeneity is a POST-LINEAR-
    SHIM residual, so l=1 has been nulled and cannot be fitted to it -- but the magnet is still described as
    asymmetric in RL, and an odd asymmetry with no l=1 must live at l=3."""
    s = ScannerLimits.of("swoop")
    R = 0.999 * s.b0_validity_radius
    hi = float(np.linalg.norm(s.b0_gradient(np.array([[R, 0, 0]]))[0]))
    lo = float(np.linalg.norm(s.b0_gradient(np.array([[-R, 0, 0]]))[0]))
    assert hi > 1.5 * lo, f"the two sides of the bore are alike: {hi*1e3:.3f} vs {lo*1e3:.3f} mT/m"
    assert s.b0_harmonic_Z2X is not None and s.b0_asymmetry_axis == (1.0, 0.0, 0.0)
    leaf = scc.get_limit("hyperfine_swoop_64mT", "homogeneity", "b0_harmonic_Z2X")
    assert "POST-LINEAR-SHIM" in leaf["context"] and "l<=2 cannot" in leaf["context"]

def test_the_field_law_is_none_for_a_machine_that_does_not_publish_one():
    """Which is every machine but one, and the reason is the SHAPE rather than the magnitude. A shimmed
    superconducting magnet's residual is parts per million and a permanent magnet's parts per thousand, but
    what decides whether a law exists is whether the spatial form was published. `None` keeps meaning 'the
    catalogue does not know' rather than standing in for zero."""
    for name in ("prisma", "connectom", "magnus"):
        s = ScannerLimits.of(name)
        assert s.b0_harmonic_Z2 is None and s.b0_offset([[0, 0, 0.05]]) is None


def test_a_published_homogeneity_is_a_magnitude_and_not_a_law():
    """The Prisma's homogeneity IS published and measured -- Gach 2020 Table III, 0.386 ppm pk-pk over a
    24 cm DSV -- and it still buys no field law, because a shape-free spread over a sphere does not
    determine a harmonic expansion. The same table refutes the obvious guess: a pure Z2 anchored to the
    24 cm figure predicts 0.821 ppm over 35 cm where 2.378 was measured, so the residual is not a bowl.
    Recording the magnitude while leaving the law absent is the distinction this catalogue has to keep."""
    entry = scc.get_scanner("siemens_magnetom_prisma_3T")
    assert scc.leaf_si(entry, "homogeneity", "b0_homogeneity") == pytest.approx(0.386e-6)
    assert ScannerLimits.of("prisma").b0_harmonic_Z2 is None       # magnitude known, shape not

    # the two DSV rows, which is what makes "not a bowl" a measurement rather than an opinion
    from dmipy_sim.acquisition import solid_harmonics as sh
    rng = np.random.default_rng(0)
    u = rng.normal(size=(200000, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    pk = lambda R: np.ptp(sh.evaluate({"Z2": 1.0}, u * R))
    predicted_35 = 0.386 * pk(0.175) / pk(0.12)
    assert predicted_35 == pytest.approx(0.821, rel=0.01)          # a pure bowl's prediction
    assert predicted_35 < 0.5 * 2.378                              # against the measured 35 cm figure


def test_the_field_offset_is_zero_at_isocentre_and_refused_beyond_its_anchor():
    s = ScannerLimits.of("swoop")
    assert s.b0_offset([[0.0, 0.0, 0.0]])[0] == pytest.approx(0.0, abs=1e-15)
    at8 = s.b0_offset([[0.0, 0.0, 0.08]])[0]
    # along the bore only the bowl acts, so half the radius is a quarter the offset
    assert s.b0_offset([[0.0, 0.0, 0.04]])[0] == pytest.approx(at8 / 4, rel=1e-9)
    # across it the odd term dominates near isocentre: 2.9 kHz at +8 cm against 0.9 at -8
    # the offset at a given point is a property of the LAW, not a published number; what is published is
    # the spread over the DSV, so that is what is asserted
    rng = np.random.default_rng(1)
    u = rng.normal(size=(60000, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    P = u * (0.999 * s.b0_validity_radius * rng.random(60000) ** (1 / 3))[:, None]
    assert np.ptp(s.b0_offset(P) / s.field_T * 1e6) == pytest.approx(1100.0, rel=0.03)
    hi = float(s.b0_offset(np.array([[0.0755, 0, 0]])))
    lo = float(s.b0_offset(np.array([[-0.0755, 0, 0]])))
    assert abs(hi - lo) > 0.2 * max(abs(hi), abs(lo))   # the odd harmonic, doing its job
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
    raw = scc.get_limit("hyperfine_swoop_64mT", "homogeneity", "b0_harmonic_Z2")["value"]
    # a solid-harmonic coefficient of dB/B0 is already dimensionless per length, so SI is a no-op --
    # unlike the ppm leaves beside it, which carry a factor of a million
    assert scc.get_limit("hyperfine_swoop_64mT", "homogeneity", "b0_harmonic_Z2", si=True) == raw
    assert scc.get_limit("hyperfine_swoop_64mT", "homogeneity", "b0_homogeneity", si=True) == \
        scc.get_limit("hyperfine_swoop_64mT", "homogeneity", "b0_homogeneity")["value"] * 1e-6
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
    # the cited anchor is the STEEPEST gradient inside the DSV, which a harmonic law need not place on the
    # R/L axis -- and for this magnet does not: the l=2 bowl is steepest along B0 while the l=3 term tilts it
    rng = np.random.default_rng(2)
    u = rng.normal(size=(80000, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    P = u * (0.999 * s.b0_validity_radius)
    steep = np.linalg.norm(s.b0_gradient(P), axis=1).max()
    assert steep == pytest.approx(1.4e-3, rel=0.02)
    cited = scc.get_limit("hyperfine_swoop_64mT", "homogeneity", "background_gradient")
    assert steep * 1e3 == pytest.approx(cited["value"], rel=2e-3)


def test_the_background_gradient_vanishes_at_isocentre_because_the_magnet_was_shimmed():
    """This REVERSES what an earlier, inadmissible law said. That law carried a free linear term fitted to
    the 1100 ppm figure, which gave a non-zero gradient at the origin and an apparent ADC bias at the one
    point one would expect to be clean. But the source states the 1100 ppm is measured AFTER linear shimming,
    so the l=1 content is nulled and fitting it twice was double-counting. Every harmonic of order two or
    more has zero gradient at the origin, so a linearly shimmed magnet does not encode diffusion at
    isocentre: the background gradient grows from nothing."""
    s = ScannerLimits.of("swoop")
    np.testing.assert_allclose(s.b0_gradient(np.zeros((1, 3)))[0], 0.0, atol=1e-15)
    assert s.b0_offset(np.zeros((1, 3)))[0] == pytest.approx(0.0, abs=1e-15)
    # and it grows monotonically away from there along any ray
    along = [float(np.linalg.norm(s.b0_gradient(np.array([[0.0, 0.0, t]]))[0])) for t in (0.0, 0.02, 0.05, 0.078)]
    assert all(np.diff(along) > 0)

def test_a_derivative_is_refused_beyond_the_law_s_anchor_and_absent_without_one():
    s = ScannerLimits.of("swoop")
    with pytest.raises(ValueError, match="never constrained"):
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


# ── a per-axis figure needs an axis assignment nobody has published (dmipy-sim#349) ──────────────────
def test_the_two_published_gradient_specs_disagree_and_both_are_recorded():
    """A regulatory filing and three peer-reviewed papers give different per-axis gradient amplitudes for the
    same machine -- 39 against 25.7 mT/m on the third axis, a factor 1.52. It is not a hardware generation
    and not one paper's error. The catalogue records BOTH rather than adjudicating, because adjudicating
    would be inventing a fact."""
    fda = scc.get_limit("hyperfine_swoop_64mT", "gradient", "per_axis_amplitude_fda")["value"]
    lit = scc.get_limit("hyperfine_swoop_64mT", "gradient", "per_axis_amplitude_literature")["value"]
    assert fda[2] / lit[2] == pytest.approx(1.52, rel=0.02)
    assert max(fda[:2]) - min(fda[:2]) < 2.0 and max(lit[:2]) - min(lit[:2]) < 2.0   # first two axes agree
    # the scalar the deliverability check uses is the WEAKEST axis, so it is safe under either spec
    s = ScannerLimits.of("swoop")
    assert s.G_max * 1e3 == pytest.approx(min(min(fda), min(lit)), rel=1e-9), \
        "G_max must be the weakest axis across BOTH specs, not the weakest of one of them"


def test_a_direction_dependent_maximum_is_refused_while_the_axis_mapping_is_unknown():
    """The refusal is the feature. Per-axis amplitudes are unusable without knowing which physical direction
    each vendor axis names, and for this machine that is not public: the filing and the literature conflict,
    and Hyperfine's own hysteresis patent contradicts the inference the other sources support. Returning the
    better-supported reading would produce a number indistinguishable from a measurement."""
    s = ScannerLimits.of("swoop")
    assert s.gradient_axis_assignment is None
    with pytest.raises(ValueError, match="not public|does not declare"):
        s.G_max_along([0, 1, 0])
    # a machine that DOES declare one answers, and the answer respects the weakest axis along that direction
    declared = replace(s, gradient_axis_assignment=((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)))
    assert declared.G_max_along([1, 0, 0]) == pytest.approx(s.G_max)
    assert declared.G_max_along([1, 1, 0]) > s.G_max        # an oblique direction shares the load


def test_the_conflict_and_the_measurement_that_narrows_it_are_both_written_down():
    leaf = scc.get_limit("hyperfine_swoop_64mT", "gradient", "per_axis_amplitude_literature")
    for phrase in ("IRRECONCILABLE", "Not a hardware generation", "83.600", "hysteresis"):
        assert phrase in leaf["context"], f"the leaf does not record {phrase!r}"
    # and the slew disagreement is explicitly marked as irrelevant to the concomitant field
    assert "never see a slew rate" in leaf["context"]


def test_the_field_law_is_one_admissible_member_and_says_how_wide_the_family_is():
    """Two published scalars against fifteen solid-harmonic coefficients is an underdetermined fit, and the
    catalogue records the width of what is left open rather than presenting one member as the answer.

    The split is the useful part. Both constraints are scale-like, so the SCALE is determined -- every
    admissible law has the same steepest gradient, by construction. The SHAPE is not: sampled laws disagree
    about the background gradient's direction by a median 71 degrees, which is not a correction to a known
    field but a different one. So a quantity driven by the worst gradient is trustworthy and one driven by
    where the gradient points is not."""
    s = ScannerLimits.of("swoop")
    assert s.b0_direction_spread_deg == pytest.approx(71.0)
    leaf = scc.get_limit("hyperfine_swoop_64mT", "homogeneity", "b0_harmonic_Z2")
    assert "ONE ADMISSIBLE LAW, NOT THE LAW" in leaf["context"]
    assert "400 laws" in leaf["context"] and "18 per cent" in leaf["context"]
    # the point spread is far wider than the shell-average one: shape is open where scale is not
    point = scc.get_limit("hyperfine_swoop_64mT", "homogeneity", "b0_gradient_point_spread")["value"]
    assert point == pytest.approx(0.36)
    spread_leaf = scc.get_limit("hyperfine_swoop_64mT", "homogeneity", "b0_gradient_point_spread")
    assert "1.400 mT/m" in spread_leaf["context"]     # the scale is fixed where the shape is not


def test_the_order_three_argument_is_over_mixtures_not_pure_terms():
    """The conclusion is right and the reasoning had to be fixed. Bounding each order by its PURE terms is
    too weak -- pure l=3 reaches only 1.572 against a required 1.591, which would have argued for l=4.
    Mixtures within an order do better: l<=3 reaches 1.971 and l<=2 only 1.364. So l>=3 is required and l=3
    suffices, and the leaf states the bound over mixtures."""
    leaf = scc.get_limit("hyperfine_swoop_64mT", "homogeneity", "b0_harmonic_Z2X")
    assert "ALL MIXTURES" in leaf["context"]
    for n in ("1.591", "1.364", "1.971"):
        assert n in leaf["context"], f"the leaf does not record {n}"
@pytest.mark.slow
def test_every_catalogued_doi_resolves_to_the_paper_it_claims():
    """A citation that does not resolve reads as provenance and is not. This caught a real one: the de Vos
    concomitant-field DOI in this catalogue pointed at "2D sodium MRI of the human calf", and it was the
    source the entire alpha = 1/2 inference rested on.

    Network-gated and marked slow, because it is the only test here that leaves the machine. It checks that
    the DOI resolves AND that the title it resolves to shares vocabulary with the title we recorded -- a DOI
    that resolves to the wrong paper is exactly the failure mode, so mere resolution is not enough."""
    import urllib.error
    import urllib.request

    cat = scc.SCANNER_CONSTANTS
    dois = {k: re.search(r"10\.\d{4,9}/\S+", c.get("doi_or_url", ""))
            for k, c in cat["citations"].items()}
    dois = {k: m.group(0) for k, m in dois.items() if m}
    assert len(dois) > 10, "the catalogue lost its DOIs"

    bad = []
    for key, doi in sorted(dois.items()):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(f"https://api.crossref.org/works/{doi}",
                                           headers={"Accept": "application/json"}), timeout=20) as r:
                title = json.load(r)["message"]["title"][0]
        except urllib.error.URLError as e:                      # offline, or Crossref is down
            pytest.skip(f"no network for the DOI check: {e}")
        except Exception:
            bad.append(f"{key}: {doi} did not resolve")
            continue
        ours = set(re.findall(r"[a-z]{5,}", cat["citations"][key]["title"].lower()))
        theirs = set(re.findall(r"[a-z]{5,}", re.sub(r"<[^>]+>", " ", title).lower()))
        if ours and not (ours & theirs):
            bad.append(f"{key}: {doi} resolves to {title[:60]!r}, which shares no words with ours")
    assert not bad, "\n".join(bad)


def test_the_hcp_pair_are_two_machines_and_neither_is_the_one_they_are_mistaken_for():
    """HCP's co-registered 3 T / 7 T subjects were scanned on two machines the catalogue holds separately,
    because in both cases the obvious nearby entry is the wrong one.

    The 7 T is an Agilent magnet with a MAGNETOM 7T Plus console and an SC72 gradient, not a Terra; the 3 T is
    the WU-Minn Connectome Skyra, not the MGH Connectom and not a stock Skyra. Reaching for the lookalike
    overstates the 7 T gradient by 14 % and the 3 T gradient by 3x, which is why the separation is a test and
    not a comment."""
    hcp7 = scc.get_scanner("siemens_magnetom_7t_plus_cmrr")
    terra = scc.get_scanner("siemens_magnetom_terra_7T")
    assert hcp7["field_T"] == terra["field_T"] == 7.0
    assert scc.leaf_si(hcp7, "gradient", "max_amplitude") == pytest.approx(70e-3)
    assert scc.leaf_si(terra, "gradient", "max_amplitude") == pytest.approx(80e-3)

    skyra = scc.get_scanner("siemens_connectome_skyra_3T")
    assert scc.leaf_si(skyra, "gradient", "max_amplitude") == pytest.approx(100e-3)
    assert scc.leaf_si(scc.get_scanner("siemens_magnetom_connectom_3T"),
                       "gradient", "max_amplitude") == pytest.approx(300e-3)


def test_the_connectome_skyra_trades_amplitude_against_slew_by_regime():
    """Its two published regimes go opposite ways: diffusion gets the amplitude, readout gets the slew. A
    single (Gmax, slew) pair cannot describe this machine, so asking for the bare maxima and using them
    together would describe a waveform it cannot play."""
    e = scc.get_scanner("siemens_connectome_skyra_3T")
    assert scc.leaf_si(e, "gradient", "max_amplitude") > scc.leaf_si(e, "gradient", "max_amplitude_readout")
    assert scc.leaf_si(e, "gradient", "max_slew_rate_diffusion") < scc.leaf_si(e, "gradient", "max_slew_rate")


def test_a_refused_coefficient_is_recorded_with_its_refusal_and_not_left_blank():
    """The Connectome Skyra's gradient-nonlinearity coefficients are withheld by the vendor. That is a fact
    about the world with a citation, so it is a leaf carrying the quote -- not a missing key, which would be
    indistinguishable from nobody having looked."""
    leaf = scc.get_limit("siemens_connectome_skyra_3T", "gradient_nonlinearity",
                         "spherical_harmonic_coefficients")
    assert leaf["value"] is None
    assert leaf["confidence"] == "NEEDS VERIFICATION"
    assert "proprietary information" in leaf["context"]
    assert leaf["source_key"] in scc.SCANNER_CONSTANTS["citations"]


def test_a_homogeneity_without_a_stated_metric_says_so():
    """A ppm figure is meaningless until it says pk-pk, VRMS or FWHM: Gach 2020 measured 0.386 against 0.024
    on one Prisma, a factor of 16. CMRR's 7 T figure does not state its metric, so the leaf must carry that
    caveat rather than let the number be compared with one that does."""
    leaf = scc.get_limit("siemens_magnetom_7t_plus_cmrr", "homogeneity", "b0_homogeneity")
    assert leaf["value"] == 5.0 and leaf["unit"] == "ppm"
    assert "METRIC IS NOT STATED" in leaf["context"]
    assert ScannerLimits.of("siemens_magnetom_7t_plus_cmrr").b0_harmonic_Z2 is None


def test_the_vendor_homogeneity_series_is_order_four_and_still_not_a_field_law():
    """A spread over ONE sphere fixes only a magnitude, but a series over NESTED spheres fixes a radial order:
    a degree-n solid harmonic has VRMS proportional to R**n, so the log-log slope is n.

    Three vendors' published series all give n ~ 4, not the 2 a bowl-shaped residual would give -- a magnet
    cancels its low orders over the imaging volume and what survives is high-order. The order is re-derived
    here from the raw rows rather than read back, so a mistyped leaf fails. It still licenses no field law:
    VRMS over a sphere is angularly blind, and degree 2 alone holds five different terms."""
    series = {                                    # DSV cm -> guaranteed maximum VRMS ppm, Gach Tables S2/S3/S5
        "siemens_magnetom_prisma_3T": {10: 0.001, 20: 0.02, 30: 0.10, 40: 0.20, 45: 0.50, 50: 1.50},
        "philips_ingenia_3T": {10: 0.0022, 20: 0.022, 30: 0.08, 40: 0.45, 45: 1.20},
        "ge_signa_premier_3T": {20: 0.05, 30: 0.15, 40: 0.50, 45: 1.50, 50: 4.00},
    }
    for key, rows in series.items():
        dsv = sorted(rows)
        n = np.polyfit(np.log(np.array(dsv) / 2.0), np.log([rows[k] for k in dsv]), 1)[0]
        leaf = scc.get_limit(key, "homogeneity", "b0_spec_radial_order")
        assert leaf["value"] == pytest.approx(n, abs=0.01), f"{key}: leaf says {leaf['value']}, rows give {n}"
        assert leaf["confidence"] == "derived"
        assert 3.0 < n < 5.0, f"{key}: order {n} is nowhere near the published envelope"
        assert ScannerLimits.of(key).b0_harmonic_Z2 is None, f"{key} grew a field law from a radial order"


def test_the_prisma_residual_is_degree_two_only_before_the_second_order_shim():
    """The one place a degree-2 residual is measurable is the gradshim-only state, and it does not survive
    shimming: the same magnet's full-shim residual is not a single harmonic at all, its piecewise order running
    1.0 at 10-20 cm to 3.9 at 30-35 cm. That contrast is why b0_homogeneity records a magnitude and refuses a
    shape, and it is a fact about this magnet -- Gach's Vida fits 1.21 in the same state."""
    def order(rows):
        dsv = sorted(rows)
        return np.polyfit(np.log(np.array(dsv) / 2.0), np.log([rows[k] for k in dsv]), 1)[0]

    gradshim = {10: 0.012, 20: 0.040, 24: 0.058, 30: 0.094, 35: 0.136}      # Gach Table S10, Prisma
    full = {10: 0.007, 20: 0.014, 24: 0.019, 30: 0.037, 35: 0.067}
    leaf = scc.get_limit("siemens_magnetom_prisma_3T", "homogeneity", "b0_measured_radial_order")
    assert leaf["value"] == pytest.approx(order(gradshim), abs=0.01)
    assert order(gradshim) == pytest.approx(2.0, abs=0.15), "the gradshim-only residual stopped looking degree-2"

    lo = np.log(full[20] / full[10]) / np.log(2.0)
    hi = np.log(full[35] / full[30]) / np.log(35 / 30)
    assert lo < 1.5 < hi, "the full-shim residual is supposed to be a MIXTURE, not one harmonic"
    assert ScannerLimits.of("prisma").b0_harmonic_Z2 is None


def test_every_three_tesla_homogeneity_is_vrms_at_a_stated_dsv():
    """Gach 2020 exists because a bare 'ppm' is ambiguous between pk-pk, VRMS and FWHM, which differ 16x on one
    machine. So every homogeneity number the catalogue carries must say its metric and its DSV in the leaf, or
    it cannot be compared with the others -- the CMRR 7 T figure is recorded precisely as the case that cannot."""
    for key, value in (("siemens_magnetom_prisma_3T", 0.02), ("philips_ingenia_3T", 0.022),
                       ("ge_signa_premier_3T", 0.05)):
        leaf = scc.get_limit(key, "homogeneity", "b0_homogeneity_vrms")
        assert leaf["value"] == value and leaf["unit"] == "ppm"
        assert "20 cm DSV" in leaf["context"] and "VRMS" in leaf["context"]
        assert leaf["confidence"] == "cited"
