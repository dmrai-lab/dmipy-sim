"""The scanner catalogue is one source with typed views (#173 piece 1).

Every class of the band-limit certificate and every Pulseq preset is the cited JSON's number for that machine;
one scanner resolves from any of its names; a slew regime is a choice, not a second catalogue; and what the
catalogue does not know is ``None`` and listed, never a number standing in for one.
"""
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
    assert {"certificate_bruker_bga_s", "certificate_micro_insert", "certificate_extreme_insert",
            "pulseq_clinical_typical", "pulseq_preclinical_bruker", "pulseq_example_system"} <= unverified
    for k in ("bruker_bga_s", "micro_insert", "extreme_insert", "clinical_typical", "preclinical_bruker"):
        assert ScannerLimits.of(k).kind == "envelope", k


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
