"""Numerical bias and Monte Carlo uncertainty must certify the same requested answer."""
import pytest
from dmipy_sim.math.accuracy import ErrorBound, assess_accuracy
from dmipy_sim.math.step_law import assess_step_certificate, STEP_METRIC

METRIC = "max_abs_complex_signal_error"


def bound(value, confidence=1.0, metric=METRIC):
    return ErrorBound(value, metric, confidence, "independent refinement or analytic bound")


def assess(numerical, statistical=.001, **kw):
    return assess_accuracy(target=.01, metric=METRIC, confidence=.99,
                           statistical=bound(statistical), numerical=numerical,
                           required=("trajectory", "representation"), **kw)


def test_individually_small_errors_can_exceed_the_joint_answer_budget():
    report = assess({"trajectory":bound(.006), "representation":bound(.004)})
    assert report["status"] == "over_budget"
    assert report["total_upper"] == pytest.approx(.011)


def test_reducing_monte_carlo_uncertainty_does_not_reduce_numerical_bias():
    terms = {"trajectory":bound(.008), "representation":bound(.003)}
    for statistical in (.01, .001, .000001):
        report = assess(terms, statistical)
        assert report["status"] == "over_budget"
        assert report["numerical_upper"] == pytest.approx(.011)


@pytest.mark.parametrize("terms", [
    {"representation":bound(.001)},
    {"trajectory":bound(None), "representation":bound(.001)},
])
def test_missing_or_unresolved_evidence_never_certifies_an_answer(terms):
    report = assess(terms)
    assert report["status"] == "unresolved" and "trajectory" in report["unresolved"]
    assert report["total_upper"] is None


def test_a_relative_diffusion_spectrum_check_cannot_certify_absolute_signal_error():
    terms = {"trajectory":bound(.001, metric="relative_diffusion_spectrum_sup"),
             "representation":bound(.001)}
    with pytest.raises(ValueError, match="requested metric"):
        assess(terms)


def test_joint_confidence_does_not_assume_independent_error_sources():
    report = assess({"trajectory":bound(.001, .995), "representation":bound(.001, .99)})
    assert report["joint_confidence"] == pytest.approx(.985)
    assert report["status"] == "unresolved"


def test_complete_bounded_evidence_passes_at_its_stated_metric():
    report = assess({"trajectory":bound(.001), "representation":bound(.002)})
    assert report["status"] == "passed" and report["total_upper"] == pytest.approx(.004)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_error_bounds_are_refused(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        bound(value)


def _cert(**c):
    """A step certificate of one octave carrying the band's numbers."""
    octave = dict(f_lo_hz=5.0, f_hi_hz=1300.0, n_bins=260, eps=c["eps_measured"], eps_se=c["eps_se"], floor=c["floor"])
    return dict(c, metric=STEP_METRIC, octaves=[octave])


def test_a_noisy_step_estimate_cannot_pass_by_making_the_acceptance_band_wider():
    cert = _cert(eps_measured=.0008, eps_se=.0002, floor=.0005, eps_declared=.001, resolves=True)
    # The former gate accepted .0008 <= .001 + 3*.0002.
    result = assess_step_certificate(cert)
    assert result["status"] == "over_budget" and result["upper"] == pytest.approx(.0014)


def test_unresolved_step_measurement_is_not_a_measured_pass():
    cert = _cert(eps_measured=0, eps_se=.0001, floor=.002, eps_declared=.001, resolves=False)
    assert assess_step_certificate(cert)["status"] == "unresolved"
    cert = _cert(eps_measured=0, eps_se=.0001, floor=.0002, eps_declared=.001, resolves=True)
    assert assess_step_certificate(cert)["status"] == "passed"


def test_the_step_check_reads_the_worst_octave_and_refuses_a_band_rms():
    """One octave over the tolerance fails the check however quiet the rest of the band; one unresolved octave leaves
    the check unresolved; a certificate without octaves (the band's rms) is refused, since it does not read the sup."""
    good = dict(f_lo_hz=40.0, f_hi_hz=1300.0, n_bins=250, eps=.0002, eps_se=.00002, floor=.0001)
    cert = dict(eps_measured=.0009, eps_se=.0001, floor=.0003, eps_declared=.001, resolves=True, metric=STEP_METRIC,
                octaves=[dict(f_lo_hz=5.0, f_hi_hz=5.0, n_bins=1, eps=.0009, eps_se=.0001, floor=.0003), good])
    result = assess_step_certificate(cert)
    assert result["status"] == "over_budget" and result["upper"] == pytest.approx(.0012)
    cert["octaves"][0].update(floor=.004)
    assert assess_step_certificate(cert)["status"] == "unresolved"
    with pytest.raises(ValueError, match="octaves"):
        assess_step_certificate(dict(eps_measured=.0001, eps_se=.00001, floor=.0001, eps_declared=.001, resolves=True))
