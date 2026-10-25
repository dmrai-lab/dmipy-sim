"""Numerical bias and Monte Carlo uncertainty must certify the same requested answer."""
import pytest
from dmipy_sim.math.accuracy import ErrorBound, assess_accuracy
from dmipy_sim.math.step_law import assess_step_certificate

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
    terms = {"trajectory":bound(.001, metric="relative_diffusion_spectrum_rms"),
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


def test_a_noisy_step_estimate_cannot_pass_by_making_the_acceptance_band_wider():
    cert = dict(eps_measured=.0008, eps_se=.0002, floor=.0005, eps_declared=.001, resolves=True)
    # The former gate accepted .0008 <= .001 + 3*.0002.
    result = assess_step_certificate(cert)
    assert result["status"] == "over_budget" and result["upper"] == pytest.approx(.0014)


def test_unresolved_step_measurement_is_not_a_measured_pass():
    cert = dict(eps_measured=0, eps_se=.0001, floor=.002, eps_declared=.001, resolves=False)
    assert assess_step_certificate(cert)["status"] == "unresolved"
    cert.update(floor=.0002, resolves=True)
    assert assess_step_certificate(cert)["status"] == "passed"
