"""The selection of the recorded values by CPU family (``tests/recorded_bits.py``): a recorded canary selects its
family in and out of the recorded-bits job; a canary of no recorded family fails in the job and skips elsewhere, both
naming it; a build other than the recorded one is an error in the job and a skip elsewhere; every family records every
array of every reference walk."""
import pytest

from tests.recorded_bits import FAMILIES, RECORDED_WITH, check_build, family_of

FORGED = "0" * 64


@pytest.mark.parametrize("in_job", [True, False], ids=["in_job", "elsewhere"])
def test_a_recorded_canary_selects_its_family(in_job):
    for canary_hash, values in FAMILIES.items():
        assert family_of(canary_hash, in_job=in_job) is values


def test_an_unrecorded_canary_fails_in_the_recorded_bits_job_naming_it():
    with pytest.raises(pytest.fail.Exception, match=f"unrecorded CPU family: this host's canary {FORGED}"):
        family_of(FORGED, in_job=True)


def test_an_unrecorded_canary_skips_elsewhere_naming_it():
    with pytest.raises(pytest.skip.Exception, match=f"unrecorded CPU family: this host's canary {FORGED}"):
        family_of(FORGED, in_job=False)


def test_another_build_is_an_error_in_the_job_and_a_skip_elsewhere():
    other = ("0.0.0", "0.0.0")
    check_build(RECORDED_WITH, in_job=True)
    check_build(RECORDED_WITH, in_job=False)
    with pytest.raises(AssertionError, match="the recorded-bits job runs jax / jaxlib"):
        check_build(other, in_job=True)
    with pytest.raises(pytest.skip.Exception, match="recorded bits are bits of jax / jaxlib"):
        check_build(other, in_job=False)


def test_every_family_records_every_array_of_every_reference_walk():
    from tests.geometry.test_mixed_myelinated_plain_cell import FIELD_ARRAYS, NO_FIELD_ARRAYS, RECORDED_WALKS
    for canary_hash, values in FAMILIES.items():
        assert len(canary_hash) == 64 and tuple(values) == RECORDED_WALKS
        assert tuple(values["field"]) == FIELD_ARRAYS and tuple(values["no_field"]) == NO_FIELD_ARRAYS
        assert len(values["forward"]) == 5
