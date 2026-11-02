"""The size of a Monte Carlo test is derived: ``se_tol`` is the tolerance of ``z_family`` standard errors and
``n_resolving`` the smallest walker count whose tolerance is half of the deviation the test must catch."""
import pytest

from tests.conftest import n_resolving, se_tol, z_family


def test_the_family_quantile_is_the_two_sided_normal_one():
    assert z_family() == pytest.approx(3.2905, abs=1e-3)              # 1e-3 shared by one assertion
    assert z_family(2) == pytest.approx(3.4808, abs=1e-3)             # and by two: 5e-4 each
    assert z_family(3, alpha=0.05) == pytest.approx(2.3940, abs=1e-3)


@pytest.mark.parametrize("comparisons", [1, 3, 8])
@pytest.mark.parametrize("runs", [1, 2])
def test_the_resolving_size_is_the_smallest_whose_tolerance_is_half_the_deviation(comparisons, runs):
    n = n_resolving(0.05, 0.7, comparisons=comparisons, runs=runs)
    assert se_tol(n, 0.7, comparisons=comparisons, runs=runs) <= 0.025 < se_tol(n - 1, 0.7, comparisons=comparisons, runs=runs)
