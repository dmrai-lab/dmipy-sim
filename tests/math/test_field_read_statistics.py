"""Read calibration against closed-form linear and quadratic fields, including shortened intervals."""
import numpy as np
import pytest
from dmipy_sim.engine.accumulate import field_read_statistics


def test_linear_fields_have_exact_trapezoids_and_length_weighted_short_intervals():
    dt = .1
    time = np.arange(11) * dt
    reads = np.stack([time, 2 * time], axis=-1)[None]
    result = field_read_statistics(reads, step_s=dt, rounds_per_block=5, blocks_per_window=2, checks=(3, 5))
    assert np.max(np.abs(result["checks"][:, :, 0])) < 1e-15
    # Two blocks, each a three-round interval and a shortened two-round interval.
    expected = 2 * dt ** 2 * (3 ** 2 + (2 * 2 / 3) ** 2)
    np.testing.assert_allclose(result["checks"][0, 0, 1, 0], expected * np.array([1, 4]))
    np.testing.assert_allclose(result["qv_increments"][0, 0], 10 * dt ** 2 * np.array([1, 4]))


def test_quadratic_integral_errors_accumulate_over_a_window_before_squaring():
    dt = .02
    t = np.arange(21) * dt
    reads = np.broadcast_to(t[None, :, None] ** 2, (3, 21, 1))
    result = field_read_statistics(reads, step_s=dt, rounds_per_block=5, blocks_per_window=2, checks=(3,))
    # Composite trapezoid error for t^2 is sum(interval_length^3)/6.
    per_block = dt ** 3 * (3 ** 3 + 2 ** 3 - 5) / 6
    np.testing.assert_allclose(result["checks"][:, :, 0, 0, 0], 2 * per_block, rtol=1e-10)
    assert result["qv_field"].shape == (3, 2, 1)


def test_nonfinite_or_incomplete_reference_series_is_refused():
    with pytest.raises(ValueError, match="whole windows"):
        field_read_statistics(np.zeros((2, 10, 1)), step_s=.1, rounds_per_block=5, blocks_per_window=1, checks=(3,))
    with pytest.raises(ValueError, match="finite"):
        field_read_statistics(np.full((2, 6, 1), np.nan), step_s=.1, rounds_per_block=5, blocks_per_window=1, checks=(3,))
