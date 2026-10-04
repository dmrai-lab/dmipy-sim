"""The closed form's grouping of measurements by the waveform they play (``replay._group_waveforms``) is the greedy
grouping by the exact test, found through a window of sorted projections (dmrai-lab/dmipy-sim#605): the same
``(group, first)`` as testing every ungrouped row against every row."""
import numpy as np
import pytest

from dmipy_sim.replay.replay import _group_waveforms


def _every_row(s, rtol):
    """The grouping written out: each ungrouped row opens a group and takes every ungrouped row within ``rtol`` of
    the largest amplitude of it, everywhere."""
    s = np.asarray(s, np.float64)
    tol = rtol * (float(np.abs(s).max()) or 1.0)
    group = np.full(s.shape[0], -1, np.int64)
    first = []
    for i in range(s.shape[0]):
        if group[i] < 0:
            group[(group < 0) & (np.abs(s - s[i]).max(axis=1) <= tol)] = len(first)
            first.append(i)
    return group, np.asarray(first, np.int64)


def _cases(rng):
    t = np.linspace(0.0, 1.0, 400)
    shape = np.where(t < 0.3, 1.0, np.where(t < 0.5, 0.0, np.where(t < 0.8, -1.0, 0.0)))   # zero-mean, refocused
    yield "one shape, many amplitudes", shape[None, :] * rng.uniform(0.1, 1.0, 60)[:, None]
    yield "repeated rows", np.repeat(rng.normal(size=(7, 400)), 9, axis=0)[rng.permutation(63)]
    near = shape[None, :] * (1.0 + rng.choice([0.0, 4e-6, 9.9e-6, 1.01e-5, 3e-5], size=50))[:, None]
    yield "rows a tolerance apart, chained", near
    yield "every row its own", rng.normal(size=(40, 400))
    yield "zeros and a shell", np.vstack([np.zeros((5, 400)), shape[None, :] * np.linspace(0.2, 1.0, 30)[:, None]])
    yield "one row", shape[None, :]


@pytest.mark.parametrize("rtol", [1e-5, 1e-3])
def test_the_windowed_grouping_is_the_grouping_by_every_row(rtol):
    rng = np.random.default_rng(3)
    for name, s in _cases(rng):
        g, f = _group_waveforms(s, rtol=rtol)
        g_ref, f_ref = _every_row(s, rtol)
        np.testing.assert_array_equal(g, g_ref, err_msg=name)
        np.testing.assert_array_equal(f, f_ref, err_msg=name)
