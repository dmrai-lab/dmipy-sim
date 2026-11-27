"""The master key uses both words of a walk seed in either JAX precision mode."""

import json
import os
import subprocess
import sys

import jax
import numpy as np
import pytest

from dmipy_sim.engine.physics import seed_keys, seed_walkers
from dmipy_sim.geometry.base import Box1D


def _check_small_seeds():
    for seed in (0, 1, 2**31 - 1, 2**31, 2**32 - 1):
        np.testing.assert_array_equal(seed_keys(seed)[0], jax.random.PRNGKey(seed), strict=True)


def _master_words():
    seeds = (2**32, 2**32 + 5, 0x6245DFE9B349429E)
    return [np.asarray(seed_keys(seed)[0]).tolist() for seed in seeds]


def _with_x64(code):
    return subprocess.check_output(
        [sys.executable, "-B", "-c", "import jax; assert jax.config.x64_enabled; " + code],
        env={**os.environ, "JAX_ENABLE_X64": "1"},
        text=True,
    )


def test_small_seeds_match_jax_in_both_modes():
    _check_small_seeds()
    _with_x64("from tests.test_seed_keys import _check_small_seeds; _check_small_seeds()")


def test_large_seeds_keep_both_words_in_both_modes():
    words_on = json.loads(_with_x64(
        "import json; from tests.test_seed_keys import _master_words; print(json.dumps(_master_words()))"
    ))
    assert _master_words() == words_on == [[1, 0], [1, 5], [0x6245DFE9, 0xB349429E]]


def test_maximum_unsigned_seed_is_accepted():
    key = seed_keys(np.uint64(2**64 - 1))[0]
    np.testing.assert_array_equal(key, np.array([0xFFFFFFFF, 0xFFFFFFFF], dtype=np.uint32), strict=True)


def test_seeds_with_the_same_low_word_place_different_first_walkers():
    first = seed_walkers(Box1D(10e-6), 1, 5)[1][0]
    other = seed_walkers(Box1D(10e-6), 1, 2**32 + 5)[1][0]
    assert not np.array_equal(first, other)


@pytest.mark.parametrize("seed", [
    pytest.param(-1, id="negative"),
    pytest.param(2**64, id="too-large"),
    pytest.param(1.5, id="float"),
    pytest.param("1", id="str"),
    pytest.param(True, id="bool"),
    pytest.param(np.bool_(True), id="numpy-bool"),
    pytest.param(np.timedelta64(1, "ns"), id="numpy-timedelta"),
])
def test_invalid_seeds_are_refused_by_name(seed):
    with pytest.raises(ValueError, match=r"\bseed\b.*integer.*2\^64"):
        seed_keys(seed)
