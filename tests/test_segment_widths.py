"""The widths of a time-segment layout's fields (#770): a segment vector's component, a tier channel and a background
component each take the fewest bits whose largest code reaches its largest value in steps of at most its derived step,
a field that rounds to zero on every row takes none, and a walker's fields are placed first-fit into 32-bit words; the
words round-trip on the host and decode the same in the kernels; the step of a partition's segments holds the worst
encoding the envelope admits at the floor's budget."""
from __future__ import annotations

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_replay.constants import GAMMA
from dmipy_sim.replay.compression import container_eps
from dmipy_sim.replay.time_segments import FIELD_BITS_MAX, field_code, pack_words, rounding_eps, word_layout
from dmipy_replay.replay.time_segments import Partition, _decode, unpack_words

D_WATER = 3e-9


def test_the_width_is_the_fewest_bits_within_the_step():
    """``bits = ceil(log2(a / step + 1)) + 1``: the largest code ``2^(bits-1) - 1`` reaches ``a`` in steps no larger
    than ``step``; at least 3 bits; 0 bits within half a step of zero or at an infinite step; refused past 24."""
    a = np.array([1.0, 3.0, 4.0, 1000.0, 0.4, 0.6, 7.0])
    bits, scale = field_code(a, 1.0)
    assert list(bits) == [3, 3, 4, 11, 0, 3, 4]
    live = bits > 0
    assert np.all(scale[live] <= 1.0) and np.all(scale[live] * (2.0 ** (bits[live] - 1) - 1) == pytest.approx(a[live]))
    assert list(field_code([5.0], np.inf)[0]) == [0]
    with pytest.raises(ValueError, match="bits"):
        field_code([2.0 ** FIELD_BITS_MAX], 1.0)


def test_the_free_water_example():
    """Free water at the floor target 1e-2 (``eps = sqrt(0.1 x 0.01) = 0.032`` rad), a PGSE of 10 ms lobes 30 ms apart
    at b 3000 s/mm^2: the reach of each lobe is ``sqrt(b_max / b_unit)``, the step ``sqrt(12 eps^2 / 2) / reach``, and a
    lobe's vector -- ``gamma`` times its duration times the walker's mean displacement over it, about six rms at 1e9
    walkers -- takes 10 or 11 bits per component: three components of both lobes in two 32-bit words."""
    eps = rounding_eps(1e-2)
    assert eps == pytest.approx(np.sqrt(0.1 * 0.01)) == pytest.approx(container_eps(1e-2))
    se = d.pgse([[0, 0, 1]], 10e-3, 30e-3, gradient_strengths=0.05, n_t=501, slew_rate=np.inf)
    p = Partition.of("pgse", se)
    reach = p.reach(1.0, 3e9)
    b_unit = (GAMMA * 10e-3) ** 2 * (30e-3 - 10e-3 / 3)
    np.testing.assert_allclose(reach, np.sqrt(3e9 / b_unit), rtol=1e-6)
    step = np.sqrt(12 * eps ** 2 / 2) / reach[0]
    disp = 6 * np.sqrt(2 * D_WATER * 40e-3)                              # the later lobe's mean displacement, six rms
    a = GAMMA * 10e-3 * disp * np.ones(3)
    bits, _ = field_code(a, step)
    assert set(bits) <= {10, 11}
    layout, n_words = word_layout(np.concatenate([bits, bits]))
    assert n_words == 2 and all(b == bits[0] for _, _, b in layout)


@pytest.mark.parametrize("bits", [[3, 7, 10], [10, 10, 10], [11, 11, 10], [11, 11, 11], [16, 3, 16], [24, 8, 5, 0, 9, 9, 9]])
def test_the_words_round_trip(bits):
    """Every code of every layout comes back exactly, a value within half a step, the extremes and zero included,
    through the host decoder and the JAX one alike; a field never crosses a word; a 0-bit field decodes to zero."""
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    bits = np.asarray(bits)
    step = 1e-8
    a = np.where(bits > 0, step * 0.9 * (2.0 ** (bits - 1) - 1), 0.0)
    got, scale = field_code(a, step)
    assert list(got) == list(bits)
    layout, n_words = word_layout(got)
    assert n_words >= -(-sum(int(b) for b in bits) // 32)
    for w, off, b in layout:
        assert off + b <= 32
    m = rng.uniform(-1, 1, (64, 128, len(bits))) * a
    m[0, 0] = a; m[0, 1] = -a; m[0, 2] = 0.0
    words = pack_words(m, scale, layout, n_words)
    assert words.dtype == np.uint32 and words.shape == m.shape[:2] + (n_words,)
    back = unpack_words(words, scale, layout)
    assert back.dtype == np.float32
    assert np.all(np.abs(back - m) <= 0.5 * scale + 2.0 ** -23 * np.abs(m) + 1e-30)
    dev = _decode(jnp.asarray(words.view(np.int32)), jnp.asarray(scale, jnp.float32), layout)
    host = np.stack([np.zeros(m.shape[:2], np.float32) if v is None else np.asarray(v) for v in dev], -1)
    np.testing.assert_array_equal(host, back)


def test_first_fit_fills_a_word_before_opening_another():
    """Short fields share a word, a field that does not fit the open words opens the next; the order is the fields'."""
    layout, n = word_layout([10, 10, 10, 2, 9, 9, 9, 5])
    assert n == 2
    assert layout[:4] == ((0, 0, 10), (0, 10, 10), (0, 20, 10), (0, 30, 2))
    assert layout[4:] == ((1, 0, 9), (1, 9, 9), (1, 18, 9), (1, 27, 5))
