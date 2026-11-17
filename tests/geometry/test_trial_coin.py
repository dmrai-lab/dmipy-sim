"""The coin of a plain wall's transmission trial: a 32-bit word compared in integers with ``floor(p 2^32)``.

Exact, by enumeration: for each float32 probability ``p`` the words ``w`` of ``[0, 2^21)`` that
:func:`~dmipy_sim.geometry._boundary.trial_granted` grants are counted and equal ``min(t, 2^21)``, ``t = floor(p 2^32)``
computed in exact rationals from the float32's value; the decision at the words around ``t`` is ``w < t``; and the
realised probability ``t / 2^32`` lies in ``(p - 2^-32, p]``. The control is the float32 uniform the coin replaces,
``(w >> 9) 2^-23 < p``, which realises a probability above ``p`` (at ``p = 1e-4`` by ``1.66e-8``): the two schemes are
told apart on the same enumeration.

One sampling check beside the exactness: a fixed key, 4 000 walkers' words as the walk draws them, and the granted
fraction at three probabilities within 4 standard errors of ``t / 2^32`` (a false rejection about 6.3e-5 per
comparison, two-sided normal).
"""
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_sim.geometry._boundary import trial_granted

TWO32 = 2 ** 32
SPAN = 2 ** 21                                                   # the enumerated words: [0, 2^21)
F32 = np.float32

#: the probabilities, as float32 values
PS = {
    "zero": F32(0.0),
    "one": F32(1.0),
    "largest_below_one": np.nextafter(F32(1.0), F32(0.0)),
    "half": F32(0.5),
    "1e-9": F32(1e-9),
    "1e-6": F32(1e-6),
    "1e-4": F32(1e-4),
    "1e-3": F32(1e-3),
    "0.0123": F32(0.0123),
    "one_third": F32(1.0 / 3.0),
    "0.9": F32(0.9),
    "one_minus_2^-24": F32(1.0 - 2.0 ** -24),
    "smallest_normal": np.finfo(np.float32).tiny,
    "p_2^32_integer": F32(12345 * 2.0 ** -32),
}


def _t(p):
    """``floor(p 2^32)`` in exact rationals from the float32's value."""
    return int(np.floor(Fraction(float(p)) * TWO32))


def _granted(words, p):
    return np.asarray(trial_granted(jnp.asarray(words, jnp.uint32), jnp.float32(p)))


@pytest.mark.parametrize("name", list(PS))
def test_the_coin_grants_exactly_the_words_below_floor_p_2_32(name):
    p = PS[name]
    assert Fraction(float(p)) == Fraction(float(F32(p)))                           # a float32 value
    t = _t(p)
    if name == "p_2^32_integer":
        assert Fraction(float(p)) * TWO32 == t                                     # p 2^32 is an integer
    words = np.arange(SPAN, dtype=np.uint32)
    n = int(_granted(words, p).sum())
    assert n == min(t, SPAN)
    probe = [w for w in (0, t - 1, t, t + 1, TWO32 - 1) if 0 <= w < TWO32]
    got = _granted(np.asarray(probe, np.uint32), p)
    want = [True] * len(probe) if p >= 1 else [False] * len(probe) if p <= 0 else [w < t for w in probe]
    assert list(got) == want, (probe, list(got))
    realised = Fraction(t, TWO32)
    assert Fraction(float(p)) - Fraction(1, TWO32) < realised <= Fraction(float(p))


def test_the_float32_uniform_realises_above_p_and_the_word_below():
    """The control: at ``p = 1e-4`` the uniform ``(w >> 9) 2^-23 < p`` grants ``ceil(p 2^23) 2^9`` words of the 2^32,
    a realised ``1.000165939e-04`` above ``p32 = 9.999999747e-05``; the word grants ``floor(p 2^32)``, below it. Both
    counted on the same enumeration (every granted word of either rule lies in ``[0, 2^21)``)."""
    p = PS["1e-4"]
    words = np.arange(SPAN, dtype=np.uint32)
    old = int(((words >> 9).astype(np.float64) * 2.0 ** -23 < float(p)).sum())
    new = int(_granted(words, p).sum())
    assert old == int(np.ceil(Fraction(float(p)) * 2 ** 23)) * 2 ** 9 < SPAN
    fp = Fraction(float(p))
    assert Fraction(old, TWO32) > fp, "the float32 uniform realises above p"
    assert f"{float(Fraction(old, TWO32)):.9e}" == "1.000165939e-04"
    assert Fraction(new, TWO32) <= fp < Fraction(new + 1, TWO32), "the word realises floor(p 2^32) / 2^32"


@pytest.mark.parametrize("p", [0.3, 1e-3, 1.0 - 1e-3])
def test_the_coin_samples_its_probability(p):
    """4 000 walkers' trial words (32 each, the walk's draw ``bits(key, (budget,), uint32)``) from the fixed key
    ``PRNGKey(20261117)``: the granted fraction lies within 4 standard errors of ``t / 2^32``."""
    keys = jax.random.split(jax.random.PRNGKey(20261117), 4000)
    words = jax.vmap(lambda k: jax.random.bits(k, (32,), dtype=jnp.uint32))(keys)
    p32 = F32(p)
    q = _t(p32) / TWO32
    n = int(words.size)
    frac = float(_granted(np.asarray(words).ravel(), p32).mean())
    se = np.sqrt(q * (1.0 - q) / n)
    assert abs(frac - q) <= 4.0 * se, (frac, q, se)
