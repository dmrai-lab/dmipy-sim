"""The realised acceptance of a membrane trial: :func:`~dmipy_sim.geometry._boundary.realised_acceptance` is the
threshold the kernel's trial decides with, and its distance from ``x / (1 + x)`` is within the derived bound.

The kernel is the oracle. For every row of the edge table, in each dtype of the function's argument, the threshold
read from :func:`~dmipy_sim.geometry._boundary.trial_granted`'s decisions over the words by bisection equals the
function's and the row's stated value; on 100 000 random finite float32 inputs (any magnitude, both signs) the
jitted kernel's decision at the words ``0, t - 1, t, t + 1, 2^32 - 1`` is ``w < t``. The realised probability is
``floor(p 2^32) / 2^32`` (``1`` where ``p >= 1``) of the kernel's float32 ``p`` wherever ``p`` is in ``[0, 1]``, and
never above it; a negative or NaN ``p`` is never granted, one above ``1`` always.

The bound, ``|r - p| < 2^-32 + eps p`` with ``eps = (3 u + u^2) / (1 - u)``, ``u = 2^-24`` for the float32 trial and
``eps = u + (1 + u) 2 v / (1 - v)``, ``v = 2^-53``, for the float64 form (its derivation is the docstring of
``tools/realised_acceptance_bound.py``, which checks it over every float32 ``x`` of ``[2^-34, 2^26]``; its record is
``docs/realised-acceptance-bound-*.json``), is checked here in exact rationals on a few thousand inputs: the corner
just below ``p = 2^-32``, the application range, the saturation at ``x = 2^24`` and random pairs whose product rounds.
Each statement is shown to fail when perturbed: the old relative ``2^-23`` on an input whose float32 ``p`` is
``1.46 x 2^-23`` off, the ceiling in place of the floor, and the bound without its absolute ``2^-32``.
"""
import json
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_sim.geometry._boundary import TRIAL_WORDS, realised_acceptance, transmit_probability, trial_granted

ROOT = Path(__file__).resolve().parents[2]
TWO32 = 2 ** 32
U = Fraction(1, 2 ** 24)
V = Fraction(1, 2 ** 53)
EPS = {np.float32: (3 * U + U * U) / (1 - U), np.float64: U + (1 + U) * 2 * V / (1 - V)}
F = np.float32
INF, NAN = float("inf"), float("nan")


def f32(v):
    """The float32 value of ``v`` as a Python float."""
    return float(F(v))


#: ``(name, kappa_over_D, d_perp, threshold with float32 arguments, threshold with float64 arguments)``; every
#: input is a float32 value. ``*`` in a name marks a row whose threshold differs between the two dtypes.
EDGES = [
    ("x = 0 (kappa_over_D 0)", 0.0, f32(1e-8), 0, 0),
    ("x = 0 (d_perp 0)", 1e3, 0.0, 0, 0),
    ("subnormal d_perp", 1.0, f32(1e-40), 0, 0),
    ("subnormal d_perp, large kappa_over_D *", f32(1e37), f32(1e-40), 0, 8572743),
    ("product underflows", f32(1e-20), f32(1e-20), 0, 0),
    ("x = 2^-34", 2.0 ** -35, 1.0, 0, 0),
    ("x = 2^-33", 2.0 ** -34, 1.0, 0, 0),
    ("x = 2^-32", 2.0 ** -33, 1.0, 1, 1),
    ("x = 0.1124 *", f32(0.0562), 1.0, 433975456, 433975488),
    ("x = 1", 0.5, 1.0, 2 ** 31, 2 ** 31),
    ("x = 2^24 - 1, the largest float32 p below 1", (2.0 ** 24 - 1) / 2, 1.0, 2 ** 32 - 256, 2 ** 32 - 256),
    ("x = 2^24, the first float32 p = 1 *", 2.0 ** 23, 1.0, 2 ** 32, 2 ** 32 - 256),
    ("x = 1e30, the clamp", f32(5e29), 1.0, 2 ** 32, 2 ** 32),
    ("product overflows", 2.0 ** 100, 2.0 ** 100, 2 ** 32, 2 ** 32),
    ("infinite kappa_over_D", INF, f32(1e-9), 2 ** 32, 2 ** 32),
    ("infinite d_perp", 1e3, INF, 2 ** 32, 2 ** 32),
    ("inf x 0", INF, 0.0, 0, 0),
    ("NaN kappa_over_D", NAN, 1.0, 0, 0),
    ("NaN d_perp", 1.0, NAN, 0, 0),
    ("2 kappa_over_D overflows, x = 4 *", 2.0 ** 127, 2.0 ** -126, 2 ** 32, 3435973888),
    ("one binade below, x = 4", 2.0 ** 126, 2.0 ** -125, 3435973888, 3435973888),
    ("x = -0.5", -0.25, 1.0, 0, 0),
    ("x = -2", -1.0, 1.0, 2 ** 32, 2 ** 32),
]

#: an input whose float32 ``p`` is ``1.4615 x 2^-23`` from the formula (relative): above the old ``2^-23``
OLD_EPS_COUNTEREXAMPLE = (551766.125, 6.9668641515530094e-12)
#: the largest float32 below ``2^-32``: ``p32`` is just below the first step, the realised probability ``0``
GRID_CORNER = float(np.nextafter(F(2.0 ** -32), F(0.0)))


@pytest.fixture(params=[np.float32, np.float64], ids=["float32", "float64"])
def dtype(request):
    """The dtype of the function's arguments; float64 under JAX's x64 mode."""
    if request.param is np.float64:
        request.getfixturevalue("x64")
    return request.param


def p_formula(k, d):
    """``x / (1 + x)``, ``x = 2 kappa_over_D d_perp``, in exact rationals of the inputs."""
    x = 2 * Fraction(float(k)) * Fraction(float(d))
    return x / (1 + x)


def kernel_threshold(k, d, dtype):
    """The number of words the kernel grants, read from its decisions alone by bisection (they are ``w < t``)."""
    p = transmit_probability(jnp.asarray(k, dtype), jnp.asarray(d, dtype))
    granted = jax.jit(lambda w: trial_granted(w, p))
    g = lambda w: bool(granted(jnp.uint32(w)))
    if not g(0):
        return 0
    if g(TWO32 - 1):
        return TWO32
    lo, hi = 0, TWO32 - 1
    while hi - lo > 1:
        m = (lo + hi) // 2
        lo, hi = (m, hi) if g(m) else (lo, m)
    return hi


# ── the edges, against the kernel ───────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("row", EDGES, ids=[r[0] for r in EDGES])
def test_every_edge_is_the_kernels_threshold(row, dtype):
    name, k, d, t32, t64 = row
    want = t32 if dtype is np.float32 else t64
    assert (t32 != t64) == name.endswith("*")
    t = realised_acceptance(k, d, dtype)
    assert isinstance(t, int)
    assert t == kernel_threshold(k, d, dtype) == want


def test_the_threshold_saturates_at_two_to_the_24():
    """``p32 = 1 - 2^-24`` (the largest float32 below 1) at ``x = 2^24 - 1`` gives ``2^32 - 256``; at ``x = 2^24`` the
    sum ``1 + x`` rounds to ``x`` and ``p32 = 1``: every word is granted. Over every float32 ``x`` of ``[2^23, 2^25]``
    the first with ``p32 = 1`` is ``2^24``; up to ``2^25`` the sum rounds to the even neighbour and ``p32`` alternates
    between ``1`` and ``1 - 2^-23``; from ``2^25`` on it is ``1``."""
    x = np.arange(F(2.0 ** 23).view(np.uint32), F(2.0 ** 25).view(np.uint32) + 1, dtype=np.uint32).view(F)
    p = np.asarray(transmit_probability(jnp.asarray(x / F(2.0)), jnp.float32(1.0)))
    first = int(np.argmax(p >= 1))
    assert x[first] == 2.0 ** 24 and p[first - 1] == np.nextafter(F(1.0), F(0.0))
    assert realised_acceptance(x[first - 1] / 2, 1.0) == TWO32 - 256
    assert realised_acceptance(x[first] / 2, 1.0) == TRIAL_WORDS
    assert realised_acceptance(F(2.0 ** 24 + 2) / 2, 1.0) == TWO32 - 512                  # 1 + x rounds to x + 2
    assert np.all(p[x >= 2.0 ** 25] == 1) and not np.all(p[x >= 2.0 ** 24] == 1)


def test_float64_needs_x64_and_another_dtype_is_refused():
    with pytest.raises(ValueError, match="x64"):
        realised_acceptance(1.0, 1.0, np.float64)
    with pytest.raises(ValueError, match="dtype"):
        realised_acceptance(1.0, 1.0, np.float16)


# ── random inputs: the kernel's decision is w < t, and t is floor(p 2^32) ─────────────────────────────────────────────
def _random_inputs(n=100_000, seed=20261117):
    """``n`` finite float32 pairs drawn uniformly over the bit patterns: any magnitude, both signs."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(2):
        b = rng.integers(0, TWO32, size=3 * n, dtype=np.uint64).astype(np.uint32).view(F)
        out.append(b[np.isfinite(b)][:n])
    return out


def test_the_kernel_grants_exactly_the_words_below_the_threshold(dtype):
    k, d = _random_inputs()
    t = realised_acceptance(k, d, dtype)
    assert t.dtype == np.uint64 and t.shape == k.shape
    kernel = jax.jit(lambda w, k, d: trial_granted(w, transmit_probability(k, d)))
    kj, dj = jnp.asarray(k, dtype), jnp.asarray(d, dtype)
    ti = t.astype(np.int64)
    for w in (np.zeros_like(ti), ti - 1, ti, ti + 1, np.full_like(ti, TWO32 - 1)):
        ok = (w >= 0) & (w < TWO32)
        got = np.asarray(kernel(jnp.asarray(np.where(ok, w, 0).astype(np.uint32)), kj, dj))
        assert np.array_equal(got[ok], (w < ti)[ok])


def test_the_realised_probability_is_the_floor_of_p_and_never_above_it(dtype):
    """Where the kernel's float32 ``p`` is in ``[0, 1]``: ``t = floor(p 2^32)`` (``2^32`` at ``p = 1``) and
    ``t / 2^32 <= p``, in exact arithmetic (a float32 times ``2^32`` is exact in float64). A NaN or negative ``p``
    gives ``t = 0``, a ``p`` above 1 gives ``2^32``."""
    k, d = _random_inputs()
    t = realised_acceptance(k, d, dtype)
    p = np.asarray(transmit_probability(jnp.asarray(k, dtype), jnp.asarray(d, dtype))).astype(F).astype(np.float64)
    inside = (p >= 0) & (p < 1)
    assert inside.sum() > 1000 and (p == 1).sum() > 1000
    assert np.array_equal(t[inside], np.floor(p[inside] * TWO32).astype(np.uint64))
    assert np.all(t[inside].astype(np.float64) <= p[inside] * TWO32)
    assert np.all(t[p == 1] == TWO32)
    assert np.all(t[np.isnan(p) | (p < 0)] == 0) and np.all(t[p > 1] == TWO32)
    assert (p < 0).sum() > 0 and (p > 1).sum() > 0
    if dtype is np.float32:                                                          # inf x 0 after an overflow
        assert np.isnan(p).sum() > 0
    for j in np.flatnonzero(inside)[:200]:                                           # the same, in Fraction
        r = Fraction(int(t[j]), TWO32)
        assert r == Fraction(int(np.floor(Fraction(float(p[j])) * TWO32)), TWO32) and r <= Fraction(float(p[j]))


# ── the bound, in exact rationals, on a small exact subset ──────────────────────────────────────────────────────────
def _x_subset():
    """A few thousand float32 ``x``: every float32 within 256 of the grid corner, 2000 drawn from the application
    range ``[2^-34, 0.1124]`` over its bit patterns, and every float32 within 256 of ``2^24``."""
    rng = np.random.default_rng(7)
    bits = lambda v: int(F(v).view(np.uint32))
    around = lambda v: np.arange(bits(v) - 256, bits(v) + 257, dtype=np.uint32)
    app = rng.integers(bits(2.0 ** -34), bits(0.1124) + 1, size=2000).astype(np.uint32)
    return np.concatenate([around(2.0 ** -32), app, around(2.0 ** 24)]).view(F)


def _pairs(n=3000, seed=11):
    """``n`` float32 pairs whose product rounds: ``x`` log-uniform in ``[1e-9, 31]``, ``d_perp`` in ``[1e-10, 1e-5]``."""
    rng = np.random.default_rng(seed)
    x = np.exp(rng.uniform(np.log(1e-9), np.log(31.0), n))
    d = np.exp(rng.uniform(np.log(1e-10), np.log(1e-5), n)).astype(F)
    return (x / (2.0 * d.astype(np.float64))).astype(F), d


def _ratios(k, d, dtype, eps=None):
    t = realised_acceptance(k, d, dtype)
    eps = EPS[dtype] if eps is None else eps
    out = []
    for kj, dj, tj in zip(k, d, t):
        p = p_formula(kj, dj)
        out.append(abs(Fraction(int(tj), TWO32) - p) / (Fraction(1, TWO32) + eps * p))
    return out


def test_the_bound_holds_on_the_x_subset(dtype):
    x = _x_subset()
    k = (x / F(2.0)).astype(F)
    r = _ratios(k, np.ones_like(k), dtype)
    assert max(r) < 1
    assert max(r) > 1 - Fraction(1, 2 ** 20)                                         # the corner is all but tight


def test_the_bound_holds_on_pairs_whose_product_rounds(dtype):
    k, d = _pairs()
    x_hat = (F(2.0) * k * d).astype(np.float64)
    x = 2.0 * k.astype(np.float64) * d.astype(np.float64)                            # exact: 48 bits
    assert np.max(np.abs(x_hat - x) / x) <= float(U)                                 # one rounding
    assert np.any(x_hat != x)
    assert max(_ratios(k, d, dtype)) <= 1


# ── each statement fails when perturbed ─────────────────────────────────────────────────────────────────────────────
def test_the_old_relative_two_to_the_minus_23_fails_and_eps_holds():
    k, d = OLD_EPS_COUNTEREXAMPLE
    assert f32(k) == k and f32(d) == d
    p32 = Fraction(float(transmit_probability(jnp.float32(k), jnp.float32(d))))
    p = p_formula(k, d)
    rel = abs(p32 - p) / p
    assert rel > Fraction(1, 2 ** 23)                                                # the old statement fails
    assert rel <= EPS[np.float32]
    assert f"{float(rel / Fraction(1, 2 ** 23)):.4f}" == "1.4615"


def test_a_third_of_eps_fails_on_the_x_subset(dtype):
    """A wrong constant, ``eps / 3`` (``u`` for the float32 trial), is violated on the same subset that holds the
    stated one: the check tells the constant apart."""
    x = _x_subset()
    k = (x / F(2.0)).astype(F)
    assert max(_ratios(k, np.ones_like(k), dtype, EPS[dtype] / 3)) > 1


def test_the_ceiling_in_place_of_the_floor_is_caught():
    """At ``x = 1e-4`` (a float32 ``p`` below ``2^-9``, whose ``p 2^32`` is not an integer; from ``2^-9`` on it is, and
    the realised probability is the float32 ``p`` itself)."""
    k, d = f32(5e-5), 1.0
    t = realised_acceptance(k, d)
    p32 = Fraction(float(transmit_probability(jnp.float32(k), jnp.float32(d))))
    ceil = -(-p32 * TWO32 // 1)
    assert ceil != t and Fraction(int(ceil), TWO32) > p32                            # ceiling: not t, and above p
    assert Fraction(t, TWO32) <= p32


def test_the_bound_without_its_absolute_term_fails_at_the_grid_corner():
    k = GRID_CORNER / 2
    assert realised_acceptance(k, 1.0) == 0
    p = p_formula(k, 1.0)
    dist = p                                                                         # realised 0
    assert dist > EPS[np.float32] * p                                                # eps p alone fails
    assert dist < Fraction(1, TWO32) + EPS[np.float32] * p


def test_outside_the_domain_the_float32_trial_is_off_the_formula():
    """``2 kappa_over_D`` overflowing and a subnormal operand flushed: documented, kernel-exact, not held to the
    bound (float32 arguments). With float64 arguments the same inputs are the kernel's too (the edge table)."""
    for k, d in ((2.0 ** 127, 2.0 ** -126), (f32(1e37), f32(1e-40))):
        dist = abs(Fraction(realised_acceptance(k, d), TWO32) - p_formula(k, d))
        assert dist > Fraction(1, TWO32)


# ── the record of the exhaustive sweep ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["application", "full"])
def test_the_recorded_sweep_covers_every_float32_and_holds(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location("realised_acceptance_bound",
                                                  ROOT / "tools" / "realised_acceptance_bound.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    rec = json.loads((ROOT / "docs" / f"realised-acceptance-bound-{name}.json").read_text())
    assert rec["range"] == name
    assert [s["form"] for s in rec["sweeps"]] == ["float32", "float64", "variant"]
    assert {s["form"]: Fraction(s["eps"]) for s in rec["sweeps"]}["float32"] == EPS[np.float32]
    assert {s["form"]: Fraction(s["eps"]) for s in rec["sweeps"]}["float64"] == EPS[np.float64]
    assert tool.check(rec) == []
    other = "full" if name == "application" else "application"
    assert tool.check(dict(rec, range=other)), "a record relabelled with the other range is refused"
