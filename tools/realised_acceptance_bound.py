"""The distance of a membrane trial's realised acceptance from ``x / (1 + x)``, checked over every float32 ``x``.

    python tools/realised_acceptance_bound.py --range application --out FILE.json      # x in [2^-34, 0.1124]
    python tools/realised_acceptance_bound.py --range full --out FILE.json             # x in [2^-34, 2^26]
    python tools/realised_acceptance_bound.py --check FILE.json                        # the record's verdicts

The trial's realised probability is ``r = t / 2^32``, ``t`` the threshold
:func:`~dmipy_sim.geometry._boundary.realised_acceptance` returns, and the formula's probability is ``p = x / (1 + x)``,
``x = 2 kappa_over_D d_perp``, in exact arithmetic of the inputs. The stated bound is

    ``|r - p| < 2^-32 + eps p``

with, per way of forming the probability,

* ``float32`` (the walk's trial, ``realised_acceptance(..., dtype=numpy.float32)``): ``eps = (3 u + u^2) / (1 - u)``,
  ``u = 2^-24``. The product ``2 kappa_over_D d_perp``, the sum ``1 + x`` and the quotient round once each (relative
  ``|d_i| <= u``), so ``p32 / p = (1 + d0) / (1 + d0 x / (1 + x)) (1 + d2) / (1 + d1)``, which lies in
  ``[(1 - u)^2 / (1 + u), (1 + u)^2 / (1 - u)]``: ``|p32 - p| <= eps p``. Where ``p32 = 1`` (from ``x = 2^24`` on, where
  ``1 + x`` rounds to ``x``) the realised probability is ``p32`` itself and the distance ``1 / (1 + x)``, just below
  ``2^-24``, is inside ``eps p``. The floor lowers ``p32`` by less than ``2^-32``. A product below float32's smallest
  normal (``x < 2^-126``) rounds by more than ``u`` but realises ``0``, within ``p < 2^-32``; one that overflows, or an
  ``x`` beyond the clamp at ``1e30``, realises ``1``, within ``1 / (1 + x)``.
* ``float64`` (``realised_acceptance(..., dtype=numpy.float64)``: the inputs and the formula in float64, the
  probability cast to float32 by the trial): the product of two float32 values is exact in float64, the sum and the
  quotient round once each in float64 (``v = 2^-53``: ``|p64 - p| <= e64 p``, ``e64 = 2 v / (1 - v)``) and the cast
  once in float32: ``eps = u + (1 + u) e64``.
* ``variant`` (a measurement, not the walk's trial): the probability formed in float64 from the float32 inputs and
  the threshold ``floor(p64 2^32)`` with no cast to float32: ``eps = e64``.

Over ``x`` in the range every float32 ``x_hat`` is taken, with ``kappa_over_D = x_hat / 2`` and ``d_perp = 1`` (the
product is then exact, so the sweep holds the sum, the quotient, the cast and the floor to the bound); below ``2^-34``
the realised probability is ``0`` and the distance ``p < 2^-32`` by the statement above, and from ``2^25`` on
``p32 = 1``. The ratio ``|r - p| / (2^-32 + eps p)`` is computed in float64 (``p`` within ``2^-52 p``, so the ratio
within ``2^-52 p / 2^-32 <= 2^-20``), and every value above ``1 - 2^-16`` is recomputed in exact rationals; the record states the largest ratio, where
it is attained, and that ratio in exact rationals. The composition with the product's rounding is checked on 100 000
random float32 pairs (``x`` log-uniform in ``[1e-9, 31]``, ``d_perp`` log-uniform in ``[1e-10, 1e-5]`` m), every ratio
in exact rationals, beside the largest relative error of the product alone (at most ``u``) and of ``p32`` alone. The
sweep also counts the values that violate the bound with a wrong constant, a third of ``eps``
(``perturbed_violations``, certain violations: ratio above ``1 + 2^-16``), which ``--check`` requires to be non-zero
over the full range for the ``float32`` and ``float64`` forms. The variant's distance is the floor's to within
``2^-52 p``, far below what a float64 ratio resolves next to ``2^-32``, so its constant is stated and not told apart.

Runs on the CPU (``JAX_PLATFORMS=cpu``) from the repository root; the float64 forms run inside ``jax.enable_x64``.
"""
import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TWO32 = 2 ** 32
U = Fraction(1, 2 ** 24)
V = Fraction(1, 2 ** 53)
E64 = 2 * V / (1 - V)
#: ``eps`` of each way of forming the probability, exact
EPS = {"float32": (3 * U + U * U) / (1 - U), "float64": U + (1 + U) * E64, "variant": E64}
#: a wrong constant for each form, a third of its ``eps`` (``u`` for the float32 trial): the sweep counts the values
#: that violate the bound with it, which shows the sweep can fail
PERTURBED_EPS = {form: eps / 3 for form, eps in EPS.items()}
#: the ranges of ``x``: ``[lo, hi]`` as float32 values
RANGES = {"application": (2.0 ** -34, 0.1124), "full": (2.0 ** -34, 2.0 ** 26)}
CHUNK = 1 << 22
#: a float64 ratio above this is recomputed in exact rationals
EXACT_ABOVE = 1.0 - 2.0 ** -16
RANDOM_PAIRS = 100_000
RANDOM_SEED = 20261117


def float32_range(lo, hi):
    """The bit patterns ``[b0, b1]`` of the float32 values in ``[lo, hi]`` (both positive)."""
    b0 = int(np.float32(lo).view(np.uint32))
    b1 = int(np.float32(hi).view(np.uint32))
    if float(np.float32(lo)) < lo:
        b0 += 1
    if float(np.float32(hi)) > hi:
        b1 -= 1
    return b0, b1


def variant_threshold(kappa_over_D, d_perp):
    """``floor(p64 2^32)`` (``2^32`` where ``p64 >= 1``, ``0`` where ``p64 <= 0`` or NaN) with ``p64 = x / (1 + x)``,
    ``x = min(2 kappa_over_D d_perp, 1e30)`` formed in float64 from the inputs as given (float32 values)."""
    k = np.asarray(kappa_over_D, np.float32).astype(np.float64)
    d = np.asarray(d_perp, np.float32).astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        x = np.minimum(2.0 * k * d, float(np.float32(1e30)))
        p = x / (1.0 + x)
        inside = (p > 0.0) & (p < 1.0)
        t = np.floor(np.where(inside, p, 0.0) * float(TWO32)).astype(np.uint64)
    return np.where(p >= 1.0, np.uint64(TWO32), t)


def thresholds(form, k, d):
    """The threshold of each trial, by ``form`` (``float32``, ``float64`` or ``variant``)."""
    import jax

    from dmipy_sim.geometry._boundary import realised_acceptance
    if form == "float32":
        return realised_acceptance(k, d, np.float32)
    if form == "float64":
        with jax.enable_x64(True):
            return realised_acceptance(k, d, np.float64)
    return variant_threshold(k, d)


def exact_ratio(t, x, eps):
    """``|t / 2^32 - p| / (2^-32 + eps p)``, ``p = x / (1 + x)``, in exact rationals (``x`` a Fraction)."""
    p = x / (1 + x)
    return abs(Fraction(int(t), TWO32) - p) / (Fraction(1, TWO32) + eps * p)


def sweep(form, lo, hi):
    """Every float32 ``x_hat`` in ``[lo, hi]`` through ``form`` with ``kappa_over_D = x_hat / 2``, ``d_perp = 1``."""
    eps = EPS[form]
    eps_f = float(eps)
    b0, b1 = float32_range(lo, hi)
    n = 0
    worst = (-1.0, None, None)
    exact_checked = 0
    exact_worst = Fraction(0)
    zero_realised = 0
    saturated = 0
    perturbed = 0
    eps_wrong = float(PERTURBED_EPS[form])
    for s in range(b0, b1 + 1, CHUNK):
        bits = np.arange(s, min(s + CHUNK, b1 + 1), dtype=np.uint32)
        x = bits.view(np.float32)
        k = (x / np.float32(2.0)).astype(np.float32)
        assert np.array_equal(k.astype(np.float64) * 2.0, x.astype(np.float64))      # the product is exact
        t = np.asarray(thresholds(form, k, np.ones_like(k)), np.uint64)
        xe = x.astype(np.float64)
        pe = xe / (1.0 + xe)
        ratio = np.abs(t.astype(np.float64) / TWO32 - pe) / (2.0 ** -32 + eps_f * pe)
        i = int(np.argmax(ratio))
        if ratio[i] > worst[0]:
            worst = (float(ratio[i]), float(x[i]), int(t[i]))
        for j in np.flatnonzero(ratio > EXACT_ABOVE):
            r = exact_ratio(t[j], Fraction(float(x[j])), eps)
            exact_checked += 1
            exact_worst = max(exact_worst, r)
            if r > 1:
                raise AssertionError(f"{form}: the bound fails at x = {float(x[j])!r}: ratio {float(r)!r}")
        wrong = np.abs(t.astype(np.float64) / TWO32 - pe) / (2.0 ** -32 + eps_wrong * pe)
        perturbed += int((wrong > 1.0 + 2.0 ** -16).sum())
        zero_realised += int((t == 0).sum())
        saturated += int((t == TWO32).sum())
        n += bits.size
    r_at_worst = exact_ratio(worst[2], Fraction(worst[1]), eps)
    return {
        "form": form, "x_lo": float(np.uint32(b0).view(np.float32)), "x_hi": float(np.uint32(b1).view(np.float32)),
        "n": n, "eps": str(eps), "eps_over_u": float(eps / U),
        "max_ratio": worst[0], "max_ratio_exact": float(r_at_worst), "max_ratio_exact_le_1": bool(r_at_worst <= 1),
        "max_at_x": worst[1], "max_at_threshold": worst[2],
        "exact_checked": exact_checked, "exact_checked_max": float(exact_worst), "zero_realised": zero_realised,
        "saturated": saturated, "perturbed_eps": str(PERTURBED_EPS[form]), "perturbed_violations": perturbed,
    }


def random_pairs():
    """The 100 000 random float32 pairs: ``x`` log-uniform in ``[1e-9, 31]``, ``d_perp`` in ``[1e-10, 1e-5]`` m."""
    rng = np.random.default_rng(RANDOM_SEED)
    x = np.exp(rng.uniform(np.log(1e-9), np.log(31.0), RANDOM_PAIRS))
    d = np.exp(rng.uniform(np.log(1e-10), np.log(1e-5), RANDOM_PAIRS)).astype(np.float32)
    k = (x / (2.0 * d.astype(np.float64))).astype(np.float32)
    return k, d


def composition(form):
    """The bound on the random pairs, every ratio in exact rationals; the product's and ``p``'s relative errors."""
    import jax.numpy as jnp

    from dmipy_sim.geometry._boundary import transmit_probability
    eps = EPS[form]
    k, d = random_pairs()
    t = np.asarray(thresholds(form, k, d), np.uint64)
    worst, at = Fraction(0), None
    for j in range(k.size):
        x = 2 * Fraction(float(k[j])) * Fraction(float(d[j]))
        r = exact_ratio(t[j], x, eps)
        if r > worst:
            worst, at = r, j
    out = {"form": form, "n": int(k.size), "eps": str(eps), "max_ratio_exact": float(worst),
           "max_ratio_exact_le_1": bool(worst <= 1), "max_at": [float(k[at]), float(d[at])]}
    if form == "float32":
        x_exact = 2.0 * k.astype(np.float64) * d.astype(np.float64)                 # exact: 48 bits
        x_hat = np.float32(2.0) * k * d
        out["product_rel_max_over_u"] = float(np.max(np.abs(x_hat.astype(np.float64) - x_exact) / x_exact) / float(U))
        p32 = np.asarray(transmit_probability(jnp.asarray(k), jnp.asarray(d)), np.float64)
        rel = [abs(Fraction(float(p32[j])) / (x / (1 + x)) - 1)
               for j, x in ((j, 2 * Fraction(float(k[j])) * Fraction(float(d[j]))) for j in range(k.size))]
        out["p32_rel_max_over_2^-23"] = float(max(rel) / (2 * U))
    return out


def run(range_name):
    lo, hi = RANGES[range_name]
    rec = {"range": range_name, "x_range": [lo, hi], "sweeps": [], "random_pairs": []}
    for form in ("float32", "float64", "variant"):
        t0 = time.perf_counter()
        s = sweep(form, lo, hi)
        s["seconds"] = round(time.perf_counter() - t0, 1)
        rec["sweeps"].append(s)
        print(json.dumps(s), flush=True)
    for form in ("float32", "float64", "variant"):
        c = composition(form)
        rec["random_pairs"].append(c)
        print(json.dumps(c), flush=True)
    return rec


def check(rec):
    """The record's verdicts: every sweep covered every float32 of its range and every ratio is at most 1."""
    bad = []
    for s in rec["sweeps"]:
        b0, b1 = float32_range(s["x_lo"], s["x_hi"])
        if s["n"] != b1 - b0 + 1:
            bad.append(f"{s['form']}: {s['n']} values, the range holds {b1 - b0 + 1}")
        if not (s["max_ratio_exact_le_1"] and s["max_ratio_exact"] <= 1 and s["exact_checked_max"] <= 1):
            bad.append(f"{s['form']}: ratio above 1")
        if rec["range"] == "full" and s["form"] != "variant" and s["perturbed_violations"] == 0:
            bad.append(f"{s['form']}: a third of eps violates nowhere: the sweep does not test the constant")
    for c in rec["random_pairs"]:
        if not (c["max_ratio_exact_le_1"] and c["max_ratio_exact"] <= 1):
            bad.append(f"{c['form']} random pairs: ratio above 1")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--range", choices=sorted(RANGES))
    ap.add_argument("--out")
    ap.add_argument("--check")
    a = ap.parse_args(argv)
    if a.check:
        bad = check(json.loads(Path(a.check).read_text()))
        print("\n".join(bad) if bad else "every ratio at most 1 over every float32 of the range")
        return 1 if bad else 0
    rec = run(a.range)
    if a.out:
        Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    bad = check(rec)
    print("\n".join(bad) if bad else "every ratio at most 1 over every float32 of the range")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
