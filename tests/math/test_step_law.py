"""The walk's step law (dmrai-lab/dmipy-sim#714, #715): the pair of lengths the engine draws gives the step the
Gaussian step's moments, the stepped matrix method is the oracle of what the step does in a pore, and the engine
draws what the law says."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.math import step_law as sl

D0 = 2e-9


def test_the_pair_matches_the_gaussian_steps_first_three_even_moments():
    m2, m4, m6 = sl.step_moments(sl.PAIR_LENGTHS, sl.PAIR_WEIGHTS)
    assert m2 == pytest.approx(1.0, abs=1e-12) and m4 == pytest.approx(1.0, abs=1e-12) and m6 == pytest.approx(1.0, abs=1e-12)
    f2, f4, f6 = sl.step_moments(sl.FIXED_LENGTHS, sl.FIXED_WEIGHTS)
    assert f2 == 1.0 and f4 == pytest.approx(0.6) and f6 == pytest.approx(0.2571, abs=1e-4)   # the fixed step's shortfall
    assert sl.LONGEST_STEP == sl.PAIR_LONG == max(sl.PAIR_LENGTHS)
    assert sl.PAIR_SHORT == pytest.approx(0.78267, abs=1e-5) and sl.PAIR_LONG == pytest.approx(1.64947, abs=1e-5)
    assert sl.PAIR_P_LONG == pytest.approx(0.18377, abs=1e-5)


def test_the_bulk_rate_error_is_first_order_for_the_fixed_step_and_third_order_for_the_pair():
    """A mode k decays per step by E[sinc(k delta)] against exp(-(k delta0)^2 / 6): the fixed step's rate is wrong
    by (k delta0)^2 / 30 to leading order, the pair's by O((k delta0)^6)."""
    x = np.array([0.1, 0.2, 0.4, 0.8])
    exact = -x ** 2 / 6
    fixed = np.log(sl.bulk_multiplier(x, sl.FIXED_LENGTHS, sl.FIXED_WEIGHTS)) / exact - 1
    pair = np.log(sl.bulk_multiplier(x)) / exact - 1
    np.testing.assert_allclose(fixed, x ** 2 / 30, rtol=0.05)         # the next order shows at 0.8
    assert np.all(np.abs(pair) < 2e-3 * x ** 6) and abs(pair[-1]) < 5e-4 and fixed[-1] > 2e-2


def test_the_engine_draws_the_law():
    import jax
    from dmipy_sim.engine.physics import pair_step
    s = np.asarray(jax.vmap(pair_step)(jax.random.split(jax.random.PRNGKey(3), 200_000)))
    n2 = (s ** 2).sum(1)
    assert n2.mean() == pytest.approx(1.0, abs=5e-3)                  # E[delta^2] = 6 D dt
    assert (n2 ** 2).mean() * 36 / 60 == pytest.approx(1.0, abs=1.5e-2)   # E[delta^4] = 60 D^2 dt^2
    assert np.sqrt(n2).max() <= sl.LONGEST_STEP * (1 + 1e-5)
    lengths = np.unique(np.round(np.sqrt(n2), 4))
    assert len(lengths) == 2 and np.allclose(lengths, sl.PAIR_LENGTHS, atol=2e-4)
    assert np.mean(np.sqrt(n2) > 1.2) == pytest.approx(sl.PAIR_P_LONG, abs=4e-3)
    assert np.allclose(np.mean(s / np.sqrt(n2)[:, None], axis=0), 0.0, atol=6e-3)   # the direction is uniform


def test_the_slab_operator_is_the_sinc_law_and_the_pair_removes_it():
    """The engine's own reflect on Box1D, projected on the slab's cosines: the fixed step's operator is
    diag(sinc(k delta)) to quadrature precision and the pair's slowest-mode rate error is below 1e-4 where the
    fixed step's is lambda dt / 5."""
    L = 8.5e-6
    sp = sl.SteppedPore("plane", L, D0, d.Box1D(L), n_modes=12)
    dt_p = 1e-4
    delta0 = np.sqrt(6 * D0 * dt_p)
    T, q = sp.operator(delta0, n_rad=80, n_a=32, n_b=48)
    assert q["asym"] < 1e-6 and q["inv"] < 1e-6 and q["diag_err"] < 1e-4
    i1 = np.argsort(sp.lam)[1]
    unit = D0 * sp.lam[i1] * dt_p / 5
    assert sp.rate_error(T, dt_p)[i1] == pytest.approx(unit, rel=0.03)
    Tp = sp.law_operator(dt_p, n_rad=80, n_a=32, n_b=48)
    assert abs(sp.rate_error(Tp, dt_p)[i1]) < 1e-2 * unit


@pytest.mark.parametrize("shape, geometry, c_wall, c_fixed", [
    ("cylinder", lambda: d.Cylinder(5.2e-6, (0.0, 0.0, 1.0)), 0.44, 1.27),
    ("sphere", lambda: d.Sphere(2.8e-6), 0.71, 1.43),
])
def test_the_curved_wall_term_is_first_order_and_the_same_for_every_matched_law(shape, geometry, c_wall, c_fixed):
    """Under the pair the pore's slowest mode keeps a first-order error of c_wall * lambda dt / 5 (the specular
    reflection on a curved wall, at delta0 / a = 0.2), the same under the Gaussian step; the fixed step's is
    c_fixed, its bulk term plus a wall term smaller than the pair's (the terms are not additive)."""
    diam = 2 * (5.2e-6 if shape == "cylinder" else 2.8e-6)
    sp = sl.SteppedPore(shape, diam, D0, geometry(), n_modes=(6, 6))
    dt_p = 1e-4 if shape == "cylinder" else 2.6e-5                   # delta0 / a = 0.21 and 0.2
    i1 = np.argsort(sp.lam)[1]
    unit = D0 * sp.lam[i1] * dt_p / 5
    kw = dict(n_rad=40, n_th=12, n_a=24, n_b=48)
    pair = sp.rate_error(sp.law_operator(dt_p, **kw), dt_p)[i1] / unit
    fixed = sp.rate_error(sp.law_operator(dt_p, sl.FIXED_LENGTHS, sl.FIXED_WEIGHTS, **kw), dt_p)[i1] / unit
    from scipy import special
    x, w = special.roots_genlaguerre(4, 0.5)
    gauss = sp.rate_error(sp.law_operator(dt_p, np.sqrt(4 * x / 6), w / w.sum(), **kw), dt_p)[i1] / unit
    assert pair == pytest.approx(c_wall, abs=0.06)
    assert gauss == pytest.approx(pair, abs=0.03)
    assert fixed == pytest.approx(c_fixed, abs=0.1)
